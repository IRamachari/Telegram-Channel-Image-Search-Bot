"""
Channel monitor — watches indexed channels for:
  - New images → auto-embed
  - Deleted messages → remove vectors
"""

from __future__ import annotations
import asyncio
import logging
import os
import re
from pathlib import Path

from telethon import TelegramClient, events
from telethon.tl.types import (
    MessageMediaPhoto,
    MessageMediaDocument,
    UpdateDeleteChannelMessages,
    PeerChannel,
)

from config import config
from database import VectorDB
from embeddings import CLIPEmbedder
import tg_client
from tg_client import msg_link

log = logging.getLogger(__name__)


def _is_image(msg) -> bool:
    if not msg or not msg.media:
        return False
    if isinstance(msg.media, MessageMediaPhoto):
        return True
    if isinstance(msg.media, MessageMediaDocument):
        doc = msg.media.document
        return doc and getattr(doc, "mime_type", "").startswith("image/")
    return False


def _msg_link(entity, msg_id):
    return msg_link(entity, msg_id)


def _md(text) -> str:
    """Escape dynamic text for Telegram's legacy Markdown (v1) escape set."""
    out = str(text)
    for ch, esc in (("_", "\\_"), ("*", "\\*"), ("`", "\\`"), ("[", "\\[")):
        out = out.replace(ch, esc)
    return out


def _unmarked(channel_id) -> int:
    """Strip Telegram's -100 marked-id prefix so ids match the indexer."""
    from tg_client import channel_url_id
    try:
        return int(channel_url_id(channel_id))
    except (TypeError, ValueError):
        return int(channel_id)


class ChannelMonitor:
    def __init__(self, db: VectorDB, embedder: CLIPEmbedder,
                 db_lock: asyncio.Lock | None = None) -> None:
        self.db = db
        self.embedder = embedder
        # Shared write lock — passed from SearchBot so the monitor, indexer,
        # and cleanup never hit ChromaDB/SQLite concurrently.
        self._db_lock: asyncio.Lock = db_lock or asyncio.Lock()
        self.client: TelegramClient | None = None
        self.running = False
        self._watch_ids: set[int] = set()
        self._stats = {
            "new_images": 0,
            "deleted_images": 0,
            "errors": 0,
        }
        self._notify_callback = None
        # Telethon dispatches event handlers sequentially on the loop. If a
        # handler awaits the DB write lock inline, a long index run starves the
        # dispatcher (and with it the MTProto pings) for hours. DB writes are
        # therefore handed to a tracked background task.
        self._bg: set[asyncio.Task] = set()
        self._keep_alive_task: asyncio.Task | None = None
        Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _spawn(self, coro) -> None:
        """Fire-and-forget task with a strong reference (see asyncio docs)."""
        if not self.running:
            # Refuse work once stop() has begun: a task spawned after _bg was
            # cleared would hold no strong reference and could be GC'd, and would
            # write a vector the operator just asked to stop.
            coro.close()
            return
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    @property
    def stats(self):
        return dict(self._stats)

    async def start(self, notify_callback=None) -> str:
        """
        Start monitoring all indexed channels.
        notify_callback(text) is called on new/deleted events.
        Returns a status message.
        """
        if self.running:
            return "Already running."

        channels = self.db.list_channels()
        if not channels:
            return "No channels indexed. /index first."

        from tg_client import user_session_available, make_user_client
        if not user_session_available():
            return "No session. /login first."

        self._notify_callback = notify_callback
        self.client = make_user_client()
        await self.client.connect()

        if not await self.client.is_user_authorized():
            await self.client.disconnect()
            return "Session expired. /login again."

        # resolve channel IDs
        self._watch_ids = set()
        ch_names = {}
        for ch_id_str in channels:
            try:
                ch_id = int(ch_id_str)
                entity = await self.client.get_entity(ch_id)
                self._watch_ids.add(entity.id)
                ch_names[entity.id] = getattr(entity, "title", str(ch_id))
            except Exception as exc:
                log.warning("Can't resolve channel %s: %s", ch_id_str, exc)

        if not self._watch_ids:
            await self.client.disconnect()
            return "No channels could be resolved."

        @self.client.on(events.NewMessage(chats=list(self._watch_ids)))
        async def on_new(event):
            await self._handle_new(event)

        @self.client.on(events.Raw(types=[UpdateDeleteChannelMessages]))
        async def on_delete(event):
            await self._handle_delete(event)

        self.running = True
        self._stats = {"new_images": 0, "deleted_images": 0, "errors": 0}

        names = ", ".join(ch_names.values())
        log.info("Monitor started for: %s", names)

        # Telethon already pings MTProto itself; this is a liveness *watchdog*
        # that detects a dead socket and reconnects. The task reference is kept
        # (the event loop only holds weak references) and cancelled in stop().
        self._keep_alive_task = asyncio.create_task(self._watchdog())
        self._keep_alive_task.add_done_callback(lambda t: self._task_died("watchdog", t))

        return f"✅ Monitoring {len(self._watch_ids)} channels:\n{names}"

    def _task_died(self, name: str, task: "asyncio.Task") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("Monitor %s task failed: %s", name, exc, exc_info=exc)
            self._stats["errors"] += 1

    async def _watchdog(self):
        """
        Telethon's own keep-alive pings MTProto every 60 s; ``is_connected()``
        is only a user-set flag and cannot detect a dead socket. This loop
        probes the connection and reconnects when it has silently dropped —
        otherwise the monitor stays "Running" while nothing is indexed.
        """
        while self.running and self.client:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise
            except Exception:
                return
            if not self.running or not self.client:
                return
            try:
                if not self.client.is_connected():
                    log.warning("Monitor: MTProto connection lost — reconnecting")
                await self.client.get_me()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("Monitor keepalive/connect failed: %s", exc)
                try:
                    await self.client.connect()
                except asyncio.CancelledError:
                    raise
                except Exception as exc2:
                    log.error("Monitor reconnect failed: %s", exc2)

    async def stop(self) -> str:
        if not self.running:
            # Still make sure a client isn't left connected.
            if self.client is not None:
                try:
                    await self.client.disconnect()
                except Exception:
                    pass
                self.client = None
            return "Not running."
        # Flip the flag FIRST: _handle_new/_handle_delete and _spawn consult it,
        # so no new work can start once teardown begins.
        self.running = False

        # Cancel the watchdog before anything else: its old finally clause used
        # to reset running=False and keep a reference to a dead client.
        if self._keep_alive_task is not None:
            self._keep_alive_task.cancel()
            try:
                await self._keep_alive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._keep_alive_task = None

        for t in list(self._bg):
            t.cancel()
        if self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
        self._bg.clear()

        if self.client:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None
        return (
            f"⏹ Monitor stopped.\n\n"
            f"📸 New images indexed: {self._stats['new_images']}\n"
            f"🗑 Deleted removed: {self._stats['deleted_images']}\n"
            f"❌ Errors: {self._stats['errors']}"
        )

    async def _handle_new(self, event):
        """
        Auto-embed new images. Downloads and embeds inline (slow, but harmless),
        then hands the DB write to a background task so Telethon's dispatcher is
        never blocked by the write lock for the duration of an index run.
        """
        if not self.running:
            return
        msg = event.message
        if not _is_image(msg):
            return
        if not self.embedder.is_ready():
            return

        tmp = None
        try:
            entity = await event.get_chat()
            # event.chat_id is a Bot-API style *marked* id (-100…). Everywhere
            # else the pipeline uses the unmarked entity.id; using the marked id
            # here produced duplicate doc-ids and a watermark the indexer never
            # reads.
            ch_id = _unmarked(entity.id)
            doc_id = f"{ch_id}_{msg.id}"
            tmp = os.path.join(config.TEMP_DIR, f"{doc_id}.jpg")

            await msg.download_media(file=tmp)
            if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                return

            emb = await asyncio.to_thread(self.embedder.embed_path, tmp)
            if emb is None:
                return

            meta = {
                "channel_id": ch_id,
                "channel_name": getattr(entity, "title", str(ch_id)),
                "message_id": msg.id,
                "date": msg.date.isoformat() if msg.date else "",
                "link": _msg_link(entity, msg.id),
                "private": getattr(entity, "username", None) is None,
            }
            self._spawn(self._persist_new(doc_id, emb, meta, ch_id, msg.id,
                                          getattr(entity, "title", ch_id)))
        except Exception as exc:
            self._stats["errors"] += 1
            log.error("Monitor new image error: %s", exc)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    async def _persist_new(self, doc_id, emb, meta, ch_id, msg_id, title):
        try:
            async with self._db_lock:
                # Self-heal rows written by the old (marked-id) code path so a
                # single image can never be indexed twice under both ids.
                # Deleting an id that does not exist is a no-op.
                self.db.delete([f"-100{ch_id}_{msg_id}"])
                self.db.upsert([doc_id], [emb], [meta])
                if self.db.get_watermark(ch_id) < msg_id:
                    self.db.set_watermark(ch_id, msg_id)
            self._stats["new_images"] += 1
            log.info("Monitor: +1 image from %s (msg %d)", title, msg_id)
            if self._notify_callback:
                try:
                    await self._notify_callback(
                        f"📸 New image indexed from "
                        f"*{_md(title)}*\n"
                        f"Total: *{self.db.count():,}*"
                    )
                except Exception:
                    pass
        except Exception as exc:
            self._stats["errors"] += 1
            log.error("Monitor persist error: %s", exc)

    async def _handle_delete(self, event):
        """Remove vectors for deleted messages — serialised via _db_lock."""
        if not self.running:
            return
        if not hasattr(event, "channel_id") or not hasattr(event, "messages"):
            return

        ch_id = _unmarked(event.channel_id)
        if ch_id not in self._watch_ids:
            return

        deleted_ids = [f"{ch_id}_{msg_id}" for msg_id in event.messages]

        if deleted_ids:
            # Hand off instead of awaiting the lock inline (see _handle_new).
            self._spawn(self._persist_delete(ch_id, deleted_ids))

    async def _persist_delete(self, ch_id, deleted_ids):
        try:
            async with self._db_lock:
                self.db.delete(deleted_ids)
            self._stats["deleted_images"] += len(deleted_ids)
            log.info("Monitor: -%d vectors (channel %d)", len(deleted_ids), ch_id)
            if self._notify_callback:
                try:
                    await self._notify_callback(
                        f"🗑 Removed *{len(deleted_ids)}* deleted images\n"
                        f"Total: *{self.db.count():,}*"
                    )
                except Exception:
                    pass
        except Exception as exc:
            self._stats["errors"] += 1
            log.error("Monitor delete error: %s", exc)

    async def cleanup_channel(self, channel_input, progress_callback=None) -> dict:
        """
        Check all indexed messages — remove vectors for deleted ones.
        """
        from tg_client import make_user_client
        client = make_user_client()
        await client.connect()
        if not await client.is_user_authorized():
            await client.disconnect()
            raise RuntimeError("Session expired.")

        try:
            from indexer import resolve_channel
            entity = await resolve_channel(client, channel_input)
            ch_id = entity.id
            ch_name = getattr(entity, "title", str(ch_id))

            indexed = self.db.get_ids_for_channel(ch_id)
            if not indexed:
                return {"channel": ch_name, "checked": 0, "removed": 0,
                        "remaining": self.db.count()}

            to_delete = []
            checked = 0
            batch_size = 100

            msg_ids = [item["message_id"] for item in indexed]
            doc_ids = {item["message_id"]: item["id"] for item in indexed}

            for i in range(0, len(msg_ids), batch_size):
                batch = msg_ids[i:i + batch_size]
                try:
                    messages = await client.get_messages(entity, ids=batch)
                    if messages is not None:
                        for j, msg in enumerate(messages):
                            if msg is None:
                                mid = batch[j]
                                did = doc_ids.get(mid)
                                if did:
                                    to_delete.append(did)
                    checked += len(batch)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("Cleanup batch error: %s", exc)

                if progress_callback:
                    try:
                        await progress_callback(checked, len(msg_ids), len(to_delete))
                    except Exception:
                        pass

            if to_delete:
                # ── serialised write ──────────────────────────────────────────
                async with self._db_lock:
                    self.db.delete(to_delete)
                # ─────────────────────────────────────────────────────────────
                log.info("Cleanup: removed %d from %s", len(to_delete), ch_name)

            return {
                "channel": ch_name,
                "checked": checked,
                "removed": len(to_delete),
                "remaining": self.db.count(),
            }

        finally:
            await client.disconnect()
