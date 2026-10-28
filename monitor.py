"""
Channel monitor — watches indexed channels for:
  - New images → auto-embed
  - Deleted messages → remove vectors
"""
from __future__ import annotations
import asyncio
import logging
import os
from pathlib import Path

from telethon import TelegramClient, events
from telethon.tl.types import (
    UpdateDeleteChannelMessages,
)

from config import config
from database import VectorDB
from embeddings import CLIPEmbedder
from tg_client import msg_link, prime_entity_cache, make_user_client
from media import is_image as media_is_image, download_thumb_type

log = logging.getLogger(__name__)


def _is_image(msg) -> bool:
    return media_is_image(msg)


def _msg_link(entity, msg_id):
    return msg_link(entity, msg_id)


def _md(text) -> str:
    out = str(text)
    for ch, esc in (("_", "\\_"), ("*", "\\*"), ("`", "\\`"), ("[", "\\[")):
        out = out.replace(ch, esc)
    return out


def _unmarked(channel_id) -> int:
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
        self._db_lock: asyncio.Lock = db_lock or asyncio.Lock()
        self.client: TelegramClient | None = None
        self.running = False
        self._watch_ids: set[int] = set()
        self._stats = {"new_images": 0, "deleted_images": 0, "errors": 0}
        self._notify_callback = None
        self._bg: set[asyncio.Task] = set()
        self._keep_alive_task: asyncio.Task | None = None
        Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    def _spawn(self, coro) -> None:
        if not self.running:
            coro.close()
            return
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    @property
    def stats(self):
        return dict(self._stats)

    async def start(self, notify_callback=None) -> str:
        if self.running:
            return "Already running."

        channels = self.db.list_channels()
        if not channels:
            return "No channels indexed. /index first."

        from tg_client import user_session_available
        if not user_session_available():
            return "No session. /login first."

        self._notify_callback = notify_callback
        self.client = make_user_client()
        await self.client.connect()

        if not await self.client.is_user_authorized():
            await self.client.disconnect()
            return "Session expired. /login again."

        # Prime the entity cache so bare channel ids resolve under StringSession.
        await prime_entity_cache(self.client)

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

        # Register handlers WITHOUT a chats filter so channels indexed later
        # (hot-add) are picked up without re-registering.
        @self.client.on(events.NewMessage())
        async def on_new(event):
            if _unmarked(getattr(event, "chat_id", 0)) not in self._watch_ids:
                return
            await self._handle_new(event)

        @self.client.on(events.Raw(types=[UpdateDeleteChannelMessages]))
        async def on_delete(event):
            await self._handle_delete(event)

        self.running = True
        self._stats = {"new_images": 0, "deleted_images": 0, "errors": 0}

        names = ", ".join(ch_names.values())
        log.info("Monitor started for: %s", names)

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
            if self.client is not None:
                try:
                    await self.client.disconnect()
                except Exception:
                    pass
                self.client = None
            return "Not running."
        self.running = False

        if self._keep_alive_task is not None:
            self._keep_alive_task.cancel()
            try:
                await self._keep_alive_task
            except (asyncio.CancelledError, Exception):
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

    async def watch(self, channel_id) -> bool:
        """Hot-add a channel to a running monitor without restart."""
        if not self.running or self.client is None:
            return False
        try:
            await prime_entity_cache(self.client)
            entity = await self.client.get_entity(int(channel_id))
            self._watch_ids.add(entity.id)
            log.info("Monitor hot-added channel %s", getattr(entity, "title", channel_id))
            return True
        except Exception as exc:
            log.warning("Monitor hot-add failed for %s: %s", channel_id, exc)
            return False

    async def _handle_new(self, event):
        if not self.running:
            return
        msg = event.message
        if not media_is_image(msg):
            return
        if not self.embedder.is_ready():
            return

        tmp = None
        try:
            entity = await event.get_chat()
            ch_id = _unmarked(entity.id)
            if ch_id not in self._watch_ids:
                return
            doc_id = f"{ch_id}_{msg.id}"
            tmp = os.path.join(config.TEMP_DIR, f"{doc_id}.jpg")

            thumb = download_thumb_type(msg)
            await msg.download_media(file=tmp, thumb=thumb)
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
            def _write():
                # Drop a stale vector from an older monitor/indexer id scheme
                # for the same message before upserting the current one.
                self.db.delete([f"-100{ch_id}_{msg_id}"])
                self.db.upsert([doc_id], [emb], [meta])

            # SQLite transactions + HNSW mutations are slow enough to stall the
            # event loop (which also runs the Bot API polling) — run them in a
            # worker thread, exactly like the indexer's _flush_locked does.
            async with self._db_lock:
                await asyncio.to_thread(_write)
                # Never race a bulk index run: jumping the watermark ahead of
                # the indexer's gap-free prefix leaves an unretryable hole,
                # because set_watermark() only ever moves forward. The vector
                # above is already safe — only the watermark advance waits.
                from indexer import index_in_progress
                if (not index_in_progress(ch_id)
                        and self.db.get_watermark(ch_id) < msg_id):
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
        if not self.running:
            return
        if not hasattr(event, "channel_id") or not hasattr(event, "messages"):
            return

        ch_id = _unmarked(event.channel_id)
        # Resolve via DB membership rather than _watch_ids, so deletes for
        # channels that are indexed but not actively watched still get cleaned.
        # get_ids_for_channel pages through the whole channel — keep that off
        # the event loop.
        try:
            indexed = await asyncio.to_thread(
                self.db.get_ids_for_channel, ch_id)
            if not indexed:
                # fall back: allow if in watch set
                if ch_id not in self._watch_ids:
                    return
        except Exception:
            if ch_id not in self._watch_ids:
                return

        deleted_ids = [f"{ch_id}_{msg_id}" for msg_id in event.messages]

        if deleted_ids:
            self._spawn(self._persist_delete(ch_id, deleted_ids))

    async def _persist_delete(self, ch_id, deleted_ids):
        try:
            async with self._db_lock:
                await asyncio.to_thread(self.db.delete, deleted_ids)
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
        from tg_client import make_user_client
        client = make_user_client(ephemeral=True)
        await client.connect()
        if not await client.is_user_authorized():
            await client.disconnect()
            raise RuntimeError("Session expired.")

        try:
            from indexer import resolve_channel
            entity = await resolve_channel(client, channel_input)
            ch_id = _unmarked(entity.id)
            ch_name = getattr(entity, "title", str(ch_id))

            indexed = await asyncio.to_thread(
                self.db.get_ids_for_channel, ch_id)
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
                async with self._db_lock:
                    await asyncio.to_thread(self.db.delete, to_delete)
                log.info("Cleanup: removed %d from %s", len(to_delete), ch_name)

            return {
                "channel": ch_name,
                "checked": checked,
                "removed": len(to_delete),
                "remaining": self.db.count(),
            }

        finally:
            await client.disconnect()
