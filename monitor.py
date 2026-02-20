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
    MessageMediaPhoto,
    MessageMediaDocument,
    UpdateDeleteChannelMessages,
    PeerChannel,
)

from config import config
from database import VectorDB
from embeddings import CLIPEmbedder

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
    u = getattr(entity, "username", None)
    return f"https://t.me/{u}/{msg_id}" if u else f"https://t.me/c/{entity.id}/{msg_id}"


class ChannelMonitor:
    def __init__(self, db: VectorDB, embedder: CLIPEmbedder) -> None:
        self.db = db
        self.embedder = embedder
        self.client: TelegramClient | None = None
        self.running = False
        self._watch_ids: set[int] = set()
        self._stats = {
            "new_images": 0,
            "deleted_images": 0,
            "errors": 0,
        }
        self._notify_callback = None
        Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    @property
    def stats(self):
        return dict(self._stats)

    async def start(self, notify_callback=None) -> str:
        """
        Start monitoring all indexed channels.
        notify_callback(text) called on events.
        Returns status message.
        """
        if self.running:
            return "Already running."

        channels = self.db.list_channels()
        if not channels:
            return "No channels indexed. /index first."

        if not os.path.exists(config.SESSION_FILE):
            return "No session. /login first."

        self._notify_callback = notify_callback
        self.client = TelegramClient(
            config.SESSION_PATH, config.API_ID, config.API_HASH,
        )
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

        # register new message handler
        @self.client.on(events.NewMessage(chats=list(self._watch_ids)))
        async def on_new(event):
            await self._handle_new(event)

        # register raw update handler for deletions
        @self.client.on(events.Raw(types=[UpdateDeleteChannelMessages]))
        async def on_delete(event):
            await self._handle_delete(event)

        self.running = True
        self._stats = {"new_images": 0, "deleted_images": 0, "errors": 0}

        names = ", ".join(ch_names.values())
        log.info("Monitor started for: %s", names)

        # keep client running in background
        asyncio.create_task(self._keep_alive())

        return f"✅ Monitoring {len(self._watch_ids)} channels:\n{names}"

    async def _keep_alive(self):
        """Keep Telethon client alive."""
        try:
            while self.running and self.client and self.client.is_connected():
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("Monitor keepalive error: %s", exc)
        finally:
            self.running = False

    async def stop(self) -> str:
        if not self.running:
            return "Not running."
        self.running = False
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
        """Auto-embed new images."""
        msg = event.message
        if not _is_image(msg):
            return

        if not self.embedder.is_ready():
            return

        ch_id = event.chat_id
        doc_id = f"{ch_id}_{msg.id}"
        tmp = os.path.join(config.TEMP_DIR, f"{doc_id}.jpg")

        try:
            entity = await event.get_chat()
            await msg.download_media(file=tmp)

            if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                return

            emb = self.embedder.embed_path(tmp)
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

            self.db.upsert([doc_id], [emb], [meta])
            self.db.set_watermark(ch_id, max(
                self.db.get_watermark(ch_id), msg.id,
            ))
            self._stats["new_images"] += 1

            log.info("Monitor: +1 image from %s (msg %d)",
                     getattr(entity, "title", ch_id), msg.id)

            if self._notify_callback:
                try:
                    await self._notify_callback(
                        f"📸 New image indexed from "
                        f"*{getattr(entity, 'title', ch_id)}*\n"
                        f"Total: *{self.db.count():,}*"
                    )
                except Exception:
                    pass

        except Exception as exc:
            self._stats["errors"] += 1
            log.error("Monitor new image error: %s", exc)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    async def _handle_delete(self, event):
        """Remove vectors for deleted messages."""
        if not hasattr(event, "channel_id") or not hasattr(event, "messages"):
            return

        ch_id = event.channel_id
        if ch_id not in self._watch_ids:
            return

        deleted_ids = []
        for msg_id in event.messages:
            doc_id = f"{ch_id}_{msg_id}"
            deleted_ids.append(doc_id)

        if deleted_ids:
            try:
                self.db.delete(deleted_ids)
                self._stats["deleted_images"] += len(deleted_ids)
                log.info("Monitor: -%d vectors (channel %d)",
                         len(deleted_ids), ch_id)

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
        client = TelegramClient(
            config.SESSION_PATH, config.API_ID, config.API_HASH,
        )
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
                await client.disconnect()
                return {"channel": ch_name, "checked": 0, "removed": 0}

            to_delete = []
            checked = 0
            batch_size = 100

            # check in batches
            msg_ids = [item["message_id"] for item in indexed]
            doc_ids = {item["message_id"]: item["id"] for item in indexed}

            for i in range(0, len(msg_ids), batch_size):
                batch = msg_ids[i:i + batch_size]
                try:
                    # get_messages returns None for deleted ones
                    messages = await client.get_messages(
                        entity, ids=batch,
                    )
                    for j, msg in enumerate(messages):
                        mid = batch[j]
                        if msg is None:
                            # message deleted
                            did = doc_ids.get(mid)
                            if did:
                                to_delete.append(did)
                    checked += len(batch)
                except Exception as exc:
                    log.warning("Cleanup batch error: %s", exc)

                if progress_callback:
                    await progress_callback(checked, len(msg_ids), len(to_delete))

            if to_delete:
                self.db.delete(to_delete)
                log.info("Cleanup: removed %d from %s", len(to_delete), ch_name)

            return {
                "channel": ch_name,
                "checked": checked,
                "removed": len(to_delete),
                "remaining": self.db.count(),
            }

        finally:
            await client.disconnect()
