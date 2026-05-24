from __future__ import annotations
import asyncio
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass, field

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import (
    MessageMediaPhoto, MessageMediaDocument,
    PhotoSize, PhotoSizeProgressive, PhotoStrippedSize, PhotoCachedSize,
)

from config import config
from database import VectorDB
from embeddings import CLIPEmbedder

log = logging.getLogger(__name__)
MIN_PHOTO_SIZE = 224

try:
    import cryptg
    HAS_CRYPTG = True
except ImportError:
    HAS_CRYPTG = False

try:
    from fast_telethon import download_file as fast_download
    HAS_FAST_DL = True
except ImportError:
    HAS_FAST_DL = False


def _get_parallel():
    try:
        from auto_scale import get_parallel
        return get_parallel()
    except Exception:
        return config.PARALLEL_DL


def _get_batch_size():
    try:
        from auto_scale import get_batch_size
        return get_batch_size()
    except Exception:
        return config.BATCH_SIZE


def get_speed_info():
    return {
        "cryptg": HAS_CRYPTG,
        "fast_download": HAS_FAST_DL,
        "parallel_files": _get_parallel(),
        "batch_size": _get_batch_size(),
    }


def session_exists():
    # A usable session is EITHER an env StringSession OR an on-disk .session file.
    from tg_client import user_session_available
    return user_session_available()


def session_info():
    from tg_client import has_string_session, session_location
    p = config.SESSION_FILE
    file_exists = os.path.exists(p)
    return {"path": session_location(),
            "exists": has_string_session() or file_exists,
            "string_session": has_string_session(),
            "size": os.path.getsize(p) if file_exists else 0,
            "cwd": os.getcwd()}


def _make_client():
    from tg_client import make_user_client
    return make_user_client()


async def check_session():
    if not session_exists():
        from tg_client import session_location
        return False, f"No session at {session_location()}"
    c = _make_client()
    try:
        await c.connect()
        if await c.is_user_authorized():
            me = await c.get_me()
            await c.disconnect()
            return True, f"Logged in as {me.first_name} (ID: {me.id})"
        await c.disconnect()
        return False, "Expired. /login again."
    except Exception as e:
        try:
            await c.disconnect()
        except Exception:
            pass
        return False, f"Error: {e}"


def _is_image(msg):
    if not msg.media:
        return False
    if isinstance(msg.media, MessageMediaPhoto):
        return True
    if isinstance(msg.media, MessageMediaDocument):
        d = msg.media.document
        return d and getattr(d, "mime_type", "").startswith("image/")
    return False


def _get_media_size(msg):
    if isinstance(msg.media, MessageMediaPhoto):
        p = msg.media.photo
        if p and p.sizes:
            for s in reversed(p.sizes):
                sz = getattr(s, "size", None)
                if sz:
                    return sz
                szs = getattr(s, "sizes", None)
                if szs:
                    return max(szs)
        return 100000
    if isinstance(msg.media, MessageMediaDocument):
        d = msg.media.document
        return d.size if d else 500000
    return 100000


def _pick_photo_size(msg):
    if not isinstance(msg.media, MessageMediaPhoto):
        return None
    p = msg.media.photo
    if not p or not p.sizes:
        return None
    usable = []
    for s in p.sizes:
        if isinstance(s, (PhotoStrippedSize, PhotoCachedSize)):
            continue
        if isinstance(s, PhotoSizeProgressive):
            w, h = s.w, s.h
        elif isinstance(s, PhotoSize):
            w, h = s.w, s.h
        else:
            w, h = getattr(s, "w", 0), getattr(s, "h", 0)
        if w > 0 and h > 0:
            usable.append((s.type, max(w, h)))
    if not usable:
        return None
    usable.sort(key=lambda x: x[1])
    for t, sz in usable:
        if sz >= MIN_PHOTO_SIZE:
            return t
    return usable[-1][0]


def _msg_link(entity, mid):
    u = getattr(entity, "username", None)
    return f"https://t.me/{u}/{mid}" if u else f"https://t.me/c/{entity.id}/{mid}"


def _channel_display(entity):
    t = getattr(entity, "title", None)
    u = getattr(entity, "username", None)
    if t and u:
        return f"{t} (@{u})"
    return f"{t} (private)" if t else str(entity.id)


def _is_restricted(entity):
    return getattr(entity, "noforwards", False)


async def resolve_channel(client, ch):
    if isinstance(ch, int):
        return await client.get_entity(ch)
    text = str(ch).strip()
    if text.lstrip("-").isdigit():
        return await client.get_entity(int(text))
    if "t.me/" in text:
        slug = text.rstrip("/").split("/")[-1]
        return await client.get_entity(text) if slug.startswith("+") else await client.get_entity(slug)
    return await client.get_entity(text)


@dataclass
class IndexStats:
    downloaded: int = 0
    embedded: int = 0
    skipped: int = 0
    errors: int = 0
    bytes_downloaded: int = 0
    flood_waits: int = 0
    normal_downloads: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self):
        return time.time() - self.start_time

    @property
    def speed(self):
        e = self.elapsed
        return self.embedded / e if e > 0 else 0

    @property
    def mb_downloaded(self):
        return self.bytes_downloaded / 1024 / 1024


class ChannelIndexer:
    def __init__(self, db, embedder):
        self.db = db
        self.embedder = embedder
        Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    async def estimate_channel(self, channel_input):
        c = _make_client()
        await c.connect()
        if not await c.is_user_authorized():
            await c.disconnect()
            raise RuntimeError("Session expired.")
        try:
            e = await resolve_channel(c, channel_input)
            msgs = await c.get_messages(e, limit=0)
            total = msgs.total
            sample = min(200, total)
            photos = 0
            tsize = 0
            if sample > 0:
                async for m in c.iter_messages(e, limit=sample):
                    if _is_image(m):
                        photos += 1
                        tsize += _get_media_size(m)
            ratio = photos / sample if sample > 0 else 0
            est = int(total * ratio)
            avg = tsize / photos if photos > 0 else 100000
            return {
                "channel_name": _channel_display(e),
                "is_private": getattr(e, "username", None) is None,
                "restricted": _is_restricted(e),
                "total_messages": total,
                "sample_size": sample,
                "sample_photos": photos,
                "photo_ratio": round(ratio, 3),
                "estimated_photos": est,
                "avg_photo_kb": round(avg / 1024, 1),
                "estimated_download_mb": round(est * avg / 1024 / 1024, 1),
                "watermark": self.db.get_watermark(e.id),
            }
        finally:
            await c.disconnect()

    async def index_channel(self, channel_input, limit=None, progress_callback=None, concurrency=None):
        if not session_exists():
            raise RuntimeError("No session. /login first.")
        if concurrency is None:
            concurrency = _get_parallel()
        c = _make_client()
        await c.connect()
        if not await c.is_user_authorized():
            await c.disconnect()
            raise RuntimeError("Session expired.")
        try:
            e = await resolve_channel(c, channel_input)
            return await self._fast_index(c, e, limit, progress_callback, concurrency)
        finally:
            try:
                await c.disconnect()
            except Exception:
                pass

    async def _fast_index(self, client, entity, limit, progress_cb, concurrency):
        ch_id = entity.id
        wm = self.db.get_watermark(ch_id)
        actual_batch = _get_batch_size()
        cur_c = min(concurrency, 8)
        min_c, max_c = 3, concurrency
        flood_w, clean_b = [0], [0]

        log.info("INDEX: %s max=%d start=%d batch=%d wm=%s",
                 _channel_display(entity), max_c, cur_c, actual_batch, wm)

        stats = IndexStats()
        sem = asyncio.Semaphore(cur_c)
        eq: asyncio.Queue = asyncio.Queue(maxsize=actual_batch * 5)

        async def update_sem(n):
            nonlocal sem, cur_c
            if n != cur_c:
                cur_c = n
                sem = asyncio.Semaphore(cur_c)

        async def dl_one(msg):
            did = f"{ch_id}_{msg.id}"
            tmp = os.path.join(config.TEMP_DIR, f"{did}.jpg")
            meta = {
                "channel_id": ch_id,
                "channel_name": getattr(entity, "title", str(ch_id)),
                "message_id": msg.id,
                "date": msg.date.isoformat() if msg.date else "",
                "link": _msg_link(entity, msg.id),
                "private": getattr(entity, "username", None) is None,
            }
            async with sem:
                try:
                    thumb = _pick_photo_size(msg)
                    if thumb and isinstance(msg.media, MessageMediaPhoto):
                        await client.download_media(msg, file=tmp, thumb=thumb)
                    else:
                        await msg.download_media(file=tmp)
                    stats.normal_downloads += 1
                except FloodWaitError as e:
                    stats.flood_waits += 1
                    flood_w[0] += 1
                    clean_b[0] = 0
                    await update_sem(max(min_c, cur_c - 2))
                    await asyncio.sleep(min(e.seconds, 30))
                    try:
                        await msg.download_media(file=tmp)
                    except Exception:
                        stats.errors += 1
                        return
                except Exception:
                    stats.errors += 1
                    return
                if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                    stats.skipped += 1
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    return
                stats.downloaded += 1
                stats.bytes_downloaded += os.path.getsize(tmp)
                await eq.put((did, tmp, meta))

        async def producer():
            tasks = []
            bn = 0
            try:
                async for msg in client.iter_messages(entity, limit=limit, min_id=wm, reverse=True):
                    if not _is_image(msg):
                        continue
                    tasks.append(asyncio.create_task(dl_one(msg)))
                    if len(tasks) >= cur_c * 2:
                        _, p = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        tasks = list(p)
                        bn += 1
                        if bn % 5 == 0:
                            if flood_w[0] == 0:
                                clean_b[0] += 1
                                if clean_b[0] >= 3:
                                    await update_sem(min(max_c, cur_c + 1))
                                    clean_b[0] = 0
                            flood_w[0] = 0
                        await asyncio.sleep(0.1)
            except FloodWaitError as e:
                stats.flood_waits += 1
                await update_sem(max(min_c, cur_c - 3))
                await asyncio.sleep(min(e.seconds, 60))
            except Exception as exc:
                log.error("Producer: %s", exc, exc_info=True)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await eq.put(None)

        async def consumer():
            bi, bp, bm = [], [], []
            while True:
                try:
                    item = await asyncio.wait_for(eq.get(), timeout=300)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    if bp:
                        n = await asyncio.to_thread(self._flush, bi, bp, bm)
                        stats.embedded += n
                        if bm:
                            self.db.set_watermark(ch_id, max(m["message_id"] for m in bm))
                        if progress_cb:
                            try:
                                await progress_cb(stats)
                            except Exception:
                                pass
                    break
                di, tp, mt = item
                bi.append(di)
                bp.append(tp)
                bm.append(mt)
                if len(bp) >= actual_batch:
                    n = await asyncio.to_thread(self._flush, bi, bp, bm)
                    stats.embedded += n
                    self.db.set_watermark(ch_id, max(m["message_id"] for m in bm))
                    log.info("  +%d total=%d %.1f/s %.1fMB %ds c=%d",
                             n, stats.embedded, stats.speed, stats.mb_downloaded,
                             stats.elapsed, cur_c)
                    if progress_cb:
                        try:
                            await progress_cb(stats)
                        except Exception:
                            pass
                    bi, bp, bm = [], [], []

        async def hb():
            while True:
                await asyncio.sleep(15)
                log.info("  hb dl=%d emb=%d err=%d c=%d",
                         stats.downloaded, stats.embedded, stats.errors, cur_c)

        p = asyncio.create_task(producer())
        c = asyncio.create_task(consumer())
        h = asyncio.create_task(hb())
        await asyncio.gather(p, c, return_exceptions=True)
        h.cancel()
        log.info("DONE emb=%d skip=%d err=%d flood=%d %.1fMB %.1f/s %ds DB=%d",
                 stats.embedded, stats.skipped, stats.errors, stats.flood_waits,
                 stats.mb_downloaded, stats.speed, stats.elapsed, self.db.count())
        return stats.embedded

    def _flush(self, ids, paths, metas):
        embs = self.embedder.embed_batch(paths)
        oi, oe, om = [], [], []
        for i, emb in enumerate(embs):
            if emb is not None:
                oi.append(ids[i])
                oe.append(emb)
                om.append(metas[i])
        if oi:
            self.db.upsert(oi, oe, om)
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        return len(oi)
