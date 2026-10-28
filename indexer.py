from __future__ import annotations
import asyncio
import logging
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field

from telethon import TelegramClient
from telethon.errors import FloodWaitError

from config import config
from database import VectorDB
from embeddings import CLIPEmbedder
import models
from media import (
    download_thumb_type, is_image, media_kind, media_size,
    pick_photo_size, sweep_temp_dir,
)

log = logging.getLogger(__name__)

# A single media item must not monopolise a run: a flood wait longer than this
# is treated as "skip this message" (the same outcome as any other download
# failure) instead of stalling every worker for half an hour.
_FLOOD_BUDGET_S = 1800

# Channel ids with a bulk index run in flight. The monitor consults this before
# advancing a watermark (see monitor._persist_new): jumping it ahead of the
# indexer's gap-free prefix leaves an unretryable hole, because set_watermark()
# only ever moves forward.
_active_runs: set[int] = set()


def index_in_progress(channel_id) -> bool:
    try:
        return int(channel_id) in _active_runs
    except (TypeError, ValueError):
        return False


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


def _make_client(receive_updates: bool = True):
    from tg_client import make_user_client
    # ephemeral=True: a file-session client works on an in-memory copy of the
    # session, so it never competes with the long-lived monitor for the sqlite
    # file (the source of intermittent "database is locked" runs).
    return make_user_client(receive_updates=receive_updates, ephemeral=True)


async def check_session():
    if not session_exists():
        from tg_client import session_location
        return False, f"No session at {session_location()}"
    c = _make_client(receive_updates=False)
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


# Media classification lives in ``media.py`` so the indexer and the monitor can
# never disagree about what counts as an image. These aliases keep the old
# private names working for callers that used them.
_is_image = is_image


def _pick_photo_size(msg):
    media = getattr(msg, "media", None)
    photo = getattr(media, "photo", None)
    return pick_photo_size(photo) if photo else None


def _msg_link(entity, mid):
    from tg_client import msg_link
    return msg_link(entity, mid)


def _channel_display(entity):
    t = getattr(entity, "title", None)
    u = getattr(entity, "username", None)
    if t and u:
        return f"{t} (@{u})"
    return f"{t} (private)" if t else str(entity.id)


def _is_restricted(entity):
    return getattr(entity, "noforwards", False)


# ``t.me/<user>``, ``t.me/<user>/<msg>``, ``t.me/c/<id>``, ``t.me/c/<id>/<msg>``,
# optionally with a scheme/www prefix and a query/fragment tail.
_TME_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/(?P<rest>[^\s?#]+)", re.I)


async def _resolve_numeric(client, value):
    """Resolve an id, priming the entity cache if the id is not known yet."""
    try:
        return await client.get_entity(value)
    except Exception:
        # A bare id is only resolvable when its access_hash is cached. A
        # StringSession persists no entities, so in a fresh process this fails
        # until one dialog-list pass has filled the cache. Broaden the catch so
        # transient FloodWait/other errors also trigger a cache prime + retry.
        from tg_client import prime_entity_cache
        await prime_entity_cache(client)
        return await client.get_entity(value)


async def resolve_channel(client, ch):
    """
    Turn user input into a channel entity.

    Accepts ``@username``, a bare/marked id, and every ``t.me`` form —
    including message deep links (``t.me/c/<id>/<msg>`` and
    ``t.me/<username>/<msg>``), whose trailing message id is not part of the
    entity. ``t.me/+invite`` / ``t.me/joinchat/...`` are passed through to
    Telethon, which raises when the account is not a member yet.
    """
    if isinstance(ch, int):
        return await _resolve_numeric(client, ch)
    text = str(ch).strip()
    if not text:
        raise ValueError("empty channel")
    match = _TME_RE.search(text)
    if match:
        rest = match.group("rest").strip("/")
        if rest.startswith(("+", "joinchat/")):
            return await client.get_entity(text)
        parts = [p for p in rest.split("/") if p]
        if len(parts) >= 2 and parts[0].lower() == "c" and parts[1].isdigit():
            return await _resolve_numeric(client, int(parts[1]))
        if parts:
            return await _resolve_numeric(client, parts[0].lstrip("@"))
        raise ValueError(f"not a channel link: {text}")
    if text.lstrip("-").isdigit():
        return await _resolve_numeric(client, int(text))
    return await _resolve_numeric(client, text.lstrip("@"))


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


class _Limiter:
    """
    Adjustable concurrency limiter.

    ``asyncio.Semaphore`` cannot be resized: the previous code replaced the
    whole Semaphore object on every throttle change, which left waiters parked
    on the *old* object while new tasks acquired the *new* one — the real
    in-flight count silently drifted from the advertised value. This always
    consults the live target instead.
    """

    def __init__(self, target: int):
        self._target = max(1, int(target))
        self._active = 0
        self._cond = asyncio.Condition()

    @property
    def target(self) -> int:
        return self._target

    @property
    def active(self) -> int:
        return self._active

    async def set_target(self, n: int) -> None:
        n = max(1, int(n))
        async with self._cond:
            if n != self._target:
                self._target = n
                self._cond.notify_all()

    async def __aenter__(self) -> "_Limiter":
        async with self._cond:
            await self._cond.wait_for(lambda: self._active < self._target)
            self._active += 1
        return self

    async def __aexit__(self, *exc) -> None:
        # Decrement BEFORE awaiting the condition: asyncio.Lock.acquire() takes
        # the waiting path whenever a waiter is queued (the steady state while
        # the limiter is saturated), so a cancel() delivered between the release
        # and the acquire raised CancelledError with the slot never returned —
        # permanently leaking a permit and deadlocking every later acquire.
        self._active -= 1
        n = self._active
        async with self._cond:
            self._cond.notify_all()


class ChannelIndexer:
    def __init__(self, db, embedder, db_lock=None):
        self.db = db
        self.embedder = embedder
        # Shared write lock — passed by the bot so the monitor, indexer and
        # cleanup never write to ChromaDB concurrently.
        self._db_lock = db_lock
        Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    async def estimate_channel(self, channel_input):
        c = _make_client(receive_updates=False)
        # connect() inside the try (see index_channel): an exception there
        # previously leaked the client and its session-file handle.
        try:
            await c.connect()
            if not await c.is_user_authorized():
                raise RuntimeError("Session expired.")
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
            try:
                await c.disconnect()
            except Exception:
                pass

    async def index_channel(self, channel_input, limit=None, progress_callback=None, concurrency=None):
        if not session_exists():
            raise RuntimeError("No session. /login first.")
        if concurrency is None:
            concurrency = _get_parallel()
        c = _make_client(receive_updates=False)
        # connect() is INSIDE the try: it was previously before it, so any
        # ConnectionError/FloodWaitError left a live client (socket, MTProto
        # sender task and the sqlite session handle) behind for the life of the
        # process — and the leaked handle on `<name>.session` then caused
        # "database is locked" for every later run.
        try:
            await c.connect()
            if not await c.is_user_authorized():
                raise RuntimeError("Session expired.")
            e = await resolve_channel(c, channel_input)
            return await self._fast_index(c, e, limit, progress_callback, concurrency)
        finally:
            try:
                await c.disconnect()
            except Exception:
                pass

    async def _fast_index(self, client, entity, limit, progress_cb, concurrency):
        ch_id = entity.id
        _active_runs.add(int(ch_id))
        actual_batch = _get_batch_size()
        cur_c = min(concurrency, 8)
        # min_c must never exceed max_c: with concurrency < 3 the old
        # "back off to floor 3" raised the limiter ABOVE the configured max,
        # which is exactly the direction that provokes flood-waits.
        min_c, max_c = min(3, concurrency), max(3, concurrency)
        flood_w, clean_b = [0], [0]

        log.info("INDEX: %s max=%d start=%d batch=%d wm=%s",
                 _channel_display(entity), max_c, cur_c, actual_batch, wm)

        stats = IndexStats()
        limiter = _Limiter(cur_c)

        # ── watermark bookkeeping ────────────────────────────────────────────
        # Downloads finish out of order, so the batch that lands in the queue
        # first can already contain the highest ids. Advancing the watermark to
        # that value on a crash/OOM/reboot would permanently skip every message
        # that was still queued below it. We therefore only ever advance the
        # watermark across a gap-free prefix of what has actually been flushed.
        pending: set[int] = set()   # produced by the producer, not yet persisted
        failed: set[int] = set()    # produced but could not be embedded (skip)
        produced_max = [wm]         # highest message id the producer has seen

        def _safe_watermark() -> int:
            # Only messages that WILL be persisted hold the watermark back.
            # Messages that failed to embed are tracked separately and do not
            # hold the gap-free prefix (otherwise one bad image pins the
            # watermark below it forever).
            effective = pending - failed
            if effective:
                return min(effective) - 1
            return produced_max[0]

        def _reject(msg_id, tmp_path) -> None:
            """A message that will not be persisted this run."""
            pending.discard(msg_id)
            # Telethon writes progressively, so a failed download can leave a
            # partial file. Nothing else removes these.
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

        async def _commit_watermark() -> None:
            candidate = max(wm, _safe_watermark())
            if candidate > self.db.get_watermark(ch_id):
                self.db.set_watermark(ch_id, candidate)

        async def update_sem(n):
            n_i = n
            if n_i != cur_c_i[0]:
                cur_c_i[0] = n_i
            await limiter.set_target(n)

        cur_c_i = [cur_c]

        eq: asyncio.Queue = asyncio.Queue(maxsize=max(1, actual_batch * 5))

        async def dl_one(msg):
            did = f"{ch_id}_{msg.id}"
            # The temp name must be unique per component: the monitor uses the
            # identical scheme in the same directory, and two writers on one
            # file meant each deleting the other's download.
            tmp = os.path.join(config.TEMP_DIR, f"idx{os.getpid()}_{did}.jpg")
            meta = {
                "channel_id": ch_id,
                "channel_name": getattr(entity, "title", str(ch_id)),
                "message_id": msg.id,
                "date": msg.date.isoformat() if msg.date else "",
                "link": _msg_link(entity, msg.id),
                "private": getattr(entity, "username", None) is None,
            }
            try:
                async with limiter:
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
                        await update_sem(max(min_c, cur_c_i[0] - 2))
                        # Honour the real flood-wait window (capped so a single
                        # bad wait can't wedge the run forever), then retry once.
                        await asyncio.sleep(min(max(e.seconds, 1), 300))
                        try:
                            await msg.download_media(file=tmp, thumb=thumb)
                        except Exception:
                            stats.errors += 1
                            _reject(msg.id, tmp)
                            return
                    except Exception:
                        stats.errors += 1
                        _reject(msg.id, tmp)
                        return
                    if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                        stats.skipped += 1
                        _reject(msg.id, tmp)
                        return
                    stats.downloaded += 1
                    try:
                        stats.bytes_downloaded += os.path.getsize(tmp)
                    except OSError:
                        stats.errors += 1
                        _reject(msg.id, tmp)
                        return
                    await eq.put((did, tmp, meta))
            except asyncio.CancelledError:
                # Telethon leaves a partially-written file when a download is
                # cancelled, and nothing else sweeps temp_images.
                _reject(msg.id, tmp)
                raise

        async def producer():
            tasks = []
            bn = 0
            seen = 0          # messages yielded by the iterator (all types)
            last = wm         # resume cursor: last message id actually seen
            try:
                while True:
                    remaining = None if limit is None else max(0, limit - seen)
                    if limit is not None and remaining <= 0:
                        break
                    try:
                        # offset_id is EXCLUSIVE, so resuming from `last` picks
                        # up right after the message that survived the flood
                        # wait (using min_id would re-download that one).
                        # wait_time=1 paces GetHistory so huge backfills don't
                        # provoke the very flood-wait handled below.
                        async for msg in client.iter_messages(
                            entity, limit=remaining, offset_id=last,
                            reverse=True, wait_time=1.0,
                        ):
                            last = msg.id
                            produced_max[0] = msg.id
                            seen += 1
                            if not _is_image(msg):
                                continue
                            pending.add(msg.id)
                            tasks.append(asyncio.create_task(dl_one(msg)))
                            if len(tasks) >= cur_c_i[0] * 2:
                                _, p = await asyncio.wait(
                                    tasks, return_when=asyncio.FIRST_COMPLETED)
                                tasks = list(p)
                                bn += 1
                                if bn % 5 == 0:
                                    if flood_w[0] == 0:
                                        clean_b[0] += 1
                                        if clean_b[0] >= 3:
                                            await update_sem(min(max_c, cur_c_i[0] + 1))
                                            clean_b[0] = 0
                                    flood_w[0] = 0
                                await asyncio.sleep(0.1)
                        break  # iterator exhausted — genuinely finished
                    except FloodWaitError as e:
                        # iter_messages is single-shot: once the flood-wait
                        # escapes, that iterator is dead. Rebuild it from the
                        # resume cursor instead of silently ending the run.
                        stats.flood_waits += 1
                        log.warning(
                            "Producer flood wait %ds @ id=%d — resuming", e.seconds, last)
                        await update_sem(max(min_c, cur_c_i[0] - 3))
                        await asyncio.sleep(min(max(e.seconds, 1), 300))
                        continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Surface producer failures instead of masquerading as success:
                # a swallowed error here means "Done ✅" with a partial index.
                log.error("Producer: %s", exc, exc_info=True)
                raise
            finally:
                # Wait for outstanding downloads...
                if tasks:
                    try:
                        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
                    except asyncio.CancelledError:
                        for t in tasks:
                            t.cancel()
                        raise
                    except Exception:
                        pass
                # Emit the sentinel. Bounded so that a dead consumer can never
                # wedge the producer here. Note this deliberately does NOT drop
                # queued rows to make room: they are still consumed before the
                # sentinel is reached, and dropping one would leak its .jpg and
                # leave its id stuck in `pending`.
                try:
                    await asyncio.wait_for(eq.put(None), timeout=120)
                except asyncio.TimeoutError:
                    log.warning("Producer: consumer never drained the queue; "
                                "some downloads may be lost")
                except asyncio.CancelledError:
                    raise

        async def consumer():
            bi, bp, bm = [], [], []
            # Backstop only: the sentinel (None) is the authoritative terminator.
            # The idle timeout guards against a producer that died without
            # emitting one. It is sized generously because it resets only on a
            # successful get: the producer can chain flood-waits (each capped at
            # 300 s) plus its own per-download flood sleeps, so a long enough
            # chained gap would otherwise make the consumer "time out" first —
            # after which the producer blocks forever on a full queue.
            idle_timeout = 7200
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(eq.get(), timeout=idle_timeout)
                    except asyncio.TimeoutError:
                        log.warning("Consumer idle %ds — producer stalled?", idle_timeout)
                        break
                    if item is not None:
                        di, tp, mt = item
                        bi.append(di)
                        bp.append(tp)
                        bm.append(mt)
                        if len(bp) < actual_batch:
                            continue
                    if bp:
                        persisted = await self._flush_locked(bi, bp, bm)
                        stats.embedded += len(persisted)
                        # Only messages that actually produced a vector may be
                        # released: a row whose embedding was None is still
                        # unpersisted. Track those as failed so they don't pin
                        # the watermark forever, but still remove them from pending.
                        batch_msg_ids = [m.get("message_id") for m in bm]
                        persisted_set = set(persisted)
                        for mid in batch_msg_ids:
                            if mid is None:
                                continue
                            if mid in persisted_set:
                                pending.discard(mid)
                            else:
                                failed.add(mid)
                                pending.discard(mid)
                        await _commit_watermark()
                        log.info("  +%d total=%d %.1f/s %.1fMB %ds c=%d",
                                 len(persisted), stats.embedded, stats.speed,
                                 stats.mb_downloaded, stats.elapsed, cur_c_i[0])
                        if progress_cb:
                            try:
                                await progress_cb(stats)
                            except Exception:
                                pass
                        bi, bp, bm = [], [], []
                    if item is None:
                        break
            finally:
                # A crash here must never leak .jpg files: temp_images filling the
                # disk takes the whole box down.
                for p_ in bp:
                    try:
                        os.remove(p_)
                    except OSError:
                        pass
                # Rows already handed to the queue but never dequeued have no
                # other owner once this task exits, and the producer is cancelled
                # too — so it can never drain them either.
                while True:
                    try:
                        leftover = eq.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if leftover is None:
                        continue
                    try:
                        os.remove(leftover[1])
                    except (OSError, IndexError):
                        pass

        async def hb():
            while True:
                await asyncio.sleep(15)
                log.info("  hb dl=%d emb=%d err=%d c=%d active=%d",
                         stats.downloaded, stats.embedded, stats.errors,
                         cur_c_i[0], limiter.active)

        with self.embedder.indexing():
            p = asyncio.create_task(producer())
            c = asyncio.create_task(consumer())
            h = asyncio.create_task(hb())
            try:
                # No return_exceptions: a producer/consumer bug must fail the run.
                await asyncio.gather(p, c)
            finally:
                # Cancel, then WAIT (bounded) so unwinding tasks finish their own
                # finally blocks before this coroutine is torn down. Awaiting
                # unconditionally inside a cancelled frame is legal only because
                # _on_shutdown cancels once; asyncio.wait cannot raise.
                for t in (p, c, h):
                    t.cancel()
                await asyncio.wait({p, c, h}, timeout=10)

        # Commit the watermark even when no vectors were produced, so the
        # channel is recorded as indexed (empty runs must still register).
        try:
            await _commit_watermark()
        except Exception:
            pass
        try:
            await asyncio.to_thread(self.db.flush_state, 10.0)
        except Exception:
            pass

        log.info("DONE emb=%d skip=%d err=%d flood=%d %.1fMB %.1f/s %ds DB=%d",
                 stats.embedded, stats.skipped, stats.errors, stats.flood_waits,
                 stats.mb_downloaded, stats.speed, stats.elapsed, self.db.count())
        if stats.embedded > 0:
            try:
                models.mark_vectors_indexed()
            except Exception as exc:
                log.warning("Could not record model fingerprint: %s", exc)
        _active_runs.discard(int(ch_id))
        return stats.embedded

    def _flush(self, ids, paths, metas):
        """
        Embed a batch, upsert the results, and report which rows succeeded.

        Returns the list of message_ids that actually produced a vector. The
        caller must only discards those from `pending`: rows whose embedding
        came back None were silently skipped, and releasing them let the
        watermark jump over messages that were never persisted.
        """
        persisted: list[int] = []
        try:
            embs = self.embedder.embed_batch(paths)
            oi, oe, om = [], [], []
            for i, emb in enumerate(embs):
                if emb is not None:
                    oi.append(ids[i])
                    oe.append(emb)
                    om.append(metas[i])
                    persisted.append(metas[i].get("message_id"))
            if oi:
                self.db.upsert(oi, oe, om)
            return persisted
        finally:
            # Always clean up, even if embed_batch raised: one leaked .jpg per
            # failed batch eventually fills the disk.
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

    async def _flush_locked(self, ids, paths, metas) -> list[int]:
        """Run _flush, serialising the ChromaDB write against the monitor."""
        if self._db_lock is None:
            return await asyncio.to_thread(self._flush, ids, paths, metas)
        async with self._db_lock:
            return await asyncio.to_thread(self._flush, ids, paths, metas)
