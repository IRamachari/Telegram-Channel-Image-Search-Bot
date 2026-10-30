from __future__ import annotations
import asyncio
import io as io_module
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, CallbackQueryHandler,
    ContextTypes, filters,
)
from telegram.error import Conflict, RetryAfter

from config import config
from database import VectorDB
from embeddings import CLIPEmbedder
from searcher import ImageSearcher
import models
import system_info
import state as _state

log = logging.getLogger(__name__)

LOGIN_PHONE, LOGIN_CODE, LOGIN_2FA = range(3)
INDEX_CHANNEL, INDEX_LIMIT = range(10, 12)
IMPORT_WAITING = 20
TUNE_WAIT = 30

# A valid backup part is exactly what export_db() produces:
#   backup_<label>_<YYYYMMDD>_<HHMMSS>.partNNofMM
# Part numbers are unbounded: a store large enough for >99 parts is possible,
# and the old \d{2} made such an export impossible to import.
_BACKUP_PART_RE = re.compile(
    r"^backup_.+_\d{8}_\d{6}\.part\d+of\d+$", re.IGNORECASE)

# /import staging limits. Parts are streamed to TEMP_DIR instead of being kept
# in RAM; these caps stop a runaway upload from filling the disk before `done`.
_IMPORT_MAX_PARTS = 400
_IMPORT_MAX_BYTES = 16 * 1024 ** 3
# db_manager still accepts the archive as bytes, so the final hand-off must fit
# in memory. Refuse anything larger with a clear message instead of OOMing.
_IMPORT_MAX_HANDOFF_BYTES = 2 * 1024 ** 3


def _stage_import_part(data: bytes, parts: list) -> str:
    """
    Write one /import backup part to TEMP_DIR and return its path.

    Staging on disk keeps a multi-GB multi-part upload out of RAM; the caps
    (see _IMPORT_MAX_* above) stop a runaway or wrong-file upload from filling
    the disk before the operator types `done`.
    """
    if len(parts) >= _IMPORT_MAX_PARTS:
        raise RuntimeError(
            f"Too many parts (limit {_IMPORT_MAX_PARTS}). "
            "Start /import again and send the correct files.")
    staged = sum(os.path.getsize(p) for p, _ in parts)
    if staged + len(data) > _IMPORT_MAX_BYTES:
        raise RuntimeError(
            "Staged parts exceed the 16 GB limit. "
            "Start /import again and send the correct files.")
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="import_part_", dir=config.TEMP_DIR)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return tmp


def _discard_import_parts(parts) -> None:
    """Delete staged /import part files (best effort — never raises)."""
    for item in parts or []:
        try:
            os.unlink(item[0])
        except (OSError, IndexError, TypeError):
            pass


def _looks_like_backup_part(name: str) -> bool:
    return bool(name) and bool(_BACKUP_PART_RE.match(name.strip()))


def _cap_lines(lines: list[str], limit: int) -> list[str]:
    """Keep generated messages below Telegram's 4096-character ceiling."""
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"  … +{len(lines) - limit} more"]


def _admin(uid):
    # Fail closed: an empty ADMIN_IDS used to grant every command — including
    # /login, which rewrites the operator's session file — to every user.
    # effective_user can be None for channel-post-like contexts; deny those too
    # instead of raising AttributeError deep inside a handler.
    if uid is None:
        return False
    return config.admins_configured and uid in config.ADMIN_IDS


def _acquire_single_instance_lock():
    """
    Telegram allows exactly ONE getUpdates long-poll per bot token at a time.
    Start a second poller with the same token and Telegram 409-terminates both
    processes' polls forever — neither reliably receives updates and the log
    fills with 'Conflict: terminated by other getUpdates request'.

    This advisory lock refuses to start a second poller from the same project
    directory. The OS releases it automatically when the holder exits (even on
    SIGKILL/crash), so a dead bot never blocks the next start. The returned file
    handle must be kept alive for the whole process lifetime.
    """
    import fcntl  # POSIX-only; this bot targets Linux servers / containers
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bot.lock")
    try:
        fh = open(lock_path, "r+")
    except FileNotFoundError:
        fh = open(lock_path, "w+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            fh.seek(0)
            other = (fh.read() or "").strip() or "unknown"
        except OSError:
            other = "unknown"
        log.critical(
            "REFUSING TO START — another instance of this bot (PID %s) already"
            " runs from this directory and holds bot.lock. Telegram permits"
            " only one poller per bot token; a second one causes endless"
            " '409 Conflict' errors and a bot that never answers commands"
            " (/index, /model, ...). Find and stop the other process with:"
            "  ps aux | grep -E 'python.*(main|bot)\\.py' | grep -v grep"
            "  — then start this one again.", other)
        raise SystemExit(1)
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
    except OSError:
        pass
    return fh


def _esc(value, default: str = "") -> str:
    """Escape a dynamic value for Telegram's legacy Markdown (v1)."""
    try:
        from telegram.helpers import escape_markdown
        return escape_markdown(str(value if value not in (None, "") else default), version=1)
    except Exception:
        return str(value)


def _fmt(hits):
    if not hits:
        return "❌ No similar images found."
    lines = ["✅ *Similar images found:*\n"]
    for i, h in enumerate(hits[:10], 1):
        p = h.get("similarity", 0) * 100
        ic = "🟢" if p >= 95 else "🟡" if p >= 85 else "🟠"
        lk = " 🔒" if h.get("private") else ""
        # An absent link must not produce an empty markdown link "[..]()",
        # which Telegram rejects — fall back to plain text for that row.
        link = (h.get("link") or "").strip()
        target = f"[{h.get('message_id', 'link')}]({link})" if link else "no link"
        lines.append(f"{ic} *#{i}* — {p:.1f}%{lk}\n    📺 {_esc(h.get('channel_name'))}\n"
                     f"    🔗 {target}\n"
                     f"    📅 {(h.get('date') or '')[:10]}\n")
    if any(h.get("private") for h in hits[:10]):
        lines.append("\n🔒 = private channel")
    top_sim = hits[0].get("similarity", 0) if hits else 0
    if hits and top_sim < 0.65:
        lines.append("\n⚠️ Low confidence — no strong match in the index.")
    elif hits and top_sim < 0.80:
        lines.append("\n🟡 Moderate match — verify before acting.")
    return "\n".join(lines)


async def _edit(msg, text, **kw):
    try:
        await msg.edit_text(text, **kw)
    except RetryAfter as e:
        await asyncio.sleep(e.retry_after)
        try:
            await msg.edit_text(text, **kw)
        except Exception:
            pass
    except Exception:
        pass


def _status_text(db: VectorDB, monitor, new_images=None, removed_images=None, errors=None) -> str:
    """Build the live status block for the pinned message."""
    n = db.count()
    ch = db.list_channels()
    mon = "✅ Running" if monitor and monitor.running else "⏹ Off"
    st = monitor.stats if monitor else {"new_images": 0, "deleted_images": 0, "errors": 0}
    ni = new_images if new_images is not None else st["new_images"]
    ri = removed_images if removed_images is not None else st["deleted_images"]
    er = errors if errors is not None else st["errors"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"📊 *Image Search Bot — Live Status*\n\n"
        f"🗄 Indexed: *{n:,}* images\n"
        f"📡 Channels: *{len(ch)}*\n"
        f"🔄 Monitor: {mon}\n"
        f"📸 New since start: *{ni}*\n"
        f"🗑 Removed since start: *{ri}*\n"
        f"❌ Errors: *{er}*\n"
        f"🕐 `{ts}`"
    )


class SearchBot:
    # Maximum number of users whose duplicate results we keep in memory.
    _DUP_CACHE_MAX = 5

    def __init__(self):
        log.info("Initialising …")
        # Runs before the event loop is entered, so the sqlite connect + Chroma
        # open here cannot block update handling.
        self.db = VectorDB()
        self.embedder = CLIPEmbedder()
        self.searcher = ImageSearcher(self.db, self.embedder)
        self._tasks: set[asyncio.Task] = set()
        self._dup_cache: dict[int, list] = {}
        self._monitor = None
        # /import staging: uid -> [(temp_path, original_name)]. Parts live on
        # disk, never in RAM, so a multi-GB backup cannot OOM the process.
        self._import_parts: dict[int, list[tuple[str, str]]] = {}
        self._app: Application | None = None
        # Pinned message registry persisted across restarts: {str(chat_id): message_id}
        self._pinned: dict[str, int] = _state.get("pinned", {})
        if not isinstance(self._pinned, dict):
            self._pinned = {}
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        # Serialise ALL ChromaDB writes — prevents "database is locked" errors
        self._db_lock: asyncio.Lock = asyncio.Lock()
        # Excludes /import (which replaces the chroma_db directory on disk) from
        # any concurrently-running index/sync. Unlike _db_lock this is held for a
        # whole logical operation, which is why the indexer must NOT take it.
        self._write_gate: asyncio.Lock = asyncio.Lock()
        # Telethon bot client (MTProto) used to download user-sent files with
        # NO 20 MB limit. Started lazily on first use.
        self._mtproto = None
        self._mtproto_lock: asyncio.Lock = asyncio.Lock()

    def _track(self, coro):
        """
        Keep a strong reference to a background task.

        Accepts either a coroutine or an already-created Task: `asyncio.create_task`
        raises TypeError on a Task, which previously made _track() unusable for
        tracked-but-externally-created tasks (and silently killed /duplicates).
        """
        if isinstance(coro, asyncio.Task):
            t = coro
        else:
            t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    # ── MTProto downloads (no 20 MB Bot-API limit) ──────────────────────────────

    async def _get_mtproto(self):
        """
        Lazily start a Telethon client logged in as the bot (over MTProto).
        Reused for the lifetime of the process. Returns None if it can't start
        (e.g. missing API credentials), so callers can fall back to the Bot API.
        """
        if self._mtproto is not None:
            return self._mtproto
        async with self._mtproto_lock:
            if self._mtproto is not None:
                return self._mtproto
            if config.API_ID == 0 or not config.API_HASH or not config.BOT_TOKEN:
                log.warning("MTProto downloader unavailable: set TELEGRAM_API_ID, "
                            "TELEGRAM_API_HASH and BOT_TOKEN to download files >20 MB.")
                return None
            client = None
            try:
                from tg_client import make_bot_client
                client = make_bot_client()
                await client.start(bot_token=config.BOT_TOKEN)
                self._mtproto = client
                log.info("MTProto download client started (bot session).")
            except Exception as e:
                # A client that connected but failed to authorise keeps its
                # socket and <name>_bot.session handle open. Leaking one per
                # retry eventually locks the session file for good.
                log.error("Could not start MTProto download client: %s", e)
                if client is not None:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                self._mtproto = None
            return self._mtproto

    async def _resolve_mtproto_peer(self, client, chat_id: int):
        """
        Resolve a Bot-API chat id to an MTProto input peer.

        Telethon can fetch a private chat's user with access_hash=0 because we
        are a bot, and a basic group by id. A supergroup/channel needs an access
        hash this receive_updates=False client does not have — retry via
        get_entity so the failure is explicit rather than a bare ValueError.
        """
        try:
            return await client.get_input_entity(chat_id)
        except Exception:
            return await client.get_entity(chat_id)

    async def _download_via_mtproto(self, chat_id: int, message_id: int, path: str) -> str:
        """Download a user-sent message's media straight to `path` over MTProto."""
        client = await self._get_mtproto()
        if client is None:
            raise RuntimeError("MTProto client unavailable")
        peer = await self._resolve_mtproto_peer(client, chat_id)
        msgs = await client.get_messages(peer, ids=message_id)
        msg = msgs[0] if isinstance(msgs, list) else msgs
        if msg is None or not getattr(msg, "media", None):
            raise RuntimeError("media not found for message")
        got = await client.download_media(msg, file=path)
        if not got or not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise RuntimeError("empty download")
        return path

    async def _fetch_media_to_path(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                                   file_id: str, path: str, size_bytes: int = 0) -> str:
        """
        Save a file the user just sent to `path`. Prefers MTProto (no size cap);
        falls back to the Bot API for small files when MTProto is unavailable.
        Streaming to disk keeps a 2 GB photo or backup out of the event loop's
        memory (the old bytes API held the whole file in RAM).
        """
        chat_id = update.effective_chat.id
        message_id = update.effective_message.message_id
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            return await self._download_via_mtproto(chat_id, message_id, path)
        except Exception as e:
            big = size_bytes and size_bytes > 19 * 1024 * 1024
            if big:
                raise RuntimeError(
                    "file is larger than 20 MB and the MTProto downloader could "
                    "not fetch it — check TELEGRAM_API_ID / TELEGRAM_API_HASH / "
                    f"BOT_TOKEN (and that the bot can see this chat). ({e})"
                )
            log.warning("MTProto download failed (%s); falling back to Bot API.", e)
            tf = await ctx.bot.get_file(file_id)
            await tf.download_to_drive(custom_path=path)
            return path

    async def _fetch_media_bytes(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                                 file_id: str, size_bytes: int = 0) -> bytes:
        """
        Get the bytes of a file the user just sent. Streams to a temp file first
        (so a multi-GB photo never sits in RAM), then reads it back and deletes
        it. Prefers MTProto (no size cap); falls back to the Bot API for small
        files when MTProto is unavailable. Raises a clear error if a large file
        can't be fetched.
        """
        fd, tmp = tempfile.mkstemp(prefix="media_", dir=config.TEMP_DIR)
        os.close(fd)
        try:
            await self._fetch_media_to_path(
                update, ctx, file_id, tmp, size_bytes=size_bytes)
            with open(tmp, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ── Startup hook ──────────────────────────────────────────────────────────

    async def _on_startup(self, app: Application) -> None:
        """
        Called automatically after the bot connects.

        Boots the auto-monitor and the offline-backlog sync. Both run in a
        guarded background task rather than inline: an exception escaping
        post_init propagates out of run_polling() and kills the process, and a
        long inline sync delayed the start of polling by hours.
        """
        self._app = app
        self._track(self._bootstrap(app))

    async def _bootstrap(self, app: Application) -> None:
        try:
            log.info("Startup: checking prerequisites …")

            from indexer import session_exists
            if not session_exists():
                log.info("Startup: no session — skipping auto-monitor/sync.")
                return
            if not self.embedder.is_ready():
                log.info("Startup: model not ready — skipping auto-monitor/sync.")
                return
            try:
                empty = self.db.count() == 0
            except Exception as e:
                log.error("Startup: DB unreadable, skipping auto-monitor/sync: %s", e)
                return
            if empty:
                log.info("Startup: DB empty — skipping auto-monitor/sync.")
                return

            admin_chat = config.ADMIN_IDS[0] if config.ADMIN_IDS else None

            # A model swap invalidates existing vectors: a dimension change
            # makes ChromaDB raise on the next insert, and a same-dimension
            # swap (B/32 vs B/16) silently produces meaningless similarity.
            try:
                if models.model_changed():
                    stored = models.read_model_fingerprint() or {}
                    now = models.model_fingerprint()
                    log.warning(
                        "Model fingerprint mismatch: indexed=%s on-disk=%s — "
                        "a full re-index is required for correct results.",
                        stored, now)
                    if admin_chat:
                        await app.bot.send_message(
                            admin_chat,
                            "⚠️ *CLIP model changed* since these vectors were "
                            f"indexed.\nIndexed with: `{stored.get('active')}` "
                            f"({stored.get('size_mb')} MB)\n"
                            f"Now on disk: `{now.get('active')}` "
                            f"({now.get('size_mb')} MB)\n\n"
                            "Search results are unreliable until you `/index` again.",
                            parse_mode="Markdown")
            except Exception as e:
                log.warning("Model fingerprint check failed: %s", e)

            # The monitor and the ephemeral indexing clients contend on a
            # file-based Telethon session; make it visible before it bites.
            try:
                import tg_client
                tg_client.warn_if_file_session_shared()
            except Exception:
                pass

            # 1. Startup sync FIRST — then start monitor so they don't collide
            try:
                channels = self.db.list_channels()
            except Exception as e:
                log.error("Startup: channel list unreadable: %s", e)
                channels = {}
            log.info("Startup sync: %d channel(s) …", len(channels))
            total_new, failures = await self._sync_channels(list(channels.keys()))
            log.info("Startup sync done: +%d new image(s), %d failure(s)",
                     total_new, failures)

            # 2. Start monitor (after sync so initial write burst is done)
            await self._start_monitor(app)

            # 3. Notify admin — report the ACTUAL outcome rather than always
            # claiming success. A failed sync used to be invisible here.
            if admin_chat:
                try:
                    if failures:
                        label = ("⚠️ *Startup sync incomplete*\n"
                                 f"📸 *{total_new}* new image(s)\n"
                                 f"❌ *{failures}* channel(s) failed\n")
                    elif total_new > 0:
                        label = ("🔄 *Startup sync complete!*\n"
                                 f"📸 *{total_new}* new image(s) indexed while offline\n")
                    else:
                        label = "✅ *Bot started* — monitor active, DB up to date.\n"
                    if self._monitor and self._monitor.running:
                        mon = "✅ running"
                    else:
                        mon = "⏹ not running (see /session)"
                    await app.bot.send_message(
                        admin_chat,
                        f"{label}📡 Monitor: {mon}\n"
                        f"📊 Total DB: *{self.db.count():,}*",
                        parse_mode="Markdown",
                    )
                except Exception as e:
                    log.debug("Startup notify error: %s", e)

            await self._update_pinned_message(app.bot)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Never let startup kill the process: the bot keeps serving
            # commands and the monitor simply stays off this boot.
            log.error("Startup sync/monitor failed: %s", e, exc_info=True)

    async def _start_monitor(self, app: Application) -> None:
        """(Re)create and start the auto-monitor. Safe to call repeatedly."""
        if self._monitor is not None:
            try:
                await self._monitor.stop()
            except Exception:
                pass
            self._monitor = None
        from monitor import ChannelMonitor
        monitor = ChannelMonitor(self.db, self.embedder, db_lock=self._db_lock)

        async def notify(text: str) -> None:
            for aid in (config.ADMIN_IDS or []):
                try:
                    await app.bot.send_message(aid, text, parse_mode="Markdown")
                except Exception:
                    pass
            await self._update_pinned_message(app.bot)

        result = await monitor.start(notify_callback=notify)
        self._monitor = monitor if monitor.running else None
        log.info("Auto-monitor: %s", result)

    async def _on_shutdown(self, app: Application) -> None:
        """Stop background work, then disconnect clients cleanly."""
        # Cancel first, then await: a task killed mid-ChromaDB-write leaves the
        # sqlite file in a state only WAL recovery can repair.
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            # Bounded wait. A task parked in `await asyncio.to_thread(...)` (the
            # duplicate scan, a multi-GB /import, an export) cannot be
            # interrupted by cancel(), so gathering unconditionally made shutdown
            # block for as long as that thread — past Docker's grace period, so
            # the process was SIGKILLed mid-write, exactly the corruption this
            # was meant to avoid.
            done, pending = await asyncio.wait(
                list(self._tasks), timeout=30)
            if pending:
                log.warning("Shutdown: %d task(s) still running, abandoning: %s",
                            len(pending), [t.get_name() for t in pending])
        self._tasks.clear()

        if self._monitor is not None:
            try:
                await self._monitor.stop()
            except Exception:
                pass
            self._monitor = None

        if self._mtproto is not None:
            try:
                await self._mtproto.disconnect()
            except Exception:
                pass
            self._mtproto = None

        try:
            await asyncio.to_thread(self.db.close, True, 10.0)
        except Exception:
            pass

    async def _on_error(self, update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Global error handler. Without this, any exception in a handler — most
        commonly a Telegram 'Can't parse entities' Markdown error from dynamic
        text — is logged as 'No error handlers are registered' and the user
        sees nothing. Here we log it and try to reply once in PLAIN text so the
        reply itself can't fail to parse.

        `update` is None for polling errors and for errors raised in background
        jobs/coroutines, so it must be handled defensively.
        """
        # 409 Conflict = a second poller owns the token. The poll loser
        # receives NOTHING — retrying forever just keeps the war alive.
        if isinstance(ctx.error, Conflict):
            self._on_conflict()
            return
        log.error("Handler error: %s", ctx.error, exc_info=ctx.error)
        msg = getattr(update, "effective_message", None) if update is not None else None
        if msg is None:
            return
        try:
            from telegram.error import BadRequest
            if isinstance(ctx.error, BadRequest) and "parse entities" in str(ctx.error):
                # The original reply failed Markdown parsing; nothing to resend,
                # but tell the user *something* rather than staying silent.
                try:
                    await msg.reply_text(
                        "⚠️ Could not format that reply. Please try again.")
                except Exception:
                    pass
                return
            await msg.reply_text("⚠️ Something went wrong handling that. Please try again.")
        except Exception:
            pass

    def _on_conflict(self) -> None:
        """
        Another poller owns this bot token RIGHT NOW.

        The file lock in run() already refuses a second instance from THIS
        directory; what is left is a stale process elsewhere (old tmux/screen
        pane, an auto-restarted systemd/docker copy) or the same token on
        another machine. A poll loser receives zero updates, so exiting is
        the only correct action — retrying forever disturbs the winner and
        floods the log.
        """
        if getattr(self, "_conflict_seen", False):
            return          # already shutting down — log the reason only once
        self._conflict_seen = True
        log.critical(
            "409 Conflict: another process is polling this bot token. This"
            " instance receives no updates and is exiting. Find the other"
            " poller with:  ps aux | grep -E 'python.*(main|bot)\\.py' |"
            " grep -v grep   — also check other servers, tmux/screen panes,"
            " docker containers and auto-restarting services using the same"
            " BOT_TOKEN.")
        try:
            if self._app is not None and hasattr(self._app, "stop_running"):
                self._app.stop_running()
                # Graceful: run_polling() unwinds → post_shutdown runs. Safety
                # net: if it has not exited within 15 s (a hung task),
                # hard-exit so this process can never linger as a silent
                # poll loser that keeps disturbing the winner.
                timer = threading.Timer(15.0, os._exit, [1])
                timer.daemon = True
                timer.start()
                return
        except Exception:
            pass
        os._exit(1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _sync_channels(self, channel_ids: list):
        """
        Index images newer than the stored watermark for each channel.

        The write gate is held so an `/import` landing mid-sync cannot move
        `chroma_db` out from under the running indexer (which would silently
        discard every vector it flushes). The per-batch `_db_lock` stays
        un-held across the loop, since it would otherwise starve the monitor.

        Returns (new_images, failures). Counting failures lets startup report
        "sync incomplete" instead of always claiming success.
        """
        total = 0
        failures = 0
        for ch_id_str in channel_ids:
            try:
                from indexer import ChannelIndexer
                async with self._write_gate:
                    count = await ChannelIndexer(
                        self.db, self.embedder, db_lock=self._db_lock,
                    ).index_channel(int(ch_id_str), limit=None, progress_callback=None)
                total += count
            except Exception as e:
                failures += 1
                log.error("Sync channel %s: %s", ch_id_str, e)
        return total, failures

    async def _update_pinned_message(self, bot) -> None:
        """Edit the pinned status message in every registered chat."""
        if not self._pinned:
            return
        # Compute once, per-chat-failure-tolerant: a single bad chat used to
        # abort the whole loop, and _status_text itself raises once db.close()
        # has run or a rebind failed.
        try:
            text = _status_text(self.db, self._monitor)
        except Exception as e:
            log.debug("Pinned status build failed: %s", e)
            return
        for chat_id_str, msg_id in list(self._pinned.items()):
            try:
                await bot.edit_message_text(
                    chat_id=int(chat_id_str),
                    message_id=msg_id,
                    text=text,
                    parse_mode="Markdown",
                )
            except Exception as e:
                log.debug("Pinned update [%s]: %s", chat_id_str, e)

    # ── /start ────────────────────────────────────────────────────────────────

    async def cmd_start(self, update: Update, _):
        from indexer import session_exists
        s = "✅" if session_exists() else "❌ → /login"
        m = "✅" if self.embedder.is_ready() else "❌ → /model"
        a = models.get_active_model() or "none"
        n = self.db.count()
        mon = "✅ Running" if self._monitor and self._monitor.running else "⏹ Off"
        await update.message.reply_text(
            f"🔍 *Image Search Bot*\n\n"
            f"*Status:*\n  Session: {s}\n  Model: {m} (`{a}`)\n"
            f"  Database: *{n:,}* vectors\n  Monitor: {mon}\n\n"
            f"*Setup:* /login → /model → /index\n"
            f"*Search:* Send photo or video\n\n"
            f"*Tools:*\n"
            f"/monitor — monitor status / toggle\n"
            f"/checkupdates — sync new images now\n"
            f"/channels — manage indexed channels\n"
            f"/setpinned — pin live status message\n"
            f"/cleanup — remove deleted vectors\n"
            f"/duplicates — find duplicates\n"
            f"/export — download DB (single file)\n"
            f"/autobackup — nightly DB backup (on/off/now)\n"
            f"/import — upload DB\n"
            f"/tune — tuning guide + override\n"
            f"/system — server stats\n"
            f"/stats — DB stats",
            parse_mode="Markdown",
        )

    # ── /session ──────────────────────────────────────────────────────────────

    async def cmd_session(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        from indexer import session_info, check_session
        si = session_info()
        v, d = await check_session()
        mode = "String session (env)" if si.get("string_session") else "Session file"
        # Plain text (no Markdown): si['path'], the account name, and any error
        # string in `d` are arbitrary and must not be parsed as Markdown.
        await update.message.reply_text(
            f"🔐 Session\n"
            f"Mode: {mode}\n"
            f"{si['path']}\n"
            f"{'✅' if si['exists'] else '❌'} | Auth: {'✅' if v else '❌'}\n"
            f"{d}"
        )

    # ── /stats ────────────────────────────────────────────────────────────────

    async def cmd_stats(self, update: Update, _):
        n = self.db.count()
        ch = self.db.list_channels()
        # Fallback: derive distinct channel ids from vectors when watermarks
        # are missing, so the bot still reports channels that exist in the DB.
        if not ch and n > 0:
            try:
                ids, metas = self.db.paged_get(include=["metadatas"])
                seen = set()
                for m in metas:
                    cid = m.get("channel_id") if isinstance(m, dict) else None
                    if cid is not None:
                        seen.add(str(cid))
                if seen:
                    ch = {c: 0 for c in seen}
            except Exception:
                pass
        lines = [f"📊 *{n:,}* vectors\n"]
        for c, w in ch.items():
            lines.append(f"  • `{c}` → `{w}`")
        if not ch:
            lines.append("  No channels.")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ── /system ───────────────────────────────────────────────────────────────

    async def cmd_system(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        text = await asyncio.to_thread(system_info.format_system_stats)
        from indexer import get_speed_info
        sp = get_speed_info()
        mon = "✅" if self._monitor and self._monitor.running else "⏹"
        text += (f"\n⚡ cryptg: {'✅' if sp['cryptg'] else '❌'} | "
                 f"parallel: {sp['parallel_files']} | batch: {sp['batch_size']}\n"
                 f"🔄 Tasks: {len(self._tasks)} | Monitor: {mon}\n")
        await update.message.reply_text(text, parse_mode="Markdown")

    # ── /tune ─────────────────────────────────────────────────────────────────

    async def _tune_apply(self, args) -> str:
        """
        Apply /tune overrides and return the resulting status text. Shared by
        the direct-with-args path and the interactive 60 s window.

        Raises ValueError on bad input (caller formats it).
        """
        from auto_scale import (_ALIASES, apply_tune, format_auto_tune,
                                reset_tune, set_tune)
        wanted: dict = {}
        do_reset = False
        for a in args:
            if a.lower() == "reset":
                do_reset = True
                continue
            k, sep, v = a.partition("=")
            if not sep:
                raise ValueError(f"'{a}' is not key=value")
            wanted[_ALIASES.get(k.lower(), k.lower())] = int(v)
        if do_reset and wanted:
            raise ValueError("reset cannot be combined with overrides")
        if do_reset:
            await asyncio.to_thread(reset_tune)
        elif wanted:
            await asyncio.to_thread(set_tune, **wanted)
        # Re-apply so _applied reflects the overrides for this model size
        # (set_tune only persists; apply_tune refreshes the live values).
        applied = await asyncio.to_thread(apply_tune, self.embedder.model_size_mb)
        # Thread counts are baked into the ONNX session at build time, so a
        # changed intra/inter needs a session rebuild. Batch size and parallel
        # downloads are read per-run and take effect immediately.
        old = self.embedder._thread_tune
        new = (applied.get("intra_threads", 4), applied.get("inter_threads", 1))
        note = ""
        if old != new:
            if self.embedder.is_ready():
                ok = await asyncio.to_thread(self.embedder.reload)
                note = (
                    f"\n✅ Model rebuilt with {new[0]} intra / {new[1]} inter threads."
                    if ok else
                    "\n⚠️ Model reload failed — previous session kept if any.")
            else:
                note = ("\nℹ️ No model loaded — threads apply at the next "
                        "model load.")
        text = await asyncio.to_thread(format_auto_tune, self.embedder.model_size_mb)
        return text + note

    _TUNE_GUIDE = (
        "⚙️ *Tuning guide*\n\n"
        "Settings are detected automatically from your CPU/RAM/container "
        "limits at startup and on every /model switch — use /tune only to "
        "override them.\n\n"
        "*How to tune*\n"
        "1. Read the detected values from the last tune status.\n"
        "2. Pick values for your box:\n"
        "   • `intra` = number of PHYSICAL CPU cores (max 16) — biggest speedup\n"
        "   • `inter` = 1 (keep 1 unless you have spare cores)\n"
        "   • `batch` = 8 for ViT-L/14 (293 MB), 32 for ViT-B, 64 for tiny models\n"
        "   • `parallel` = CPU cores ×2, capped at 16\n"
        "3\\. *Reply to this message* with your settings within 60 s "
        "(or use `/tune intra=4 inter=1 batch=8 parallel=8` directly)\n"
        "4. `reset` — discard overrides, back to auto-detect\n\n"
        "*Examples*\n"
        "`intra=2 batch=4` — small 1-CPU box\n"
        "`intra=8 batch=16 parallel=12` — big server\n"
        "`reset` — back to auto-detect\n\n"
        "Bounds: intra 1-32, inter 1-8, batch 1-256, parallel 1-32.\n"
        "Thread changes rebuild the model session automatically.\n\n"
        "⏳ *Send your settings now — this window closes in 60 seconds.* "
        "(/cancel to leave)"
    )

    async def cmd_tune(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """
        /tune — tuning guide + interactive 60 s input window.

        Cached manual overrides are persisted in tune.json so they survive
        restarts (auto_scale loads the file at import, so a fresh process
        applies the same values on first load).
        """
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        args = list(ctx.args or []) if ctx else []
        if args:
            # Direct application: /tune intra=2 batch=8 / /tune reset
            try:
                text = await self._tune_apply(args)
            except ValueError as e:
                await update.message.reply_text(f"⚠️ {e}")
            except Exception as e:
                await update.message.reply_text(f"⚠️ {e}")
            else:
                await update.message.reply_text(text, parse_mode="Markdown")
            return ConversationHandler.END
        await update.message.reply_text(self._TUNE_GUIDE, parse_mode="Markdown")
        return TUNE_WAIT

    async def tune_input(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Plain-text reply inside the /tune window: parse + apply + END."""
        text = update.message.text.strip()
        if not text:
            return TUNE_WAIT
        try:
            applied = await self._tune_apply(text.split())
        except ValueError as e:
            await update.message.reply_text(
                f"⚠️ {e}\n\nExamples:\n`intra=2 batch=4`\n`intra=8 batch=16 parallel=12`\n`reset`",
                parse_mode="Markdown")
            return TUNE_WAIT
        except Exception as e:
            await update.message.reply_text(f"⚠️ {e}", parse_mode="Markdown")
            return ConversationHandler.END
        await update.message.reply_text(applied, parse_mode="Markdown")
        return ConversationHandler.END

    async def tune_timeout(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """The 60 s /tune input window expired."""
        eff = getattr(update, "effective_message", None)
        if eff is not None:
            try:
                await eff.reply_text(
                    "⌛ Tune window ended (60 s). Run /tune again.",
                    parse_mode="Markdown")
            except Exception:
                pass
        return ConversationHandler.END

    # ── /cancel ───────────────────────────────────────────────────────────────

    async def cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        cl = ctx.user_data.pop("tg_client", None)
        if cl:
            try:
                await cl.disconnect()
            except Exception:
                pass
        _discard_import_parts(self._import_parts.pop(update.effective_user.id, None))
        await update.message.reply_text("Cancelled.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    async def on_conv_timeout(self, update, ctx: ContextTypes.DEFAULT_TYPE):
        """
        Conversation timeout handler. Without it, a 5-minute idle conversation
        is simply forgotten: the connected Telethon client stored in user_data
        by the login flow (plus its open sqlite handle) and any queued import
        parts are never released.
        """
        try:
            cl = ctx.user_data.pop("tg_client", None)
            if cl:
                try:
                    await cl.disconnect()
                except Exception:
                    pass
            ctx.user_data.pop("tg_phone", None)
            ctx.user_data.pop("tg_hash", None)
            if update is not None:
                eff = getattr(update, "effective_user", None)
                if eff is not None:
                    _discard_import_parts(
                        self._import_parts.pop(eff.id, None))
        except Exception as e:
            log.debug("Conversation timeout cleanup: %s", e)
        return ConversationHandler.END

    # ── /login ────────────────────────────────────────────────────────────────

    async def login_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        if config.API_ID == 0:
            await update.message.reply_text("❌ Set API\\_ID.", parse_mode="Markdown")
            return ConversationHandler.END
        from tg_client import has_string_session
        if has_string_session():
            await update.message.reply_text(
                "ℹ️ A `TELETHON_STRING_SESSION` is already configured, so login "
                "isn't needed.\nUse /session to verify it.",
                parse_mode="Markdown")
            return ConversationHandler.END
        await update.message.reply_text(
            "🔐 Phone with country code.\n`+1234567890`\n/cancel",
            parse_mode="Markdown")
        return LOGIN_PHONE

    async def login_phone(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        phone = update.message.text.strip()
        try: await update.message.delete()
        except Exception: pass
        msg = await update.message.reply_text("📱 Sending …")
        from telethon import TelegramClient
        cl = TelegramClient(config.SESSION_PATH, config.API_ID, config.API_HASH)
        try:
            await cl.connect()
            sent = await cl.send_code_request(phone)
        except Exception as e:
            try: await cl.disconnect()
            except Exception: pass
            await msg.edit_text(f"❌ `{e}`", parse_mode="Markdown")
            return ConversationHandler.END
        ctx.user_data.update({"tg_client": cl, "tg_phone": phone, "tg_hash": sent.phone_code_hash})
        await msg.edit_text("✅ Enter code.\n/cancel", parse_mode="Markdown")
        return LOGIN_CODE

    async def login_code(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        code = update.message.text.strip().replace(" ", "").replace("-", "")
        try: await update.message.delete()
        except Exception: pass
        cl = ctx.user_data.get("tg_client")
        if not cl:
            await update.message.reply_text("❌ /login again.", parse_mode="Markdown")
            return ConversationHandler.END
        from telethon.errors import SessionPasswordNeededError
        try:
            await cl.sign_in(phone=ctx.user_data["tg_phone"], code=code,
                             phone_code_hash=ctx.user_data["tg_hash"])
        except SessionPasswordNeededError:
            await update.message.reply_text("🔒 2FA password?", parse_mode="Markdown")
            return LOGIN_2FA
        except Exception as e:
            await cl.disconnect()
            ctx.user_data.pop("tg_client", None)
            await update.message.reply_text(f"❌ `{e}`", parse_mode="Markdown")
            return ConversationHandler.END
        # get_me() can raise; without a finally the authorised client (and its
        # open sqlite session handle) stayed connected until the conversation
        # timed out.
        try:
            me = await cl.get_me()
            name = me.first_name if me else "unknown"
        finally:
            try:
                await cl.disconnect()
            except Exception:
                pass
            ctx.user_data.pop("tg_client", None)
        await update.message.reply_text(f"✅ *{name}*", parse_mode="Markdown")
        return ConversationHandler.END

    async def login_2fa(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        pw = update.message.text.strip()
        try: await update.message.delete()
        except Exception: pass
        cl = ctx.user_data.get("tg_client")
        if not cl:
            return ConversationHandler.END
        try:
            await cl.sign_in(password=pw)
        except Exception as e:
            await cl.disconnect()
            ctx.user_data.pop("tg_client", None)
            await update.message.reply_text(f"❌ `{e}`", parse_mode="Markdown")
            return ConversationHandler.END
        # Same finally as login_code: never leave an authorised client dangling.
        try:
            me = await cl.get_me()
            name = me.first_name if me else "unknown"
        finally:
            try:
                await cl.disconnect()
            except Exception:
                pass
            ctx.user_data.pop("tg_client", None)
        await update.message.reply_text(f"✅ *{name}*", parse_mode="Markdown")
        return ConversationHandler.END

    # ── /model ────────────────────────────────────────────────────────────────

    async def cmd_model(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        st = await asyncio.to_thread(system_info.get_system_stats)
        ram = st["ram_total_mb"]
        lines = ["🧠 *Models*\n", f"_RAM: {ram:.0f} MB_\n"]
        buttons = []
        for k, info in models.AVAILABLE.items():
            lines.append(models.format_model_card(k))
            lines.append("")
            f = "✅" if ram >= info["ram_mb"] else "⚠️"
            buttons.append([InlineKeyboardButton(
                f"{info['name']} ({info['size_mb']}MB) {f}", callback_data=f"dl_{k}")])
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown",
                                        reply_markup=InlineKeyboardMarkup(buttons))

    async def model_cb(self, update: Update, _):
        q = update.callback_query
        await q.answer()
        if not _admin(q.from_user.id):
            return
        key = q.data.replace("dl_", "")
        if key not in models.AVAILABLE:
            return
        info = models.AVAILABLE[key]
        old = models.get_active_model()
        if self.embedder.is_busy():
            await q.edit_message_text(
                "⚠️ An index run is in progress — wait for it to finish "
                "before switching models.", parse_mode="Markdown")
            return
        await q.edit_message_text(f"⬇️ *{info['name']}* …", parse_mode="Markdown")
        try:
            r = await asyncio.to_thread(models.download_model, key)
            if self.embedder.is_busy():
                raise RuntimeError("an index run started during download; model not switched")
            # ONNX session construction takes seconds — never on the event loop.
            reloaded = await asyncio.to_thread(self.embedder.reload)
            from auto_scale import apply_tune
            apply_tune(self.embedder.model_size_mb)
            if not reloaded:
                raise RuntimeError(self.embedder.last_error or "model reload failed")
            models.write_model_fingerprint()
            if old and old != key and self.db.count() > 0:
                w = ("\n\n⚠️ *Model changed* - existing vectors are not comparable.\n"
                     "Run /index again for reliable results.")
            else:
                w = ""
            await q.edit_message_text(
                f"{r}\nDim: `{self.embedder.get_dim()}`\n"
                f"Input: *{self.embedder.input_size}px* \\({self.embedder.crop_mode} crop\\){w}",
                parse_mode="Markdown")
        except Exception as e:
            await q.edit_message_text(f"❌ `{e}`", parse_mode="Markdown")

    # ── /export ───────────────────────────────────────────────────────────────

    _HTTP_UPLOAD_SAFE = 45 * 1024 * 1024  # Bot API sendDocument cap ≈ 50 MB

    async def cmd_export(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        if self.db.count() == 0:
            return await update.message.reply_text("❌ Empty DB.")
        msg = await update.message.reply_text("📦 Preparing …")
        try:
            from db_manager import export_db_single, split_backup
            # The zip is built from the live directory, so it must not overlap a
            # write: a page captured mid-transaction yields a silently corrupt
            # backup that only fails at /import time.
            async with self._db_lock:
                data, name, stats = await asyncio.to_thread(export_db_single, self.db)
            mb = len(data) / 1024 / 1024
            caption = (f"🗄 Backup: {stats['channels']}\n"
                       f"Vectors: {stats['vectors']:,} | {mb:.1f} MB")
            # Single file, ALWAYS — the /import side takes the native zip in
            # one piece (MTProto download has no size cap).
            if len(data) <= self._HTTP_UPLOAD_SAFE:
                await _edit(msg, f"📦 Uploading ({mb:.0f} MB) …", parse_mode="Markdown")
                doc = io_module.BytesIO(data)
                doc.name = name
                await update.message.reply_document(document=doc, caption=caption)
                await _edit(msg, "✅ Export sent!", parse_mode="Markdown")
                return
            client = await self._get_mtproto()
            if client is not None:
                try:
                    await _edit(msg, f"📦 Uploading single file via MTProto "
                                     f"({mb:.0f} MB, no Telegram limit) …",
                                parse_mode="Markdown")
                    peer = await self._resolve_mtproto_peer(
                        client, update.effective_chat.id)
                    tmpdir = tempfile.mkdtemp(prefix="exp_", dir=config.TEMP_DIR)
                    try:
                        tmp = Path(tmpdir) / name
                        try:
                            tmp.write_bytes(data)
                        except OSError as exc:
                            raise RuntimeError(f"could not stage export: {exc}")
                        await client.send_file(
                            peer, str(tmp), caption=caption, force_document=True)
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                    await _edit(msg, "✅ Export sent (single file)!", parse_mode="Markdown")
                    return
                except Exception as exc:
                    log.warning("MTProto export failed (%s) — falling back to parts", exc)
                    await _edit(msg, "⚠️ Uploading via MTProto failed — "
                                     "sending parts as fallback …",
                                parse_mode="Markdown")
            else:
                log.warning("No MTProto client — exporting in parts.")
            # Fallback: legacy multi-part chunks (works even without
            # API_ID/API_HASH). Parts are sliced from the already-built zip,
            # never rebuilt.
            parts = split_backup(data, name, stats)
            total_parts = len(parts)
            for i, (pdata, pname, _) in enumerate(parts, 1):
                doc = io_module.BytesIO(pdata)
                doc.name = pname
                part_mb = len(pdata) / 1024 / 1024
                await update.message.reply_document(
                    document=doc,
                    caption=f"📦 Part {i}/{total_parts} ({part_mb:.1f} MB)",
                )
                await asyncio.sleep(1)
            await _edit(msg,
                f"⚠️ Telegram blocked the single {mb:.0f} MB upload — "
                f"*{total_parts}* parts sent.\n"
                f"To import: /import → send all parts in order.",
                parse_mode="Markdown")
        except Exception as e:
            log.error("Export: %s", e, exc_info=True)
            await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")

    # ── /autobackup — nightly one-file backup ─────────────────────────────────

    _AUTOBACKUP_JOB_NAME = "auto_backup"

    def _backup_cfg(self) -> dict:
        enabled = _state.get("autobackup.enabled", None)
        if enabled is None:
            enabled = bool(config.AUTO_BACKUP)
        sent = _state.get("autobackup.sent", [])
        if not isinstance(sent, list):
            sent = []
        return {"enabled": bool(enabled), "sent": sent}

    async def _build_backup(self):
        """
        Build the single-file backup zip. Same ritual as /export: built under
        the DB write lock so a page captured mid-transaction can never land in
        the archive (a silently corrupt backup only fails at /import time).
        """
        from db_manager import export_db_single
        async with self._db_lock:
            return await asyncio.to_thread(export_db_single, self.db)

    async def _deliver_backup(self, target: int, data: bytes, name: str,
                              caption: str):
        """Send one backup file to chat `target`; returns (message_id, mode)."""
        if len(data) <= self._HTTP_UPLOAD_SAFE:
            doc = io_module.BytesIO(data)
            doc.name = name
            m = await self._app.bot.send_document(
                chat_id=target, document=doc, caption=caption)
            return m.message_id, "api"
        client = await self._get_mtproto()
        if client is None:
            raise RuntimeError(
                "backup exceeds the Bot API limit and no MTProto client is "
                "available (set TELEGRAM_API_ID and TELEGRAM_API_HASH)")
        peer = await self._resolve_mtproto_peer(client, target)
        tmpdir = tempfile.mkdtemp(prefix="bkp_", dir=config.TEMP_DIR)
        try:
            tmp = Path(tmpdir) / name
            tmp.write_bytes(data)
            sent = await client.send_file(
                peer, str(tmp), caption=caption, force_document=True)
            return int(getattr(sent, "id", 0) or 0), "mtproto"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def _delete_backup_messages(self, entries):
        """Best-effort deletion of superseded backup messages."""
        for chat_id, mid, mode in entries:
            try:
                if mode == "api":
                    await self._app.bot.delete_message(
                        chat_id=chat_id, message_id=mid)
                else:
                    client = await self._get_mtproto()
                    if client is not None:
                        peer = await self._resolve_mtproto_peer(client, chat_id)
                        await client.delete_messages(peer, [mid])
            except Exception as e:
                log.debug("Auto-backup cleanup: %s", e)

    async def _run_backup(self, target: int) -> str:
        """Build, deliver, record and prune one automatic backup."""
        if self.db.count() == 0:
            return "❌ Empty DB — nothing to back up."
        data, name, stats = await self._build_backup()
        mb = len(data) / 1024 / 1024
        n_ch = len(await asyncio.to_thread(self.db.list_channels))
        caption = (f"🗄 Auto backup {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                   f"{n_ch} channels · "
                   f"{stats['vectors']:,} vectors · {mb:.1f} MB")
        mid, mode = await self._deliver_backup(target, data, name, caption)
        cfg = self._backup_cfg()
        cfg["sent"].append([target, mid, mode])
        keep = max(1, config.BACKUP_KEEP)
        doomed = cfg["sent"][:-keep]
        cfg["sent"] = cfg["sent"][-keep:]
        _state.put("autobackup.sent", cfg["sent"])
        await self._delete_backup_messages(doomed)
        kept = len(cfg["sent"])
        return (f"✅ Backup sent ({mb:.1f} MB, {'Bot API' if mode == 'api' else 'MTProto'}) —\n"
                f"{stats['vectors']:,} vectors, {n_ch} channels.\n"
                f"Keeping the newest {min(keep, kept)} backup(s).")

    async def _auto_backup_job(self, ctx: ContextTypes.DEFAULT_TYPE):
        cfg = self._backup_cfg()
        if not cfg["enabled"]:
            return
        if not config.BACKUP_CHAT_ID and not config.admins_configured:
            log.warning("Auto-backup skipped: set BACKUP_CHAT_ID or ADMIN_IDS "
                        "in .env so the backup has somewhere to go.")
            return
        target = config.BACKUP_CHAT_ID or config.ADMIN_IDS[0]
        try:
            text = await self._run_backup(target)
            log.info("Auto-backup: %s", text.replace("\n", " | "))
        except Exception as e:
            log.error("Auto-backup failed: %s", e, exc_info=True)

    async def cmd_autobackup(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        args = (update.message.text or "").split()[1:]
        arg0 = args[0].lower() if args else ""
        if arg0 == "on":
            _state.put("autobackup.enabled", True)
            return await update.message.reply_text(
                "✅ Auto-backup enabled.", parse_mode="Markdown")
        if arg0 == "off":
            _state.put("autobackup.enabled", False)
            return await update.message.reply_text(
                "⏹ Auto-backup disabled.", parse_mode="Markdown")
        if arg0 == "now":
            msg = await update.message.reply_text("📦 Preparing backup …")
            try:
                text = await self._run_backup(update.effective_chat.id)
                await _edit(msg, text, parse_mode="Markdown")
            except Exception as e:
                log.error("Autobackup now: %s", e, exc_info=True)
                await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")
            return
        cfg = self._backup_cfg()
        tag = "✅ enabled" if cfg["enabled"] else "⏹ disabled"
        target = config.BACKUP_CHAT_ID or (
            config.ADMIN_IDS[0] if config.ADMIN_IDS else 0)
        nxt = "—"
        try:
            jq = getattr(self._app, "job_queue", None) if self._app else None
            jobs = jq.get_jobs_by_name(self._AUTOBACKUP_JOB_NAME) if jq else []
            if jobs and getattr(jobs[0], "next_t", None):
                nxt = jobs[0].next_t.isoformat(sep=" ", timespec="minutes")
        except Exception:
            pass
        await update.message.reply_text(
            f"🗄 *Auto-backup*\n"
            f"Status: {tag}\n"
            f"Schedule: daily at `{max(0, min(23, config.BACKUP_HOUR)):02d}:00` "
            f"(server time)\n"
            f"Target: `{target or 'unset'}`\n"
            f"Kept copies: {config.BACKUP_KEEP}\n"
            f"Next run: `{nxt}`\n\n"
            f"`/autobackup now` — run immediately (back here)\n"
            f"`/autobackup on|off` — enable/disable",
            parse_mode="Markdown")

    # ── /import ───────────────────────────────────────────────────────────────

    async def import_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        cur = self.db.count()
        w = f"\n⚠️ Replaces *{cur:,}* current vectors.\n" if cur > 0 else "\n"
        # A previously abandoned /import may have left parts staged on disk.
        _discard_import_parts(self._import_parts.get(update.effective_user.id))
        self._import_parts[update.effective_user.id] = []
        await update.message.reply_text(
            f"📥 *Import Database*\n{w}"
            f"Send the `.zip` backup file.\n"
            f"For multi-part: send all parts in order, then type `done`.\n\n"
            f"/cancel to abort",
            parse_mode="Markdown")
        return IMPORT_WAITING

    async def _rebind_db(self) -> None:
        """
        Re-open the vector store after an /import replaced the files on disk.

        import_db moves the whole chroma_db directory away and extracts a new
        one under the store's directory lock, so old handles must not keep
        serving reads or writes through the swap window. Rebuild db/searcher,
        and restart the monitor only if it was running (never auto-start one
        that was deliberately switched off).
        """
        was_running = self._monitor is not None and self._monitor.running
        if self._monitor is not None:
            # Stop BEFORE rebinding: a live monitor holds the old db and would
            # keep writing into the directory the import just replaced.
            try:
                await self._monitor.stop()
            except Exception:
                pass
            self._monitor = None
        old_db = self.db
        # Keep serving if the reopen fails: the old handle may be stale, but a
        # bot that answers is better than a crashed handler. VectorDB() opens
        # SQLite and usearch and starts its writer threads, so it must not run
        # on the event loop.
        try:
            self.db = await asyncio.to_thread(VectorDB)
        except Exception as e:
            log.error("Reopen of vector store failed, keeping old handle: %s", e)
            return
        # Release the previous instance's background writer threads —
        # otherwise each /import leaks them. close(wait=True) joins them (up
        # to its timeout), so it must not run on the event loop.
        if old_db is not None and old_db is not self.db:
            try:
                await asyncio.to_thread(old_db.close, True, 10.0)
            except Exception:
                pass
        self.searcher = ImageSearcher(self.db, self.embedder)
        if was_running and self._app is not None:
            try:
                await self._start_monitor(self._app)
            except Exception as e:
                log.error("Monitor restart after import failed: %s", e)
        if self._app is not None:
            await self._update_pinned_message(self._app.bot)

    async def import_file(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if update.message.text:
            text = update.message.text.strip().lower()
            if text == "done":
                parts = self._import_parts.get(uid, [])
                if not parts:
                    await update.message.reply_text("❌ No parts received.", parse_mode="Markdown")
                    return IMPORT_WAITING
                msg = await update.message.reply_text(f"📥 Assembling {len(parts)} parts …")
                try:
                    from db_manager import import_db_parts
                    # The final hand-off must fit in memory (db_manager accepts
                    # the archive as bytes): refuse anything larger with a
                    # clear message instead of OOMing the process.
                    staged_bytes = sum(os.path.getsize(p) for p, _ in parts)
                    if staged_bytes > _IMPORT_MAX_HANDOFF_BYTES:
                        raise RuntimeError(
                            f"Backup is {staged_bytes / 1024 ** 3:.1f} GB — over "
                            f"the {_IMPORT_MAX_HANDOFF_BYTES // 1024 ** 3} GB "
                            "in-memory limit. Import it on a machine with more "
                            "RAM.")
                    items = []
                    for p, orig in parts:
                        with open(p, "rb") as fh:
                            items.append((fh.read(), orig))
                    # Hold the write gate for the WHOLE import: import_db moves
                    # chroma_db aside and rmtrees it, so a monitor write landing
                    # in that window goes to a deleted directory and is lost.
                    async with self._write_gate:
                        r = await asyncio.to_thread(import_db_parts, items)
                        await self._rebind_db()
                    extra = ""
                    adv = (r or {}).get("watermarks_advanced")
                    if isinstance(adv, int) and adv > 0:
                        extra = f"\n♻️ {adv} watermark(s) kept ahead of the backup"
                    if (r or {}).get("unknown_model"):
                        extra += ("\n\n⚠️ *This backup has no model fingerprint*\n"
                                  "It may have been indexed with a different CLIP "
                                  "model or an older preprocessing version.\n"
                                  "Run `/index` again for reliable search results.")
                    await _edit(msg,
                        f"✅ *Imported!*\nVectors: *{r['vectors']:,}*\n"
                        f"Channels: *{r['channels']}*{extra}", parse_mode="Markdown")
                except Exception as e:
                    # The old DB is restored on disk by import_db, but this
                    # process's handles may be stale — reopen regardless.
                    await self._rebind_db()
                    await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")
                finally:
                    # ALWAYS release the staged part files: if _rebind_db raised
                    # above, the early return left a multi-GB backup pinned on
                    # disk and the conversation stuck in IMPORT_WAITING forever.
                    _discard_import_parts(self._import_parts.pop(uid, None))
                return ConversationHandler.END
            else:
                await update.message.reply_text(
                    "Send a `.zip` file or type `done` after sending all parts.",
                    parse_mode="Markdown")
                return IMPORT_WAITING

        doc = update.message.document
        if not doc:
            if update.message.photo:
                # Absorb photos so the conversation keeps control instead of
                # falling through to the search handler mid-import.
                await update.message.reply_text(
                    "📎 Send the `.zip` backup (or its parts), not a photo.",
                    parse_mode="Markdown")
            else:
                await update.message.reply_text("📎 Send a file.", parse_mode="Markdown")
            return IMPORT_WAITING
        name = doc.file_name or ""
        sz = (doc.file_size or 0) / 1024 / 1024
        # A single .zip must fit in memory at hand-off (import_db takes bytes).
        if (doc.file_size or 0) > _IMPORT_MAX_HANDOFF_BYTES:
            await update.message.reply_text(
                f"❌ Backup is {sz:.1f} GB — over the "
                f"{_IMPORT_MAX_HANDOFF_BYTES // 1024 ** 3} GB in-memory limit.",
                parse_mode="Markdown")
            return IMPORT_WAITING
        msg = await update.message.reply_text(f"📥 Downloading `{name}` ({sz:.1f} MB) …",
                                               parse_mode="Markdown")
        try:
            data = await self._fetch_media_bytes(
                update, ctx, doc.file_id, size_bytes=doc.file_size or 0)
        except Exception as e:
            await _edit(msg, f"❌ Download failed: `{e}`", parse_mode="Markdown")
            return IMPORT_WAITING
        if name.endswith(".zip"):
            if len(data) > _IMPORT_MAX_HANDOFF_BYTES:
                # file_size can be missing/0 — enforce the cap on the real bytes.
                await _edit(msg,
                    f"❌ Backup is {len(data) / 1024 ** 3:.1f} GB — over the "
                    f"{_IMPORT_MAX_HANDOFF_BYTES // 1024 ** 3} GB in-memory limit.",
                    parse_mode="Markdown")
                return IMPORT_WAITING
            await _edit(msg, "📥 Importing …", parse_mode="Markdown")
            try:
                from db_manager import import_db
                # See the parts path: the whole import runs under the write
                # gate so no monitor/index write hits the moved directory.
                async with self._write_gate:
                    r = await asyncio.to_thread(import_db, data)
                    await self._rebind_db()
                extra = ""
                adv = (r or {}).get("watermarks_advanced")
                if isinstance(adv, int) and adv > 0:
                    extra = f"\n♻️ {adv} watermark(s) kept ahead of the backup"
                if (r or {}).get("unknown_model"):
                    extra += ("\n\n⚠️ *This backup has no model fingerprint*\n"
                              "It may have been indexed with a different CLIP "
                              "model or an older preprocessing version.\n"
                              "Run `/index` again for reliable search results.")
                await _edit(msg, f"✅ *Imported!*\nVectors: *{r['vectors']:,}*{extra}",
                            parse_mode="Markdown")
            except Exception as e:
                await self._rebind_db()
                await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")
            finally:
                _discard_import_parts(self._import_parts.pop(uid, None))
            return ConversationHandler.END
        # Multi-part backups: stage the bytes on DISK with the filename, so a
        # multi-GB upload never sits in RAM, and the part ordering/geometry
        # can be validated before anything is written.
        if _looks_like_backup_part(name):
            parts = self._import_parts.setdefault(uid, [])
            try:
                staged = _stage_import_part(data, parts)
            except RuntimeError as e:
                await _edit(msg, f"❌ {e}", parse_mode="Markdown")
                return IMPORT_WAITING
            parts.append((staged, name))
            staged_mb = sum(os.path.getsize(p) for p, _ in parts) / 1024 / 1024
            await _edit(msg,
                f"📦 Part received ({sz:.1f} MB — {staged_mb:.1f} MB staged)\n"
                f"Send next part or type `done` to import.",
                parse_mode="Markdown")
            return IMPORT_WAITING
        await _edit(msg, "❌ Not a `.zip` or a valid `partNNofMM` backup part.",
                    parse_mode="Markdown")
        return IMPORT_WAITING

    # ── /monitor ──────────────────────────────────────────────────────────────

    async def cmd_monitor(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        running = self._monitor and self._monitor.running
        buttons = []
        if running:
            st = self._monitor.stats
            buttons.append([InlineKeyboardButton("⏹ Stop", callback_data="mon_stop")])
            buttons.append([InlineKeyboardButton("📊 Status", callback_data="mon_status")])
            await update.message.reply_text(
                f"📡 *Monitor Running* \\(auto-on by default\\)\n\n"
                f"📸 New indexed: *{st['new_images']}*\n"
                f"🗑 Deleted removed: *{st['deleted_images']}*\n"
                f"❌ Errors: *{st['errors']}*\n\n"
                f"_Restarts automatically on bot reboot._",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(buttons))
        else:
            if self.db.count() == 0:
                return await update.message.reply_text("❌ Index channels first.", parse_mode="Markdown")
            if not self.embedder.is_ready():
                return await update.message.reply_text("❌ /model first.", parse_mode="Markdown")
            buttons.append([InlineKeyboardButton("▶️ Start Monitor", callback_data="mon_start")])
            ch = self.db.list_channels()
            await update.message.reply_text(
                f"📡 *Channel Monitor*\n\n"
                f"Watches *{len(ch)}* indexed channel(s) for:\n"
                f"  📸 New images → auto-embed\n"
                f"  🗑 Deleted images → remove vectors\n\n"
                f"_Normally starts automatically on bot boot._",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(buttons))

    async def monitor_cb(self, update: Update, _):
        q = update.callback_query
        await q.answer()
        if not _admin(q.from_user.id):
            return
        if q.data == "mon_start":
            # Stop the existing monitor first: replacing self._monitor without
            # stopping the old one leaves its Telethon client connected with
            # its NewMessage/Delete handlers still active — double indexing
            # and an orphaned MTProto session.
            if self._monitor is not None:
                try:
                    await self._monitor.stop()
                except Exception:
                    pass
                self._monitor = None
            await q.edit_message_text("📡 Starting monitor …", parse_mode="Markdown")
            from monitor import ChannelMonitor
            self._monitor = ChannelMonitor(self.db, self.embedder, db_lock=self._db_lock)
            bot = q.get_bot()

            async def notify(text):
                for aid in (config.ADMIN_IDS or []):
                    try:
                        await bot.send_message(aid, text, parse_mode="Markdown")
                    except Exception:
                        pass
                await self._update_pinned_message(bot)

            result = await self._monitor.start(notify_callback=notify)
            if not self._monitor.running:
                self._monitor = None
            # `result` embeds raw channel titles from monitor.start(); escape
            # them so a title containing _ or * can't fail the whole edit.
            await q.edit_message_text(f"📡 {_esc(result)}", parse_mode="Markdown")

        elif q.data == "mon_stop":
            if self._monitor:
                result = await self._monitor.stop()
                self._monitor = None
                await q.edit_message_text(result, parse_mode="Markdown")

        elif q.data == "mon_status":
            if self._monitor:
                st = self._monitor.stats
                await q.edit_message_text(
                    f"📡 *Monitor Status*\n\n"
                    f"📸 New: *{st['new_images']}*\n"
                    f"🗑 Deleted: *{st['deleted_images']}*\n"
                    f"❌ Errors: *{st['errors']}*\n"
                    f"📊 DB: *{self.db.count():,}*",
                    parse_mode="Markdown")

    # ── /channels ─────────────────────────────────────────────────────────────

    _CHANNELS_MAX_ROWS = 25

    async def _channels_view(self):
        """
        Build the /channels panel. Returns (text, keyboard) — the keyboard has
        a per-channel remove button (per row) plus Refresh. Empty DB returns
        (text, None) with a hint instead.
        """
        overview = await asyncio.to_thread(self.db.channel_overview)
        if not overview:
            return ("📭 *No channels indexed yet.*\n"
                    "Use /index to add one.", None)
        running = bool(self._monitor and self._monitor.running)
        lines = [f"📡 *Indexed channels* — *{len(overview)}*"
                 f" | Monitor: {'▶️ running' if running else '⏹ off'}\n"]
        buttons = []
        shown = 0
        for tid in sorted(overview)[:self._CHANNELS_MAX_ROWS]:
            e = overview[tid]
            wm = e.get("watermark")
            lines.append(
                f"\n• {_esc(e['name'])}\n"
                f"  🆔 `{tid}` | 📦 *{e['count']:,}* vectors"
                + (f" | 💧 watermark msg {wm:,}" if wm is not None else ""))
            buttons.append([InlineKeyboardButton(
                f"🗑 {e['name'][:28]}",
                callback_data=f"chn:rm:{tid}")])
            shown += 1
        if len(overview) > shown:
            lines.append(f"\n… +{len(overview) - shown} more channels "
                         f"(not listed, already in the buttons)")
            buttons.extend([
                [InlineKeyboardButton(f"🗑 Remove {t}", callback_data=f"chn:rm:{t}")]
                for t in sorted(overview)[shown:]])
        buttons.append([InlineKeyboardButton("♻️ Refresh", callback_data="chn:refresh")])
        lines.append("\n🗑 removes a channel: its vectors and monitor "
                     "watermark are deleted permanently.")
        return ("\n".join(lines), InlineKeyboardMarkup(buttons))

    async def cmd_channels(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        try:
            text, kb = await self._channels_view()
        except Exception as e:
            log.error("Channels view: %s", e, exc_info=True)
            return await update.message.reply_text(f"❌ `{e}`", parse_mode="Markdown")
        await update.message.reply_text(
            text, parse_mode="Markdown", reply_markup=kb,
            disable_web_page_preview=True)

    async def channel_cb(self, update: Update, _):
        q = update.callback_query
        await q.answer()
        if not _admin(q.from_user.id):
            return
        parts = (q.data or "").split(":")
        try:
            if parts[1] == "rm":
                tid = int(parts[2])
                overview = await asyncio.to_thread(self.db.channel_overview)
                e = overview.get(tid)
                if not e:
                    await q.edit_message_text(
                        "⚠️ Channel no longer present — refreshing panel")
                    return
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Yes, delete everything",
                                          callback_data=f"chn:rmOK:{tid}")],
                    [InlineKeyboardButton("❌ Cancel", callback_data=f"chn:cancel:{tid}")],
                ])
                wm = e.get("watermark")
                await q.edit_message_text(
                    f"⚠️ *Remove channel?*\n\n"
                    f"{_esc(e['name'])} (`{tid}`)\n"
                    f"pack *{e['count']:,}* vectors"
                    + (f" | 💧 watermark msg {wm:,}" if wm is not None else "") +
                    "\n\nThis permanently deletes all its vectors from the "
                    "index and stops the monitor watching it.",
                    parse_mode="Markdown", reply_markup=kb)

            elif parts[1] == "rmOK":
                tid = int(parts[2])
                try:
                    await q.edit_message_text("🗑 Removing …")
                    n = await asyncio.to_thread(self.db.forget_channel, tid)
                except Exception as exc:
                    log.error("Channel removal %s: %s", tid, exc, exc_info=exc)
                    await q.edit_message_text(f"❌ `{exc}`", parse_mode="Markdown")
                    return
                if self._monitor is not None:
                    self._monitor.unwatch(tid)
                bot = q.get_bot()
                await self._update_pinned_message(bot)
                text, kb = await self._channels_view()
                await q.edit_message_text(
                    f"✅ Removed `{tid}` ({n:,} vectors deleted).\n\n" + text,
                    parse_mode="Markdown", reply_markup=kb,
                    disable_web_page_preview=True)

            elif parts[1] == "cancel":
                text, kb = await self._channels_view()
                await q.edit_message_text(
                    text, parse_mode="Markdown", reply_markup=kb,
                    disable_web_page_preview=True)

            elif parts[1] == "refresh":
                await q.edit_message_text("♻️ Refreshing …")
                text, kb = await self._channels_view()
                await q.edit_message_text(
                    text, parse_mode="Markdown", reply_markup=kb,
                    disable_web_page_preview=True)
        except Exception as e:
            log.error("Channels callback: %s", e, exc_info=True)
            try:
                await q.edit_message_text(f"❌ `{e}`", parse_mode="Markdown")
            except Exception:
                pass

    # ── /setpinned ────────────────────────────────────────────────────────────

    async def cmd_setpinned(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """
        Create and pin a live status message in this chat.
        The bot auto-edits it whenever images are indexed or removed.
        """
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        text = _status_text(self.db, self._monitor, new_images=0, removed_images=0, errors=0)
        msg = await update.message.reply_text(text, parse_mode="Markdown")
        pinned_ok = True
        try:
            await ctx.bot.pin_chat_message(
                chat_id=update.effective_chat.id,
                message_id=msg.message_id,
                disable_notification=True,
            )
        except Exception as e:
            pinned_ok = False
            log.debug("Pin error: %s", e)
        chat_id_str = str(update.effective_chat.id)
        self._pinned[chat_id_str] = msg.message_id
        _state.put("pinned", self._pinned)
        if pinned_ok:
            confirm = ("✅ *Live status pinned!*\n"
                       "Updates automatically whenever images are indexed or removed.")
        else:
            # Don't claim success when the pin actually failed (usually a
            # missing "pin messages" admin right in this chat).
            confirm = ("⚠️ Status message created, but I could not pin it "
                       "(missing *Pin Messages* permission).\n"
                       "It still updates here in place.")
        await update.message.reply_text(confirm, parse_mode="Markdown")

    # ── /checkupdates ─────────────────────────────────────────────────────────

    async def cmd_checkupdates(self, update: Update, _):
        """
        Manually scan all indexed channels for images added since last run.
        Same logic as the automatic startup sync.
        """
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        from indexer import session_exists
        if not session_exists():
            return await update.message.reply_text("❌ /login first.", parse_mode="Markdown")
        if not self.embedder.is_ready():
            return await update.message.reply_text("❌ /model first.", parse_mode="Markdown")
        ch = self.db.list_channels()
        if not ch:
            return await update.message.reply_text("❌ No indexed channels.", parse_mode="Markdown")
        msg = await update.message.reply_text(
            f"🔄 Checking *{len(ch)}* channel(s) for new images …",
            parse_mode="Markdown",
        )
        self._track(self._do_checkupdates(msg, list(ch.keys())))

    async def _do_checkupdates(self, msg, channel_ids: list):
        total_new = 0
        lines = []
        for ch_id_str in channel_ids:
            try:
                from indexer import ChannelIndexer
                # Gated like /index and /import so a concurrent import cannot
                # move the DB directory mid-run. No per-batch _db_lock across
                # the loop — that would block monitor writes for hours.
                async with self._write_gate:
                    count = await ChannelIndexer(
                        self.db, self.embedder, db_lock=self._db_lock,
                    ).index_channel(int(ch_id_str), limit=None, progress_callback=None)
                if count > 0:
                    lines.append(f"  ✅ `{ch_id_str}` → +{count} new")
                    total_new += count
            except Exception as e:
                lines.append(f"  ❌ `{ch_id_str}`: {e}")

        detail = "\n".join(lines) if lines else "  _No new images found._"
        icon = "✅" if total_new > 0 else "💤"
        await _edit(
            msg,
            f"{icon} *Check complete*\n\n"
            f"📸 New images: *{total_new}*\n\n"
            f"{detail}\n\n"
            f"📊 Total DB: *{self.db.count():,}*",
            parse_mode="Markdown",
        )
        if total_new > 0 and self._app:
            await self._update_pinned_message(self._app.bot)

    # ── /cleanup ──────────────────────────────────────────────────────────────

    async def cmd_cleanup(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        ch = self.db.list_channels()
        if not ch:
            return await update.message.reply_text("❌ No channels.", parse_mode="Markdown")
        buttons = []
        for cid in ch:
            buttons.append([InlineKeyboardButton(f"🧹 Channel {cid}", callback_data=f"clean_{cid}")])
        buttons.append([InlineKeyboardButton("🧹 All Channels", callback_data="clean_all")])
        await update.message.reply_text(
            "🧹 *Cleanup Deleted Images*\n\n"
            "Checks if indexed messages still exist. "
            "Removes vectors for deleted ones.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons))

    async def cleanup_cb(self, update: Update, _):
        q = update.callback_query
        await q.answer()
        if not _admin(q.from_user.id):
            return
        if q.data == "clean_all":
            channels = list(self.db.list_channels().keys())
            await q.edit_message_text(f"🧹 Cleaning *{len(channels)}* channels …", parse_mode="Markdown")
            self._track(self._do_cleanup_all(q, channels))
        elif q.data.startswith("clean_"):
            cid = q.data.replace("clean_", "")
            await q.edit_message_text(f"🧹 Cleaning channel `{cid}` …", parse_mode="Markdown")
            self._track(self._do_cleanup(q, int(cid)))

    async def _do_cleanup(self, q, channel_id):
        try:
            from monitor import ChannelMonitor
            cm = ChannelMonitor(self.db, self.embedder, db_lock=self._db_lock)
            async def prog(checked, total, removed):
                if checked % 200 == 0:
                    await _edit(q.message, f"🧹 Checking {checked}/{total} … removed {removed}",
                                parse_mode="Markdown")
            # Gate the whole run so /import cannot swap the DB underneath it.
            async with self._write_gate:
                r = await cm.cleanup_channel(channel_id, progress_callback=prog)
            await _edit(q.message,
                f"✅ *Cleanup done*\n\n"
                f"📡 {_esc(r['channel'])}\n"
                f"✅ Checked: *{r['checked']}*\n"
                f"🗑 Removed: *{r['removed']}*\n"
                f"📊 Remaining: *{r['remaining']:,}*",
                parse_mode="Markdown")
            if r["removed"] > 0 and self._app:
                await self._update_pinned_message(self._app.bot)
        except Exception as e:
            await _edit(q.message, f"❌ `{e}`", parse_mode="Markdown")

    async def _do_cleanup_all(self, q, channel_ids):
        total_removed = 0
        total_checked = 0
        for i, cid in enumerate(channel_ids, 1):
            try:
                from monitor import ChannelMonitor
                cm = ChannelMonitor(self.db, self.embedder, db_lock=self._db_lock)
                await _edit(q.message, f"🧹 Channel {i}/{len(channel_ids)} (`{cid}`) …",
                            parse_mode="Markdown")
                async with self._write_gate:
                    r = await cm.cleanup_channel(int(cid))
                total_removed += r["removed"]
                total_checked += r["checked"]
            except Exception as e:
                log.error("Cleanup %s: %s", cid, e)
        try:
            await _edit(q.message,
                f"✅ *All channels cleaned*\n\n"
                f"✅ Checked: *{total_checked}*\n"
                f"🗑 Removed: *{total_removed}*\n"
                f"📊 DB: *{self.db.count():,}*",
                parse_mode="Markdown")
            if total_removed > 0 and self._app:
                await self._update_pinned_message(self._app.bot)
        except Exception as e:
            log.error("Cleanup summary failed: %s", e)

    # ── /duplicates ───────────────────────────────────────────────────────────

    async def cmd_duplicates(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        c = self.db.count()
        if c < 2:
            return await update.message.reply_text("❌ Need ≥2 images.", parse_mode="Markdown")
        est = max(1, c // 100)
        t = f"{est}s" if est < 60 else f"{est // 60}m"
        await update.message.reply_text(
            f"🔍 *Duplicates*\n📊 *{c:,}* images | 🎯 95% | ⏱ ~{t}",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔍 Start", callback_data="dup_start")],
                [InlineKeyboardButton("❌ Cancel", callback_data="dup_cancel")]]))

    async def dup_cb(self, update: Update, _):
        q = update.callback_query
        await q.answer()
        if not _admin(q.from_user.id):
            return
        if q.data == "dup_cancel":
            return await q.edit_message_text("Cancelled.")
        if q.data == "dup_noop":
            return
        if q.data == "dup_start":
            await q.edit_message_text(f"🔍 *Scanning…* 0/{self.db.count()}", parse_mode="Markdown")
            self._track(self._do_dups(q))
            return
        if q.data.startswith("dup_page_"):
            raw = q.data.replace("dup_page_", "")
            # A crafted/negative value used to raise ValueError into the generic
            # error handler.
            try:
                page = int(raw)
            except ValueError:
                return
            groups = self._dup_cache.get(q.from_user.id, [])
            if groups:
                await self._dup_page(q.message, groups, page)
            else:
                # The entry was evicted (cache is capped) or this is a stale
                # button from a previous run — say so instead of doing nothing.
                try:
                    await q.edit_message_text(
                        "⌛ Results expired. Run /duplicates again.")
                except Exception:
                    pass
            return

    async def _do_dups(self, q):
        try:
            try:
                from db_manager import DuplicateFinder
                pq: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def on_p(d, t, f):
                    # Called from the worker thread inside asyncio.to_thread.
                    # asyncio.Queue is NOT thread-safe: put_nowait from a foreign
                    # thread sets a Future's result off-loop, so the progress
                    # only refreshed when the 1s wait_for happened to re-arm.
                    try:
                        loop.call_soon_threadsafe(pq.put_nowait, (d, t, f))
                    except RuntimeError:
                        pass      # loop already closed during shutdown

                async def upd():
                    last = 0.0
                    while True:
                        try:
                            d, t, f = await asyncio.wait_for(pq.get(), timeout=1)
                            now = time.time()
                            if now - last >= 3:
                                last = now
                                await _edit(q.message, f"🔍 {d:,}/{t:,} | Groups: *{f}*",
                                            parse_mode="Markdown")
                        except asyncio.TimeoutError:
                            continue
                        except asyncio.CancelledError:
                            break
                ut = asyncio.create_task(upd())
                self._track(ut)   # strong reference — the loop only keeps weak ones
                try:
                    groups = await asyncio.to_thread(
                        DuplicateFinder(self.db).find_all, threshold=0.95, progress_callback=on_p)
                finally:
                    ut.cancel()
                    try:
                        await ut
                    except (asyncio.CancelledError, Exception):
                        pass
            except Exception as e:
                await _edit(q.message, f"❌ `{e}`", parse_mode="Markdown")
                return
            if not groups:
                return await _edit(q.message, "✅ *No duplicates!*", parse_mode="Markdown")
            # The cache held every duplicate id/link/name for the process
            # lifetime (unbounded). Keep only the last few users.
            while len(self._dup_cache) >= self._DUP_CACHE_MAX:
                self._dup_cache.pop(next(iter(self._dup_cache)))
            self._dup_cache[q.from_user.id] = groups
            td = sum(g["count"] for g in groups)
            await _edit(q.message, f"✅ *{len(groups)}* groups, *{td}* duplicates",
                        parse_mode="Markdown")
            await asyncio.sleep(0.5)
            await self._dup_page(q.message, groups, 0)
        except Exception as e:
            await _edit(q.message, f"❌ `{e}`", parse_mode="Markdown")

    async def _dup_page(self, message, groups, page):
        pp = 5
        tp = max(1, (len(groups) + pp - 1) // pp)
        # Clamp: a stale button from a previous run (or a crafted callback value)
        # used to render an empty page instead of a valid one.
        if page < 0 or page >= tp:
            page = 0
        pg = groups[page * pp:(page + 1) * pp]
        td = sum(g["count"] for g in groups)
        lines = [f"🔍 *Duplicates* {page + 1}/{tp}\n_{len(groups)} groups, {td} reposts total_\n"]
        for gi, g in enumerate(pg, start=page * pp + 1):
            src = g["source"]
            src_date = (src.get('date') or '')[:10] or 'unknown'
            lock = ' 🔒' if src.get('private') else ''
            lines.append(f"▸ *Group {gi}* — {g['count']} repost(s)")
            lines.append(
                f"  📌 [original]({src['link']}) `{src_date}`"
                f" — {_esc(src.get('channel_name'))}{lock}"
            )
            for m in g["matches"][:3]:
                m_date = (m.get('date') or '')[:10] or 'unknown'
                m_lock = ' 🔒' if m.get('private') else ''
                lines.append(
                    f"  🔁 {m['similarity']*100:.1f}% [repost]({m['link']}) `{m_date}`"
                    f" — {_esc(m.get('channel_name'))}{m_lock}"
                )
            if g["count"] > 3:
                lines.append(f"  _+{g['count'] - 3} more reposts_")
            lines.append("")
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️", callback_data=f"dup_page_{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{tp}", callback_data="dup_noop"))
        if page < tp - 1:
            nav.append(InlineKeyboardButton("➡️", callback_data=f"dup_page_{page + 1}"))
        try:
            await message.edit_text("\n".join(lines), parse_mode="Markdown",
                                    disable_web_page_preview=True,
                                    reply_markup=InlineKeyboardMarkup([nav]))
        except Exception: pass

    # ── /index ────────────────────────────────────────────────────────────────

    async def index_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        from indexer import session_exists, check_session
        if not session_exists():
            await update.message.reply_text("❌ /login first.", parse_mode="Markdown")
            return ConversationHandler.END
        if not self.embedder.is_ready():
            await update.message.reply_text("❌ /model first.", parse_mode="Markdown")
            return ConversationHandler.END
        msg = await update.message.reply_text("🔐 Checking …")
        v, d = await check_session()
        if not v:
            await msg.edit_text(f"❌ {_esc(d)}", parse_mode="Markdown")
            return ConversationHandler.END
        await msg.edit_text(
            f"✅ {d}\n\n📡 Send channel.\n`@user` `https://t.me/+x` `-100xxx`\n/cancel",
            parse_mode="Markdown")
        return INDEX_CHANNEL

    async def index_channel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.strip()
        ctx.user_data["idx_ch"] = text
        msg = await update.message.reply_text("📊 Analysing …")
        try:
            ch = int(text) if text.lstrip("-").isdigit() else text
            from indexer import ChannelIndexer, get_speed_info
            est = await ChannelIndexer(self.db, self.embedder).estimate_channel(ch)
            ctx.user_data["idx_est"] = est
            dim = self.embedder.get_dim() or 768
            sp = system_info.estimate_space(est["estimated_photos"], dim)
            si = get_speed_info()
            photos = est["estimated_photos"]
            em = photos / (25 if si["cryptg"] else 15) / 60 if photos > 0 else 0
            wm = f"📌 `{est['watermark']}`\n" if est["watermark"] > 0 else ""
            bd = "🔒" if est["is_private"] else "🌐"
            if est["restricted"]: bd += " 🚫"
            await msg.edit_text(
                f"📡 *{_esc(est['channel_name'])}* {bd}\n\n{wm}"
                f"📨 *{est['total_messages']:,}* | 🖼 *~{photos:,}* ({est['photo_ratio']*100:.0f}%)\n"
                f"💾 *{sp['total_mb']:.1f} MB* | 💿 *{sp['disk_free_mb']:.0f} MB* "
                f"{'✅' if sp['fits'] else '⚠️'} | ⏱ *~{em:.0f}m*\n\n"
                f"Number or `all` | /cancel", parse_mode="Markdown")
        except Exception as e:
            await msg.edit_text(f"❌ `{e}`\nRetry or /cancel", parse_mode="Markdown")
            return INDEX_CHANNEL
        return INDEX_LIMIT

    async def index_limit(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.strip().lower()
        ch = ctx.user_data.get("idx_ch", "")
        # "0".isdigit() is True, so limit=0 used to be accepted and the run
        # immediately indexed nothing while reporting "Done".
        limit = None if text == "all" else int(text) if text.isdigit() else None
        if text != "all" and (limit is None or limit <= 0):
            await update.message.reply_text("Number or `all`.", parse_mode="Markdown")
            return INDEX_LIMIT
        msg = await update.message.reply_text(f"⏳ Indexing `{ch}` …", parse_mode="Markdown")
        self._track(self._do_index(msg, ch, limit))
        return ConversationHandler.END

    async def _do_index(self, msg, channel, limit):
        try:
            ch = int(channel) if str(channel).lstrip("-").isdigit() else channel
            from indexer import ChannelIndexer, IndexStats
            last_t = [0.0]
            async def prog(s: IndexStats):
                now = time.time()
                if now - last_t[0] < 5: return
                last_t[0] = now
                m, sec = divmod(int(s.elapsed), 60)
                fl = f"\n⏸ {s.flood_waits}" if s.flood_waits else ""
                await _edit(msg,
                    f"⏳ ✅*{s.embedded}* ⬇️*{s.downloaded}* ({s.mb_downloaded:.1f}MB)\n"
                    f"⏭{s.skipped} ❌{s.errors} 🚀*{s.speed:.1f}/s* ⏱*{m}m{sec}s*{fl}",
                    parse_mode="Markdown")
            # Guard against a concurrent /import swapping the DB underneath us:
            # the indexer binds self.db once, so an import mid-run would have it
            # write into the moved-then-deleted directory while its watermark
            # landed in the new one.
            async with self._write_gate:
                count = await ChannelIndexer(
                    self.db, self.embedder, db_lock=self._db_lock,
                ).index_channel(ch, limit=limit, progress_callback=prog)
            # Re-read through self.db after the run so a rebind during the
            # run is still reflected in the summary — outside the gate, so
            # /import and /cleanup are unblocked as soon as the run ends.
            ss = await asyncio.to_thread(system_info.get_system_stats)
            await _edit(msg,
                f"✅ *Done!* 📸*{count}* 📊*{self.db.count():,}* "
                f"💾*{ss['chroma_mb']:.1f}MB* 🧠*{ss['ram_used_mb']:.0f}MB*",
                parse_mode="Markdown")
            if count > 0 and self._app:
                await self._update_pinned_message(self._app.bot)
            if self._monitor is not None and getattr(self._monitor, "running", False):
                try:
                    await self._monitor.watch(ch)
                except Exception as e:
                    log.debug("Monitor hot-add failed: %s", e)
        except Exception as e:
            await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")

    # ── photo / doc ───────────────────────────────────────────────────────────

    async def on_photo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self.embedder.is_ready():
            return await update.message.reply_text("❌ /model first")
        msg = await update.message.reply_text("🔍 …")
        photo = update.message.photo[-1]
        try:
            data = await self._fetch_media_bytes(
                update, ctx, photo.file_id, size_bytes=getattr(photo, "file_size", 0) or 0)
        except Exception as e:
            return await _edit(msg, f"⚠️ Download failed: `{e}`", parse_mode="Markdown")
        try:
            hits = await asyncio.to_thread(self.searcher.duplicates, data=data)
            if not hits:
                # No near-duplicate above THRESHOLD — still show the closest
                # matches so a first sighting of an image is never a dead end.
                hits = await asyncio.to_thread(self.searcher.search, data=data, n=8)
            await _edit(msg, _fmt(hits), parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as e:
            await _edit(msg, f"⚠️ `{e}`", parse_mode="Markdown")

    async def on_doc(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        # Some uploads carry no mime_type at all — fall back to the filename
        # extension so a photo sent as a file is still searchable.
        name = (doc.file_name or "").lower()
        is_image_doc = ((doc.mime_type or "").startswith("image/") or
                        name.endswith((".jpg", ".jpeg", ".png", ".webp",
                                       ".bmp", ".gif", ".tiff", ".tif",
                                       ".heic", ".heif")))
        if not is_image_doc:
            # A video sent as a FILE is indexed by its poster thumbnail, so
            # searching it must use the same frame instead of ignoring it.
            if (doc.mime_type or "").startswith("video/"):
                if not self.embedder.is_ready():
                    return await update.message.reply_text("❌ /model first")
                thumb = getattr(doc, "thumbnail", None)
                if thumb is None:
                    return await update.message.reply_text(
                        "⚠️ This video has no thumbnail to search.")
                msg = await update.message.reply_text("🔍 …")
                return await self._search_thumb(ctx, thumb, msg)
            return
        if not self.embedder.is_ready():
            return await update.message.reply_text("❌ /model first")
        msg = await update.message.reply_text("🔍 …")
        try:
            data = await self._fetch_media_bytes(
                update, ctx, doc.file_id, size_bytes=doc.file_size or 0)
        except Exception as e:
            return await _edit(msg, f"⚠️ Download failed: `{e}`", parse_mode="Markdown")
        try:
            hits = await asyncio.to_thread(self.searcher.duplicates, data=data)
            if not hits:
                # No near-duplicate above THRESHOLD — still show the closest
                # matches so a first sighting of an image is never a dead end.
                hits = await asyncio.to_thread(self.searcher.search, data=data, n=8)
            await _edit(msg, _fmt(hits), parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as e:
            await _edit(msg, f"⚠️ `{e}`", parse_mode="Markdown")

    async def _search_thumb(self, ctx: ContextTypes.DEFAULT_TYPE, thumb, msg):
        """
        Embed a video/GIF poster thumbnail and answer with matches.

        Channel indexing embeds the poster frame of videos (``media_kind`` ==
        "thumb"), so a video sent to the bot has to be searched through that
        same frame — otherwise the query vector and the indexed vectors come
        from different pixels and never match.
        """
        try:
            f = await ctx.bot.get_file(thumb.file_id)
            data = bytes(await f.download_as_bytearray())
        except Exception as e:
            return await _edit(msg, f"⚠️ Download failed: `{e}`", parse_mode="Markdown")
        try:
            hits = await asyncio.to_thread(self.searcher.duplicates, data=data)
            if not hits:
                hits = await asyncio.to_thread(self.searcher.search, data=data, n=8)
            await _edit(msg, _fmt(hits), parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as e:
            await _edit(msg, f"⚠️ `{e}`", parse_mode="Markdown")

    async def on_video(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        """Search a native video / GIF / round video note by its poster frame."""
        m = update.message
        media = m.video or m.animation or m.video_note
        if media is None:
            return
        if not self.embedder.is_ready():
            return await m.reply_text("❌ /model first")
        thumb = getattr(media, "thumbnail", None)
        if thumb is None:
            return await m.reply_text(
                "⚠️ This video has no thumbnail — nothing to match against.\n"
                "Only videos with a poster frame are searchable.")
        msg = await m.reply_text("🔍 …")
        await self._search_thumb(ctx, thumb, msg)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        # Fail fast with a CLEAR message instead of an InvalidToken traceback
        # from deep inside PTB (a missing BOT_TOKEN was an opaque crash).
        if not config.BOT_TOKEN:
            log.error("BOT_TOKEN is not set. Put the @BotFather token in "
                      ".env and restart.")
            raise SystemExit(1)
        if not config.admins_configured:
            # Fail closed: without ADMIN_IDS every admin command (including
            # /login, which rewrites the session file) is refused for EVERYONE.
            log.warning("ADMIN_IDS is empty — all admin commands are LOCKED. "
                        "Set at least one numeric Telegram user id in .env.")
        # Exactly ONE poller per token per directory. A second instance is the
        # cause of the endless '409 Conflict: terminated by other getUpdates
        # request' war, in which neither bot reliably receives updates and
        # every command (/index, /model, ...) appears broken.
        self._lock_fh = _acquire_single_instance_lock()
        # PTB's default HTTP timeouts are only 5 s (read/connect/write/pool).
        # On a slow or jittery residential line a perfectly healthy API call
        # (e.g. the reply to /system) can exceed that and surface as
        # telegram.error.TimedOut — the user just sees "⚠️ Something went
        # wrong". Raising the sockets absorbs home-network latency spikes
        # without masking real outages (the error handler still reports them).
        # All four builder methods exist since PTB 20.0 (pinned >=20.7).
        app = (
            Application.builder()
            .token(config.BOT_TOKEN)
            .read_timeout(30)      # API call responses (long replies, edits)
            .write_timeout(60)     # uploads: /export zips, import media
            .connect_timeout(30)
            .pool_timeout(30)
            .post_init(self._on_startup)   # auto-start monitor + sync on every boot
            .post_shutdown(self._on_shutdown)
            .build()
        )
        self._app = app

        app.add_error_handler(self._on_error)

        app.add_handler(CommandHandler("start",         self.cmd_start))
        app.add_handler(CommandHandler("stats",         self.cmd_stats))
        app.add_handler(CommandHandler("channels",     self.cmd_channels))
        app.add_handler(CommandHandler("status",        self.cmd_stats))
        app.add_handler(CommandHandler("system",        self.cmd_system))
        app.add_handler(CommandHandler("session",       self.cmd_session))
        app.add_handler(CommandHandler("export",        self.cmd_export))
        app.add_handler(CommandHandler("autobackup",    self.cmd_autobackup))
        app.add_handler(CommandHandler("duplicates",    self.cmd_duplicates))
        app.add_handler(CommandHandler("monitor",       self.cmd_monitor))
        app.add_handler(CommandHandler("cleanup",       self.cmd_cleanup))
        app.add_handler(CommandHandler("checkupdates",  self.cmd_checkupdates))  # NEW
        app.add_handler(CommandHandler("setpinned",     self.cmd_setpinned))     # NEW


        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("login", self.login_start)],
            states={
                LOGIN_PHONE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_phone)],
                LOGIN_CODE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_code)],
                LOGIN_2FA:   [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_2fa)],
                # A timed-out login must release the connected Telethon client
                # (and its sqlite session handle) — see on_conv_timeout.
                ConversationHandler.TIMEOUT: [
                    MessageHandler(filters.ALL, self.on_conv_timeout)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=300,
        ))
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("import", self.import_start)],
            states={
                IMPORT_WAITING: [
                    MessageHandler(filters.Document.ALL, self.import_file),
                    MessageHandler(filters.PHOTO, self.import_file),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.import_file),
                ],
                # A timed-out import must release the staged part files.
                ConversationHandler.TIMEOUT: [
                    MessageHandler(filters.ALL, self.on_conv_timeout)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=600,
        ))
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("index", self.index_start)],
            states={
                INDEX_CHANNEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.index_channel)],
                INDEX_LIMIT:   [MessageHandler(filters.TEXT & ~filters.COMMAND, self.index_limit)],
                ConversationHandler.TIMEOUT: [
                    MessageHandler(filters.ALL, self.on_conv_timeout)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=300,
        ))
        # /tune: guide → interactive 60 s settings window (plain-text reply),
        # with the direct-argument form /tune intra=2 batch=8 applying instantly.
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("tune", self.cmd_tune)],
            states={
                TUNE_WAIT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.tune_input)],
                ConversationHandler.TIMEOUT: [
                    MessageHandler(filters.ALL, self.tune_timeout)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=60,
        ))

        app.add_handler(CommandHandler("model", self.cmd_model))
        app.add_handler(CallbackQueryHandler(self.model_cb,   pattern=r"^dl_"))
        app.add_handler(CallbackQueryHandler(self.dup_cb,     pattern=r"^dup_"))
        app.add_handler(CallbackQueryHandler(self.monitor_cb, pattern=r"^mon_"))
        app.add_handler(CallbackQueryHandler(self.channel_cb, pattern=r"^chn_"))
        app.add_handler(CallbackQueryHandler(self.cleanup_cb, pattern=r"^clean_"))
        app.add_handler(MessageHandler(filters.PHOTO,        self.on_photo))
        app.add_handler(MessageHandler(filters.VIDEO | filters.ANIMATION |
                                       filters.VIDEO_NOTE, self.on_video))
        app.add_handler(MessageHandler(filters.Document.ALL, self.on_doc))
        app.add_handler(CommandHandler("cancel", self.cmd_cancel))

        # Nightly auto-backup (feature 10). Needs the job-queue extra.
        if getattr(app, "job_queue", None) is None:
            log.warning("Auto-backup unavailable: reinstall with "
                        "'pip install \"python-telegram-bot[job-queue]\"'.")
        else:
            try:
                import datetime as _dtmod
                hour = max(0, min(23, int(config.BACKUP_HOUR)))
                local_tz = datetime.now().astimezone().tzinfo
                app.job_queue.run_daily(
                    self._auto_backup_job,
                    time=_dtmod.time(hour=hour, tzinfo=local_tz),
                    name=self._AUTOBACKUP_JOB_NAME)
                log.info("Auto-backup scheduled daily at %02d:00 (server time).",
                         hour)
            except Exception as e:
                log.warning("Could not schedule auto-backup: %s", e)

        log.info("Bot polling …")
        # timeout=30 raises the long-poll window (default 10). PTB derives the
        # effective read timeout as read_timeout + timeout, so this also raises
        # the socket timeout automatically — ~5 round trips per minute instead of
        # ~6 per second, each idle connection nearly free.
        # allowed_updates keeps the JSON payload to what the handlers can actually
        # use, dropping chat_member/poll/join-request traffic entirely.
        app.run_polling(
            drop_pending_updates=True,
            timeout=30,
            # Deliberately WITHOUT "edited_message": PTB's MessageHandler also
            # matches edited messages (effective_message falls back to it), so
            # editing a caption re-ran the CLIP search and editing an /import
            # part re-downloaded and re-appended it, breaking the parts set.
            allowed_updates=["message", "callback_query"],
        )
