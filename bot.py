from __future__ import annotations
import asyncio
import io as io_module
import logging
import time

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, CallbackQueryHandler,
    ContextTypes, filters,
)
from telegram.error import RetryAfter, TimedOut

from config import config
from database import VectorDB
from embeddings import CLIPEmbedder
from searcher import ImageSearcher
import models
import system_info

log = logging.getLogger(__name__)

LOGIN_PHONE, LOGIN_CODE, LOGIN_2FA = range(3)
INDEX_CHANNEL, INDEX_LIMIT = range(10, 12)
IMPORT_WAITING = 20
IMPORT_PARTS = 21
CLEANUP_CHANNEL = 30


def _admin(uid):
    return not config.ADMIN_IDS or uid in config.ADMIN_IDS


def _fmt(hits):
    if not hits:
        return "❌ No similar images found."
    lines = ["✅ *Similar images found:*\n"]
    for i, h in enumerate(hits[:10], 1):
        p = h["similarity"] * 100
        ic = "🟢" if p >= 95 else "🟡" if p >= 85 else "🟠"
        lk = " 🔒" if h.get("private") else ""
        lines.append(f"{ic} *#{i}* — {p:.1f}%{lk}\n    📺 {h['channel_name']}\n"
                     f"    🔗 [message]({h['link']})\n    📅 {h['date'][:10]}\n")
    if any(h.get("private") for h in hits[:10]):
        lines.append("\n🔒 = private channel")
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


class SearchBot:
    def __init__(self):
        log.info("Initialising …")
        self.db = VectorDB()
        self.embedder = CLIPEmbedder()
        self.searcher = ImageSearcher(self.db, self.embedder)
        self._tasks: set[asyncio.Task] = set()
        self._dup_cache: dict[int, list] = {}
        self._monitor = None
        self._import_parts: dict[int, list[bytes]] = {}

    def _track(self, coro):
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    # ── /start ──
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
            f"*Search:* Send photo\n\n"
            f"*Tools:*\n"
            f"/monitor — auto\\-detect new/deleted\n"
            f"/cleanup — remove deleted vectors\n"
            f"/duplicates — find duplicates\n"
            f"/export — download DB\n"
            f"/import — upload DB\n"
            f"/autotune — optimize\n"
            f"/system — server stats\n"
            f"/stats — DB stats",
            parse_mode="Markdown",
        )

    # ── /session ──
    async def cmd_session(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        from indexer import session_info, check_session
        si = session_info()
        v, d = await check_session()
        await update.message.reply_text(
            f"🔐 *Session*\n`{si['path']}`\n"
            f"{'✅' if si['exists'] else '❌'} | Auth: {'✅' if v else '❌'}\n{d}",
            parse_mode="Markdown",
        )

    # ── /stats ──
    async def cmd_stats(self, update: Update, _):
        n = self.db.count()
        ch = self.db.list_channels()
        lines = [f"📊 *{n:,}* vectors\n"]
        for c, w in ch.items():
            lines.append(f"  • `{c}` → `{w}`")
        if not ch:
            lines.append("  No channels.")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ── /system ──
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

    # ── /autotune ──
    async def cmd_autotune(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        from auto_scale import format_auto_tune, apply_tune
        await asyncio.to_thread(apply_tune, self.embedder.model_size_mb)
        text = await asyncio.to_thread(format_auto_tune, self.embedder.model_size_mb)
        await update.message.reply_text(text + "\n✅ _Applied\\._", parse_mode="Markdown")

    # ── /cancel ──
    async def cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        cl = ctx.user_data.pop("tg_client", None)
        if cl:
            try:
                await cl.disconnect()
            except Exception:
                pass
        self._import_parts.pop(update.effective_user.id, None)
        await update.message.reply_text("Cancelled.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    # ── /login ──
    async def login_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        if config.API_ID == 0:
            await update.message.reply_text("❌ Set API\\_ID\\.", parse_mode="Markdown")
            return ConversationHandler.END
        await update.message.reply_text(
            "🔐 Phone with country code\\.\n`+1234567890`\n/cancel",
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
        await msg.edit_text("✅ Enter code\\.\n/cancel", parse_mode="Markdown")
        return LOGIN_CODE

    async def login_code(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        code = update.message.text.strip().replace(" ", "").replace("-", "")
        try: await update.message.delete()
        except Exception: pass
        cl = ctx.user_data.get("tg_client")
        if not cl:
            await update.message.reply_text("❌ /login again\\.", parse_mode="Markdown")
            return ConversationHandler.END
        from telethon.errors import SessionPasswordNeededError
        try:
            await cl.sign_in(phone=ctx.user_data["tg_phone"], code=code,
                             phone_code_hash=ctx.user_data["tg_hash"])
        except SessionPasswordNeededError:
            await update.message.reply_text("🔒 2FA password\\?", parse_mode="Markdown")
            return LOGIN_2FA
        except Exception as e:
            await cl.disconnect()
            ctx.user_data.pop("tg_client", None)
            await update.message.reply_text(f"❌ `{e}`", parse_mode="Markdown")
            return ConversationHandler.END
        me = await cl.get_me()
        await cl.disconnect()
        ctx.user_data.pop("tg_client", None)
        await update.message.reply_text(f"✅ *{me.first_name}*", parse_mode="Markdown")
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
        me = await cl.get_me()
        await cl.disconnect()
        ctx.user_data.pop("tg_client", None)
        await update.message.reply_text(f"✅ *{me.first_name}*", parse_mode="Markdown")
        return ConversationHandler.END

    # ── /model ──
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
        await q.edit_message_text(f"⬇️ *{info['name']}* …", parse_mode="Markdown")
        try:
            r = await asyncio.to_thread(models.download_model, key)
            self.embedder.reload()
            from auto_scale import apply_tune
            apply_tune(self.embedder.model_size_mb)
            w = "\n⚠️ Re\\-index recommended\\." if old and old != key and self.db.count() > 0 else ""
            await q.edit_message_text(f"{r}\nDim: `{self.embedder.get_dim()}`{w}", parse_mode="Markdown")
        except Exception as e:
            await q.edit_message_text(f"❌ `{e}`", parse_mode="Markdown")

    # ── /export ──
    async def cmd_export(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        if self.db.count() == 0:
            return await update.message.reply_text("❌ Empty DB.")
        msg = await update.message.reply_text("📦 Preparing …")
        try:
            from db_manager import export_db
            parts = await asyncio.to_thread(export_db, self.db)
            stats = parts[0][2]
            total_parts = len(parts)

            if total_parts == 1:
                await _edit(msg, f"📦 Uploading ({stats['zip_mb']} MB) …", parse_mode="Markdown")
                doc = io_module.BytesIO(parts[0][0])
                doc.name = parts[0][1]
                await update.message.reply_document(
                    document=doc,
                    caption=(f"🗄 Backup: {stats['channels']}\n"
                             f"Vectors: {stats['vectors']:,} | {stats['zip_mb']} MB"),
                )
                await _edit(msg, "✅ Export sent\\!", parse_mode="Markdown")
            else:
                await _edit(msg,
                    f"📦 Database is *{stats['zip_mb']} MB*\n"
                    f"Splitting into *{total_parts}* parts …",
                    parse_mode="Markdown")
                for i, (data, name, _) in enumerate(parts, 1):
                    doc = io_module.BytesIO(data)
                    doc.name = name
                    part_mb = len(data) / 1024 / 1024
                    await update.message.reply_document(
                        document=doc,
                        caption=f"📦 Part {i}/{total_parts} ({part_mb:.1f} MB)",
                    )
                    await asyncio.sleep(1)
                await _edit(msg,
                    f"✅ *{total_parts}* parts sent\\!\n\n"
                    f"To import: /import → send all parts in order\\.",
                    parse_mode="Markdown")
        except Exception as e:
            log.error("Export: %s", e, exc_info=True)
            await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")

    # ── /import ──
    async def import_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        cur = self.db.count()
        w = f"\n⚠️ Replaces *{cur:,}* current vectors\\.\n" if cur > 0 else "\n"
        self._import_parts[update.effective_user.id] = []
        await update.message.reply_text(
            f"📥 *Import Database*\n{w}"
            f"Send the `.zip` backup file\\.\n"
            f"For multi\\-part: send all parts in order, then type `done`\\.\n\n"
            f"/cancel to abort",
            parse_mode="Markdown")
        return IMPORT_WAITING

    async def import_file(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id

        # handle text message "done" for multi-part
        if update.message.text:
            text = update.message.text.strip().lower()
            if text == "done":
                parts = self._import_parts.get(uid, [])
                if not parts:
                    await update.message.reply_text("❌ No parts received\\.", parse_mode="Markdown")
                    return IMPORT_WAITING
                msg = await update.message.reply_text(
                    f"📥 Assembling {len(parts)} parts …")
                try:
                    from db_manager import import_db_parts
                    r = await asyncio.to_thread(import_db_parts, parts)
                    self.db = VectorDB()
                    self.searcher = ImageSearcher(self.db, self.embedder)
                    self._import_parts.pop(uid, None)
                    await _edit(msg,
                        f"✅ *Imported\\!*\nVectors: *{r['vectors']:,}*\n"
                        f"Channels: *{r['channels']}*", parse_mode="Markdown")
                except Exception as e:
                    self.db = VectorDB()
                    self.searcher = ImageSearcher(self.db, self.embedder)
                    await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")
                return ConversationHandler.END
            else:
                await update.message.reply_text(
                    "Send a `.zip` file or type `done` after sending all parts\\.",
                    parse_mode="Markdown")
                return IMPORT_WAITING

        doc = update.message.document
        if not doc:
            await update.message.reply_text("📎 Send a file\\.", parse_mode="Markdown")
            return IMPORT_WAITING

        name = doc.file_name or ""
        sz = (doc.file_size or 0) / 1024 / 1024

        msg = await update.message.reply_text(f"📥 Downloading `{name}` ({sz:.1f} MB) …",
                                               parse_mode="Markdown")
        tf = await ctx.bot.get_file(doc.file_id)
        data = bytes(await tf.download_as_bytearray())

        # single zip file
        if name.endswith(".zip"):
            await _edit(msg, "📥 Importing …", parse_mode="Markdown")
            try:
                from db_manager import import_db
                r = await asyncio.to_thread(import_db, data)
                self.db = VectorDB()
                self.searcher = ImageSearcher(self.db, self.embedder)
                self._import_parts.pop(uid, None)
                await _edit(msg, f"✅ *Imported\\!*\nVectors: *{r['vectors']:,}*",
                            parse_mode="Markdown")
            except Exception as e:
                self.db = VectorDB()
                self.searcher = ImageSearcher(self.db, self.embedder)
                await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")
            return ConversationHandler.END

        # multi-part file
        if "part" in name.lower():
            parts = self._import_parts.setdefault(uid, [])
            parts.append(data)
            await _edit(msg,
                f"📦 Part {len(parts)} received ({sz:.1f} MB)\\.\n"
                f"Send next part or type `done` to import\\.",
                parse_mode="Markdown")
            return IMPORT_WAITING

        await _edit(msg, "❌ Send `.zip` or part files\\.", parse_mode="Markdown")
        return IMPORT_WAITING

    # ── /monitor ──
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
                f"📡 *Monitor Running*\n\n"
                f"📸 New indexed: *{st['new_images']}*\n"
                f"🗑 Deleted removed: *{st['deleted_images']}*\n"
                f"❌ Errors: *{st['errors']}*",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(buttons))
        else:
            if self.db.count() == 0:
                return await update.message.reply_text("❌ Index channels first\\.", parse_mode="Markdown")
            if not self.embedder.is_ready():
                return await update.message.reply_text("❌ /model first\\.", parse_mode="Markdown")
            buttons.append([InlineKeyboardButton("▶️ Start Monitor", callback_data="mon_start")])
            ch = self.db.list_channels()
            await update.message.reply_text(
                f"📡 *Channel Monitor*\n\n"
                f"Watches *{len(ch)}* indexed channels for:\n"
                f"  📸 New images → auto\\-embed\n"
                f"  🗑 Deleted images → remove vectors\n",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(buttons))

    async def monitor_cb(self, update: Update, _):
        q = update.callback_query
        await q.answer()
        if not _admin(q.from_user.id):
            return

        if q.data == "mon_start":
            await q.edit_message_text("📡 Starting monitor …", parse_mode="Markdown")
            from monitor import ChannelMonitor
            self._monitor = ChannelMonitor(self.db, self.embedder)

            chat_id = q.message.chat_id
            bot = q.get_bot()

            async def notify(text):
                try:
                    await bot.send_message(chat_id, text, parse_mode="Markdown")
                except Exception:
                    pass

            result = await self._monitor.start(notify_callback=notify)
            await q.edit_message_text(f"📡 {result}", parse_mode="Markdown")

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

    # ── /cleanup ──
    async def cmd_cleanup(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        ch = self.db.list_channels()
        if not ch:
            return await update.message.reply_text("❌ No channels\\.", parse_mode="Markdown")

        buttons = []
        for cid in ch:
            buttons.append([InlineKeyboardButton(
                f"🧹 Channel {cid}", callback_data=f"clean_{cid}")])
        buttons.append([InlineKeyboardButton("🧹 All Channels", callback_data="clean_all")])

        await update.message.reply_text(
            "🧹 *Cleanup Deleted Images*\n\n"
            "Checks if indexed messages still exist\\.\n"
            "Removes vectors for deleted ones\\.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons))

    async def cleanup_cb(self, update: Update, _):
        q = update.callback_query
        await q.answer()
        if not _admin(q.from_user.id):
            return

        if q.data == "clean_all":
            channels = list(self.db.list_channels().keys())
            await q.edit_message_text(
                f"🧹 Cleaning *{len(channels)}* channels …",
                parse_mode="Markdown")
            self._track(self._do_cleanup_all(q, channels))
        elif q.data.startswith("clean_"):
            cid = q.data.replace("clean_", "")
            await q.edit_message_text(f"🧹 Cleaning channel `{cid}` …", parse_mode="Markdown")
            self._track(self._do_cleanup(q, int(cid)))

    async def _do_cleanup(self, q, channel_id):
        try:
            from monitor import ChannelMonitor
            cm = ChannelMonitor(self.db, self.embedder)

            async def prog(checked, total, removed):
                if checked % 200 == 0:
                    await _edit(q.message,
                        f"🧹 Checking {checked}/{total} … removed {removed}",
                        parse_mode="Markdown")

            r = await cm.cleanup_channel(channel_id, progress_callback=prog)
            await _edit(q.message,
                f"✅ *Cleanup done*\n\n"
                f"📡 {r['channel']}\n"
                f"✅ Checked: *{r['checked']}*\n"
                f"🗑 Removed: *{r['removed']}*\n"
                f"📊 Remaining: *{r['remaining']:,}*",
                parse_mode="Markdown")
        except Exception as e:
            await _edit(q.message, f"❌ `{e}`", parse_mode="Markdown")

    async def _do_cleanup_all(self, q, channel_ids):
        total_removed = 0
        total_checked = 0
        for i, cid in enumerate(channel_ids, 1):
            try:
                from monitor import ChannelMonitor
                cm = ChannelMonitor(self.db, self.embedder)
                await _edit(q.message,
                    f"🧹 Channel {i}/{len(channel_ids)} (`{cid}`) …",
                    parse_mode="Markdown")
                r = await cm.cleanup_channel(int(cid))
                total_removed += r["removed"]
                total_checked += r["checked"]
            except Exception as e:
                log.error("Cleanup %s: %s", cid, e)
        await _edit(q.message,
            f"✅ *All channels cleaned*\n\n"
            f"✅ Checked: *{total_checked}*\n"
            f"🗑 Removed: *{total_removed}*\n"
            f"📊 DB: *{self.db.count():,}*",
            parse_mode="Markdown")

    # ── /duplicates ──
    async def cmd_duplicates(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        c = self.db.count()
        if c < 2:
            return await update.message.reply_text("❌ Need ≥2 images\\.", parse_mode="Markdown")
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
            page = int(q.data.replace("dup_page_", ""))
            groups = self._dup_cache.get(q.from_user.id, [])
            if groups:
                await self._dup_page(q.message, groups, page)

    async def _do_dups(self, q):
        try:
            from db_manager import DuplicateFinder
            pq: asyncio.Queue = asyncio.Queue()
            def on_p(d, t, f):
                try: pq.put_nowait((d, t, f))
                except Exception: pass
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
                    except asyncio.TimeoutError: continue
                    except asyncio.CancelledError: break
            ut = asyncio.create_task(upd())
            groups = await asyncio.to_thread(
                DuplicateFinder(self.db).find_all, threshold=0.95, progress_callback=on_p)
            ut.cancel()
            if not groups:
                return await _edit(q.message, "✅ *No duplicates\\!*", parse_mode="Markdown")
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
        tp = (len(groups) + pp - 1) // pp
        pg = groups[page * pp:(page + 1) * pp]
        td = sum(g["count"] for g in groups)
        lines = [f"🔍 *Duplicates* {page + 1}/{tp}\n_{len(groups)} groups, {td} total_\n"]
        for gi, g in enumerate(pg, start=page * pp + 1):
            src = g["source"]
            lines.append(f"▸ *Group {gi}* ({g['count']})")
            lines.append(f"  📌 [{src['channel_name']}]({src['link']})"
                         f"{' 🔒' if src.get('private') else ''}")
            for m in g["matches"][:3]:
                lines.append(f"  🟢 {m['similarity']*100:.1f}% "
                             f"[{m['channel_name']}]({m['link']})"
                             f"{' 🔒' if m.get('private') else ''}")
            if g["count"] > 3:
                lines.append(f"  _\\+{g['count'] - 3} more_")
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

    # ── /index ──
    async def index_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        from indexer import session_exists, check_session
        if not session_exists():
            await update.message.reply_text("❌ /login first\\.", parse_mode="Markdown")
            return ConversationHandler.END
        if not self.embedder.is_ready():
            await update.message.reply_text("❌ /model first\\.", parse_mode="Markdown")
            return ConversationHandler.END
        msg = await update.message.reply_text("🔐 Checking …")
        v, d = await check_session()
        if not v:
            await msg.edit_text(f"❌ {d}", parse_mode="Markdown")
            return ConversationHandler.END
        await msg.edit_text(
            f"✅ {d}\n\n📡 Send channel\\.\n`@user` `https://t.me/+x` `-100xxx`\n/cancel",
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
                f"📡 *{est['channel_name']}* {bd}\n\n{wm}"
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
        limit = None if text == "all" else int(text) if text.isdigit() else None
        if text != "all" and limit is None:
            await update.message.reply_text("Number or `all`\\.", parse_mode="Markdown")
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
            count = await ChannelIndexer(self.db, self.embedder).index_channel(
                ch, limit=limit, progress_callback=prog)
            ss = await asyncio.to_thread(system_info.get_system_stats)
            await _edit(msg,
                f"✅ *Done\\!* 📸*{count}* 📊*{self.db.count():,}* "
                f"💾*{ss['chroma_mb']:.1f}MB* 🧠*{ss['ram_used_mb']:.0f}MB*",
                parse_mode="Markdown")
        except Exception as e:
            await _edit(msg, f"❌ `{e}`", parse_mode="Markdown")

    # ── photo ──
    async def on_photo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self.embedder.is_ready():
            return await update.message.reply_text("❌ /model first")
        msg = await update.message.reply_text("🔍 …")
        tf = await ctx.bot.get_file(update.message.photo[-1].file_id)
        data = bytes(await tf.download_as_bytearray())
        try:
            hits = await asyncio.to_thread(self.searcher.duplicates, data=data)
            await _edit(msg, _fmt(hits), parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as e:
            await _edit(msg, f"⚠️ `{e}`", parse_mode="Markdown")

    async def on_doc(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        doc = update.message.document
        if not (doc.mime_type or "").startswith("image/"):
            return
        if not self.embedder.is_ready():
            return await update.message.reply_text("❌ /model first")
        msg = await update.message.reply_text("🔍 …")
        tf = await ctx.bot.get_file(doc.file_id)
        data = bytes(await tf.download_as_bytearray())
        try:
            hits = await asyncio.to_thread(self.searcher.duplicates, data=data)
            await _edit(msg, _fmt(hits), parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as e:
            await _edit(msg, f"⚠️ `{e}`", parse_mode="Markdown")

    # ── run ──
    def run(self):
        app = Application.builder().token(config.BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("system", self.cmd_system))
        app.add_handler(CommandHandler("session", self.cmd_session))
        app.add_handler(CommandHandler("autotune", self.cmd_autotune))
        app.add_handler(CommandHandler("export", self.cmd_export))
        app.add_handler(CommandHandler("duplicates", self.cmd_duplicates))
        app.add_handler(CommandHandler("monitor", self.cmd_monitor))
        app.add_handler(CommandHandler("cleanup", self.cmd_cleanup))

        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("login", self.login_start)],
            states={
                LOGIN_PHONE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_phone)],
                LOGIN_CODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_code)],
                LOGIN_2FA: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_2fa)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=300))
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("import", self.import_start)],
            states={
                IMPORT_WAITING: [
                    MessageHandler(filters.Document.ALL, self.import_file),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.import_file),
                ],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=600))
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("index", self.index_start)],
            states={
                INDEX_CHANNEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.index_channel)],
                INDEX_LIMIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.index_limit)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=300))

        app.add_handler(CommandHandler("model", self.cmd_model))
        app.add_handler(CallbackQueryHandler(self.model_cb, pattern=r"^dl_"))
        app.add_handler(CallbackQueryHandler(self.dup_cb, pattern=r"^dup_"))
        app.add_handler(CallbackQueryHandler(self.monitor_cb, pattern=r"^mon_"))
        app.add_handler(CallbackQueryHandler(self.cleanup_cb, pattern=r"^clean_"))
        app.add_handler(MessageHandler(filters.PHOTO, self.on_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.on_doc))
        app.add_handler(CommandHandler("cancel", self.cmd_cancel))

        log.info("Bot polling …")
        app.run_polling(drop_pending_updates=True)
