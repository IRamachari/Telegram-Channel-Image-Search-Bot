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


def _admin(uid):
    return not config.ADMIN_IDS or uid in config.ADMIN_IDS


def _fmt(hits):
    if not hits:
        return "❌ No similar images found."
    lines = ["✅ *Similar images found:*\n"]
    for i, h in enumerate(hits[:10], 1):
        pct = h["similarity"] * 100
        icon = "🟢" if pct >= 95 else "🟡" if pct >= 85 else "🟠"
        lock = " 🔒" if h.get("private") else ""
        lines.append(
            f"{icon} *#{i}* — {pct:.1f}%{lock}\n"
            f"    📺 {h['channel_name']}\n"
            f"    🔗 [message]({h['link']})\n"
            f"    📅 {h['date'][:10]}\n"
        )
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
    except (TimedOut, Exception):
        pass


class SearchBot:
    def __init__(self):
        log.info("Initialising bot …")
        self.db = VectorDB()
        self.embedder = CLIPEmbedder()
        self.searcher = ImageSearcher(self.db, self.embedder)
        self._tasks: set[asyncio.Task] = set()
        self._dup_cache: dict[int, list] = {}
        try:
            from auto_scale import get_applied
            t = get_applied()
            log.info("Auto: batch=%d parallel=%d", t["batch_size"], t["parallel_downloads"])
        except Exception:
            pass

    def _track(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ── /start ──
    async def cmd_start(self, update: Update, _):
        from indexer import session_exists
        s = "✅" if session_exists() else "❌ → /login"
        m = "✅" if self.embedder.is_ready() else "❌ → /model"
        a = models.get_active_model() or "none"
        n = self.db.count()
        await update.message.reply_text(
            "🔍 *Image Search Bot*\n\n"
            f"*Status:*\n  Session: {s}\n  Model: {m} (`{a}`)\n"
            f"  Database: *{n:,}* vectors\n\n"
            "*Setup:*\n1\\. /login\n2\\. /model\n3\\. /index\n\n"
            "*Use:* Send photo → search\n\n"
            "*Tools:*\n"
            "/duplicates — find duplicates\n"
            "/export — download DB\n"
            "/import — upload DB\n"
            "/autotune — optimize\n"
            "/system — server stats\n"
            "/stats — DB stats\n"
            "/session — debug",
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
            f"🔐 *Session*\nPath: `{si['path']}`\n"
            f"Exists: {'✅' if si['exists'] else '❌'}\n"
            f"Auth: {'✅' if v else '❌'}\n{d}",
            parse_mode="Markdown",
        )

    # ── /stats ──
    async def cmd_stats(self, update: Update, _):
        n = self.db.count()
        ch = self.db.list_channels()
        lines = [f"📊 *{n:,}* vectors\n"]
        for c, w in ch.items():
            lines.append(f"  • `{c}` → msg `{w}`")
        if not ch:
            lines.append("  No channels yet.")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ── /system ──
    async def cmd_system(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        text = await asyncio.to_thread(system_info.format_system_stats)
        from indexer import get_speed_info
        sp = get_speed_info()
        text += (
            f"\n⚡ *Speed*\n  cryptg: {'✅' if sp['cryptg'] else '❌'}\n"
            f"  parallel: {sp['parallel_files']}\n  batch: {sp['batch_size']}\n"
            f"\n🔄 Tasks: {len(self._tasks)}\n"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    # ── /autotune ──
    async def cmd_autotune(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        from auto_scale import format_auto_tune, apply_tune
        await asyncio.to_thread(apply_tune, self.embedder.model_size_mb)
        text = await asyncio.to_thread(format_auto_tune, self.embedder.model_size_mb)
        text += "\n✅ _Applied automatically\\._"
        await update.message.reply_text(text, parse_mode="Markdown")

    # ── /cancel ──
    async def cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        cl = ctx.user_data.pop("tg_client", None)
        if cl:
            try:
                await cl.disconnect()
            except Exception:
                pass
        await update.message.reply_text("Cancelled.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    # ── /login ──
    async def login_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        if config.API_ID == 0 or not config.API_HASH:
            await update.message.reply_text("❌ Set API\\_ID and API\\_HASH\\.", parse_mode="Markdown")
            return ConversationHandler.END
        await update.message.reply_text(
            "🔐 *Login*\n\nSend phone with country code\\.\n`+1234567890`\n\n/cancel to abort",
            parse_mode="Markdown",
        )
        return LOGIN_PHONE

    async def login_phone(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        phone = update.message.text.strip()
        try:
            await update.message.delete()
        except Exception:
            pass
        msg = await update.message.reply_text("📱 Sending code …")
        from telethon import TelegramClient
        client = TelegramClient(config.SESSION_PATH, config.API_ID, config.API_HASH)
        try:
            await client.connect()
            sent = await client.send_code_request(phone)
        except Exception as exc:
            try:
                await client.disconnect()
            except Exception:
                pass
            await msg.edit_text(f"❌ `{exc}`", parse_mode="Markdown")
            return ConversationHandler.END
        ctx.user_data["tg_client"] = client
        ctx.user_data["tg_phone"] = phone
        ctx.user_data["tg_hash"] = sent.phone_code_hash
        await msg.edit_text("✅ Code sent\\! Enter it\\.\n/cancel to abort", parse_mode="Markdown")
        return LOGIN_CODE

    async def login_code(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        code = update.message.text.strip().replace(" ", "").replace("-", "")
        try:
            await update.message.delete()
        except Exception:
            pass
        client = ctx.user_data.get("tg_client")
        if not client:
            await update.message.reply_text("❌ Lost\\. /login again\\.", parse_mode="Markdown")
            return ConversationHandler.END
        from telethon.errors import SessionPasswordNeededError
        try:
            await client.sign_in(phone=ctx.user_data["tg_phone"], code=code,
                                 phone_code_hash=ctx.user_data["tg_hash"])
        except SessionPasswordNeededError:
            await update.message.reply_text("🔒 2FA password\\?\n/cancel", parse_mode="Markdown")
            return LOGIN_2FA
        except Exception as exc:
            await client.disconnect()
            ctx.user_data.pop("tg_client", None)
            await update.message.reply_text(f"❌ `{exc}`", parse_mode="Markdown")
            return ConversationHandler.END
        me = await client.get_me()
        await client.disconnect()
        ctx.user_data.pop("tg_client", None)
        await update.message.reply_text(
            f"✅ *{me.first_name}* \\(ID: `{me.id}`\\)\nUse /index", parse_mode="Markdown",
        )
        return ConversationHandler.END

    async def login_2fa(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        pw = update.message.text.strip()
        try:
            await update.message.delete()
        except Exception:
            pass
        client = ctx.user_data.get("tg_client")
        if not client:
            await update.message.reply_text("❌ Lost\\.", parse_mode="Markdown")
            return ConversationHandler.END
        try:
            await client.sign_in(password=pw)
        except Exception as exc:
            await client.disconnect()
            ctx.user_data.pop("tg_client", None)
            await update.message.reply_text(f"❌ `{exc}`", parse_mode="Markdown")
            return ConversationHandler.END
        me = await client.get_me()
        await client.disconnect()
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
                f"{info['name']} ({info['size_mb']}MB) {f}", callback_data=f"dl_{k}",
            )])
        await update.message.reply_text(
            "\n".join(lines), parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

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
        await q.edit_message_text(f"⬇️ *{info['name']}* \\({info['size_mb']} MB\\) …", parse_mode="Markdown")
        try:
            r = await asyncio.to_thread(models.download_model, key)
            self.embedder.reload()
            from auto_scale import apply_tune
            apply_tune(self.embedder.model_size_mb)
            w = "\n\n⚠️ Re\\-index for best results\\." if old and old != key and self.db.count() > 0 else ""
            await q.edit_message_text(f"{r}\nDim: `{self.embedder.get_dim()}`{w}", parse_mode="Markdown")
        except Exception as exc:
            await q.edit_message_text(f"❌ `{exc}`", parse_mode="Markdown")

    # ── /export ──
    async def cmd_export(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        if self.db.count() == 0:
            return await update.message.reply_text("❌ Empty DB.")
        msg = await update.message.reply_text("📦 Preparing …")
        try:
            from db_manager import get_db_size_mb, export_db
            size = await asyncio.to_thread(get_db_size_mb)
            if size > 45:
                return await _edit(msg,
                    f"❌ Too large ({size:.0f} MB)\\.\n`tar czf backup.tar.gz chroma_db/`",
                    parse_mode="Markdown")
            await _edit(msg, f"📦 Compressing {size:.1f} MB …", parse_mode="Markdown")
            zb, st = await asyncio.to_thread(export_db)
            doc = io_module.BytesIO(zb)
            doc.name = f"backup_{int(time.time())}.zip"
            await update.message.reply_document(
                document=doc,
                caption=f"🗄 Vectors: {self.db.count():,} | {st['zip_size_mb']} MB",
            )
            await _edit(msg, "✅ Sent\\!", parse_mode="Markdown")
        except Exception as exc:
            await _edit(msg, f"❌ `{exc}`", parse_mode="Markdown")

    # ── /import ──
    async def import_start(self, update: Update, _):
        if not _admin(update.effective_user.id):
            await update.message.reply_text("⛔")
            return ConversationHandler.END
        cur = self.db.count()
        w = f"\n⚠️ Replaces current *{cur:,}* vectors\\.\n" if cur > 0 else "\n"
        await update.message.reply_text(
            f"📥 *Import*\n\nSend `.zip` backup\\.{w}/cancel to abort",
            parse_mode="Markdown",
        )
        return IMPORT_WAITING

    async def import_file(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not update.message.document:
            await update.message.reply_text("📎 Send a `.zip` file\\.", parse_mode="Markdown")
            return IMPORT_WAITING
        doc = update.message.document
        if not (doc.file_name or "").endswith(".zip"):
            await update.message.reply_text("❌ Must be `.zip`\\.", parse_mode="Markdown")
            return IMPORT_WAITING
        msg = await update.message.reply_text("📥 Downloading …")
        try:
            tf = await ctx.bot.get_file(doc.file_id)
            data = bytes(await tf.download_as_bytearray())
            await _edit(msg, "📥 Importing …", parse_mode="Markdown")
            from db_manager import import_db
            r = await asyncio.to_thread(import_db, data)
            self.db = VectorDB()
            self.searcher = ImageSearcher(self.db, self.embedder)
            await _edit(msg, f"✅ *Imported\\!*\nVectors: *{r['vectors']:,}*", parse_mode="Markdown")
        except Exception as exc:
            self.db = VectorDB()
            self.searcher = ImageSearcher(self.db, self.embedder)
            await _edit(msg, f"❌ `{exc}`", parse_mode="Markdown")
        return ConversationHandler.END

    # ── /duplicates ──
    async def cmd_duplicates(self, update: Update, _):
        if not _admin(update.effective_user.id):
            return await update.message.reply_text("⛔")
        c = self.db.count()
        if c < 2:
            return await update.message.reply_text("❌ Need ≥2 images\\.", parse_mode="Markdown")
        est = max(1, c // 100)
        t = f"{est}s" if est < 60 else f"{est // 60}m {est % 60}s"
        await update.message.reply_text(
            f"🔍 *Find Duplicates*\n\n📊 *{c:,}* images\n🎯 Threshold: *95%*\n⏱ ~{t}",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔍 Start", callback_data="dup_start")],
                [InlineKeyboardButton("❌ Cancel", callback_data="dup_cancel")],
            ]),
        )

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
            await q.edit_message_text(
                f"🔍 *Scanning…*\nChecked: *0* / *{self.db.count()}*",
                parse_mode="Markdown",
            )
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
            finder = DuplicateFinder(self.db)
            pq: asyncio.Queue = asyncio.Queue()

            def on_prog(done, total, found):
                try:
                    pq.put_nowait((done, total, found))
                except Exception:
                    pass

            async def updater():
                last = 0.0
                while True:
                    try:
                        d, t, f = await asyncio.wait_for(pq.get(), timeout=1)
                        now = time.time()
                        if now - last >= 3:
                            last = now
                            pct = d * 100 // t if t else 0
                            await _edit(q.message,
                                f"🔍 *Scanning…*\n{d:,}/{t:,} ({pct}%)\nGroups: *{f}*",
                                parse_mode="Markdown")
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            ut = asyncio.create_task(updater())
            groups = await asyncio.to_thread(finder.find_all, threshold=0.95, progress_callback=on_prog)
            ut.cancel()

            if not groups:
                return await _edit(q.message, "✅ *No duplicates\\!*\nNothing ≥95%\\.", parse_mode="Markdown")

            self._dup_cache[q.from_user.id] = groups
            td = sum(g["count"] for g in groups)
            await _edit(q.message,
                f"✅ *Done\\!*\n🔍 *{len(groups)}* groups\n📸 *{td}* duplicates",
                parse_mode="Markdown")
            await asyncio.sleep(0.5)
            await self._dup_page(q.message, groups, 0)
        except Exception as exc:
            await _edit(q.message, f"❌ `{exc}`", parse_mode="Markdown")

    async def _dup_page(self, message, groups, page):
        pp = 5
        tp = (len(groups) + pp - 1) // pp
        start = page * pp
        pg = groups[start:start + pp]
        td = sum(g["count"] for g in groups)
        lines = [f"🔍 *Duplicates* — {page + 1}/{tp}\n_{len(groups)} groups, {td} total_\n"]
        for gi, g in enumerate(pg, start=start + 1):
            src = g["source"]
            lk = " 🔒" if src.get("private") else ""
            lines.append(f"▸ *Group {gi}* ({g['count']} dupes)")
            lines.append(f"  📌 [{src['channel_name']}]({src['link']}){lk}")
            for m in g["matches"][:3]:
                p = m["similarity"] * 100
                ml = " 🔒" if m.get("private") else ""
                lines.append(f"  🟢 {p:.1f}% [{m['channel_name']}]({m['link']}){ml}")
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
        except Exception:
            pass

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
        msg = await update.message.reply_text("🔐 Checking session …")
        v, d = await check_session()
        if not v:
            await msg.edit_text(f"❌ {d}\n/login again\\.", parse_mode="Markdown")
            return ConversationHandler.END
        await msg.edit_text(
            f"✅ {d}\n\n📡 *Send channel*\n\n"
            "`@username`\n`https://t.me/+hash`\n`-1001234567890`\n\n/cancel",
            parse_mode="Markdown",
        )
        return INDEX_CHANNEL

    async def index_channel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.strip()
        ctx.user_data["idx_ch"] = text
        msg = await update.message.reply_text("📊 Analysing …")
        try:
            ch = int(text) if text.lstrip("-").isdigit() else text
            from indexer import ChannelIndexer, get_speed_info
            ix = ChannelIndexer(self.db, self.embedder)
            est = await ix.estimate_channel(ch)
            ctx.user_data["idx_est"] = est
            dim = self.embedder.get_dim() or 768
            sp = system_info.estimate_space(est["estimated_photos"], dim)
            si = get_speed_info()
            badges = "🔒 Private" if est["is_private"] else "🌐 Public"
            if est["restricted"]:
                badges += " 🚫"
            photos = est["estimated_photos"]
            base = 25 if si["cryptg"] else 15
            em = photos / base / 60 if photos > 0 else 0
            wm = f"📌 Last: `{est['watermark']}`\n" if est["watermark"] > 0 else ""
            await msg.edit_text(
                f"📡 *{est['channel_name']}*\n{badges}\n\n{wm}"
                f"📨 *{est['total_messages']:,}* msgs\n"
                f"🖼 *~{photos:,}* photos ({est['photo_ratio']*100:.0f}%)\n"
                f"📏 Avg: *{est['avg_photo_kb']:.0f} KB*\n\n"
                f"💾 DB: *{sp['total_mb']:.1f} MB*\n"
                f"💿 Free: *{sp['disk_free_mb']:.0f} MB* {'✅' if sp['fits'] else '⚠️'}\n"
                f"⏱ ~*{em:.0f} min*\n\n"
                f"📝 Number or `all`\n/cancel",
                parse_mode="Markdown",
            )
        except Exception as exc:
            await msg.edit_text(f"❌ `{exc}`\nRetry or /cancel", parse_mode="Markdown")
            return INDEX_CHANNEL
        return INDEX_LIMIT

    async def index_limit(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.strip().lower()
        ch = ctx.user_data.get("idx_ch", "")
        if text == "all":
            limit = None
        else:
            try:
                limit = int(text)
            except ValueError:
                await update.message.reply_text("Number or `all`\\.", parse_mode="Markdown")
                return INDEX_LIMIT
        msg = await update.message.reply_text(f"⏳ Indexing `{ch}` …", parse_mode="Markdown")
        self._track(self._do_index(msg, ch, limit))
        return ConversationHandler.END

    async def _do_index(self, msg, channel, limit):
        try:
            ch = int(channel) if str(channel).lstrip("-").isdigit() else channel
            from indexer import ChannelIndexer, IndexStats
            ix = ChannelIndexer(self.db, self.embedder)
            last_t = [0.0]

            async def prog(s: IndexStats):
                now = time.time()
                if now - last_t[0] < 5:
                    return
                last_t[0] = now
                m, sec = divmod(int(s.elapsed), 60)
                fl = f"\n⏸ Floods: {s.flood_waits}" if s.flood_waits else ""
                await _edit(msg,
                    f"⏳ *Indexing…*\n\n✅ *{s.embedded}*\n"
                    f"⬇️ *{s.downloaded}* ({s.mb_downloaded:.1f} MB)\n"
                    f"⏭ {s.skipped} ❌ {s.errors}\n"
                    f"🚀 *{s.speed:.1f}/s*\n⏱ *{m}m {sec}s*{fl}",
                    parse_mode="Markdown")

            count = await ix.index_channel(ch, limit=limit, progress_callback=prog)
            ss = await asyncio.to_thread(system_info.get_system_stats)
            await _edit(msg,
                f"✅ *Done\\!*\n\n📸 *{count}*\n📊 *{self.db.count():,}*\n"
                f"💾 *{ss['chroma_mb']:.1f} MB*\n"
                f"🧠 *{ss['ram_used_mb']:.0f}/{ss['ram_total_mb']:.0f} MB*",
                parse_mode="Markdown")
        except Exception as exc:
            await _edit(msg, f"❌ `{exc}`", parse_mode="Markdown")

    # ── photo search ──
    async def on_photo(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self.embedder.is_ready():
            return await update.message.reply_text("❌ /model first")
        msg = await update.message.reply_text("🔍 …")
        tf = await ctx.bot.get_file(update.message.photo[-1].file_id)
        data = bytes(await tf.download_as_bytearray())
        try:
            hits = await asyncio.to_thread(self.searcher.duplicates, data=data)
            await _edit(msg, _fmt(hits), parse_mode="Markdown", disable_web_page_preview=True)
        except Exception as exc:
            await _edit(msg, f"⚠️ `{exc}`", parse_mode="Markdown")

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
        except Exception as exc:
            await _edit(msg, f"⚠️ `{exc}`", parse_mode="Markdown")

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

        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("login", self.login_start)],
            states={
                LOGIN_PHONE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_phone)],
                LOGIN_CODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_code)],
                LOGIN_2FA: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.login_2fa)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=300,
        ))
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("import", self.import_start)],
            states={
                IMPORT_WAITING: [MessageHandler(filters.Document.ALL, self.import_file)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=300,
        ))
        app.add_handler(ConversationHandler(
            entry_points=[CommandHandler("index", self.index_start)],
            states={
                INDEX_CHANNEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.index_channel)],
                INDEX_LIMIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.index_limit)],
            },
            fallbacks=[CommandHandler("cancel", self.cmd_cancel)],
            conversation_timeout=300,
        ))

        app.add_handler(CommandHandler("model", self.cmd_model))
        app.add_handler(CallbackQueryHandler(self.model_cb, pattern=r"^dl_"))
        app.add_handler(CallbackQueryHandler(self.dup_cb, pattern=r"^dup_"))
        app.add_handler(MessageHandler(filters.PHOTO, self.on_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.on_doc))
        app.add_handler(CommandHandler("cancel", self.cmd_cancel))

        log.info("Bot polling …")
        app.run_polling(drop_pending_updates=True)
