"""
Central Telegram (Telethon / MTProto) client factory.

Everything that needs to talk to Telegram over MTProto goes through here so
there is a single, consistent place that decides *how* a session is loaded:

  1. A Telethon ``StringSession`` taken from the ``TELETHON_STRING_SESSION``
     environment variable (recommended for cloud / Docker — no files needed).
  2. Otherwise a classic on-disk ``<name>.session`` file (created by /login,
     or restored from ``TELETHON_SESSION_B64`` in entrypoint.sh).

There are two kinds of client:

  * **User client**  — logged in as *your* Telegram account. Used to read and
    index channels you are a member of (public or private).

  * **Bot client**   — logged in as the *bot* (via BOT_TOKEN) over MTProto.
    Used ONLY to download files that users send to the bot. The Bot HTTP API
    caps downloads at 20 MB ("File is too big"); MTProto does not, so large
    photos / documents / backup zips can be pulled with no size limit.

IMPORTANT — session-string compatibility:
    The string MUST be a **Telethon** StringSession. A **Pyrogram** session
    string is a completely different format and will NOT work here. Generate a
    compatible one with:  ``python generate_session.py``
"""

from __future__ import annotations

import logging
import os

from telethon import TelegramClient
from telethon.sessions import StringSession

from config import config

log = logging.getLogger(__name__)


# ── session discovery ───────────────────────────────────────────────────────

def has_string_session() -> bool:
    """True if a Telethon StringSession was supplied via the environment."""
    return bool(config.STRING_SESSION)


def user_session_available() -> bool:
    """
    True if *some* usable user session exists — either a StringSession in the
    environment or a ``.session`` file on disk. This is what the rest of the
    app should check instead of looking for the file directly.
    """
    return has_string_session() or os.path.exists(config.SESSION_FILE)


def session_location() -> str:
    """Human-readable description of where the user session is coming from."""
    if has_string_session():
        return "TELETHON_STRING_SESSION (env)"
    return config.SESSION_FILE


# ── client builders ─────────────────────────────────────────────────────────

def make_user_client() -> TelegramClient:
    """
    Build (but do not connect) a Telethon client for the *user* account.

    Uses the StringSession from the environment when present, otherwise the
    on-disk session file. Either way it carries the same API_ID / API_HASH.
    """
    if has_string_session():
        session: object = StringSession(config.STRING_SESSION)
    else:
        session = config.SESSION_PATH
    return TelegramClient(session, config.API_ID, config.API_HASH)


def make_bot_client() -> TelegramClient:
    """
    Build (but do not connect / start) a Telethon client that will log in as
    the *bot* over MTProto, used purely for large-file downloads.

    ``receive_updates=False`` is critical: it stops this client from consuming
    bot updates, which would otherwise conflict with python-telegram-bot's
    long-polling ("terminated by other getUpdates request").
    """
    if config.BOT_STRING_SESSION:
        session: object = StringSession(config.BOT_STRING_SESSION)
    else:
        session = config.BOT_SESSION_PATH
    return TelegramClient(
        session, config.API_ID, config.API_HASH, receive_updates=False,
    )
