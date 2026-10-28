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
from telethon.sessions import SQLiteSession, StringSession

from config import config

log = logging.getLogger(__name__)

# Prefix Telegram prepends to channel/supergroup ids in Bot-API style "marked"
# ids (-100<id>). A real https://t.me/c/<id>/<msg> link must NOT contain it.
_MARKED_PREFIX = "-100"


# ── session discovery ───────────────────────────────────────────────────────

def _string_session_valid(s: str) -> bool:
    """True if ``s`` is a non-empty, well-formed Telethon StringSession.

    A malformed value (truncated, wrong format, bad base64 padding, Pyrogram
    string, ...) must never reach ``StringSession()``: Telethon raises
    ``binascii.Error`` / ``ValueError`` at construction time, which propagates
    up through every caller (monitor start, /index, startup sync) as an
    unhandled exception. Treating an invalid string as "no session" lets the
    bot fall back to the on-disk file session or prompt for /login instead of
    crashing the whole handler.
    """
    if not s:
        return False
    try:
        StringSession(s)
        return True
    except Exception:
        log.warning("TELETHON_STRING_SESSION is set but malformed; ignoring it.")
        return False


def has_string_session() -> bool:
    """True if a valid Telethon StringSession was supplied via the env."""
    return _string_session_valid(config.STRING_SESSION)


def user_session_available() -> bool:
    """
    True if *some* usable user session exists — either a StringSession in the
    environment or a ``.session`` file on disk. This is what the rest of the
    app should check instead of looking for the file directly.
    """
    return has_string_session() or os.path.exists(config.SESSION_FILE)


def warn_if_file_session_shared() -> None:
    """
    Warn when the on-disk session is used together with the monitor.

    The monitor holds a long-lived SQLiteSession on ``<name>.session``. Every
    ephemeral client (indexer / estimate / cleanup) now snapshots that file into
    an in-memory StringSession instead of opening a second writer on it, so the
    remaining contention is only with a concurrent /login. A StringSession in
    the environment removes the file entirely and is still the recommended
    deployment for a monitor + indexer combination.
    """
    if not has_string_session() and os.path.exists(config.SESSION_FILE):
        log.warning(
            "Using the on-disk session %s: the monitor holds the only writer, "
            "but a concurrent /login can still contend for it. Set "
            "TELETHON_STRING_SESSION to remove the file entirely.",
            config.SESSION_FILE)


def session_location() -> str:
    """Human-readable description of where the user session is coming from."""
    if has_string_session():
        return "TELETHON_STRING_SESSION (env)"
    return config.SESSION_FILE


def channel_url_id(channel_id) -> str:
    """
    Convert a (possibly Bot-API "marked") channel id into the integer that
    belongs in a ``https://t.me/c/<id>/<msg>`` link.

    Telegram marks channel/supergroup peer ids as ``-100<real_id>``; the public
    message-link form drops that prefix entirely. Passing the marked id yields
    a dead link (see core.telegram.org/api/links).
    """
    s = str(int(channel_id))
    if s.startswith(_MARKED_PREFIX):
        return s[len(_MARKED_PREFIX):]
    return s.lstrip("-")


def msg_link(entity, message_id: int) -> str:
    """
    Public deep link to a message in a channel. Public channels use their
    username, private ones the ``/c/<unmarked-id>/<msg>`` form.
    """
    username = getattr(entity, "username", None)
    if username:
        return f"https://t.me/{username}/{message_id}"
    return f"https://t.me/c/{channel_url_id(entity.id)}/{message_id}"


# ── client builders ─────────────────────────────────────────────────────────

def _ephemeral_session():
    """
    In-memory snapshot of the on-disk session, or ``None``.

    Ephemeral clients (indexer, estimate, cleanup) used to open their own
    SQLiteSession on the SAME ``.session`` file the long-lived monitor holds;
    two writers on one rollback-journal sqlite file raise "database is locked"
    mid-run. Copying the auth key into a StringSession keeps the same account
    without a shared file. Returns ``None`` when the file cannot be read, so
    the caller falls back to the old behaviour instead of failing.
    """
    if not os.path.exists(config.SESSION_FILE):
        return None
    base = None
    try:
        base = SQLiteSession(config.SESSION_FILE)
        if not base.auth_key:
            return None
        session = StringSession()
        session.set_dc(base.dc_id, base.server_address, base.port)
        session.auth_key = base.auth_key
        return session
    except Exception as exc:
        log.warning("Could not snapshot %s (%s); sharing the file session.",
                    config.SESSION_FILE, exc)
        return None
    finally:
        if base is not None:
            try:
                base.close()
            except Exception:
                pass


async def prime_entity_cache(client) -> int:
    """
    Fill the client's entity cache from the dialog list.

    A bare channel id (``-100…`` or the unmarked int stored in the watermark
    state) can only be resolved when the entity — with its access_hash — is
    already known. A StringSession persists no entities, so in a fresh process
    ``get_entity(id)`` falls through to ``channels.GetChannelsRequest`` with
    ``access_hash=0``, which Telegram rejects for user accounts. One
    ``iter_dialogs()`` pass (a handful of requests) makes every chat the
    account is in resolvable by id again. Returns the number of dialogs seen.
    """
    seen = 0
    try:
        async for _ in client.iter_dialogs(limit=None):
            seen += 1
    except Exception as exc:
        log.warning("Could not prime the entity cache: %s", exc)
    if seen:
        log.debug("Primed entity cache with %d dialog(s)", seen)
    return seen


def make_user_client(receive_updates: bool = True, catch_up: bool = False,
                     ephemeral: bool = False) -> TelegramClient:
    """
    Build (but do not connect) a Telethon client for the *user* account.

    Uses the StringSession from the environment when present, otherwise the
    on-disk session file. Either way it carries the same API_ID / API_HASH.

    `receive_updates` defaults to True because the long-running ChannelMonitor
    registers event handlers on its client. Ephemeral callers that only download
    (the indexer's per-run client) should pass False to save bandwidth.

    `catch_up` makes Telethon replay the updates missed while the client was
    disconnected; only the long-lived monitor wants that.

    `ephemeral` makes a file-session client work on an in-memory copy of the
    session so it never competes with the monitor for the sqlite file.
    """
    if has_string_session():
        session: object = StringSession(config.STRING_SESSION)
    elif ephemeral:
        session = _ephemeral_session() or config.SESSION_PATH
    else:
        session = config.SESSION_PATH
    return TelegramClient(
        session, config.API_ID, config.API_HASH,
        # A 12-hour index run should not abort on the 6th connection blip.
        connection_retries=10,
        retry_delay=2,
        # Let Telethon absorb short flood-waits in its own request retry loop
        # rather than routing every one through the Python handler.
        flood_sleep_threshold=120,
        # Indexing thousands of entities in one process.
        entity_cache_limit=20_000,
        receive_updates=receive_updates,
        catch_up=catch_up and receive_updates,
    )


def make_bot_client() -> TelegramClient:
    """
    Build (but do not connect / start) a Telethon client that will log in as
    the *bot* over MTProto, used purely for large-file downloads.

    ``receive_updates=False`` is critical: it stops this client from consuming
    bot updates, which would otherwise conflict with python-telegram-bot's
    long-polling ("terminated by other getUpdates request").
    """
    if config.BOT_STRING_SESSION and _string_session_valid(config.BOT_STRING_SESSION):
        session: object = StringSession(config.BOT_STRING_SESSION)
    else:
        if config.BOT_STRING_SESSION:
            log.warning("BOT_STRING_SESSION is set but malformed; using file session.")
        session = config.BOT_SESSION_PATH
    return TelegramClient(
        session, config.API_ID, config.API_HASH, receive_updates=False,
    )
