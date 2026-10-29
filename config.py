from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.resolve()
load_dotenv(PROJECT_DIR / ".env")


def _int_list(raw: str) -> list[int]:
    out = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except ValueError:
            log = __import__("logging").getLogger(__name__)
            log.warning("Ignoring non-integer ADMIN_IDS entry: %r", x)
    return out


@dataclass(frozen=True)
class Config:
    API_ID: int          = int(os.getenv("TELEGRAM_API_ID", "0"))
    API_HASH: str        = os.getenv("TELEGRAM_API_HASH", "")
    SESSION_NAME: str    = os.getenv("TELEGRAM_SESSION", "indexer_session")
    BOT_TOKEN: str       = os.getenv("BOT_TOKEN", "")
    # Telethon StringSession for the *user* account (optional). When set, the
    # bot authenticates from this string instead of a .session file — ideal for
    # Docker / cloud hosts. MUST be a Telethon string (NOT Pyrogram). Generate
    # one with `python generate_session.py`.
    STRING_SESSION: str  = os.getenv("TELETHON_STRING_SESSION", "").strip()
    # Optional StringSession for the *bot* MTProto download client. Usually
    # left empty — the bot session is created automatically from BOT_TOKEN.
    BOT_STRING_SESSION: str = os.getenv("BOT_TELETHON_STRING_SESSION", "").strip()
    CHROMA_DIR: str      = str(PROJECT_DIR / os.getenv("CHROMA_DIR", "chroma_db"))
    COLLECTION: str      = os.getenv("COLLECTION_NAME", "telegram_images")
    TEMP_DIR: str        = str(PROJECT_DIR / os.getenv("TEMP_DIR", "temp_images"))
    BATCH_SIZE: int      = int(os.getenv("BATCH_SIZE", "32"))
    THRESHOLD: float     = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    MAX_RESULTS: int     = int(os.getenv("MAX_RESULTS", "10"))
    PARALLEL_DL: int     = int(os.getenv("PARALLEL_DOWNLOADS", "10"))
    # Nightly auto-backup (/autobackup). Sends the single-file DB backup to
    # BACKUP_CHAT_ID (or the first ADMIN_IDS entry) once a day and keeps only
    # the newest BACKUP_KEEP copies in that chat. BACKUP_HOUR is read in the
    # SERVER's local timezone.
    AUTO_BACKUP: bool    = os.getenv(
        "AUTO_BACKUP", "1") not in ("0", "false", "False", "no", "NO", "off", "OFF", "")
    BACKUP_CHAT_ID: int  = int(os.getenv("BACKUP_CHAT_ID", "0") or 0)
    BACKUP_HOUR: int     = int(os.getenv("BACKUP_HOUR", "4") or 0)
    BACKUP_KEEP: int     = int(os.getenv("BACKUP_KEEP", "7") or 0)
    ADMIN_IDS: list[int] = field(
        default_factory=lambda: _int_list(os.getenv("ADMIN_IDS", ""))
    )

    @property
    def admins_configured(self) -> bool:
        """
        True when at least one admin is configured.

        An empty ADMIN_IDS used to mean "every user is an admin", which handed
        /login (it rewrites the session file), /import and /export to anyone.
        The bot now fails closed and refuses to start without an admin.
        """
        return bool(self.ADMIN_IDS)

    @property
    def SESSION_PATH(self) -> str:
        return str(PROJECT_DIR / self.SESSION_NAME)

    @property
    def SESSION_FILE(self) -> str:
        return str(PROJECT_DIR / f"{self.SESSION_NAME}.session")

    @property
    def BOT_SESSION_PATH(self) -> str:
        # On-disk session file used by the bot MTProto download client when no
        # BOT_TELETHON_STRING_SESSION is provided.
        return str(PROJECT_DIR / f"{self.SESSION_NAME}_bot")


config = Config()
