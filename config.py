from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.resolve()
load_dotenv(PROJECT_DIR / ".env")


def _int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class Config:
    API_ID: int          = int(os.getenv("TELEGRAM_API_ID", "0"))
    API_HASH: str        = os.getenv("TELEGRAM_API_HASH", "")
    SESSION_NAME: str    = os.getenv("TELEGRAM_SESSION", "indexer_session")
    BOT_TOKEN: str       = os.getenv("BOT_TOKEN", "")
    CHROMA_DIR: str      = str(PROJECT_DIR / os.getenv("CHROMA_DIR", "chroma_db"))
    COLLECTION: str      = os.getenv("COLLECTION_NAME", "telegram_images")
    TEMP_DIR: str        = str(PROJECT_DIR / os.getenv("TEMP_DIR", "temp_images"))
    BATCH_SIZE: int      = int(os.getenv("BATCH_SIZE", "32"))
    THRESHOLD: float     = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    MAX_RESULTS: int     = int(os.getenv("MAX_RESULTS", "10"))
    PARALLEL_DL: int     = int(os.getenv("PARALLEL_DOWNLOADS", "10"))
    ADMIN_IDS: list[int] = field(
        default_factory=lambda: _int_list(os.getenv("ADMIN_IDS", ""))
    )

    @property
    def SESSION_PATH(self) -> str:
        return str(PROJECT_DIR / self.SESSION_NAME)

    @property
    def SESSION_FILE(self) -> str:
        return str(PROJECT_DIR / f"{self.SESSION_NAME}.session")


config = Config()
