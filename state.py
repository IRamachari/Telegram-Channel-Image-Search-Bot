"""
Lightweight persistent state — survives bot restarts.
Stored as JSON next to the chroma_db directory.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

from config import config

log = logging.getLogger(__name__)
_STATE_FILE = Path(config.CHROMA_DIR).parent / "bot_state.json"


def _load() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception as e:
            log.warning("State load error: %s", e)
    return {}


def _save(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def get(key: str, default=None):
    return _load().get(key, default)


def put(key: str, value) -> None:
    s = _load()
    s[key] = value
    _save(s)


def delete(key: str) -> None:
    s = _load()
    s.pop(key, None)
    _save(s)
