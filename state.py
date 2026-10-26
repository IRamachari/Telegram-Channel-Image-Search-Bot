"""
Lightweight persistent state — survives bot restarts.
Stored as JSON next to the chroma_db directory.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path

from config import config

log = logging.getLogger(__name__)
# Kept INSIDE the database directory so it is included in /export backups and
# replaced by /import, the same as the index watermarks.
_STATE_FILE = Path(config.CHROMA_DIR) / "bot_state.json"


def _load() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception as e:
            log.warning("State load error: %s", e)
    return {}


def _save(state: dict) -> None:
    """
    Persist state atomically.

    ``Path.write_text`` truncates first and writes second, so a crash in between
    (OOM kill is routine on a small box) leaves an empty or half-written file
    and the pinned-message registry is lost. A same-directory temp file plus
    ``os.replace`` is atomic on POSIX.
    """
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_name(_STATE_FILE.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(state, indent=2))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, _STATE_FILE)


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
