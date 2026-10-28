"""
Lightweight persistent state — survives bot restarts.
Stored as JSON next to the chroma_db directory.
"""
from __future__ import annotations
import json
import logging
import os
import threading
import uuid
from pathlib import Path

from config import config

log = logging.getLogger(__name__)
_STATE_FILE = Path(config.CHROMA_DIR) / "bot_state.json"
_LOCK = threading.Lock()


def _load() -> dict:
    if _STATE_FILE.exists():
        try:
            data = json.loads(_STATE_FILE.read_text())
            if isinstance(data, dict):
                return data
            log.warning("State file is not a JSON object — ignoring")
        except Exception as e:
            log.warning("State load error: %s", e)
    return {}


def _save(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_name(
        f"{_STATE_FILE.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(state, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _STATE_FILE)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def get(key: str, default=None):
    with _LOCK:
        return _load().get(key, default)


def put(key: str, value) -> None:
    with _LOCK:
        s = _load()
        s[key] = value
        _save(s)


def delete(key: str) -> None:
    with _LOCK:
        s = _load()
        s.pop(key, None)
        _save(s)
