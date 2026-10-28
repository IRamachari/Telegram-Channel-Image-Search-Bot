from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import config

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
ONNX_PATH = MODEL_DIR / "clip_visual.onnx"
STATE_FILE = MODEL_DIR / "model_state.json"

# `size_bytes` and `sha256` are the exact values published by the Hugging Face
# API for each LFS file. They are verified after every download, so a truncated
# transfer or a proxy's HTML error page can never be installed as a model.
AVAILABLE = {
    "vit-b-32": {
        "name": "CLIP ViT-B/32",
        "url": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 85,
        "size_bytes": 89117001,
        "sha256": "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
        "ram_mb": 350,
        "accuracy": "⭐⭐⭐ Good",
        "speed": "⚡⚡⚡ Fast",
        "best_for": "General use, low-RAM servers",
        "min_instance": "Micro (512 MB)",
    },
    "vit-b-16": {
        "name": "CLIP ViT-B/16",
        "url": "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 83.4,
        "size_bytes": 87461602,
        "sha256": "44eece4fe5fe4e0359a88268a327adf758633a1aade3917690b952bef1501f96",
        "ram_mb": 500,
        "accuracy": "⭐⭐⭐⭐ Better",
        "speed": "⚡⚡ Medium",
        "best_for": "Better matching, handles edits",
        "min_instance": "Small (1 GB)",
    },
    "vit-l-14": {
        "name": "CLIP ViT-L/14",
        "url": "https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 292.8,
        "size_bytes": 307001500,
        "sha256": "9a658de8944a2001719beabf6c5030cfe7b3dc80e85983f69fc648d6b4d7ec0f",
        "ram_mb": 1200,
        "accuracy": "⭐⭐⭐⭐⭐ Best",
        "speed": "⚡ Slow",
        "best_for": "Maximum accuracy",
        "min_instance": "Medium (2 GB)",
    },
    # 336px variants: same CLIP preprocessing (mean/std, shortest-edge resize +
    # centre crop), only the input resolution changes. Measurably better
    # zero-shot retrieval than their 224px counterparts on fine detail.
    "vit-l-14-336": {
        "name": "CLIP ViT-L/14 @336px",
        "url": "https://huggingface.co/Xenova/clip-vit-large-patch14-336/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 293.1,
        "size_bytes": 307331740,
        "sha256": "ee6c57374a9450be119b3c9f8768ff52ea27bfa521482b0eedeaa90b59a2ba9b",
        "ram_mb": 1800,
        "accuracy": "⭐⭐⭐⭐⭐ Best+",
        "speed": "⚡ Slow",
        "best_for": "Highest quality; detail-heavy channels",
        "min_instance": "Large (2-3 GB)",
        # "center": resize shortest edge then centre-crop (OpenAI CLIP standard)
        "crop": "center",
        "input": 336,
    },
    # NOTE: Chinese-CLIP is NOT offered. Its ONNX exports (Xenova org) are
    # combined text+image graphs that require input_ids/attention_mask, and no
    # vision-only export exists — verified against every Xenova chinese-clip
    # repo. A URL pointing at a non-existent vision_model_quantized.onnx made
    # /model fail with HTTP 404 for that entry.
}


def model_settings(key: str) -> dict:
    """Preprocessing settings for a model key, with safe defaults."""
    info = AVAILABLE.get(key) or {}
    # Default to the historical behaviour: shortest-edge resize + centre crop
    # at 224px. Anything unknown must not silently change the preprocessing.
    return {
        "crop": info.get("crop", "center"),
        "input": int(info.get("input", 224)),
    }


def get_active_model() -> str | None:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text()).get("active")
        except Exception:
            return None
    return None


def get_active_settings() -> dict:
    """
    Preprocessing settings for the model that is currently on disk.

    Falls back to the historical defaults (224px, shortest-edge + centre crop)
    so an unknown or missing model never silently changes preprocessing.
    """
    key = get_active_model()
    if not key:
        return {"crop": "center", "input": 224, "active": None}
    out = model_settings(key)
    out["active"] = key
    return out


def set_active_model(key: str) -> None:
    # Keep the preprocessing mode next to the model key: the embedder reads it
    # to decide how to resize, and non-CLIP architectures differ here.
    payload = {"active": key}
    payload.update(model_settings(key))
    # Atomic: a crash mid-write must not leave an unparseable state file, which
    # would silently reset the active model (and its preprocessing) to defaults.
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, STATE_FILE)


def is_model_downloaded() -> bool:
    return ONNX_PATH.exists()


def model_file_size_mb() -> float:
    if ONNX_PATH.exists():
        return ONNX_PATH.stat().st_size / 1024 / 1024
    return 0


# ── Download + integrity ─────────────────────────────────────────────────────

MIN_MODEL_BYTES = 1024 * 1024
_CHUNK = 256 * 1024
_TIMEOUT = 30


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_range_total(value: str | None) -> int | None:
    """Total size out of a `Content-Range: bytes 100-999/1000` header."""
    if not value or "/" not in value:
        return None
    tail = value.rsplit("/", 1)[-1].strip()
    return int(tail) if tail.isdigit() else None


def _resume_download(url: str, dest: Path, expected: int, progress=None) -> None:
    """
    Fetch `url` into `dest`, continuing from whatever is already there.

    The partial file is only ever appended to, so a connection that dies at
    90 % of a 293 MB model costs one retry instead of a full re-download.
    """
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have > expected:
        log.warning("Discarding oversized partial download (%d > %d bytes)",
                    have, expected)
        dest.unlink()
        have = 0
    if expected and have == expected:
        return                      # already complete — the caller verifies it
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        log.info("Resuming download at %d/%d bytes", have, expected or 0)
    try:
        resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:
            return                  # server says we already hold every byte
        raise
    with resp:
        if have and getattr(resp, "status", 200) != 206:
            log.warning("Server ignored the Range request — restarting download")
            have = 0
        total = (_content_range_total(resp.headers.get("Content-Range"))
                 or expected
                 or (int(resp.headers.get("Content-Length") or 0) + have))
        with open(dest, "ab" if have else "wb") as fh:
            done = have
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress is not None:
                    try:
                        progress(done, total)
                    except Exception:
                        pass


def verify_model(path: str | Path | None = None, key: str | None = None) -> dict:
    """
    Check a model file against the registry (size, sha256, ONNX plausibility).

    `key` picks the expected spec; without it the active model is used, and
    when that is unknown only the generic checks (size floor, not-text) run.
    """
    p = Path(path) if path else ONNX_PATH
    key = key or get_active_model()
    info = AVAILABLE.get(key or "", {})
    expected_bytes = int(info.get("size_bytes") or 0)
    expected_sha = info.get("sha256") or None
    out = {
        "path": str(p), "key": key, "ok": False,
        "size_bytes": 0, "expected_bytes": expected_bytes,
        "sha256": None, "expected_sha256": expected_sha, "reason": "",
    }
    if not p.exists():
        out["reason"] = "file not found"
        return out
    size = p.stat().st_size
    out["size_bytes"] = size
    if size < MIN_MODEL_BYTES:
        out["reason"] = f"file is only {size} bytes"
        return out
    if expected_bytes and size != expected_bytes:
        out["reason"] = f"size {size} != expected {expected_bytes} bytes"
        return out
    with open(p, "rb") as fh:
        head = fh.read(512)
    # A proxy/captive-portal page is text; a real ONNX protobuf starts with a
    # binary field tag, never '<' or '{'.
    if head[:1] in (b"<", b"{") or b"<html" in head.lower():
        out["reason"] = "file is text/HTML, not an ONNX model"
        return out
    if expected_sha:
        out["sha256"] = _sha256_of(p)
        if out["sha256"] != expected_sha:
            out["reason"] = "sha256 mismatch"
            return out
    out["ok"] = True
    out["reason"] = "verified" if expected_sha else "size ok (no sha256 in registry)"
    return out


def download_model(key: str, progress=None) -> str:
    """
    Download + verify a model, then install it and record it as active.

    `progress(done, total)` is optional and only used for reporting. A network
    failure keeps the `.part` file so the next attempt resumes; a body that is
    complete but wrong is deleted, because it can never become valid.
    """
    if key not in AVAILABLE:
        raise ValueError(f"Unknown model: {key}")
    info = AVAILABLE[key]
    tmp = ONNX_PATH.parent / (ONNX_PATH.name + ".part")
    expected = int(info.get("size_bytes") or 0)
    if expected:
        free = shutil.disk_usage(ONNX_PATH.parent).free
        if free < expected + 64 * 1024 * 1024:
            raise RuntimeError(
                f"Not enough disk space: need ~{expected / 1048576:.0f} MB, "
                f"have {free / 1048576:.0f} MB free")
    log.info("Downloading %s from %s", info["name"], info["url"])
    try:
        _resume_download(info["url"], tmp, expected, progress)
        check = verify_model(tmp, key=key)
        if not check["ok"]:
            if (check["size_bytes"] and check["expected_bytes"]
                    and check["size_bytes"] < check["expected_bytes"]):
                # Truncated: keep it, the next call resumes from here.
                raise RuntimeError(
                    f"incomplete download ({check['size_bytes']} of "
                    f"{check['expected_bytes']} bytes) — run again to resume")
            tmp.unlink(missing_ok=True)
            raise RuntimeError(check["reason"])
        # Never leave the previous model deleted while the new one is unproven:
        # the old code removed it first, leaving a window with no model at all.
        os.replace(tmp, str(ONNX_PATH))
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}")
    set_active_model(key)
    size = model_file_size_mb()
    return f"✅ {info['name']} downloaded ({size:.0f} MB, sha256 verified)"


# Fingerprint of the embedding model currently in use. Stored INSIDE the
# database directory so it travels with the vectors in every /export backup and
# is replaced by /import — which is exactly when a mismatch must be detected.
#
# The record has TWO halves on purpose:
#   disk    — what is on disk right now (written whenever the file changes)
#   vectors — the model that actually produced the vectors in the store, written
#             only by mark_vectors_indexed() after a successful index run.
# Recording the disk half at download time (which is all a download can know)
# must never silence the "re-index required" warning.
FINGERPRINT_FILE = Path(config.CHROMA_DIR) / "model_fingerprint.json"
FINGERPRINT_VERSION = 2

# Historical default resolution, used only when nothing else is known.
_DEFAULT = 224

_KEEP = object()


def _prep_version():
    """Preprocessing-pipeline version, or None when it cannot be resolved."""
    try:
        from embeddings import PREP_VERSION
        return PREP_VERSION
    except Exception:
        return None


def model_fingerprint() -> dict:
    """
    Cheap identity of the model AND preprocessing pipeline on disk.

    Switching CLIP models invalidates the existing vectors: a dimension change
    makes the vector store reject the next insert, and a same-dimension change
    (B/32 vs B/16 both emit 512-d) silently produces meaningless similarity
    scores with no error. Changing the preprocessing (Image.draft /
    reducing_gap) has the same effect, so the prep version is part of it.
    """
    fp = {"active": get_active_model(), "size_mb": 0.0, "input": _DEFAULT,
          "crop": "center", "prep": _prep_version()}
    if ONNX_PATH.exists():
        st = ONNX_PATH.stat()
        fp["size_mb"] = round(st.st_size / 1024 / 1024, 2)
        # Name + size + seconds-resolution mtime: enough to detect a model swap
        # without hashing hundreds of MB on every start.
        fp["mtime"] = int(st.st_mtime)
    # Resolution and crop mode decide the actual pixels, so two models that
    # happen to share a file size are still distinguished.
    try:
        s = model_settings(fp.get("active"))
        fp["input"] = s["input"]
        fp["crop"] = s["crop"]
    except Exception:
        pass
    return fp


def read_fingerprint() -> dict:
    """
    Whole fingerprint record, normalised to {version, disk, vectors, indexed_at}.

    Returns {} when nothing has ever been recorded. Legacy (flat, v1) files are
    mapped onto BOTH halves so upgrading does not force a spurious re-index.
    """
    if not FINGERPRINT_FILE.exists():
        return {}
    try:
        data = json.loads(FINGERPRINT_FILE.read_text())
    except Exception as exc:
        log.warning("Unreadable model fingerprint %s: %s", FINGERPRINT_FILE, exc)
        return {"version": FINGERPRINT_VERSION, "disk": {}, "vectors": None,
                "indexed_at": None, "corrupt": True}
    if not isinstance(data, dict):
        return {"version": FINGERPRINT_VERSION, "disk": {}, "vectors": None,
                "indexed_at": None, "corrupt": True}
    if "disk" in data or "vectors" in data:
        return {
            "version": int(data.get("version") or FINGERPRINT_VERSION),
            "disk": data.get("disk") or {},
            "vectors": data.get("vectors"),
            "indexed_at": data.get("indexed_at"),
        }
    return {"version": 1, "disk": data, "vectors": data,
            "indexed_at": data.get("indexed_at")}


def read_model_fingerprint() -> dict | None:
    """The identity of the model that produced the stored vectors (or None)."""
    return read_fingerprint().get("vectors")


def _write_fingerprint(disk: dict, vectors=_KEEP, indexed_at=_KEEP) -> None:
    cur = read_fingerprint()
    rec = {
        "version": FINGERPRINT_VERSION,
        "disk": disk,
        "vectors": cur.get("vectors") if vectors is _KEEP else vectors,
        "indexed_at": cur.get("indexed_at") if indexed_at is _KEEP else indexed_at,
    }
    tmp = FINGERPRINT_FILE.with_name(FINGERPRINT_FILE.name + ".tmp")
    try:
        FINGERPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(rec, indent=2))
        os.replace(tmp, FINGERPRINT_FILE)
    except Exception as exc:
        log.warning("Could not write model fingerprint: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def write_model_fingerprint() -> None:
    """
    Record the model that is on disk. Never touches the vectors half.

    This is safe to call right after a download. Marking the vectors as
    up-to-date is mark_vectors_indexed()'s job — a download proves nothing
    about the vectors that are already in the store.
    """
    _write_fingerprint(model_fingerprint())


def mark_vectors_indexed() -> None:
    """
    Record that the vectors in the store were produced by the model on disk.

    Call ONLY after a complete (re-)index with the current model — i.e. at the
    end of a successful index run, never on a download or a partial run.
    """
    disk = model_fingerprint()
    _write_fingerprint(disk, vectors=disk, indexed_at=int(time.time()))


def model_status() -> dict:
    """
    Compare the model on disk with the model that produced the vectors.

    state: "ok"          vectors match the model on disk
           "missing"     nothing recorded yet (fresh store, or a store indexed
                         before fingerprinting existed) — the caller knows
                         whether the store is empty and can decide to warn
           "unreadable"  fingerprint file is corrupt
           "mismatch"    vectors came from a different model / preprocessing
    """
    disk = model_fingerprint()
    rec = read_fingerprint()
    vectors = rec.get("vectors")
    if rec.get("corrupt"):
        state, detail = "unreadable", "model fingerprint file is corrupt"
    elif not rec:
        state, detail = "missing", "no model fingerprint recorded yet"
    elif not vectors:
        state, detail = "missing", "only the on-disk model is recorded"
    elif vectors != disk:
        state, detail = "mismatch", "vectors were built with a different model"
    else:
        state, detail = "ok", "vectors match the model on disk"
    return {"state": state, "detail": detail, "disk": disk,
            "vectors": vectors, "indexed_at": rec.get("indexed_at")}


def model_changed() -> bool:
    """True when the on-disk model differs from the one the vectors came from."""
    return model_status()["state"] in ("mismatch", "unreadable")


def format_model_card(key: str) -> str:
    m = AVAILABLE[key]
    active = get_active_model()
    tag = " 👈 ACTIVE" if key == active else ""
    s = model_settings(key)
    res = f"  📐 Input: {s['input']}px\n" if s["input"] != 224 else ""
    # Show the real file size for the installed model, the catalogue size
    # otherwise: they can differ if the registry entry is ever wrong.
    size = m["size_mb"]
    if key == active and ONNX_PATH.exists():
        size = round(model_file_size_mb(), 1)
    return (
        f"*{m['name']}*{tag}\n"
        f"  📦 Size: {size} MB\n"
        f"  🧠 RAM: {m['ram_mb']} MB minimum\n"
        f"{res}"
        f"  🎯 Accuracy: {m['accuracy']}\n"
        f"  ⚡ Speed: {m['speed']}\n"
        f"  💡 Best for: {m['best_for']}\n"
        f"  🖥 Min instance: {m['min_instance']}"
    )
