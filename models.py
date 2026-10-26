from __future__ import annotations
import json
import logging
import os
import urllib.request
import sys
from pathlib import Path

from config import config

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
ONNX_PATH = MODEL_DIR / "clip_visual.onnx"
STATE_FILE = MODEL_DIR / "model_state.json"

AVAILABLE = {
    "vit-b-32": {
        "name": "CLIP ViT-B/32",
        "url": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 85,
        "ram_mb": 350,
        "accuracy": "⭐⭐⭐ Good",
        "speed": "⚡⚡⚡ Fast",
        "best_for": "General use, low-RAM servers",
        "min_instance": "Micro (512 MB)",
    },
    "vit-b-16": {
        "name": "CLIP ViT-B/16",
        "url": "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 110,
        "ram_mb": 500,
        "accuracy": "⭐⭐⭐⭐ Better",
        "speed": "⚡⚡ Medium",
        "best_for": "Better matching, handles edits",
        "min_instance": "Small (1 GB)",
    },
    "vit-l-14": {
        "name": "CLIP ViT-L/14",
        "url": "https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 350,
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
        "size_mb": 307,
        "ram_mb": 1800,
        "accuracy": "⭐⭐⭐⭐⭐ Best+",
        "speed": "⚡ Slow",
        "best_for": "Highest quality; detail-heavy channels",
        "min_instance": "Large (2-3 GB)",
        # "center": resize shortest edge then centre-crop (OpenAI CLIP standard)
        "crop": "center",
        "input": 336,
    },
    # Chinese-CLIP: a proper CLIP trained on large-scale Chinese data. Its
    # feature extractor resizes straight to (336,336) rather than resizing the
    # shortest edge and cropping, hence crop="stretch".
    "chinese-clip-l-336": {
        "name": "Chinese-CLIP ViT-L/14 @336px",
        "url": "https://huggingface.co/Xenova/chinese-clip-vit-large-patch14-336px/resolve/main/onnx/vision_model_quantized.onnx",
        "size_mb": 307,
        "ram_mb": 1800,
        "accuracy": "⭐⭐⭐⭐ Multilingual",
        "speed": "⚡ Slow",
        "best_for": "CJK / multilingual channel content",
        "min_instance": "Large (2-3 GB)",
        "crop": "stretch",
        "input": 336,
    },
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
    # to decide how to resize, and Chinese-CLIP differs from OpenAI CLIP here.
    payload = {"active": key}
    payload.update(model_settings(key))
    STATE_FILE.write_text(json.dumps(payload, indent=2))


def is_model_downloaded() -> bool:
    return ONNX_PATH.exists()


def model_file_size_mb() -> float:
    if ONNX_PATH.exists():
        return ONNX_PATH.stat().st_size / 1024 / 1024
    return 0


# Fingerprint of the embedding model currently in use. Stored INSIDE the
# database directory so it travels with the vectors in every /export backup and
# is replaced by /import — which is exactly when a mismatch must be detected.
FINGERPRINT_FILE = Path(config.CHROMA_DIR) / "model_fingerprint.json"

# Historical default resolution, used only when nothing else is known.
_DEFAULT = 224


def model_fingerprint() -> dict:
    """
    Cheap identity of the model AND preprocessing pipeline that produced the
    embeddings on disk.

    Switching CLIP models invalidates the existing vectors: a dimension change
    makes ChromaDB raise on the next insert, and a same-dimension change (B/32
    vs B/16) silently produces meaningless similarity scores with no error.
    Changing the preprocessing (Image.draft / reducing_gap) has the same effect,
    so the prep version is part of the fingerprint.
    """
    fp = {"active": get_active_model(), "size_mb": 0.0, "input": _DEFAULT,
          "crop": "center"}
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
    try:
        from embeddings import PREP_VERSION
        fp["prep"] = PREP_VERSION
    except Exception:
        pass
    return fp


def read_model_fingerprint() -> dict | None:
    if FINGERPRINT_FILE.exists():
        try:
            return json.loads(FINGERPRINT_FILE.read_text())
        except Exception:
            return None
    return None


def write_model_fingerprint() -> None:
    try:
        FINGERPRINT_FILE.write_text(json.dumps(model_fingerprint(), indent=2))
    except Exception as exc:
        log.warning("Could not write model fingerprint: %s", exc)


def model_changed() -> bool:
    """True when the on-disk model differs from the one the vectors came from."""
    stored = read_model_fingerprint()
    if stored is None:
        return False          # nothing recorded — treat as compatible
    return stored != model_fingerprint()


def download_model(key: str) -> str:
    if key not in AVAILABLE:
        raise ValueError(f"Unknown model: {key}")
    info = AVAILABLE[key]
    tmp = str(ONNX_PATH) + ".tmp"
    log.info("Downloading %s from %s", info["name"], info["url"])
    try:
        urllib.request.urlretrieve(info["url"], tmp)
        if not Path(tmp).exists() or Path(tmp).stat().st_size < 1024 * 1024:
            raise RuntimeError("Downloaded model is suspiciously small — aborting.")
        # Never leave the previous model deleted while the new one is unproven:
        # the old code removed it first, leaving a window with no model at all.
        os.replace(tmp, str(ONNX_PATH))
    except Exception as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise RuntimeError(f"Download failed: {exc}")
    set_active_model(key)
    size = model_file_size_mb()
    return f"✅ {info['name']} downloaded ({size:.0f} MB)"


def format_model_card(key: str) -> str:
    m = AVAILABLE[key]
    active = get_active_model()
    tag = " 👈 ACTIVE" if key == active else ""
    s = model_settings(key)
    res = f"  📐 Input: {s['input']}px\n" if s["input"] != 224 else ""
    return (
        f"*{m['name']}*{tag}\n"
        f"  📦 Size: {m['size_mb']} MB\n"
        f"  🧠 RAM: {m['ram_mb']} MB minimum\n"
        f"{res}"
        f"  🎯 Accuracy: {m['accuracy']}\n"
        f"  ⚡ Speed: {m['speed']}\n"
        f"  💡 Best for: {m['best_for']}\n"
        f"  🖥 Min instance: {m['min_instance']}"
    )
