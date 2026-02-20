from __future__ import annotations
import json
import logging
import os
import urllib.request
import sys
from pathlib import Path

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
}


def get_active_model() -> str | None:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text()).get("active")
    return None


def set_active_model(key: str) -> None:
    STATE_FILE.write_text(json.dumps({"active": key}, indent=2))


def is_model_downloaded() -> bool:
    return ONNX_PATH.exists()


def model_file_size_mb() -> float:
    if ONNX_PATH.exists():
        return ONNX_PATH.stat().st_size / 1024 / 1024
    return 0


def download_model(key: str) -> str:
    if key not in AVAILABLE:
        raise ValueError(f"Unknown model: {key}")
    info = AVAILABLE[key]
    tmp = str(ONNX_PATH) + ".tmp"
    log.info("Downloading %s from %s", info["name"], info["url"])
    try:
        urllib.request.urlretrieve(info["url"], tmp)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Download failed: {exc}")
    if ONNX_PATH.exists():
        os.remove(ONNX_PATH)
    os.rename(tmp, str(ONNX_PATH))
    set_active_model(key)
    size = model_file_size_mb()
    return f"✅ {info['name']} downloaded ({size:.0f} MB)"


def format_model_card(key: str) -> str:
    m = AVAILABLE[key]
    active = get_active_model()
    tag = " 👈 ACTIVE" if key == active else ""
    return (
        f"*{m['name']}*{tag}\n"
        f"  📦 Size: {m['size_mb']} MB\n"
        f"  🧠 RAM: {m['ram_mb']} MB minimum\n"
        f"  🎯 Accuracy: {m['accuracy']}\n"
        f"  ⚡ Speed: {m['speed']}\n"
        f"  💡 Best for: {m['best_for']}\n"
        f"  🖥 Min instance: {m['min_instance']}"
    )
