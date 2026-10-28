from __future__ import annotations
import shutil
from pathlib import Path

import psutil

from config import config


def get_system_stats() -> dict:
    # cpu_count() can be None in restricted containers; never print "None cores".
    cpu_count = psutil.cpu_count(logical=True) or __import__("os").cpu_count() or 1
    cpu_pct = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/")

    chroma_path = Path(config.CHROMA_DIR)
    chroma_size = 0
    if chroma_path.exists():
        for f in chroma_path.rglob("*"):
            if f.is_file():
                chroma_size += f.stat().st_size

    model_path = Path(__file__).parent / "clip_visual.onnx"
    model_mb = model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0

    return {
        "cpu_count": cpu_count,
        "cpu_pct": cpu_pct,
        "ram_total_mb": mem.total / 1024 / 1024,
        "ram_used_mb": mem.used / 1024 / 1024,
        "ram_avail_mb": mem.available / 1024 / 1024,
        "ram_pct": mem.percent,
        "disk_total_mb": disk.total / 1024 / 1024,
        "disk_used_mb": disk.used / 1024 / 1024,
        "disk_free_mb": disk.free / 1024 / 1024,
        "disk_pct": (disk.used / disk.total) * 100,
        "chroma_mb": chroma_size / 1024 / 1024,
        "model_mb": model_mb,
    }


def format_system_stats() -> str:
    s = get_system_stats()
    return (
        "🖥 *System Resources*\n\n"
        f"*CPU:* {s['cpu_count']} cores — {s['cpu_pct']:.0f}%\n"
        f"{_bar(s['cpu_pct'])}\n\n"
        f"*RAM:* {s['ram_used_mb']:.0f} / {s['ram_total_mb']:.0f} MB "
        f"({s['ram_avail_mb']:.0f} MB free)\n"
        f"{_bar(s['ram_pct'])}\n\n"
        f"*Disk:* {s['disk_used_mb']:.0f} / {s['disk_total_mb']:.0f} MB "
        f"({s['disk_free_mb']:.0f} MB free)\n"
        f"{_bar(s['disk_pct'])}\n\n"
        f"*ChromaDB:* {s['chroma_mb']:.1f} MB\n"
        f"*Model:* {s['model_mb']:.0f} MB\n"
    )


def _bar(pct, width=20):
    filled = int(pct / 100 * width)
    empty = width - filled
    icon = "🟢" if pct < 60 else "🟡" if pct < 85 else "🔴"
    return f"{icon} `[{'█' * filled}{'░' * empty}]` {pct:.0f}%"


def per_image_bytes(vector_dim: int = 768) -> int:
    """Bytes one indexed image costs (vector blob + HNSW + row overhead)."""
    bytes_per = vector_dim * 4
    overhead = 256
    multiplier = 1.5
    return int((bytes_per + overhead) * multiplier)
    bytes_per = vector_dim * 4
    overhead = 256
    multiplier = 1.5
    per_image = (bytes_per + overhead) * multiplier
    total = photo_count * per_image
    disk_free = shutil.disk_usage("/").free
    return {
        "per_image_kb": per_image / 1024,
        "total_mb": total / 1024 / 1024,
        "disk_free_mb": disk_free / 1024 / 1024,
        "fits": total < disk_free * 0.9,
    }
