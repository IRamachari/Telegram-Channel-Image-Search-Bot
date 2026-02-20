from __future__ import annotations
import logging
import shutil

import psutil

log = logging.getLogger(__name__)


def detect_specs() -> dict:
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False) or cpu_count
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    return {
        "cpu_logical": cpu_count,
        "cpu_physical": cpu_physical,
        "ram_total_mb": mem.total / 1024 / 1024,
        "ram_avail_mb": mem.available / 1024 / 1024,
        "disk_free_mb": disk.free / 1024 / 1024,
    }


def auto_tune(model_size_mb: float = 0) -> dict:
    specs = detect_specs()
    cpu = specs["cpu_logical"]
    ram = specs["ram_total_mb"]
    disk = specs["disk_free_mb"]

    intra_threads = max(2, min(cpu // 2, 16))
    inter_threads = max(1, min(cpu // 4, 8))

    if model_size_mb > 300:
        base_batch = 8
    elif model_size_mb > 100:
        base_batch = 32
    else:
        base_batch = 64

    if ram >= 16384:
        ram_mult = 4
    elif ram >= 8192:
        ram_mult = 2
    elif ram >= 4096:
        ram_mult = 1.5
    elif ram >= 2048:
        ram_mult = 1
    else:
        ram_mult = 0.5

    batch_size = max(4, min(int(base_batch * ram_mult), 256))
    power = 1
    while power * 2 <= batch_size:
        power *= 2
    batch_size = power

    if ram >= 8192 and cpu >= 8:
        parallel = 15
    elif ram >= 4096:
        parallel = min(12, cpu)
    elif ram >= 2048:
        parallel = min(10, cpu)
    elif ram >= 1024:
        parallel = min(8, cpu)
    else:
        parallel = min(5, cpu)

    if ram >= 32768 and cpu >= 16:
        model = "vit-l-14"
        model_name = "ViT-L/14 (you have enough power)"
    elif ram >= 4096:
        model = "vit-b-16"
        model_name = "ViT-B/16 (balanced)"
    else:
        model = "vit-b-32"
        model_name = "ViT-B/32 (lightweight)"

    max_images = int(disk * 1024 / 2 * 0.8)

    result = {
        "specs": specs,
        "intra_threads": intra_threads,
        "inter_threads": inter_threads,
        "batch_size": batch_size,
        "parallel_downloads": parallel,
        "recommended_model": model,
        "recommended_model_name": model_name,
        "max_images_estimate": max_images,
    }

    log.info(
        "Auto-tune: cpu=%d ram=%.0fMB model=%.0fMB → "
        "batch=%d parallel=%d threads=%d/%d",
        cpu, ram, model_size_mb, batch_size, parallel,
        intra_threads, inter_threads,
    )
    return result


_applied: dict = {}


def apply_tune(model_size_mb: float = 0) -> dict:
    global _applied
    _applied = auto_tune(model_size_mb)
    return _applied


def get_applied() -> dict:
    if not _applied:
        return apply_tune()
    return _applied


def get_batch_size() -> int:
    return get_applied().get("batch_size", 32)


def get_parallel() -> int:
    return get_applied().get("parallel_downloads", 8)


def get_threads() -> tuple[int, int]:
    a = get_applied()
    return a.get("intra_threads", 4), a.get("inter_threads", 2)


def format_auto_tune(model_size_mb: float = 0) -> str:
    r = auto_tune(model_size_mb)
    s = r["specs"]
    return (
        "⚙️ *Auto\\-Tune Report*\n\n"
        "*Hardware Detected:*\n"
        f"  🖥 CPU: {s['cpu_physical']} cores / "
        f"{s['cpu_logical']} threads\n"
        f"  🧠 RAM: {s['ram_total_mb']:.0f} MB "
        f"({s['ram_avail_mb']:.0f} MB free)\n"
        f"  💿 Disk: {s['disk_free_mb']:.0f} MB free\n\n"
        "*Applied Settings:*\n"
        f"  📦 Batch size: `{r['batch_size']}`\n"
        f"  ⬇️ Parallel downloads: `{r['parallel_downloads']}`\n"
        f"  🔧 ONNX threads: `{r['intra_threads']}` intra / "
        f"`{r['inter_threads']}` inter\n"
        f"  🧠 Model: `{r['recommended_model_name']}`\n"
        f"  📸 Max images: ~{r['max_images_estimate']:,}\n"
    )
