from __future__ import annotations
import json
import logging
import os
import shutil
import threading
from pathlib import Path

import psutil

log = logging.getLogger(__name__)


def detect_specs() -> dict:
    # psutil.cpu_count() returns None when the topology can't be determined
    # (common in restricted containers). Without the fallback, cpu//2 raises
    # TypeError and /autotune fails with a bare error to the admin.
    cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
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

    # ORT intra-op threads should follow the PHYSICAL core count: the docs state
    # "INTRA Threads Total = Number of physical CPU Cores" and note that leaving
    # intra_op_num_threads at 0 also enables affinitization. Dividing the logical
    # count in half threw away ~half the available compute on the single hottest
    # operation this bot performs. Conservative by design: the embedding pool is
    # capped at 2 threads, so this never saturates a shared box.
    cpu = specs["cpu_physical"]
    intra_threads = max(2, min(cpu, 16))
    inter_threads = 1
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

    # Recommendation only — the operator can always override with /model. The
    # 336px variant is preferred whenever RAM allows because it is measurably
    # more accurate than its 224px counterpart on fine detail.
    if ram >= 32768 and cpu >= 16:
        model = "vit-l-14"
        model_name = "ViT-L/14 (you have enough power)"
    elif ram >= 2048:
        model = "vit-l-14-336"
        model_name = "ViT-L/14 @336px (best accuracy)"
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

# ── Manual overrides (/tune) ─────────────────────────────────────────────────
# Persisted in tune.json next to the code so they survive restarts. A typo
# can't push ORT into spawning hundreds of threads or the downloader into
# hammering Telegram's flood limits: every knob is bounded here.
_OVERRIDE_FILE = Path(__file__).parent / "tune.json"
_TUNE_KEYS = {
    "intra_threads": (1, 32),
    "inter_threads": (1, 8),
    "batch_size": (1, 256),
    "parallel_downloads": (1, 32),
}
_ALIASES = {
    "intra": "intra_threads",
    "intra_op": "intra_threads",
    "inter": "inter_threads",
    "inter_op": "inter_threads",
    "batch": "batch_size",
    "parallel": "parallel_downloads",
    "dl": "parallel_downloads",
}
_tune_mu = threading.Lock()


def _load_overrides() -> dict:
    """Read tune.json, ignoring unknown/corrupt entries defensively."""
    try:
        data = json.loads(_OVERRIDE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Ignoring corrupt %s: %s", _OVERRIDE_FILE, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    for k, v in data.items():
        k = _ALIASES.get(str(k).lower(), str(k).lower())
        if k in _TUNE_KEYS and isinstance(v, int) and not isinstance(v, bool):
            lo, hi = _TUNE_KEYS[k]
            out[k] = max(lo, min(hi, v))
    return out


def _save_overrides() -> None:
    tmp = _OVERRIDE_FILE.with_name(_OVERRIDE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(_overrides, indent=2), encoding="utf-8")
    os.replace(tmp, _OVERRIDE_FILE)


def apply_tune(model_size_mb: float = 0) -> dict:
    """Run auto_tune and lay manual /tune overrides on top of it."""
    global _applied
    with _tune_mu:
        r = auto_tune(model_size_mb)
        # Sorted list so the report is stable and the dict stays JSON-safe.
        r["_manual"] = sorted(_overrides)
        for k, v in _overrides.items():
            r[k] = v
        _applied = r
        return r


def set_tune(**kv) -> dict:
    """
    Validate + persist manual overrides. Raises ValueError on a bad key or
    value; the caller is responsible for re-running apply_tune() afterwards.
    """
    global _overrides
    with _tune_mu:
        norm = {}
        for k, v in kv.items():
            k = _ALIASES.get(str(k).lower(), str(k).lower())
            if k not in _TUNE_KEYS:
                raise ValueError(f"unknown setting '{k}'")
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"'{k}' must be an integer")
            lo, hi = _TUNE_KEYS[k]
            if not lo <= v <= hi:
                raise ValueError(f"'{k}' must be between {lo} and {hi}")
            norm[k] = v
        _overrides.update(norm)
        _save_overrides()
    return dict(_overrides)


def reset_tune() -> None:
    """Drop every manual override (and tune.json) — back to full auto."""
    global _overrides
    with _tune_mu:
        _overrides = {}
        try:
            _OVERRIDE_FILE.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.warning("Could not remove %s: %s", _OVERRIDE_FILE, exc)


# Loaded once at import: get_batch_size()/get_threads() consult _applied, which
# apply_tune() fills — so a fresh process picks up persisted overrides on its
# very first model load, with no explicit call needed.
_overrides: dict = _load_overrides()


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
    r = apply_tune(model_size_mb)
    s = r["specs"]
    manual = set(r.get("_manual", ()))

    def mark(k: str) -> str:
        return " ✋" if k in manual else ""

    text = (
        "⚙️ *Tune Report*\n\n"
        "*Hardware Detected:*\n"
        f"  🖥 CPU: {s['cpu_physical']} cores / "
        f"{s['cpu_logical']} threads\n"
        f"  🧠 RAM: {s['ram_total_mb']:.0f} MB "
        f"({s['ram_avail_mb']:.0f} MB free)\n"
        f"  💿 Disk: {s['disk_free_mb']:.0f} MB free\n\n"
        "*Applied Settings:*\n"
        f"  📦 Batch size: `{r['batch_size']}`{mark('batch_size')}\n"
        f"  ⬇️ Parallel downloads: `{r['parallel_downloads']}`"
        f"{mark('parallel_downloads')}\n"
        f"  🔧 ONNX threads: `{r['intra_threads']}` intra / "
        f"`{r['inter_threads']}` inter{mark('intra_threads')}\n"
        f"  🧠 Model: `{r['recommended_model_name']}`\n"
        f"  📸 Max images: ~{r['max_images_estimate']:,}\n"
    )
    if manual:
        text += "\n✋ manual override — /tune reset for full auto\n"
    return text
