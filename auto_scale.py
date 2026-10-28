from __future__ import annotations
import json
import logging
import os
import shutil
import threading
from pathlib import Path

import psutil

log = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).parent

# ── Container limits ─────────────────────────────────────────────────────────
# psutil reads /proc/cpuinfo and /proc/meminfo, which most container runtimes do
# NOT namespace. A 1 vCPU / 512 MB container therefore reports the whole host,
# and the old code happily spawned 16 ONNX intra-op threads and recommended a
# 1.8 GB model on a box that cannot hold it. Prefer the cgroup quota whenever it
# is smaller than what psutil sees.
_CG_MEM_V2 = Path("/sys/fs/cgroup/memory.max")
_CG_MEM_V2_USED = Path("/sys/fs/cgroup/memory.current")
_CG_MEM_V1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CG_MEM_V1_USED = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
_CG_CPU_V2 = Path("/sys/fs/cgroup/cpu.max")
_CG_CPU_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CG_CPU_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")

# cgroup v1 reports "unlimited" as a near-2**63 sentinel rather than "max".
_UNLIMITED_BYTES = 1 << 60


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except Exception:
        return None


def _cgroup_memory_mb() -> tuple[float, float] | None:
    """(limit_mb, used_mb) from the cgroup, or None when absent/unlimited."""
    raw = _read(_CG_MEM_V2)
    used = _read(_CG_MEM_V2_USED)
    if raw is None:
        raw = _read(_CG_MEM_V1)
        used = _read(_CG_MEM_V1_USED)
    if not raw or raw == "max" or not raw.isdigit():
        return None
    limit = int(raw)
    if limit <= 0 or limit >= _UNLIMITED_BYTES:
        return None
    used_mb = int(used) / 1024 / 1024 if used and used.isdigit() else 0.0
    return limit / 1024 / 1024, used_mb


def _cgroup_cpu() -> float | None:
    """CPU quota in whole-ish cores (1.5 for a 150% quota), or None."""
    raw = _read(_CG_CPU_V2)
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            period = int(parts[1])
            if period > 0:
                return int(parts[0]) / period
    quota, period = _read(_CG_CPU_V1_QUOTA), _read(_CG_CPU_V1_PERIOD)
    if (quota and period and quota.isdigit() and period.isdigit()
            and int(quota) > 0 and int(period) > 0):
        return int(quota) / int(period)
    return None


def detect_specs() -> dict:
    # psutil.cpu_count() returns None when the topology can't be determined
    # (common in restricted containers). Without the fallback, cpu//2 raises
    # TypeError and /autotune fails with a bare error to the admin.
    cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    cpu_physical = psutil.cpu_count(logical=False) or cpu_count
    mem = psutil.virtual_memory()
    ram_total_mb = mem.total / 1024 / 1024
    ram_avail_mb = mem.available / 1024 / 1024
    disk = shutil.disk_usage(_PROJECT_DIR)

    limited = False
    cg_mem = _cgroup_memory_mb()
    if cg_mem:
        limit_mb, used_mb = cg_mem
        if limit_mb < ram_total_mb:
            ram_total_mb = limit_mb
            # The host's "available" is meaningless here: what matters is how
            # much of the cgroup allowance is left.
            ram_avail_mb = max(0.0, limit_mb - used_mb)
            limited = True
    cg_cpu = _cgroup_cpu()
    if cg_cpu:
        quota = max(1, int(round(cg_cpu)))
        if quota < cpu_physical:
            cpu_physical = quota
            limited = True
        cpu_count = min(cpu_count, quota)

    return {
        "cpu_logical": cpu_count,
        "cpu_physical": cpu_physical,
        "ram_total_mb": ram_total_mb,
        "ram_avail_mb": ram_avail_mb,
        "disk_free_mb": disk.free / 1024 / 1024,
        "cgroup_limited": limited,
    }


# Model preference, most accurate first. A model is only recommended when the
# box can hold it with headroom: the registry's ram_mb is a *minimum*, so
# picking the biggest model that merely fits leaves nothing for the OS, Python
# and the vector store — which is how auto-tune used to OOM 2 GB boxes with a
# 1.8 GB model.
_REC_HEADROOM = 1.35
_REC_ORDER = ("vit-l-14-336", "vit-l-14", "vit-b-16", "vit-b-32")
_REC_NOTES = {"vit-l-14-336": "best accuracy", "vit-l-14": "224px",
              "vit-b-16": "balanced", "vit-b-32": "lightweight"}


def _recommend(ram_mb: float, cpu: int) -> tuple[str, str]:
    from models import AVAILABLE
    for key in _REC_ORDER:
        info = AVAILABLE[key]
        # A ViT-L/14 on a single core is unusably slow, however much RAM it has.
        if cpu < 2 and key.startswith("vit-l"):
            continue
        if ram_mb >= info["ram_mb"] * _REC_HEADROOM:
            return key, f"{info['name']} ({_REC_NOTES[key]})"
    info = AVAILABLE["vit-b-32"]
    return "vit-b-32", f"{info['name']} ({_REC_NOTES['vit-b-32']})"


def auto_tune(model_size_mb: float = 0) -> dict:
    specs = detect_specs()
    cpu = specs["cpu_physical"]
    ram = specs["ram_total_mb"]
    disk = specs["disk_free_mb"]

    # ORT intra-op threads should follow the PHYSICAL core count: the docs state
    # "INTRA Threads Total = Number of physical CPU Cores" and note that leaving
    # intra_op_num_threads at 0 also enables affinitization. Dividing the logical
    # count in half threw away ~half the available compute on the single hottest
    # operation this bot performs. Capped at 16, and 1 on a single-core box (the
    # old floor of 2 oversubscribed 1 vCPU containers).
    intra_threads = max(1, min(cpu, 16))
    inter_threads = 1

    # Tiers match the real file sizes: ViT-L/14 is ~293 MB on disk, the B models
    # are ~85 MB. The old 300 MB cutoff sat just above the real L/14 size, which
    # silently gave the biggest model the middle tier.
    if model_size_mb > 250:
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

    model, model_name = _recommend(ram, cpu)

    # One image costs a 512-d float32 vector twice (SQLite blob + HNSW graph)
    # plus row/graph overhead — see system_info.per_image_bytes(). The old
    # 2 KB/image figure was below the raw vector size itself.
    from system_info import per_image_bytes
    per_kb = per_image_bytes() / 1024
    max_images = int(disk * 0.8 / per_kb) if per_kb else 0

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
_OVERRIDE_FILE = _PROJECT_DIR / "tune.json"
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


def _save_overrides(values: dict) -> None:
    tmp = _OVERRIDE_FILE.with_name(_OVERRIDE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(values, indent=2), encoding="utf-8")
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
        # Persist first: if the write fails the live process keeps the values
        # the next restart would have had, instead of silently diverging.
        merged = dict(_overrides)
        merged.update(norm)
        _save_overrides(merged)
        _overrides = merged
    return dict(_overrides)


def reset_tune() -> None:
    """Drop every manual override (and tune.json) — back to full auto."""
    global _overrides
    with _tune_mu:
        _overrides = {}
        try:
            _OVERRIDE_FILE.unlink(missing_ok=True)
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
    return a.get("intra_threads", 4), a.get("inter_threads", 1)


def format_auto_tune(model_size_mb: float = 0) -> str:
    # Reporting only: never mutate the live settings just because someone asked
    # for the report (apply_tune() is the caller's job).
    r = _applied or apply_tune(model_size_mb)
    s = r["specs"]
    manual = set(r.get("_manual", ()))
    limited = " · container limit" if s.get("cgroup_limited") else ""

    def mark(k: str) -> str:
        return " ✋" if k in manual else ""

    text = (
        "⚙️ *Tune Report*\n\n"
        "*Hardware Detected:*\n"
        f"  🖥 CPU: {s['cpu_physical']} cores / "
        f"{s['cpu_logical']} threads{limited}\n"
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
