#!/usr/bin/env python3
"""Download a CLIP vision model into the project directory.

Usage:
    python download_model.py                  # smallest model (vit-b-32)
    python download_model.py vit-l-14-336     # any key from the registry
    python download_model.py --list           # show the catalogue
    python download_model.py --verify [key]   # check the file already on disk

The download is resumable and verified (exact size + sha256 from the registry)
before it replaces the installed model, so an interrupted or proxied transfer
can never leave a broken `clip_visual.onnx` behind.
"""
from __future__ import annotations

import argparse
import sys

from models import (AVAILABLE, ONNX_PATH, download_model, get_active_model,
                    is_model_downloaded, model_file_size_mb, set_active_model,
                    verify_model)

DEFAULT_KEY = "vit-b-32"


def _progress(done: int, total: int) -> None:
    mb = done / 1024 / 1024
    if total:
        pct = 100 * done / total
        sys.stdout.write(f"\r  {mb:.0f} / {total / 1024 / 1024:.0f} MB "
                         f"({pct:.0f}%)")
    else:
        sys.stdout.write(f"\r  {mb:.0f} MB")
    sys.stdout.flush()


def _list() -> int:
    active = get_active_model()
    print("Models (key — size / min RAM / input):")
    for key, info in AVAILABLE.items():
        tag = "  👈 active" if key == active else ""
        res = info.get("input", 224)
        print(f"  {key:<14} {info['size_mb']:>6} MB  "
              f"{info['ram_mb']:>5} MB RAM  {res}px{tag}")
    if is_model_downloaded():
        print(f"\nOn disk: {ONNX_PATH.name} "
              f"({model_file_size_mb():.0f} MB)")
    else:
        print("\nNo model on disk yet.")
    return 0


def _verify(key: str | None) -> int:
    check = verify_model(key=key)
    print(f"{check['path']}")
    print(f"  size:   {check['size_bytes']} bytes "
          f"(expected {check['expected_bytes'] or 'unknown'})")
    if check["sha256"]:
        print(f"  sha256: {check['sha256']}")
    print(f"  {'✅ ' if check['ok'] else '❌ '}{check['reason']}")
    return 0 if check["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download a CLIP model.")
    p.add_argument("key", nargs="?", default=None,
                   help=f"model key (default: {DEFAULT_KEY})")
    p.add_argument("--list", action="store_true", help="list available models")
    p.add_argument("--verify", action="store_true",
                   help="verify the installed file instead of downloading")
    args = p.parse_args(argv)

    if args.list:
        return _list()
    key = args.key or DEFAULT_KEY
    if key not in AVAILABLE:
        print(f"❌ Unknown model '{key}'. Try --list.")
        return 1
    if args.verify:
        return _verify(key)

    if is_model_downloaded():
        check = verify_model(key=key)
        if check["ok"]:
            print(f"✅ Exists and verified: {ONNX_PATH} "
                  f"({check['size_bytes'] // 1024 // 1024} MB)")
            if get_active_model() != key:
                set_active_model(key)
                print(f"ℹ️  Recorded '{key}' as the active model.")
            return 0
        print(f"⚠️  Existing {ONNX_PATH.name} failed verification "
              f"({check['reason']}) — downloading a clean copy.")

    print(f"Downloading {AVAILABLE[key]['name']} → {ONNX_PATH}")
    try:
        msg = download_model(key, progress=_progress)
    except Exception as exc:
        print(f"\n❌ {exc}")
        return 1
    print(f"\n{msg}")
    print("ℹ️  Switching models invalidates existing vectors — run /index "
          "again for reliable results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
