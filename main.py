#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import faulthandler
import importlib
import logging
import sys

# Print a Python-level stack trace on fatal signals (SIGSEGV, SIGBUS, …).
# faulthandler writes straight to fd 2, so the trace survives even a hard
# native crash that would otherwise leave the log without any output.
faulthandler.enable()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
)
logging.getLogger("telethon").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.WARNING)

log = logging.getLogger("main")


def _preflight() -> None:
    """
    Import the native extensions one at a time with a stderr marker before each.

    The bot's heavy dependencies are C extensions (onnxruntime, usearch,
    numkong, cryptg, Pillow). If one of them segfaults at import, Python
    cannot raise — the process just dies — and logging setup produces no
    output yet. These flushed stderr markers, combined with faulthandler's
    stack trace, pinpoint the failing module in a single restart.
    """
    for mod in ("numpy", "PIL", "onnxruntime", "usearch", "cryptg",
                "telethon", "telegram"):
        print(f"[preflight] importing {mod} …", file=sys.stderr, flush=True)
        importlib.import_module(mod)
    print("[preflight] all imports OK", file=sys.stderr, flush=True)


def do_index(args):
    _preflight()
    from database import VectorDB
    from embeddings import CLIPEmbedder
    from indexer import ChannelIndexer
    from models import mark_vectors_indexed
    db = VectorDB()
    emb = CLIPEmbedder()
    if not emb.is_ready():
        print("❌ No model. Run: python download_model.py")
        sys.exit(1)
    n = asyncio.run(ChannelIndexer(db, emb).index_channel(args.channel, limit=args.limit))
    if n and n > 0:
        try:
            mark_vectors_indexed()
        except Exception as exc:
            log.warning("Could not record model fingerprint: %s", exc)
    print(f"✅ {n} images. Total: {db.count()}")
    try:
        db.close(wait=True)
    except Exception:
        pass


def do_bot(_):
    _preflight()
    from bot import SearchBot
    SearchBot().run()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    ix = sub.add_parser("index")
    ix.add_argument("channel")
    ix.add_argument("--limit", type=int, default=None)
    sub.add_parser("bot")
    args = p.parse_args()
    {"index": do_index, "bot": do_bot}.get(args.cmd, lambda _: p.print_help())(args)


if __name__ == "__main__":
    main()
