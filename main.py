#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
)
logging.getLogger("telethon").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.WARNING)

log = logging.getLogger("main")


def do_index(args):
    from database import VectorDB
    from embeddings import CLIPEmbedder
    from indexer import ChannelIndexer
    db = VectorDB()
    emb = CLIPEmbedder()
    if not emb.is_ready():
        print("❌ No model. Run: python download_model.py")
        sys.exit(1)
    n = asyncio.run(ChannelIndexer(db, emb).index_channel(args.channel, limit=args.limit))
    print(f"✅ {n} images. Total: {db.count()}")


def do_bot(_):
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
