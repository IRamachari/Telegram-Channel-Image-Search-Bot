from __future__ import annotations
import io
import logging
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

from config import config
from database import VectorDB

log = logging.getLogger(__name__)

PART_SIZE = 49 * 1024 * 1024  # 49 MB per part (Telegram limit ~50MB)


def _safe_filename(text: str) -> str:
    return re.sub(r'[^\w\-]', '_', text)[:40]


def _get_channel_label(db: VectorDB) -> str:
    channels = db.list_channels()
    if not channels:
        return "empty"
    total = db.count()
    all_data = db.col.get(limit=min(total, 1), include=["metadatas"])
    names = set()
    if all_data["metadatas"]:
        # get unique channel names from metadata
        full = db.col.get(limit=total, include=["metadatas"])
        for m in full["metadatas"]:
            n = m.get("channel_name", "")
            if n:
                names.add(n)
    if not names:
        return "channels_" + "_".join(channels.keys())
    return "_".join(_safe_filename(n) for n in sorted(names))


def get_db_size_mb() -> float:
    p = Path(config.CHROMA_DIR)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def _create_zip() -> bytes:
    chroma = Path(config.CHROMA_DIR)
    if not chroma.exists():
        raise RuntimeError("No database found.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(chroma.rglob("*")):
            if f.is_file():
                zf.write(f, str(f.relative_to(chroma.parent)))
    return buf.getvalue()


def export_db(db: VectorDB) -> list[tuple[bytes, str, dict]]:
    """
    Export database as zip.
    Returns list of (bytes, filename, stats).
    Single item if <49MB, multiple parts if larger.
    """
    zip_bytes = _create_zip()
    label = _get_channel_label(db)
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_mb = get_db_size_mb()
    zip_mb = len(zip_bytes) / 1024 / 1024
    vectors = db.count()

    stats = {
        "vectors": vectors,
        "raw_mb": round(raw_mb, 1),
        "zip_mb": round(zip_mb, 1),
        "channels": label,
    }

    if len(zip_bytes) <= PART_SIZE:
        name = f"backup_{label}_{ts}.zip"
        return [(zip_bytes, name, stats)]

    # split into parts
    parts = []
    total = len(zip_bytes)
    num_parts = (total + PART_SIZE - 1) // PART_SIZE
    stats["parts"] = num_parts

    for i in range(num_parts):
        start = i * PART_SIZE
        chunk = zip_bytes[start:start + PART_SIZE]
        pname = f"backup_{label}_{ts}.part{i + 1:02d}of{num_parts:02d}"
        parts.append((chunk, pname, stats))

    return parts


def import_db(zip_bytes: bytes) -> dict:
    """Import from single zip."""
    chroma = Path(config.CHROMA_DIR)
    backup = Path(config.CHROMA_DIR + f"_bak_{int(time.time())}")

    try:
        buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            if not names:
                raise RuntimeError("Empty zip.")
            if not any("chroma" in n.lower() for n in names):
                raise RuntimeError("Not a ChromaDB backup.")
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid zip file.")

    if chroma.exists():
        shutil.move(str(chroma), str(backup))

    try:
        buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(chroma.parent)
        db = VectorDB()
        count = db.count()
        channels = db.list_channels()
        if backup.exists():
            shutil.rmtree(str(backup))
        return {"vectors": count, "channels": len(channels), "success": True}
    except Exception as exc:
        if chroma.exists():
            shutil.rmtree(str(chroma))
        if backup.exists():
            shutil.move(str(backup), str(chroma))
        raise RuntimeError(f"Failed (restored backup): {exc}")


def import_db_parts(parts: list[bytes]) -> dict:
    """Import from multiple parts (reassemble then import)."""
    combined = b"".join(parts)
    return import_db(combined)


class DuplicateFinder:
    def __init__(self, db: VectorDB):
        self.db = db

    def find_all(self, threshold=0.95, page_size=100, progress_callback=None):
        col = self.db.col
        total = col.count()
        if total < 2:
            return []
        all_data = col.get(include=["embeddings", "metadatas"], limit=total)
        ids = all_data["ids"]
        embeddings = all_data["embeddings"]
        metadatas = all_data["metadatas"]
        if not ids or not embeddings:
            return []
        seen = set()
        groups = []
        for i in range(len(ids)):
            if ids[i] in seen:
                continue
            results = col.query(
                query_embeddings=[embeddings[i]],
                n_results=min(20, total),
                include=["metadatas", "distances"],
            )
            matches = []
            for j in range(len(results["ids"][0])):
                mid = results["ids"][0][j]
                if mid == ids[i] or mid in seen:
                    continue
                sim = 1.0 - results["distances"][0][j]
                if sim >= threshold:
                    meta = results["metadatas"][0][j]
                    matches.append({
                        "id": mid,
                        "similarity": round(sim, 4),
                        "channel_name": meta.get("channel_name", ""),
                        "message_id": meta.get("message_id", 0),
                        "link": meta.get("link", ""),
                        "date": meta.get("date", ""),
                        "private": meta.get("private", False),
                    })
            if matches:
                seen.add(ids[i])
                for m in matches:
                    seen.add(m["id"])
                src = metadatas[i]
                groups.append({
                    "source": {
                        "id": ids[i],
                        "channel_name": src.get("channel_name", ""),
                        "message_id": src.get("message_id", 0),
                        "link": src.get("link", ""),
                        "date": src.get("date", ""),
                        "private": src.get("private", False),
                    },
                    "matches": sorted(matches, key=lambda x: x["similarity"], reverse=True),
                    "count": len(matches),
                })
            if progress_callback and (i + 1) % page_size == 0:
                progress_callback(i + 1, total, len(groups))
        if progress_callback:
            progress_callback(len(ids), total, len(groups))
        groups.sort(key=lambda g: g["count"], reverse=True)
        return groups
