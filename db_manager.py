from __future__ import annotations
import io
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path

from config import config
from database import VectorDB

log = logging.getLogger(__name__)
TELEGRAM_FILE_LIMIT = 49 * 1024 * 1024


def get_db_size_mb() -> float:
    chroma_path = Path(config.CHROMA_DIR)
    if not chroma_path.exists():
        return 0
    total = 0
    for f in chroma_path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1024 / 1024


def export_db() -> tuple[bytes, dict]:
    chroma_path = Path(config.CHROMA_DIR)
    if not chroma_path.exists():
        raise RuntimeError("No database found.")
    buf = io.BytesIO()
    file_count = 0
    total_size = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(chroma_path.rglob("*")):
            if f.is_file():
                arcname = str(f.relative_to(chroma_path.parent))
                zf.write(f, arcname)
                file_count += 1
                total_size += f.stat().st_size
    zip_bytes = buf.getvalue()
    if len(zip_bytes) > TELEGRAM_FILE_LIMIT:
        raise RuntimeError(
            f"Database too large for Telegram "
            f"({len(zip_bytes) // 1024 // 1024} MB). "
            f"Use: tar czf backup.tar.gz chroma_db/"
        )
    return zip_bytes, {
        "files": file_count,
        "raw_size_mb": round(total_size / 1024 / 1024, 1),
        "zip_size_mb": round(len(zip_bytes) / 1024 / 1024, 1),
    }


def import_db(zip_bytes: bytes) -> dict:
    chroma_path = Path(config.CHROMA_DIR)
    backup_path = Path(config.CHROMA_DIR + f"_backup_{int(time.time())}")
    try:
        buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            if not names:
                raise RuntimeError("Empty zip file.")
            if not any("chroma" in n.lower() for n in names):
                raise RuntimeError("Not a ChromaDB backup.")
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid zip file.")
    if chroma_path.exists():
        shutil.move(str(chroma_path), str(backup_path))
    try:
        buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            zf.extractall(chroma_path.parent)
        db = VectorDB()
        count = db.count()
        channels = db.list_channels()
        if backup_path.exists():
            shutil.rmtree(str(backup_path))
        return {"vectors": count, "channels": len(channels), "success": True}
    except Exception as exc:
        if chroma_path.exists():
            shutil.rmtree(str(chroma_path))
        if backup_path.exists():
            shutil.move(str(backup_path), str(chroma_path))
        raise RuntimeError(f"Import failed (backup restored): {exc}")


class DuplicateFinder:
    def __init__(self, db: VectorDB) -> None:
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
