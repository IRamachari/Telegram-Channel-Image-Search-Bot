from __future__ import annotations
import io
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from config import config
from database import VectorDB, _DIR_LOCK, _SWAP_LOCK, _bump_dir_epoch

log = logging.getLogger(__name__)

PART_SIZE = 49 * 1024 * 1024  # 49 MB per part (Telegram limit ~50MB)

MAX_MEMBERS = 200_000
MAX_TOTAL_BYTES = 16 * 1024 ** 3
MAX_FILE_BYTES = 8 * 1024 ** 3
_COPY_CHUNK = 1 << 20

# Part names produced by export_db(): backup_<label>_<YYYYmmdd>_<HHMMSS>.partNNofMM
_PART_RE = re.compile(r"^backup_.+_(\d{8})_(\d{6})\.part(\d+)of(\d+)$", re.I)


def _l2_normalize(a: "np.ndarray") -> "np.ndarray":
    norms = np.sqrt(np.einsum("id,id->i", a, a, optimize=False))
    inv = np.zeros_like(norms, dtype=np.float32)
    np.divide(1.0, norms, out=inv, where=norms > 0)
    a *= inv[:, None].astype(np.float32)
    return a


def _safe_filename(text: str) -> str:
    return re.sub(r'[^\w\-]', '_', text)[:40]


def _allowed_prefix() -> str:
    return Path(config.CHROMA_DIR).name.rstrip("/") + "/"


_STORE_ROOT_FILES = {
    "vectors.sqlite3", "vectors.sqlite3-wal", "vectors.sqlite3-shm",
    "hnsw.bin", "hnsw.bin.tmp", "hnsw.rev",
    "indexing_state.json", "model_fingerprint.json", "bot_state.json",
}


def _archive_layout(infos) -> str:
    prefix = _allowed_prefix()
    nested = [i for i in infos if i.filename.replace("\\", "/").startswith(prefix)]
    if not nested:
        return "flat"
    return "nested" if len(nested) == len(infos) else "malformed"


def _validate_zip_members(zf: zipfile.ZipFile):
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS:
        raise RuntimeError("Archive contains too many files.")
    total = sum(i.file_size for i in infos)
    if total > MAX_TOTAL_BYTES:
        raise RuntimeError("Archive is too large once unpacked.")

    layout = _archive_layout(infos)
    if layout == "malformed":
        raise RuntimeError(
            "Archive mixes prefixed and unprefixed entries — "
            "it is corrupt or has been modified.")
    prefix = _allowed_prefix()
    for info in infos:
        name = info.filename.replace("\\", "/")
        parts = [p for p in name.split("/") if p]
        if not parts:
            raise RuntimeError("Archive contains an empty entry name.")
        if name.startswith("/") or ":" in parts[0]:
            raise RuntimeError(f"Unsafe path in archive: {info.filename}")
        if any(p == ".." for p in parts):
            raise RuntimeError(f"Unsafe path in archive: {info.filename}")
        if info.file_size > MAX_FILE_BYTES:
            raise RuntimeError(f"Archive member too large: {info.filename}")

        if layout == "nested":
            if not name.startswith(prefix):
                raise RuntimeError(f"Unexpected archive entry: {info.filename}")
        else:
            if len(parts) > 4:
                raise RuntimeError(f"Unexpected archive entry: {info.filename}")
            if len(parts) == 1 and parts[0] not in _STORE_ROOT_FILES:
                raise RuntimeError(f"Unexpected archive entry: {info.filename}")
    return infos, layout


def _safe_extract(zf: zipfile.ZipFile, dest_root: Path) -> int:
    infos, layout = _validate_zip_members(zf)
    if layout == "flat":
        dest_root = Path(config.CHROMA_DIR)
    dest_root = dest_root.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    written = 0
    # Free-space check up front.
    try:
        free = shutil.disk_usage(str(dest_root)).free
        if free < MAX_TOTAL_BYTES // 4:
            log.warning("Low free disk space (%d bytes) during extract", free)
    except Exception:
        pass
    for info in infos:
        name = info.filename.replace("\\", "/")
        dest = (dest_root / name).resolve()
        try:
            dest.relative_to(dest_root)
        except ValueError:
            raise RuntimeError(f"Archive entry escapes the database directory: {name}")
        if info.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info, "r") as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(_COPY_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_TOTAL_BYTES:
                    raise RuntimeError("Archive exceeded the size budget while unpacking.")
                out.write(chunk)
    return written


def _get_channel_label(db: VectorDB) -> str:
    channels = db.list_channels()
    if not channels:
        return "empty"
    total = db.count()
    names = set()
    if total:
        ids, metas = db.paged_get(include=["metadatas"])
        for m in metas:
            n = m.get("channel_name", "") if isinstance(m, dict) else ""
            if n:
                names.add(n)
    if not names:
        label = "channels_" + "_".join(str(k) for k in channels.keys())
    else:
        label = "_".join(_safe_filename(n) for n in sorted(names))
    if len(label) > 100:
        label = label[:100] + "_x%dch" % len(names)
    return label


def get_db_size_mb() -> float:
    p = Path(config.CHROMA_DIR)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def _create_zip(db: VectorDB) -> bytes:
    chroma = Path(config.CHROMA_DIR)
    if not chroma.exists():
        raise RuntimeError("No database found.")
    if not db.checkpoint_wal():
        log.warning("Export: WAL checkpoint was blocked; the archive may be "
                    "missing the newest transactions. Retry when indexing is idle.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(chroma.rglob("*")):
            if not f.is_file():
                continue
            if f.name.endswith(("-wal", "-shm", ".tmp")) or ".tmp" in f.name:
                continue
            try:
                zf.write(f, str(f.relative_to(chroma.parent)))
            except FileNotFoundError:
                continue
            except OSError as exc:
                log.debug("Backup skip %s: %s", f, exc)
    return buf.getvalue()


def export_db_single(db: VectorDB) -> tuple[bytes, str, dict]:
    """
    Build ONE zip holding the whole store (the /export single-file format).

    Returns (zip_bytes, filename, stats). Callers decide how to deliver the
    bytes: Bot API up to ~50 MB, MTProto beyond, split_backup() as the
    last-resort fallback.
    """
    zip_bytes = _create_zip(db)
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
    return zip_bytes, f"backup_{label}_{ts}.zip", stats


def split_backup(data: bytes, name: str, stats: dict) -> list[tuple[bytes, str, dict]]:
    """
    Split a complete single-file backup into the legacy multi-part chunks
    (backup_..._ts.partNNofMM). Small files stay whole. Filenames must keep
    matching the /import regex so old parts remain importable.
    """
    if len(data) <= PART_SIZE:
        return [(data, name, stats)]
    num = (len(data) + PART_SIZE - 1) // PART_SIZE
    stats = dict(stats)
    stats["parts"] = num
    base = name[:-4] if name.endswith(".zip") else name
    return [
        (data[i * PART_SIZE:(i + 1) * PART_SIZE],
         f"{base}.part{i + 1:02d}of{num:02d}", stats)
        for i in range(num)
    ]


def export_db(db: VectorDB) -> list[tuple[bytes, str, dict]]:
    """Legacy multi-part export (kept for compatibility with the /import
    parts path): single file when small, chunks when large."""
    return split_backup(*export_db_single(db))


def import_db(zip_bytes: bytes) -> dict:
    chroma = Path(config.CHROMA_DIR)

    stale_state: dict = {}
    db_snap = None
    try:
        db_snap = VectorDB()
        stale_state = dict(db_snap.list_channels())
    except Exception:
        stale_state = {}
    finally:
        if db_snap is not None:
            try:
                db_snap.close(wait=True)
            except Exception:
                pass

    try:
        buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            if not names:
                raise RuntimeError("Empty zip.")
            basenames = {n.replace("\\", "/").rsplit("/", 1)[-1]
                         for n in names}
            if "chroma.sqlite3" in basenames:
                raise RuntimeError(
                    "This backup uses the old ChromaDB format. The bot now "
                    "stores vectors natively (vectors.sqlite3); since every "
                    "old vector also predates the current model fingerprint, "
                    "run /index to rebuild instead of importing.")
            if "vectors.sqlite3" not in basenames:
                raise RuntimeError("Not a database backup.")
            _validate_zip_members(zf)
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid zip file.")

    backup = Path(config.CHROMA_DIR + f"_bak_{int(time.time()*1000)}_{os.getpid()}")

    with _DIR_LOCK:
        # Invalidate retired writers before touching the directory.
        with _SWAP_LOCK:
            _bump_dir_epoch()
        if chroma.exists():
            shutil.move(str(chroma), str(backup))

        db = None
        try:
            buf = io.BytesIO(zip_bytes)
            with zipfile.ZipFile(buf, "r") as zf:
                _safe_extract(zf, chroma.parent)
            db = VectorDB()
            if not db.checkpoint_wal():
                log.warning("Import: WAL checkpoint busy; proceeding anyway.")
            count = db.count()
            if count == 0:
                raise RuntimeError("Imported backup contains no vectors.")
            # integrity check
            try:
                _con = __import__("sqlite3").connect(str(Path(config.CHROMA_DIR) / "vectors.sqlite3"))
                try:
                    _con.execute("PRAGMA busy_timeout=30000;")
                    ic = _con.execute("PRAGMA integrity_check;").fetchone()
                    if ic and ic[0] != "ok":
                        raise RuntimeError(f"Imported database failed integrity check: {ic[0]}")
                finally:
                    _con.close()
            except RuntimeError:
                raise
            except Exception as exc:
                log.warning("Import integrity check skipped: %s", exc)
            channels = db.list_channels()
            unknown_model = False
            if count > 0:
                try:
                    import models as _models
                    if _models.read_model_fingerprint() is None:
                        unknown_model = True
                        log.warning(
                            "Imported %d vectors with no model fingerprint: "
                            "they may have been indexed with a different "
                            "CLIP model or preprocessing version.", count)
                except Exception:
                    pass
            restored = _merge_watermarks_forward(db, stale_state)
            if backup.exists():
                shutil.rmtree(str(backup), ignore_errors=True)
            return {"vectors": count, "channels": len(channels),
                    "success": True, "watermarks_advanced": restored,
                    "unknown_model": unknown_model}
        except Exception as exc:
            if chroma.exists():
                shutil.rmtree(str(chroma), ignore_errors=True)
            if backup.exists():
                try:
                    os.replace(str(backup), str(chroma))
                except OSError:
                    log.error("Could not restore backup %s: %s", backup, exc)
            raise RuntimeError(f"Failed (restored backup): {exc}")
        finally:
            if db is not None:
                try:
                    db.close(wait=True)
                except Exception:
                    pass


def _merge_watermarks_forward(db: "VectorDB", stale_state: dict) -> int:
    if not stale_state:
        return 0
    restored = 0
    try:
        current = db.list_channels()
        for ch, old in stale_state.items():
            cur = current.get(ch, 0)
            if isinstance(old, int) and old > cur:
                db.set_watermark(ch, old)
                restored += 1
        # Flush so the merged watermarks actually land on disk before import
        # returns (the writer is asynchronous).
        try:
            db.flush_state(timeout=10.0)
        except Exception:
            pass
        return restored
    except Exception as exc:
        log.warning("Watermark merge after import failed: %s", exc)
        return 0


def import_db_parts(parts) -> dict:
    if not parts:
        raise RuntimeError("No parts received.")

    normalised = []
    for item in parts:
        if isinstance(item, (tuple, list)):
            data, name = item[0], item[1]
        else:
            data, name = item, ""
        normalised.append((bytes(data), str(name or "")))

    parsed = []
    total_expected = None
    stamp = None
    for data, name in normalised:
        m = _PART_RE.match(name.strip())
        if not m:
            raise RuntimeError(f"Not a valid backup part: {name or '<unnamed>'}")
        ts = m.group(1) + m.group(2)
        idx, num = int(m.group(3)), int(m.group(4))
        if num < 1 or idx < 1 or idx > num:
            raise RuntimeError(f"Invalid part numbering: {name!r}")
        if total_expected is None:
            total_expected = num
            stamp = ts
        elif num != total_expected or ts != stamp:
            raise RuntimeError("Parts belong to different backups — send one set at a time.")
        parsed.append((idx, data))

    if len({i for i, _ in parsed}) != len(parsed):
        raise RuntimeError("Duplicate part numbers received.")
    if len(parsed) != total_expected:
        raise RuntimeError(f"Missing parts: got {len(parsed)} of {total_expected}.")

    ordered = sorted(parsed, key=lambda t: t[0])
    if [i for i, _ in ordered] != list(range(1, total_expected + 1)):
        raise RuntimeError("Parts are out of order — send all parts in order, then 'done'.")

    # Geometry check: sizes must be <= PART_SIZE, last part > 0.
    for idx, data in ordered[:-1]:
        if len(data) > PART_SIZE or len(data) == 0:
            raise RuntimeError(
                f"Part {idx} has invalid size ({len(data)} bytes).")
    last = ordered[-1]
    if not 0 < len(last[1]) <= PART_SIZE:
        raise RuntimeError(f"Part {total_expected} has an invalid size "
                           f"({len(last[1])} bytes).")

    # Stream reassembly to a temp file to avoid doubling peak RAM.
    fd, tmp_path = tempfile.mkstemp(prefix="backup_", suffix=".zip")
    try:
        with os.fdopen(fd, "wb") as out:
            for _, data in ordered:
                out.write(data)
        with open(tmp_path, "rb") as fh:
            combined = fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return import_db(combined)


class DuplicateFinder:
    def __init__(self, db):
        self.db = db

    @staticmethod
    def _day(date_str):
        return (date_str or '')[:10]

    def find_all(self, threshold=0.95, page_size=100, progress_callback=None):
        total = self.db.count()
        if total < 2:
            return []

        ids: list = []
        metadatas: list = []
        emb_chunks: list = []
        for page_ids, page_metas, page_embs in self.db.iter_vectors(page=page_size or 1000):
            ids.extend(page_ids)
            metadatas.extend(page_metas)
            emb_chunks.append(page_embs)
        if len(ids) < 2 or not emb_chunks:
            return []
        # Pre-allocate the matrix to avoid repeated vstack copies.
        n = len(ids)
        dim = emb_chunks[0].shape[1]
        embeddings = np.empty((n, dim), dtype=np.float32)
        off = 0
        for chunk in emb_chunks:
            embeddings[off:off + chunk.shape[0]] = chunk
            off += chunk.shape[0]
        del emb_chunks
        embeddings = _l2_normalize(embeddings)

        group_seen = set()
        match_seen = set()
        groups = []

        block = max(1, min(512, len(ids)))
        for base in range(0, len(ids), block):
            sims = embeddings[base:base + block] @ embeddings.T
            for r in range(sims.shape[0]):
                i = base + r
                src_id = ids[i]
                if src_id in group_seen or src_id in match_seen:
                    continue
                row = sims[r]

                above = np.nonzero(row >= threshold)[0]
                if above.size == 0:
                    continue
                above = above[np.argsort(row[above])[::-1]]

                src_meta = metadatas[i]
                src_day = self._day(src_meta.get('date', ''))

                matches = []
                for j in above:
                    mid = ids[int(j)]
                    if mid == src_id or mid in group_seen or mid in match_seen:
                        continue
                    match_day = self._day((metadatas[int(j)] or {}).get('date', ''))
                    if match_day == src_day:
                        continue
                    matches.append({
                        'id': mid,
                        'similarity': round(float(row[int(j)]), 4),
                        'channel_name': (metadatas[int(j)] or {}).get('channel_name', ''),
                        'message_id': (metadatas[int(j)] or {}).get('message_id', 0),
                        'link': (metadatas[int(j)] or {}).get('link', ''),
                        'date': (metadatas[int(j)] or {}).get('date', ''),
                        'private': (metadatas[int(j)] or {}).get('private', False),
                    })

                if matches:
                    group_seen.add(src_id)
                    for m in matches:
                        match_seen.add(m['id'])
                    matches.sort(key=lambda x: x['date'])
                    groups.append({
                        'source': {
                            'id': src_id,
                            'channel_name': src_meta.get('channel_name', ''),
                            'message_id': src_meta.get('message_id', 0),
                            'link': src_meta.get('link', ''),
                            'date': src_meta.get('date', ''),
                            'private': src_meta.get('private', False),
                        },
                        'matches': matches,
                        'count': len(matches),
                    })

            del sims
            if progress_callback:
                progress_callback(min(base + block, len(ids)), len(ids), len(groups))

        if progress_callback:
            progress_callback(len(ids), len(ids), len(groups))

        groups.sort(key=lambda g: g['count'], reverse=True)
        return groups
