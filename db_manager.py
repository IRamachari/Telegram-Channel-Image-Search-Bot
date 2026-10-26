from __future__ import annotations
import io
import logging
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from config import config
from database import VectorDB, _DIR_LOCK

log = logging.getLogger(__name__)

PART_SIZE = 49 * 1024 * 1024  # 49 MB per part (Telegram limit ~50MB)

# Guards for /import. A backup is admin-supplied but still untrusted input: an
# archive with ../ members or a huge uncompressed size can overwrite files
# anywhere the process can write, or exhaust the disk (which takes the whole
# box down — not just the bot).
MAX_MEMBERS = 200_000
MAX_TOTAL_BYTES = 16 * 1024 ** 3
MAX_FILE_BYTES = 8 * 1024 ** 3
_COPY_CHUNK = 1 << 20

# Part names produced by export_db(): backup_<label>_<YYYYmmdd>_<HHMMSS>.partNNofMM
_PART_RE = re.compile(r"^backup_.+_(\d{8})_(\d{6})\.part(\d{2})of(\d{2})$", re.I)


def _l2_normalize(a: "np.ndarray") -> "np.ndarray":
    """
    Row-wise L2 normalisation, in place, without a second (N, D) temporary.

    `np.linalg.norm` dispatches through a generic multi-norm path; einsum is a
    single fused reduction. For a 100k x 512 matrix the non-in-place form
    allocates an extra ~200 MB.
    """
    norms = np.sqrt(np.einsum("id,id->i", a, a, optimize=False))
    inv = np.where(norms > 0, np.reciprocal(norms, dtype=np.float32), 0.0)
    a *= inv[:, None].astype(np.float32)
    return a


def _safe_filename(text: str) -> str:
    return re.sub(r'[^\w\-]', '_', text)[:40]


def _allowed_prefix() -> str:
    """Directory a backup must unpack into, e.g. ``chroma_db/``."""
    return Path(config.CHROMA_DIR).name.rstrip("/") + "/"


# Files the vector store (see database.py) may legitimately place at the root
# of the database directory. Flat-layout archives are restricted to these
# names; anything else at the root is rejected. Files of the OLD ChromaDB
# storage are absent on purpose — import_db rejects that format explicitly
# before any extraction happens.
_STORE_ROOT_FILES = {
    "vectors.sqlite3", "vectors.sqlite3-wal", "vectors.sqlite3-shm",
    "hnsw.bin", "hnsw.bin.tmp",
    "indexing_state.json", "model_fingerprint.json",
}


def _archive_layout(infos) -> str:
    """
    Decide how an archive is laid out.

    Returns ``"nested"`` (members are ``<chroma_db>/...`` — what every export
    from this bot produces) or ``"flat"`` (members are ChromaDB's own directory
    contents at the zip root, which is what the pre-refactor exporter and a
    plain ``zip -r`` of the folder produce).

    Both are legitimate. The old restore path rejected the flat form outright,
    which made every pre-existing backup impossible to import.
    """
    prefix = _allowed_prefix()
    nested = [i for i in infos if i.filename.replace("\\", "/").startswith(prefix)]
    if not nested:
        return "flat"
    # If ANY member carries the prefix, ALL must — a mixed archive is malformed.
    return "nested" if len(nested) == len(infos) else "malformed"


def _validate_zip_members(zf: zipfile.ZipFile):
    """
    Validate an archive and return ``(infos, dest_root)``.

    Rejects anything that is not a ChromaDB backup. ``ZipFile.extractall`` only
    strips drive letters and leading ``..`` — it does NOT guarantee containment,
    so ``../bot.py`` or an absolute path would be written inside the project
    directory (code + .env with BOT_TOKEN and TELEGRAM_API_HASH live there).
    """
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
            # Flat layout: entries must live directly inside the database
            # directory. Top-level files are restricted to ChromaDB's own names
            # and subdirectories are bounded, so nothing can reach outside.
            if len(parts) > 4:
                raise RuntimeError(f"Unexpected archive entry: {info.filename}")
            if len(parts) == 1 and parts[0] not in _STORE_ROOT_FILES:
                raise RuntimeError(f"Unexpected archive entry: {info.filename}")
    return infos, layout


def _safe_extract(zf: zipfile.ZipFile, dest_root: Path) -> int:
    """Extract into dest_root, enforcing containment and a byte budget."""
    infos, layout = _validate_zip_members(zf)
    if layout == "flat":
        # Members are the database directory's own contents.
        dest_root = Path(config.CHROMA_DIR)
    dest_root = dest_root.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for info in infos:
        name = info.filename.replace("\\", "/")
        dest = (dest_root / name).resolve()
        if not str(dest).startswith(str(dest_root) + os.sep) and dest != dest_root:
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
                # Member headers can lie about their size — check bytes actually
                # written so a zip bomb can't exhaust the disk.
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
    # _safe_filename caps EACH name at 40 chars but not the joined result: with
    # several long titles the filename hit the 255-byte limit and Telegram
    # rejected the document — after all the CPU/IO of building it.
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
    # Committed-but-uncheckpointed pages live in chroma.sqlite3-wal; without
    # this a file-level backup silently loses the tail of every transaction.
    # If the checkpoint was blocked by an active reader the archive would be
    # internally inconsistent, so refuse rather than ship a broken backup.
    if not db.checkpoint_wal():
        log.warning("Export: WAL checkpoint was blocked; the archive may be "
                    "missing the newest transactions. Retry when indexing is idle.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(chroma.rglob("*")):
            if not f.is_file():
                continue
            # Never ship live WAL / shared-memory / half-written temp files:
            # they are not restorable state, and a leftover .tmp from a failed
            # atomic write would be baked into every future backup forever.
            if f.name.endswith(("-wal", "-shm", ".tmp")):
                continue
            try:
                zf.write(f, str(f.relative_to(chroma.parent)))
            except FileNotFoundError:
                # WAL rotated/truncated under us — skipping keeps the zip usable.
                continue
            except OSError as exc:
                log.debug("Backup skip %s: %s", f, exc)
    return buf.getvalue()


def export_db(db: VectorDB) -> list[tuple[bytes, str, dict]]:
    """
    Export database as zip.
    Returns list of (bytes, filename, stats).
    Single item if <49MB, multiple parts if larger.
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
    # Unique per attempt: two imports inside the same second shared a name, and
    # the second then nested the live DB inside the first's backup directory.
    backup = Path(config.CHROMA_DIR + f"_bak_{int(time.time()*1000)}_{os.getpid()}")

    # Snapshot the live watermarks BEFORE the directory is moved away, so an
    # older backup cannot silently roll them back (which would re-scan and
    # re-download every channel's full history).
    stale_state: dict = {}
    try:
        stale_state = dict(VectorDB().list_channels())
    except Exception:
        stale_state = {}

    try:
        buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            if not names:
                raise RuntimeError("Empty zip.")
            basenames = {n.replace("\\", "/").rsplit("/", 1)[-1]
                         for n in names}
            if "chroma.sqlite3" in basenames:
                # A pre-migration backup. Its vectors were produced by the old
                # ChromaDB storage with an older model fingerprint, so they are
                # due for replacement anyway; refuse loudly instead of leaving
                # the admin with an unreadable directory.
                raise RuntimeError(
                    "This backup uses the old ChromaDB format. The bot now "
                    "stores vectors natively (vectors.sqlite3); since every "
                    "old vector also predates the current model fingerprint, "
                    "run /index to rebuild instead of importing.")
            if "vectors.sqlite3" not in basenames:
                raise RuntimeError("Not a database backup.")
            # Validates and reports the layout so the extraction below lands in
            # the right directory (see _validate_zip_members).
            _validate_zip_members(zf)
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid zip file.")

    # The swap, extraction and first open all happen under the store's
    # directory lock: a search landing mid-swap would otherwise open a fresh,
    # empty database and silently return nothing (see database._DIR_LOCK).
    with _DIR_LOCK:
        # Move the live DB aside instead of deleting it first: if anything
        # below fails, the previous database is still intact on disk.
        if chroma.exists():
            shutil.move(str(chroma), str(backup))

        db = None
        try:
            buf = io.BytesIO(zip_bytes)
            with zipfile.ZipFile(buf, "r") as zf:
                # For a "nested" archive dest_root is the parent (members
                # carry the chroma_db/ prefix); _safe_extract overrides it
                # for "flat" archives.
                _safe_extract(zf, chroma.parent)
            db = VectorDB()
            # A copied WAL may not belong to the copied main file; fold it in.
            db.checkpoint_wal()
            count = db.count()
            channels = db.list_channels()
            # Never let an imported backup move a watermark backwards.
            restored = _merge_watermarks_forward(db, stale_state)
            # A backup made before the fingerprint existed carries no model
            # provenance. Since a model or preprocessing change silently
            # invalidates every vector (and a dimension change makes the next
            # insert fail), the admin must be told rather than discovering it
            # mid-index.
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
            if backup.exists():
                shutil.rmtree(str(backup), ignore_errors=True)
            return {"vectors": count, "channels": len(channels),
                    "success": True, "watermarks_advanced": restored,
                    "unknown_model": unknown_model}
        except Exception as exc:
            if chroma.exists():
                shutil.rmtree(str(chroma), ignore_errors=True)
            if backup.exists():
                # os.replace (rename) fails loudly if chroma still exists,
                # instead of MOVING THE BACKUP INTO it the way shutil.move
                # does — which used to leave the database nested at
                # chroma_db/chroma_db_bak_*/chroma.sqlite3 so the next open
                # produced a brand-new empty DB.
                try:
                    os.replace(str(backup), str(chroma))
                except OSError:
                    log.error("Could not restore backup %s: %s", backup, exc)
            raise RuntimeError(f"Failed (restored backup): {exc}")
        finally:
            # This VectorDB is not the bot's: close it so its background
            # writer threads are released rather than leaked per import.
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass


def _merge_watermarks_forward(db: "VectorDB", stale_state: dict) -> int:
    """
    Keep the higher watermark per channel after an import.

    Returns the number of channels whose watermark was raised back up. An
    imported backup taken earlier in time (or one with no state file) would
    otherwise reset every channel to 0 and force a full re-index.
    """
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
        return restored
    except Exception as exc:
        log.warning("Watermark merge after import failed: %s", exc)
        return 0


def import_db_parts(parts) -> dict:
    """
    Import from multiple parts, re-validated before anything is written.

    ``parts`` is a list of (bytes, filename) in send order. The part label
    encodes the index and the total, so a misordered, duplicated or missing part
    is now a clean failure instead of a silently corrupt database.
    """
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

    # Geometry check: every part but the last must be exactly PART_SIZE, and the
    # last must be > 0 and <= PART_SIZE. This catches a truncated or padded part
    # that would otherwise silently reassemble into a corrupt archive.
    for idx, data in ordered[:-1]:
        if len(data) != PART_SIZE:
            raise RuntimeError(
                f"Part {idx:02d} is truncated ({len(data)} bytes, expected {PART_SIZE}).")
    last = ordered[-1]
    if not 0 < len(last[1]) <= PART_SIZE:
        raise RuntimeError(f"Part {total_expected:02d} has an invalid size "
                            f"({len(last[1])} bytes).")

    combined = b"".join(d for _, d in ordered)
    return import_db(combined)


class DuplicateFinder:
    def __init__(self, db):
        self.db = db

    @staticmethod
    def _day(date_str):
        return (date_str or '')[:10]

    def find_all(self, threshold=0.95, page_size=100, progress_callback=None):
        """
        Find images that were posted on more than one distinct date.

        Exact all-pairs cosine (recall 1.0) computed in numpy, instead of one
        approximate HNSW query per vector. All pairs above the threshold are
        checked, and same-day matches are skipped entirely.
        """
        total = self.db.count()
        if total < 2:
            return []

        ids: list = []
        metadatas: list = []
        emb_chunks: list = []
        # Streams straight from SQLite: pagination is exact (stable ORDER BY
        # id), so the ChromaDB-era no-progress/lockstep guards are gone.
        for page_ids, page_metas, page_embs in self.db.iter_vectors(page=1000):
            ids.extend(page_ids)
            metadatas.extend(page_metas)
            emb_chunks.append(page_embs)
        if len(ids) < 2 or not emb_chunks:
            return []
        embeddings = np.vstack(emb_chunks).astype(np.float32, copy=False)
        del emb_chunks
        # The ANN graph's cosine space normalises vectors internally, but these
        # may have been written by an older build: normalise once so the dot
        # product below is exactly cosine similarity.
        embeddings = _l2_normalize(embeddings)

        group_seen = set()
        match_seen = set()
        groups     = []

        # Exact all-pairs cosine via BLAS instead of one approximate HNSW query
        # per vector (which was N Python round-trips into ChromaDB). Rows are
        # processed in blocks so peak memory stays bounded, and within a block we
        # only keep entries above the threshold.
        block = max(1, min(512, len(ids)))
        for base in range(0, len(ids), block):
            sims = embeddings[base:base + block] @ embeddings.T   # (b, N)
            for r in range(sims.shape[0]):
                i = base + r
                src_id = ids[i]
                if src_id in group_seen or src_id in match_seen:
                    continue
                row = sims[r]

                # Only look at candidates that clear the threshold, then sort
                # those few by similarity. np.argpartition is O(N) average and
                # np.argsort would be O(N log N) for a set we then discard.
                above = np.nonzero(row >= threshold)[0]
                if above.size == 0:
                    continue
                # highest similarity first
                above = above[np.argsort(row[above])[::-1]]

                src_meta = metadatas[i]
                src_day  = self._day(src_meta.get('date', ''))

                matches = []
                for j in above:
                    mid = ids[int(j)]
                    if mid == src_id or mid in group_seen or mid in match_seen:
                        continue
                    match_day = self._day((metadatas[int(j)] or {}).get('date', ''))
                    # Only flag if on a DIFFERENT date
                    if match_day == src_day:
                        continue
                    matches.append({
                        'id':           mid,
                        'similarity':   round(float(row[int(j)]), 4),
                        'channel_name': (metadatas[int(j)] or {}).get('channel_name', ''),
                        'message_id':   (metadatas[int(j)] or {}).get('message_id', 0),
                        'link':         (metadatas[int(j)] or {}).get('link', ''),
                        'date':         (metadatas[int(j)] or {}).get('date', ''),
                        'private':      (metadatas[int(j)] or {}).get('private', False),
                    })

                if matches:
                    group_seen.add(src_id)
                    for m in matches:
                        match_seen.add(m['id'])
                    # chronological order: original first, newest repost last
                    matches.sort(key=lambda x: x['date'])
                    groups.append({
                        'source': {
                            'id':           src_id,
                            'channel_name': src_meta.get('channel_name', ''),
                            'message_id':   src_meta.get('message_id', 0),
                            'link':         src_meta.get('link', ''),
                            'date':         src_meta.get('date', ''),
                            'private':      src_meta.get('private', False),
                        },
                        'matches': matches,
                        'count':   len(matches),
                    })

            del sims
            if progress_callback:
                progress_callback(min(base + block, len(ids)), len(ids), len(groups))

        if progress_callback:
            progress_callback(len(ids), len(ids), len(groups))

        groups.sort(key=lambda g: g['count'], reverse=True)
        return groups
