from __future__ import annotations
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from usearch.index import Index, MetricKind

from config import config

log = logging.getLogger(__name__)

# Everything lives inside config.CHROMA_DIR so /export (which zips the whole
# directory) and the panel workflows keep working unchanged.
_STATE_FILE = Path(config.CHROMA_DIR) / "indexing_state.json"
_SQLITE_FILE = Path(config.CHROMA_DIR) / "vectors.sqlite3"
_HNSW_FILE = Path(config.CHROMA_DIR) / "hnsw.bin"
# Revision of the SQLite state the graph in `_HNSW_FILE` was built from. The
# graph is a derived cache and a key set alone cannot tell a fresh graph from
# one whose vectors were overwritten in place (upsert reuses keys), so the
# open-time check compares revisions instead. See `_open_hnsw`.
_REV_FILE = Path(config.CHROMA_DIR) / "hnsw.rev"

_PAGE = 1000
# A write of the ANN graph is a full-file rewrite; during a backfill batches
# land every few seconds, so saving each one would burn the disk for nothing.
# The saver waits for a quiet window instead.
_HNSW_SAVE_DEBOUNCE = 2.0

_SCHEMA = (
    # `key` is the integer HNSW key; `id` is the external (string) record id.
    "CREATE TABLE IF NOT EXISTS vectors ("
    " id TEXT PRIMARY KEY,"
    " key INTEGER NOT NULL UNIQUE,"
    " dim INTEGER NOT NULL,"
    " embedding BLOB NOT NULL,"
    " metadata TEXT NOT NULL,"
    " channel_id INTEGER)",
    "CREATE INDEX IF NOT EXISTS idx_vectors_channel ON vectors(channel_id)",
    "CREATE TABLE IF NOT EXISTS store_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)",
)

# ChromaDB's cosine space stores f32 vectors and reports distance as
# 1 - cosine_similarity; usearch's MetricKind.Cos with dtype f32 does exactly
# the same (verified to 1.2e-7), so every consumer of `distances` keeps working.

# Makes the directory swap in import_db exclusive against reads: without it, a
# search landing between the swap's move and extract phases would open a
# fresh, empty database and silently return nothing. It is an RLock so
# import_db can hold it across the whole swap while the VectorDB() it
# constructs re-enters. Every VectorDB method takes it per operation, so /find
# during an import briefly waits instead of returning wrong results.
_DIR_LOCK = threading.RLock()

# Held while import_db replaces the database directory. Background writers take
# it only around the final rename / state-file write, and re-check
# `_DIR_EPOCH`: a writer that started before the swap must not drop its file
# into the freshly extracted directory. Lock order is always
# `_DIR_LOCK` -> `_hnsw_mu` -> `_SWAP_LOCK`.
_SWAP_LOCK = threading.RLock()
_DIR_EPOCH = 0


def _bump_dir_epoch() -> int:
    """Invalidate in-flight background writes. Caller holds `_SWAP_LOCK`."""
    global _DIR_EPOCH
    _DIR_EPOCH += 1
    return _DIR_EPOCH


def read_watermarks() -> dict:
    """
    Read the watermark file without opening the store.

    ``VectorDB()`` loads the HNSW graph (rebuilding it when stale) and starts two
    threads, which is far too much work for a caller that only needs the
    per-channel watermarks — `import_db` snapshots them before a swap.
    """
    if _STATE_FILE.exists():
        try:
            data = json.loads(_STATE_FILE.read_text())
            if isinstance(data, dict):
                return data
            log.warning("Indexing state file is not a JSON object — ignoring")
        except Exception as e:
            log.warning("Indexing state load error (%s): %s", _STATE_FILE, e)
    return {}


def _serialized(fn):
    """Run a VectorDB method under `_DIR_LOCK` (see the note above)."""
    def wrapper(*args, **kwargs):
        with _DIR_LOCK:
            return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def _atomic_write_json(path: Path, obj) -> None:
    """
    Persist JSON atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except (OSError, AttributeError):
            pass
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _read_rev_file() -> "int | None":
    """Revision the on-disk graph claims to reflect; None when unknown."""
    try:
        return int(_REV_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _connect() -> sqlite3.Connection:
    """A short-lived connection. SQLite WAL + busy_timeout make this cheap."""
    con = sqlite3.connect(str(_SQLITE_FILE), timeout=30)
    con.execute("PRAGMA busy_timeout=30000;")
    return con


def _channel_id_of(meta: dict):
    try:
        v = meta.get("channel_id")
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _norm_channel_id(channel_id):
    """Canonical string form used for watermark keys."""
    try:
        return str(int(channel_id))
    except (TypeError, ValueError):
        return str(channel_id)


class VectorDB:
    """
    Vector store over SQLite (source of truth) + usearch HNSW (derived cache).
    """

    @_serialized
    def __init__(self) -> None:
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)

        try:
            con = sqlite3.connect(str(_SQLITE_FILE), timeout=30)
            try:
                for pragma in ("journal_mode=WAL", "busy_timeout=30000"):
                    try:
                        con.execute(f"PRAGMA {pragma};")
                    except sqlite3.DatabaseError as exc:
                        log.warning("SQLite pragma %s skipped: %s",
                                    pragma, exc)
                for stmt in _SCHEMA:
                    con.execute(stmt)
                con.execute(
                    "INSERT OR IGNORE INTO store_meta(k, v) "
                    "SELECT 'rev', '0'")
                con.execute(
                    "INSERT OR IGNORE INTO store_meta(k, v) "
                    "SELECT 'next_key', COALESCE(MAX(key), 0) FROM vectors")
                con.commit()
            finally:
                con.close()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot initialise vector store at {_SQLITE_FILE}: {exc}"
            ) from exc

        self.dim = self._read_dim()
        self._db_rev = self._read_rev()

        self._hnsw = None
        self._key2sid: dict[int, str] = {}
        self._hnsw_mu = threading.Lock()
        self._hnsw_dirty = False
        self._last_mut = 0.0
        self._graph_rev = -1
        self._hnsw_wake = threading.Event()
        self._closed = False
        self._close_initiated = False
        self._count_cache = None
        # Cached mean of all stored embeddings, used for hubness-corrected
        # (centered) re-ranking in query(). Invalidated on every write.
        self._db_mean = None
        self._db_mean_rev = -1
        self._hnsw_saver = threading.Thread(
            target=self._hnsw_saver_loop, name="hnsw-saver", daemon=True)

        total = self.count()
        if total:
            self._key2sid = {
                int(k): sid for k, sid in
                self._sql("SELECT key, id FROM vectors", fetch="all")
            }
            self._hnsw = self._open_hnsw(total)
            log.info("Vector store ready (%s vectors, %d-dim)", total, self.dim)
        else:
            log.info("Vector store ready (empty)")

        self._state: dict = self._load_state()
        self._state_lock = threading.Lock()
        self._state_dirty = False
        self._state_written = threading.Event()
        self._state_written.set()
        self._writer_wake = threading.Event()
        self._writer = threading.Thread(
            target=self._writer_loop, name="wm-writer", daemon=True)
        self._writer.start()

        self._hnsw_saver.start()

    def _sql(self, sql, args=(), fetch="none"):
        con = _connect()
        try:
            cur = con.execute(sql, args)
            if fetch == "all":
                return cur.fetchall()
            if fetch == "one":
                return cur.fetchone()
            con.commit()
            return cur.rowcount
        finally:
            con.close()

    def _read_rev(self) -> int:
        row = self._sql("SELECT v FROM store_meta WHERE k='rev'", fetch="one")
        try:
            return int(row[0]) if row else 0
        except (TypeError, ValueError):
            return 0

    def _stored_dim(self, con) -> "int | None":
        row = con.execute(
            "SELECT v FROM store_meta WHERE k='dim'").fetchone()
        if row:
            return int(row[0])
        row = con.execute("SELECT MIN(dim) FROM vectors").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def _read_dim(self) -> "int | None":
        con = _connect()
        try:
            return self._stored_dim(con)
        finally:
            con.close()

    def _open_hnsw(self, total: int) -> Index:
        assert self.dim is not None and total > 0
        rev_on_disk = _read_rev_file()
        try:
            idx = Index(ndim=self.dim, metric=MetricKind.Cos, dtype="f32")
            idx.load(str(_HNSW_FILE))
            on_disk = {int(k) for k in idx.keys}
            if (rev_on_disk is not None and rev_on_disk == self._db_rev
                    and idx.size == total and on_disk == set(self._key2sid)
                    and int(idx.ndim) == int(self.dim)):
                self._graph_rev = rev_on_disk
                return idx
            if rev_on_disk is None:
                log.info("ANN index has no revision marker — rebuilding from %d "
                         "stored vectors", total)
            else:
                log.info("ANN index out of sync (rev %s vs %s, %d keys vs %d "
                         "rows, %d-dim vs %d-dim) — rebuilding",
                         rev_on_disk, self._db_rev, idx.size, total,
                         int(idx.ndim), int(self.dim))
        except FileNotFoundError:
            log.info("No ANN index on disk — building from %d stored vectors", total)
        except Exception as exc:
            log.warning("ANN index unusable (%s) — rebuilding from SQLite", exc)
        return self._rebuild_hnsw()

    def _rebuild_hnsw(self) -> Index:
        assert self.dim is not None
        rev = self._db_rev
        idx = Index(ndim=self.dim, metric=MetricKind.Cos, dtype="f32")
        offset = 0
        while True:
            rows = self._sql(
                "SELECT key, embedding, dim FROM vectors ORDER BY key "
                "LIMIT ? OFFSET ?", (_PAGE, offset), fetch="all")
            if not rows:
                break
            dims = {int(r[2]) for r in rows}
            if dims != {self.dim}:
                raise RuntimeError(
                    f"vectors table holds mixed dimensions {sorted(dims)} "
                    f"(expected {self.dim}) — run /index to rebuild")
            keys = np.asarray([r[0] for r in rows], dtype=np.int64)
            vecs = np.frombuffer(
                b"".join(r[1] for r in rows), dtype=np.float32
            ).reshape(len(rows), self.dim)
            idx.add(keys, vecs)
            if len(rows) < _PAGE:
                break
            offset += _PAGE
        with self._hnsw_mu:
            self._hnsw = idx
            self._graph_rev = rev
            self._mark_mutated()
        log.info("ANN index rebuilt (%d vectors)", idx.size)
        return idx

    def _mark_mutated(self) -> None:
        """Caller must hold `_hnsw_mu`."""
        self._last_mut = time.monotonic()
        self._hnsw_dirty = True
        self._hnsw_wake.set()

    def _hnsw_saver_loop(self) -> None:
        while not self._closed:
            self._hnsw_wake.wait()
            self._hnsw_wake.clear()
            if self._closed:
                return
            while True:
                if self._closed:
                    return
                if not self._hnsw_dirty:
                    break
                quiet = time.monotonic() - self._last_mut
                if quiet < _HNSW_SAVE_DEBOUNCE:
                    self._hnsw_wake.wait(_HNSW_SAVE_DEBOUNCE - quiet)
                    self._hnsw_wake.clear()
                    continue
                self._save_hnsw_now()
                continue

    def _save_hnsw_now(self) -> None:
        with self._hnsw_mu:
            if self._hnsw is None or not self._hnsw_dirty or self._closed:
                return
            rev = self._graph_rev
            tmp = _HNSW_FILE.with_name(
                f"hnsw.bin.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            rev_tmp = None
            try:
                self._hnsw.save(str(tmp))
                with _SWAP_LOCK:
                    if self._closed:
                        return
                    os.replace(str(tmp), _HNSW_FILE)
                    rev_tmp = _REV_FILE.with_name(
                        f"hnsw.rev.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
                    with open(rev_tmp, "w") as fh:
                        fh.write(str(rev))
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(str(rev_tmp), _REV_FILE)
                    self._hnsw_dirty = False
            except Exception as exc:
                log.warning("ANN index save failed — will retry: %s", exc)
                self._hnsw_wake.wait(5.0)
            finally:
                for leftover in (tmp, rev_tmp):
                    if leftover is None:
                        continue
                    try:
                        leftover.unlink()
                    except OSError:
                        pass

    @_serialized
    def checkpoint_wal(self) -> bool:
        self._save_hnsw_now()
        try:
            con = sqlite3.connect(str(_SQLITE_FILE), timeout=30)
            try:
                con.execute("PRAGMA busy_timeout=30000;")
                row = con.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
                con.commit()
                if row and row[0]:
                    log.warning("WAL checkpoint busy (readers active): %s", row)
                    return False
                return True
            finally:
                con.close()
        except Exception as exc:
            log.warning("WAL checkpoint failed: %s", exc)
            return False

    @_serialized
    def upsert(self, ids, embeddings, metadatas):
        embs = np.asarray(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        n = len(ids)
        if n == 0:
            return
        if not (len(metadatas) == n == embs.shape[0]):
            raise ValueError(
                f"upsert: ids/metadata/embedding counts disagree "
                f"({n}/{len(metadatas)}/{embs.shape[0]})")

        if len(set(ids)) != n:
            dups = n - len(set(ids))
            last = {sid: i for i, sid in enumerate(ids)}
            ids = list(dict.fromkeys(ids))
            metadatas = [metadatas[last[sid]] for sid in ids]
            embs = embs[[last[sid] for sid in ids]]
            n = len(ids)
            log.warning("upsert: collapsed %d duplicate ids in batch", dups)

        dim = embs.shape[1]
        con = _connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            stored = self._stored_dim(con)
            if stored is None:
                con.execute(
                    "INSERT INTO store_meta(k, v) VALUES('dim', ?)",
                    (str(dim),))
            elif stored != dim:
                raise RuntimeError(
                    f"Dimension mismatch: the index holds {stored}-dim "
                    f"vectors but this batch has {dim}. The model changed "
                    f"without a re-index — run /index to rebuild.")
            qmarks = ",".join("?" * n)
            existing = dict(con.execute(
                f"SELECT id, key FROM vectors WHERE id IN ({qmarks})",
                list(ids)))
            counter = con.execute(
                "SELECT v FROM store_meta WHERE k='next_key'").fetchone()
            kmax = int(counter[0]) if counter else 0
            kmax = max(kmax, con.execute(
                "SELECT COALESCE(MAX(key), 0) FROM vectors").fetchone()[0])
            rows = []
            new_keys = []
            new_vecs = []
            for i, (sid, meta) in enumerate(zip(ids, metadatas)):
                meta = meta or {}
                old = existing.get(sid)
                if old is not None:
                    key = int(old)
                else:
                    kmax += 1
                    key = kmax
                rows.append((
                    sid, key, dim, embs[i].tobytes(),
                    json.dumps(meta), _channel_id_of(meta)))
                new_keys.append(key)
                new_vecs.append(embs[i])
            con.executemany(
                "INSERT INTO vectors(id, key, dim, embedding, metadata, channel_id)"
                " VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "key=excluded.key, dim=excluded.dim, "
                "embedding=excluded.embedding, metadata=excluded.metadata, "
                "channel_id=excluded.channel_id", rows)
            con.execute(
                "UPDATE store_meta SET v = ? WHERE k='next_key'", (str(kmax),))
            con.execute(
                "UPDATE store_meta SET v = CAST(CAST(v AS INTEGER) + 1 AS TEXT)"
                " WHERE k='rev'")
            rev = int(con.execute(
                "SELECT v FROM store_meta WHERE k='rev'").fetchone()[0])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

        self._db_rev = rev
        self._count_cache = None
        self._db_mean = None
        self._db_mean_rev = -1

        try:
            with self._hnsw_mu:
                if self._hnsw is None:
                    self._hnsw = Index(
                        ndim=dim, metric=MetricKind.Cos, dtype="f32")
                    self.dim = dim
                for old in existing.values():
                    if old is not None:
                        self._hnsw.remove(np.int64(old))
                        self._key2sid.pop(int(old), None)
                keys_arr = np.asarray(new_keys, dtype=np.int64)
                vecs_arr = np.asarray(new_vecs, dtype=np.float32)
                self._hnsw.add(keys_arr, vecs_arr)
                for sid, key in zip(ids, new_keys):
                    self._key2sid[int(key)] = sid
                self._graph_rev = rev
                self._mark_mutated()
        except Exception as exc:
            log.error("ANN mirror failed (%s) — vectors are committed; the "
                      "graph revision is now behind and will be rebuilt on "
                      "next start", exc)

    @_serialized
    def query(self, embedding, n=10):
        empty = {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        if self._hnsw is None or not self._hnsw.size:
            return empty
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if self.dim is not None and q.shape[0] != self.dim:
            raise RuntimeError(
                f"Query is {q.shape[0]}-dim but the index holds "
                f"{self.dim}-dim vectors — the model changed; run /index.")
        # If the in-memory graph fell behind (mirror failure), rebuild before
        # serving so results are never stale.
        if self._graph_rev != self._db_rev:
            log.warning("ANN graph stale (rev %s vs %s) — rebuilding",
                        self._graph_rev, self._db_rev)
            self._rebuild_hnsw()
        size = int(self._hnsw.size)
        n = max(1, min(int(n), size))

        # Hubness-corrected re-ranking (Nearest-Neighbor / distribution
        # normalisation). CLIP-family embeddings suffer from "hubs" that
        # dominate the nearest-neighbour lists, so a query for an image that IS
        # in the index can still return a different image as the top hit. We
        # keep the raw cosine for the reported similarity (so the existing
        # 0-1 scale and thresholds stay meaningful) but ORDER the results by a
        # centred cosine, which down-weights hubs. The candidate pool is taken
        # from the exact raw-cosine top-K so the re-rank never has to load the
        # whole table; the true match of a photo query is always within a few
        # dozen positions by raw cosine.
        K = min(size, max(n * 8, 64))
        res = self._hnsw.search(q.reshape(1, -1), count=K, exact=True)
        cand_keys = [int(k) for k in res.keys]
        cand_dists = [float(d) for d in res.distances]
        cand_sids = [self._key2sid.get(k) for k in cand_keys]
        # keep only resolvable
        usable = [(k, sid, dist) for k, sid, dist in zip(cand_keys, cand_sids, cand_dists)
                  if sid is not None]
        if not usable:
            return empty

        # Cached mean of the DB embeddings, invalidated on writes.
        if self._db_mean is None or self._db_mean_rev != self._db_rev:
            self._db_mean = self._compute_db_mean()
            self._db_mean_rev = self._db_rev

        # Load the candidate embeddings from SQLite for centred scoring.
        cand_sid_list = [u[1] for u in usable]
        qmarks = ",".join("?" * len(cand_sid_list))
        rows = self._sql(
            f"SELECT id, embedding, metadata FROM vectors WHERE id IN ({qmarks})",
            cand_sid_list, fetch="all")
        meta_by_sid = {r[0]: (json.loads(r[2]) if r[2] else {}) for r in rows}
        emb_by_sid = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}

        mu = self._db_mean
        qc = q - (mu if mu is not None else 0.0)
        qn = float(np.linalg.norm(qc)) or 1.0

        scored = []
        for key, sid, raw_dist in usable:
            if sid not in emb_by_sid:
                continue
            cvec = emb_by_sid[sid]
            if self.dim is not None and cvec.shape[0] != self.dim:
                continue
            cc = cvec - (mu if mu is not None else 0.0)
            cn = float(np.linalg.norm(cc)) or 1.0
            centered_sim = float(np.dot(cc, qc) / (cn * qn))
            meta = meta_by_sid.get(sid, {})
            scored.append((sid, meta, raw_dist, centered_sim))

        if not scored:
            return empty
        # Order by centred similarity (descending); tie-break by raw distance.
        scored.sort(key=lambda x: (-x[3], x[2]))
        top = scored[:n]

        out_ids, out_metas, out_dists = [], [], []
        for sid, meta, raw_dist, _ in top:
            out_ids.append(sid)
            out_metas.append(meta)
            out_dists.append(raw_dist)
        if not out_ids:
            return empty
        return {"ids": [out_ids], "metadatas": [out_metas],
                "distances": [out_dists]}

    def _compute_db_mean(self):
        # Mean of all stored (unit-normalised) embeddings. Cached per revision.
        total = self.count()
        if total == 0:
            return None
        acc = None
        cnt = 0
        for ids, metas, embs in self.iter_vectors(page=1000):
            if acc is None:
                acc = np.zeros(embs.shape[1], dtype=np.float64)
            acc += embs.astype(np.float64).sum(axis=0)
            cnt += embs.shape[0]
        if cnt == 0:
            return None
        return (acc / cnt).astype(np.float32)

    @_serialized
    def count(self):
        if self._count_cache is not None:
            return self._count_cache
        row = self._sql("SELECT COUNT(*) FROM vectors", fetch="one")
        self._count_cache = int(row[0]) if row else 0
        return self._count_cache

    @_serialized
    def delete(self, ids: list[str]):
        if not ids:
            return
        ids = list(dict.fromkeys(ids))
        con = _connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            qmarks = ",".join("?" * len(ids))
            rows = con.execute(
                f"SELECT id, key FROM vectors WHERE id IN ({qmarks})",
                ids).fetchall()
            if not rows:
                con.commit()
                return
            con.executemany(
                "DELETE FROM vectors WHERE id = ?", [(r[0],) for r in rows])
            con.execute(
                "UPDATE store_meta SET v = CAST(CAST(v AS INTEGER) + 1 AS TEXT)"
                " WHERE k='rev'")
            rev = int(con.execute(
                "SELECT v FROM store_meta WHERE k='rev'").fetchone()[0])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        self._db_rev = rev
        self._count_cache = None
        self._db_mean = None
        self._db_mean_rev = -1
        try:
            with self._hnsw_mu:
                if self._hnsw is not None:
                    for _, key in rows:
                        self._hnsw.remove(np.int64(key))
                for _, key in rows:
                    self._key2sid.pop(int(key), None)
                self._graph_rev = rev
                self._mark_mutated()
        except Exception as exc:
            log.error("ANN removal failed (%s) — rows are deleted; the graph "
                      "revision is now behind and will be rebuilt on next "
                      "start", exc)

    @_serialized
    def get_all_ids(self) -> list[str]:
        ids, _ = self.paged_get(page=1000)
        return ids

    @_serialized
    def get_ids_for_channel(self, channel_id: int) -> list[dict]:
        results: list[dict] = []
        page = 1000
        last = ""
        for _ in range(100_000):
            rows = self._sql(
                "SELECT id, metadata, channel_id FROM vectors "
                "WHERE (channel_id = ? OR channel_id IS NULL) "
                "AND id > ? ORDER BY id LIMIT ?",
                (channel_id, last, page), fetch="all")
            if not rows:
                break
            for doc_id, mtext, cid in rows:
                meta = json.loads(mtext) if mtext else {}
                if cid is None and _channel_id_of(meta) != channel_id:
                    continue
                results.append({
                    "id": doc_id,
                    "message_id": meta.get("message_id", 0),
                })
            last = rows[-1][0]
            if len(rows) < page:
                break
        return results

    def channel_overview(self) -> dict[int, dict]:
        """
        Per-channel bookkeeping for the /channels panel.

        A "channel" groups rows by their true channel id — both the
        channel_id column and (for legacy rows whose column is NULL) the id
        inside the metadata JSON, exactly the two shapes get_ids_for_channel()
        matches, so any key returned here is a valid forget_channel() target.

        Returns: {true_channel_id: {"name": str, "count": int, "watermark": int}}
        """
        out: dict[int, dict] = {}
        with _DIR_LOCK:
            con = _connect()
            try:
                rows = con.execute(
                    "SELECT channel_id, COUNT(*) FROM vectors "
                    "GROUP BY channel_id").fetchall()
                for cid, cnt in rows:
                    if cid is None:
                        continue
                    e = out.setdefault(int(cid), {"name": str(cid), "count": 0})
                    e["count"] += int(cnt)
                    if e["name"] == str(cid):
                        srow = con.execute(
                            "SELECT metadata FROM vectors WHERE channel_id = ? "
                            "AND metadata IS NOT NULL LIMIT 1", (cid,)).fetchone()
                        if srow:
                            try:
                                nm = (json.loads(srow[0]) or {}).get("channel_name")
                            except Exception:
                                nm = None
                            if nm:
                                e["name"] = str(nm)
                # Legacy rows (NULL column value) keep the channel in metadata.
                page, last = 1000, ""
                for _ in range(100_000):
                    nrows = con.execute(
                        "SELECT id, metadata FROM vectors "
                        "WHERE channel_id IS NULL AND id > ? ORDER BY id LIMIT ?",
                        (last, page)).fetchall()
                    if not nrows:
                        break
                    for doc_id, mtext in nrows:
                        last = doc_id
                        try:
                            meta = json.loads(mtext) if mtext else {}
                        except Exception:
                            meta = {}
                        cid = _channel_id_of(meta)
                        if cid is None:
                            continue
                        e = out.setdefault(cid, {"name": str(cid), "count": 0})
                        e["count"] += 1
                        nm = meta.get("channel_name")
                        if nm and e["name"] == str(cid):
                            e["name"] = str(nm)
            finally:
                con.close()
        wm = self.list_channels()
        for tid, e in out.items():
            w = wm.get(_norm_channel_id(tid))
            if w is not None:
                e["watermark"] = int(w)
        return out

    def forget_channel(self, channel_id: int) -> int:
        """
        Remove a channel entirely: delete every vector that belongs to it
        (column id or metadata id — same matching as get_ids_for_channel) and
        drop its monitor watermark. Returns the vector count removed.

        Deliberately NOT @_serialized: delete() takes _DIR_LOCK per chunk and
        that lock is not re-entrant.
        """
        docs = self.get_ids_for_channel(channel_id)
        ids = [d["id"] for d in docs]
        total = len(ids)
        for i in range(0, total, 1000):
            self.delete(ids[i:i + 1000])
        key = _norm_channel_id(channel_id)
        with self._state_lock:
            if self._state.pop(key, None) is not None:
                self._state_dirty = True
                self._state_written.clear()
        self._flush_state()
        return total

    @_serialized
    def paged_get(self, where=None, include=("ids",), page: int = 1000):
        if "embeddings" in include:
            raise ValueError(
                "paged_get does not return embeddings — use iter_vectors()")
        want_meta = "metadatas" in include
        cols = "id, metadata" if want_meta else "id"
        sql = f"SELECT {cols} FROM vectors"
        args: list = []
        if where:
            if set(where) != {"channel_id"}:
                raise ValueError(
                    "Only the {'channel_id': <int>} filter is supported")
            sql += " WHERE channel_id = ? AND id > ?"
            args.append(where["channel_id"])
        else:
            sql += " WHERE id > ?"
        sql += " ORDER BY id LIMIT ?"
        ids: list = []
        extras: list = []
        last = ""
        for _ in range(100_000):
            rows = self._sql(sql, (*args, last, page), fetch="all")
            if not rows:
                break
            ids.extend(r[0] for r in rows)
            if want_meta:
                extras.extend(
                    (json.loads(r[1]) if r[1] else {}) for r in rows)
            last = rows[-1][0]
            if len(rows) < page:
                break
        return ids, extras

    def iter_vectors(self, page: int = 1000):
        last = ""
        running_dim = None
        for _ in range(100_000):
            with _DIR_LOCK:
                con = _connect()
                try:
                    rows = con.execute(
                        "SELECT id, metadata, embedding, dim FROM vectors "
                        "WHERE id > ? ORDER BY id LIMIT ?",
                        (last, page)).fetchall()
                finally:
                    con.close()
            if not rows:
                return
            ids = [r[0] for r in rows]
            metas = [(json.loads(r[1]) if r[1] else {}) for r in rows]
            dims = {int(r[3]) for r in rows}
            if len(dims) != 1:
                raise RuntimeError(
                    f"vectors table holds mixed dimensions {sorted(dims)} — "
                    f"run /index to rebuild the store")
            page_dim = dims.pop()
            if running_dim is None:
                running_dim = page_dim
            elif page_dim != running_dim:
                raise RuntimeError(
                    f"vectors table holds mixed dimensions "
                    f"({running_dim} vs {page_dim}) — run /index to rebuild")
            if self.dim is not None and page_dim != self.dim:
                raise RuntimeError(
                    f"vectors table holds {page_dim}-dim vectors but store "
                    f"expects {self.dim}-dim — run /index to rebuild")
            embs = np.frombuffer(
                b"".join(r[2] for r in rows), dtype=np.float32
            ).reshape(len(rows), page_dim)
            yield ids, metas, embs
            last = rows[-1][0]
            if len(rows) < page:
                return

    def _load_state(self):
        return read_watermarks()

    def _save_state(self, state):
        _atomic_write_json(_STATE_FILE, state)

    def get_watermark(self, channel_id):
        key = _norm_channel_id(channel_id)
        with self._state_lock:
            return self._state.get(key, 0)

    def set_watermark(self, channel_id, msg_id):
        key = _norm_channel_id(channel_id)
        with self._state_lock:
            if self._state.get(key, 0) >= msg_id:
                return
            self._state[key] = msg_id
            self._state_dirty = True
            self._state_written.clear()
        self._flush_state()

    def list_channels(self):
        with self._state_lock:
            return dict(self._state)

    def reload_state(self):
        """Re-read the watermark file from disk into the in-memory cache."""
        fresh = read_watermarks()
        with self._state_lock:
            self._state = fresh

    def _flush_state(self):
        self._writer_wake.set()

    def flush_state(self, timeout: float = 5.0) -> bool:
        if self._closed:
            return True
        self._writer_wake.set()
        return self._state_written.wait(timeout)

    def _writer_loop(self):
        while not self._closed:
            self._writer_wake.wait()
            self._writer_wake.clear()
            if self._closed:
                return
            while True:
                with self._state_lock:
                    if not self._state_dirty:
                        self._state_written.set()
                        break
                    snapshot = dict(self._state)
                    self._state_dirty = False
                    epoch = _DIR_EPOCH
                try:
                    with _SWAP_LOCK:
                        if _DIR_EPOCH != epoch or self._closed:
                            with self._state_lock:
                                self._state_dirty = True
                                self._state_written.set()
                            break
                        _atomic_write_json(_STATE_FILE, snapshot)
                except Exception as exc:
                    with self._state_lock:
                        self._state_dirty = True
                    log.warning("Watermark write failed (%s) — will retry: %s",
                                _STATE_FILE, exc)
                    self._writer_wake.wait(5.0)
                    continue
                with self._state_lock:
                    if self._state_dirty:
                        continue
                    self._state_written.set()
                break

    def close(self, wait: bool = False, timeout: float = 10.0) -> bool:
        if getattr(self, "_close_initiated", False):
            return True
        self._close_initiated = True
        if wait:
            flushed = self.flush_state(timeout)
            self._closed = True
            self._writer_wake.set()
            self._hnsw_wake.set()
            for thread in (self._writer, self._hnsw_saver):
                if thread.is_alive() and thread is not threading.current_thread():
                    thread.join(timeout)
            return flushed
        self._writer_wake.set()
        self._hnsw_wake.set()
        return True
