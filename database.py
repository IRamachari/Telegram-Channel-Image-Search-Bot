from __future__ import annotations
import json
import logging
import os
import sqlite3
import threading
import time
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

    ``Path.write_text`` truncates first and writes second, so a crash in between
    leaves an empty/partial file. A same-directory temp file + ``os.replace`` is
    the documented way to make this atomic (both live on the same filesystem).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


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


class VectorDB:
    """
    Vector store over SQLite (source of truth) + usearch HNSW (derived cache).

    Correctness never depends on when ``hnsw.bin`` was last written: on open,
    the on-disk graph is compared against the SQLite key set and rebuilt from
    the stored embeddings if it is missing, stale or corrupt. A crash at any
    point therefore costs one rebuild, never a lost vector.
    """

    @_serialized
    def __init__(self) -> None:
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)

        # Apply WAL mode + busy timeout BEFORE anything else opens the file.
        # WAL lets searches run concurrently with a batch commit, and
        # busy_timeout makes SQLite retry for up to 30 s instead of raising
        # "database is locked" when two threads hit the DB at once.
        # journal_mode is persistent, so the pragma is needed only once — but
        # running it on every open is harmless and self-repairs a database
        # that arrived without it (e.g. extracted from a backup).
        try:
            con = sqlite3.connect(str(_SQLITE_FILE), timeout=30)
            try:
                for pragma in ("journal_mode=WAL", "busy_timeout=30000"):
                    try:
                        con.execute(f"PRAGMA {pragma};")
                    except sqlite3.DatabaseError as exc:
                        # WAL is unavailable on some filesystems (network
                        # mounts); SQLite still works in its default mode.
                        log.warning("SQLite pragma %s skipped: %s",
                                    pragma, exc)
                for stmt in _SCHEMA:
                    con.execute(stmt)
                con.commit()
            finally:
                con.close()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot initialise vector store at {_SQLITE_FILE}: {exc}"
            ) from exc

        self.dim = self._read_dim()

        # ── ANN index + concurrency state ──────────────────────────────────
        # `_hnsw_mu` serialises graph mutations (add/remove) against the
        # background save. Searches do not take it: usearch supports concurrent
        # search during add, and a save never touches the in-memory graph.
        self._hnsw = None            # type: Index | None
        self._key2sid: dict[int, str] = {}
        self._hnsw_mu = threading.Lock()
        self._hnsw_dirty = False
        self._last_mut = 0.0
        self._hnsw_wake = threading.Event()
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

        # ── watermark state ────────────────────────────────────────────────
        # Cached in memory so get_watermark()/list_channels() are O(1) instead
        # of a full file read, and so concurrent writers cannot clobber each
        # other's entries (see set_watermark). Persisted by a single writer
        # thread so the fsync never lands on the event loop.
        self._state: dict = self._load_state()
        self._state_lock = threading.Lock()
        self._state_dirty = False
        self._written = False
        self._writer_wake = threading.Event()
        self._writer = threading.Thread(
            target=self._writer_loop, name="wm-writer", daemon=True)
        self._writer.start()

        self._hnsw_saver.start()

    # ── sqlite plumbing ─────────────────────────────────────────────────────

    def _sql(self, sql, args=(), fetch="none"):
        con = _connect()
        try:
            cur = con.execute(sql, args)
            if fetch == "all":
                return cur.fetchall()
            if fetch == "one":
                return cur.fetchone()
            con.commit()
            return cur
        finally:
            con.close()

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

    # ── ANN index lifecycle ────────────────────────────────────────────────

    def _open_hnsw(self, total: int) -> Index:
        """
        Load ``hnsw.bin`` if it matches the SQLite key set; rebuild otherwise.
        """
        assert self.dim is not None and total > 0
        try:
            idx = Index(ndim=self.dim, metric=MetricKind.Cos, dtype="f32")
            idx.load(str(_HNSW_FILE))
            on_disk = {int(k) for k in idx.keys}
            if idx.size == total and on_disk == set(self._key2sid):
                return idx
            log.info("ANN index out of sync (%d on disk vs %d rows) — rebuilding",
                     idx.size, total)
        except FileNotFoundError:
            log.info("No ANN index on disk — building from %d stored vectors", total)
        except Exception as exc:
            log.warning("ANN index unusable (%s) — rebuilding from SQLite", exc)
        return self._rebuild_hnsw()

    def _rebuild_hnsw(self) -> Index:
        assert self.dim is not None
        idx = Index(ndim=self.dim, metric=MetricKind.Cos, dtype="f32")
        offset = 0
        while True:
            rows = self._sql(
                "SELECT key, embedding FROM vectors ORDER BY key "
                "LIMIT ? OFFSET ?", (_PAGE, offset), fetch="all")
            if not rows:
                break
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
            self._mark_mutated()
        log.info("ANN index rebuilt (%d vectors)", idx.size)
        return idx

    def _mark_mutated(self) -> None:
        """Caller must hold `_hnsw_mu`."""
        self._last_mut = time.monotonic()
        self._hnsw_dirty = True
        self._hnsw_wake.set()

    def _hnsw_saver_loop(self) -> None:
        """
        Coalescing writer for the ANN graph file.

        Waits for a quiet window (no mutation for `_HNSW_SAVE_DEBOUNCE` s) so a
        backfill does not serialise the whole graph to disk after every batch.
        If the process dies before a save lands, the open-time consistency
        check simply rebuilds from SQLite — timing here can never lose data.
        """
        while True:
            self._hnsw_wake.wait()
            self._hnsw_wake.clear()
            while True:
                if not self._hnsw_dirty:
                    break
                quiet = time.monotonic() - self._last_mut
                if quiet < _HNSW_SAVE_DEBOUNCE:
                    # Returns True when another mutation arrived; either way
                    # re-evaluate (more mutations may have queued).
                    self._hnsw_wake.wait(_HNSW_SAVE_DEBOUNCE - quiet)
                    self._hnsw_wake.clear()
                    continue
                self._save_hnsw_now()
                continue  # re-check: a mutation may have landed during the save

    def _save_hnsw_now(self) -> None:
        with self._hnsw_mu:
            if self._hnsw is None or not self._hnsw_dirty:
                return
            try:
                tmp = _HNSW_FILE.with_name("hnsw.bin.tmp")
                self._hnsw.save(str(tmp))
                os.replace(str(tmp), _HNSW_FILE)
                self._hnsw_dirty = False
            except Exception as exc:
                # ENOSPC/EIO: keep the dirty flag so the next quiet window
                # retries; the open-time check heals a lost file regardless.
                log.warning("ANN index save failed — will retry: %s", exc)
                self._hnsw_wake.wait(5.0)

    # ── maintenance ────────────────────────────────────────────────────────

    @_serialized
    def checkpoint_wal(self) -> bool:
        """
        Fold the write-ahead log into ``vectors.sqlite3`` and flush the ANN
        graph file, so a plain directory copy is a consistent snapshot.

        Returns True when the checkpoint actually truncated the WAL. TRUNCATE
        needs exclusive access; if another connection holds a read transaction
        it reports busy=1 instead of blocking, and the backup must know about
        it. A failed graph save is NOT fatal for backups — the importing side
        rebuilds ``hnsw.bin`` from SQLite if it is stale.
        """
        self._save_hnsw_now()
        try:
            con = sqlite3.connect(str(_SQLITE_FILE), timeout=30)
            try:
                con.execute("PRAGMA busy_timeout=30000;")
                row = con.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
                con.commit()
                if row and row[0]:
                    # busy == 1: the WAL was NOT truncated
                    log.warning("WAL checkpoint busy (readers active): %s", row)
                    return False
                return True
            finally:
                con.close()
        except Exception as exc:
            log.warning("WAL checkpoint failed: %s", exc)
            return False

    # ── queries ─────────────────────────────────────────────────────────────

    @_serialized
    def upsert(self, ids, embeddings, metadatas):
        """
        Insert or replace records, mirroring the committed rows into the ANN
        graph. SQLite commits FIRST: if the process dies between the commit and
        the graph update, the open-time consistency check rebuilds the graph —
        the reverse order could serve vectors that were never persisted.
        """
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

        dim = embs.shape[1]
        con = _connect()
        try:
            with con:  # single atomic transaction
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
                kmax = con.execute(
                    "SELECT COALESCE(MAX(key), 0) FROM vectors").fetchone()[0]
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
                    "INSERT OR REPLACE INTO vectors"
                    "(id, key, dim, embedding, metadata, channel_id)"
                    " VALUES(?,?,?,?,?,?)", rows)
        finally:
            con.close()

        # Committed. Mirror into the graph; a failure here only delays
        # visibility until the next open-time rebuild.
        try:
            with self._hnsw_mu:
                if self._hnsw is None:
                    self._hnsw = Index(
                        ndim=dim, metric=MetricKind.Cos, dtype="f32")
                    self.dim = dim
                for old in existing.values():
                    if old is not None:
                        # Same key reused on overwrite: usearch forbids adding
                        # a live duplicate key, but allows re-adding a removed
                        # one, so drop then re-add.
                        self._hnsw.remove(np.int64(old))
                        self._key2sid.pop(int(old), None)
                keys_arr = np.asarray(new_keys, dtype=np.int64)
                vecs_arr = np.asarray(new_vecs, dtype=np.float32)
                self._hnsw.add(keys_arr, vecs_arr)
                for sid, key in zip(ids, new_keys):
                    self._key2sid[int(key)] = sid
                self._mark_mutated()
        except Exception as exc:
            log.error("ANN mirror failed (%s) — vectors are stored and the "
                      "graph will be rebuilt on next start", exc)

    @_serialized
    def query(self, embedding, n=10):
        """
        ChromaDB-shaped result: ``{"ids": [[..]], "metadatas": [[..]],
        "distances": [[..]]}`` with distance = 1 - cosine similarity.
        """
        empty = {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        total = self.count()
        if total == 0 or self._hnsw is None:
            return empty
        q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if self.dim is not None and q.shape[1] != self.dim:
            raise RuntimeError(
                f"Query is {q.shape[1]}-dim but the index holds "
                f"{self.dim}-dim vectors — the model changed; run /index.")
        n = max(1, min(int(n), total))
        res = self._hnsw.search(q, count=n)
        pairs = []
        for k, d in zip(res.keys, res.distances):
            sid = self._key2sid.get(int(k))
            if sid is not None:
                pairs.append((sid, float(d)))
        if not pairs:
            return empty
        qmarks = ",".join("?" * len(pairs))
        metas = {
            sid: json.loads(mtext) if mtext else {}
            for sid, mtext in self._sql(
                f"SELECT id, metadata FROM vectors WHERE id IN ({qmarks})",
                [s for s, _ in pairs], fetch="all")
        }
        out_ids, out_metas, out_dists = [], [], []
        for sid, dist in pairs:
            if sid not in metas:      # deleted between count() and now
                continue
            out_ids.append(sid)
            out_metas.append(metas[sid])
            out_dists.append(dist)
        if not out_ids:
            return empty
        return {"ids": [out_ids], "metadatas": [out_metas],
                "distances": [out_dists]}

    @_serialized
    def count(self):
        row = self._sql("SELECT COUNT(*) FROM vectors", fetch="one")
        return int(row[0]) if row else 0

    @_serialized
    def delete(self, ids: list[str]):
        if not ids:
            return
        ids = list(dict.fromkeys(ids))
        con = _connect()
        try:
            with con:  # atomic transaction
                qmarks = ",".join("?" * len(ids))
                rows = con.execute(
                    f"SELECT id, key FROM vectors WHERE id IN ({qmarks})",
                    ids).fetchall()
                if not rows:
                    return
                con.executemany(
                    "DELETE FROM vectors WHERE id = ?",
                    [(r[0],) for r in rows])
        finally:
            con.close()
        with self._hnsw_mu:
            if self._hnsw is not None:
                for _, key in rows:
                    # Removing an absent key is a safe no-op.
                    self._hnsw.remove(np.int64(key))
            for _, key in rows:
                self._key2sid.pop(int(key), None)
            self._mark_mutated()

    @_serialized
    def get_all_ids(self) -> list[str]:
        ids, _ = self.paged_get(page=1000)
        return ids

    @_serialized
    def get_ids_for_channel(self, channel_id: int) -> list[dict]:
        """
        All indexed message ids for one channel.

        Paginates with limit+offset under a stable ``ORDER BY id`` so rows
        inserted between pages shift forward instead of being skipped.
        """
        results: list[dict] = []
        page = 1000
        offset = 0
        for _ in range(100_000):
            rows = self._sql(
                "SELECT id, metadata FROM vectors WHERE channel_id = ? "
                "ORDER BY id LIMIT ? OFFSET ?",
                (channel_id, page, offset), fetch="all")
            if not rows:
                break
            for doc_id, mtext in rows:
                meta = json.loads(mtext) if mtext else {}
                results.append({
                    "id": doc_id,
                    "message_id": meta.get("message_id", 0),
                })
            if len(rows) < page:
                break
            offset += page
        return results

    @_serialized
    def paged_get(self, where=None, include=("ids",), page: int = 1000):
        """
        Page through the collection, tolerating records with no metadata.

        `where` accepts the single filter shape the bot uses,
        ``{"channel_id": <int>}``.
        """
        want_meta = "metadatas" in include
        want_emb = "embeddings" in include
        cols = "id" + (", metadata" if want_meta else "") \
                    + (", embedding, dim" if want_emb else "")
        sql = f"SELECT {cols} FROM vectors"
        args: list = []
        if where:
            if set(where) != {"channel_id"}:
                raise ValueError(
                    "Only the {'channel_id': <int>} filter is supported")
            sql += " WHERE channel_id = ?"
            args.append(where["channel_id"])
        sql += " ORDER BY id LIMIT ? OFFSET ?"

        ids: list = []
        extras: list = []
        offset = 0
        for _ in range(100_000):
            rows = self._sql(sql, (*args, page, offset), fetch="all")
            if not rows:
                break
            ids.extend(r[0] for r in rows)
            if want_meta:
                extras.extend(
                    (json.loads(r[1]) if r[1] else {}) for r in rows)
            if len(rows) < page:
                break
            offset += page
        return ids, extras

    def iter_vectors(self, page: int = 1000):
        """
        Stream (ids, metadatas, embeddings) pages in stable ``ORDER BY id``
        order. Used by the duplicate scan, which needs every stored vector.

        The lock is taken per page, NOT across yields: a generator abandoned
        mid-stream would otherwise hold it until garbage collection. Pagination
        stays correct across lock releases because ``ORDER BY id`` is stable.
        """
        offset = 0
        for _ in range(100_000):
            with _DIR_LOCK:
                con = _connect()
                try:
                    rows = con.execute(
                        "SELECT id, metadata, embedding, dim FROM vectors "
                        "ORDER BY id LIMIT ? OFFSET ?",
                        (page, offset)).fetchall()
                finally:
                    con.close()
            if not rows:
                return
            ids = [r[0] for r in rows]
            metas = [(json.loads(r[1]) if r[1] else {}) for r in rows]
            embs = np.frombuffer(
                b"".join(r[2] for r in rows), dtype=np.float32
            ).reshape(len(rows), rows[0][3])
            yield ids, metas, embs
            if len(rows) < page:
                return
            offset += page

    # ── indexing watermarks ────────────────────────────────────────────────

    def _load_state(self):
        """Read the watermark file. Tolerates a partial/corrupt file."""
        if _STATE_FILE.exists():
            try:
                return json.loads(_STATE_FILE.read_text())
            except Exception as e:
                # A partial write used to raise here and killed every
                # index/monitor run; degrade to "nothing indexed yet" instead.
                log.warning("Indexing state load error (%s): %s", _STATE_FILE, e)
        return {}

    def _save_state(self, state):
        _atomic_write_json(_STATE_FILE, state)

    def get_watermark(self, channel_id):
        with self._state_lock:
            return self._state.get(str(channel_id), 0)

    def set_watermark(self, channel_id, msg_id):
        """
        Advance a channel watermark.

        The whole read-modify-write is under `_state_lock` because `_load_state`
        returns the ENTIRE dict: two writers for different channels used to
        clobber each other (the loser's entry silently reverted to 0), causing a
        full re-scan of a channel on every sync.

        The lock is a plain threading.Lock and only ever held for the duration
        of an in-memory dict update plus a queued write — never across an await
        or a blocking call — so it cannot stall the event loop.
        """
        with self._state_lock:
            key = str(channel_id)
            if self._state.get(key, 0) >= msg_id:
                return
            self._state[key] = msg_id
            self._state_dirty = True
        self._flush_state()

    def list_channels(self):
        with self._state_lock:
            return dict(self._state)

    # ── state persistence ─────────────────────────────────────────────────

    def _flush_state(self):
        """
        Persist the watermark state off the calling thread.

        Watermarks are advanced once per flushed batch, i.e. hundreds of times
        during a backfill. Doing the temp-write + fsync + rename inline put a
        1-10 ms stall on the asyncio event loop each time (blocking every
        command and monitor event), so writes are handed to a dedicated writer
        thread and coalesced.
        """
        self._writer_wake.set()

    def _writer_loop(self):
        """Single writer thread: coalesces writes and never blocks the loop."""
        while True:
            self._writer_wake.wait()
            self._writer_wake.clear()
            # Loop until there is genuinely nothing left to persist: an update
            # can land while we are writing.
            while True:
                with self._state_lock:
                    if not self._state_dirty:
                        break
                    snapshot = dict(self._state)
                try:
                    _atomic_write_json(_STATE_FILE, snapshot)
                except Exception as exc:
                    # The dirty flag is deliberately NOT cleared until the write
                    # succeeds. Clearing it early dropped the update on ENOSPC /
                    # EIO for good, and since close() polls the same flag it
                    # could not distinguish "flushed" from "cleared then failed"
                    # — losing the watermark and forcing a full channel re-scan.
                    log.warning("Watermark write failed (%s) — will retry: %s",
                                _STATE_FILE, exc)
                    # Back off briefly so a persistent failure cannot spin.
                    self._writer_wake.wait(5.0)
                    continue
                # Only now is the persisted copy known to match the snapshot.
                with self._state_lock:
                    if self._state_dirty:
                        continue      # newer data arrived; write again
                    self._written = True
                break

    def close(self) -> None:
        """
        Ask both background writers to drain. Never blocks the event loop.

        Neither writer clears its dirty flag until its write succeeds: if the
        saver could not persist, the pending state must survive (a lost
        watermark means a full re-scan of that channel; a lost graph file is
        rebuilt from SQLite on the next open). Per-operation SQLite
        connections have no persistent handle to release.
        """
        self._writer_wake.set()
        self._hnsw_wake.set()
