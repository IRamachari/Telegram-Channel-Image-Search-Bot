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
            return data if isinstance(data, dict) else {}
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

    ``Path.write_text`` truncates first and writes second, so a crash in between
    leaves an empty/partial file. A same-directory temp file + ``os.replace`` is
    the documented way to make this atomic (both live on the same filesystem).
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
            pass          # not every filesystem/platform allows this
    finally:
        try:
            os.unlink(tmp)
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
                # `rev` is bumped by every write transaction; `next_key` is the
                # allocator for the integer HNSW key (never derived from
                # MAX(key) under a deferred transaction — two processes would
                # hand out the same key). Seeded from the current maximum so a
                # store created by an older build upgrades in place.
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

        # ── ANN index + concurrency state ──────────────────────────────────
        # `_hnsw_mu` serialises graph mutations (add/remove) against the
        # background save. Searches do not take it: usearch supports concurrent
        # search during add, and a save never touches the in-memory graph.
        self._hnsw = None            # type: Index | None
        self._key2sid: dict[int, str] = {}
        self._hnsw_mu = threading.Lock()
        self._hnsw_dirty = False
        self._last_mut = 0.0
        self._graph_rev = -1         # revision the in-memory graph reflects
        self._hnsw_wake = threading.Event()
        self._closed = False
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
        self._state_written = threading.Event()
        self._state_written.set()    # nothing pending on a fresh instance
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
            # The connection is closed on the way out, so the cursor would be
            # unusable: report the row count instead of handing one back.
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

    # ── ANN index lifecycle ────────────────────────────────────────────────

    def _open_hnsw(self, total: int) -> Index:
        """
        Load ``hnsw.bin`` if it matches the SQLite key set; rebuild otherwise.
        """
        assert self.dim is not None and total > 0
        rev_on_disk = _read_rev_file()
        try:
            idx = Index(ndim=self.dim, metric=MetricKind.Cos, dtype="f32")
            idx.load(str(_HNSW_FILE))
            on_disk = {int(k) for k in idx.keys}
            # A key set alone cannot prove the graph is current: `upsert` reuses
            # the key of an existing id, so an in-place update that never reached
            # the graph leaves size and keys identical. The revision marker is
            # written together with the graph file and is the real freshness
            # oracle. `ndim` is checked too — `load()` adopts the file's
            # dimensionality, so a graph built for another model would otherwise
            # pass while every query fails.
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
        # The revision is read once: every committed row is included in the
        # rebuild, so the graph reflects exactly this revision.
        rev = self._db_rev
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
        """
        Coalescing writer for the ANN graph file.

        Waits for a quiet window (no mutation for `_HNSW_SAVE_DEBOUNCE` s) so a
        backfill does not serialise the whole graph to disk after every batch.
        If the process dies before a save lands, the open-time consistency
        check simply rebuilds from SQLite — timing here can never lose data.
        """
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
                    # Returns True when another mutation arrived; either way
                    # re-evaluate (more mutations may have queued).
                    self._hnsw_wake.wait(_HNSW_SAVE_DEBOUNCE - quiet)
                    self._hnsw_wake.clear()
                    continue
                self._save_hnsw_now()
                continue  # re-check: a mutation may have landed during the save

    def _save_hnsw_now(self) -> None:
        with self._hnsw_mu:
            if self._hnsw is None or not self._hnsw_dirty or self._closed:
                return
            rev = self._graph_rev
            # Unique temp name: `_hnsw_mu` is per instance, but several
            # instances (bot, import, re-open) and even a second process
            # (`python main.py index`) can save at the same time. A shared
            # `hnsw.bin.tmp` would let one writer rename a half-written file
            # produced by the other.
            tmp = _HNSW_FILE.with_name(
                f"hnsw.bin.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            rev_tmp = None
            try:
                self._hnsw.save(str(tmp))
                # The rename is the only step that must be exclusive against an
                # /import swapping the directory, so it takes `_SWAP_LOCK`
                # (never `_DIR_LOCK`: that is the outer lock of every write
                # method and taking it here would invert the lock order).
                with _SWAP_LOCK:
                    if self._closed:
                        return
                    os.replace(str(tmp), _HNSW_FILE)
                    rev_tmp = _REV_FILE.with_name(
                        f"hnsw.rev.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
                    rev_tmp.write_text(str(rev))
                    os.replace(str(rev_tmp), _REV_FILE)
                    self._hnsw_dirty = False
            except Exception as exc:
                # ENOSPC/EIO: keep the dirty flag so the next quiet window
                # retries; the open-time check heals a lost file regardless.
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
        the graph update, the open-time revision check rebuilds the graph —
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

        # Collapse duplicate ids inside one batch (last one wins, matching
        # INSERT OR REPLACE). Without this a new id appearing twice would get
        # two different keys, SQLite would keep only the second row while the
        # graph kept both, and `query` would return the same id twice.
        if len(set(ids)) != n:
            last = {sid: i for i, sid in enumerate(ids)}
            ids = list(dict.fromkeys(ids))
            metadatas = [metadatas[last[sid]] for sid in ids]
            embs = embs[[last[sid] for sid in ids]]
            n = len(ids)

        dim = embs.shape[1]
        con = _connect()
        try:
            # BEGIN IMMEDIATE, not the implicit DEFERRED transaction of
            # `with con:`: the key allocator reads a counter and writes it back,
            # and a deferred read lock would let a second process
            # (`python main.py index`) allocate the same key. The UNIQUE
            # constraint on `key` would then turn INSERT OR REPLACE into a
            # silent DELETE of the other process's row.
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
            # Self-heal a counter that is behind the data (a store restored
            # from a build without the allocator).
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
                "INSERT OR REPLACE INTO vectors"
                "(id, key, dim, embedding, metadata, channel_id)"
                " VALUES(?,?,?,?,?,?)", rows)
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

        # Committed. Mirror into the graph; a failure here leaves
        # `_graph_rev` behind, so the next open rebuilds from SQLite.
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
                self._graph_rev = rev
                self._mark_mutated()
        except Exception as exc:
            log.error("ANN mirror failed (%s) — vectors are committed; the "
                      "graph revision is now behind and will be rebuilt on "
                      "next start", exc)

    @_serialized
    def query(self, embedding, n=10):
        """
        ChromaDB-shaped result: ``{"ids": [[..]], "metadatas": [[..]],
        "distances": [[..]]}`` with distance = 1 - cosine similarity.
        """
        empty = {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        if self._hnsw is None or not self._hnsw.size:
            return empty
        q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if self.dim is not None and q.shape[1] != self.dim:
            raise RuntimeError(
                f"Query is {q.shape[1]}-dim but the index holds "
                f"{self.dim}-dim vectors — the model changed; run /index.")
        # Clamp to the graph actually being searched instead of running a
        # COUNT(*) on every photo (searcher.search already counts once).
        n = max(1, min(int(n), int(self._hnsw.size)))
        # Exact (linear) search: correctness over speed. The approximate
        # HNSW path could miss the true nearest neighbour on small or
        # clustered sets, which surfaced as "wrong image" results. Exact
        # search is SIMD-optimised in usearch and fast enough for the channel
        # sizes this bot targets.
        res = self._hnsw.search(q, count=n, exact=True)
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
            if sid not in metas:      # deleted between search and lookup
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
                con.execute(
                    "UPDATE store_meta SET v = CAST(CAST(v AS INTEGER) + 1 AS TEXT)"
                    " WHERE k='rev'")
                rev = int(con.execute(
                    "SELECT v FROM store_meta WHERE k='rev'").fetchone()[0])
        finally:
            con.close()
        self._db_rev = rev
        # The graph update is best effort, like upsert's mirror: the rows are
        # already gone, so a usearch failure must not surface as a failed
        # delete. Leaving `_graph_rev` behind makes the next open rebuild.
        try:
            with self._hnsw_mu:
                if self._hnsw is not None:
                    for _, key in rows:
                        # Removing an absent key is a safe no-op.
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
        """
        All indexed message ids for one channel.

        Keyset pagination (``id > last``): a ``LIMIT/OFFSET`` walk over a table
        that the monitor is deleting from skips rows, and one that the indexer
        is inserting into returns duplicates. `id` is the primary key, so the
        keyset cursor is exact.
        """
        results: list[dict] = []
        page = 1000
        last = ""
        for _ in range(100_000):
            rows = self._sql(
                "SELECT id, metadata FROM vectors "
                "WHERE channel_id = ? AND id > ? ORDER BY id LIMIT ?",
                (channel_id, last, page), fetch="all")
            if not rows:
                break
            for doc_id, mtext in rows:
                meta = json.loads(mtext) if mtext else {}
                results.append({
                    "id": doc_id,
                    "message_id": meta.get("message_id", 0),
                })
            last = rows[-1][0]
            if len(rows) < page:
                break
        return results

    @_serialized
    def paged_get(self, where=None, include=("ids",), page: int = 1000):
        """
        Page through the collection, tolerating records with no metadata.

        `where` accepts the single filter shape the bot uses,
        ``{"channel_id": <int>}``. Only ids and metadatas are returned;
        `iter_vectors` is the way to stream embeddings.
        """
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
        """
        Stream (ids, metadatas, embeddings) pages in ``ORDER BY id`` order.
        Used by the duplicate scan, which needs every stored vector.

        The lock is taken per page, NOT across yields: a generator abandoned
        mid-stream would otherwise hold it until garbage collection. A keyset
        cursor (``id > last``) keeps the walk exact across those releases —
        offset pagination skips rows deleted by the monitor and repeats rows
        inserted by the indexer.
        """
        last = ""
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
            embs = np.frombuffer(
                b"".join(r[2] for r in rows), dtype=np.float32
            ).reshape(len(rows), dims.pop())
            yield ids, metas, embs
            last = rows[-1][0]
            if len(rows) < page:
                return

    # ── indexing watermarks ────────────────────────────────────────────────

    def _load_state(self):
        """Read the watermark file. Tolerates a partial/corrupt file."""
        return read_watermarks()

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
            self._state_written.clear()
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

    def flush_state(self, timeout: float = 5.0) -> bool:
        """
        Block until the pending watermark write has landed (or `timeout`).

        The writer is asynchronous, so a caller that is about to replace the
        directory (/import) or exit must be able to wait for it; otherwise the
        merged watermarks are read back from a file that was never updated and
        every channel is re-scanned. Returns True when nothing is pending.
        """
        if self._closed:
            return True
        if not self._state_dirty:
            return True
        self._writer_wake.set()
        return self._state_written.wait(timeout)

    def _writer_loop(self):
        """Single writer thread: coalesces writes and never blocks the loop."""
        while not self._closed:
            self._writer_wake.wait()
            self._writer_wake.clear()
            if self._closed:
                return
            # Loop until there is genuinely nothing left to persist: an update
            # can land while we are writing.
            while True:
                with self._state_lock:
                    if not self._state_dirty:
                        self._state_written.set()
                        break
                    snapshot = dict(self._state)
                    # Cleared HERE, not after the write: leaving it set made the
                    # `continue` below unconditional, so the writer re-snapshotted
                    # and re-fsynced the file forever (one full write per loop
                    # iteration, for the life of the process).
                    self._state_dirty = False
                    epoch = _DIR_EPOCH
                try:
                    with _SWAP_LOCK:
                        if _DIR_EPOCH != epoch or self._closed:
                            # /import replaced the directory: this snapshot
                            # belongs to the retired store and must not be
                            # written over the imported state file.
                            with self._state_lock:
                                self._state_written.set()
                            break
                        _atomic_write_json(_STATE_FILE, snapshot)
                except Exception as exc:
                    # Put the update back: dropping it on ENOSPC / EIO means a
                    # lost watermark and a full channel re-scan.
                    with self._state_lock:
                        self._state_dirty = True
                    log.warning("Watermark write failed (%s) — will retry: %s",
                                _STATE_FILE, exc)
                    # Back off briefly so a persistent failure cannot spin.
                    self._writer_wake.wait(5.0)
                    continue
                # Only now is the persisted copy known to match the snapshot.
                with self._state_lock:
                    if self._state_dirty:
                        continue      # newer data arrived; write again
                    self._state_written.set()
                break

    def close(self, wait: bool = False, timeout: float = 10.0) -> bool:
        """
        Stop both background writers.

        ``wait=False`` (the default, used on the event loop) only signals them,
        so it never blocks. ``wait=True`` additionally flushes the pending
        watermark synchronously and joins both threads; `/import` uses it
        because the next step re-reads the state file and replaces the
        directory. Returns True when the watermark state is on disk.
        """
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
        # Non-blocking (event-loop) path: ask the writers to flush but do not
        # mark the instance closed, otherwise the daemon writer exits before it
        # can persist the pending watermark/graph. The threads are daemons and
        # terminate with the process; /import uses wait=True to join them.
        self._writer_wake.set()
        self._hnsw_wake.set()
        return True
