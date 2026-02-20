from __future__ import annotations
import json
import logging
from pathlib import Path

import chromadb

from config import config

log = logging.getLogger(__name__)
_STATE_FILE = Path(config.CHROMA_DIR) / "indexing_state.json"


class VectorDB:
    def __init__(self) -> None:
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.col = self.client.get_or_create_collection(
            name=config.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("ChromaDB ready  (%s vectors)", self.col.count())

    def upsert(self, ids, embeddings, metadatas):
        self.col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding, n=10):
        return self.col.query(
            query_embeddings=[embedding],
            n_results=min(n, self.col.count() or 1),
            include=["metadatas", "distances"],
        )

    def count(self):
        return self.col.count()

    def _load_state(self):
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text())
        return {}

    def _save_state(self, state):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, indent=2))

    def get_watermark(self, channel_id):
        return self._load_state().get(str(channel_id), 0)

    def set_watermark(self, channel_id, msg_id):
        s = self._load_state()
        s[str(channel_id)] = msg_id
        self._save_state(s)

    def list_channels(self):
        return self._load_state()
