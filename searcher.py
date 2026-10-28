from __future__ import annotations
from config import config
from database import VectorDB
from embeddings import CLIPEmbedder


class ImageSearcher:
    def __init__(self, db: VectorDB, embedder: CLIPEmbedder) -> None:
        self.db = db
        self.embedder = embedder

    def search(self, *, path=None, data=None, n=None):
        n = n or config.MAX_RESULTS
        if data:
            emb = self.embedder.embed_bytes_query(data)
        elif path:
            emb = self.embedder.embed_path_query(path)
        else:
            raise ValueError("supply path or data")
        if self.db.count() == 0:
            return []
        raw = self.db.query(emb, n=n)
        results = []
        for i in range(len(raw["ids"][0])):
            meta = raw["metadatas"][0][i]
            results.append({
                "id": raw["ids"][0][i],
                "similarity": round(1.0 - raw["distances"][0][i], 4),
                "channel_name": meta.get("channel_name", ""),
                "message_id": meta.get("message_id", 0),
                "link": meta.get("link", ""),
                "date": meta.get("date", ""),
                "private": meta.get("private", False),
            })
        return results

    def duplicates(self, *, path=None, data=None, threshold=None):
        thr = threshold if threshold is not None else config.THRESHOLD
        hits = self.search(path=path, data=data, n=20)
        return [h for h in hits if h["similarity"] >= thr]
