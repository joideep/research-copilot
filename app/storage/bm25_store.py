"""Sparse (keyword) index using BM25, complementing the dense vector store.

Pure semantic search often misses exact terms — names, acronyms, numbers,
API names — that BM25 catches reliably. Persisted to disk as a pickle so it
survives restarts alongside Chroma.
"""
import os
import pickle
import re

from rank_bm25 import BM25Okapi

from app.config import settings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Store:
    def __init__(self):
        self.chunks: list[dict] = []   # parallel list of {id, text, metadata}
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if os.path.exists(settings.bm25_index_path):
            with open(settings.bm25_index_path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            self._rebuild_index()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(settings.bm25_index_path), exist_ok=True)
        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump({"chunks": self.chunks}, f)

    def _rebuild_index(self) -> None:
        if not self.chunks:
            self.bm25 = None
            return
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def add_chunks(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)
        self._rebuild_index()
        self._save()

    def search(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or settings.top_k_sparse
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
        hits = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            chunk = self.chunks[i]
            hits.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": scores[i] / max_score,  # normalize to 0-1
            })
        return hits
