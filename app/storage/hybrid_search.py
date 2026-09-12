"""Combines dense (semantic) and sparse (BM25) search into one ranked list.

Uses a simple weighted score fusion on normalized scores. Both stores already
return 0-1 normalized scores, so this is a straightforward blend rather than
full reciprocal rank fusion — simple, and easy to reason about when tuning
dense_weight in config.py.
"""
from app.config import settings
from app.storage.vector_store import VectorStore
from app.storage.bm25_store import BM25Store


class HybridSearcher:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

    def index(self, chunks: list[dict]) -> None:
        self.vector_store.add_chunks(chunks)
        self.bm25_store.add_chunks(chunks)

    def search(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or settings.top_k_final
        dense_hits = self.vector_store.search(query)
        sparse_hits = self.bm25_store.search(query)

        combined: dict[str, dict] = {}
        for hit in dense_hits:
            combined[hit["id"]] = {
                **hit,
                "dense_score": hit["score"],
                "sparse_score": 0.0,
            }
        for hit in sparse_hits:
            if hit["id"] in combined:
                combined[hit["id"]]["sparse_score"] = hit["score"]
            else:
                combined[hit["id"]] = {
                    **hit,
                    "dense_score": 0.0,
                    "sparse_score": hit["score"],
                }

        w = settings.dense_weight
        for hit in combined.values():
            hit["final_score"] = w * hit["dense_score"] + (1 - w) * hit["sparse_score"]

        ranked = sorted(combined.values(), key=lambda h: h["final_score"], reverse=True)
        return ranked[:top_k]
