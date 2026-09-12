"""Dense (semantic) vector store backed by ChromaDB with local embeddings.

Embeddings run locally via sentence-transformers, so ingesting and searching
your library costs no API calls.
"""
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


class VectorStore:
    def __init__(self, collection_name: str = "research_library"):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )

    def search(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or settings.top_k_dense
        embedder = get_embedder()
        query_embedding = embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        hits = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for id_, text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns a distance (lower = closer); convert to a
            # 0-1 similarity score so it combines cleanly with BM25 scores.
            similarity = 1.0 / (1.0 + dist)
            hits.append({"id": id_, "text": text, "metadata": meta, "score": similarity})
        return hits

    def count(self) -> int:
        return self.collection.count()
