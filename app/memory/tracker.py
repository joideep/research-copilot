"""Tracks what's been ingested and queried, so the assistant can surface
connections like 'you read something related to this a few weeks ago'.

Deliberately simple — a JSON file, not a database — since a single user's
reading history is small. This is the piece most RAG demos skip; it's what
turns search into something closer to a research memory.
"""
import json
import os
from datetime import datetime, timezone

from app.config import settings


class MemoryTracker:
    def __init__(self):
        self.path = settings.memory_db_path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {"documents": [], "queries": []}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_document(self, source: str, title: str, doc_type: str, topics: list[str] = None) -> None:
        self.data["documents"].append({
            "source": source,
            "title": title,
            "doc_type": doc_type,
            "topics": topics or [],
            "added_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def record_query(self, query: str, matched_titles: list[str]) -> None:
        self.data["queries"].append({
            "query": query,
            "matched_titles": matched_titles,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def related_past_reads(self, current_titles: list[str], limit: int = 3) -> list[dict]:
        """Naive relatedness: past queries that matched any of the same titles."""
        related = []
        for q in reversed(self.data["queries"]):
            if any(t in q["matched_titles"] for t in current_titles):
                related.append(q)
            if len(related) >= limit:
                break
        return related

    def library_summary(self) -> dict:
        docs = self.data["documents"]
        return {
            "total_documents": len(docs),
            "by_type": {
                t: len([d for d in docs if d["doc_type"] == t])
                for t in set(d["doc_type"] for d in docs)
            } if docs else {},
            "most_recent": docs[-5:] if docs else [],
        }
