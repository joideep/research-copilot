"""Splits document text into overlapping chunks for embedding.

Uses a simple sentence-aware sliding window rather than a fixed character
cut, so chunks don't split mid-sentence where avoidable.
"""
import re
import uuid

from app.config import settings
from app.ingestion.loaders import Document


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_document(
    doc: Document,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[dict]:
    """Return a list of chunk dicts: {id, text, metadata}."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    sentences = split_sentences(doc.text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)

            # Build overlap: keep trailing sentences whose combined length
            # is <= chunk_overlap, to carry context into the next chunk.
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "id": str(uuid.uuid4()),
            "text": chunk_text,
            "metadata": {
                "source": doc.source,
                "title": doc.title,
                "doc_type": doc.doc_type,
                "added_at": doc.added_at,
                "chunk_index": i,
            },
        })
    return result
