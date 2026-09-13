"""CLI for ingesting a document without going through the API.

Usage:
    python scripts/ingest_example.py --pdf path/to/file.pdf
    python scripts/ingest_example.py --url https://example.com/article
    python scripts/ingest_example.py --arxiv 2005.14165
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.loaders import load_pdf, load_url, load_arxiv
from app.ingestion.chunker import chunk_document
from app.storage.hybrid_search import HybridSearcher
from app.memory.tracker import MemoryTracker


def main():
    parser = argparse.ArgumentParser(description="Ingest a document into the research library")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to a local PDF file")
    group.add_argument("--url", help="URL of an article to ingest")
    group.add_argument("--arxiv", help="arXiv ID, e.g. 2005.14165")
    args = parser.parse_args()

    if args.pdf:
        doc = load_pdf(args.pdf)
    elif args.url:
        doc = load_url(args.url)
    else:
        doc = load_arxiv(args.arxiv)

    print(f"Loaded: {doc.title} ({len(doc.text)} chars)")

    chunks = chunk_document(doc)
    print(f"Chunked into {len(chunks)} pieces")

    searcher = HybridSearcher()
    searcher.index(chunks)

    memory = MemoryTracker()
    memory.record_document(doc.source, doc.title, doc.doc_type)

    print(f"Indexed '{doc.title}' into the library.")


if __name__ == "__main__":
    sys.exit(main())
