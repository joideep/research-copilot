# Research Copilot

A personal research assistant that ingests papers, articles, and notes, lets you
query across your library with hybrid (semantic + keyword) search, and uses an
agent with tools to actually do research work — not just answer questions about
one document at a time.

## Why this exists

Most "chat with your PDFs" demos stop at basic RAG. This project goes further:

- **Hybrid search** (dense embeddings + BM25) instead of pure vector similarity,
  which misses exact terms, names, and numbers.
- **An agent with tools**, not a single prompt — it can search your library,
  search the live web for recent papers, summarize, and compare sources.
- **A lightweight memory layer** that tracks what you've read and when, so the
  assistant can say "you read something related to this three weeks ago."
- **An eval harness** with a small fixed set of test queries, so retrieval
  quality is measured over time instead of eyeballed.

## Architecture

```
                       ┌─────────────────┐
   PDFs / URLs /       │   Ingestion      │
   arXiv IDs   ───────▶│ (loaders +       │
                        │  chunker)        │
                        └────────┬─────────┘
                                 │ chunks
                                 ▼
                        ┌──────────────────┐
                        │  Storage layer    │
                        │  - Chroma (dense) │
                        │  - BM25 (sparse)  │
                        └────────┬──────────┘
                                 │ hybrid_search()
                                 ▼
   User query ──────▶   ┌──────────────────┐      ┌───────────────┐
                        │   Agent           │─────▶│ Memory tracker │
                        │  (tool-calling    │      └───────────────┘
                        │   loop, Claude)   │
                        └────────┬──────────┘
                                 │
                        ┌────────┴──────────┐
                        │  Tools:            │
                        │  - search_library   │
                        │  - search_web        │
                        │  - summarize_paper   │
                        │  - compare_sources    │
                        └───────────────────┘
```

FastAPI exposes this as an API; a Streamlit app gives you a usable UI on top.

## Setup

```bash
git clone <your-repo-url>
cd research-copilot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then fill in your ANTHROPIC_API_KEY
```

### Ingest something

```bash
python scripts/ingest_example.py --arxiv 2005.14165   # a paper by arXiv ID
python scripts/ingest_example.py --pdf path/to/file.pdf
python scripts/ingest_example.py --url https://example.com/article
```

### Run the API

```bash
uvicorn app.api.main:app --reload
```

### Run the UI

```bash
streamlit run frontend/streamlit_app.py
```

### Run the eval harness

```bash
python -m app.eval.eval_harness
```

## Project layout

```
app/
  ingestion/      loaders.py (PDF, URL, arXiv), chunker.py
  storage/         vector_store.py (Chroma), bm25_store.py, hybrid_search.py
  agent/           tools.py, agent.py (tool-calling loop)
  memory/          tracker.py (reading history, topic tags)
  api/             main.py (FastAPI app)
  eval/            eval_harness.py, test_queries.json
frontend/
  streamlit_app.py
scripts/
  ingest_example.py
```

## Roadmap / ideas to extend

- Swap Chroma for Qdrant if you outgrow local storage
- Add a real knowledge graph (e.g. topic co-occurrence) instead of flat tags
- Scheduled ingestion from an RSS feed of a field you follow
- Multi-user support if you want to share it

## Tech stack

Python, FastAPI, Streamlit, ChromaDB, `rank_bm25`, sentence-transformers
(local embeddings, no API cost), Anthropic API for generation and tool use.
