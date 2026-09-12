"""FastAPI app exposing the research copilot as an API.

Endpoints:
  POST /ingest/pdf    - ingest an uploaded PDF
  POST /ingest/url     - ingest a web article
  POST /ingest/arxiv    - ingest a paper by arXiv ID
  POST /query           - ask the agent a question
  GET  /library/summary  - summary of what's ingested so far
"""
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingestion.loaders import load_pdf, load_url, load_arxiv
from app.ingestion.chunker import chunk_document
from app.storage.hybrid_search import HybridSearcher
from app.memory.tracker import MemoryTracker
from app.agent.agent import ResearchAgent

app = FastAPI(title="Research Copilot API")

searcher = HybridSearcher()
memory = MemoryTracker()
agent = ResearchAgent()


class UrlIngestRequest(BaseModel):
    url: str


class ArxivIngestRequest(BaseModel):
    arxiv_id: str


class QueryRequest(BaseModel):
    message: str


def _ingest_and_index(doc) -> dict:
    chunks = chunk_document(doc)
    searcher.index(chunks)
    memory.record_document(doc.source, doc.title, doc.doc_type)
    return {"title": doc.title, "source": doc.source, "chunks_indexed": len(chunks)}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        doc = load_pdf(tmp_path)
        return _ingest_and_index(doc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/url")
async def ingest_url(req: UrlIngestRequest):
    try:
        doc = load_url(req.url)
        return _ingest_and_index(doc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/arxiv")
async def ingest_arxiv(req: ArxivIngestRequest):
    try:
        doc = load_arxiv(req.arxiv_id)
        return _ingest_and_index(doc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
async def query(req: QueryRequest):
    try:
        result = agent.run(req.message)
        return {"answer": result["answer"], "tool_calls": result["tool_calls"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/library/summary")
async def library_summary():
    return memory.library_summary()


@app.get("/health")
async def health():
    return {"status": "ok", "documents_indexed": searcher.vector_store.count()}
