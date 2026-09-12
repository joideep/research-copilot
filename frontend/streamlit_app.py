"""Streamlit UI for the research copilot.

Run with: streamlit run frontend/streamlit_app.py
Talks directly to the agent/search layer (not through the API) to keep the
demo self-contained — swap in requests to the FastAPI backend if you want
this to run against a separately deployed service.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from app.ingestion.loaders import load_pdf, load_url, load_arxiv
from app.ingestion.chunker import chunk_document
from app.storage.hybrid_search import HybridSearcher
from app.memory.tracker import MemoryTracker
from app.agent.agent import ResearchAgent

st.set_page_config(page_title="Research Copilot", page_icon="📚", layout="wide")


@st.cache_resource
def get_components():
    return HybridSearcher(), MemoryTracker(), ResearchAgent()


searcher, memory, agent = get_components()

st.title("📚 Research Copilot")

tab_chat, tab_ingest, tab_library = st.tabs(["Chat", "Ingest", "Library"])

with tab_ingest:
    st.subheader("Add to your library")
    ingest_type = st.radio("Source type", ["PDF upload", "URL", "arXiv ID"], horizontal=True)

    if ingest_type == "PDF upload":
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded and st.button("Ingest PDF"):
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("Loading and chunking..."):
                doc = load_pdf(tmp_path)
                chunks = chunk_document(doc)
                searcher.index(chunks)
                memory.record_document(doc.source, doc.title, doc.doc_type)
            st.success(f"Indexed '{doc.title}' ({len(chunks)} chunks)")

    elif ingest_type == "URL":
        url = st.text_input("Article URL")
        if url and st.button("Ingest URL"):
            with st.spinner("Fetching and chunking..."):
                doc = load_url(url)
                chunks = chunk_document(doc)
                searcher.index(chunks)
                memory.record_document(doc.source, doc.title, doc.doc_type)
            st.success(f"Indexed '{doc.title}' ({len(chunks)} chunks)")

    else:
        arxiv_id = st.text_input("arXiv ID (e.g. 2005.14165)")
        if arxiv_id and st.button("Ingest arXiv paper"):
            with st.spinner("Fetching paper and chunking..."):
                doc = load_arxiv(arxiv_id)
                chunks = chunk_document(doc)
                searcher.index(chunks)
                memory.record_document(doc.source, doc.title, doc.doc_type)
            st.success(f"Indexed '{doc.title}' ({len(chunks)} chunks)")

with tab_library:
    st.subheader("Library summary")
    summary = memory.library_summary()
    col1, col2 = st.columns(2)
    col1.metric("Total documents", summary["total_documents"])
    col2.json(summary["by_type"])
    st.write("Most recently added:")
    for doc in reversed(summary["most_recent"]):
        st.write(f"- **{doc['title']}** ({doc['doc_type']}) — {doc['added_at'][:10]}")

with tab_chat:
    st.subheader("Ask your research copilot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question about your library..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = agent.run(prompt)
            st.write(result["answer"])
            if result["tool_calls"]:
                with st.expander("Tool calls made"):
                    st.json(result["tool_calls"])

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
