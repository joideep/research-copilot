"""Loaders that turn a PDF, URL, or arXiv ID into a plain-text Document.

Each loader returns a `Document` dataclass with the raw text plus metadata
(source, title, date added) that the rest of the pipeline relies on.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Document:
    text: str
    source: str                # file path, URL, or arXiv ID
    title: str
    doc_type: str               # "pdf" | "url" | "arxiv"
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict = field(default_factory=dict)


def load_pdf(path: str) -> Document:
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    full_text = "\n".join(text_parts).strip()

    title = path.split("/")[-1]
    if reader.metadata and reader.metadata.title:
        title = reader.metadata.title

    return Document(text=full_text, source=path, title=title, doc_type="pdf")


def load_url(url: str) -> Document:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "research-copilot/0.1"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else url

    # Prefer <article> content if present, otherwise fall back to <body>.
    article = soup.find("article")
    container = article if article else soup.body
    text = container.get_text(separator="\n").strip() if container else soup.get_text()
    text = re.sub(r"\n{3,}", "\n\n", text)

    return Document(text=text, source=url, title=title, doc_type="url")


def load_arxiv(arxiv_id: str) -> Document:
    """Fetch abstract + metadata from arXiv's API, and download the PDF for full text."""
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    resp = requests.get(api_url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")

    entry = soup.find("entry")
    if entry is None:
        raise ValueError(f"No arXiv entry found for id {arxiv_id}")

    title = entry.find("title").get_text().strip()
    summary = entry.find("summary").get_text().strip()

    pdf_url = None
    for link in entry.find_all("link"):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href")
            break

    full_text = summary
    if pdf_url:
        try:
            pdf_resp = requests.get(pdf_url, timeout=30)
            pdf_resp.raise_for_status()
            tmp_path = f"/tmp/{arxiv_id.replace('/', '_')}.pdf"
            with open(tmp_path, "wb") as f:
                f.write(pdf_resp.content)
            reader = PdfReader(tmp_path)
            full_text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        except Exception:
            # Fall back to the abstract if the PDF download/parse fails.
            full_text = summary

    return Document(
        text=full_text,
        source=f"arxiv:{arxiv_id}",
        title=title,
        doc_type="arxiv",
        extra={"abstract": summary, "pdf_url": pdf_url},
    )
