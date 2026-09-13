"""Tools available to the agent. Each tool is a plain Python function plus a
JSON-schema definition Claude uses to decide when and how to call it.

Kept separate from agent.py so tools can be tested or extended independently.
"""
from app.storage.hybrid_search import HybridSearcher
from app.memory.tracker import MemoryTracker

_searcher: HybridSearcher | None = None
_memory: MemoryTracker | None = None


def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


def get_memory() -> MemoryTracker:
    global _memory
    if _memory is None:
        _memory = MemoryTracker()
    return _memory


def search_library(query: str, top_k: int = 6) -> dict:
    """Search the user's ingested library with hybrid dense+sparse search."""
    top_k = int(top_k)
    hits = get_searcher().search(query, top_k=top_k)
    titles = list({h["metadata"]["title"] for h in hits})
    get_memory().record_query(query, titles)
    related = get_memory().related_past_reads(titles)

    return {
        "results": [
            {
                "title": h["metadata"]["title"],
                "source": h["metadata"]["source"],
                "text": h["text"],
                "score": round(h["final_score"], 3),
            }
            for h in hits
        ],
        "related_past_queries": [r["query"] for r in related],
    }


def search_web(query: str) -> dict:
    """Search the live web for recent papers/articles not in the local library.

    NOTE: this is a stub — plug in a real search API (e.g. Tavily, Serper,
    or the arXiv API for paper-specific search) before relying on this in
    production. Returns a clear message instead of raising so the agent can
    tell the user it isn't wired up yet, rather than crashing.
    """
    return {
        "error": (
            "Web search isn't connected yet. This tool is a placeholder — "
            "hook it up to a real search provider (e.g. Tavily, Serper, or "
            "the arXiv API) to enable it."
        )
    }


def library_summary() -> dict:
    """Return a summary of what's currently in the library."""
    return get_memory().library_summary()


# JSON schemas Claude uses to know what each tool does and what arguments it takes.
TOOL_DEFINITIONS = [
    {
        "name": "search_library",
        "description": (
            "Search the user's personal research library (ingested papers, "
            "articles, notes) using hybrid semantic + keyword search. Use this "
            "first for any question about content the user has already saved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "top_k": {"type": "integer", "description": "Number of results to return", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the live web for recent papers or articles not in the "
            "user's local library. Use when the library doesn't have relevant "
            "results or the user asks for something recent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "library_summary",
        "description": "Get a summary of what's currently in the user's library (counts, types, recent additions).",
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_IMPLEMENTATIONS = {
    "search_library": lambda **kwargs: search_library(**kwargs),
    "search_web": lambda **kwargs: search_web(**kwargs),
    "library_summary": lambda **kwargs: library_summary(**kwargs),
}
