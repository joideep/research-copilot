"""Small retrieval eval harness.

The point isn't statistical rigor with two example queries — it's having a
*repeatable* check you run after every change to chunking, embeddings, or
scoring weights, so you know if you made retrieval better or worse instead
of guessing from a couple of manual tries.

Edit test_queries.json to match documents actually in your library, and add
to it as you go. Each entry checks whether any of the top results' source
strings contain an expected substring — a cheap proxy for "did it find the
right document," not full relevance judgment.
"""
import json
import os

from app.storage.hybrid_search import HybridSearcher

TEST_QUERIES_PATH = os.path.join(os.path.dirname(__file__), "test_queries.json")


def load_test_queries() -> list[dict]:
    with open(TEST_QUERIES_PATH, "r") as f:
        return json.load(f)


def run_eval(top_k: int = 5) -> dict:
    searcher = HybridSearcher()
    test_queries = load_test_queries()

    results = []
    hits = 0

    for tq in test_queries:
        query = tq["query"]
        expected = tq["expected_source_contains"].lower()

        search_results = searcher.search(query, top_k=top_k)
        found = any(
            expected in r["metadata"]["source"].lower()
            or expected in r["metadata"]["title"].lower()
            or expected in r["text"].lower()
            for r in search_results
        )
        if found:
            hits += 1

        results.append({
            "query": query,
            "expected_contains": expected,
            "found": found,
            "top_result_titles": [r["metadata"]["title"] for r in search_results[:3]],
        })

    total = len(test_queries)
    summary = {
        "total_queries": total,
        "hits": hits,
        "hit_rate": round(hits / total, 3) if total else 0.0,
        "details": results,
    }
    return summary


if __name__ == "__main__":
    summary = run_eval()
    print(json.dumps(summary, indent=2))
    print(f"\nHit rate: {summary['hit_rate']:.1%} ({summary['hits']}/{summary['total_queries']})")
