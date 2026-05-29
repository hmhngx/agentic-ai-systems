"""Offline tests for corpus_bridge — corpus/query determinism and schema."""

from __future__ import annotations

from collections import Counter

from src.corpus_bridge import get_corpus_and_queries


def test_get_corpus_and_queries_returns_correct_counts() -> None:
    """Corpus and query set sizes must match Day 3 seeds for valid recall@5."""
    corpus, queries = get_corpus_and_queries()
    assert len(corpus) == 500, (
        "Corpus must have exactly 500 chunks (seed=42) — wrong size breaks "
        "BM25/Qdrant alignment and invalidates the benchmark."
    )
    assert len(queries) == 20, (
        "Query set must have exactly 20 queries (seed=99) — stratified "
        "recall needs 4 queries per topic."
    )


def test_corpus_is_deterministic() -> None:
    """Repeated loads must yield identical text so offline and CI runs are comparable."""
    c1, q1 = get_corpus_and_queries()
    c2, q2 = get_corpus_and_queries()
    assert c1[0]["text"] == c2[0]["text"], (
        "Corpus must be deterministic (same seed) — non-deterministic corpora "
        "make recall@5 irreproducible across machines."
    )
    assert q1[0]["query_text"] == q2[0]["query_text"], (
        "Queries must be deterministic (seed=99) — query drift changes which "
        "chunks ground truth expects."
    )


def test_corpus_has_required_keys() -> None:
    """Every chunk must expose fields needed for BM25, RRF, and Qdrant payload lookup."""
    corpus, _ = get_corpus_and_queries()
    required = {
        "chunk_id",
        "text",
        "topic",
        "source",
        "page",
        "chunk_index",
        "id",
    }
    for chunk in corpus[:10]:
        missing = required - set(chunk.keys())
        assert not missing, (
            f"Missing keys: {missing} — incomplete chunks break chunk_id lookup "
            "during RRF fusion and reranker text resolution."
        )


def test_corpus_topic_distribution() -> None:
    """Even 100-chunk-per-topic split ensures BM25 and dense see all domains equally."""
    corpus, _ = get_corpus_and_queries()
    dist = Counter(c["topic"] for c in corpus)
    assert len(dist) == 5, (
        f"Expected 5 topics, got {len(dist)}: {dict(dist)} — uneven topics "
        "skew recall and hide per-domain retrieval failures."
    )
    assert all(v == 100 for v in dist.values()), (
        f"Uneven topic distribution: {dict(dist)} — imbalanced corpora let one "
        "topic dominate aggregate recall."
    )


def test_query_topic_distribution() -> None:
    """Stratified queries (4 per topic) exercise each domain in the ablation table."""
    _, queries = get_corpus_and_queries()
    dist = Counter(q["topic"] for q in queries)
    assert all(v == 4 for v in dist.values()), (
        f"Expected 4 queries per topic, got: {dict(dist)} — missing topics "
        "leave retrieval gaps undetected in the benchmark."
    )


def test_query_text_is_from_corpus() -> None:
    """Query fragments must come from source chunks so recall@5 measures real retrieval."""
    corpus, queries = get_corpus_and_queries()
    corpus_by_id = {str(c["id"]): c for c in corpus}
    for q in queries:
        source_chunk = corpus_by_id.get(str(q["source_chunk_id"]))
        if source_chunk:
            assert q["query_text"] in source_chunk["text"], (
                f"Query text not found in source chunk {q['source_chunk_id']} — "
                "synthetic queries must be in-corpus fragments or recall is meaningless."
            )
