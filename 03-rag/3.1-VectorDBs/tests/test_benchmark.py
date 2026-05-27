"""Offline recall@k tests and integration benchmark pipeline tests."""

from __future__ import annotations

import pytest

from src.benchmark import compute_recall_at_k, run_benchmark, run_filtered_benchmark
from src.corpus import get_query_set
from src.embedder import embed_texts

_EXPECTED_TOPICS = {
    "machine_learning",
    "ocean_biology",
    "ancient_history",
    "cooking",
    "urban_architecture",
}
_BENCHMARK_ROW_KEYS = {
    "ef_search",
    "recall@5",
    "ann_p50_ms",
    "ann_p95_ms",
    "exact_p50_ms",
    "exact_p95_ms",
    "queries_run",
}


def test_recall_perfect():
    """Verify perfect overlap yields recall@k == 1.0."""
    ann = [1, 2, 3, 4, 5]
    gt = [1, 2, 3, 4, 5]
    assert compute_recall_at_k(ann, gt, k=5) == 1.0, (
        "expected recall 1.0 because all five IDs match"
    )


def test_recall_zero():
    """Verify no overlap yields recall@k == 0.0."""
    ann = [6, 7, 8, 9, 10]
    gt = [1, 2, 3, 4, 5]
    assert compute_recall_at_k(ann, gt, k=5) == 0.0, (
        "expected recall 0.0 because no IDs overlap"
    )


def test_recall_partial():
    """Verify partial overlap yields the correct recall fraction."""
    ann = [1, 2, 6, 7, 8]
    gt = [1, 2, 3, 4, 5]
    result = compute_recall_at_k(ann, gt, k=5)
    assert abs(result - 0.4) < 1e-9, (
        "expected recall 0.4 because 2 of 5 ground-truth IDs appear in ANN top-5"
    )


def test_recall_order_independent():
    """Verify recall@k ignores rank order and only checks set membership."""
    ann = [5, 4, 3, 2, 1]
    gt = [1, 2, 3, 4, 5]
    assert compute_recall_at_k(ann, gt, k=5) == 1.0, (
        "expected recall 1.0 because all five IDs are present regardless of order"
    )


def test_recall_k_cutoff_respected():
    """Verify only the first k IDs from each list participate in recall@k."""
    ann = [1, 2, 3, 4, 99]
    gt = [1, 2, 3, 4, 5]
    result = compute_recall_at_k(ann, gt, k=4)
    assert result == 1.0, (
        "expected recall 1.0 at k=4 because ann[:4] and gt[:4] are identical sets"
    )


def test_recall_returns_float():
    """Verify compute_recall_at_k always returns a Python float."""
    assert isinstance(compute_recall_at_k([1], [1], k=1), float), (
        "expected float return type because recall is a fractional metric"
    )


@pytest.mark.integration
def test_run_benchmark_returns_correct_structure(populated_collection, small_corpus):
    """Verify run_benchmark returns one row per ef_search with required metrics."""
    client, collection_name = populated_collection
    queries = get_query_set(small_corpus, n_queries=5, seed=99)
    q_embeds = embed_texts([q["query_text"] for q in queries], input_type="query")
    results = run_benchmark(
        client,
        queries,
        q_embeds,
        ef_search_values=[64],
        collection_name=collection_name,
    )
    assert len(results) == 1, "expected one row because ef_search_values has one entry"
    row = results[0]
    assert set(row.keys()) == _BENCHMARK_ROW_KEYS, (
        f"expected keys {_BENCHMARK_ROW_KEYS} because benchmark schema is fixed"
    )
    assert row["ef_search"] == 64, "expected ef_search 64 because that was requested"
    assert row["queries_run"] == 5, "expected 5 queries because n_queries=5"
    assert 0.0 <= row["recall@5"] <= 1.0, (
        "expected recall in [0,1] because it is a fraction of matched IDs"
    )
    assert row["ann_p50_ms"] > 0, "expected positive ANN latency because search ran"
    assert row["ann_p95_ms"] >= row["ann_p50_ms"], (
        "expected p95 >= p50 because p95 is a higher percentile"
    )
    assert row["exact_p50_ms"] > 0, (
        "expected positive exact latency because ground-truth search ran"
    )


@pytest.mark.integration
def test_run_benchmark_recall_improves_with_ef(populated_collection, small_corpus):
    """Verify higher ef_search does not materially lower recall (non-decreasing)."""
    client, collection_name = populated_collection
    queries = get_query_set(small_corpus, n_queries=5, seed=99)
    q_embeds = embed_texts([q["query_text"] for q in queries], input_type="query")
    results = run_benchmark(
        client,
        queries,
        q_embeds,
        ef_search_values=[16, 256],
        collection_name=collection_name,
    )
    assert len(results) == 2, "expected two rows because two ef_search values were run"
    recall_16 = results[0]["recall@5"]
    recall_256 = results[1]["recall@5"]
    assert recall_256 >= recall_16 - 0.01, (
        "expected recall at ef=256 >= recall at ef=16 - 0.01 because higher ef "
        "should not reduce recall on a small corpus"
    )


@pytest.mark.integration
def test_run_filtered_benchmark_returns_all_topics(populated_collection, small_corpus):
    """Verify filtered benchmark emits one result row per topic in the query set."""
    client, collection_name = populated_collection
    queries = get_query_set(small_corpus, n_queries=5, seed=99)
    q_embeds = embed_texts([q["query_text"] for q in queries], input_type="query")
    results = run_filtered_benchmark(
        client,
        queries,
        q_embeds,
        ef_search=64,
        collection_name=collection_name,
    )
    returned_topics = {r["topic"] for r in results}
    assert returned_topics == _EXPECTED_TOPICS, (
        f"expected all five topics because filtered benchmark groups by topic"
    )


@pytest.mark.integration
def test_run_filtered_benchmark_recall_in_range(populated_collection, small_corpus):
    """Verify filtered and unfiltered recall values are valid fractions."""
    client, collection_name = populated_collection
    queries = get_query_set(small_corpus, n_queries=5, seed=99)
    q_embeds = embed_texts([q["query_text"] for q in queries], input_type="query")
    results = run_filtered_benchmark(
        client,
        queries,
        q_embeds,
        ef_search=64,
        collection_name=collection_name,
    )
    for r in results:
        assert 0.0 <= r["recall_unfiltered"] <= 1.0, (
            f"expected recall_unfiltered in [0,1] for topic {r['topic']}"
        )
        assert 0.0 <= r["recall_filtered"] <= 1.0, (
            f"expected recall_filtered in [0,1] for topic {r['topic']}"
        )
