"""Offline tests for RRF fusion — the hybrid rank-combination core."""

from __future__ import annotations

from src.rrf_fusion import RRF_K, fuse_rrf, get_top_k_chunk_ids


def test_rrf_k_constant() -> None:
    """RRF_K=60 is the Cormack et al. 2009 default validated on TREC benchmarks."""
    assert RRF_K == 60, (
        f"RRF_K must be 60 (Cormack et al. 2009), got {RRF_K}. "
        "k=60 is not arbitrary — it is empirically validated across TREC benchmarks."
    )


def test_rrf_k_is_int() -> None:
    """k must be an integer so rank arithmetic stays exact in fusion."""
    assert isinstance(RRF_K, int), (
        f"RRF_K must be int, got {type(RRF_K)} — float k would break rank indexing."
    )


def test_fuse_rrf_returns_correct_structure(
    sample_dense_results: list[dict], sample_bm25_results: list[dict]
) -> None:
    """Fused rows must carry scores, ranks, and system flags for diagnostics and reranking."""
    results = fuse_rrf(sample_dense_results, sample_bm25_results)
    assert isinstance(results, list), "fuse_rrf must return a list of fused dicts."
    for r in results:
        required = {"chunk_id", "rrf_score", "final_rank", "in_dense", "in_bm25"}
        missing = required - set(r.keys())
        assert not missing, (
            f"Missing keys: {missing} — downstream reranker and recall need full metadata."
        )


def test_fuse_rrf_ranks_are_1_indexed(
    sample_dense_results: list[dict], sample_bm25_results: list[dict]
) -> None:
    """final_rank must be 1..N so get_top_k_chunk_ids and recall@5 slice correctly."""
    results = fuse_rrf(sample_dense_results, sample_bm25_results)
    ranks = [r["final_rank"] for r in results]
    assert ranks[0] == 1, (
        f"First result must have final_rank=1, got {ranks[0]} — 0-based ranks break top-k."
    )
    assert ranks == list(range(1, len(results) + 1)), (
        f"Ranks must be sequential 1-indexed: {ranks} — gaps break ordered rerank input."
    )


def test_fuse_rrf_formula_correct() -> None:
    """Rank-1 in both systems must score 2/(k+1) — verifies 1/(k+rank) with 1-indexed ranks."""
    both_rank1_dense = [
        {"chunk_id": "test_doc", "dense_score": 0.9, "dense_rank": 1}
    ]
    both_rank1_bm25 = [
        {"chunk_id": "test_doc", "bm25_score": 8.0, "bm25_rank": 1}
    ]
    results = fuse_rrf(both_rank1_dense, both_rank1_bm25, k=60)
    assert len(results) == 1, "Single doc in both lists must produce one fused row."
    expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    actual_score = results[0]["rrf_score"]
    assert abs(actual_score - expected_score) < 1e-9, (
        f"RRF score wrong: expected {expected_score:.9f}, got {actual_score:.9f}. "
        "Formula must be 1/(k+rank) with 1-indexed ranks, not 1/(k+rank-1)."
    )


def test_fuse_rrf_formula_rank2() -> None:
    """Rank-2 dense-only doc must score 1/(k+2) and rank below rank-1."""
    dense = [
        {"chunk_id": "doc_a", "dense_score": 0.9, "dense_rank": 1},
        {"chunk_id": "doc_b", "dense_score": 0.8, "dense_rank": 2},
    ]
    bm25: list[dict] = []
    results = fuse_rrf(dense, bm25, k=60)
    score_a = next(r["rrf_score"] for r in results if r["chunk_id"] == "doc_a")
    score_b = next(r["rrf_score"] for r in results if r["chunk_id"] == "doc_b")
    assert abs(score_a - 1.0 / 61) < 1e-9, f"Rank-1 score wrong: {score_a}"
    assert abs(score_b - 1.0 / 62) < 1e-9, f"Rank-2 score wrong: {score_b}"
    assert score_a > score_b, (
        "Rank 1 must score higher than rank 2 — inverted ordering would rerank wrong docs."
    )


def test_fuse_rrf_bm25_only_docs_appear_in_output(
    sample_dense_results: list[dict],
    sample_bm25_results: list[dict],
    tiny_chunk_ids: list[str],
) -> None:
    """BM25-only chunk_ids must survive fusion — hybrid search's reason to exist."""
    results = fuse_rrf(sample_dense_results, sample_bm25_results)
    fused_ids = {r["chunk_id"] for r in results}
    bm25_only_ids = {tiny_chunk_ids[1], tiny_chunk_ids[3], tiny_chunk_ids[9]}
    for cid in bm25_only_ids:
        assert cid in fused_ids, (
            f"BM25-only chunk {cid} missing from fused results. "
            "This defeats the purpose of hybrid search — BM25 exists to surface "
            "exact-match documents that dense retrieval misses entirely."
        )


def test_fuse_rrf_in_dense_in_bm25_flags(
    sample_dense_results: list[dict],
    sample_bm25_results: list[dict],
    tiny_chunk_ids: list[str],
) -> None:
    """in_dense/in_bm25 flags must reflect which retriever contributed each chunk."""
    results = fuse_rrf(sample_dense_results, sample_bm25_results)
    result_map = {r["chunk_id"]: r for r in results}
    assert result_map[tiny_chunk_ids[0]]["in_dense"] is True
    assert result_map[tiny_chunk_ids[0]]["in_bm25"] is True
    assert result_map[tiny_chunk_ids[1]]["in_dense"] is False
    assert result_map[tiny_chunk_ids[1]]["in_bm25"] is True
    assert result_map[tiny_chunk_ids[8]]["in_dense"] is True
    assert result_map[tiny_chunk_ids[8]]["in_bm25"] is False


def test_fuse_rrf_consensus_doc_ranks_higher_than_single_system(
    sample_dense_results: list[dict],
    sample_bm25_results: list[dict],
    tiny_chunk_ids: list[str],
) -> None:
    """Docs ranked highly in both systems must outscore single-system hits."""
    results = fuse_rrf(sample_dense_results, sample_bm25_results)
    result_map = {r["chunk_id"]: r for r in results}
    score_consensus = result_map[tiny_chunk_ids[0]]["rrf_score"]
    score_bm25_only = result_map[tiny_chunk_ids[1]]["rrf_score"]
    assert score_consensus > score_bm25_only, (
        f"Consensus doc ({score_consensus:.6f}) must score higher than "
        f"single-system doc ({score_bm25_only:.6f}). RRF's core invariant."
    )


def test_fuse_rrf_empty_dense(sample_bm25_results: list[dict]) -> None:
    """With no dense hits, fusion must still emit all BM25 candidates."""
    results = fuse_rrf([], sample_bm25_results)
    assert isinstance(results, list), "Empty dense must still return a list."
    assert len(results) == len(sample_bm25_results), (
        "Empty dense: should fall back to BM25 results only"
    )
    for r in results:
        assert r["in_dense"] is False
        assert r["in_bm25"] is True


def test_fuse_rrf_empty_bm25(sample_dense_results: list[dict]) -> None:
    """With no BM25 hits, fusion must still emit all dense candidates."""
    results = fuse_rrf(sample_dense_results, [])
    assert len(results) == len(sample_dense_results), (
        "Empty BM25: fused list must match dense candidate count."
    )
    for r in results:
        assert r["in_dense"] is True
        assert r["in_bm25"] is False


def test_fuse_rrf_both_empty() -> None:
    """Both empty inputs must yield [] without error — OOV queries must not crash."""
    results = fuse_rrf([], [])
    assert results == [], "Both empty: must return empty list, not crash"


def test_fuse_rrf_sorted_descending(
    sample_dense_results: list[dict], sample_bm25_results: list[dict]
) -> None:
    """Results must be descending by rrf_score so reranker reads true priority order."""
    results = fuse_rrf(sample_dense_results, sample_bm25_results)
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        "Results must be sorted by RRF score descending — unsorted lists "
        "send wrong candidates to the cross-encoder."
    )


def test_fuse_rrf_k_60_vs_k_0_difference() -> None:
    """k=60 flattens rank cliffs versus k=0 — validates smoothing constant choice."""
    dense_1 = [{"chunk_id": "a", "dense_score": 0.9, "dense_rank": 1}]
    dense_2 = [{"chunk_id": "a", "dense_score": 0.9, "dense_rank": 2}]
    score_rank1_k60 = fuse_rrf(dense_1, [], k=60)[0]["rrf_score"]
    score_rank2_k60 = fuse_rrf(dense_2, [], k=60)[0]["rrf_score"]
    ratio_k60 = score_rank1_k60 / score_rank2_k60
    assert ratio_k60 < 1.02, (
        f"k=60 should make rank-1 and rank-2 nearly equal (ratio < 1.02), "
        f"got {ratio_k60:.4f} — wrong k over-weights top ranks."
    )
    score_rank1_k0 = 1.0 / (0 + 1)
    score_rank2_k0 = 1.0 / (0 + 2)
    assert score_rank1_k0 / score_rank2_k0 == 2.0, "k=0 sanity check"


def test_get_top_k_chunk_ids(
    sample_dense_results: list[dict], sample_bm25_results: list[dict]
) -> None:
    """get_top_k_chunk_ids must return the highest final_rank chunk_ids for recall@k."""
    results = fuse_rrf(sample_dense_results, sample_bm25_results)
    top3 = get_top_k_chunk_ids(results, k=3)
    assert len(top3) == 3, "k=3 must return exactly three chunk_ids when enough results exist."
    assert all(isinstance(i, str) for i in top3), (
        "chunk_ids must be strings to match Qdrant payload and ground truth."
    )
    assert top3[0] == results[0]["chunk_id"], (
        "First ID must be top-ranked — wrong slice breaks recall@5 measurement."
    )


def test_get_top_k_handles_k_larger_than_results(sample_dense_results: list[dict]) -> None:
    """k larger than fused list must return all available ids without raising."""
    results = fuse_rrf(sample_dense_results, [])
    top20 = get_top_k_chunk_ids(results, k=20)
    assert len(top20) == len(results), (
        "k > len(results): return all available, don't crash"
    )
