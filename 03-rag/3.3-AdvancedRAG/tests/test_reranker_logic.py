"""Offline tests for reranker constants and build_rerank_input (no Cohere calls)."""

from __future__ import annotations

import os

import pytest

from src.reranker import (
    CANDIDATE_POOL,
    RERANK_MODEL,
    RERANK_TOP_N,
    build_rerank_input,
    rerank,
)


def test_reranker_constants() -> None:
    """Pool and top-N bounds control cross-encoder cost and LLM context pollution."""
    assert RERANK_TOP_N == 5, (
        f"RERANK_TOP_N must be 5 (topK pathology — >5 floods LLM context), "
        f"got {RERANK_TOP_N}"
    )
    assert CANDIDATE_POOL == 20, (
        f"CANDIDATE_POOL must be 20 (cross-encoder input size), got {CANDIDATE_POOL}"
    )
    assert "rerank" in RERANK_MODEL.lower(), (
        f"RERANK_MODEL should be a Cohere rerank model, got {RERANK_MODEL}"
    )


def test_rerank_top_n_not_greater_than_candidate_pool() -> None:
    """Final top-N cannot exceed the fused pool fed to the cross-encoder."""
    assert RERANK_TOP_N < CANDIDATE_POOL, (
        f"RERANK_TOP_N ({RERANK_TOP_N}) must be < CANDIDATE_POOL ({CANDIDATE_POOL}) — "
        "otherwise reranking cannot narrow the candidate set."
    )


def test_build_rerank_input_structure(tiny_corpus: list[dict]) -> None:
    """build_rerank_input must yield (chunk_id, text) pairs for the Cohere API."""
    fused = [
        {
            "chunk_id": c["chunk_id"],
            "rrf_score": 0.03,
            "final_rank": i + 1,
            "in_dense": True,
            "in_bm25": False,
        }
        for i, c in enumerate(tiny_corpus[:5])
    ]
    candidates = build_rerank_input(fused, tiny_corpus, top_candidate=5)
    assert isinstance(candidates, list), "Candidates must be a list of tuples."
    assert len(candidates) == 5, "top_candidate=5 must return five (id, text) pairs."
    for chunk_id, text in candidates:
        assert isinstance(chunk_id, str), "chunk_id must be str for API document keys."
        assert isinstance(text, str), "text must be str for cross-encoder input."
        assert len(text) > 0, "Empty text would make reranking meaningless."


def test_build_rerank_input_returns_correct_texts(tiny_corpus: list[dict]) -> None:
    """Resolved text must match corpus — wrong text sends the reranker the wrong document."""
    fused = [
        {
            "chunk_id": tiny_corpus[0]["chunk_id"],
            "rrf_score": 0.03,
            "final_rank": 1,
            "in_dense": True,
            "in_bm25": False,
        }
    ]
    candidates = build_rerank_input(fused, tiny_corpus, top_candidate=1)
    assert candidates[0][0] == tiny_corpus[0]["chunk_id"], (
        "chunk_id must match fused result — ID mismatch breaks recall attribution."
    )
    assert candidates[0][1] == tiny_corpus[0]["text"], (
        "text must match corpus chunk — wrong body yields false rerank scores."
    )


def test_build_rerank_input_top_candidate_respected(tiny_corpus: list[dict]) -> None:
    """top_candidate caps how many fused rows are passed to the cross-encoder."""
    fused = [
        {
            "chunk_id": c["chunk_id"],
            "rrf_score": 0.03,
            "final_rank": i + 1,
            "in_dense": True,
            "in_bm25": False,
        }
        for i, c in enumerate(tiny_corpus)
    ]
    candidates = build_rerank_input(fused, tiny_corpus, top_candidate=3)
    assert len(candidates) == 3, (
        f"top_candidate=3: expected 3 candidates, got {len(candidates)} — "
        "pool size controls reranker latency and cost."
    )


def test_build_rerank_input_missing_chunk_id_raises(tiny_corpus: list[dict]) -> None:
    """Unknown chunk_id must raise — silent skip would hide corpus/Qdrant desync."""
    fused = [
        {
            "chunk_id": "nonexistent-uuid-0000",
            "rrf_score": 0.03,
            "final_rank": 1,
            "in_dense": True,
            "in_bm25": False,
        }
    ]
    with pytest.raises(KeyError):
        build_rerank_input(fused, tiny_corpus, top_candidate=1)


def test_rerank_empty_key_returns_empty(tiny_corpus: list[dict]) -> None:
    """Missing COHERE_API_KEY must return [] so the pipeline falls back to RRF top-N."""
    import src.reranker as reranker_mod

    original = os.environ.pop("COHERE_API_KEY", None)
    reranker_mod._cohere_client = None
    reranker_mod._cohere_warned = False
    try:
        candidates = [(c["chunk_id"], c["text"]) for c in tiny_corpus[:3]]
        result = rerank("test query", candidates, top_n=3)
        assert result == [], (
            "Empty COHERE_API_KEY must return [] — reranker unavailable, not a crash"
        )
    finally:
        reranker_mod._cohere_client = None
        reranker_mod._cohere_warned = False
        if original:
            os.environ["COHERE_API_KEY"] = original


def test_finalize_with_dense_backfill_preserves_dense_recall() -> None:
    """Final top-k must stay within high-ef dense top-5 regardless of rerank order."""
    from src.reranker import finalize_with_dense_backfill

    dense_hits = [{"chunk_id": str(i)} for i in range(10)]
    # Cohere promotes ids outside dense top-5 — they must be ignored.
    reranked_ids = ["99", "98", "0", "1"]
    final = finalize_with_dense_backfill(
        reranked_ids, dense_hits, k=5, precision_slots=2
    )
    assert final == ["0", "1", "2", "3", "4"], (
        "Only dense top-5 ids may appear; cohere picks outside that set are dropped."
    )


def test_finalize_with_dense_backfill_reorders_within_dense_top() -> None:
    """Cohere may swap order among dense top-5 for the precision slots."""
    from src.reranker import finalize_with_dense_backfill

    dense_hits = [{"chunk_id": str(i)} for i in range(10)]
    final = finalize_with_dense_backfill(
        ["3", "1"], dense_hits, k=5, precision_slots=2
    )
    assert final == ["3", "1", "0", "2", "4"]


def test_build_rerank_input_with_dense_anchor_prepends_dense(
    tiny_corpus: list[dict],
) -> None:
    """Dense anchor ensures exact neighbours are always in the rerank pool."""
    from src.reranker import build_rerank_input_with_dense_anchor

    dense_hits = [
        {"chunk_id": tiny_corpus[i]["chunk_id"], "score": 0.9 - i * 0.01}
        for i in range(5)
    ]
    fused = [
        {
            "chunk_id": tiny_corpus[9]["chunk_id"],
            "rrf_score": 0.05,
            "final_rank": 1,
            "in_dense": True,
            "in_bm25": True,
        }
    ]
    pairs = build_rerank_input_with_dense_anchor(
        fused, dense_hits, tiny_corpus, top_candidate=3, dense_anchor=2
    )
    assert pairs[0][0] == dense_hits[0]["chunk_id"]
    assert pairs[1][0] == dense_hits[1]["chunk_id"]
    assert len(pairs) == 3


def test_rate_limit_backoff_never_zero_when_interval_disabled() -> None:
    """429 backoff must stay >= min even when proactive throttle is disabled."""
    import src.reranker as reranker_mod

    original_interval = reranker_mod._COHERE_MIN_INTERVAL_SEC
    original_min = reranker_mod._COHERE_429_MIN_BACKOFF_SEC
    try:
        reranker_mod._COHERE_MIN_INTERVAL_SEC = 0.0
        reranker_mod._COHERE_429_MIN_BACKOFF_SEC = 6.5
        assert reranker_mod._rate_limit_backoff_sec(0) == 6.5
        assert reranker_mod._rate_limit_backoff_sec(1) == 13.0
    finally:
        reranker_mod._COHERE_MIN_INTERVAL_SEC = original_interval
        reranker_mod._COHERE_429_MIN_BACKOFF_SEC = original_min


def test_extend_cohere_cooldown_blocks_until_elapsed(monkeypatch) -> None:
    """Cooldown from a 429 must delay the next throttle wait."""
    import src.reranker as reranker_mod

    times = iter([100.0, 100.0, 106.0, 106.0])
    monkeypatch.setattr(reranker_mod.time, "monotonic", lambda: next(times))
    sleeps: list[float] = []
    monkeypatch.setattr(
        reranker_mod.time, "sleep", lambda s: sleeps.append(float(s))
    )

    reranker_mod._cohere_cooldown_until = 0.0
    reranker_mod._extend_cohere_cooldown(6.0)
    reranker_mod._wait_cohere_cooldown()

    assert sleeps == [6.0], "Must sleep until cooldown expires after a 429."


def test_rerank_returns_correct_structure_when_available(tiny_corpus: list[dict]) -> None:
    """Live Cohere rerank must return scored rows with ranks when API key is set."""
    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not set — skipping live reranker test")
    candidates = [(c["chunk_id"], c["text"]) for c in tiny_corpus[:5]]
    results = rerank("neural networks gradient descent", candidates, top_n=3)
    assert len(results) <= 3, "top_n=3 must cap reranker output length."
    for r in results:
        required = {"chunk_id", "rerank_score", "rerank_rank", "original_rrf_rank"}
        missing = required - set(r.keys())
        assert not missing, (
            f"Missing keys: {missing} — benchmark needs rerank scores for ablation."
        )
        assert 0.0 <= r["rerank_score"] <= 1.0, (
            "rerank_score must be in [0,1] — out-of-range scores break reporting."
        )
        assert r["rerank_rank"] >= 1, (
            "rerank_rank must be 1-indexed for final top-k selection."
        )
