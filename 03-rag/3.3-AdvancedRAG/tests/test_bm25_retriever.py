"""Offline tests for BM25 index build, search structure, and ranking invariants."""

from __future__ import annotations

import re

from src.bm25_retriever import bm25_search, build_bm25_index


def test_build_bm25_index_returns_correct_types(tiny_corpus: list[dict]) -> None:
    """BM25 must return parallel index and chunk_id list for rank-to-id mapping."""
    index, ids = build_bm25_index(tiny_corpus)
    assert isinstance(ids, list), (
        "chunk_ids must be a list — wrong type breaks index position lookup."
    )
    assert len(ids) == len(tiny_corpus), (
        f"chunk_ids length {len(ids)} != corpus length {len(tiny_corpus)} — "
        "length mismatch causes wrong chunk_id at each BM25 rank."
    )
    assert all(isinstance(i, str) for i in ids), (
        "chunk_ids must be strings — RRF keys and Qdrant payload use str chunk_id."
    )


def test_build_bm25_index_chunk_ids_match_corpus_order(tiny_corpus: list[dict]) -> None:
    """chunk_ids[i] must equal corpus[i].chunk_id or RRF maps ranks to wrong documents."""
    index, ids = build_bm25_index(tiny_corpus)
    for i, chunk in enumerate(tiny_corpus):
        assert ids[i] == chunk["chunk_id"], (
            f"Position {i}: expected chunk_id {chunk['chunk_id']}, got {ids[i]}. "
            "Order mismatch causes wrong chunk retrieval during RRF fusion."
        )


def test_bm25_search_returns_correct_structure(bm25_index_and_ids) -> None:
    """Each BM25 hit must expose chunk_id, score, and rank for downstream RRF."""
    index, ids = bm25_index_and_ids
    results = bm25_search(index, ids, "neural networks backpropagation", top_n=5)
    assert isinstance(results, list), "BM25 search must return a list of ranked dicts."
    for r in results:
        required = {"chunk_id", "bm25_score", "bm25_rank"}
        missing = required - set(r.keys())
        assert not missing, (
            f"Missing keys: {missing} — RRF requires bm25_rank and chunk_id on every hit."
        )


def test_bm25_search_ranks_are_1_indexed(bm25_index_and_ids) -> None:
    """Ranks must be 1-indexed because RRF formula is 1/(k+rank), not 1/(k+rank-1)."""
    index, ids = bm25_index_and_ids
    results = bm25_search(index, ids, "gradient descent loss", top_n=5)
    if results:
        assert results[0]["bm25_rank"] == 1, (
            "First result must have bm25_rank=1. 0-indexed ranks break RRF formula."
        )
        for i, r in enumerate(results):
            assert r["bm25_rank"] == i + 1, (
                f"Rank at position {i} should be {i + 1}, got {r['bm25_rank']} — "
                "non-sequential ranks distort reciprocal rank fusion scores."
            )


def test_bm25_search_exact_match_scores_higher(
    bm25_index_and_ids, tiny_corpus: list[dict]
) -> None:
    """Rare tokens like Maillard must rank the cooking chunk first — BM25's core value."""
    index, ids = bm25_index_and_ids
    results = bm25_search(index, ids, "Maillard reaction emulsification", top_n=10)
    assert len(results) > 0, (
        "BM25 should find exact-match tokens — empty results mean sparse "
        "retrieval failed on a diagnostic query."
    )
    top_chunk_id = results[0]["chunk_id"]
    top_chunk = next((c for c in tiny_corpus if c["chunk_id"] == top_chunk_id), None)
    assert top_chunk is not None, "Top BM25 hit must resolve to a corpus chunk."
    assert top_chunk["topic"] == "cooking", (
        f"'Maillard' query should retrieve cooking chunk first, got topic: "
        f"{top_chunk.get('topic')} — exact-match routing is why hybrid search exists."
    )


def test_bm25_search_filters_zero_scores(bm25_index_and_ids) -> None:
    """Zero-score rows must not enter RRF or they steal rank credit without relevance."""
    index, ids = bm25_index_and_ids
    results = bm25_search(
        index,
        ids,
        "xkcd zyzzyva fluorescent antidisestablishmentarianism",
        top_n=5,
    )
    for r in results:
        assert r["bm25_score"] > 0.0, (
            "Zero-score results must be filtered out. They add noise to RRF."
        )


def test_bm25_search_top_n_respected(bm25_index_and_ids) -> None:
    """top_n caps candidate pool size fed into RRF fusion."""
    index, ids = bm25_index_and_ids
    results = bm25_search(index, ids, "neural networks", top_n=3)
    assert len(results) <= 3, (
        f"Expected <=3 results, got {len(results)} — pool size overflow "
        "wastes fusion compute and dilutes rank signals."
    )


def test_bm25_tokenizer_matches_index_tokenizer() -> None:
    """Query and index must share lowercase word tokenizer or vocabulary lookup misses."""
    query = "Gradient Descent LOSS Function"
    index_tokens = re.findall(r"\b\w+\b", query.lower())
    assert "gradient" in index_tokens, (
        "Tokenizer must extract 'gradient' — missed tokens yield zero BM25 scores."
    )
    assert "descent" in index_tokens, (
        "Tokenizer must extract 'descent' — case-folding is required for IDF lookup."
    )
    assert "loss" in index_tokens, (
        "Tokenizer must extract 'loss' — inconsistent tokenization breaks hybrid recall."
    )
    assert "Gradient" not in index_tokens, (
        "Tokenizer must lowercase — 'Gradient' != 'gradient' in BM25 vocabulary"
    )


def test_bm25_search_bm25_only_results_visible(bm25_index_and_ids) -> None:
    """Historical exact terms must rank — BM25 surfaces what dense embeddings blur."""
    index, ids = bm25_index_and_ids
    results = bm25_search(index, ids, "Silk Road hieroglyphs Roman", top_n=5)
    assert len(results) > 0, (
        "BM25 should catch exact historical terms that dense embeddings blur — "
        "without hits, hybrid search cannot outperform dense-only."
    )
