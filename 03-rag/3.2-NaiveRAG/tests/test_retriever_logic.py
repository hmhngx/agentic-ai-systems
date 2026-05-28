"""Offline tests for ``src/retriever.py`` business logic.

The retriever sits between Qdrant and the LLM and applies three RAG-
specific rules:

  1. Drop any candidate with score < MIN_SCORE_THRESHOLD (default 0.25).
  2. Drop near-duplicate chunks (Jaccard > 0.95) on the same page.
  3. Return status='NO_RESULTS' when the filtered list is empty.

These rules are correctness-critical: skipping (1) lets noise reach
the LLM, skipping (2) wastes prompt tokens on duplicate context,
skipping (3) is THE main RAG hallucination vector because an empty
context combined with temperature=0 still produces a confident
answer from parametric memory.

The ONLY mocking allowed in the whole suite happens here: we patch
``src.retriever.search`` and ``src.retriever.embed_query`` so we can
exercise the business logic without Qdrant or external embedding APIs. Everything
else uses real implementations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.retriever import MIN_SCORE_THRESHOLD, retrieve


def test_min_score_threshold_value() -> None:
    """MIN_SCORE_THRESHOLD defaults to 0.25 for OpenRouter embeddings."""
    assert MIN_SCORE_THRESHOLD == 0.25, \
        f"MIN_SCORE_THRESHOLD must default to 0.25, got {MIN_SCORE_THRESHOLD}. " \
        "Below 0.25 the retrieved chunk is unlikely to be semantically " \
        "related to the query - lowering the threshold lets noise into the " \
        "prompt and the LLM hallucinates from irrelevant context."


def test_min_score_threshold_is_float() -> None:
    """Threshold must be float so >= comparison against float scores is well-defined."""
    assert isinstance(MIN_SCORE_THRESHOLD, float), \
        f"MIN_SCORE_THRESHOLD must be float, got {type(MIN_SCORE_THRESHOLD).__name__}. " \
        "An int threshold compared against float scores invites subtle " \
        "off-by-one comparison surprises that flake retrieval tests."


def test_retrieve_returns_ok_for_high_scores(sample_results: list[dict]) -> None:
    """High-score retrieval must return status='OK' so the LLM is called."""
    mock_client = MagicMock()
    with patch("src.retriever.search", return_value=list(sample_results)), \
         patch("src.retriever.embed_query",
               return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "test query", top_k=5, quiet=True)

    assert status == "OK", \
        f"Expected status 'OK' when all scores are above {MIN_SCORE_THRESHOLD}, " \
        f"got {status!r}. " \
        "An incorrect NO_RESULTS here would suppress legitimate answers."
    assert len(results) > 0, \
        f"OK status with empty results is contradictory - got {len(results)} results."


def test_retrieve_returns_no_results_for_low_scores(
    low_score_results: list[dict],
) -> None:
    """All-below-threshold retrieval must return NO_RESULTS so LLM is NOT called."""
    mock_client = MagicMock()
    with patch("src.retriever.search", return_value=list(low_score_results)), \
         patch("src.retriever.embed_query",
               return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "test query", top_k=5, quiet=True)

    assert status == "NO_RESULTS", \
        f"Expected NO_RESULTS for all scores below {MIN_SCORE_THRESHOLD}, " \
        f"got {status!r}. " \
        "A wrong OK here sends an empty/garbage context to the LLM, which " \
        "is the textbook hallucination case for RAG."
    assert len(results) == 0, \
        f"NO_RESULTS must come with an empty list, got {len(results)} results. " \
        "A non-empty list would let a caller ignore the status string and " \
        "still call the LLM with junk context."


def test_retrieve_returns_no_results_for_empty_search() -> None:
    """Empty Qdrant response must collapse to NO_RESULTS, not silent OK."""
    mock_client = MagicMock()
    with patch("src.retriever.search", return_value=[]), \
         patch("src.retriever.embed_query",
               return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "test query", top_k=5, quiet=True)

    assert status == "NO_RESULTS", \
        f"Empty search response must collapse to NO_RESULTS, got {status!r}. " \
        "Treating empty as OK invites the LLM to answer from parametric memory."
    assert results == [], \
        f"Empty search must produce empty result list, got {results!r}."


def test_retrieve_filters_below_threshold() -> None:
    """Mixed-score input must be split: above threshold kept, below dropped."""
    mock_client = MagicMock()
    mixed = [
        {"text": "A above", "page_num": 1, "source_pdf": "f.pdf",
         "chunk_id": "1", "score": 0.85, "rank": 1},
        {"text": "B below", "page_num": 1, "source_pdf": "f.pdf",
         "chunk_id": "2", "score": 0.20, "rank": 2},
        {"text": "C above", "page_num": 2, "source_pdf": "f.pdf",
         "chunk_id": "3", "score": 0.72, "rank": 3},
    ]
    with patch("src.retriever.search", return_value=mixed), \
         patch("src.retriever.embed_query",
               return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "test", top_k=5, quiet=True)

    assert status == "OK", \
        f"Expected OK when at least one score is above threshold, got {status!r}."
    assert len(results) == 2, \
        f"Expected 2 results after threshold filter, got {len(results)}. " \
        "Filter drift here means low-confidence chunks leak into the prompt."
    assert all(r["score"] >= MIN_SCORE_THRESHOLD for r in results), \
        f"Some kept results below threshold: " \
        f"{[r['score'] for r in results if r['score'] < MIN_SCORE_THRESHOLD]}. " \
        "A single under-threshold chunk is enough to ground a wrong answer."


def test_retrieve_deduplicates_same_page_similar_text() -> None:
    """Exact duplicates on the same page must collapse - keep the higher score."""
    mock_client = MagicMock()
    long_text = (
        "Neural networks learn through backpropagation by adjusting weights "
        "to minimize loss functions in a supervised learning setting."
    )
    dupes = [
        {"text": long_text, "page_num": 1, "source_pdf": "f.pdf",
         "chunk_id": "1", "score": 0.90, "rank": 1},
        {"text": long_text, "page_num": 1, "source_pdf": "f.pdf",
         "chunk_id": "2", "score": 0.85, "rank": 2},
        {"text": "Coral reefs support marine biodiversity in tropical oceans.",
         "page_num": 2, "source_pdf": "f.pdf",
         "chunk_id": "3", "score": 0.75, "rank": 3},
    ]
    with patch("src.retriever.search", return_value=dupes), \
         patch("src.retriever.embed_query",
               return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "test", top_k=5, quiet=True)

    assert status == "OK", \
        f"Expected OK after dedup leaves >=1 chunk, got {status!r}."
    assert len(results) == 2, \
        f"Expected 2 results after dedup (exact same-page duplicate removed), " \
        f"got {len(results)}. " \
        "Duplicate chunks waste prompt tokens and concentrate the LLM's " \
        "attention on a single repeated claim, distorting the answer."

    kept_ids = {r["chunk_id"] for r in results}
    assert "1" in kept_ids, \
        f"Higher-score duplicate (chunk 1, score 0.90) was removed; " \
        f"kept ids: {kept_ids}. " \
        "Dedup must always keep the higher-ranked instance - dropping it " \
        "loses retrieval quality."


def test_retrieve_status_values_are_correct_strings(
    sample_results: list[dict],
) -> None:
    """Status must be the literal strings 'OK' / 'NO_RESULTS' - never bool/None."""
    mock_client = MagicMock()
    with patch("src.retriever.search", return_value=list(sample_results)), \
         patch("src.retriever.embed_query",
               return_value=np.zeros(1024, dtype=np.float32)):
        _, status = retrieve(mock_client, "test", top_k=5, quiet=True)

    assert status in ("OK", "NO_RESULTS"), \
        f"Status must be 'OK' or 'NO_RESULTS', got {status!r}. " \
        "The CLI dispatches on this string with ==; any other sentinel " \
        "(True/None/0) silently routes to the wrong branch."
    assert isinstance(status, str), \
        f"Status must be str, got {type(status).__name__}. " \
        "Non-string status breaks the if-comparison in naive_rag._answer_one."


def test_retrieve_reranks_reference_like_chunks_lower() -> None:
    """Reference-like chunks should be down-ranked behind body content."""
    mock_client = MagicMock()
    candidates = [
        {
            "text": "In Proceedings of the 47th International ACM SIGIR Conference.",
            "page_num": 10,
            "source_pdf": "f.pdf",
            "chunk_id": "r1",
            "score": 0.33,
            "rank": 1,
        },
        {
            "text": "This paper introduces a benchmark for RAG robustness and analyzes temperature effects.",
            "page_num": 1,
            "source_pdf": "f.pdf",
            "chunk_id": "b1",
            "score": 0.30,
            "rank": 2,
        },
    ]

    with patch("src.retriever.search", return_value=candidates), \
         patch("src.retriever.embed_query", return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "what is this paper about", top_k=5, quiet=True)

    assert status == "OK"
    assert results[0]["chunk_id"] == "b1", \
        "Body content should outrank reference-like chunk after reranking."


def test_retrieve_broad_query_boosts_early_page_body_chunk() -> None:
    """Broad queries should prioritize page-1 body chunk over references."""
    mock_client = MagicMock()
    candidates = [
        {
            "text": "In Proceedings of the 47th International ACM SIGIR Conference. 2024.",
            "page_num": 10,
            "source_pdf": "f.pdf",
            "chunk_id": "r2",
            "score": 0.31,
            "rank": 1,
        },
        {
            "text": "We approach the RAG LLM as a black box and quantify temperature effects.",
            "page_num": 1,
            "source_pdf": "f.pdf",
            "chunk_id": "b2",
            "score": 0.27,
            "rank": 2,
        },
    ]
    with patch("src.retriever.search", return_value=candidates), \
         patch("src.retriever.embed_query", return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "summarize this paper", top_k=5, quiet=True)

    assert status == "OK"
    assert results[0]["chunk_id"] == "b2", \
        "Page-1 body chunk should win on broad summary intents."


def test_retrieve_broad_query_limits_reference_chunks() -> None:
    """Broad overview queries should include at most one reference-like chunk."""
    mock_client = MagicMock()
    candidates = [
        {
            "text": "In Proceedings of the 47th International ACM SIGIR Conference. 2024.",
            "page_num": 10,
            "source_pdf": "f.pdf",
            "chunk_id": "r1",
            "score": 0.36,
            "rank": 1,
        },
        {
            "text": "Findings of the Association for Computational Linguistics: EMNLP 2024.",
            "page_num": 10,
            "source_pdf": "f.pdf",
            "chunk_id": "r2",
            "score": 0.35,
            "rank": 2,
        },
        {
            "text": "This paper introduces a benchmark and analyzes perturbation robustness.",
            "page_num": 1,
            "source_pdf": "f.pdf",
            "chunk_id": "b1",
            "score": 0.30,
            "rank": 3,
        },
    ]
    with patch("src.retriever.search", return_value=candidates), \
         patch("src.retriever.embed_query", return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "what is this paper about", top_k=5, quiet=True)

    assert status == "OK"
    ref_ids = {"r1", "r2"}
    kept_refs = [r for r in results if r["chunk_id"] in ref_ids]
    assert len(kept_refs) <= 1, \
        "Broad query should keep at most one reference-like chunk."


def test_retrieve_reference_query_keeps_reference_chunks() -> None:
    """Citation/reference queries should not suppress reference-like chunks."""
    mock_client = MagicMock()
    candidates = [
        {
            "text": "In Proceedings of the 47th International ACM SIGIR Conference. 2024.",
            "page_num": 10,
            "source_pdf": "f.pdf",
            "chunk_id": "r3",
            "score": 0.31,
            "rank": 1,
        },
        {
            "text": "In Findings of the Association for Computational Linguistics: EMNLP 2024.",
            "page_num": 10,
            "source_pdf": "f.pdf",
            "chunk_id": "r4",
            "score": 0.30,
            "rank": 2,
        },
        {
            "text": "Body text about experimental design.",
            "page_num": 2,
            "source_pdf": "f.pdf",
            "chunk_id": "b3",
            "score": 0.29,
            "rank": 3,
        },
    ]
    with patch("src.retriever.search", return_value=candidates), \
         patch("src.retriever.embed_query", return_value=np.zeros(1024, dtype=np.float32)):
        results, status = retrieve(mock_client, "how many citations are there", top_k=5, quiet=True)

    assert status == "OK"
    top_ids = {r["chunk_id"] for r in results[:2]}
    assert "r3" in top_ids or "r4" in top_ids, \
        "Citation query should keep reference-like chunks near top."
