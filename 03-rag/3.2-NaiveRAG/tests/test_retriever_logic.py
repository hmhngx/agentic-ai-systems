"""Offline tests for ``src/retriever.py`` business logic.

The retriever sits between Qdrant and the LLM and applies three RAG-
specific rules:

  1. Drop any candidate with score < MIN_SCORE_THRESHOLD (0.40).
  2. Drop near-duplicate chunks (Jaccard > 0.95) on the same page.
  3. Return status='NO_RESULTS' when the filtered list is empty.

These rules are correctness-critical: skipping (1) lets noise reach
the LLM, skipping (2) wastes prompt tokens on duplicate context,
skipping (3) is THE main RAG hallucination vector because an empty
context combined with temperature=0 still produces a confident
answer from parametric memory.

The ONLY mocking allowed in the whole suite happens here: we patch
``src.retriever.search`` and ``src.retriever.embed_query`` so we can
exercise the business logic without Qdrant or Voyage. Everything
else uses real implementations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.retriever import MIN_SCORE_THRESHOLD, retrieve


def test_min_score_threshold_value() -> None:
    """MIN_SCORE_THRESHOLD == 0.40 - empirically tuned floor for voyage-3."""
    assert MIN_SCORE_THRESHOLD == 0.40, \
        f"MIN_SCORE_THRESHOLD must be 0.40, got {MIN_SCORE_THRESHOLD}. " \
        "Below 0.40 the retrieved chunk is unlikely to be semantically " \
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
         "chunk_id": "2", "score": 0.35, "rank": 2},
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
