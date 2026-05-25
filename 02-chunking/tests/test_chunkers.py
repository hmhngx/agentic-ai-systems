"""Unit tests for src.chunkers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.chunkers import (
    _coalesce_single_sentence_groups,
    _cosine_similarity,
    _greedy_merge_by_char_limit,
    _sentences,
    naive_chunk,
    recursive_chunk,
    semantic_chunk,
    sentence_chunk,
)

# Lines 28-29 in src/chunkers.py: intentionally uncovered — defensive branch
# (spaCy model missing OSError with install message).

_SENTENCE_END = ".!?\"'"


def _chunk(fn: Callable[[str], list[str]], text: str) -> list[str]:
    return fn(text)


def test_naive_chunk_basic(sample_plain_text):
    """Naive chunker returns multiple non-empty string chunks on long text."""
    result = _chunk(naive_chunk, sample_plain_text)
    assert isinstance(result, list), "expected list return type"
    assert len(result) >= 3, "expected at least three chunks for long multi-topic text"
    assert all(isinstance(c, str) for c in result), "expected every chunk to be str"
    assert all(len(c.strip()) > 0 for c in result), "expected no empty chunks"


def test_naive_chunk_no_empty_chunks(sample_plain_text):
    """Naive chunks contain no blank or untrimmed whitespace-only segments."""
    result = _chunk(naive_chunk, sample_plain_text)
    assert not any(c.strip() == "" for c in result), "expected no empty chunks"
    assert not any(c.strip() != c for c in result), "expected chunks to be stripped"


def test_naive_chunk_coverage(sample_plain_text):
    """Naive chunk output retains at least 85% of source characters."""
    result = _chunk(naive_chunk, sample_plain_text)
    all_chunk_text = " ".join(result)
    assert len(all_chunk_text) >= len(sample_plain_text) * 0.85, (
        "expected at least 85% character coverage across chunks"
    )


def test_naive_chunk_minimal_input(minimal_text):
    """Naive chunker handles a single short sentence without raising."""
    result = _chunk(naive_chunk, minimal_text)
    assert isinstance(result, list), "expected list return on minimal input"


def test_naive_chunk_empty_input(empty_text):
    """Naive chunker handles empty string without raising."""
    result = _chunk(naive_chunk, empty_text)
    assert isinstance(result, list), "expected list return on empty input"


def test_naive_chunk_types_consistent(sample_plain_text):
    """Naive chunk elements are strings only."""
    result = _chunk(naive_chunk, sample_plain_text)
    assert all(type(c) is str for c in result), "expected plain str elements, not subclasses"


def test_recursive_chunk_basic(sample_plain_text):
    """Recursive chunker returns multiple non-empty string chunks on long text."""
    result = _chunk(recursive_chunk, sample_plain_text)
    assert isinstance(result, list), "expected list return type"
    assert len(result) >= 3, "expected at least three chunks for long multi-topic text"
    assert all(isinstance(c, str) for c in result), "expected every chunk to be str"
    assert all(len(c.strip()) > 0 for c in result), "expected no empty chunks"


def test_recursive_chunk_no_empty_chunks(sample_plain_text):
    """Recursive chunks contain no blank or untrimmed segments."""
    result = _chunk(recursive_chunk, sample_plain_text)
    assert not any(c.strip() == "" for c in result), "expected no empty chunks"
    assert not any(c.strip() != c for c in result), "expected chunks to be stripped"


def test_recursive_chunk_coverage(sample_plain_text):
    """Recursive chunk output retains at least 85% of source characters."""
    result = _chunk(recursive_chunk, sample_plain_text)
    all_chunk_text = " ".join(result)
    assert len(all_chunk_text) >= len(sample_plain_text) * 0.85, (
        "expected at least 85% character coverage across chunks"
    )


def test_recursive_chunk_minimal_input(minimal_text):
    """Recursive chunker handles a single short sentence without raising."""
    result = _chunk(recursive_chunk, minimal_text)
    assert isinstance(result, list), "expected list return on minimal input"


def test_recursive_chunk_empty_input(empty_text):
    """Recursive chunker handles empty string without raising."""
    result = _chunk(recursive_chunk, empty_text)
    assert isinstance(result, list), "expected list return on empty input"


def test_recursive_chunk_types_consistent(sample_plain_text):
    """Recursive chunk elements are strings only."""
    result = _chunk(recursive_chunk, sample_plain_text)
    assert all(type(c) is str for c in result), "expected plain str elements, not subclasses"


def test_sentence_chunk_basic(sample_plain_text):
    """Sentence chunker returns multiple non-empty string chunks on long text."""
    result = _chunk(sentence_chunk, sample_plain_text)
    assert isinstance(result, list), "expected list return type"
    assert len(result) >= 3, "expected at least three chunks for long multi-topic text"
    assert all(isinstance(c, str) for c in result), "expected every chunk to be str"
    assert all(len(c.strip()) > 0 for c in result), "expected no empty chunks"


def test_sentence_chunk_no_empty_chunks(sample_plain_text):
    """Sentence chunks contain no blank or untrimmed segments."""
    result = _chunk(sentence_chunk, sample_plain_text)
    assert not any(c.strip() == "" for c in result), "expected no empty chunks"
    assert not any(c.strip() != c for c in result), "expected chunks to be stripped"


def test_sentence_chunk_coverage(sample_plain_text):
    """Sentence chunk output retains at least 85% of source characters."""
    result = _chunk(sentence_chunk, sample_plain_text)
    all_chunk_text = " ".join(result)
    assert len(all_chunk_text) >= len(sample_plain_text) * 0.85, (
        "expected at least 85% character coverage across chunks"
    )


def test_sentence_chunk_minimal_input(minimal_text):
    """Sentence chunker handles a single short sentence without raising."""
    result = _chunk(sentence_chunk, minimal_text)
    assert isinstance(result, list), "expected list return on minimal input"


def test_sentence_chunk_empty_input(empty_text):
    """Sentence chunker handles empty string without raising."""
    result = _chunk(sentence_chunk, empty_text)
    assert isinstance(result, list), "expected list return on empty input"


def test_sentence_chunk_types_consistent(sample_plain_text):
    """Sentence chunk elements are strings only."""
    result = _chunk(sentence_chunk, sample_plain_text)
    assert all(type(c) is str for c in result), "expected plain str elements, not subclasses"


@pytest.mark.slow
def test_semantic_chunk_basic(sample_plain_text):
    """Semantic chunker returns multiple non-empty string chunks on long text."""
    result = _chunk(semantic_chunk, sample_plain_text)
    assert isinstance(result, list), "expected list return type"
    assert len(result) >= 3, "expected at least three chunks for long multi-topic text"
    assert all(isinstance(c, str) for c in result), "expected every chunk to be str"
    assert all(len(c.strip()) > 0 for c in result), "expected no empty chunks"


@pytest.mark.slow
def test_semantic_chunk_no_empty_chunks(sample_plain_text):
    """Semantic chunks contain no blank or untrimmed segments."""
    result = _chunk(semantic_chunk, sample_plain_text)
    assert not any(c.strip() == "" for c in result), "expected no empty chunks"
    assert not any(c.strip() != c for c in result), "expected chunks to be stripped"


@pytest.mark.slow
def test_semantic_chunk_coverage(sample_plain_text):
    """Semantic chunk output retains at least 85% of source characters."""
    result = _chunk(semantic_chunk, sample_plain_text)
    all_chunk_text = " ".join(result)
    assert len(all_chunk_text) >= len(sample_plain_text) * 0.85, (
        "expected at least 85% character coverage across chunks"
    )


@pytest.mark.slow
def test_semantic_chunk_minimal_input(minimal_text):
    """Semantic chunker handles a single short sentence without raising."""
    result = _chunk(semantic_chunk, minimal_text)
    assert isinstance(result, list), "expected list return on minimal input"


@pytest.mark.slow
def test_semantic_chunk_empty_input(empty_text):
    """Semantic chunker handles empty string without raising."""
    result = _chunk(semantic_chunk, empty_text)
    assert isinstance(result, list), "expected list return on empty input"


@pytest.mark.slow
def test_semantic_chunk_types_consistent(sample_plain_text):
    """Semantic chunk elements are strings only."""
    result = _chunk(semantic_chunk, sample_plain_text)
    assert all(type(c) is str for c in result), "expected plain str elements, not subclasses"


def test_naive_chunk_has_fixed_size(sample_plain_text):
    """Naive chunks do not exceed chunk_size by more than ten percent."""
    result = _chunk(naive_chunk, sample_plain_text)
    assert all(len(c) <= 1100 for c in result), "expected chunk length <= 1100 characters"


def test_recursive_chunk_respects_paragraphs():
    """Recursive splitter avoids starting chunks mid-word on paragraph boundaries."""
    paragraphs = [
        "Machine learning models learn patterns from labeled datasets.",
        "Ocean biology studies plankton blooms and coral reef ecosystems.",
        "Cooking techniques include sautéing braising and careful seasoning.",
        "Deep learning stacks neural layers for vision and language tasks.",
        "Marine mammals navigate using echolocation and seasonal migrations.",
    ]
    text = "\n\n".join(paragraphs)
    result = _chunk(recursive_chunk, text)
    for chunk in result:
        if chunk and chunk[0].isalpha():
            assert not chunk[0].islower(), (
                "expected chunk not to start mid-word (lowercase continuation)"
            )


def test_sentence_chunk_no_mid_sentence_cuts(sample_plain_text):
    """Most sentence chunks end on sentence-boundary punctuation."""
    result = _chunk(sentence_chunk, sample_plain_text)
    assert len(result) > 0, "expected at least one chunk to evaluate"
    ending_ok = 0
    for chunk in result:
        stripped = chunk.strip()
        if stripped and stripped[-1] in _SENTENCE_END:
            ending_ok += 1
    ratio = ending_ok / len(result)
    assert ratio >= 0.8, "expected at least 80% of chunks to end at sentence boundaries"


@pytest.mark.slow
def test_semantic_chunk_topic_separation(sample_plain_text):
    """Semantic chunker splits multi-topic text into at least two chunks."""
    result = _chunk(semantic_chunk, sample_plain_text)
    assert len(result) >= 2, "expected topic boundaries to yield multiple semantic chunks"


def test_sentences_whitespace_only():
    """_sentences returns [] for whitespace-only input."""
    assert _sentences("   \n\t  ") == [], "expected no sentences from blank text"


def test_greedy_merge_empty_parts():
    """_greedy_merge_by_char_limit returns [] when parts is empty."""
    assert _greedy_merge_by_char_limit([]) == [], "expected empty list for empty parts"


def test_cosine_similarity_zero_vector():
    """_cosine_similarity returns 0.0 when either vector has zero norm."""
    zero = np.zeros(4)
    unit = np.array([1.0, 0.0, 0.0, 0.0])
    assert _cosine_similarity(zero, unit) == 0.0, "expected zero similarity for zero vector"


def test_coalesce_single_group_unchanged():
    """_coalesce_single_sentence_groups leaves a single group unchanged."""
    groups = [["Only one sentence here."]]
    assert _coalesce_single_sentence_groups(groups) == groups, "expected single group passthrough"


def test_coalesce_merges_adjacent_singletons():
    """_coalesce_single_sentence_groups merges consecutive one-sentence groups."""
    groups = [["First."], ["Second."]]
    merged = _coalesce_single_sentence_groups(groups)
    assert len(merged) == 1, "expected two singleton groups merged into one"
    assert merged[0] == ["First.", "Second."], "expected sentences combined in order"


def test_sentence_chunk_no_spacy_sentences(monkeypatch):
    """sentence_chunk returns [] when _sentences yields no segments."""
    monkeypatch.setattr("src.chunkers._sentences", lambda _text: [])
    result = _chunk(sentence_chunk, "Non-empty but unsplit text.")
    assert result == [], "expected empty chunks when no sentences are detected"


@pytest.mark.slow
def test_semantic_chunk_no_sentences(monkeypatch):
    """semantic_chunk returns [] when _sentences yields no segments."""
    monkeypatch.setattr("src.chunkers._sentences", lambda _text: [])
    result = _chunk(semantic_chunk, "Non-empty but unsplit text.")
    assert result == [], "expected empty chunks when no sentences are detected"


@pytest.mark.slow
def test_semantic_chunk_single_sentence_returns_one_chunk(minimal_text):
    """semantic_chunk returns exactly one chunk for a single-sentence document."""
    result = _chunk(semantic_chunk, minimal_text)
    assert len(result) == 1, "expected one chunk for single-sentence input"
    assert result[0] == minimal_text.strip(), "expected chunk text to match input sentence"
