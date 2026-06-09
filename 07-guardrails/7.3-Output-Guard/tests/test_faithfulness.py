from __future__ import annotations

import pytest
from src.faithfulness_checker import FaithfulnessChecker, _content_tokens, _faithfulness_score
from src.types import OutputGuardConfig, Verdict

_CTX = [
    "RAG stands for Retrieval-Augmented Generation.",
    "The technique was introduced in the 2020 paper by Lewis et al.",
    "The system retrieves documents from a vector database using embedding-based similarity search.",
    "The retrieved documents are provided as context to the LLM for grounded text generation.",
]


@pytest.fixture
def checker():
    return FaithfulnessChecker(OutputGuardConfig())


# --- Unit: _content_tokens ---

def test_content_tokens_removes_stopwords():
    tokens = _content_tokens("the quick brown fox")
    assert "the" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_content_tokens_lowercases():
    tokens = _content_tokens("RAG Retrieval")
    assert "rag" in tokens
    assert "retrieval" in tokens


def test_content_tokens_filters_single_chars():
    tokens = _content_tokens("a b c dog")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "dog" in tokens


# --- Unit: _faithfulness_score ---

def test_faithful_answer_scores_high():
    answer = (
        "RAG retrieves documents from a vector database using embedding-based "
        "similarity search and provides them as context to the LLM for grounded generation."
    )
    score = _faithfulness_score(answer, _CTX)
    assert score >= 0.60


def test_fabricated_answer_scores_low():
    answer = "RAG uses blockchain technology to store and verify documents on a distributed ledger."
    score = _faithfulness_score(answer, _CTX)
    assert score < 0.60


def test_empty_context_scores_zero():
    score = _faithfulness_score("Any answer here about RAG.", [])
    assert score == 0.0


def test_empty_answer_scores_zero():
    score = _faithfulness_score("", _CTX)
    assert score == 0.0


# --- FaithfulnessChecker ---

def test_faithful_answer_allows(checker):
    answer = (
        "RAG, or Retrieval-Augmented Generation, retrieves documents from a vector "
        "database using embedding-based similarity search, then provides them as "
        "context to the LLM for grounded generation."
    )
    r = checker.check(answer, _CTX)
    assert r.check == "faithfulness"
    assert r.verdict is Verdict.ALLOW
    assert r.score >= 0.60


def test_fabricated_answer_retries(checker):
    answer = "RAG was invented by OpenAI in 2022 as part of the GPT-4 project."
    r = checker.check(answer, _CTX)
    assert r.verdict is Verdict.RETRY
    assert r.score < 0.60


def test_empty_context_retries(checker):
    r = checker.check("Some answer about RAG systems that seems reasonable.", [])
    assert r.verdict is Verdict.RETRY


def test_check_field_is_faithfulness(checker):
    r = checker.check("Some answer about RAG.", _CTX)
    assert r.check == "faithfulness"


def test_latency_is_positive(checker):
    r = checker.check("RAG retrieves documents using embeddings.", _CTX)
    assert r.latency_ms >= 0.0


def test_score_in_range(checker):
    r = checker.check("Some answer about RAG systems.", _CTX)
    assert 0.0 <= r.score <= 1.0
