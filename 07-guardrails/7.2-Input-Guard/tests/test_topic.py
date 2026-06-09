from __future__ import annotations

import pytest

from src.topic_checker import TopicChecker, _score
from src.types import GuardConfig, Verdict


@pytest.fixture(scope="module")
def checker():
    return TopicChecker(GuardConfig(use_llm=False))


# --- _score unit tests ---

def test_score_zero_for_empty():
    assert _score("") == 0.0


def test_score_zero_for_off_topic():
    assert _score("What is the best pizza recipe?") == 0.0


def test_score_positive_for_domain_words():
    s = _score("How do RAG embeddings work?")
    assert s > 0.0


def test_score_clamped_to_one():
    query = "rag retrieval embedding vector llm agent guardrail token prompt chunking"
    assert _score(query) <= 1.0


def test_score_threshold_boundary():
    # "What is RAG?" → "rag" is 1 hit out of 3 tokens = 0.33 ≥ 0.08
    assert _score("What is RAG?") >= 0.08


# --- TopicChecker ALLOW cases ---

@pytest.mark.parametrize("query", [
    "What is RAG and how does it use embeddings?",
    "How does FAISS work for vector similarity search?",
    "Explain chunking strategies for LLM context windows.",
    "What are the tradeoffs between dense and sparse retrieval?",
    "How do transformer attention mechanisms work?",
    "What is LangChain and how does it relate to agents?",
    "Explain fine-tuning vs RAG tradeoffs.",
    "How do I build a knowledge graph for RAG?",
])
def test_allows_on_topic(checker, query):
    result = checker.check(query)
    assert result.verdict == Verdict.ALLOW, (
        f"Expected ALLOW for {query!r}, got {result.verdict} (reason: {result.reason})"
    )
    assert result.check == "topic"


# --- TopicChecker BLOCK cases ---

@pytest.mark.parametrize("query", [
    "What is the best pizza recipe for a dinner party?",
    "Write me a poem about the ocean.",
    "Who won the World Cup in 2022?",
    "How do I bake sourdough bread?",
    "What movies are playing this weekend?",
    "",
])
def test_blocks_off_topic(checker, query):
    result = checker.check(query)
    assert result.verdict == Verdict.BLOCK, (
        f"Expected BLOCK for {query!r}, got {result.verdict}"
    )
    assert result.check == "topic"


def test_block_reason_contains_redirect(checker):
    result = checker.check("What is the best pizza?")
    assert "RAG" in result.reason or "LLM" in result.reason or "agentic" in result.reason


def test_returns_latency(checker):
    result = checker.check("What is RAG?")
    assert result.latency_ms >= 0.0
