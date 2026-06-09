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


# ---------------------------------------------------------------------------
# LLM escalation path (borderline zone: 0.04 ≤ score < 0.08)
# Query below scores 0.05 (1 domain hit / 20 tokens).
# ---------------------------------------------------------------------------

_BORDERLINE_QUERY = (
    "RAG is just a buzzword I keep hearing but I have no idea "
    "what anyone means when they say it"
)

# Clearly off-topic (score = 0.0 < _BORDERLINE_FLOOR): LLM must not be tried.
_CLEAR_OT_QUERY = "What is the best pizza recipe for a dinner party?"


def test_llm_allows_borderline_topic():
    """LLM returning 'on_topic' should override the offline block → ALLOW."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=True)
    checker = TopicChecker(config)
    with patch("src.topic_checker.llm_topic_check", return_value="on_topic") as mock:
        result = checker.check(_BORDERLINE_QUERY)
    assert result.verdict == Verdict.ALLOW
    assert result.check == "topic"
    assert "LLM override" in result.reason
    mock.assert_called_once_with(_BORDERLINE_QUERY, config)


def test_llm_blocks_borderline_topic():
    """LLM returning 'off_topic' should keep the block and mention LLM."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=True)
    checker = TopicChecker(config)
    with patch("src.topic_checker.llm_topic_check", return_value="off_topic"):
        result = checker.check(_BORDERLINE_QUERY)
    assert result.verdict == Verdict.BLOCK
    assert "LLM confirmed" in result.reason
    assert result.check == "topic"


def test_llm_topic_error_falls_back_to_block():
    """LLMJudgeError should fall back to the conservative offline BLOCK."""
    from unittest.mock import patch
    from src.llm_judge import LLMJudgeError
    config = GuardConfig(use_llm=True)
    checker = TopicChecker(config)
    with patch("src.topic_checker.llm_topic_check", side_effect=LLMJudgeError("timeout")):
        result = checker.check(_BORDERLINE_QUERY)
    assert result.verdict == Verdict.BLOCK
    assert result.check == "topic"


def test_clearly_off_topic_does_not_call_llm():
    """Scores below _BORDERLINE_FLOOR must BLOCK without LLM involvement."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=True)
    checker = TopicChecker(config)
    with patch("src.topic_checker.llm_topic_check") as mock:
        result = checker.check(_CLEAR_OT_QUERY)
    assert result.verdict == Verdict.BLOCK
    mock.assert_not_called()


def test_llm_topic_not_called_when_disabled():
    """use_llm=False must never invoke the LLM regardless of score zone."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=False)
    checker = TopicChecker(config)
    with patch("src.topic_checker.llm_topic_check") as mock:
        checker.check(_BORDERLINE_QUERY)
    mock.assert_not_called()
