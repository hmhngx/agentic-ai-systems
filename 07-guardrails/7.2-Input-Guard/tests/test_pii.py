from __future__ import annotations

import pytest

from src.pii_checker import PIIChecker, _luhn_check
from src.types import GuardConfig, Verdict


@pytest.fixture(scope="module")
def checker():
    return PIIChecker(GuardConfig(use_llm=False))


# --- Luhn algorithm unit tests ---

def test_luhn_valid_visa():
    # 4532015112830366 is a mathematically valid Luhn number
    assert _luhn_check("4532015112830366") is True


def test_luhn_invalid_visa():
    # Flip last digit: Luhn fails
    assert _luhn_check("4532015112830367") is False


def test_luhn_too_short():
    assert _luhn_check("123456789012") is False  # 12 digits, min is 13


# --- PII detection: BLOCK cases ---

@pytest.mark.parametrize("query,pii_label", [
    ("My email is john.doe@gmail.com", "email"),
    ("Contact me at alice+work@company.org", "email"),
    ("Call me at (415) 555-1234", "phone number"),
    ("Phone: 800-555-0199", "phone number"),
    ("My SSN is 123-45-6789", "SSN"),
    ("Social: 987-65-4320", "SSN"),
    ("Card number: 4532 0151 1283 0366 exp 12/26", "credit card"),
    ("Pay with 4532-0151-1283-0366", "credit card"),
])
def test_blocks_high_confidence_pii(checker, query, pii_label):
    result = checker.check(query)
    assert result.verdict == Verdict.BLOCK, (
        f"Expected BLOCK for {query!r}, got {result.verdict} ({result.reason})"
    )
    assert result.check == "pii"
    assert result.score >= 0.8
    assert pii_label in result.reason


# --- PII detection: ALLOW cases ---

@pytest.mark.parametrize("query", [
    "What is RAG?",
    "How do embeddings work?",
    "Explain transformer attention mechanisms.",
    "What are the tradeoffs between BM25 and dense retrieval?",
    "",
    "4532 0151 1283 0367",  # fails Luhn → not a valid card
])
def test_allows_clean_queries(checker, query):
    result = checker.check(query)
    assert result.verdict == Verdict.ALLOW, (
        f"Expected ALLOW for {query!r}, got {result.verdict} ({result.reason})"
    )
    assert result.check == "pii"


def test_returns_latency(checker):
    result = checker.check("What is RAG?")
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# LLM escalation path (soft zone: 0.40 ≤ confidence < 0.80)
# "Please forward the results to Sarah Johnson" → full name, confidence 0.60
# ---------------------------------------------------------------------------

_SOFT_QUERY = "Please forward the results to Sarah Johnson"


def test_llm_clears_soft_pii():
    """LLM returning 'no_pii' should override the soft block → ALLOW."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=True)
    checker = PIIChecker(config)
    with patch("src.pii_checker.llm_pii_check", return_value="no_pii") as mock:
        result = checker.check(_SOFT_QUERY)
    assert result.verdict == Verdict.ALLOW
    assert result.check == "pii"
    mock.assert_called_once_with(_SOFT_QUERY, config)


def test_llm_confirms_soft_pii():
    """LLM returning 'pii_detected' should keep the block and mention LLM."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=True)
    checker = PIIChecker(config)
    with patch("src.pii_checker.llm_pii_check", return_value="pii_detected"):
        result = checker.check(_SOFT_QUERY)
    assert result.verdict == Verdict.BLOCK
    assert "LLM" in result.reason
    assert result.check == "pii"


def test_llm_pii_error_falls_back_to_block():
    """LLMJudgeError should fall back to the conservative offline BLOCK."""
    from unittest.mock import patch
    from src.llm_judge import LLMJudgeError
    config = GuardConfig(use_llm=True)
    checker = PIIChecker(config)
    with patch("src.pii_checker.llm_pii_check", side_effect=LLMJudgeError("timeout")):
        result = checker.check(_SOFT_QUERY)
    assert result.verdict == Verdict.BLOCK
    assert result.check == "pii"


def test_hard_pii_never_calls_llm():
    """High-confidence PII (≥ hard threshold) must BLOCK without LLM involvement."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=True)
    checker = PIIChecker(config)
    with patch("src.pii_checker.llm_pii_check") as mock:
        result = checker.check("My SSN is 123-45-6789")
    assert result.verdict == Verdict.BLOCK
    mock.assert_not_called()


def test_llm_not_called_when_disabled():
    """use_llm=False must never invoke the LLM regardless of confidence zone."""
    from unittest.mock import patch
    config = GuardConfig(use_llm=False)
    checker = PIIChecker(config)
    with patch("src.pii_checker.llm_pii_check") as mock:
        checker.check(_SOFT_QUERY)
    mock.assert_not_called()
