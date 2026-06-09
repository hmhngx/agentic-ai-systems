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
