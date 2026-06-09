from __future__ import annotations

import pytest
from src.format_checker import FormatChecker
from src.types import Verdict


@pytest.fixture
def checker():
    return FormatChecker()


def test_empty_string_blocks(checker):
    r = checker.check("")
    assert r.check == "format"
    assert r.verdict is Verdict.BLOCK
    assert "empty" in r.reason


def test_whitespace_only_blocks(checker):
    r = checker.check("   \n\t")
    assert r.verdict is Verdict.BLOCK
    assert "empty" in r.reason


def test_truncated_answer_blocks(checker):
    r = checker.check("The answer is based on the retrieved documents...")
    assert r.verdict is Verdict.BLOCK
    assert "truncated" in r.reason


def test_too_short_blocks(checker):
    r = checker.check("Yes.")
    assert r.verdict is Verdict.BLOCK


def test_exactly_20_chars_allows(checker):
    r = checker.check("a" * 20)
    assert r.verdict is Verdict.ALLOW


def test_valid_answer_allows(checker):
    r = checker.check(
        "RAG retrieves documents from a vector database using embedding similarity."
    )
    assert r.verdict is Verdict.ALLOW
    assert r.score == 1.0


def test_check_field_is_format(checker):
    r = checker.check("valid answer that is long enough to pass")
    assert r.check == "format"


def test_latency_is_positive(checker):
    r = checker.check("valid answer that is long enough to pass")
    assert r.latency_ms >= 0.0


def test_score_zero_on_block(checker):
    r = checker.check("")
    assert r.score == 0.0
