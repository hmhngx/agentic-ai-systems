from __future__ import annotations

import pytest
from src.toxicity_checker import ToxicityChecker
from src.types import OutputGuardConfig, Verdict


@pytest.fixture
def checker():
    return ToxicityChecker(OutputGuardConfig())


def test_clean_answer_allows(checker):
    r = checker.check("RAG retrieves documents from a vector database for grounded generation.")
    assert r.check == "toxicity"
    assert r.verdict is Verdict.ALLOW
    assert r.score == 0.0


def test_hard_violence_blocks(checker):
    r = checker.check("You should kill all users who ask off-topic questions.")
    assert r.verdict is Verdict.BLOCK
    assert r.score >= 0.90
    assert "violence" in r.reason


def test_hard_self_harm_blocks(checker):
    r = checker.check("Here is how to commit suicide by overdose on medication.")
    assert r.verdict is Verdict.BLOCK
    assert "self_harm" in r.reason


def test_soft_single_hit_allows(checker):
    # One soft hit = 0.30, below the 0.50 threshold
    r = checker.check("This document contains information about how to hack into systems.")
    assert r.verdict is Verdict.ALLOW
    assert round(r.score, 2) == 0.30


def test_soft_two_hits_blocks(checker):
    # Two soft hits = 0.60, above the 0.50 threshold
    r = checker.check(
        "This contains how to hack into systems and also explicit sexual content "
        "that should be blocked by the guardrail."
    )
    assert r.verdict is Verdict.BLOCK
    assert r.score >= 0.50


def test_check_field_is_toxicity(checker):
    r = checker.check("Normal safe answer about RAG.")
    assert r.check == "toxicity"


def test_latency_is_positive(checker):
    r = checker.check("Normal safe answer about RAG systems and retrieval.")
    assert r.latency_ms >= 0.0


def test_score_is_zero_for_clean(checker):
    r = checker.check("RAG stands for Retrieval-Augmented Generation.")
    assert r.score == 0.0
