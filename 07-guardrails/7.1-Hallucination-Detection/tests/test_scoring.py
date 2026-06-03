"""Tests for the scoring policy: composition, gates, and invariants."""

from __future__ import annotations

import pytest

from src.checks import check_citations, check_retrieval, check_support
from src.detector import detect
from src.scoring import compose, is_refusal
from src.types import (
    DetectorConfig,
    HARD_FALLBACK_VERDICTS,
    RetrievedChunk,
    Verdict,
)


CONFIG = DetectorConfig()


def test_config_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        DetectorConfig(w_support=0.5, w_citation=0.5, w_retrieval=0.5)


def test_threshold_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        DetectorConfig(threshold=1.5)


class TestIsRefusal:
    def test_detects_naive_rag_refusal_phrase(self):
        assert is_refusal(
            "I do not have enough information in the retrieved documents.", CONFIG
        )

    def test_detects_our_own_fallback(self):
        assert is_refusal("I could not find relevant context.", CONFIG)

    def test_normal_answer_is_not_a_refusal(self):
        assert not is_refusal("The capital of France is Paris.", CONFIG)


class TestCompose:
    def _signals(self, answer, chunks):
        return (
            check_support(answer, chunks, CONFIG),
            check_citations(answer, chunks, CONFIG),
            check_retrieval(chunks, CONFIG),
        )

    def test_compose_is_bounded(self):
        s, c, r = self._signals("anything", [RetrievedChunk("Doc 1", "x", 0.5)])
        assert 0.0 <= compose(s, c, r, CONFIG) <= 1.0

    def test_perfect_signals_compose_to_one(self):
        # Verbatim claim, valid citation, full high-score retrieval -> 1.0.
        chunk = RetrievedChunk("Doc 1", "alpha beta gamma delta", 1.0)
        chunks = [chunk] * CONFIG.target_chunk_count
        s, c, r = self._signals("alpha beta gamma delta [Doc 1].", chunks)
        assert s.mean_support == pytest.approx(1.0)
        assert c.validity == 1.0
        assert compose(s, c, r, CONFIG) == pytest.approx(1.0)


class TestGatePrecedence:
    """A fabricated citation outranks an (also-present) unsupported diagnosis."""

    def test_fabrication_beats_unsupported(self):
        chunks = [RetrievedChunk("Doc 1", "Apollo 11 launched in 1969.", 0.3)]
        # Off-topic AND cites a non-existent Doc 9 -> FABRICATED wins.
        report = detect(
            "capital of australia",
            "The capital of Australia is Canberra [Doc 9].",
            chunks, config=CONFIG,
        )
        assert report.verdict == Verdict.FABRICATED_CITATION

    def test_no_retrieval_beats_everything(self):
        report = detect("q", "Confident answer [Doc 9].", [], config=CONFIG)
        assert report.verdict == Verdict.NO_RETRIEVAL
        assert report.score == 0.0


class TestInvariants:
    """Properties that must hold for any input."""

    CASES = [
        ("q", "", []),
        ("q", "Plain answer.", []),
        ("q", "Apollo 11 launched in 1969 [Doc 1].",
         [RetrievedChunk("Doc 1", "Apollo 11 launched in 1969.", 0.9)]),
        ("q", "Off topic [Doc 5].",
         [RetrievedChunk("Doc 1", "Apollo.", 0.9)]),
        ("q", "I do not have enough information.",
         [RetrievedChunk("Doc 1", "Apollo.", 0.1)]),
    ]

    @pytest.mark.parametrize("question,answer,chunks", CASES)
    def test_score_always_in_unit_interval(self, question, answer, chunks):
        report = detect(question, answer, chunks, config=CONFIG)
        assert 0.0 <= report.score <= 1.0

    @pytest.mark.parametrize("question,answer,chunks", CASES)
    def test_served_implies_safe_verdict_and_score(self, question, answer, chunks):
        report = detect(question, answer, chunks, config=CONFIG)
        if report.served:
            assert report.verdict not in HARD_FALLBACK_VERDICTS
            assert report.score >= CONFIG.threshold
            assert report.served_answer == report.answer
        else:
            assert report.served_answer == CONFIG.fallback_message

    @pytest.mark.parametrize("question,answer,chunks", CASES)
    def test_hard_fallback_verdicts_never_serve(self, question, answer, chunks):
        report = detect(question, answer, chunks, config=CONFIG)
        if report.verdict in HARD_FALLBACK_VERDICTS:
            assert report.served is False
