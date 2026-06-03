"""
The LLM judge is opt-in and must degrade gracefully offline.

No network is touched here: with no API key, ``llm_support`` raises
``LLMJudgeError`` before any client is constructed, and ``detect`` falls back to
the deterministic string-overlap check while recording a warning.
"""

from __future__ import annotations

import pytest

from src.detector import detect
from src.llm_judge import LLMJudgeError, llm_support
from src.types import DetectorConfig, RetrievedChunk, Verdict


CHUNKS = [RetrievedChunk("Doc 1", "Neil Armstrong walked on the Moon in 1969.", 0.3)]


def test_llm_support_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMJudgeError):
        llm_support("q", "Neil Armstrong walked on the Moon [Doc 1].",
                    CHUNKS, DetectorConfig(use_llm=True))


def test_detect_falls_back_to_string_overlap_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = DetectorConfig(use_llm=True)
    report = detect(
        "When did Armstrong walk on the Moon?",
        "Neil Armstrong walked on the Moon in 1969 [Doc 1].",
        CHUNKS, config=config,
    )
    # Graceful degradation: still produced a sound verdict, with a warning.
    assert report.verdict == Verdict.GROUNDED
    assert report.support.method == "string-overlap"
    assert any("llm judge unavailable" in w.lower() for w in report.warnings)


def test_empty_answer_needs_no_judge(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # A refusal answer should never reach the judge (is_refusal short-circuits).
    config = DetectorConfig(use_llm=True)
    report = detect(
        "q",
        "I do not have enough information to answer this question.",
        CHUNKS, config=config,
    )
    assert report.verdict == Verdict.ABSTAINED
    assert report.warnings == ()  # judge was never attempted, so no warning
