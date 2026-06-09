from __future__ import annotations

import os
import pytest
from pydantic import ValidationError

from src.types import (
    CheckResult,
    GuardrailInternalError,
    OutputCheckResultModel,
    OutputGuardConfig,
    OutputGuardReport,
    Verdict,
)


def test_verdict_values():
    assert Verdict.ALLOW.value == "ALLOW"
    assert Verdict.BLOCK.value == "BLOCK"
    assert Verdict.RETRY.value == "RETRY"


def test_verdict_is_str():
    assert isinstance(Verdict.ALLOW, str)


def test_check_result_frozen():
    cr = CheckResult(check="format", verdict=Verdict.ALLOW, reason="ok", score=1.0, latency_ms=1.0)
    with pytest.raises(Exception):
        cr.score = 0.5  # type: ignore[misc]


def test_output_guard_report_passed_allow():
    cr = CheckResult(check="format", verdict=Verdict.ALLOW, reason="ok", score=1.0, latency_ms=1.0)
    report = OutputGuardReport(
        answer="hello world", verdict=Verdict.ALLOW, reason="all good",
        checks=(cr,), total_latency_ms=2.0,
    )
    assert report.passed() is True


def test_output_guard_report_passed_block():
    cr = CheckResult(check="format", verdict=Verdict.BLOCK, reason="empty", score=0.0, latency_ms=1.0)
    report = OutputGuardReport(
        answer="", verdict=Verdict.BLOCK, reason="empty",
        checks=(cr,), total_latency_ms=1.0,
    )
    assert report.passed() is False


def test_output_guard_report_to_dict_keys():
    cr = CheckResult(check="faithfulness", verdict=Verdict.RETRY, reason="low", score=0.3, latency_ms=5.0)
    report = OutputGuardReport(
        answer="some answer", verdict=Verdict.RETRY, reason="low",
        checks=(cr,), total_latency_ms=6.0,
    )
    d = report.to_dict()
    assert set(d.keys()) == {"answer", "verdict", "reason", "checks", "total_latency_ms"}
    assert d["verdict"] == "RETRY"
    assert len(d["checks"]) == 1
    assert d["checks"][0]["check"] == "faithfulness"


def test_output_check_result_model_valid():
    m = OutputCheckResultModel(
        check="faithfulness", verdict="ALLOW", reason="ok", score=0.8, latency_ms=10.0
    )
    assert m.score == 0.8


def test_output_check_result_model_score_out_of_range():
    with pytest.raises(ValidationError):
        OutputCheckResultModel(
            check="faithfulness", verdict="ALLOW", reason="ok", score=1.5, latency_ms=10.0
        )


def test_output_check_result_model_negative_score():
    with pytest.raises(ValidationError):
        OutputCheckResultModel(
            check="format", verdict="BLOCK", reason="bad", score=-0.1, latency_ms=1.0
        )


def test_output_check_result_model_invalid_check():
    with pytest.raises(ValidationError):
        OutputCheckResultModel(
            check="injection", verdict="BLOCK", reason="bad", score=0.9, latency_ms=1.0
        )


def test_output_check_result_model_invalid_verdict():
    with pytest.raises(ValidationError):
        OutputCheckResultModel(
            check="format", verdict="SKIP", reason="ok", score=1.0, latency_ms=1.0
        )


def test_output_guard_config_defaults():
    cfg = OutputGuardConfig()
    assert cfg.use_llm is False
    assert cfg.faithfulness_retry_threshold == 0.60
    assert cfg.toxicity_block_threshold == 0.50
    assert cfg.toxicity_soft_threshold == 0.25


def test_output_guard_config_from_env(monkeypatch):
    monkeypatch.setenv("OUTPUT_GUARD_USE_LLM", "1")
    monkeypatch.setenv("OUTPUT_GUARD_FAITHFULNESS_THRESHOLD", "0.75")
    monkeypatch.setenv("OUTPUT_GUARD_TOXICITY_THRESHOLD", "0.60")
    cfg = OutputGuardConfig.from_env()
    assert cfg.use_llm is True
    assert cfg.faithfulness_retry_threshold == 0.75
    assert cfg.toxicity_block_threshold == 0.60


def test_guardrail_internal_error_is_runtime_error():
    exc = GuardrailInternalError("schema fail")
    assert isinstance(exc, RuntimeError)
