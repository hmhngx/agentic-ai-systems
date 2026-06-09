from src.types import (
    CheckResult,
    CheckResultModel,
    GuardConfig,
    GuardrailInternalError,
    InputGuardReport,
    Verdict,
)
from pydantic import ValidationError


def test_verdict_is_str_enum():
    assert Verdict.ALLOW == "ALLOW"
    assert Verdict.BLOCK == "BLOCK"


def test_check_result_frozen():
    r = CheckResult(
        check="injection",
        verdict=Verdict.BLOCK,
        reason="test",
        score=1.0,
        latency_ms=0.5,
    )
    assert r.verdict is Verdict.BLOCK
    try:
        r.score = 0.0  # type: ignore[misc]
        assert False, "should be frozen"
    except Exception:
        pass


def test_input_guard_report_allowed():
    r = CheckResult(check="injection", verdict=Verdict.ALLOW, reason="ok", score=0.0, latency_ms=0.1)
    report = InputGuardReport(
        query="hello",
        verdict=Verdict.ALLOW,
        reason="all checks passed",
        checks=(r,),
        total_latency_ms=0.5,
    )
    assert report.allowed() is True
    assert report.to_dict()["verdict"] == "ALLOW"


def test_check_result_model_validates():
    m = CheckResultModel(
        check="topic",
        verdict="BLOCK",
        reason="off-topic",
        score=0.9,
        latency_ms=1.2,
    )
    assert m.verdict == "BLOCK"


def test_check_result_model_rejects_bad_score():
    try:
        CheckResultModel(
            check="topic",
            verdict="BLOCK",
            reason="off-topic",
            score=1.5,  # > 1.0 — should fail
            latency_ms=1.2,
        )
        assert False, "should raise ValidationError"
    except ValidationError:
        pass


def test_guard_config_defaults():
    cfg = GuardConfig()
    assert cfg.use_llm is False
    assert cfg.topic_threshold == 0.08
    assert cfg.pii_hard_threshold == 0.80
