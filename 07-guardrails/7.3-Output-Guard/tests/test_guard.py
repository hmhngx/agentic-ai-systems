from __future__ import annotations

import pytest
from src.guard import OutputGuard
from src.types import GuardrailInternalError, OutputGuardConfig, Verdict

_CTX = [
    "RAG stands for Retrieval-Augmented Generation.",
    "The system retrieves documents from a vector database using embedding-based similarity search.",
    "The retrieved documents are provided as context to the LLM for grounded text generation.",
]

_FAITHFUL = (
    "RAG retrieves documents from a vector database using embedding-based similarity search, "
    "then provides them as context to the LLM for grounded text generation."
)
_UNFAITHFUL = "RAG uses blockchain technology to store documents on a distributed ledger."
_TOXIC = "You should kill all users who ask irrelevant questions."
_EMPTY = ""
_TRUNCATED = "The answer is based on the retrieved documents..."
_SHORT = "Yes."


@pytest.fixture
def guard():
    return OutputGuard(OutputGuardConfig())


def test_faithful_answer_passes(guard):
    report = guard.check(_FAITHFUL, _CTX)
    assert report.passed() is True
    assert report.verdict is Verdict.ALLOW


def test_unfaithful_answer_retries(guard):
    report = guard.check(_UNFAITHFUL, _CTX)
    assert report.verdict is Verdict.RETRY
    assert not report.passed()


def test_toxic_answer_blocks(guard):
    report = guard.check(_TOXIC, _CTX)
    assert report.verdict is Verdict.BLOCK
    assert not report.passed()


def test_empty_answer_blocks(guard):
    report = guard.check(_EMPTY, _CTX)
    assert report.verdict is Verdict.BLOCK


def test_truncated_answer_blocks(guard):
    report = guard.check(_TRUNCATED, _CTX)
    assert report.verdict is Verdict.BLOCK


def test_short_answer_blocks(guard):
    report = guard.check(_SHORT, _CTX)
    assert report.verdict is Verdict.BLOCK


def test_format_block_short_circuits(guard):
    # Toxicity and faithfulness must NOT run when format fails.
    report = guard.check(_EMPTY, _CTX)
    assert len(report.checks) == 1
    assert report.checks[0].check == "format"


def test_toxicity_block_short_circuits(guard):
    # Faithfulness must NOT run when toxicity blocks.
    report = guard.check(_TOXIC, _CTX)
    checks_run = [c.check for c in report.checks]
    assert "faithfulness" not in checks_run
    assert "toxicity" in checks_run


def test_all_three_stages_run_on_faithful(guard):
    report = guard.check(_FAITHFUL, _CTX)
    checks_run = [c.check for c in report.checks]
    assert checks_run == ["format", "toxicity", "faithfulness"]


def test_report_total_latency_is_positive(guard):
    report = guard.check(_FAITHFUL, _CTX)
    assert report.total_latency_ms > 0.0


def test_report_to_dict_structure(guard):
    report = guard.check(_FAITHFUL, _CTX)
    d = report.to_dict()
    assert d["verdict"] == "ALLOW"
    assert len(d["checks"]) == 3


def test_invalid_check_result_raises_guardrail_error(monkeypatch):
    """If a checker produces an out-of-range score, GuardrailInternalError fires."""
    from src.format_checker import FormatChecker
    from src.types import CheckResult

    guard = OutputGuard(OutputGuardConfig())

    def _bad_check(answer):
        return CheckResult(
            check="format", verdict=Verdict.ALLOW, reason="ok",
            score=99.0,  # violates ge=0.0, le=1.0
            latency_ms=1.0,
        )

    monkeypatch.setattr(guard._format, "check", _bad_check)
    with pytest.raises(GuardrailInternalError):
        guard.check("some answer that is long enough to pass", _CTX)
