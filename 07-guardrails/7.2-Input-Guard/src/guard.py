from __future__ import annotations

import time

from guardrails import Guard

from src.injection_checker import InjectionChecker
from src.pii_checker import PIIChecker
from src.topic_checker import TopicChecker
from src.types import (
    CheckResult,
    CheckResultModel,
    GuardConfig,
    GuardrailInternalError,
    InputGuardReport,
    Verdict,
)

# Guard created once at module load. Guard.for_pydantic() registers
# CheckResultModel as the schema enforcer — any CheckResult that would violate
# the Pydantic field constraints raises GuardrailInternalError.
_check_result_guard: Guard = Guard.for_pydantic(output_class=CheckResultModel)


def _validate(result: CheckResult) -> None:
    """Validate a CheckResult against the guardrails-ai Pydantic schema.

    Uses CheckResultModel directly (the model backing the Guard) for validation.
    Raises GuardrailInternalError on any schema violation.
    """
    try:
        CheckResultModel(
            check=result.check,
            verdict=result.verdict.value,
            reason=result.reason,
            score=result.score,
            latency_ms=result.latency_ms,
        )
    except Exception as exc:
        raise GuardrailInternalError(
            f"CheckResult schema validation failed for check='{result.check}': {exc}"
        ) from exc


class InputGuard:
    """Pre-LLM security firewall.

    Runs three checks in series: injection → PII → topic.
    Short-circuits on the first BLOCK — no downstream compute is spent.

    Usage:
        guard = InputGuard()
        report = guard.check(user_query)
        if not report.allowed():
            return report.reason
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        self._config = config or GuardConfig.from_env()
        self._checkers = [
            InjectionChecker(),
            PIIChecker(self._config),
            TopicChecker(self._config),
        ]

    def check(self, query: str) -> InputGuardReport:
        t0 = time.perf_counter()
        checks: list[CheckResult] = []

        for checker in self._checkers:
            result = checker.check(query)
            _validate(result)
            checks.append(result)
            if result.verdict is Verdict.BLOCK:
                break

        total_latency = (time.perf_counter() - t0) * 1000
        blocking = next((c for c in checks if c.verdict is Verdict.BLOCK), None)

        return InputGuardReport(
            query=query,
            verdict=Verdict.BLOCK if blocking else Verdict.ALLOW,
            reason=blocking.reason if blocking else "all checks passed",
            checks=tuple(checks),
            total_latency_ms=total_latency,
        )
