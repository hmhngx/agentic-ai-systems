from __future__ import annotations

import time

from src.faithfulness_checker import FaithfulnessChecker
from src.format_checker import FormatChecker
from src.toxicity_checker import ToxicityChecker
from src.types import (
    CheckResult,
    GuardrailInternalError,
    OutputCheckResultModel,
    OutputGuardConfig,
    OutputGuardReport,
    Verdict,
)


def _validate(result: CheckResult) -> None:
    """Validate a CheckResult against the Pydantic schema.

    Raises GuardrailInternalError on any field constraint violation.
    """
    try:
        OutputCheckResultModel(
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


class OutputGuard:
    """Post-generation output validation waterfall.

    Runs three checks in series: format → toxicity → faithfulness.
    Short-circuits on BLOCK. Returns RETRY when faithfulness score < threshold.
    """

    def __init__(self, config: OutputGuardConfig | None = None) -> None:
        self._config = config or OutputGuardConfig.from_env()
        self._format = FormatChecker()
        self._toxicity = ToxicityChecker(self._config)
        self._faithfulness = FaithfulnessChecker(self._config)

    def check(self, answer: str, context: list[str]) -> OutputGuardReport:
        t0 = time.perf_counter()
        checks: list[CheckResult] = []

        # Stage 1: structural format check
        fmt = self._format.check(answer)
        _validate(fmt)
        checks.append(fmt)
        if fmt.verdict is Verdict.BLOCK:
            return OutputGuardReport(
                answer=answer,
                verdict=Verdict.BLOCK,
                reason=fmt.reason,
                checks=tuple(checks),
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Stage 2: toxicity content check
        tox = self._toxicity.check(answer)
        _validate(tox)
        checks.append(tox)
        if tox.verdict is Verdict.BLOCK:
            return OutputGuardReport(
                answer=answer,
                verdict=Verdict.BLOCK,
                reason=tox.reason,
                checks=tuple(checks),
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Stage 3: faithfulness check (may return RETRY)
        faith = self._faithfulness.check(answer, context)
        _validate(faith)
        checks.append(faith)

        return OutputGuardReport(
            answer=answer,
            verdict=faith.verdict,
            reason=faith.reason,
            checks=tuple(checks),
            total_latency_ms=(time.perf_counter() - t0) * 1000,
        )
