from __future__ import annotations

import time

from pydantic import BaseModel, Field, ValidationError

from src.types import CheckResult, Verdict


class RAGAnswerModel(BaseModel):
    answer: str = Field(min_length=20)


class FormatChecker:
    """Stage 1: structural validation (<5ms, CPU-only)."""

    def check(self, answer: str) -> CheckResult:
        t0 = time.perf_counter()

        if not answer or not answer.strip():
            return CheckResult(
                check="format",
                verdict=Verdict.BLOCK,
                reason="answer is empty",
                score=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if answer.rstrip().endswith("..."):
            return CheckResult(
                check="format",
                verdict=Verdict.BLOCK,
                reason="answer appears truncated (ends with '...')",
                score=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            RAGAnswerModel(answer=answer)
        except ValidationError as exc:
            msg = exc.errors()[0]["msg"]
            return CheckResult(
                check="format",
                verdict=Verdict.BLOCK,
                reason=f"schema validation failed: {msg}",
                score=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return CheckResult(
            check="format",
            verdict=Verdict.ALLOW,
            reason="format valid",
            score=1.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
