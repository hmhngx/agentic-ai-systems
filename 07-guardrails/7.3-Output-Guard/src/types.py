from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class GuardrailInternalError(RuntimeError):
    """Raised when guardrails-ai schema validation fails on a CheckResult."""


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    RETRY = "RETRY"


@dataclass(frozen=True)
class CheckResult:
    check: str
    verdict: Verdict
    reason: str
    score: float
    latency_ms: float


@dataclass(frozen=True)
class OutputGuardReport:
    answer: str
    verdict: Verdict
    reason: str
    checks: tuple[CheckResult, ...]
    total_latency_ms: float

    def passed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    def to_dict(self) -> dict:
        return {
            "answer": self.answer[:200],
            "verdict": self.verdict.value,
            "reason": self.reason,
            "checks": [
                {
                    "check": c.check,
                    "verdict": c.verdict.value,
                    "reason": c.reason,
                    "score": round(c.score, 4),
                    "latency_ms": round(c.latency_ms, 3),
                }
                for c in self.checks
            ],
            "total_latency_ms": round(self.total_latency_ms, 3),
        }


class OutputCheckResultModel(BaseModel):
    """guardrails-ai Pydantic schema contract for each CheckResult."""

    check: Literal["format", "toxicity", "faithfulness"]
    verdict: Literal["ALLOW", "BLOCK", "RETRY"]
    reason: str
    score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OutputGuardConfig:
    use_llm: bool = False
    llm_model: str = "openai/gpt-4o-mini"
    faithfulness_retry_threshold: float = 0.60
    toxicity_block_threshold: float = 0.50
    toxicity_soft_threshold: float = 0.25

    @classmethod
    def from_env(cls) -> OutputGuardConfig:
        api_key_present = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
        return cls(
            use_llm=_env_bool("OUTPUT_GUARD_USE_LLM", api_key_present),
            llm_model=os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
            faithfulness_retry_threshold=_env_float(
                "OUTPUT_GUARD_FAITHFULNESS_THRESHOLD", 0.60
            ),
            toxicity_block_threshold=_env_float("OUTPUT_GUARD_TOXICITY_THRESHOLD", 0.50),
            toxicity_soft_threshold=_env_float(
                "OUTPUT_GUARD_TOXICITY_SOFT_THRESHOLD", 0.25
            ),
        )
