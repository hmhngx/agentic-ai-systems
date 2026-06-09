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


@dataclass(frozen=True)
class CheckResult:
    check: str        # "injection" | "pii" | "topic"
    verdict: Verdict
    reason: str
    score: float      # 0.0–1.0
    latency_ms: float


@dataclass(frozen=True)
class InputGuardReport:
    query: str
    verdict: Verdict
    reason: str
    checks: tuple[CheckResult, ...]
    total_latency_ms: float

    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    def to_dict(self) -> dict:
        return {
            "query": self.query,
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


class CheckResultModel(BaseModel):
    """Pydantic schema used by guardrails-ai Guard for structural validation."""

    check: Literal["injection", "pii", "topic"]
    verdict: Literal["ALLOW", "BLOCK"]
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
class GuardConfig:
    use_llm: bool = False
    llm_model: str = "openai/gpt-4o-mini"
    topic_threshold: float = 0.08
    pii_hard_threshold: float = 0.80
    pii_soft_threshold: float = 0.40

    @classmethod
    def from_env(cls) -> GuardConfig:
        return cls(
            use_llm=_env_bool("INPUT_GUARD_USE_LLM", False),
            llm_model=os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
            topic_threshold=_env_float("INPUT_GUARD_TOPIC_THRESHOLD", 0.08),
            pii_hard_threshold=_env_float("INPUT_GUARD_PII_THRESHOLD", 0.80),
        )
