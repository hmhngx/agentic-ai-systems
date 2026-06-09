from __future__ import annotations

import re
import time

from src.types import CheckResult, OutputGuardConfig, Verdict

# Hard patterns — any single match → immediate BLOCK (score 0.95).
_HARD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("violence", re.compile(r"\b(kill|murder|assassinate|shoot|stab|blow\s+up)\b", re.I)),
    ("violence", re.compile(r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)\b", re.I)),
    ("self_harm", re.compile(r"\b(suicide|self.harm|cut\s+yourself|overdose\s+on)\b", re.I)),
    ("self_harm", re.compile(r"\bhow\s+to\s+(kill|hurt)\s+(yourself|myself)\b", re.I)),
]

# Soft patterns — each match adds 0.30 to the score.
_SOFT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("hate_speech", re.compile(r"\b(hate|despise|loathe)\s+(all\s+)?(people|humans|group)\b", re.I)),
    ("hate_speech", re.compile(r"\b(inferior|subhuman|worthless)\s+(race|group|people)\b", re.I)),
    ("explicit", re.compile(r"\bexplicit\s+sexual\s+content\b", re.I)),
    ("explicit", re.compile(r"\bgraphic\s+violent?\s+content\b", re.I)),
    ("harassment", re.compile(r"\b(threaten|harass|stalk|doxx)\b", re.I)),
    ("harmful", re.compile(r"\b(illegal\s+drug|drug\s+synthesis|how\s+to\s+hack)\b", re.I)),
]


class ToxicityChecker:
    """Stage 2: keyword content filter (<5ms offline, optional LLM escalation)."""

    def __init__(self, config: OutputGuardConfig | None = None) -> None:
        self._config = config or OutputGuardConfig()

    def check(self, answer: str) -> CheckResult:
        t0 = time.perf_counter()

        for family, pattern in _HARD_PATTERNS:
            if pattern.search(answer):
                return CheckResult(
                    check="toxicity",
                    verdict=Verdict.BLOCK,
                    reason=f"toxic content detected: {family} (hard block)",
                    score=0.95,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        soft_hits = sum(1 for _, p in _SOFT_PATTERNS if p.search(answer))
        score = min(1.0, soft_hits * 0.30)

        if score > self._config.toxicity_block_threshold:
            return CheckResult(
                check="toxicity",
                verdict=Verdict.BLOCK,
                reason=(
                    f"toxic content score {score:.2f} exceeds "
                    f"threshold {self._config.toxicity_block_threshold:.2f}"
                ),
                score=score,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if (
            self._config.toxicity_soft_threshold <= score <= self._config.toxicity_block_threshold
            and self._config.use_llm
        ):
            from src.llm_judge import LLMJudgeError, llm_toxicity_check

            try:
                llm_score = llm_toxicity_check(answer, self._config)
                if llm_score > self._config.toxicity_block_threshold:
                    return CheckResult(
                        check="toxicity",
                        verdict=Verdict.BLOCK,
                        reason=f"LLM judge: toxic content score {llm_score:.2f}",
                        score=llm_score,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
            except LLMJudgeError:
                pass

        return CheckResult(
            check="toxicity",
            verdict=Verdict.ALLOW,
            reason="no toxic content detected",
            score=score,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
