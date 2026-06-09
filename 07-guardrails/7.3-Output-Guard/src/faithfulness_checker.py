from __future__ import annotations

import re
import time

from src.types import CheckResult, OutputGuardConfig, Verdict

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "it", "its", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "my", "our", "your", "his", "her",
    "their", "not", "no", "as", "so", "if", "then", "than", "about",
})


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _faithfulness_score(answer: str, context: list[str]) -> float:
    """Mean per-sentence token containment across all context chunks.

    Containment = |sentence_tokens ∩ context_tokens| / |sentence_tokens|.
    Sentences with no content tokens are skipped.
    Returns 0.0 when context is empty or answer has no scorable sentences.
    """
    if not context:
        return 0.0

    context_tokens: set[str] = set()
    for chunk in context:
        context_tokens |= _content_tokens(chunk)

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    per_sentence: list[float] = []
    for sentence in sentences:
        s_tokens = _content_tokens(sentence)
        if not s_tokens:
            continue
        containment = len(s_tokens & context_tokens) / len(s_tokens)
        per_sentence.append(containment)

    if not per_sentence:
        return 0.0
    return sum(per_sentence) / len(per_sentence)


class FaithfulnessChecker:
    """Stage 3: token-containment faithfulness check (<20ms offline).

    score >= faithfulness_retry_threshold → ALLOW
    score <  faithfulness_retry_threshold → RETRY (orchestrator may retry once)
    """

    def __init__(self, config: OutputGuardConfig | None = None) -> None:
        self._config = config or OutputGuardConfig()

    def check(self, answer: str, context: list[str]) -> CheckResult:
        t0 = time.perf_counter()

        score = _faithfulness_score(answer, context)
        threshold = self._config.faithfulness_retry_threshold

        if score >= threshold:
            return CheckResult(
                check="faithfulness",
                verdict=Verdict.ALLOW,
                reason=f"faithfulness score {score:.3f} >= threshold {threshold:.2f}",
                score=score,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if self._config.use_llm:
            from src.llm_judge import LLMJudgeError, llm_faithfulness_check

            try:
                llm_score = llm_faithfulness_check(answer, context, self._config)
                score = llm_score
                if score >= threshold:
                    return CheckResult(
                        check="faithfulness",
                        verdict=Verdict.ALLOW,
                        reason=f"LLM judge faithfulness score {score:.3f} >= threshold {threshold:.2f}",
                        score=score,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
            except LLMJudgeError:
                pass  # fall through to offline score

        return CheckResult(
            check="faithfulness",
            verdict=Verdict.RETRY,
            reason=f"faithfulness score {score:.3f} < threshold {threshold:.2f}",
            score=score,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
