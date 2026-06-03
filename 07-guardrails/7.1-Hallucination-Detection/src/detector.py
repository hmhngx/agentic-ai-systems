"""
detector.py - The orchestrator: run the three checks, then assess.

This is the one function callers import: ``detect(question, answer, chunks)``.
It wires the checks to the scorer and handles the optional LLM-judge swap,
including the graceful fall back to the deterministic string-overlap check if
the judge is unavailable or errors. It performs no I/O of its own.
"""

from __future__ import annotations

from src.checks import check_citations, check_retrieval, check_support
from src.scoring import assess, is_refusal
from src.types import DetectorConfig, GroundednessReport, RetrievedChunk


def detect(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    config: DetectorConfig | None = None,
) -> GroundednessReport:
    """Score one RAG turn and return a groundedness report + verdict.

    The support check (Check 1) uses deterministic string overlap by default.
    When ``config.use_llm`` is set and there is something to judge (chunks
    present and the answer is not already a refusal), it tries the OpenRouter
    NLI judge first and falls back to string overlap on any failure, recording
    a warning. Checks 2 and 3 are always the deterministic functions.
    """
    config = config or DetectorConfig.from_env()
    warnings: list[str] = []

    citations = check_citations(answer, chunks, config)
    retrieval = check_retrieval(chunks, config)

    use_judge = config.use_llm and bool(chunks) and not is_refusal(answer, config)
    if use_judge:
        # Imported here so the offline path never imports the LLM stack.
        from src.llm_judge import LLMJudgeError, llm_support

        try:
            support = llm_support(question, answer, chunks, config)
        except LLMJudgeError as exc:
            warnings.append(
                f"LLM judge unavailable ({exc}); used string overlap instead."
            )
            support = check_support(answer, chunks, config)
    else:
        support = check_support(answer, chunks, config)

    return assess(
        question=question,
        answer=answer,
        support=support,
        citations=citations,
        retrieval=retrieval,
        config=config,
        extra_warnings=tuple(warnings),
    )
