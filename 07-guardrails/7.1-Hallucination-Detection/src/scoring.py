"""
scoring.py - Turn the three check results into one score and one verdict.

This module owns the *policy*: how the measurements combine, which override
gates fire, and how the failure mode is diagnosed. Keeping it separate from
``checks.py`` means the policy can change (re-weight, re-threshold) without
touching the measurement code, and every branch here is unit-testable.

The composed groundedness score is:

    raw = w_support * mean_support
        + w_citation * citation_validity
        + w_retrieval * retrieval_adequacy

Then, in precedence order, hard override gates can *cap* the score below the
threshold and assign a specific verdict. Capping (rather than letting the blend
decide) is what guarantees that a fabricated citation or an unsupported answer
can never be served, no matter how the other terms shake out.
"""

from __future__ import annotations

from src.text import normalize
from src.types import (
    CitationResult,
    DetectorConfig,
    GroundednessReport,
    HARD_FALLBACK_VERDICTS,
    RetrievalResult,
    SupportResult,
    Verdict,
)


def is_refusal(answer: str, config: DetectorConfig) -> bool:
    """True when the answer is the underlying RAG's own "I don't know".

    A correct refusal is not a hallucination - it is the system behaving
    exactly as a grounded RAG should when the context is absent. We detect it
    so it can be labelled ``ABSTAINED`` rather than mis-diagnosed as unsupported
    (it would be unsupported: a refusal shares no facts with the chunks).
    """
    norm = normalize(answer)
    return any(marker in norm for marker in config.refusal_markers)


def compose(
    support: SupportResult,
    citations: CitationResult,
    retrieval: RetrievalResult,
    config: DetectorConfig,
) -> float:
    """Weighted blend of the three signals, clamped to [0, 1]."""
    raw = (
        config.w_support * support.mean_support
        + config.w_citation * citations.validity
        + config.w_retrieval * retrieval.adequacy
    )
    return min(max(raw, 0.0), 1.0)


def assess(
    question: str,
    answer: str,
    support: SupportResult,
    citations: CitationResult,
    retrieval: RetrievalResult,
    config: DetectorConfig,
    extra_warnings: tuple[str, ...] = (),
) -> GroundednessReport:
    """Resolve check results into the final report.

    Precedence (first match wins), implementing
    ``NO_RETRIEVAL > FABRICATED_CITATION > ABSTAINED > UNSUPPORTED > UNCITED >
    GROUNDED``:

      1. No chunks retrieved        -> score 0.0, NO_RETRIEVAL.
      2. Fabricated citation present-> score capped < threshold, FABRICATED_CITATION.
      3. Answer is a refusal        -> ABSTAINED (served = fallback).
      4. Claims not supported       -> score capped < threshold, UNSUPPORTED.
      5. Supported but uncited      -> UNCITED (served iff score >= threshold).
      6. Supported and cited        -> GROUNDED (served iff score >= threshold).

    Steps 2 and 4 swap order relative to the textual precedence on purpose:
    fabrication is checked before abstention/support because an invented
    citation is the strongest distrust signal and should be the reported
    diagnosis even when the answer is also unsupported.
    """
    raw = compose(support, citations, retrieval, config)
    cap = config.threshold - 1e-9  # forces "below threshold" => fallback
    warnings: list[str] = list(extra_warnings)

    # Gate 1: nothing retrieved - the context is absent, full stop.
    if retrieval.n_chunks == 0:
        return _report(
            question, answer, 0.0, Verdict.NO_RETRIEVAL, support,
            citations, retrieval, config, warnings,
        )

    # Gate 2: a fabricated citation outranks every other diagnosis.
    if citations.fabricated_ids:
        warnings.append(
            "Answer cites non-existent chunk(s): "
            + ", ".join(citations.fabricated_ids)
        )
        return _report(
            question, answer, min(raw, cap), Verdict.FABRICATED_CITATION,
            support, citations, retrieval, config, warnings,
        )

    # Gate 3: the RAG correctly declined - not a hallucination.
    if is_refusal(answer, config):
        return _report(
            question, answer, min(raw, cap), Verdict.ABSTAINED,
            support, citations, retrieval, config, warnings,
        )

    # Gate 4: claims are not grounded in the chunks - the classic hallucination.
    supported = support.supported_ratio >= config.support_ratio_threshold
    if not supported:
        return _report(
            question, answer, min(raw, cap), Verdict.UNSUPPORTED,
            support, citations, retrieval, config, warnings,
        )

    # Supported. Distinguish cited vs uncited; the threshold decides serving.
    if not citations.has_citations:
        if raw >= config.threshold:
            warnings.append(
                "Answer is supported by the chunks but contains no inline "
                "[Doc N] citations; served with reduced confidence."
            )
        return _report(
            question, answer, raw, Verdict.UNCITED, support,
            citations, retrieval, config, warnings,
        )

    return _report(
        question, answer, raw, Verdict.GROUNDED, support,
        citations, retrieval, config, warnings,
    )


def _report(
    question: str,
    answer: str,
    score: float,
    verdict: Verdict,
    support: SupportResult,
    citations: CitationResult,
    retrieval: RetrievalResult,
    config: DetectorConfig,
    warnings: list[str],
) -> GroundednessReport:
    """Construct the report, deriving the serve decision consistently.

    An answer is served only when its verdict is non-fatal *and* the score
    clears the threshold. Hard-fallback verdicts never serve the answer even if
    a score somehow cleared the bar - the verdict is the final authority.
    """
    served = verdict not in HARD_FALLBACK_VERDICTS and score >= config.threshold
    served_answer = answer if served else config.fallback_message
    return GroundednessReport(
        question=question,
        answer=answer,
        score=score,
        verdict=verdict,
        served=served,
        served_answer=served_answer,
        support=support,
        citations=citations,
        retrieval=retrieval,
        warnings=tuple(warnings),
    )
