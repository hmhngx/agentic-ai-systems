"""
types.py - Immutable data contracts for the hallucination detector.

Everything the detector consumes and produces is a frozen dataclass defined
here, so the boundary between "any RAG" and the guardrail is a small, explicit,
testable surface:

    input:  question: str, answer: str, chunks: list[RetrievedChunk]
    output: GroundednessReport

We use stdlib ``dataclasses`` rather than pydantic on purpose: the inputs are
constructed programmatically by an adapter (not parsed from untrusted JSON at a
network boundary), so the value of runtime coercion is low and the value of zero
extra dependencies is high. ``DetectorConfig`` centralises every tunable so the
scoring policy lives in one auditable place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


# The single fallback string the guardrail serves whenever it refuses an answer.
# Kept here (not buried in scoring) because it is part of the public contract:
# the Day-10 goal is that absent-context questions surface exactly this.
FALLBACK_MESSAGE: str = "I could not find relevant context."


class Verdict(str, Enum):
    """The diagnosed failure mode of a single RAG turn.

    Subclassing ``str`` makes the enum JSON-serialisable and lets ``--json``
    output and test assertions compare against plain strings.

    The values are ordered by *severity of distrust* in ``_VERDICT_PRECEDENCE``
    (in ``scoring.py``); the first matching condition wins, so a fabricated
    citation is reported even if the answer is also unsupported.
    """

    GROUNDED = "GROUNDED"                      # claims trace to chunks, citations valid
    NO_RETRIEVAL = "NO_RETRIEVAL"              # nothing was retrieved - context absent
    FABRICATED_CITATION = "FABRICATED_CITATION"  # cites a [Doc N] that does not exist
    UNSUPPORTED = "UNSUPPORTED"                # claims are not present in the chunks
    UNCITED = "UNCITED"                        # supported, but no inline [Doc N] at all
    ABSTAINED = "ABSTAINED"                    # the underlying RAG correctly refused


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved piece of evidence.

    ``chunk_id`` is the *citable label* the generator was told to reference -
    for the naive-RAG adapter this is ``"Doc 1" .. "Doc N"`` (rank order). The
    detector is RAG-agnostic: any pipeline that labels its chunks and emits
    matching inline citations can be wrapped.

    ``score`` is the retrieval (query<->chunk) similarity, if the RAG exposes
    it. It is deliberately *optional* and only lightly weighted - a high
    retrieval score does not imply a grounded answer, and a low one does not
    imply a bad answer. Groundedness is measured against the chunk *text*, not
    its retrieval score.
    """

    chunk_id: str
    text: str
    score: float | None = None


@dataclass(frozen=True)
class ClaimSupport:
    """Per-sentence (per-claim) grounding evidence, for diagnostics."""

    claim: str
    support: float          # max(containment, span_ratio) over all chunks, in [0, 1]
    containment: float      # set-overlap signal (paraphrase-tolerant)
    span_ratio: float       # difflib contiguous-run signal (verbatim-extraction)
    best_chunk_id: str | None
    supported: bool         # support >= config.claim_support_threshold


@dataclass(frozen=True)
class SupportResult:
    """Check 1 output: did the answer use the retrieved chunks?"""

    supported_ratio: float                 # fraction of claims that are supported
    mean_support: float                    # mean per-claim support (feeds the score)
    per_claim: tuple[ClaimSupport, ...]
    method: str                            # "string-overlap" or "llm-judge"


@dataclass(frozen=True)
class CitationResult:
    """Check 2 output: are the answer's [Doc N] citations real chunk IDs?"""

    cited_ids: tuple[str, ...]             # distinct labels cited, e.g. ("Doc 1",)
    valid_ids: tuple[str, ...]             # the real chunk_id space
    fabricated_ids: tuple[str, ...]        # cited but not real - a hard red flag
    validity: float                        # |cited & valid| / |cited|, 0.0 if none
    has_citations: bool


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieval adequacy: did we even retrieve a usable amount of evidence?"""

    n_chunks: int
    adequacy: float                        # in [0, 1]
    mean_score: float | None               # mean retrieval score when exposed


@dataclass(frozen=True)
class GroundednessReport:
    """The guardrail's verdict on one RAG turn - the public output.

    ``served_answer`` is what a caller should actually show the user: the
    original ``answer`` when ``served`` is True, otherwise the fallback string.
    The numeric ``score`` and the diagnostic ``verdict`` explain *why*.
    """

    question: str
    answer: str
    score: float
    verdict: Verdict
    served: bool
    served_answer: str
    support: SupportResult
    citations: CitationResult
    retrieval: RetrievalResult
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Flatten to plain JSON-serialisable primitives for ``--json`` output."""
        return {
            "question": self.question,
            "answer": self.answer,
            "score": round(self.score, 4),
            "verdict": self.verdict.value,
            "served": self.served,
            "served_answer": self.served_answer,
            "support": {
                "method": self.support.method,
                "supported_ratio": round(self.support.supported_ratio, 4),
                "mean_support": round(self.support.mean_support, 4),
                "per_claim": [
                    {
                        "claim": c.claim,
                        "support": round(c.support, 4),
                        "containment": round(c.containment, 4),
                        "span_ratio": round(c.span_ratio, 4),
                        "best_chunk_id": c.best_chunk_id,
                        "supported": c.supported,
                    }
                    for c in self.support.per_claim
                ],
            },
            "citations": {
                "cited_ids": list(self.citations.cited_ids),
                "valid_ids": list(self.citations.valid_ids),
                "fabricated_ids": list(self.citations.fabricated_ids),
                "validity": round(self.citations.validity, 4),
                "has_citations": self.citations.has_citations,
            },
            "retrieval": {
                "n_chunks": self.retrieval.n_chunks,
                "adequacy": round(self.retrieval.adequacy, 4),
                "mean_score": (
                    None if self.retrieval.mean_score is None
                    else round(self.retrieval.mean_score, 4)
                ),
            },
            "warnings": list(self.warnings),
        }


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, ignoring blank/invalid values."""
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
class DetectorConfig:
    """Every tunable knob for the scoring policy, in one auditable place.

    The three weights express the policy that *grounding in evidence* (support)
    dominates, citation hygiene matters but is secondary, and retrieval
    adequacy is only a faint prior. They must sum to 1.0 (validated below) so
    the composed score stays in [0, 1].
    """

    threshold: float = 0.5                 # Check 3: refuse if score < threshold
    w_support: float = 0.60
    w_citation: float = 0.25
    w_retrieval: float = 0.15
    claim_support_threshold: float = 0.50  # tau_claim: per-claim "is it grounded?"
    support_ratio_threshold: float = 0.50  # tau_support: gate for UNSUPPORTED
    target_chunk_count: int = 5            # chunks at/above which retrieval is "full"
    use_llm: bool = False                  # swap Check 1 for an OpenRouter NLI judge
    llm_model: str = "openai/gpt-4o-mini"
    fallback_message: str = FALLBACK_MESSAGE
    refusal_markers: tuple[str, ...] = (
        "i do not have enough information",
        "i don't have enough information",
        "i could not find relevant context",
        "i cannot answer this question based on",
    )

    def __post_init__(self) -> None:
        total = self.w_support + self.w_citation + self.w_retrieval
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"DetectorConfig weights must sum to 1.0, got {total:.6f} "
                f"(support={self.w_support}, citation={self.w_citation}, "
                f"retrieval={self.w_retrieval})."
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}.")

    @classmethod
    def from_env(cls) -> "DetectorConfig":
        """Build a config, letting environment variables override the defaults.

        Recognised: GROUNDEDNESS_THRESHOLD, GROUNDEDNESS_W_SUPPORT,
        GROUNDEDNESS_W_CITATION, GROUNDEDNESS_W_RETRIEVAL, USE_LLM,
        OPENROUTER_CHAT_MODEL.
        """
        return cls(
            threshold=_env_float("GROUNDEDNESS_THRESHOLD", 0.50),
            w_support=_env_float("GROUNDEDNESS_W_SUPPORT", 0.60),
            w_citation=_env_float("GROUNDEDNESS_W_CITATION", 0.25),
            w_retrieval=_env_float("GROUNDEDNESS_W_RETRIEVAL", 0.15),
            use_llm=_env_bool("USE_LLM", False),
            llm_model=os.environ.get("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
        )


# Verdicts whose answer must never be served to the user, regardless of score.
HARD_FALLBACK_VERDICTS: frozenset[Verdict] = frozenset(
    {
        Verdict.NO_RETRIEVAL,
        Verdict.FABRICATED_CITATION,
        Verdict.UNSUPPORTED,
        Verdict.ABSTAINED,
    }
)
