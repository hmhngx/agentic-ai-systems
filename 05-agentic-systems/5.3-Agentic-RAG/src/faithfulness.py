"""Offline RAGAS-faithfulness proxy + root-failure-mode diagnosis.

faithfulness = supported_claims / total_claims, where a claim (a sentence of the
answer) is supported iff >= SUPPORT_FRAC of its DISCRIMINATIVE content tokens
appear in the union of retrieved chunks. Corpus-ubiquitous terms (idf 0, e.g.
'helios') are excluded so they neither prop up nor sink a claim; OOV words are
KEPT (they are the fabrication signal). With USE_RAGAS=1 the real `ragas`
faithfulness metric is used instead, behind the same return contract.

Failure mode (the 'trace a hallucination to its root cause' goal):
  NO_RETRIEVAL          - nothing retrieved
  ABSTAINED             - answer is the system's own refusal (safe, not a bug)
  GROUNDED              - every claim supported
  WRONG_CHUNKS          - unsupported AND retrieval grounded weakly (retrieval bug)
  UNSUPPORTED_GENERATION- unsupported despite the answer terms being retrievable
                          (generation drifted off otherwise-adequate context)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config
from src.generator import GROUND_MIN, REFUSAL
from src.retriever import get_space
from src.text import content_tokens

SUPPORT_FRAC = 0.5
_CITE_RE = re.compile(r"\[doc \d+\]", re.IGNORECASE)
_SENT_RE = re.compile(r"[^.;]+")


@dataclass
class FaithReport:
    score: float
    supported: int
    total: int
    unsupported_claims: list[str]
    failure_mode: str
    is_refusal: bool = False
    extra: dict = field(default_factory=dict)

    def asdict(self) -> dict:
        return {
            "score": self.score, "supported": self.supported, "total": self.total,
            "unsupported_claims": self.unsupported_claims,
            "failure_mode": self.failure_mode, "is_refusal": self.is_refusal,
        }


def _claims(answer: str) -> list[str]:
    stripped = _CITE_RE.sub("", answer)
    return [s.strip() for s in _SENT_RE.findall(stripped) if s.strip()]


def score(question: str, answer: str, chunks: list[dict]) -> FaithReport:
    if config.use_ragas():
        return _score_ragas(question, answer, chunks)

    if not chunks:
        return FaithReport(0.0, 0, 0, [], "NO_RETRIEVAL")

    if REFUSAL.lower() in answer.lower():
        return FaithReport(1.0, 0, 0, [], "ABSTAINED", is_refusal=True)

    space = get_space()
    chunk_tokens: set[str] = set()
    for c in chunks:
        chunk_tokens |= set(content_tokens(c["text"]))

    claims = _claims(answer)
    if not claims:
        return FaithReport(1.0, 0, 0, [], "GROUNDED")

    supported, unsupported = 0, []
    for claim in claims:
        # discriminative denominator: drop corpus-ubiquitous terms, keep OOV
        terms = [t for t in content_tokens(claim) if t not in space.ubiquitous]
        if not terms:
            supported += 1
            continue
        hit = sum(1 for t in terms if t in chunk_tokens)
        if hit / len(terms) >= SUPPORT_FRAC:
            supported += 1
        else:
            unsupported.append(claim)

    total = len(claims)
    faith = supported / total

    if not unsupported:
        mode = "GROUNDED"
    else:
        # root cause: did retrieval even ground the question?
        qterms = set(content_tokens(question))
        best_mass = space.idf_mass(qterms & set(content_tokens(chunks[0]["text"])))
        mode = "WRONG_CHUNKS" if best_mass < GROUND_MIN else "UNSUPPORTED_GENERATION"

    return FaithReport(faith, supported, total, unsupported, mode)


def _score_ragas(question: str, answer: str, chunks: list[dict]) -> FaithReport:
    """Real RAGAS faithfulness (only when USE_RAGAS=1)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness as ragas_faithfulness

    ds = Dataset.from_dict({
        "question": [question], "answer": [answer],
        "contexts": [[c["text"] for c in chunks]],
    })
    result = evaluate(ds, metrics=[ragas_faithfulness])
    val = float(result["faithfulness"])
    mode = "GROUNDED" if val >= 0.70 else "UNSUPPORTED_GENERATION"
    return FaithReport(val, 0, 1, [] if val >= 0.70 else [answer], mode,
                       extra={"ragas": True})
