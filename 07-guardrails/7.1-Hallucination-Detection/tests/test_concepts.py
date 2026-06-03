"""
Each Day-10 *understand* concept, encoded as an executable assertion.

These are the nuances the detector exists to teach; if any regresses, the
guardrail's pedagogy is broken even if the headline demo still passes.
"""

from __future__ import annotations

from src.detector import detect
from src.types import DetectorConfig, Verdict


CONFIG = DetectorConfig()

APOLLO = [
    ("Doc 1", "Apollo 11 launched on July 16, 1969 from Florida.", 0.30),
    ("Doc 2", "Neil Armstrong walked on the Moon on July 20, 1969.", 0.29),
    ("Doc 3", "The command module was named Columbia.", 0.27),
]


def _chunks(triples):
    from src.types import RetrievedChunk

    return [RetrievedChunk(cid, txt, sc) for cid, txt, sc in triples]


def test_low_retrieval_score_does_not_imply_bad_answer():
    """A faithful extraction from a LOW-scoring chunk is still GROUNDED."""
    chunks = _chunks([
        ("Doc 1", "Photosynthesis converts sunlight into glucose using "
                  "chlorophyll in chloroplasts.", 0.04),  # very low retrieval score
    ])
    report = detect(
        "How does photosynthesis work?",
        "Photosynthesis converts sunlight into glucose using chlorophyll "
        "in chloroplasts [Doc 1].",
        chunks, config=CONFIG,
    )
    assert report.verdict == Verdict.GROUNDED
    assert report.served is True


def test_high_retrieval_score_does_not_imply_good_answer():
    """A hallucination over a HIGH-scoring chunk is still UNSUPPORTED."""
    chunks = _chunks([("Doc 1", "Apollo 11 launched in 1969.", 0.98)])
    report = detect(
        "What is the powerhouse of the cell?",
        "The mitochondria is the powerhouse of the cell [Doc 1].",
        chunks, config=CONFIG,
    )
    assert report.verdict == Verdict.UNSUPPORTED
    assert report.served is False


def test_supported_but_uncited_is_served_with_a_warning():
    """Missing citations alone do not condemn a well-supported answer."""
    report = detect(
        "When did Armstrong walk on the Moon?",
        "Neil Armstrong walked on the Moon on July 20, 1969.",  # no [Doc N]
        _chunks(APOLLO), config=CONFIG,
    )
    assert report.verdict == Verdict.UNCITED
    assert report.served is True
    assert report.citations.has_citations is False
    assert any("no inline" in w.lower() for w in report.warnings)


def test_valid_citation_does_not_make_a_claim_grounded():
    """Citing a REAL chunk that does not support the claim is UNSUPPORTED."""
    report = detect(
        "What is the capital of Australia?",
        "The capital of Australia is Canberra [Doc 1].",
        _chunks(APOLLO), config=CONFIG,
    )
    assert report.citations.fabricated_ids == ()       # the citation is real
    assert report.verdict == Verdict.UNSUPPORTED        # but the claim is not in it
    assert report.served is False


def test_correct_abstention_is_not_a_hallucination():
    """The RAG's own refusal is diagnosed ABSTAINED, not UNSUPPORTED."""
    report = detect(
        "What language ran the Apollo guidance computer?",
        "I do not have enough information in the retrieved documents to "
        "answer this question.",
        _chunks(APOLLO), config=CONFIG,
    )
    assert report.verdict == Verdict.ABSTAINED
    assert report.served is False


def test_fabricated_citation_is_its_own_failure_mode():
    """Inventing a [Doc N] is diagnosed distinctly from being unsupported."""
    report = detect(
        "Who wrote Pride and Prejudice?",
        "Jane Austen wrote Pride and Prejudice in 1813 [Doc 9].",
        _chunks(APOLLO), config=CONFIG,
    )
    assert report.verdict == Verdict.FABRICATED_CITATION
    assert "Doc 9" in report.citations.fabricated_ids
    assert report.served is False


def test_groundedness_measures_evidence_not_world_correctness():
    """A factually TRUE claim absent from the context is still UNSUPPORTED.

    Groundedness is answer<->evidence entailment, not correctness: "Paris is
    the capital of France" is true, but if it is not in the chunks the guardrail
    must still refuse - it cannot verify it from the provided context.
    """
    report = detect(
        "What is the capital of France?",
        "The capital of France is Paris [Doc 1].",
        _chunks(APOLLO), config=CONFIG,
    )
    assert report.verdict == Verdict.UNSUPPORTED
    assert report.served is False
