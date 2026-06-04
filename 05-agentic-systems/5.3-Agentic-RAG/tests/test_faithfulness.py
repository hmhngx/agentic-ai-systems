from src.faithfulness import score, FaithReport, SUPPORT_FRAC
from src.generator import REFUSAL


def _chunks():
    return [{"doc_id": "D1", "label": "Doc 1", "score": 0.9, "rank": 1,
             "text": "Helios retention: indexed documents are retained for 90 days, "
                     "after which they are automatically purged."}]


def test_extracted_answer_is_fully_faithful():
    ans = "indexed documents are retained for 90 days [Doc 1]."
    rep = score("How long are documents kept?", ans, _chunks())
    assert isinstance(rep, FaithReport)
    assert rep.score == 1.0
    assert rep.failure_mode == "GROUNDED"
    assert rep.unsupported_claims == []


def test_guess_answer_is_unfaithful_and_diagnosed():
    # 'keep'/'deleting' are not in the chunk -> unsupported claim
    ans = "helios keep documents deleting is described in the retrieved context [Doc 1]."
    rep = score("How long does Helios keep documents before deleting them?", ans, _chunks())
    assert rep.score < 0.70
    assert rep.failure_mode in ("WRONG_CHUNKS", "UNSUPPORTED_GENERATION")
    assert rep.unsupported_claims


def test_no_chunks_is_no_retrieval():
    rep = score("anything", "some answer", [])
    assert rep.score == 0.0
    assert rep.failure_mode == "NO_RETRIEVAL"


def test_refusal_is_abstained_not_hallucination():
    rep = score("unanswerable", REFUSAL, _chunks())
    assert rep.failure_mode == "ABSTAINED"
    assert rep.is_refusal is True


def test_support_fraction_threshold_constant():
    assert SUPPORT_FRAC == 0.5
