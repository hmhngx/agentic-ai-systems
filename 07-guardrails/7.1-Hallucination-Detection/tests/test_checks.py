"""Unit tests for the three checks in isolation (no scoring, no verdicts)."""

from __future__ import annotations

from src.checks import check_citations, check_retrieval, check_support
from src.types import DetectorConfig, RetrievedChunk


CONFIG = DetectorConfig()

CHUNKS = [
    RetrievedChunk("Doc 1", "Apollo 11 launched on July 16, 1969 from Florida.", 0.3),
    RetrievedChunk("Doc 2", "Neil Armstrong walked on the Moon on July 20, 1969.", 0.3),
    RetrievedChunk("Doc 3", "The command module was named Columbia.", 0.3),
]


# --- Check 1: support ------------------------------------------------------
class TestSupport:
    def test_faithfully_extracted_claim_is_supported(self):
        result = check_support(
            "Neil Armstrong walked on the Moon on July 20, 1969 [Doc 2].",
            CHUNKS, CONFIG,
        )
        assert result.supported_ratio == 1.0
        assert result.mean_support >= 0.8
        assert result.per_claim[0].best_chunk_id == "Doc 2"

    def test_offtopic_claim_is_unsupported(self):
        result = check_support(
            "Photosynthesis converts sunlight into glucose using chlorophyll.",
            CHUNKS, CONFIG,
        )
        assert result.supported_ratio == 0.0
        assert result.mean_support < 0.2

    def test_no_chunks_yields_zero_support(self):
        result = check_support("Anything at all.", [], CONFIG)
        assert result.supported_ratio == 0.0
        assert result.mean_support == 0.0

    def test_citations_are_stripped_before_overlap(self):
        # The bracket label must not itself count as overlap.
        plain = check_support("Columbia was the command module.", CHUNKS, CONFIG)
        cited = check_support("Columbia was the command module [Doc 3].", CHUNKS, CONFIG)
        assert plain.mean_support == cited.mean_support

    def test_containment_and_span_are_both_reported(self):
        result = check_support(
            "Neil Armstrong walked on the Moon on July 20, 1969.", CHUNKS, CONFIG,
        )
        claim = result.per_claim[0]
        assert 0.0 <= claim.span_ratio <= claim.containment <= 1.0

    def test_method_is_string_overlap(self):
        result = check_support("X.", CHUNKS, CONFIG)
        assert result.method == "string-overlap"


# --- Check 2: citations ----------------------------------------------------
class TestCitations:
    def test_valid_citations(self):
        result = check_citations("A [Doc 1]. B [Doc 2].", CHUNKS, CONFIG)
        assert result.validity == 1.0
        assert result.fabricated_ids == ()
        assert result.has_citations is True
        assert result.cited_ids == ("Doc 1", "Doc 2")

    def test_fabricated_citation_detected(self):
        result = check_citations("Jane Austen wrote it [Doc 9].", CHUNKS, CONFIG)
        assert result.fabricated_ids == ("Doc 9",)
        assert result.validity == 0.0
        assert result.has_citations is True

    def test_partially_fabricated(self):
        result = check_citations("A [Doc 1] and B [Doc 7].", CHUNKS, CONFIG)
        assert result.fabricated_ids == ("Doc 7",)
        assert result.validity == 0.5

    def test_uncited_answer(self):
        result = check_citations("No citations here at all.", CHUNKS, CONFIG)
        assert result.has_citations is False
        assert result.validity == 0.0
        assert result.cited_ids == ()

    def test_spacing_tolerant_extraction(self):
        result = check_citations("Spaced [ Doc 2 ].", CHUNKS, CONFIG)
        assert result.cited_ids == ("Doc 2",)


# --- Retrieval adequacy ----------------------------------------------------
class TestRetrieval:
    def test_empty_retrieval(self):
        result = check_retrieval([], CONFIG)
        assert result.n_chunks == 0
        assert result.adequacy == 0.0
        assert result.mean_score is None

    def test_adequacy_in_unit_interval(self):
        result = check_retrieval(CHUNKS, CONFIG)
        assert 0.0 <= result.adequacy <= 1.0
        assert result.mean_score is not None

    def test_no_scores_falls_back_to_count_adequacy(self):
        chunks = [RetrievedChunk("Doc 1", "text", None)]
        result = check_retrieval(chunks, CONFIG)
        assert result.mean_score is None
        assert result.adequacy == min(1 / CONFIG.target_chunk_count, 1.0)
