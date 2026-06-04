from src.reflexion import refine, SYNONYM_MAP
from src.retriever import retrieve


def test_refine_expands_casual_terms_to_formal_synonyms():
    q = "How long does Helios keep documents before deleting them?"
    refined = refine(q, attempts_log=[{"query": q, "faithfulness": 0.0,
                                       "diagnosis": "WRONG_CHUNKS"}])
    low = refined.lower()
    assert "retention" in low or "retained" in low   # keep -> retain/retention
    assert "purged" in low                            # deleting -> purged


def test_refined_query_sharpens_retrieval_onto_owner_doc():
    q = "How long does Helios keep documents before deleting them?"
    refined = refine(q, attempts_log=[{"query": q, "faithfulness": 0.0,
                                       "diagnosis": "WRONG_CHUNKS"}])
    assert retrieve(refined)[0]["doc_id"] == "D1"


def test_refine_differs_from_previous_query():
    q = "What is the Helios carbon footprint per query?"  # no synonyms available
    prev = [{"query": q, "faithfulness": 0.0, "diagnosis": "WRONG_CHUNKS"}]
    refined = refine(q, attempts_log=prev)
    assert refined != q   # must change so the loop makes progress (even if unhelpful)


def test_synonym_map_is_generic_vocabulary_bridging():
    assert "keep" in SYNONYM_MAP and "retention" in SYNONYM_MAP["keep"]
    assert "cost" in SYNONYM_MAP
