from src.corpus import DOCUMENTS, documents
from src.text import tokenize


def test_seven_documents_with_unique_ids():
    assert len(DOCUMENTS) == 7
    ids = [d["doc_id"] for d in DOCUMENTS]
    assert ids == [f"D{i}" for i in range(1, 8)]


def test_documents_returns_a_copy():
    a = documents()
    a.append({"doc_id": "X", "text": "x"})
    assert len(documents()) == 7  # mutating the returned list must not affect the source


def test_helios_is_ubiquitous_distinctive_terms_are_owned():
    toks = {d["doc_id"]: set(tokenize(d["text"])) for d in DOCUMENTS}
    # 'helios' in every doc => idf 0 => non-discriminative
    assert all("helios" in t for t in toks.values())
    # distinctive terms each owned by exactly one doc
    owned = {"retention": "D1", "purged": "D1", "quota": "D3",
             "pricing": "D2", "embedding": "D4"}
    for term, owner in owned.items():
        holders = [doc_id for doc_id, t in toks.items() if term in t]
        assert holders == [owner], f"{term} should be owned only by {owner}, got {holders}"
    # 'documents' is shared by D1 and D5 (the ambiguity that motivates reflexion)
    holders = sorted(doc_id for doc_id, t in toks.items() if "documents" in t)
    assert holders == ["D1", "D5"]
