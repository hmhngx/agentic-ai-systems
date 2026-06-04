from src.retriever import retrieve, get_space, TOP_K


def test_distinctive_query_ranks_owner_first():
    chunks = retrieve("Helios query quota per day")
    assert chunks[0]["doc_id"] == "D3"          # 'quota' is owned by D3
    assert chunks[0]["label"] == "Doc 1"
    assert chunks[0]["rank"] == 1


def test_embedding_query_ranks_d4_first():
    chunks = retrieve("Which embedding model does Helios use for collections?")
    assert chunks[0]["doc_id"] == "D4"


def test_top_k_limits_results_and_labels_are_sequential():
    chunks = retrieve("Helios documents")
    assert len(chunks) <= TOP_K
    assert [c["label"] for c in chunks] == [f"Doc {i}" for i in range(1, len(chunks) + 1)]


def test_oov_only_query_returns_empty():
    assert retrieve("carbon footprint emissions") == []


def test_get_space_is_singleton():
    assert get_space() is get_space()
