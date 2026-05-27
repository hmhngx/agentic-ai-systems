"""Offline tests for synthetic corpus and query-set generation."""

from __future__ import annotations

from collections import Counter

from src.corpus import generate_corpus, get_query_set

_CHUNK_KEYS = {"id", "text", "topic", "source", "page", "chunk_index"}
_QUERY_KEYS = {"query_text", "source_chunk_id", "topic"}
_EXPECTED_TOPICS = {
    "machine_learning",
    "ocean_biology",
    "ancient_history",
    "cooking",
    "urban_architecture",
}


def test_corpus_length():
    """Verify default corpus has 500 chunks with all required metadata keys."""
    corpus = generate_corpus(n_chunks=500)
    assert len(corpus) == 500, "expected 500 chunks because n_chunks=500"
    for chunk in corpus:
        assert set(chunk.keys()) == _CHUNK_KEYS, (
            f"expected chunk keys {_CHUNK_KEYS} because corpus schema is fixed"
        )


def test_corpus_deterministic():
    """Verify same seed yields identical corpus and different seeds diverge."""
    c1 = generate_corpus(seed=42)
    c2 = generate_corpus(seed=42)
    assert c1[0]["text"] == c2[0]["text"], (
        "expected identical first chunk text because seed=42 is fixed"
    )
    assert c1[499]["topic"] == c2[499]["topic"], (
        "expected identical last chunk topic because seed=42 is fixed"
    )
    c3 = generate_corpus(seed=99)
    assert c1[0]["text"] != c3[0]["text"], (
        "expected different first chunk text because seed=99 differs from 42"
    )


def test_corpus_id_sequence():
    """Verify chunk IDs are a contiguous 0..n-1 sequence for Qdrant point IDs."""
    corpus = generate_corpus(n_chunks=500)
    assert [c["id"] for c in corpus] == list(range(500)), (
        "expected ids 0..499 because each chunk gets a sequential id"
    )


def test_corpus_topic_distribution():
    """Verify five topics each have exactly 100 chunks (even split)."""
    corpus = generate_corpus(n_chunks=500)
    dist = Counter(c["topic"] for c in corpus)
    assert len(dist) == 5, "expected 5 topics because corpus has five topic pools"
    assert all(v == 100 for v in dist.values()), (
        "expected 100 chunks per topic because 500 / 5 = 100 with no rounding"
    )


def test_corpus_valid_topics():
    """Verify corpus only uses the five defined topic keys."""
    corpus = generate_corpus(n_chunks=500)
    assert set(c["topic"] for c in corpus) == _EXPECTED_TOPICS, (
        "expected exactly the five TOPIC_SENTENCES keys in the corpus"
    )


def test_corpus_text_length():
    """Verify every chunk has non-trivial text for embedding and retrieval."""
    corpus = generate_corpus()
    assert all(len(c["text"]) >= 100 for c in corpus), (
        "expected each chunk text >= 100 chars because four sentences are joined"
    )


def test_corpus_page_range():
    """Verify simulated page numbers stay within the documented 1..50 range."""
    corpus = generate_corpus()
    assert all(1 <= c["page"] <= 50 for c in corpus), (
        "expected page in 1..50 because rng.integers uses high=51"
    )


def test_query_set_length(corpus):
    """Verify default query set has 20 stratified queries."""
    q = get_query_set(corpus)
    assert len(q) == 20, "expected 20 queries because n_queries defaults to 20"


def test_query_set_keys(corpus):
    """Verify each query dict exposes the three required fields."""
    q = get_query_set(corpus)
    for qi in q:
        assert set(qi.keys()) == _QUERY_KEYS, (
            f"expected query keys {_QUERY_KEYS} because query schema is fixed"
        )


def test_query_set_topic_distribution(corpus):
    """Verify stratified sampling yields four queries per topic."""
    q = get_query_set(corpus)
    dist = Counter(qi["topic"] for qi in q)
    assert len(dist) == 5, "expected 5 topics because queries are stratified"
    assert all(v == 4 for v in dist.values()), (
        "expected 4 queries per topic because 20 / 5 = 4"
    )


def test_query_set_deterministic(corpus):
    """Verify query set is reproducible for a fixed seed."""
    q1 = get_query_set(corpus, seed=99)
    q2 = get_query_set(corpus, seed=99)
    assert q1[0]["query_text"] == q2[0]["query_text"], (
        "expected identical query text because seed=99 is fixed"
    )
    assert q1[0]["source_chunk_id"] == q2[0]["source_chunk_id"], (
        "expected identical source_chunk_id because seed=99 is fixed"
    )


def test_query_text_is_substring_of_source_chunk(corpus):
    """Verify queries are derived from corpus chunks, not fabricated text."""
    q = get_query_set(corpus)
    for qi in q:
        source_chunk = corpus[qi["source_chunk_id"]]
        assert qi["query_text"] in source_chunk["text"], (
            f"expected query text in source chunk {qi['source_chunk_id']} "
            "because queries use the first sentence of the source chunk"
        )
