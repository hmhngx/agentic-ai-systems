"""Integration tests for Qdrant client operations (auto-skip if Qdrant unavailable)."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client.models import Distance

from src.embedder import EMBEDDING_DIM, embed_single
from src.qdrant_ops import (
    count_vectors,
    create_hnsw_collection,
    get_client,
    search_flat,
    search_hnsw,
)

_VALID_TOPICS = {
    "machine_learning",
    "ocean_biology",
    "ancient_history",
    "cooking",
    "urban_architecture",
}

pytestmark = pytest.mark.integration


def test_get_client_connects(qdrant_client):
    """Verify get_client can reach Qdrant and list collections."""
    client = get_client()
    info = client.get_collections()
    assert hasattr(info, "collections"), (
        "expected collections attribute because get_collections returns a response object"
    )


def test_create_hnsw_collection_creates_collection(qdrant_client):
    """Verify create_hnsw_collection registers a new collection in Qdrant."""
    client = qdrant_client
    name = f"test_hnsw_{uuid.uuid4().hex[:8]}"
    create_hnsw_collection(client, dim=EMBEDDING_DIM, collection_name=name)
    collections = [c.name for c in client.get_collections().collections]
    assert name in collections, (
        f"expected {name} in collection list because create_hnsw_collection just ran"
    )
    if client.collection_exists(collection_name=name):
        client.delete_collection(collection_name=name)


def test_hnsw_collection_has_correct_config(populated_collection):
    """Verify HNSW collection vector size, distance, and graph parameters."""
    client, collection_name = populated_collection
    info = client.get_collection(collection_name)
    assert info.config.params.vectors.size == EMBEDDING_DIM, (
        f"expected vector size {EMBEDDING_DIM} because collection was created with that dim"
    )
    assert info.config.params.vectors.distance == Distance.COSINE, (
        "expected COSINE distance because semantic embeddings use cosine similarity"
    )
    assert info.config.hnsw_config.m == 16, (
        "expected m=16 because create_hnsw_collection sets production default"
    )
    assert info.config.hnsw_config.ef_construct == 200, (
        "expected ef_construct=200 because create_hnsw_collection sets baseline"
    )


def test_flat_collection_has_m_zero(populated_flat_collection):
    """Verify FLAT collection disables HNSW graph (m=0)."""
    client, collection_name = populated_flat_collection
    info = client.get_collection(collection_name)
    assert info.config.hnsw_config.m == 0, (
        "expected m=0 because create_flat_collection disables the HNSW graph"
    )


def test_upsert_correct_count(populated_collection):
    """Verify all 50 small_corpus chunks were upserted."""
    client, collection_name = populated_collection
    n = count_vectors(client, collection_name)
    assert n == 50, "expected 50 points because small_corpus has 50 chunks"


def test_upsert_payload_preserved(populated_collection):
    """Verify chunk metadata is stored in Qdrant payload fields."""
    client, collection_name = populated_collection
    points = client.retrieve(collection_name=collection_name, ids=[0], with_payload=True)
    assert len(points) == 1, "expected one point because id=0 was requested"
    payload = points[0].payload
    assert "topic" in payload, "expected topic in payload because upsert stores it"
    assert "source" in payload, "expected source in payload because upsert stores it"
    assert "text" in payload, "expected text in payload because upsert stores it"
    assert "page" in payload, "expected page in payload because upsert stores it"
    assert payload["topic"] in _VALID_TOPICS, (
        f"expected topic in {_VALID_TOPICS} because corpus uses those keys"
    )


def test_search_hnsw_returns_top_k(populated_collection):
    """Verify HNSW search returns top_k integer IDs and a positive latency."""
    client, collection_name = populated_collection
    query_vec = embed_single("neural networks gradient descent")
    ids, latency_ms = search_hnsw(
        client,
        query_vec,
        top_k=5,
        ef_search=64,
        collection_name=collection_name,
    )
    assert len(ids) == 5, "expected 5 ids because top_k=5"
    assert all(isinstance(i, int) for i in ids), (
        "expected int point ids because chunk ids are integers"
    )
    assert latency_ms > 0.0, "expected positive latency because search executed"
    assert latency_ms < 5000.0, (
        "expected latency under 5s because 50-vector search should be fast"
    )


def test_search_hnsw_ids_are_valid(populated_collection):
    """Verify returned IDs are within the 0..49 range of the small corpus."""
    client, collection_name = populated_collection
    query_vec = embed_single("neural networks gradient descent")
    ids, _ = search_hnsw(
        client,
        query_vec,
        top_k=5,
        ef_search=64,
        collection_name=collection_name,
    )
    assert all(0 <= i < 50 for i in ids), (
        "expected ids in 0..49 because collection has 50 chunks"
    )


def test_search_flat_returns_top_k(populated_collection):
    """Verify exact search returns top_k results with positive latency."""
    client, collection_name = populated_collection
    query_vec = embed_single("coral reef marine biology")
    ids, latency_ms = search_flat(
        client, query_vec, top_k=5, collection_name=collection_name
    )
    assert len(ids) == 5, "expected 5 ids because top_k=5"
    assert latency_ms > 0.0, "expected positive latency because search executed"


def test_search_flat_is_deterministic(populated_collection):
    """Verify exact search returns the same ranking on repeated calls."""
    client, collection_name = populated_collection
    query_vec = embed_single("coral reef marine biology")
    ids1, _ = search_flat(
        client, query_vec, top_k=5, collection_name=collection_name
    )
    ids2, _ = search_flat(
        client, query_vec, top_k=5, collection_name=collection_name
    )
    assert ids1 == ids2, (
        "expected identical id lists because exact search is deterministic"
    )


def test_payload_filter_reduces_results_to_topic(populated_collection):
    """Verify topic_filter restricts results to the requested payload topic."""
    client, collection_name = populated_collection
    query_vec = embed_single("machine learning")
    ids, _ = search_hnsw(
        client,
        query_vec,
        top_k=5,
        ef_search=64,
        topic_filter="machine_learning",
        collection_name=collection_name,
    )
    points = client.retrieve(
        collection_name=collection_name, ids=ids, with_payload=True
    )
    returned_topics = {p.payload["topic"] for p in points}
    assert returned_topics == {"machine_learning"}, (
        f"expected only machine_learning topics but got {returned_topics}"
    )


def test_search_hnsw_ef_search_affects_latency(populated_collection):
    """Verify different ef_search values run without error and return top_k ids."""
    client, collection_name = populated_collection
    query_vec = embed_single("test query for latency")
    ids_fast, lat_fast = search_hnsw(
        client,
        query_vec,
        top_k=5,
        ef_search=16,
        collection_name=collection_name,
    )
    ids_slow, lat_slow = search_hnsw(
        client,
        query_vec,
        top_k=5,
        ef_search=256,
        collection_name=collection_name,
    )
    assert len(ids_fast) == 5, "expected 5 ids for ef_search=16"
    assert len(ids_slow) == 5, "expected 5 ids for ef_search=256"
    assert lat_fast > 0.0, "expected positive latency for ef_search=16"
    assert lat_slow > 0.0, "expected positive latency for ef_search=256"
