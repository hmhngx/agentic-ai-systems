"""Shared pytest fixtures for the VectorDBs benchmark module."""

from __future__ import annotations

import importlib
from uuid import uuid4

import pytest

from src.corpus import generate_corpus, get_query_set
from src.embedder import EMBEDDING_DIM, embed_texts
from src.qdrant_ops import (
    create_flat_collection,
    create_hnsw_collection,
    get_client,
    upsert_chunks,
)


def _reset_embedder_state() -> None:
    """Clear cached embedding client so hash fallback is used."""
    emb = importlib.import_module("src.embedder")
    emb._openrouter_loaded = False
    emb._openrouter_client = None
    emb._fallback_warned = False


def _qdrant_reachable() -> bool:
    """Return True if Qdrant responds on the configured URL."""
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def _delete_test_collections(client) -> None:
    """Remove collections created by integration tests."""
    try:
        names = [c.name for c in client.get_collections().collections]
    except Exception:
        return
    for name in names:
        if name.startswith("test_"):
            try:
                client.delete_collection(collection_name=name)
            except Exception:
                pass


@pytest.fixture(scope="session")
def corpus():
    """500-chunk deterministic corpus shared across tests."""
    return generate_corpus(n_chunks=500, seed=42)


@pytest.fixture(scope="session")
def query_set(corpus):
    """20 stratified queries derived from the session corpus."""
    return get_query_set(corpus, n_queries=20, seed=99)


@pytest.fixture(scope="session")
def small_corpus():
    """50-chunk corpus for integration tests that do not need 500 vectors."""
    return generate_corpus(n_chunks=50, seed=42)


@pytest.fixture(scope="session")
def sample_embeddings(small_corpus, monkeypatch_session):
    """Hash-fallback embeddings for all 50 small_corpus chunks."""
    monkeypatch_session.delenv("OPENROUTER_API_KEY", raising=False)
    _reset_embedder_state()
    texts = [c["text"] for c in small_corpus]
    return embed_texts(texts, input_type="document")


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch for env vars used by sample_embeddings."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def qdrant_client():
    """Qdrant client; skips test if localhost:6333 is unavailable."""
    try:
        client = get_client()
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant not running — integration test skipped")
    yield client
    _delete_test_collections(client)


@pytest.fixture
def populated_collection(qdrant_client, small_corpus, sample_embeddings):
    """HNSW collection with 50 upserted chunks; unique name per test."""
    client = qdrant_client
    name = f"test_chunks_hnsw_{uuid4().hex[:8]}"
    create_hnsw_collection(client, dim=EMBEDDING_DIM, collection_name=name)
    upsert_chunks(client, name, small_corpus, sample_embeddings, batch_size=100)
    yield client, name
    if client.collection_exists(collection_name=name):
        client.delete_collection(collection_name=name)


@pytest.fixture
def populated_flat_collection(qdrant_client, small_corpus, sample_embeddings):
    """FLAT (m=0) collection with 50 upserted chunks; unique name per test."""
    client = qdrant_client
    name = f"test_chunks_flat_{uuid4().hex[:8]}"
    create_flat_collection(client, dim=EMBEDDING_DIM, collection_name=name)
    upsert_chunks(client, name, small_corpus, sample_embeddings, batch_size=100)
    yield client, name
    if client.collection_exists(collection_name=name):
        client.delete_collection(collection_name=name)
