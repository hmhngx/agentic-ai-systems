"""Integration tests for ``src/vector_store.py``.

Every test in this file talks to a real Qdrant instance on
``localhost:6333``. The ``qdrant_client`` fixture in tests/conftest.py
auto-skips the entire file when Qdrant is unreachable, so these tests
contribute zero failures in pure-offline runs (e.g. CI without
docker).

We deliberately do not mock the qdrant client: vector_store.py is
THE adapter layer around Qdrant, and a mock would just be testing
the mock's behavior, not Qdrant's contract. The point of this file
is to catch breaking Qdrant-version changes early.
"""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest

from src.vector_store import (
    collection_exists,
    create_collection,
    get_client,
    search,
    upsert_chunks,
)


pytestmark = pytest.mark.integration


def _vector_count(client, collection_name: str) -> int:
    """Read the vector count via the canonical ``count`` API.

    The ``info.vectors_count`` / ``info.points_count`` attributes are
    inconsistent across qdrant-client versions (older versions populate
    the former, newer the latter, some populate neither until indexing
    finishes). ``client.count(exact=True)`` is the version-stable path.
    """
    return int(client.count(collection_name=collection_name, exact=True).count)


def test_get_client_connects(qdrant_client) -> None:
    """get_client() must reach Qdrant and return a usable client object."""
    client = get_client()
    info = client.get_collections()
    assert hasattr(info, "collections"), \
        f"client.get_collections() returned {type(info).__name__} without " \
        "a 'collections' attribute - the qdrant-client API contract changed " \
        "and the rest of vector_store.py needs an audit."


def test_create_collection_exists_after_creation(qdrant_client) -> None:
    """create_collection must actually create the named collection on the server."""
    test_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
    names = [c.name for c in qdrant_client.get_collections().collections]
    assert test_name in names, \
        f"Collection '{test_name}' not found in Qdrant after creation; " \
        f"existing: {names}. " \
        "If create_collection silently no-ops, the next upsert raises a " \
        "confusing 'collection not found' error far from the root cause."


def test_create_collection_hnsw_config(qdrant_client) -> None:
    """HNSW config must match the documented production values (m=16, ef=200)."""
    test_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
    info = qdrant_client.get_collection(test_name)

    assert info.config.hnsw_config.m == 16, \
        f"HNSW m={info.config.hnsw_config.m}, expected 16. " \
        "m controls graph connectivity; deviating from 16 trades recall " \
        "and RAM in ways the team has not benchmarked."
    assert info.config.hnsw_config.ef_construct == 200, \
        f"HNSW ef_construct={info.config.hnsw_config.ef_construct}, expected 200. " \
        "ef_construct controls build-time graph quality; lower values hurt " \
        "recall, higher values slow ingest."
    assert info.config.params.vectors.size == 1024, \
        f"Vector size={info.config.params.vectors.size}, expected 1024. " \
        "Mismatch with EMBEDDING_DIM means every upsert is rejected."


def test_upsert_chunks_correct_count(qdrant_client, sample_chunks) -> None:
    """upsert_chunks must persist exactly len(chunks) points to the collection."""
    test_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
        rng = np.random.default_rng(seed=11)
        embeddings = rng.standard_normal((len(sample_chunks), 1024)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        upsert_chunks(qdrant_client, sample_chunks, embeddings)

        count = _vector_count(qdrant_client, test_name)

    assert count == len(sample_chunks), \
        f"Expected {len(sample_chunks)} vectors in collection, got {count}. " \
        "A count mismatch means we silently lost chunks during upsert - " \
        "the user asks about page 3 and the retriever has no chunk to return."


def test_upsert_payload_preserved(qdrant_client, sample_chunks) -> None:
    """Payload (text/page_num/source_pdf/chunk_id) must survive round-trip to Qdrant."""
    from src.vector_store import _uuid_to_point_id

    test_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
        rng = np.random.default_rng(seed=22)
        embeddings = rng.standard_normal((len(sample_chunks), 1024)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        upsert_chunks(qdrant_client, sample_chunks, embeddings)

    sample_chunk = sample_chunks[0]
    point_id = _uuid_to_point_id(sample_chunk["chunk_id"])
    points = qdrant_client.retrieve(
        collection_name=test_name, ids=[point_id], with_payload=True
    )
    assert len(points) == 1, \
        f"Expected 1 point retrieved by id, got {len(points)}. " \
        "Point id derivation broke - chunks are unreachable by chunk_id."

    payload = points[0].payload
    assert "text" in payload, \
        "Payload missing 'text' - the LLM has nothing to ground on at " \
        "retrieval time; format_context_block would emit empty bodies."
    assert "page_num" in payload, \
        "Payload missing 'page_num' - citations cannot reference a page."
    assert "source_pdf" in payload, \
        "Payload missing 'source_pdf' - footer cannot show the source file."
    assert "chunk_id" in payload, \
        "Payload missing 'chunk_id' - we lose the stable cross-system handle " \
        "we need for log correlation and dedup."


def test_search_returns_top_k(qdrant_client, sample_chunks) -> None:
    """search must return at most top_k well-formed result dicts in rank order."""
    test_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
        rng = np.random.default_rng(seed=33)
        embeddings = rng.standard_normal((len(sample_chunks), 1024)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        upsert_chunks(qdrant_client, sample_chunks, embeddings)

        query_vec = rng.standard_normal(1024).astype(np.float32)
        query_vec = query_vec / float(np.linalg.norm(query_vec))
        results = search(qdrant_client, query_vec, top_k=3)

    assert len(results) <= 3, \
        f"top_k=3 returned {len(results)} results - search ignored its " \
        "limit and will overflow downstream prompt-token budgets."
    assert all(isinstance(r["score"], float) for r in results), \
        f"Some scores are non-float: " \
        f"{[type(r['score']).__name__ for r in results]}. " \
        "Threshold comparison (>= 0.40) breaks if scores are int or None."
    assert all(1 <= r["rank"] <= 3 for r in results), \
        f"Ranks outside [1, 3]: {[r['rank'] for r in results]}. " \
        "Ranks are 1-indexed for human-readable display in the footer."


def test_search_top_k_cap_enforced(qdrant_client, sample_chunks) -> None:
    """top_k > 5 must be clamped to 5 - defense against the topK pathology."""
    test_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
        rng = np.random.default_rng(seed=44)
        embeddings = rng.standard_normal((len(sample_chunks), 1024)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        upsert_chunks(qdrant_client, sample_chunks, embeddings)

        query_vec = rng.standard_normal(1024).astype(np.float32)
        query_vec = query_vec / float(np.linalg.norm(query_vec))
        with nullcontext():
            results = search(qdrant_client, query_vec, top_k=10)

    assert len(results) <= 5, \
        f"top_k cap not enforced: got {len(results)} results for top_k=10. " \
        "Above 5 chunks the LLM's attention dilutes across distractors and " \
        "answer quality degrades (the documented topK pathology)."


def test_collection_exists_true_after_upsert(qdrant_client, sample_chunks) -> None:
    """collection_exists must return True once a populated collection exists."""
    test_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
        rng = np.random.default_rng(seed=55)
        embeddings = rng.standard_normal((len(sample_chunks), 1024)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        upsert_chunks(qdrant_client, sample_chunks, embeddings)

        assert collection_exists(qdrant_client) is True, \
            "collection_exists returned False after a successful upsert - " \
            "the CLI would re-ingest every run, wasting Voyage API calls."


def test_collection_exists_false_for_missing(qdrant_client) -> None:
    """collection_exists must return False for a name that doesn't exist."""
    with patch("src.vector_store.COLLECTION_NAME", "definitely_does_not_exist_xyz"):
        assert collection_exists(qdrant_client) is False, \
            "collection_exists returned True for a non-existent collection - " \
            "the CLI would skip ingestion and then crash on first search."
