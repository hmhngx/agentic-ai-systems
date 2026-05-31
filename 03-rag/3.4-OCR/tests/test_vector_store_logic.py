"""Offline source checks and Qdrant integration tests for vector_store metadata filters."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import numpy as np
import pytest

from src.vector_store import (
    _uuid_to_point_id,
    collection_exists,
    create_collection,
    search_by_type,
    upsert_chunks,
)

OCR_DIR = Path(__file__).resolve().parent.parent


def test_search_by_type_uses_qdrant_filter_not_python() -> None:
    """Verifies server-side Qdrant filters — Python post-filtering fails at scale."""
    source = (OCR_DIR / "src" / "vector_store.py").read_text(encoding="utf-8")
    assert "FieldCondition" in source, (
        "search_by_type must use Qdrant FieldCondition for server-side filtering. "
        "Python post-filtering requires fetching all chunks — catastrophic at scale."
    )
    assert "MatchValue" in source or "MatchAny" in source, (
        "search_by_type must use Qdrant MatchValue/MatchAny for type filtering"
    )
    assert "Filter(" in source or "Filter(must" in source, (
        "search_by_type must construct a Qdrant Filter object"
    )


def test_payload_fields_documented() -> None:
    """Verifies all RAG payload fields are stored for section and type filtering."""
    source = (OCR_DIR / "src" / "vector_store.py").read_text(encoding="utf-8")
    required_payload_fields = [
        "chunk_type",
        "region_type",
        "page_num",
        "heading_path",
        "source_pdf",
        "reading_order",
        "table_title",
        "token_count",
    ]
    for field in required_payload_fields:
        assert f'"{field}"' in source or f"'{field}'" in source, (
            f"Payload field '{field}' not referenced in vector_store.py"
        )


def test_collection_name_from_env() -> None:
    """Verifies separate collection name avoids colliding with Day 3/4 indexes."""
    source = (OCR_DIR / "src" / "vector_store.py").read_text(encoding="utf-8")
    assert "COLLECTION_NAME" in source
    assert "doc_ingest_ocr" in source, (
        "Default collection name must be 'doc_ingest_ocr' to avoid colliding with Day 3/4"
    )


def _sample_chunks() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(uuid.uuid4()),
            "text": "Neural networks backpropagation",
            "chunk_type": "prose",
            "region_type": "text",
            "page_num": 1,
            "heading_path": ["Chapter 1"],
            "source_pdf": "test.pdf",
            "reading_order": 0,
            "bbox": [0, 0, 1, 1],
            "table_title": None,
            "chunk_index": 0,
            "token_count": 8,
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "text": "[TABLE: Accuracy] | Model | Score |\n|---|---|\n| BERT | 0.91 |",
            "chunk_type": "table",
            "region_type": "table",
            "page_num": 2,
            "heading_path": ["Chapter 2"],
            "source_pdf": "test.pdf",
            "reading_order": 5,
            "bbox": [0, 0, 1, 1],
            "table_title": "Accuracy",
            "chunk_index": 0,
            "token_count": 15,
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "text": "Conclusion and future work",
            "chunk_type": "prose",
            "region_type": "text",
            "page_num": 3,
            "heading_path": ["Chapter 3"],
            "source_pdf": "test.pdf",
            "reading_order": 10,
            "bbox": [0, 0, 1, 1],
            "table_title": None,
            "chunk_index": 0,
            "token_count": 5,
        },
    ]


@pytest.fixture
def seeded_qdrant_collection(qdrant_client: Any) -> Generator[tuple[str, list[dict[str, Any]]], None, None]:
    """Create a temporary collection with three mixed-type chunks for integration tests."""
    test_name = f"test_ocr_{uuid.uuid4().hex[:8]}"
    chunks = _sample_chunks()
    embeddings = np.random.randn(3, 1024).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    with patch("src.vector_store.COLLECTION_NAME", test_name):
        create_collection(qdrant_client, dim=1024)
        upsert_chunks(qdrant_client, chunks, embeddings)
        yield test_name, chunks

    try:
        if qdrant_client.collection_exists(collection_name=test_name):
            qdrant_client.delete_collection(collection_name=test_name)
    except Exception:
        pass


@pytest.mark.integration
def test_create_collection_and_upsert(qdrant_client: Any) -> None:
    """Verifies collection creation and upsert store vectors for retrieval smoke tests."""
    test_name = f"test_ocr_{uuid.uuid4().hex[:8]}"
    chunks = _sample_chunks()
    embeddings = np.random.randn(3, 1024).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    try:
        with patch("src.vector_store.COLLECTION_NAME", test_name):
            create_collection(qdrant_client, dim=1024)
            assert qdrant_client.collection_exists(collection_name=test_name), (
                "Collection must exist after create_collection — "
                "without a target collection, upsert has nowhere to store chunks"
            )
            upsert_chunks(qdrant_client, chunks, embeddings)
            assert collection_exists(qdrant_client) is True, (
                "collection_exists must be True after upsert — "
                "pipeline skips ingest when it thinks the index is empty"
            )

            count_result = qdrant_client.count(collection_name=test_name, exact=True)
            assert int(count_result.count) == 3, (
                "Expected 3 vectors after upsert — incomplete index breaks retrieval"
            )
    finally:
        try:
            if qdrant_client.collection_exists(collection_name=test_name):
                qdrant_client.delete_collection(collection_name=test_name)
        except Exception:
            pass


@pytest.mark.integration
def test_search_by_type_returns_only_table_chunks(
    qdrant_client: Any,
    seeded_qdrant_collection: tuple[str, list[dict[str, Any]]],
) -> None:
    """Verifies chunk_type filter returns only table chunks for table-specific queries."""
    test_name, _chunks = seeded_qdrant_collection
    query_vec = np.random.randn(1024).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    with patch("src.vector_store.COLLECTION_NAME", test_name):
        results = search_by_type(qdrant_client, query_vec, "table", top_k=5)

    for r in results:
        assert r.get("chunk_type") == "table", (
            f"search_by_type('table') returned a non-table chunk: {r.get('chunk_type')}"
        )


@pytest.mark.integration
def test_payload_heading_path_retrievable(
    qdrant_client: Any,
    seeded_qdrant_collection: tuple[str, list[dict[str, Any]]],
) -> None:
    """Verifies heading_path payload survives upsert for section-scoped filtering."""
    test_name, chunks = seeded_qdrant_collection
    prose_chunk = chunks[0]
    point_id = _uuid_to_point_id(prose_chunk["chunk_id"])

    results = qdrant_client.retrieve(
        collection_name=test_name,
        ids=[point_id],
        with_payload=True,
    )
    assert results, "Expected to retrieve upserted prose chunk by point ID"
    payload = results[0].payload
    assert payload is not None
    assert "heading_path" in payload
    assert isinstance(payload["heading_path"], list)
    assert "Chapter 1" in payload["heading_path"]
