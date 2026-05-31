"""Offline tests for embedder structure — hash fallbacks would corrupt semantic retrieval."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from src import embedder
from src.embedder import BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL, _normalize

OCR_DIR = Path(__file__).resolve().parent.parent


def test_embedding_constants() -> None:
    """Verifies embedding config matches Qdrant collection dimension and Voyage model."""
    assert EMBEDDING_DIM == 1024, f"DIM must be 1024, got {EMBEDDING_DIM}"
    assert EMBEDDING_MODEL == "voyage-3", f"Model must be voyage-3, got {EMBEDDING_MODEL}"
    assert BATCH_SIZE == 50, f"BATCH_SIZE must be 50, got {BATCH_SIZE}"


def test_no_hash_fallback_in_embedder() -> None:
    """Verifies no hash fallback — fake vectors would make table Markdown unretrievable."""
    source = (OCR_DIR / "src" / "embedder.py").read_text(encoding="utf-8")
    assert "hashlib" not in source, (
        "embedder.py must have NO hash fallback. Day 6 requires semantic embeddings. "
        "Hash fallback would embed table Markdown as meaningless token hashes."
    )
    assert "blake2" not in source
    assert "sha256" not in source


def test_normalize_zero_vector() -> None:
    """Verifies zero-vector normalize avoids NaN propagation into Qdrant."""
    zero = np.zeros(1024, dtype=np.float32)
    result = _normalize(zero)
    assert result.shape == (1024,)
    assert not np.isnan(result).any(), "Zero vector normalize must not produce NaN"


def test_normalize_produces_unit_vector() -> None:
    """Verifies L2 normalization so cosine similarity equals dot product."""
    vec = np.random.randn(1024).astype(np.float32)
    result = _normalize(vec)
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-5, f"_normalize must produce unit vector, norm={norm}"


def test_normalize_matrix() -> None:
    """Verifies batch normalization for consistent similarity scores across chunks."""
    mat = np.random.randn(5, 1024).astype(np.float32)
    result = _normalize(mat)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"All row norms must be 1.0 after normalize, got: {norms}"
    )


def test_embed_documents_signature() -> None:
    """Verifies embed_chunks API accepts corpus texts for document-type embedding."""
    sig = inspect.signature(embedder.embed_chunks)
    assert "texts" in sig.parameters or "chunks" in sig.parameters, (
        "embed_chunks must accept texts or chunks parameter"
    )


def test_embed_query_signature() -> None:
    """Verifies embed_query API accepts query text for query-type embedding."""
    sig = inspect.signature(embedder.embed_query)
    assert "text" in sig.parameters, "embed_query must accept text parameter"
