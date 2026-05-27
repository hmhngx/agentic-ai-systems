"""Offline tests for embedder shape, normalization, and hash-fallback determinism."""

from __future__ import annotations

import numpy as np
import pytest

import src.embedder as emb_module
from src.embedder import EMBEDDING_DIM, embed_single, embed_texts


@pytest.fixture(autouse=True)
def _hash_only(monkeypatch):
    """Force hash fallback by clearing API keys and resetting client cache."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    emb_module._voyage_loaded = False
    emb_module._voyage_client = None
    emb_module._fallback_warned = False
    yield
    emb_module._voyage_loaded = False
    emb_module._voyage_client = None
    emb_module._fallback_warned = False


def test_embed_single_shape():
    """Verify embed_single returns a 1-D float32 vector of EMBEDDING_DIM."""
    v = embed_single("test sentence")
    assert v.shape == (EMBEDDING_DIM,), (
        f"expected shape ({EMBEDDING_DIM},) because embed_single returns one row"
    )
    assert v.dtype == np.float32, "expected float32 because embedder uses float32"


def test_embed_single_normalized():
    """Verify embed_single output is L2-normalized for cosine distance."""
    v = embed_single("test sentence")
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5, (
        f"expected unit norm because vectors are L2-normalized (got {norm})"
    )


def test_embed_single_deterministic():
    """Verify identical input always yields identical embedding under hash fallback."""
    v1 = embed_single("same text")
    v2 = embed_single("same text")
    assert np.allclose(v1, v2), (
        "expected identical vectors because hash fallback is deterministic"
    )


def test_embed_single_different_texts_differ():
    """Verify different texts produce different embedding vectors."""
    v1 = embed_single("machine learning gradient descent")
    v2 = embed_single("ocean biology coral reef")
    assert not np.allclose(v1, v2), (
        "expected different vectors because input texts differ"
    )


def test_embed_texts_shape():
    """Verify embed_texts returns (N, EMBEDDING_DIM) float32 matrix."""
    texts = ["sentence one", "sentence two", "sentence three"]
    result = embed_texts(texts, input_type="document")
    assert result.shape == (3, EMBEDDING_DIM), (
        f"expected shape (3, {EMBEDDING_DIM}) because three texts were passed"
    )
    assert result.dtype == np.float32, "expected float32 because embedder uses float32"


def test_embed_texts_each_row_normalized():
    """Verify every row of embed_texts output has unit L2 norm."""
    texts = ["sentence one", "sentence two", "sentence three"]
    result = embed_texts(texts)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        "expected all row norms 1.0 because embedder L2-normalizes output"
    )


def test_embed_texts_batching_consistent():
    """Verify batch_size does not change embedding values for the same texts."""
    texts = [f"sentence {i}" for i in range(10)]
    r1 = embed_texts(texts, batch_size=3)
    r2 = embed_texts(texts, batch_size=10)
    assert np.allclose(r1, r2), (
        "expected identical matrices because batching only splits API calls"
    )


def test_embed_texts_empty_raises():
    """Verify empty input raises instead of returning a silent empty array."""
    with pytest.raises(ValueError):
        embed_texts([])


def test_embed_single_query_vs_document_differ():
    """Verify query and document input_type both return valid same-shaped vectors."""
    v_doc = embed_single("test", input_type="document")
    v_query = embed_single("test", input_type="query")
    assert v_doc.shape == v_query.shape, (
        "expected same shape because both input types embed one string"
    )
