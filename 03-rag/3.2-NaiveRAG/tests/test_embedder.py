"""Offline tests for ``src/embedder.py``.

These tests verify module-level contracts that the rest of the
pipeline depends on:

  * The embedding model identifier and dimensionality must match what
    ``vector_store.create_collection`` is configured for. A mismatch
    silently breaks every upsert and every search.
  * ``_normalize`` must produce unit vectors and must handle the
    zero-norm edge case without dividing by zero (we'd inject NaN
    into Qdrant otherwise).
  * The module must NOT ship a hash-based embedding fallback. The
    fallback would produce non-semantic vectors that look healthy at
    insert time but retrieve garbage at query time.

No network calls. No VOYAGE_API_KEY required.
"""

from __future__ import annotations

import inspect
import pathlib

import numpy as np


def test_constants_correct() -> None:
    """Model/dim/batch constants must match the rest of the pipeline."""
    from src.embedder import BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "voyage-3", \
        f"EMBEDDING_MODEL must be 'voyage-3', got {EMBEDDING_MODEL!r}. " \
        "Other Voyage models output different dims and the Qdrant " \
        "collection is configured for 1024 - mismatch = unusable index."
    assert EMBEDDING_DIM == 1024, \
        f"EMBEDDING_DIM must be 1024 (voyage-3 default), got {EMBEDDING_DIM}. " \
        "Any drift here means create_collection builds a wrong-size index " \
        "and every upsert is rejected by Qdrant."
    assert BATCH_SIZE == 50, \
        f"BATCH_SIZE must be 50, got {BATCH_SIZE}. " \
        "Larger batches exceed Voyage's token budget for 300-token chunks; " \
        "smaller batches multiply API calls and slow ingest unnecessarily."


def test_normalize_zero_vector() -> None:
    """Zero vector must stay zero, not become NaN from divide-by-zero."""
    from src.embedder import _normalize
    zero = np.zeros(1024, dtype=np.float32)
    result = _normalize(zero)
    assert result.shape == (1024,), \
        f"Expected shape (1024,), got {result.shape}. " \
        "Shape drift means downstream Qdrant insert fails the dim check."
    assert np.allclose(result, np.zeros(1024)), \
        f"Zero vector did not stay zero, got max abs = {np.max(np.abs(result))}. " \
        "Divide-by-zero -> NaN -> Qdrant rejects the point and ingest crashes mid-batch."


def test_normalize_unit_vector() -> None:
    """A non-zero vector must come back with L2 norm == 1.0."""
    from src.embedder import _normalize
    rng = np.random.default_rng(seed=42)
    vec = rng.standard_normal(1024).astype(np.float32)
    result = _normalize(vec)
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-5, \
        f"Norm {norm} != 1.0 within 1e-5. " \
        "Non-unit vectors break the assumption that dot product == cosine " \
        "similarity, which leaks into retrieval score interpretation."


def test_normalize_matrix() -> None:
    """Matrix normalization must produce unit rows, not unit columns."""
    from src.embedder import _normalize
    rng = np.random.default_rng(seed=7)
    mat = rng.standard_normal((10, 1024)).astype(np.float32)
    result = _normalize(mat)
    assert result.shape == (10, 1024), \
        f"Shape changed during normalize: got {result.shape}, expected (10, 1024)."
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Per-row norms not all 1.0: min={norms.min()}, max={norms.max()}. " \
        "If rows aren't unit-length, scores returned by Qdrant cannot be " \
        "compared across different queries on a common 0..1 scale."


def test_normalize_preserves_direction() -> None:
    """Direction must be preserved (only magnitude is scaled)."""
    from src.embedder import _normalize
    vec = np.array([3.0, 4.0], dtype=np.float32)
    result = _normalize(vec)
    assert np.allclose(result, [0.6, 0.8], atol=1e-5), \
        f"Direction broken: [3, 4] should normalize to [0.6, 0.8], got " \
        f"{result.tolist()}. A direction-warping normalize would silently " \
        "remap embeddings to a different region of the vector space."


def test_embed_documents_function_exists_and_typed() -> None:
    """embed_documents must accept 'texts' and declare its return type."""
    from src import embedder
    sig = inspect.signature(embedder.embed_documents)
    assert "texts" in sig.parameters, \
        f"embed_documents missing 'texts' parameter; got {list(sig.parameters)}. " \
        "The ingest path calls embed_documents(texts=[...]) by name."
    assert sig.return_annotation is not inspect.Parameter.empty, \
        "embed_documents missing return annotation - callers depend on " \
        "the np.ndarray contract for downstream upsert."


def test_embed_query_function_exists_and_typed() -> None:
    """embed_query must accept 'text' and declare its return type."""
    from src import embedder
    sig = inspect.signature(embedder.embed_query)
    assert "text" in sig.parameters, \
        f"embed_query missing 'text' parameter; got {list(sig.parameters)}. " \
        "The retriever calls embed_query(text=...) by name."
    assert sig.return_annotation is not inspect.Parameter.empty, \
        "embed_query missing return annotation - the retriever assumes " \
        "an np.ndarray shape (EMBEDDING_DIM,) for vector_store.search()."


def test_no_fallback_hash_in_embedder() -> None:
    """Embedder must not ship a hash-based fallback - it would retrieve garbage."""
    src_path = pathlib.Path(__file__).resolve().parent.parent / "src" / "embedder.py"
    source = src_path.read_text(encoding="utf-8")
    assert "hashlib" not in source, \
        "embedder.py imports hashlib - a hash fallback embeds text into " \
        "non-semantic space. The Voyage path must be mandatory (Answer B)."
    assert "blake2" not in source, \
        "embedder.py references blake2 - any hash fallback breaks RAG: " \
        "the index looks healthy but every query returns nonsense."
    assert "sha256" not in source, \
        "embedder.py references sha256 - same hash-fallback failure mode."
