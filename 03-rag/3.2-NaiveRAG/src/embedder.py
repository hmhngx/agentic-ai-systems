"""
embedder.py - OpenRouter embeddings via OpenAI-compatible API.

Design decision: use OpenRouter's OpenAI-compatible endpoint so we can keep
the provider wiring in one place and avoid vendor-specific SDK coupling.

Design decision: vectors are L2-normalized before return.
Reason: Qdrant COSINE distance auto-normalizes on insert, but normalizing
explicitly here makes the behavior transparent and allows dot product
computation to work identically to cosine similarity without ambiguity.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from openai import OpenAI


EMBEDDING_MODEL: str = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_DIM: int = int(os.environ.get("OPENROUTER_EMBEDDING_DIM", "1536"))
BATCH_SIZE: int = 50


# Cache the OpenRouter client across calls so a single CLI invocation only
# pays for client construction (and any auth handshake) once.
_openrouter_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazily construct an OpenRouter client from ``OPENROUTER_API_KEY``.

    We intentionally do NOT silently fall back to a hash embedding here:
    this module is the production embedding path for the RAG pipeline,
    and silent fallbacks would produce non-semantic vectors that retrieve
    garbage from Qdrant. If the key is missing we fail loudly so the
    caller can decide what to do (the CLI in ``naive_rag.py`` warns the
    user up front).
    """
    global _openrouter_client
    if _openrouter_client is not None:
        return _openrouter_client

    api_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. OpenRouter embeddings require an API key. "
            "Add it to .env or export it in your shell, then retry."
        )
    base_url: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    _openrouter_client = OpenAI(api_key=api_key, base_url=base_url)
    return _openrouter_client


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or a matrix of row vectors.

    Handles the zero-norm edge case by leaving zero rows at zero rather
    than dividing by zero. A zero embedding only happens for pathological
    inputs (e.g. an all-whitespace string that slipped through filters);
    surfacing it as a zero vector is better than NaN propagation.
    """
    if vec.ndim == 1:
        norm: float = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec.astype(np.float32, copy=False)
        return (vec / norm).astype(np.float32, copy=False)

    # 2D case: per-row L2 normalization.
    norms: np.ndarray = np.linalg.norm(vec, axis=1, keepdims=True)
    # Replace zeros with 1.0 so the division is a no-op for those rows;
    # we never divide by zero, and zero rows remain zero.
    safe_norms: np.ndarray = np.where(norms == 0.0, 1.0, norms)
    return (vec / safe_norms).astype(np.float32, copy=False)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed corpus chunks.

    Returns a ``(len(texts), EMBEDDING_DIM)`` float32 array, L2-normalized.

    Uses OpenRouter's OpenAI-compatible embeddings endpoint.
    """
    if not texts:
        raise ValueError("embed_documents called with empty texts list")

    client: OpenAI = _get_client()
    rows: list[list[float]] = []
    total_batches: int = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
        batch: list[str] = texts[start : start + BATCH_SIZE]
        print(f"Embedding batch {i}/{total_batches}...")
        try:
            result = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises provider-specific subclasses
            # Re-raise with batch context so debugging a partial failure
            # tells you which batch broke and at what index range.
            raise RuntimeError(
                f"embed_documents failed on batch {i}: {exc}"
            ) from exc
        rows.extend([item.embedding for item in result.data])

    matrix: np.ndarray = np.asarray(rows, dtype=np.float32)
    return _normalize(matrix)


def embed_query(text: str) -> np.ndarray:
    """Embed a single user query.

    Returns a ``(EMBEDDING_DIM,)`` float32 array, L2-normalized.

    Uses the same embedding model as document chunks.
    """
    if not text or not text.strip():
        raise ValueError("embed_query called with empty text")

    client: OpenAI = _get_client()
    try:
        result = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises provider-specific subclasses
        raise RuntimeError(f"embed_query failed: {exc}") from exc

    vec: np.ndarray = np.asarray(result.data[0].embedding, dtype=np.float32)
    return _normalize(vec)
