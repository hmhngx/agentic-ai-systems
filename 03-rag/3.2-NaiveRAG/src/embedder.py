"""
embedder.py - Voyage-3 embeddings via voyageai SDK.

Design decision: voyageai SDK is used directly, not via the Anthropic client,
because the Anthropic Python SDK v0.25+ does not expose an .embeddings attribute.
voyageai is Anthropic's preferred embedding provider and the API key can be
the same Voyage AI key or a separate one.

Design decision: input_type is a required parameter, not defaulted silently.
Reason: using "document" for queries or "query" for documents degrades retrieval
quality because voyage-3 optimizes embeddings differently for each role.
Calling code must explicitly declare intent.

Design decision: vectors are L2-normalized before return.
Reason: Qdrant COSINE distance auto-normalizes on insert, but normalizing
explicitly here makes the behavior transparent and allows dot product
computation to work identically to cosine similarity without ambiguity.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import voyageai


EMBEDDING_MODEL: str = "voyage-3"
EMBEDDING_DIM: int = 1024      # voyage-3 default output dimension
BATCH_SIZE: int = 50           # conservative batch size; voyage-3 allows up to 128
                               # but 50 stays within token limits for 300-token chunks


# Cache the voyageai client across calls so a single CLI invocation only
# pays for client construction (and any auth handshake) once.
_voyage_client: "Optional[voyageai.Client]" = None


def _get_client() -> voyageai.Client:
    """Lazily construct a Voyage client from ``VOYAGE_API_KEY``.

    We intentionally do NOT silently fall back to a hash embedding here:
    this module is the production embedding path for the RAG pipeline,
    and silent fallbacks would produce non-semantic vectors that retrieve
    garbage from Qdrant. If the key is missing we fail loudly so the
    caller can decide what to do (the CLI in ``naive_rag.py`` warns the
    user up front).
    """
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client

    api_key: Optional[str] = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. voyage-3 embeddings require a Voyage AI "
            "API key. Add it to .env or export it in your shell, then retry."
        )
    _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


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
    """Embed corpus chunks. ``input_type="document"``.

    Returns a ``(len(texts), EMBEDDING_DIM)`` float32 array, L2-normalized.

    Why ``input_type="document"``?
        voyage-3 is asymmetric: documents and queries are routed to
        slightly different output regions. The "document" mode optimizes
        for the role of being retrievable; the "query" mode optimizes
        for the role of doing the retrieving.
    """
    if not texts:
        raise ValueError("embed_documents called with empty texts list")

    client: voyageai.Client = _get_client()
    rows: list[list[float]] = []
    total_batches: int = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
        batch: list[str] = texts[start : start + BATCH_SIZE]
        print(f"Embedding batch {i}/{total_batches}...")
        try:
            result = client.embed(
                texts=batch,                  # up to BATCH_SIZE chunk strings per call
                model=EMBEDDING_MODEL,        # voyage-3: 1024-dim general-purpose model
                input_type="document",        # marks these as retrievable corpus chunks
            )
        except Exception as exc:  # noqa: BLE001 - voyageai raises a tree of subclasses
            # Re-raise with batch context so debugging a partial failure
            # tells you which batch broke and at what index range.
            raise RuntimeError(
                f"embed_documents failed on batch {i}: {exc}"
            ) from exc
        rows.extend(result.embeddings)

    matrix: np.ndarray = np.asarray(rows, dtype=np.float32)
    return _normalize(matrix)


def embed_query(text: str) -> np.ndarray:
    """Embed a single user query. ``input_type="query"``.

    Returns a ``(EMBEDDING_DIM,)`` float32 array, L2-normalized.

    Why ``input_type="query"``?
        voyage-3 uses asymmetric embedding - queries and documents are
        optimized for different distributions. Using "document" for a
        query degrades recall because the query vector lands in the
        wrong region of the embedding space.
    """
    if not text or not text.strip():
        raise ValueError("embed_query called with empty text")

    client: voyageai.Client = _get_client()
    try:
        result = client.embed(
            texts=[text],                 # single query string wrapped in a list (SDK contract)
            model=EMBEDDING_MODEL,        # same model as documents - asymmetry is via input_type
            input_type="query",           # marks this as the lookup intent, not retrievable doc
        )
    except Exception as exc:  # noqa: BLE001 - voyageai raises a tree of subclasses
        raise RuntimeError(f"embed_query failed: {exc}") from exc

    vec: np.ndarray = np.asarray(result.embeddings[0], dtype=np.float32)
    return _normalize(vec)
