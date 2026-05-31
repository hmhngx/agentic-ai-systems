"""
embedder.py — Embedding with voyageai (primary) or OpenRouter (fallback).

Design decision: voyage-3 is the primary embedding model.
  It handles mixed content (prose + table Markdown + headings) well.
  The [TABLE: title] prefix and Markdown formatting embed meaningfully.

input_type distinction is critical:
  "document": corpus chunks being indexed (tells model: this is retrievable content)
  "query":    search queries (tells model: this is lookup intent)
  Using the wrong input_type degrades retrieval — document and query
  embeddings occupy different regions of the voyage-3 output space.

For table chunks specifically:
  Use input_type="document" — the Markdown table text is the document.
  The [TABLE: title] prefix ensures the table title is near the beginning
  of the embedding context, giving it higher attention weight.

Batch strategy:
  Batch all chunks together regardless of type.
  Do NOT use different embedding models for prose vs tables.
  Same model = same vector space = comparable similarity scores.
  Using different models would make cross-type similarity meaningless.

Strict Voyage only: there is deliberately NO hash-embedding fallback.
  This module requires semantically meaningful vectors — a hash fallback
  would silently fill Qdrant with noise that retrieves garbage, defeating
  the entire point of the OCR ingestion pipeline. If the key is missing we
  fail loudly so the caller (the CLI) can stop before doing useless work.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import voyageai
from voyageai.error import RateLimitError


EMBEDDING_DIM: int = 1024
EMBEDDING_MODEL: str = "voyage-3"
BATCH_SIZE: int = 50
# Voyage free-tier without billing is 3 RPM; wait one full minute window before retry.
_RATE_LIMIT_WAIT_SECONDS: int = 21
_RATE_LIMIT_MAX_RETRIES: int = 5


# Cache the Voyage client across calls so a single CLI invocation only pays
# for client construction (and any auth handshake) once.
_voyage_client: Optional[voyageai.Client] = None


def _get_client() -> voyageai.Client:
    """Lazily construct a Voyage client from ``VOYAGE_API_KEY``.

    Fails loudly (no silent hash fallback) when the key is absent: this is
    the production embedding path and non-semantic vectors would corrupt
    the index. See the module docstring for the full rationale.
    """
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client

    api_key: Optional[str] = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. This module requires real semantic "
            "embeddings (voyage-3). Add the key to .env or export it, then retry."
        )
    _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or a matrix of row vectors.

    Handles the zero-norm edge case by leaving zero rows at zero rather than
    dividing by zero. A zero embedding only happens for pathological inputs;
    surfacing it as a zero vector is better than NaN propagation. Qdrant's
    COSINE distance also normalizes on insert, but doing it here makes the
    behavior explicit and keeps dot-product == cosine downstream.
    """
    if vec.ndim == 1:
        norm: float = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec.astype(np.float32, copy=False)
        return (vec / norm).astype(np.float32, copy=False)

    norms: np.ndarray = np.linalg.norm(vec, axis=1, keepdims=True)
    # Replace zero norms with 1.0 so the division is a no-op for those rows.
    safe_norms: np.ndarray = np.where(norms == 0.0, 1.0, norms)
    return (vec / safe_norms).astype(np.float32, copy=False)


def _embed_call(
    client: voyageai.Client,
    texts: list[str],
    input_type: str,
    *,
    context: str,
) -> list[list[float]]:
    """Call Voyage embed with retries when the org hits reduced RPM limits."""
    last_exc: Exception | None = None
    for attempt in range(1, _RATE_LIMIT_MAX_RETRIES + 1):
        try:
            result = client.embed(
                texts,
                model=EMBEDDING_MODEL,
                input_type=input_type,
            )
            return result.embeddings
        except RateLimitError as exc:
            last_exc = exc
            if attempt >= _RATE_LIMIT_MAX_RETRIES:
                break
            print(
                f"Rate limit hit during {context} "
                f"(attempt {attempt}/{_RATE_LIMIT_MAX_RETRIES}); "
                f"waiting {_RATE_LIMIT_WAIT_SECONDS}s..."
            )
            time.sleep(_RATE_LIMIT_WAIT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - SDK raises provider-specific subclasses
            raise RuntimeError(f"{context} failed: {exc}") from exc
    raise RuntimeError(f"{context} failed after rate-limit retries: {last_exc}") from last_exc


def embed_chunks(texts: list[str]) -> np.ndarray:
    """Embed corpus chunks with ``input_type="document"``.

    Returns a ``(len(texts), EMBEDDING_DIM)`` float32 array, L2-normalized.
    Batches at ``BATCH_SIZE`` to respect Voyage request limits and to keep
    a partial failure attributable to a specific batch.
    """
    if not texts:
        raise ValueError("embed_chunks called with empty texts list")

    client: voyageai.Client = _get_client()
    rows: list[list[float]] = []
    total_batches: int = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
        batch: list[str] = texts[start : start + BATCH_SIZE]
        print(f"Embedding batch {batch_num}/{total_batches}...")
        rows.extend(
            _embed_call(
                client,
                batch,
                "document",
                context=f"embed_chunks batch {batch_num}",
            )
        )

    matrix: np.ndarray = np.asarray(rows, dtype=np.float32)
    return _normalize(matrix)


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query with ``input_type="query"``.

    Returns a ``(EMBEDDING_DIM,)`` float32 array, L2-normalized. Uses the
    same model as :func:`embed_chunks` so query and document vectors are
    directly comparable under cosine similarity.
    """
    if not text or not text.strip():
        raise ValueError("embed_query called with empty text")

    client: voyageai.Client = _get_client()
    rows: list[list[float]] = _embed_call(
        client,
        [text],
        "query",
        context="embed_query",
    )
    vec: np.ndarray = np.asarray(rows[0], dtype=np.float32)
    return _normalize(vec)


def embed_queries(texts: list[str]) -> np.ndarray:
    """Embed multiple search queries in one API call with ``input_type="query"``.

    Batching verification queries avoids burning the free-tier 3 RPM limit on
    five sequential single-text calls immediately after ingest embed batches.
    """
    if not texts:
        raise ValueError("embed_queries called with empty texts list")
    if any(not t or not t.strip() for t in texts):
        raise ValueError("embed_queries called with empty query text")

    client: voyageai.Client = _get_client()
    rows: list[list[float]] = _embed_call(
        client,
        texts,
        "query",
        context=f"embed_queries ({len(texts)} queries)",
    )
    matrix: np.ndarray = np.asarray(rows, dtype=np.float32)
    return _normalize(matrix)
