"""Embedding module. Supports OpenRouter embeddings and a local hash fallback.

Architecture decision: isolated here so ``qdrant_ops.py`` and
``benchmark.py`` never import provider SDKs directly. If
the embedding provider changes, only this file changes.

Fallback semantics: when ``OPENROUTER_API_KEY`` is missing the script still
runs end-to-end -- ``_hash_embed`` produces deterministic unit vectors so
the benchmark plumbing exercises the same code paths. Those vectors are
NOT semantic, so the recall@5 numbers from a hash-only run only validate
pipeline wiring (unfiltered ANN vs exact recall is often 1.0 at N=500
because hash vectors are well-separated; filtered-topic recall stays low).
Only the OpenRouter path produces semantically meaningful recall for tuning
ef_search and production readiness.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# Dimension and model name kept as module-level constants so callers
# (qdrant_ops.create_hnsw_collection) can read them without instantiating
# a client. 1536 matches openai/text-embedding-3-small default output dim;
# the hash fallback
# pads to the same dimension so collection configs do not need to change.
EMBEDDING_DIM: int = int(os.getenv("OPENROUTER_EMBEDDING_DIM", "1536"))
EMBEDDING_MODEL: str = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Embedding APIs support batching. 50 is conservative: longer
# chunks consume more tokens per request, and 50 keeps us well under
# any per-request token cap while still amortising HTTPS overhead.
_DEFAULT_BATCH_SIZE: int = 50

# Cache the loaded client and the "we warned about fallback" state so a
# single CLI invocation prints the warning exactly once.
_openrouter_client: Optional[OpenAI] = None
_openrouter_loaded: bool = False
_fallback_warned: bool = False


def _load_client() -> Optional[OpenAI]:
    """Lazily build and cache an OpenRouter client.

    Reads ``.env`` (via ``python-dotenv``) so ``OPENROUTER_API_KEY`` can live
    in a project-local ``.env`` file. Returns ``None`` if no API key
    is available, in which case callers fall back to ``_hash_embed``.
    """
    global _openrouter_client, _openrouter_loaded
    if _openrouter_loaded:
        return _openrouter_client

    load_dotenv()
    api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        _openrouter_client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        _openrouter_client = None

    _openrouter_loaded = True
    return _openrouter_client


def _warn_fallback_once() -> None:
    """Print a one-time warning when the hash fallback is engaged."""
    global _fallback_warned
    if _fallback_warned:
        return
    print(
        "  WARN: OPENROUTER_API_KEY not found -- using local hash embeddings. "
        "recall@5 numbers will not be semantically meaningful.",
        file=sys.stderr,
    )
    _fallback_warned = True


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row in-place-safe fashion.

    Why normalize here even though Qdrant auto-normalizes when COSINE is
    selected? Two reasons:
      1. The hash fallback would otherwise return raw normal vectors with
         drifting magnitudes, which Qdrant would still normalize but our
         downstream code would not match cosine == dot.
      2. Pre-normalized vectors let benchmark callers use np.dot directly
         for any sanity checks without re-normalizing.
    Zero-norm rows are left at zero (a degenerate case for the hash path
    only).
    """
    norms: np.ndarray = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms: np.ndarray = np.where(norms == 0.0, 1.0, norms)
    return (matrix / safe_norms).astype(np.float32, copy=False)


def _hash_embed(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic local embedding for offline/testing use.

    NOT semantic. Uses SHA-256 of the text as a seed for NumPy's RNG and
    samples a standard-normal vector, then L2-normalizes. Same input ->
    same vector, which is all we need for the benchmark plumbing.
    """
    digest: bytes = hashlib.sha256(text.encode("utf-8")).digest()
    seed: int = int.from_bytes(digest[:8], byteorder="big") % (2**32)
    rng: np.random.Generator = np.random.default_rng(seed)
    vec: np.ndarray = rng.standard_normal(dim).astype(np.float32)
    norm: float = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def _embed_with_openrouter(
    client: OpenAI,
    texts: list[str],
    input_type: str,
    batch_size: int,
) -> np.ndarray:
    """Call OpenRouter's OpenAI-compatible embeddings API in batches."""
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch: list[str] = texts[start : start + batch_size]
        result = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
            extra_body={"input_type": input_type},
        )
        rows.extend(item.embedding for item in result.data)
    return np.asarray(rows, dtype=np.float32)


def embed_texts(
    texts: list[str],
    input_type: str = "document",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Embed a list of texts and return a ``(N, EMBEDDING_DIM)`` float32 array.

    ``input_type``:
      * ``"document"`` -- corpus chunks being indexed (retrievable docs).
      * ``"query"``    -- search queries (lookup intents).
      Using the wrong ``input_type`` can degrade retrieval quality because
      embedding providers map documents and queries to different regions of
      the output space on purpose.

    ``batch_size``: bound on texts per API call. Defaults to 50; bump up
    to 128 for short inputs to reduce round-trips.

    Fallback: if no OpenRouter client is available, every text is embedded
    with ``_hash_embed`` and a one-time warning is printed.

    The returned matrix is L2-normalized so cosine similarity equals the
    dot product, which lets Qdrant's COSINE distance match exactly with
    any local sanity checks we run on the same vectors.
    """
    if not texts:
        raise ValueError("texts must not be empty")

    if input_type not in {"document", "query"}:
        raise ValueError(
            f"input_type must be 'document' or 'query' (got {input_type!r})"
        )

    client: Optional[OpenAI] = _load_client()
    if client is None:
        _warn_fallback_once()
        matrix: np.ndarray = np.stack(
            [_hash_embed(text, EMBEDDING_DIM) for text in texts], axis=0
        )
    else:
        matrix = _embed_with_openrouter(client, texts, input_type, batch_size)

    return _l2_normalize(matrix)


def embed_single(text: str, input_type: str = "query") -> np.ndarray:
    """Convenience wrapper: embed one text, return ``(EMBEDDING_DIM,)`` float32.

    Used for query embedding at search time so callers do not have to
    wrap a single string in a list and unwrap a (1, D) matrix.
    """
    matrix: np.ndarray = embed_texts([text], input_type=input_type, batch_size=1)
    return matrix[0]
