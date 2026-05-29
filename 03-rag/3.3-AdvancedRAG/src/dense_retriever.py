"""dense_retriever.py - Qdrant vector search for the advanced RAG pipeline.

Design decision: reuses the Day 3 Qdrant collection (COLLECTION_NAME=chunks_hnsw).
Does NOT re-ingest under normal conditions. Assumes Day 3 was run and the
collection is populated.

If the collection is empty or missing: raises RuntimeError with the exact
command to re-run Day 3.

If the collection exists but its payload lacks the ``chunk_id`` field that
this module relies on (Day 3 did not store ``chunk_id`` in the payload),
a one-time automatic re-ingest is triggered to add the field. The vectors
are recomputed via the same OpenRouter embedder used in Day 3 / Day 4 so
the geometry of the index is unchanged.

Why keep dense retrieval in a separate module from BM25?
    Dense and sparse retrieval are independent systems with different
    failure modes. Isolating them allows independent benchmarking:
    - If BM25 returns empty results (OOV query), dense retrieval still
      runs normally.
    - If Qdrant is down, BM25 can still run independently.
    RRF fusion in rrf_fusion.py handles the combination.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import PointStruct, SearchParams


# ---------------------------------------------------------------------------
# Cross-day module bridges - WHY we don't use sys.path + ``from src.X``
#
# Day 3, Day 4 and Day 5 each ship their own ``src/`` package directory.
# A plain ``sys.path.insert + from src.embedder import ...`` would resolve
# ``src`` to whichever ``src`` Python finds first in sys.path, which in
# practice is the local Day 5 package because we already imported other
# modules from it. The collision is silent in dev (Day 5's ``src`` always
# wins) and explodes at runtime with ``ModuleNotFoundError: No module
# named 'src.embedder'``.
#
# Instead we load the target files DIRECTLY via importlib.util under
# unique synthetic module names (``_day4_embedder`` / ``_day3_qdrant_ops``)
# so the global ``src`` namespace is never touched. The unique names also
# let the modules be cached in ``sys.modules`` for the lifetime of the
# process so subsequent imports are free.
# ---------------------------------------------------------------------------
_THIS_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT_3RAG: Path = _THIS_DIR.parent.parent
_DAY3_DIR: Path = _REPO_ROOT_3RAG / "3.1-VectorDBs"
_DAY4_DIR: Path = _REPO_ROOT_3RAG / "3.2-NaiveRAG"


def _load_module_from_path(unique_name: str, file_path: Path) -> ModuleType:
    """Load a Python file as a standalone module under ``unique_name``.

    ``unique_name`` must NOT collide with any package on sys.path - that
    is the whole point of this helper. We deliberately pick names with a
    leading underscore (``_day3_qdrant_ops``, ``_day4_embedder``) so a
    future contributor cannot accidentally publish a package with the
    same name.
    """
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    if not file_path.is_file():
        raise RuntimeError(
            f"Required file not found: {file_path}. "
            f"Day 5 depends on Day 3 / Day 4 being present at sibling paths."
        )
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build import spec for {file_path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


# Day 4's embedder is the production OpenRouter path; we reuse it rather
# than re-implementing because (1) it L2-normalises to match the COSINE
# distance used by chunks_hnsw, and (2) keeping the embedder in one place
# avoids the failure mode where the corpus is embedded with model A and
# the query with model B.
try:
    _day4_embedder: ModuleType = _load_module_from_path(
        "_day4_embedder", _DAY4_DIR / "src" / "embedder.py"
    )
except RuntimeError as exc:  # pragma: no cover - explicit user-facing error
    raise RuntimeError(
        "Could not load Day 4 embedder from "
        "03-rag/3.2-NaiveRAG/src/embedder.py. This module depends on it "
        f"for OpenRouter dense embeddings. Underlying error: {exc}"
    ) from exc

EMBEDDING_DIM: int = int(_day4_embedder.EMBEDDING_DIM)
embed_documents: Callable[[list[str]], np.ndarray] = _day4_embedder.embed_documents
embed_query: Callable[[str], np.ndarray] = _day4_embedder.embed_query


# ---------------------------------------------------------------------------
# Connection + collection constants
# ---------------------------------------------------------------------------
QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "chunks_hnsw")
_EXPECTED_VECTOR_COUNT: int = 500
_QDRANT_TIMEOUT_SEC: int = 30
_REINGEST_BATCH_SIZE: int = 100


def get_qdrant_client() -> QdrantClient:
    """Connect to Qdrant and verify the server is reachable.

    Does NOT require ``chunks_hnsw`` to exist yet. A missing or empty
    collection is repaired by :func:`ensure_chunk_id_payload`, which
    creates the collection and upserts all 500 chunks (same HNSW config
    as Day 3) on first run. That lets Day 5 work on a fresh Docker
    volume without running Day 3 separately.

    Raises RuntimeError only when Qdrant itself is down (connection
    refused, timeout, etc.).
    """
    try:
        client: QdrantClient = QdrantClient(url=QDRANT_URL, timeout=_QDRANT_TIMEOUT_SEC)
        # Lightweight liveness probe: list collections. Avoids calling
        # get_collection('chunks_hnsw') which 404s loudly on a fresh DB.
        client.get_collections()
    except (ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        raise RuntimeError(
            f"Qdrant unreachable at {QDRANT_URL}. "
            f"Start it with: docker run -d -p 6333:6333 qdrant/qdrant\n"
            f"Underlying error: {exc}"
        ) from exc
    return client


def _payload_has_chunk_id(client: QdrantClient) -> bool:
    """Inspect a single point to see whether the payload exposes ``chunk_id``.

    Day 3's qdrant_ops.upsert_chunks stored {topic, source, page, chunk_index,
    text} in the payload. The advanced RAG modules (RRF, reranker) need a
    string ``chunk_id`` for cross-system joins, so we sniff one point here
    and, on a mismatch, trigger the one-time re-ingest below.

    Uses ``scroll`` with limit=1 (cheapest possible read) and returns False
    on any access error so the re-ingest path is the safe default.
    """
    try:
        scroll_result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,                          # only need one sample point
            with_payload=True,                # payload is the whole reason for the call
            with_vectors=False,               # vectors are huge; never needed here
        )
    except (ResponseHandlingException, UnexpectedResponse):
        return False
    if not scroll_result:
        return False
    payload: dict[str, Any] | None = scroll_result[0].payload
    return bool(payload) and "chunk_id" in payload


def _build_points_with_chunk_id(
    chunks: list[dict],
    embeddings: np.ndarray,
) -> list[PointStruct]:
    """Mirror of Day 3's _build_points BUT with ``chunk_id`` in the payload.

    We do not import Day 3's _build_points because adding a payload key
    upstream would require modifying Day 3's source - violating the rule
    that this module never edits other days. The bodies are kept aligned
    so reading both side-by-side reveals exactly one difference: the
    extra ``chunk_id`` field.
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
            f"row counts must match"
        )
    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=int(chunk["id"]),       # Qdrant point ID; reuse Day 3 int IDs for stable cross-day identity
                vector=vector.tolist(),    # Qdrant expects list[float]; np.ndarray is not directly serializable
                payload={
                    "chunk_id": str(chunk["id"]),  # NEW vs Day 3: canonical cross-system string key
                    "topic": chunk["topic"],
                    "source": chunk["source"],
                    "page": int(chunk["page"]),
                    "chunk_index": int(chunk["chunk_index"]),
                    "text": chunk["text"],
                },
            )
        )
    return points


def _reingest_with_chunk_id(client: QdrantClient, corpus: list[dict]) -> None:
    """Drop and rebuild ``chunks_hnsw`` with ``chunk_id`` in every payload.

    Triggered only once per environment - subsequent runs see ``chunk_id``
    on the sniffed point and skip this branch. We reuse Day 3's
    create_hnsw_collection so the HNSW parameters (m=16, ef_construct=200)
    stay identical to the benchmark's published numbers.
    """
    try:
        _day3_qdrant_ops: ModuleType = _load_module_from_path(
            "_day3_qdrant_ops", _DAY3_DIR / "src" / "qdrant_ops.py"
        )
    except RuntimeError as exc:  # pragma: no cover - explicit user-facing error
        raise RuntimeError(
            "Could not load Day 3 qdrant_ops from "
            "03-rag/3.1-VectorDBs/src/qdrant_ops.py. Required for the "
            f"one-time chunk_id payload migration. Underlying error: {exc}"
        ) from exc
    create_hnsw_collection: Callable[..., None] = _day3_qdrant_ops.create_hnsw_collection

    print(
        "  Bootstrap: creating/rebuilding collection with chunk_id payloads "
        "(one-time; embeds 500 chunks via OpenRouter)..."
    )

    texts: list[str] = [chunk["text"] for chunk in corpus]
    start: float = time.perf_counter()
    embeddings: np.ndarray = embed_documents(texts)
    elapsed: float = time.perf_counter() - start
    print(
        f"  Re-embedded {len(texts)} chunks in {elapsed:.1f}s "
        f"(dim={embeddings.shape[1]})"
    )

    create_hnsw_collection(client, dim=EMBEDDING_DIM, collection_name=COLLECTION_NAME)
    points: list[PointStruct] = _build_points_with_chunk_id(corpus, embeddings)

    total: int = len(points)
    for start_idx in range(0, total, _REINGEST_BATCH_SIZE):
        batch: list[PointStruct] = points[start_idx : start_idx + _REINGEST_BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,                     # block until WAL ack so subsequent searches see these points
        )
        done: int = min(start_idx + _REINGEST_BATCH_SIZE, total)
        print(f"  Re-ingest: upserted {done}/{total} points...")

    print("  Re-ingest complete. chunk_id is now present on every payload.")


def _collection_is_ready(client: QdrantClient) -> bool:
    """True when the collection exists, has vectors, and payloads carry chunk_id."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return False
    try:
        count: int = int(client.count(collection_name=COLLECTION_NAME, exact=True).count)
    except (ResponseHandlingException, UnexpectedResponse):
        return False
    if count < _EXPECTED_VECTOR_COUNT:
        return False
    return _payload_has_chunk_id(client)


def ensure_chunk_id_payload(client: QdrantClient, corpus: list[dict]) -> None:
    """Public bootstrap: create or repair ``chunks_hnsw`` if needed.

    Triggers a full embed + upsert when any of these hold:
      - collection does not exist (fresh Qdrant / new Docker volume)
      - fewer than 500 vectors (partial or stale ingest)
      - payloads lack ``chunk_id`` (Day 3 ingest without migration)

    Idempotent on a healthy collection: one scroll call, then return.
    """
    if _collection_is_ready(client):
        return
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(
            f"  Collection '{COLLECTION_NAME}' not found - will create and "
            f"populate {len(corpus)} chunks (Day 3 optional)."
        )
    _reingest_with_chunk_id(client, corpus)

    count: int = int(client.count(collection_name=COLLECTION_NAME, exact=True).count)
    if count < _EXPECTED_VECTOR_COUNT:
        raise RuntimeError(
            f"Bootstrap finished but '{COLLECTION_NAME}' has only {count} "
            f"vectors (expected {_EXPECTED_VECTOR_COUNT}). Check OpenRouter "
            f"embeddings and Qdrant logs."
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
# Two ef values: baseline simulates a fast production ANN (lower recall vs
# exact GT); hybrid/rerank pools use higher ef so ground-truth neighbours
# land in the candidate set for RRF and the cross-encoder.
BASELINE_HNSW_EF: int = 16
CANDIDATE_HNSW_EF: int = 128


def dense_search(
    client: QdrantClient,
    query: str,
    top_n: int = 50,
    hnsw_ef: int = CANDIDATE_HNSW_EF,
) -> list[dict]:
    """Embed the query and run Qdrant HNSW ANN search for ``top_n`` candidates.

    Returns a list of dicts:
        {"chunk_id": str, "dense_score": float, "dense_rank": int}
    with 1-indexed ranks. On a Qdrant error the underlying exception is
    re-raised with context; on a genuinely empty result we return ``[]``.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive (got {top_n})")
    if not query or not query.strip():
        raise ValueError("dense_search called with empty query")

    qvec: np.ndarray = embed_query(query)
    qvec_list: list[float] = qvec.astype(np.float32).tolist()

    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=qvec_list,                # nearest-neighbour query vector; length must equal EMBEDDING_DIM
            limit=top_n,                    # candidate pool size; matches BM25's top_n by design
            search_params=SearchParams(
                hnsw_ef=hnsw_ef,
                exact=False,                # use the HNSW graph (approximate); True is reserved for ground_truth.py
            ),
            with_payload=True,              # need payload['chunk_id'] for the cross-system join
            with_vectors=False,             # vectors are 1536-dim each; do not pay to ship them back
        )
    except (ResponseHandlingException, UnexpectedResponse) as exc:
        raise RuntimeError(
            f"Qdrant dense_search failed for query={query!r}: {exc}"
        ) from exc

    points: list[Any] = list(response.points)
    if not points:
        return []

    results: list[dict] = []
    for rank_pos, point in enumerate(points, start=1):  # 1-indexed rank
        payload: dict[str, Any] = point.payload or {}
        chunk_id_raw: Any = payload.get("chunk_id")
        if chunk_id_raw is None:
            # Defence in depth - ensure_chunk_id_payload() should have
            # repaired this already, but if a stale point sneaks through
            # we fall back to the integer point ID stringified, which
            # matches the corpus_bridge contract.
            chunk_id_raw = str(point.id)
        results.append(
            {
                "chunk_id": str(chunk_id_raw),
                "dense_score": float(point.score),
                "dense_rank": rank_pos,
            }
        )
    return results
