"""
vector_store.py — Qdrant operations with rich region metadata.

This module differs from Day 3/4 vector stores in one critical way:
  it stores STRUCTURED METADATA per chunk that enables region-aware retrieval.

Payload fields stored per point:
  chunk_id:       str    — for deduplication and lookup
  text:           str    — chunk text (for display in results)
  chunk_type:     str    — "prose"|"table"|"figure_meta"|"list"
  region_type:    str    — original Docling label
  page_num:       int    — source page
  heading_path:   list[str]  — ancestor headings (enables section filtering)
  source_pdf:     str    — source filename
  reading_order:  int    — document position (for sorting results)
  table_title:    str | None — only for table chunks
  token_count:    int

Why store heading_path in payload?
  Enables filtered searches: "find tables in the Results section"
  Payload filter: must=[FieldCondition(key="heading_path",
                                       match=MatchAny(any=["Results"]))]
  This is a production-critical feature for long documents where generic
  queries would retrieve chunks from unintended sections.

Why store chunk_type in payload?
  Enables type-specific filtered searches:
  "find all table chunks" = filter on chunk_type="table"
  "find prose about X" = filter on chunk_type="prose"
  Without this metadata, all chunks are indistinguishable by type.

COLLECTION_NAME: use a separate collection from Day 3/4 to avoid conflicts.
  Default: "doc_ingest_ocr"
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)


COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "doc_ingest_ocr")
QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")


def get_client() -> QdrantClient:
    """Connect to Qdrant. Exits the process with a clear message on failure.

    ``timeout=30`` allows for collection creation + initial index build on a
    cold Docker container; the default 5s is sometimes too tight.
    """
    try:
        client = QdrantClient(
            url=QDRANT_URL,    # REST endpoint, set via QDRANT_URL env var
            timeout=30,        # seconds: covers collection create + warm-up
        )
        # Touch the server once so connection errors surface here rather than
        # at the first real call. Listing collections is the cheapest probe.
        client.get_collections()
        return client
    except (ConnectionError, ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        print(
            "ERROR: Could not connect to Qdrant at "
            f"{QDRANT_URL} ({type(exc).__name__}: {exc}).\n"
            "Start Qdrant locally with:\n"
            "  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest",
            file=sys.stderr,
        )
        sys.exit(1)


def create_collection(client: QdrantClient, dim: int) -> None:
    """Create the chunks collection from scratch. Drops any existing version.

    Every Qdrant parameter is documented inline so the index behavior is
    transparent: silently changing an HNSW knob is the most common cause of
    recall/latency regressions in vector pipelines.
    """
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dim,                       # MUST match EMBEDDING_DIM exactly; mismatch = inserts and searches both fail
            distance=Distance.COSINE,       # COSINE = standard semantic similarity metric; Qdrant auto-normalizes on insert
            on_disk=False,                  # keep vectors in RAM: document-scale collections are tiny (a few thousand vectors)
        ),
        hnsw_config=HnswConfigDiff(
            m=16,                           # 16 edges per node: production default for ~1024-dim embeddings; trades RAM for recall
            ef_construct=200,               # build-time candidate queue: balances graph quality vs indexing speed
            full_scan_threshold=1000,       # collections under 1000 vectors use exact (brute-force) search automatically.
                                            # Most single-document ingests stay under this, so search is exact and recall@k = 1.0.
                                            # We still configure HNSW correctly so the same code scales to large corpora unchanged.
        ),
        on_disk_payload=False,              # payloads stay in RAM: small dataset, cheap, faster metadata lookups
    )

    print(f"Collection '{COLLECTION_NAME}' created with {dim}-dim COSINE HNSW index.")


def _uuid_to_point_id(chunk_id: str) -> int:
    """Convert a UUID4 string to a uint64-safe int for Qdrant point IDs.

    Qdrant point IDs must be uint64 or UUID strings. We use the int form so
    payload-indexed lookups, dedup logic, and log lines stay uniformly
    numeric. The mod by 2**63 is a safety margin against signed/unsigned
    handling differences across the HTTP and gRPC client paths — 63 bits is
    still 9.2 quintillion values, far beyond any collision risk.
    """
    return int(uuid.UUID(chunk_id)) % (2**63)


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> None:
    """Upsert ``chunks`` + matching ``embeddings`` rows into the collection.

    Batching trades per-call latency for throughput: 100 points/call turns a
    400-chunk upsert into 4 round-trips instead of 400. Qdrant recommends
    100-500 per batch as the sweet spot.
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
            "must have the same row count"
        )

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=_uuid_to_point_id(chunk["chunk_id"]),  # uint64-safe int derived from chunk UUID4
                vector=vector.tolist(),                   # Qdrant requires list[float], not np.ndarray
                payload={
                    "chunk_id":      chunk["chunk_id"],       # original UUID, kept for cross-system traceability
                    "text":          chunk["text"],           # stored verbatim so results show the exact source text
                    "chunk_type":    chunk["chunk_type"],     # enables type-filtered search (prose/table/figure_meta/list)
                    "region_type":   chunk["region_type"],    # original Docling label, retained for provenance/debugging
                    "page_num":      chunk["page_num"],       # MANDATORY for citations: humans verify answers against this page
                    "heading_path":  chunk["heading_path"],   # ancestor headings: enables section-scoped filtered retrieval
                    "source_pdf":    chunk["source_pdf"],     # basename of the source PDF (citation display)
                    "reading_order": chunk["reading_order"],  # document position: lets callers re-sort results into doc order
                    "table_title":   chunk.get("table_title"),  # only set for table chunks; None otherwise
                    "token_count":   chunk["token_count"],    # advisory: spot oddly-sized chunks in the payload
                    "bbox":          chunk.get("bbox", [0.0, 0.0, 1.0, 1.0]),  # normalized region box (future Vision LLM crop)
                },
            )
        )

    total: int = len(points)
    for start in range(0, total, batch_size):
        batch: list[PointStruct] = points[start : start + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,    # list of PointStruct for this batch; one round-trip per batch
            wait=True,       # block until WAL ack so subsequent search() calls see these points
        )
        done: int = min(start + batch_size, total)
        print(f"Upserted {done}/{total} chunks...")

    count_result = client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"Collection now contains {int(count_result.count)} vectors.")


def _point_to_result(point: Any, rank: int) -> dict[str, Any]:
    """Flatten a Qdrant scored point into a display-ready result dict."""
    payload: dict[str, Any] = point.payload or {}
    return {
        "text":          payload.get("text", ""),
        "chunk_type":    payload.get("chunk_type", ""),
        "region_type":   payload.get("region_type", ""),
        "page_num":      int(payload.get("page_num", 0)),
        "heading_path":  list(payload.get("heading_path", []) or []),
        "table_title":   payload.get("table_title"),
        "source_pdf":    payload.get("source_pdf", ""),
        "chunk_id":      payload.get("chunk_id", ""),
        "reading_order": int(payload.get("reading_order", 0)),
        "score":         float(point.score) if point.score is not None else 0.0,
        "rank":          rank,  # 1-indexed for human-readable display
    }


def search(
    client: QdrantClient,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run an unfiltered vector search. Returns ranked result dicts."""
    qvec: list[float] = query_vector.astype(np.float32).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,                          # query embedding as list[float] of length EMBEDDING_DIM
        limit=top_k,                         # top-K cutoff
        search_params=SearchParams(
            hnsw_ef=128,                     # query-time candidate queue; higher = better recall, slower
            exact=False,                     # use HNSW when applicable; Qdrant auto-uses exact for small collections
        ),
    )

    return [_point_to_result(point, rank) for rank, point in enumerate(response.points, start=1)]


def search_by_type(
    client: QdrantClient,
    query_vector: np.ndarray,
    chunk_type: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Run a vector search restricted to chunks of a specific ``chunk_type``.

    The filter is applied SERVER-SIDE via Qdrant's ``Filter`` + ``FieldCondition``
    so the HNSW traversal itself only considers matching points — this is both
    correct (no missed candidates) and far cheaper than fetching everything and
    post-filtering in Python.
    """
    qvec: list[float] = query_vector.astype(np.float32).tolist()

    type_filter: Filter = Filter(
        must=[
            FieldCondition(
                key="chunk_type",                  # payload field set in upsert_chunks
                match=MatchValue(value=chunk_type),  # exact-match on the requested type
            )
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        query_filter=type_filter,            # server-side restriction to chunk_type
        limit=top_k,
        search_params=SearchParams(
            hnsw_ef=128,
            exact=False,
        ),
    )

    return [_point_to_result(point, rank) for rank, point in enumerate(response.points, start=1)]


def _payload_to_chunk(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a Qdrant payload dict into the verifier's chunk shape."""
    payload = payload or {}
    return {
        "text": payload.get("text", ""),
        "chunk_type": payload.get("chunk_type", ""),
        "page_num": int(payload.get("page_num", 0)),
        "heading_path": list(payload.get("heading_path", []) or []),
        "table_title": payload.get("table_title"),
        "source_pdf": payload.get("source_pdf", ""),
        "reading_order": int(payload.get("reading_order", 0)),
    }


def sample_payloads_by_type(
    client: QdrantClient,
    chunk_type: str,
    limit: int = 1,
    *,
    fetch_limit: int = 32,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` chunk payloads of ``chunk_type``, earliest in doc order.

    Used by the verifier to build corpus-specific smoke-test queries instead
    of hardcoded phrases from the synthetic sample PDFs.
    """
    type_filter: Filter = Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))]
    )
    scroll_result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=type_filter,
        limit=max(limit, fetch_limit),
        with_payload=True,
        with_vectors=False,
    )
    if isinstance(scroll_result, tuple):
        records, _ = scroll_result
    else:
        records = scroll_result

    chunks: list[dict[str, Any]] = [_payload_to_chunk(point.payload) for point in records]
    chunks.sort(key=lambda c: (c["page_num"], c["reading_order"]))
    return chunks[:limit]


def count_by_type(client: QdrantClient, chunk_type: str) -> int:
    """Return the number of stored points whose ``chunk_type`` matches.

    Used by the CLI deliverable checklist (e.g. "at least N table chunks in
    collection") so the count is read back from Qdrant rather than trusting
    in-memory ingest tallies — this also works on the ``--verify-only`` path.
    """
    type_filter: Filter = Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))]
    )
    result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=type_filter,
        exact=True,   # exact count is microsecond-cheap at document scale
    )
    return int(result.count)


def collection_exists(client: QdrantClient) -> bool:
    """True iff ``COLLECTION_NAME`` exists AND already has at least one vector.

    The "has vectors" check matters because a collection can exist in an empty
    state if a previous run created it but crashed before upsert.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return False
    info = client.get_collection(collection_name=COLLECTION_NAME)
    points_count: int = int(getattr(info, "points_count", 0) or 0)
    vectors_count: int = int(getattr(info, "vectors_count", 0) or 0)
    return max(points_count, vectors_count) > 0
