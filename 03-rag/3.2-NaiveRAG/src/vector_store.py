"""
vector_store.py - Qdrant operations for naive RAG.

Design decision: collection name and URL come from environment variables,
not hardcoded. This allows the same code to point at local Docker for
development and Qdrant Cloud for production without source changes.

Design decision: collection is deleted and recreated on every ingest.
Reason: this is a learning artifact - idempotent ingestion matters more
than incremental updates at this stage. In production, use upsert with
content-hash deduplication.
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
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)


COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "naive_rag_chunks")
QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")


def get_client() -> QdrantClient:
    """Connect to Qdrant. Exits the process with a clear message on failure.

    ``timeout=30`` allows for collection creation + initial index build
    on a cold Docker container; the default 5s is sometimes too tight.
    """
    try:
        client = QdrantClient(
            url=QDRANT_URL,    # REST endpoint, set via QDRANT_URL env var
            timeout=30,        # seconds: covers collection create + warm-up
        )
        # Touch the server once so we surface connection errors here, at
        # client construction, rather than at the first real call. Listing
        # collections is the cheapest health-probe call Qdrant exposes.
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

    Every Qdrant parameter is documented inline. This is intentional -
    naive RAG fails most often because someone changed an HNSW knob
    without understanding the downstream effect on recall and latency.
    """
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dim,                       # MUST match EMBEDDING_DIM exactly; mismatch = inserts and searches both fail
            distance=Distance.COSINE,       # COSINE = standard semantic similarity metric; Qdrant auto-normalizes on insert
            on_disk=False,                  # keep vectors in RAM: PDF-scale collections are tiny (under a few thousand vectors)
        ),
        hnsw_config=HnswConfigDiff(
            m=16,                           # 16 edges per node: production default for 1024-dim embeddings; trades RAM for recall
            ef_construct=200,               # build-time candidate queue: balances graph quality vs indexing speed
            full_scan_threshold=1000,       # CRITICAL FOR THIS DEMO: collections under 1000 vectors use exact (brute-force) search automatically.
                                            # Our PDF will have ~100-400 chunks, so Qdrant ALWAYS falls back to exact search here.
                                            # That means recall@k = 1.0 by construction for this naive RAG.
                                            # We still configure HNSW correctly so the same code scales to large corpora unchanged.
        ),
        on_disk_payload=False,              # payloads stay in RAM: small dataset, cheap, faster citation lookups
    )

    print(
        f"Collection '{COLLECTION_NAME}' created with {dim}-dim COSINE HNSW index."
    )


def _uuid_to_point_id(chunk_id: str) -> int:
    """Convert a UUID4 string to a uint64-safe int for Qdrant point IDs.

    Qdrant point IDs must be either uint64 or UUID strings. We use the
    int form so downstream payload-indexed lookups, dedup logic, and
    log lines stay uniformly numeric. The mod by 2**63 is a safety
    margin against signed/unsigned int handling differences across the
    HTTP and gRPC client paths - 63 bits is still 9.2 quintillion values,
    far beyond any collision risk for a PDF-scale corpus.
    """
    return int(uuid.UUID(chunk_id)) % (2**63)


def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> None:
    """Upsert ``chunks`` + matching ``embeddings`` rows into the collection.

    Batching trades per-call latency for throughput: 100 points/call
    turns a 400-chunk upsert into 4 round-trips instead of 400. Qdrant's
    docs recommend 100-500 per batch as the sweet spot.
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
                id=_uuid_to_point_id(chunk["chunk_id"]),  # uint64-safe int derived from chunk's UUID4
                vector=vector.tolist(),                   # Qdrant requires list[float], not np.ndarray
                payload={
                    "chunk_id":    chunk["chunk_id"],     # original UUID, kept for cross-system traceability
                    "text":        chunk["text"],         # stored verbatim so citations show the exact source text
                    "page_num":    chunk["page_num"],     # MANDATORY for citations: humans verify answers against this page
                    "chunk_index": chunk["chunk_index"],  # 0-indexed position within the page (useful for debugging)
                    "source_pdf":  chunk["source_pdf"],   # basename of the source PDF (citation display)
                    "token_count": chunk["token_count"],  # advisory: lets us spot oddly-sized chunks in the payload
                },
            )
        )

    total: int = len(points)
    for start in range(0, total, batch_size):
        batch: list[PointStruct] = points[start : start + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,    # list of PointStruct for this batch; one network round-trip per batch
            wait=True,       # block until WAL ack so subsequent search() calls see these points
        )
        done: int = min(start + batch_size, total)
        print(f"Upserted {done}/{total} chunks...")

    # Confirm by re-counting from the server. ``exact=True`` is fine here
    # because PDF-scale collections finish counting in microseconds.
    count_result = client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"Collection now contains {int(count_result.count)} vectors.")


def search(
    client: QdrantClient,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """Run a vector search. Returns ranked result dicts (highest score first).

    RAG best practice: top_k between 3 and 5. We hard-cap at 5 here as
    defense-in-depth (the CLI argparse already restricts to {3,4,5}).
    Going above 5 introduces the "topK pathology": each additional chunk
    adds prompt tokens, raises the chance of a low-quality distractor
    chunk, and dilutes the LLM's attention away from the best evidence.
    """
    if top_k > 5:
        print(
            f"  WARN: top_k={top_k} exceeds RAG best-practice cap; "
            "clamping to 5 to avoid topK pathology.",
            file=sys.stderr,
        )
        top_k = 5

    qvec: list[float] = query_vector.astype(np.float32).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,                          # query embedding as list[float] of length EMBEDDING_DIM
        limit=top_k,                         # top-K cutoff; matches downstream RAG context-window budget
        search_params=SearchParams(
            hnsw_ef=128,                     # query-time candidate queue; higher = better recall, slower. 128 is conservative for small corpora.
                                             # NOTE: with full_scan_threshold=1000, our PDF collection uses exact search and this knob is moot here.
            exact=False,                     # use the HNSW index when applicable; Qdrant overrides to exact for small collections automatically
        ),
    )

    results: list[dict] = []
    for rank, point in enumerate(response.points, start=1):
        payload: dict[str, Any] = point.payload or {}
        results.append(
            {
                "text":        payload.get("text", ""),
                "page_num":    int(payload.get("page_num", 0)),
                "source_pdf":  payload.get("source_pdf", ""),
                "chunk_id":    payload.get("chunk_id", ""),
                "score":       float(point.score) if point.score is not None else 0.0,
                "rank":        rank,  # 1-indexed for human-readable display
            }
        )
    return results


def collection_exists(client: QdrantClient) -> bool:
    """True iff ``COLLECTION_NAME`` exists AND already has at least one vector.

    Used by the CLI to decide whether ingestion can be skipped. The
    "has vectors" check matters because a collection can exist in an
    empty state if a previous run created it but crashed before upsert.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return False
    info = client.get_collection(collection_name=COLLECTION_NAME)
    # qdrant-client exposes ``points_count`` (canonical) and sometimes
    # ``vectors_count`` depending on version. Read whichever is non-None.
    points_count: int = int(getattr(info, "points_count", 0) or 0)
    vectors_count: int = int(getattr(info, "vectors_count", 0) or 0)
    return max(points_count, vectors_count) > 0
