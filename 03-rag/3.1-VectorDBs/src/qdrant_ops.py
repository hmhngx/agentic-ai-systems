"""All Qdrant client operations live here.

Everything that touches the Qdrant API funnels through this module so the
benchmark code stays index-agnostic. Every Qdrant parameter on every call
carries an inline comment explaining what it does and why it is set to
that value -- this is a learning artifact, so the comments are part of
the deliverable.
"""

from __future__ import annotations

import os
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    QueryResponse,
    SearchParams,
    VectorParams,
)


# Collection and connection constants. Defined here once so every other
# module imports them from a single source of truth.
COLLECTION_NAME: str = "chunks_hnsw"
COLLECTION_NAME_FLAT: str = "chunks_flat"
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")


def get_client() -> QdrantClient:
    """Create a Qdrant client connected to the local Docker instance.

    ``timeout=30`` allows up to 30 seconds for slow operations like
    collection creation with index builds; the default 5s is sometimes
    too tight when Docker is cold-starting.
    """
    return QdrantClient(
        url=QDRANT_URL,    # REST endpoint exposed by scripts/start_qdrant.sh
        timeout=30,        # seconds: covers cold collection creation + first index build
    )


def _drop_if_exists(client: QdrantClient, collection_name: str) -> None:
    """Delete a collection only when it already exists.

    Using ``collection_exists`` keeps ``delete_collection`` from raising
    on a fresh database, which matters for the first ``--skip-upsert``-
    less run.
    """
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)


def create_hnsw_collection(
    client: QdrantClient,
    dim: int,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Create the HNSW collection. Drops any existing collection first.

    The HNSW config below is the production-default knob set for
    1024-dim semantic embeddings. Each parameter is annotated; tune
    these per the rules in the docstring of the parameter.
    """
    _drop_if_exists(client, collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,                      # must match embedding dimension (embedder.EMBEDDING_DIM); mismatch breaks insert/search
            distance=Distance.COSINE,      # COSINE chosen because semantic embeddings are L2-normalized; cosine ≈ dot product for unit vectors
            on_disk=False,                 # keep vectors in RAM for low-latency lookups (500 pts is tiny)
        ),
        hnsw_config=HnswConfigDiff(
            m=16,                          # edges per node in HNSW graph; typical range 8–64; 16 is Qdrant default for 1024-dim production workloads
            ef_construct=200,              # build-time candidate queue; higher = better graph quality + slower indexing (quality vs speed trade-off)
            full_scan_threshold=10,      # when segment has fewer points than this, Qdrant uses FLAT not HNSW; 10 (min allowed) keeps HNSW active at 500 pts
        ),
        on_disk_payload=False,             # payload in RAM; flip to True if payload size exceeds available RAM in prod
    )


def create_flat_collection(
    client: QdrantClient,
    dim: int,
    collection_name: str = COLLECTION_NAME_FLAT,
) -> None:
    """Create the FLAT (exact search) collection for illustration.

    Identical to the HNSW collection except ``m=0`` disables the HNSW
    graph entirely, so Qdrant falls back to brute-force exact search on
    every query. The benchmark itself uses ``exact=True`` on the HNSW
    collection for ground truth -- this separate collection exists to
    make the FLAT-vs-HNSW choice explicit in the data model.
    """
    _drop_if_exists(client, collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,                      # must match embedder.EMBEDDING_DIM
            distance=Distance.COSINE,      # same metric so scores compare across collections
            on_disk=False,                 # in-RAM for fair latency comparison
        ),
        hnsw_config=HnswConfigDiff(
            m=0,                           # m=0 disables HNSW graph construction -- forces brute-force exact scan
            ef_construct=200,              # ignored when m=0 but kept for clarity / future re-enable
            full_scan_threshold=10000,     # same threshold so flat behaviour is uniform across collections
        ),
        on_disk_payload=False,             # payload in RAM, matching the HNSW collection
    )


def _build_points(
    chunks: list[dict],
    embeddings: np.ndarray,
) -> list[PointStruct]:
    """Pair each chunk dict with its embedding row into a PointStruct list."""
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) "
            f"row counts must match"
        )
    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=int(chunk["id"]),       # Qdrant point IDs must be int or UUID; we reuse the chunk index
                vector=vector.tolist(),    # Qdrant expects list[float]; np.ndarray is not directly serializable
                payload={                  # arbitrary JSON dict; "text" stays here so we can show retrieved chunks
                    "topic": chunk["topic"],
                    "source": chunk["source"],
                    "page": int(chunk["page"]),
                    "chunk_index": int(chunk["chunk_index"]),
                    "text": chunk["text"],
                },
            )
        )
    return points


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    batch_size: int = 100,                 # points per upsert call; 100–500 recommended to balance throughput vs memory per request
) -> None:
    """Upsert chunks + embeddings into ``collection_name`` in batches.

    Batching trades per-call latency for throughput: 100 points/call
    turns the 500-chunk upsert into 5 round-trips instead of 500.
    Qdrant's docs recommend 100-500 per batch as the sweet spot.
    """
    points: list[PointStruct] = _build_points(chunks, embeddings)
    total: int = len(points)
    for start in range(0, total, batch_size):
        batch: list[PointStruct] = points[start : start + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,                  # batched list of PointStruct: one network round-trip per batch
            wait=True,                     # block until the WAL ack so subsequent searches see these points immediately
        )
        print(f"  Upserted {min(start + batch_size, total)}/{total} points...")


def _topic_filter(topic: str | None) -> Filter | None:
    """Build a Qdrant ``Filter`` restricting search to a single topic.

    ``Filter(must=[...])`` is an AND of conditions; ``FieldCondition`` +
    ``MatchValue`` matches an exact payload value. Returns ``None`` for
    a missing topic so the caller can pass it straight through.
    """
    if topic is None:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="topic",                  # payload field to match; restricts search space (e.g. 500 → ~100 vectors per topic)
                match=MatchValue(value=topic),  # exact-equality on topic; only points with this payload value are candidates
            )
        ]
    )


def _point_ids_from_response(response: QueryResponse) -> list[int]:
    """Extract ranked point IDs from a ``query_points`` QueryResponse."""
    return [int(p.id) for p in response.points]


def search_hnsw(
    client: QdrantClient,
    query_vector: np.ndarray,
    top_k: int = 5,
    ef_search: int = 64,
    topic_filter: str | None = None,
    collection_name: str = COLLECTION_NAME,
) -> tuple[list[int], float]:
    """Run ANN search using the HNSW index. Returns ``(point_ids, latency_ms)``.

    ``ef_search`` is the primary recall-vs-latency knob: must be
    ``>= top_k``; 64 is fast with ~95% recall on healthy graphs, 256
    nudges recall toward ~99% at higher cost.

    Uses ``client.query_points`` (qdrant-client >= 1.7); the legacy
    ``client.search`` method was removed in recent client versions.
    """
    qvec: list[float] = query_vector.astype(np.float32).tolist()
    flt: Filter | None = _topic_filter(topic_filter)

    start: float = time.perf_counter()
    response = client.query_points(
        collection_name=collection_name,    # HNSW collection to search
        query=qvec,                         # nearest-neighbor query vector; list[float] of length EMBEDDING_DIM
        limit=top_k,                        # top-K to return; matches the recall@K cutoff
        search_params=SearchParams(
            hnsw_ef=ef_search,              # dynamic candidate-list size at QUERY time; recall<->latency knob
            exact=False,                    # use the HNSW graph (approximate); True would force flat scan
        ),
        query_filter=flt,                   # when set, search space shrinks to matching payloads only (e.g. 500→~100 by topic)
    )
    latency_ms: float = (time.perf_counter() - start) * 1000.0
    return _point_ids_from_response(response), latency_ms


def search_flat(
    client: QdrantClient,
    query_vector: np.ndarray,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> tuple[list[int], float]:
    """Run exact brute-force search for ground-truth recall.

    Uses the HNSW collection with ``exact=True`` -- proves the same
    collection can serve both exact and ANN search depending on
    ``SearchParams``. The standalone FLAT collection exists for
    illustration; ``exact=True`` is sufficient in production.
    """
    qvec: list[float] = query_vector.astype(np.float32).tolist()

    start: float = time.perf_counter()
    response = client.query_points(
        collection_name=collection_name,    # same collection as ANN; only the search params differ
        query=qvec,                         # list[float], length EMBEDDING_DIM
        limit=top_k,                        # top-K for ground-truth comparison
        search_params=SearchParams(
            exact=True,                     # bypass HNSW graph; compute distance to every vector. Recall = 1.0 by definition.
        ),
    )
    latency_ms: float = (time.perf_counter() - start) * 1000.0
    return _point_ids_from_response(response), latency_ms


def count_vectors(client: QdrantClient, collection_name: str) -> int:
    """Return the exact number of points in a collection.

    Used by the ``--skip-upsert`` CLI flag to decide whether the
    collection is already populated. ``count`` with ``exact=True`` walks
    the collection synchronously; on 500 points it is effectively free.
    """
    if not client.collection_exists(collection_name=collection_name):
        return 0
    result = client.count(
        collection_name=collection_name,
        exact=True,                          # exact count (not the approximate estimate path); fine on small collections
    )
    return int(result.count)
