"""Qdrant storage for entity embeddings (local Docker by default).

Mirrors the day-3.4 vector store but the unit of storage is an ENTITY, not a chunk.
Payload carries graph attributes (type, degree, mentions, aliases) so a future
hybrid retriever can filter ("MODEL entities only") without touching the graph.
"""
from __future__ import annotations

import hashlib
import sys
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from src import config
from src.schema import Entity


def get_client() -> QdrantClient:
    try:
        client = QdrantClient(url=config.QDRANT_URL, timeout=30)
        client.get_collections()
        return client
    except (ConnectionError, ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        print(
            f"ERROR: Could not connect to Qdrant at {config.QDRANT_URL} "
            f"({type(exc).__name__}: {exc}).\nStart it with:\n"
            "  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest",
            file=sys.stderr,
        )
        sys.exit(1)


def point_id(name: str) -> int:
    """Stable uint63 id derived from the canonical entity name (idempotent upserts)."""
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (2**63)


def create_collection(client: QdrantClient, dim: int) -> None:
    if client.collection_exists(config.ENTITY_COLLECTION):
        client.delete_collection(config.ENTITY_COLLECTION)
    client.create_collection(
        collection_name=config.ENTITY_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Collection '{config.ENTITY_COLLECTION}' created ({dim}-dim COSINE).")


def upsert_entities(
    client: QdrantClient,
    entities: list[Entity],
    vectors: np.ndarray,
    degrees: dict[str, int] | None = None,
) -> None:
    if len(entities) != vectors.shape[0]:
        raise ValueError(f"entities ({len(entities)}) != vectors ({vectors.shape[0]})")
    degrees = degrees or {}
    points = [
        PointStruct(
            id=point_id(e.name),
            vector=vectors[i].tolist(),
            payload={
                "name": e.name,
                "type": e.type.value,
                "mentions": e.mentions,
                "aliases": e.aliases,
                "degree": degrees.get(e.name, 0),
            },
        )
        for i, e in enumerate(entities)
    ]
    client.upsert(collection_name=config.ENTITY_COLLECTION, points=points, wait=True)
    count = client.count(collection_name=config.ENTITY_COLLECTION, exact=True).count
    print(f"Upserted {len(points)} entities; collection now holds {int(count)}.")


def search(client: QdrantClient, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    resp = client.query_points(
        collection_name=config.ENTITY_COLLECTION,
        query=query_vector.astype(np.float32).tolist(),
        limit=top_k,
    )
    out = []
    for rank, p in enumerate(resp.points, start=1):
        payload = p.payload or {}
        out.append(
            {
                "rank": rank,
                "score": float(p.score or 0.0),
                "name": payload.get("name", ""),
                "type": payload.get("type", ""),
                "degree": int(payload.get("degree", 0)),
            }
        )
    return out
