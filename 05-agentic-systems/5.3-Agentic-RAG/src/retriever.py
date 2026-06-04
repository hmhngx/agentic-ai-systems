"""Retrieval over an in-memory Qdrant collection of tf-idf vectors.

Qdrant runs in :memory: mode — no server, no Docker, fully offline. The space
and the index are built once (lazy singletons) from the corpus. Below MIN_SCORE
(default 0.0, i.e. any in-vocab overlap) nothing is returned, so an OOV-only
query yields [] -> the NO_RETRIEVAL failure mode.
"""
from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.corpus import documents
from src.embedder import TfidfSpace, build_space

TOP_K = 4
MIN_SCORE = 0.0
_COLLECTION = "helios_chunks"

_space: Optional[TfidfSpace] = None
_client: Optional[QdrantClient] = None


def get_space() -> TfidfSpace:
    global _space
    if _space is None:
        _space = build_space(documents())
    return _space


def _get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client
    space = get_space()
    docs = documents()
    client = QdrantClient(location=":memory:")
    # Fresh in-process client each run, so a plain create is correct (no recreate).
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config=VectorParams(size=space.dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=i, vector=space.embed(d["text"]).tolist(),
                    payload={"doc_id": d["doc_id"], "text": d["text"]})
        for i, d in enumerate(docs)
    ]
    client.upsert(collection_name=_COLLECTION, points=points, wait=True)
    _client = client
    return _client


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the query, search, threshold, and label results [Doc 1..N]."""
    space = get_space()
    qvec = space.embed(query)
    if float((qvec != 0).sum()) == 0:     # zero vector => no in-vocab signal
        return []
    client = _get_client()
    hits = client.query_points(
        collection_name=_COLLECTION, query=qvec.tolist(), limit=top_k,
    ).points
    chunks: list[dict] = []
    rank = 0
    for h in hits:
        score = float(h.score) if h.score is not None else 0.0
        if score <= MIN_SCORE:
            continue
        rank += 1
        payload = h.payload or {}
        chunks.append({
            "doc_id": payload.get("doc_id", ""),
            "text": payload.get("text", ""),
            "score": score,
            "rank": rank,
            "label": f"Doc {rank}",
        })
    return chunks
