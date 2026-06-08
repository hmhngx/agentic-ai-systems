"""Build the Qdrant chunk collection + NetworkX knowledge graph from the corpus.

Call build_index() once before running queries. It recreates the Qdrant collection
on every call (idempotent), so re-indexing is safe and always fresh.
"""
from __future__ import annotations

import hashlib
import re
import sys

import networkx as nx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from src import config
from src.corpus import PASSAGES
from src.knowledge_base import build_curated_graph


def _get_client() -> QdrantClient:
    try:
        client = QdrantClient(url=config.QDRANT_URL, timeout=30)
        client.get_collections()
        return client
    except (ConnectionError, ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        print(
            f"ERROR: Cannot connect to Qdrant at {config.QDRANT_URL} "
            f"({type(exc).__name__}).\n"
            "Start it with:  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest",
            file=sys.stderr,
        )
        sys.exit(1)


def _chunk_id(chunk_id_str: str) -> int:
    """Stable uint63 derived from chunk string id."""
    return int(hashlib.md5(chunk_id_str.encode()).hexdigest(), 16) % (2**63)


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-tokens embedding (offline fallback, non-semantic)."""
    rows = np.zeros((len(texts), 256), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            rows[i, h % 256] += 1.0
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (rows / safe).astype(np.float32)


def _api_embed(texts: list[str]) -> np.ndarray:
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=config.OPENROUTER_BASE_URL)
    result = client.embeddings.create(model=config.OPENROUTER_EMBEDDING_MODEL, input=texts)
    rows = np.asarray([d.embedding for d in result.data], dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (rows / safe).astype(np.float32)


def _embed(texts: list[str]) -> np.ndarray:
    if config.use_embeddings_api():
        return _api_embed(texts)
    print("NOTE: no OPENROUTER_API_KEY — using deterministic offline (non-semantic) embeddings.")
    return _hash_embed(texts)


def _link_entities_to_chunks(graph: nx.MultiDiGraph) -> dict[str, list[str]]:
    """For each passage, return which graph entity names appear in its text (case-insensitive)."""
    entity_names = list(graph.nodes())
    result: dict[str, list[str]] = {}
    for p in PASSAGES:
        text_lower = p["text"].lower()
        found = [e for e in entity_names if e.lower() in text_lower]
        result[p["id"]] = found
    return result


def build_index() -> tuple[QdrantClient, nx.MultiDiGraph]:
    """Embed all corpus passages, upsert to Qdrant, return (client, graph)."""
    client = _get_client()
    graph = build_curated_graph()

    dim = config.embedding_dim()

    if client.collection_exists(config.CHUNK_COLLECTION):
        client.delete_collection(config.CHUNK_COLLECTION)
    client.create_collection(
        collection_name=config.CHUNK_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    chunk_entities = _link_entities_to_chunks(graph)
    texts = [p["text"] for p in PASSAGES]
    vectors = _embed(texts)

    points = [
        PointStruct(
            id=_chunk_id(p["id"]),
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": p["id"],
                "text": p["text"],
                "entities": chunk_entities.get(p["id"], []),
            },
        )
        for i, p in enumerate(PASSAGES)
    ]
    client.upsert(collection_name=config.CHUNK_COLLECTION, points=points, wait=True)
    count = client.count(collection_name=config.CHUNK_COLLECTION, exact=True).count
    print(f"Indexed {len(points)} chunks → '{config.CHUNK_COLLECTION}' ({int(count)} total).")

    return client, graph
