"""Retrieval layer: vanilla vector search and hybrid graph+vector search.

Data flow:
  vanilla_retrieve  →  [RetrievedChunk × top_k]
  hybrid_retrieve   →  HybridResult(chunks, triples)

HybridResult.fused_context() returns the string injected into the LLM prompt.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import networkx as nx
import numpy as np
from qdrant_client import QdrantClient

from src import config


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    entities: list[str]
    source: str = "vector"  # "vector" | "graph_boosted"


@dataclass
class GraphTriple:
    source_entity: str
    relation: str
    target_entity: str
    score: float = 0.0


@dataclass
class HybridResult:
    chunks: list[RetrievedChunk]
    triples: list[GraphTriple]

    def fused_context(self) -> str:
        lines = ["[Retrieved Documents]"]
        for i, c in enumerate(self.chunks, 1):
            tag = " [graph-boosted]" if c.source == "graph_boosted" else ""
            lines.append(f"[{i}] (score={c.score:.3f}{tag}) {c.text}")
        lines.append("")
        lines.append("[Entity Relationships]")
        for t in self.triples:
            lines.append(f"- {t.source_entity} → [{t.relation}] → {t.target_entity}")
        return "\n".join(lines)


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _hash_embed_single(text: str) -> np.ndarray:
    row = np.zeros(256, dtype=np.float32)
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        row[h % 256] += 1.0
    norm = float(np.linalg.norm(row))
    return (row / norm if norm > 0 else row).astype(np.float32)


def _api_embed_single(text: str) -> np.ndarray:
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=config.OPENROUTER_BASE_URL)
    result = client.embeddings.create(model=config.OPENROUTER_EMBEDDING_MODEL, input=[text])
    vec = np.asarray(result.data[0].embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm if norm > 0 else vec)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string; route to API or offline hash."""
    if config.use_embeddings_api():
        return _api_embed_single(query)
    return _hash_embed_single(query)


# ── Vanilla retriever ─────────────────────────────────────────────────────────

def vanilla_retrieve(
    client: QdrantClient,
    query_vec: np.ndarray,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Return top-k chunks by cosine similarity. No graph involvement."""
    k = top_k if top_k is not None else config.TOP_K_CHUNKS
    resp = client.query_points(
        collection_name=config.CHUNK_COLLECTION,
        query=query_vec.astype(np.float32).tolist(),
        limit=k,
    )
    return [
        RetrievedChunk(
            chunk_id=p.payload["chunk_id"],
            text=p.payload["text"],
            score=float(p.score or 0.0),
            entities=p.payload.get("entities", []),
            source="vector",
        )
        for p in resp.points
    ]


# ── Graph traversal ───────────────────────────────────────────────────────────

def bfs_1hop(
    graph: nx.MultiDiGraph,
    seed_entities: list[str],
    max_neighbors: int | None = None,
) -> list[GraphTriple]:
    """Collect all (source, relation, target) triples exactly 1 hop from seed entities.

    Includes both outgoing edges (entity → neighbor) and incoming edges (predecessor → entity).
    Results are deduplicated by (source, relation, target) and capped at MAX_TRIPLES.
    """
    max_n = max_neighbors if max_neighbors is not None else config.MAX_NEIGHBORS
    seen: set[tuple[str, str, str]] = set()
    triples: list[GraphTriple] = []

    for entity in seed_entities:
        if entity not in graph:
            continue

        # Outgoing edges: entity → neighbor
        successors = list(graph.successors(entity))[:max_n]
        for nb in successors:
            for edge_data in graph[entity][nb].values():
                key = (entity, edge_data.get("relation", "related_to"), nb)
                if key not in seen:
                    seen.add(key)
                    triples.append(
                        GraphTriple(
                            source_entity=entity,
                            relation=edge_data.get("relation", "related_to"),
                            target_entity=nb,
                            score=float(edge_data.get("weight", 1.0)),
                        )
                    )

        # Incoming edges: predecessor → entity
        predecessors = list(graph.predecessors(entity))[:max_n]
        for pred in predecessors:
            for edge_data in graph[pred][entity].values():
                key = (pred, edge_data.get("relation", "related_to"), entity)
                if key not in seen:
                    seen.add(key)
                    triples.append(
                        GraphTriple(
                            source_entity=pred,
                            relation=edge_data.get("relation", "related_to"),
                            target_entity=entity,
                            score=float(edge_data.get("weight", 1.0)),
                        )
                    )

    triples.sort(key=lambda t: t.score, reverse=True)
    return triples[: config.MAX_TRIPLES]
