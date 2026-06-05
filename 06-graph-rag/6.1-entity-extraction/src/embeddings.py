"""Entity embeddings for Qdrant.

Real path: OpenRouter `text-embedding-3-small` (1536-dim) when a key is set.
Offline path: a deterministic hashed bag-of-tokens vector (256-dim) so the whole
pipeline — including Qdrant upsert and search — runs with no key or network.
The offline vectors are NON-SEMANTIC placeholders (stable across runs); we print a
clear notice when they are used.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from src import config
from src.schema import Entity

_API_DIM = 1536
_OFFLINE_DIM = 256


def embedding_dim() -> int:
    return _API_DIM if config.use_embeddings_api() else _OFFLINE_DIM


def entity_text(e: Entity) -> str:
    """The string we embed for an entity: name + type + aliases."""
    parts = [e.name, f"type: {e.type.value}"]
    if e.aliases:
        parts.append("aka " + ", ".join(e.aliases))
    return " | ".join(parts)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (m / safe).astype(np.float32)


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic hashed bag-of-tokens embedding (offline fallback)."""
    rows = np.zeros((len(texts), _OFFLINE_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            rows[i, h % _OFFLINE_DIM] += 1.0
    return _normalize(rows)


def _api_embed(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=config.OPENROUTER_BASE_URL)
    result = client.embeddings.create(model=config.OPENROUTER_EMBEDDING_MODEL, input=texts)
    return _normalize(np.asarray([d.embedding for d in result.data], dtype=np.float32))


def embed_entities(entities: list[Entity]) -> np.ndarray:
    """Return an (n, embedding_dim()) L2-normalized float32 matrix for the entities."""
    if not entities:
        return np.zeros((0, embedding_dim()), dtype=np.float32)
    texts = [entity_text(e) for e in entities]
    if config.use_embeddings_api():
        return _api_embed(texts)
    print("NOTE: no OPENROUTER_API_KEY — using deterministic offline (non-semantic) embeddings.")
    return _hash_embed(texts)
