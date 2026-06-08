from __future__ import annotations

import os

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_EMBEDDING_MODEL = os.environ.get(
    "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
CHUNK_COLLECTION = os.environ.get("CHUNK_COLLECTION", "graph_rag_chunks")
ENTITY_COLLECTION = os.environ.get("ENTITY_COLLECTION", "graph_rag_entities")
OBSIDIAN_VAULT = os.environ.get(
    "OBSIDIAN_VAULT",
    r"C:\Users\minhh\OneDrive\Documents\Obsidian Vault",
)

MAX_NEIGHBORS = int(os.environ.get("MAX_NEIGHBORS", "10"))
MAX_TRIPLES = int(os.environ.get("MAX_TRIPLES", "20"))
TOP_K_CHUNKS = int(os.environ.get("TOP_K_CHUNKS", "5"))

_API_DIM = 1536
_OFFLINE_DIM = 256


def _key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "")


def use_llm() -> bool:
    return os.environ.get("USE_LLM", "0") == "1" and bool(_key())


def use_embeddings_api() -> bool:
    return bool(_key())


def embedding_dim() -> int:
    return _API_DIM if use_embeddings_api() else _OFFLINE_DIM
