"""Single source of truth for "real backend vs offline deterministic path".

Components ask here instead of reading os.environ directly, so the offline default
is enforced in exactly one place. Mirrors the convention in 05-agentic-systems/5.3.
"""
from __future__ import annotations

import os

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_EMBEDDING_MODEL = os.environ.get(
    "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
ENTITY_COLLECTION = os.environ.get("ENTITY_COLLECTION", "graph_rag_entities")


def _key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "")


def use_llm() -> bool:
    """True only when explicitly enabled AND a key is present."""
    return os.environ.get("USE_LLM", "0") == "1" and bool(_key())


def use_embeddings_api() -> bool:
    """True when a key is present: real semantic embeddings are then available
    regardless of which extractor produced the entities."""
    return bool(_key())
