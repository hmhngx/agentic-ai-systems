"""Resolve which backend each component uses, from env. Offline by default.

This module is the single source of truth for "are we running the real LLM /
real RAGAS path, or the deterministic offline path?". Every component asks here
rather than reading os.environ directly, so the offline default is enforced in
exactly one place and the optional real backends are opt-in and key-gated.
"""
from __future__ import annotations

import os

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def use_llm() -> bool:
    """True only when explicitly enabled AND a key is present."""
    return os.environ.get("USE_LLM", "0") == "1" and bool(os.environ.get("OPENROUTER_API_KEY"))


def use_ragas() -> bool:
    """True only when explicitly enabled AND a key is present (ragas needs an LLM)."""
    return os.environ.get("USE_RAGAS", "0") == "1" and bool(os.environ.get("OPENROUTER_API_KEY"))
