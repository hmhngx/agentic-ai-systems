"""Tests for prompt token budgeting (history trim + observation truncate)."""

import json

from src.prompt_builder import (
    build_conversation,
    build_system_prompt,
    trim_history_for_prompt,
    truncate_for_context,
)


def test_truncate_for_context():
    """Long observations must be capped before entering message history."""
    long_text = "x" * 500
    out = truncate_for_context(long_text, limit=100)
    assert len(out) < 150
    assert "truncated" in out


def test_trim_history_keeps_last_turn_only():
    """Only the latest assistant/observation pair is sent to the API by default."""
    history = [
        {"role": "assistant", "content": "turn1"},
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "turn2"},
        {"role": "user", "content": "obs2"},
    ]
    trimmed = trim_history_for_prompt(history)
    assert len(trimmed) == 2
    assert trimmed[0]["content"] == "turn2"


def test_build_conversation_uses_trimmed_history():
    """API messages must not include the full multi-turn trace."""
    history = [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "old obs"},
        {"role": "assistant", "content": "new"},
        {"role": "user", "content": "new obs"},
    ]
    messages = build_conversation("q", history)
    assert len(messages) == 4
    assert messages[-1]["content"] == "new obs"


def test_truncate_json_web_search_keeps_valid_json():
    """web_search observations should trim results[], not slice mid-JSON."""
    payload = {
        "query": "test",
        "provider": "duckduckgo",
        "results": [
            {"title": "A" * 40, "url": "https://a", "snippet": "snippet " * 30},
            {"title": "B" * 40, "url": "https://b", "snippet": "snippet " * 30},
        ],
    }
    raw = json.dumps(payload)
    out = truncate_for_context(raw, limit=400)
    parsed = json.loads(out)
    assert isinstance(parsed["results"], list)
    assert len(parsed["results"]) >= 1
    assert len(out) <= 400


def test_compact_system_prompt_is_small():
    """Compact prompt must stay small enough for low prompt-token limits."""
    assert len(build_system_prompt()) < 600
