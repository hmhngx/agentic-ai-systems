"""Shared pytest fixtures for the ReAct agent test suite."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(AGENT_DIR / ".env", override=True)


@pytest.fixture
def mock_llm_responses():
    """
    Factory: patch call_llm to return a fixed sequence of LLM strings (function-scoped).
    """
    patches = []

    def _apply(responses: list[str]):
        call_iter = iter(responses)

        def _side_effect(*_args, **_kwargs):
            try:
                return next(call_iter)
            except StopIteration:
                raise AssertionError("call_llm called more times than mock responses provided")

        p = patch("src.react_loop.call_llm", side_effect=_side_effect)
        patches.append(p)
        return p.start()

    yield _apply

    for p in reversed(patches):
        p.stop()


@pytest.fixture(scope="session")
def api_key_present():
    """Skip integration tests when OPENROUTER_API_KEY is unset."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")


@pytest.fixture(scope="session")
def sample_notes_file():
    """Path to sample_data/notes.txt used by file_read demo tests."""
    path = AGENT_DIR / "sample_data" / "notes.txt"
    if not path.is_file():
        pytest.fail("sample_data/notes.txt missing")
    return path
