"""Pytest config: put the day folder on sys.path and provide shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SAMPLE_TEXT = (
    "Nelson Liu and John Hewitt study language models at Stanford University. "
    "Percy Liang advises the project at Stanford University. "
    "GPT-3.5-Turbo was evaluated on the NaturalQuestions dataset. "
    "Claude was also evaluated on NaturalQuestions. "
    "Kevin Lin works at the University of California, Berkeley. "
    "Samaya AI collaborated with Stanford University on multi-document question answering."
)


@pytest.fixture
def sample_text() -> str:
    """A few sentences with known, clustered entities (people/orgs/models/datasets)."""
    return SAMPLE_TEXT
