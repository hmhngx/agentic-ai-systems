"""Root conftest for the 7.1 hallucination-detector test suite.

Lives at the package root so ``from src.detector import ...`` resolves the same
way it does when running ``hallucination_detector.py`` directly, regardless of
the caller's working directory.
"""

from __future__ import annotations

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parent))


def pytest_configure(config):
    """Register custom markers used by the suite."""
    config.addinivalue_line(
        "markers", "integration: requires OpenRouter API key (LLM judge / live RAG)"
    )
