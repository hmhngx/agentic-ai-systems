"""Root conftest for the 3.2-NaiveRAG test suite.

This file lives at the package root so ``from src.embedder import ...``
imports resolve the same way they do when running ``naive_rag.py``
directly. Without this, the tests would have to second-guess the
caller's working directory.
"""

from __future__ import annotations

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parent))


def pytest_configure(config):
    """Register the two custom markers the suite uses.

    ``integration`` gates Qdrant + API-key tests so they auto-skip in
    offline runs. ``slow`` flags tests that take more than ten seconds
    so CI can opt them out with ``-m 'not slow'`` when wall-clock
    matters more than coverage.
    """
    config.addinivalue_line("markers", "integration: requires Qdrant + API keys")
    config.addinivalue_line("markers", "slow: takes more than 10 seconds")
