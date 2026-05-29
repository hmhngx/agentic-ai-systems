"""Root conftest for the 3.3-AdvancedRAG test suite."""

from __future__ import annotations

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parent))


def pytest_configure(config):
    """Register custom markers for integration and slow tests."""
    config.addinivalue_line("markers", "integration: requires Qdrant + API keys")
    config.addinivalue_line("markers", "slow: takes more than 10 seconds")
