from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

os.environ.setdefault("OUTPUT_GUARD_USE_LLM", "0")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires OPENROUTER_API_KEY"
    )
