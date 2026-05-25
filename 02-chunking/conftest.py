"""Pytest path setup so `src` is importable without editable install."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
