"""Extraction façade: pick the backend, return a uniform ExtractionResult."""
from __future__ import annotations

from src import config
from src.extractor_llm import extract_llm
from src.extractor_offline import extract_offline
from src.loader import Page
from src.schema import ExtractionResult


def extract(pages: list[Page]) -> ExtractionResult:
    """Route to the LLM extractor when enabled, else the deterministic offline one."""
    if config.use_llm():
        return extract_llm(pages)
    return extract_offline(pages)
