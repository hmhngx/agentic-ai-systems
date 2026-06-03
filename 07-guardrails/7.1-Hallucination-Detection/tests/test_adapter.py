"""Unit tests for the pure part of the naive-RAG adapter (no RAG running)."""

from __future__ import annotations

import pytest

from src.rag_adapter import chunks_from_naive_rag, run_naive_rag_and_detect


def test_maps_rank_to_doc_label():
    results = [
        {"rank": 1, "text": "first", "score": 0.9, "page_num": 2},
        {"rank": 2, "text": "second", "score": 0.5, "page_num": 7},
    ]
    chunks = chunks_from_naive_rag(results)
    assert [c.chunk_id for c in chunks] == ["Doc 1", "Doc 2"]
    assert [c.text for c in chunks] == ["first", "second"]
    assert chunks[0].score == 0.9


def test_handles_missing_score():
    chunks = chunks_from_naive_rag([{"rank": 1, "text": "t", "score": None}])
    assert chunks[0].score is None


def test_empty_results():
    assert chunks_from_naive_rag([]) == []


def test_live_missing_pdf_raises_before_touching_qdrant(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    with pytest.raises(RuntimeError, match="PDF file not found"):
        run_naive_rag_and_detect("q", pdf_path="does_not_exist.pdf")


def test_live_missing_api_key_raises_before_touching_qdrant(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_naive_rag_and_detect("q")
