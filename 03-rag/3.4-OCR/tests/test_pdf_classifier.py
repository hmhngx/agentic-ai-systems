"""Offline tests for PDF classification — wrong strategy wastes OCR time or drops scanned text."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pdf_classifier import classify_pdf


def test_classify_pdf_returns_correct_structure(tiny_pdf: Path) -> None:
    """Verifies classify_pdf returns the full strategy dict needed before parsing."""
    result = classify_pdf(str(tiny_pdf))
    assert isinstance(result, dict)
    required_keys = {
        "pdf_path",
        "pdf_type",
        "total_pages",
        "digital_pages",
        "scanned_pages",
        "avg_chars_page",
        "needs_ocr",
        "recommendation",
    }
    assert required_keys.issubset(set(result.keys())), (
        f"Missing keys: {required_keys - set(result.keys())} — "
        "pipeline cannot choose extraction strategy without complete classification"
    )


def test_classify_digital_pdf_as_digital(tiny_pdf: Path) -> None:
    """Verifies born-digital PDFs skip OCR — enabling OCR wastes ~30s per page."""
    result = classify_pdf(str(tiny_pdf))
    assert result["pdf_type"] == "digital", (
        f"Born-digital PDF must classify as 'digital', got '{result['pdf_type']}'"
    )
    assert result["needs_ocr"] is False, (
        "Born-digital PDF must not require OCR — enabling OCR wastes 30s/page"
    )


def test_classify_pdf_page_counts_correct(tiny_pdf: Path) -> None:
    """Verifies page count drives per-page OCR budgeting in hybrid documents."""
    result = classify_pdf(str(tiny_pdf))
    assert result["total_pages"] == 2, (
        f"tiny_pdf has 2 pages, got {result['total_pages']}"
    )


def test_classify_pdf_avg_chars_positive(tiny_pdf: Path) -> None:
    """Verifies text density signal — zero avg chars means empty extraction and no RAG content."""
    result = classify_pdf(str(tiny_pdf))
    assert result["avg_chars_page"] > 0, (
        "Born-digital PDF must have positive avg chars/page"
    )


def test_classify_missing_file_raises() -> None:
    """Verifies missing PDFs fail fast before expensive Docling model loads."""
    with pytest.raises(FileNotFoundError):
        classify_pdf("/nonexistent/path/file.pdf")


def test_classify_pdf_recommendation_is_string(tiny_pdf: Path) -> None:
    """Verifies recommendation explains strategy — operators need actionable guidance."""
    result = classify_pdf(str(tiny_pdf))
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 10, (
        "Recommendation must be a meaningful explanation, not empty"
    )


def test_classify_pdf_digital_pages_list(tiny_pdf: Path) -> None:
    """Verifies per-page digital list enables hybrid OCR on scanned pages only."""
    result = classify_pdf(str(tiny_pdf))
    assert isinstance(result["digital_pages"], list)
    assert all(isinstance(p, int) for p in result["digital_pages"])
