"""Unit tests for src.pdf_loader."""

from __future__ import annotations

import pytest

from src.pdf_loader import load_pdf_as_text


def test_load_valid_pdf(tmp_pdf):
    """Valid PDF path returns stripped text with substantial content."""
    result = load_pdf_as_text(str(tmp_pdf))
    assert isinstance(result, str), "expected str return type from load_pdf_as_text"
    assert len(result) > 100, "expected extracted text longer than 100 characters"
    assert result.strip() == result, "expected no leading or trailing whitespace"


def test_load_missing_file():
    """Missing PDF path raises FileNotFoundError mentioning the path."""
    missing = "/nonexistent/path/file.pdf"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_pdf_as_text(missing)
    assert missing in str(exc_info.value), "expected exception message to contain file path"


def test_load_invalid_pdf(tmp_invalid_file):
    """Invalid PDF bytes raise an error rather than returning text."""
    import fitz

    with pytest.raises((ValueError, fitz.FileDataError, Exception)):
        load_pdf_as_text(str(tmp_invalid_file))


def test_pages_joined_with_double_newline(tmp_pdf):
    """Multi-page PDFs are joined with double newlines between pages."""
    result = load_pdf_as_text(str(tmp_pdf))
    assert "\n\n" in result, "expected page separator double newline in extracted text"


def test_load_blank_pdf_raises_value_error(tmp_path):
    """PDF with no extractable text raises ValueError (line 33 in pdf_loader)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    blank_pdf = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(blank_pdf), pagesize=letter)
    c.showPage()
    c.save()

    with pytest.raises(ValueError, match="no text"):
        load_pdf_as_text(str(blank_pdf))
