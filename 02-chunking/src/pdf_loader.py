"""PDF text extraction using pymupdf (fitz)."""

from pathlib import Path

import fitz


def load_pdf_as_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF using pymupdf (fitz).

    Joins pages with double newline to preserve paragraph boundaries.
    Raises FileNotFoundError if pdf_path does not exist.
    Raises ValueError if the extracted text is empty.
    Returns the full document as a single string.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc: fitz.Document | None = None
    try:
        doc = fitz.open(pdf_path)
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        text = "\n\n".join(pages).strip()
    finally:
        if doc is not None:
            doc.close()

    if len(text) == 0:
        raise ValueError("PDF extracted no text — may be scanned/image-only")

    return text
