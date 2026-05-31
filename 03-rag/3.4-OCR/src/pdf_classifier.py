"""
pdf_classifier.py — Classify PDFs before choosing extraction strategy.

Why classify before parsing?
  Different PDFs require completely different extraction strategies:
  - Born-digital: has a text layer → use Docling's native PDF backend
  - Scanned: bitmap images only → must enable OCR in Docling (slow)
  - Hybrid: mix of digital and scanned pages → per-page strategy needed

  Running OCR on a born-digital PDF wastes 30+ seconds per page.
  Skipping OCR on a scanned PDF produces empty chunks.
  Classification upfront prevents both failure modes.

Classification method:
  1. Open PDF with pymupdf (fitz)
  2. For each page: extract text via get_text("text")
  3. Count characters. If avg chars/page < TEXT_THRESHOLD: likely scanned.
  4. Check image count: if images/page > IMAGE_THRESHOLD and text is sparse: scanned.
  5. Return classification per page, not just per document.
     A 50-page report might have 48 digital pages and 2 scanned appendix pages.
"""

from __future__ import annotations

import os
from typing import Any

import fitz  # PyMuPDF: imported as ``fitz`` for historical reasons


TEXT_THRESHOLD: int = 100    # avg chars/page below this → suspect scanned
IMAGE_THRESHOLD: float = 0.8  # images per page above this → suspect image-heavy


class PDFType(str):
    """Enumeration of document-level classifications.

    Subclasses ``str`` so the values compare and serialize as plain
    strings (e.g. they drop straight into a JSON payload or an f-string)
    while still being namespaced under a single symbol.
    """

    DIGITAL = "digital"    # clean text layer, no OCR needed
    SCANNED = "scanned"    # bitmap only, OCR required
    HYBRID = "hybrid"      # mix — process page by page


def _page_is_scanned(char_count: int, image_count: int) -> bool:
    """Decide whether a single page looks scanned (bitmap-only).

    A page is treated as scanned when it has very little extractable text
    AND carries at least one raster image. The two conditions together
    avoid false positives: a genuinely short text page (e.g. a section
    divider) has little text but no image, so it stays "digital".
    """
    text_is_sparse: bool = char_count < TEXT_THRESHOLD
    has_image: bool = image_count >= 1
    return text_is_sparse and has_image


def classify_pdf(pdf_path: str) -> dict[str, Any]:
    """Classify ``pdf_path`` into a digital/scanned/hybrid strategy.

    Returns a classification dict (see module docstring for the schema).

    Raises:
        FileNotFoundError: if ``pdf_path`` does not exist on disk.
        ValueError: if the file cannot be opened as a valid PDF.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    basename: str = os.path.basename(pdf_path)

    try:
        document = fitz.open(pdf_path)
    except (RuntimeError, ValueError) as exc:
        # PyMuPDF raises RuntimeError/ValueError for corrupt or non-PDF input.
        raise ValueError(f"Not a valid PDF ({basename}): {exc}") from exc

    try:
        total_pages: int = document.page_count
        if total_pages == 0:
            raise ValueError(f"PDF has zero pages: {basename}")

        digital_pages: list[int] = []
        scanned_pages: list[int] = []
        total_chars: int = 0

        for page_index in range(total_pages):
            page = document.load_page(page_index)
            # get_text("text") returns the page's text-layer content; an
            # empty/near-empty result is the primary signal for a scan.
            page_text: str = page.get_text("text") or ""
            char_count: int = len(page_text.strip())
            total_chars += char_count

            # get_images(full=True) lists every embedded raster XObject on
            # the page; a scanned page is typically a single full-page image.
            image_count: int = len(page.get_images(full=True))

            if _page_is_scanned(char_count, image_count):
                scanned_pages.append(page_index)
            else:
                digital_pages.append(page_index)
    finally:
        # Always release the file handle even if classification raised,
        # so callers can immediately re-open the PDF with Docling.
        document.close()

    avg_chars_page: float = float(total_chars) / float(total_pages)

    if not scanned_pages:
        pdf_type: str = PDFType.DIGITAL
    elif not digital_pages:
        pdf_type = PDFType.SCANNED
    else:
        pdf_type = PDFType.HYBRID

    # OCR is needed whenever at least one page lacks an extractable text layer.
    needs_ocr: bool = len(scanned_pages) > 0

    if pdf_type == PDFType.DIGITAL:
        recommendation: str = (
            "Born-digital text layer detected on every page. "
            "Use Docling native PDF backend without OCR (fast)."
        )
    elif pdf_type == PDFType.SCANNED:
        recommendation = (
            "No usable text layer on any page. "
            "Enable Docling OCR (EasyOCR) — expect ~30s/page on CPU."
        )
    else:
        recommendation = (
            f"Hybrid document: {len(digital_pages)} digital page(s), "
            f"{len(scanned_pages)} scanned page(s). Enable OCR so the scanned "
            "pages are not dropped; digital pages still use the text layer."
        )

    print(
        f"Classified {basename}: {pdf_type} "
        f"({total_pages} pages, avg {avg_chars_page:.0f} chars/page)"
    )

    return {
        "pdf_path": pdf_path,
        "pdf_type": pdf_type,
        "total_pages": total_pages,
        "digital_pages": digital_pages,
        "scanned_pages": scanned_pages,
        "avg_chars_page": avg_chars_page,
        "needs_ocr": needs_ocr,
        "recommendation": recommendation,
    }
