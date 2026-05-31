"""
docling_parser.py — Docling-based PDF parsing into typed regions.

Why Docling over pdfplumber?
  pdfplumber extracts raw text in stream order — it cannot:
    - Distinguish a paragraph from a table from a figure caption
    - Handle scanned PDFs (no text layer to extract)
    - Detect reading order in multi-column documents
    - Parse table structure (rows, columns, spanning cells)

  Docling uses DocLayNet (RT-DETR object detector) to classify every
  bounding box into semantic region types, then TableFormer (vision
  transformer) to recover table grid structure. The result is a
  DoclingDocument with typed nodes and correct reading order.

Fallback strategy:
  If Docling raises any exception (import error, model download failure,
  PDF parsing error): fall back to pdfplumber for basic text extraction.
  The fallback produces NO table structure and NO heading detection.
  It logs a warning and tags all chunks as region_type="unknown_fallback".
  The pipeline continues — it does not crash.
  This is critical because Docling's model downloads can fail in restricted
  network environments.

Region types extracted from DoclingDocument:
  TITLE / HEADING: nodes with label in (Title, SectionHeader)
  TEXT:            nodes with label Text or Paragraph
  TABLE:           nodes with label Table (handled by table_serializer)
  LIST:            nodes with label List or ListItem
  FIGURE:          nodes with label Picture (stored as metadata-only, no text)
  CAPTION:         nodes with label Caption (attached to nearest Figure/Table)
  HEADER_FOOTER:   nodes with label PageHeader or PageFooter (discarded — noise)

Why discard headers/footers?
  Page headers ("Company Name — Confidential") and footers ("Page 1 of 50")
  repeat on every page. If embedded, they pollute the vector index with
  identical chunks that score highly for generic queries, degrading precision.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import pdfplumber


# Map Docling ``DocItemLabel`` string values to our normalized region types.
# Docling's enum values are lowercase strings (e.g. "section_header"); we
# compare on the value so the mapping is resilient to enum import quirks.
_LABEL_MAP: dict[str, str] = {
    "title": "title",
    "section_header": "heading",
    "text": "text",
    "paragraph": "text",
    "code": "text",        # code blocks embed fine as prose-like text
    "footnote": "text",
    "formula": "text",     # equations treated as text (see module/CLI notes)
    "list_item": "list",
    "table": "table",
    "picture": "figure",
    "caption": "caption",
}

# Labels that are pure page furniture: discarded entirely so they never
# pollute the vector index with repeated, low-signal chunks.
_DISCARD_LABELS: frozenset[str] = frozenset({"page_header", "page_footer"})


def _label_value(label: Any) -> str:
    """Return the lowercase string value of a Docling ``DocItemLabel``.

    ``DocItemLabel`` is a str-based enum, so ``.value`` is the canonical
    lowercase token. We fall back to ``str(label)`` for any exotic label
    type and normalize case so the lookup in ``_LABEL_MAP`` is stable.
    """
    value: str = str(getattr(label, "value", label))
    return value.strip().lower()


def _normalize_bbox(bbox: Any, page_width: float, page_height: float) -> list[float]:
    """Normalize a Docling ``BoundingBox`` to [x0, y0, x1, y1] in 0-1 space.

    Normalization makes the bbox resolution-independent so downstream
    consumers (e.g. a future Vision LLM crop step) do not need the page
    dimensions. Values are clamped to [0, 1] to absorb minor overshoot
    from coordinate-origin differences.
    """
    if page_width <= 0.0 or page_height <= 0.0:
        return [0.0, 0.0, 1.0, 1.0]

    left: float = float(getattr(bbox, "l", 0.0))
    top: float = float(getattr(bbox, "t", 0.0))
    right: float = float(getattr(bbox, "r", page_width))
    bottom: float = float(getattr(bbox, "b", page_height))

    def _clamp(value: float, denom: float) -> float:
        scaled: float = value / denom
        return max(0.0, min(1.0, scaled))

    x0: float = _clamp(min(left, right), page_width)
    x1: float = _clamp(max(left, right), page_width)
    y0: float = _clamp(min(top, bottom), page_height)
    y1: float = _clamp(max(top, bottom), page_height)
    return [x0, y0, x1, y1]


def _page_sizes(doc: Any) -> dict[int, tuple[float, float]]:
    """Build a {page_no: (width, height)} map for bbox normalization.

    ``doc.pages`` is keyed by 1-indexed page number; each ``PageItem``
    exposes a ``size`` with ``width``/``height``. We read defensively
    because some backends omit page sizes for image-only pages.
    """
    sizes: dict[int, tuple[float, float]] = {}
    pages: dict[int, Any] = getattr(doc, "pages", {}) or {}
    for page_no, page in pages.items():
        size = getattr(page, "size", None)
        width: float = float(getattr(size, "width", 0.0)) if size is not None else 0.0
        height: float = float(getattr(size, "height", 0.0)) if size is not None else 0.0
        sizes[int(page_no)] = (width, height)
    return sizes


def _table_to_markdown(table_item: Any, doc: Any) -> str:
    """Export a Docling table node to a Markdown table string.

    ``export_to_markdown`` requires the owning ``DoclingDocument`` in
    recent docling-core releases but accepted no argument in older ones.
    We try the documented (doc-passing) form first, then fall back, so
    the module works across docling-core versions.
    """
    try:
        return str(table_item.export_to_markdown(doc))
    except TypeError:
        return str(table_item.export_to_markdown())


def _docling_parse(pdf_path: str, enable_ocr: bool) -> list[dict[str, Any]]:
    """Run the real Docling pipeline. Raises on any Docling-side failure.

    Kept separate from :func:`parse_pdf_to_regions` so the caller can wrap
    the entire Docling path (imports included) in a single try/except and
    fall back to pdfplumber when anything here breaks.
    """
    # Imports live inside the function so an ImportError (e.g. Docling not
    # installed, or a model-download failure during lazy init) is caught by
    # the caller's fallback handler instead of crashing module import.
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc.document import TableItem

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    # do_ocr=True: enables EasyOCR for scanned pages (slow, ~30s/page on CPU)
    # do_ocr=False: uses native PDF text layer (fast, ~0.5s/page)
    pipeline_options.do_table_structure = True
    # do_table_structure: enables TableFormer for table grid recovery.
    # TableFormer detects rows, columns, spanning cells even without borders.

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(pdf_path)
    doc = result.document

    basename: str = os.path.basename(pdf_path)
    page_sizes: dict[int, tuple[float, float]] = _page_sizes(doc)

    # Heading stack of (level, text) tuples. Title sits at level 0; section
    # headers carry their own nesting level. Snapshotting the stack at each
    # node yields that node's full ancestor heading path.
    heading_stack: list[tuple[int, str]] = []
    regions: list[dict[str, Any]] = []
    reading_order: int = 0

    for item, tree_level in doc.iterate_items():
        label = getattr(item, "label", None)
        if label is None:
            # Group/container nodes have no label; they only hold structure.
            continue

        label_value: str = _label_value(label)
        if label_value in _DISCARD_LABELS:
            continue  # page header/footer furniture — never embedded

        region_type: Optional[str] = _LABEL_MAP.get(label_value)
        if region_type is None:
            # Unknown label: keep it as text if it carries any, else skip.
            region_type = "text" if getattr(item, "text", "") else None
        if region_type is None:
            continue

        is_table: bool = region_type == "table" and isinstance(item, TableItem)

        # Resolve text content per region type.
        if is_table:
            text: str = _table_to_markdown(item, doc)
        else:
            text = str(getattr(item, "text", "") or "")

        # Maintain the heading stack as we encounter title/heading nodes.
        if region_type == "title":
            heading_stack = [(0, text.strip())] if text.strip() else []
        elif region_type == "heading":
            level: int = int(getattr(item, "level", 1) or 1)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if text.strip():
                heading_stack.append((level, text.strip()))

        heading_path: list[str] = [t for _, t in heading_stack if t]

        # Provenance: first prov entry gives the source page and bbox.
        prov_list: list[Any] = list(getattr(item, "prov", []) or [])
        if prov_list:
            page_num: int = int(getattr(prov_list[0], "page_no", 1) or 1)
            width, height = page_sizes.get(page_num, (0.0, 0.0))
            bbox: list[float] = _normalize_bbox(
                getattr(prov_list[0], "bbox", None), width, height
            )
        else:
            page_num = 1
            bbox = [0.0, 0.0, 1.0, 1.0]

        table_data: Optional[dict[str, Any]] = None
        if is_table:
            # Carry the live Docling node + owning doc so table_serializer can
            # re-export Markdown and access cell-level rows when splitting.
            table_data = {
                "docling_item": item,
                "doc": doc,
                "markdown": text,
            }

        regions.append(
            {
                "region_id": str(uuid.uuid4()),
                "region_type": region_type,
                "text": text,
                "page_num": page_num,
                "bbox": bbox,
                "reading_order": reading_order,
                "heading_path": heading_path,
                "source_pdf": basename,
                "table_data": table_data,
            }
        )
        reading_order += 1

    # Filter empty regions. Figures legitimately carry no text (they are
    # metadata-only), so they are exempt; every other empty region is noise.
    cleaned: list[dict[str, Any]] = [
        region
        for region in regions
        if region["region_type"] == "figure" or region["text"].strip() != ""
    ]
    return cleaned


def parse_pdf_to_regions(pdf_path: str, enable_ocr: bool = False) -> list[dict[str, Any]]:
    """Parse ``pdf_path`` into typed region dicts using Docling.

    Returns a list of region dicts (schema in the module docstring). On any
    Docling failure, transparently falls back to pdfplumber text extraction
    so the pipeline never crashes on a single problematic PDF.

    Raises:
        FileNotFoundError: if ``pdf_path`` does not exist.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    basename: str = os.path.basename(pdf_path)

    try:
        regions: list[dict[str, Any]] = _docling_parse(pdf_path, enable_ocr)
    except Exception as exc:  # noqa: BLE001 - any Docling failure routes to fallback
        # Broad-but-explicit: import errors, model download failures, and
        # backend parse errors all funnel into the pdfplumber fallback.
        print(f"WARNING: Docling failed ({exc}). Falling back to pdfplumber.")
        regions = _pdfplumber_fallback(pdf_path)

    n_text: int = sum(1 for r in regions if r["region_type"] in ("text", "list"))
    n_table: int = sum(1 for r in regions if r["region_type"] == "table")
    n_heading: int = sum(1 for r in regions if r["region_type"] in ("title", "heading"))
    print(
        f"Parsed {basename}: {len(regions)} regions "
        f"({n_text} text, {n_table} tables, {n_heading} headings)"
    )
    return regions


def _pdfplumber_fallback(pdf_path: str) -> list[dict[str, Any]]:
    """pdfplumber-based extraction. Used only when Docling fails.

    Why pdfplumber fails on scanned PDFs:
      pdfplumber reads the PDF's text layer — bytes that represent Unicode
      characters. Scanned PDFs contain bitmap images with NO text layer.
      ``pages[0].extract_text()`` returns "" or None for those documents.
      This is why Docling with OCR is the correct tool for scanned docs.

    For born-digital PDFs where Docling failed for other reasons:
      pdfplumber can still extract text — just without structure.
    """
    basename: str = os.path.basename(pdf_path)
    regions: list[dict[str, Any]] = []
    reading_order: int = 0
    n_pages: int = 0

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages):
            page_text: str = page.extract_text() or ""
            if not page_text.strip():
                continue
            # Split into paragraphs on blank lines; this is the only structure
            # signal available without a layout model.
            paragraphs: list[str] = [
                block.strip()
                for block in page_text.split("\n\n")
                if block.strip()
            ]
            for paragraph in paragraphs:
                regions.append(
                    {
                        "region_id": str(uuid.uuid4()),
                        "region_type": "text",
                        "text": paragraph,
                        "page_num": page_index + 1,  # 1-indexed for citations
                        "bbox": [0.0, 0.0, 1.0, 1.0],  # full page: no layout info
                        "reading_order": reading_order,
                        "heading_path": [],            # no heading detection in fallback
                        "source_pdf": basename,
                        "table_data": None,            # no table structure in fallback
                    }
                )
                reading_order += 1

    print(f"Fallback extraction: {len(regions)} text regions from {n_pages} pages")
    return regions
