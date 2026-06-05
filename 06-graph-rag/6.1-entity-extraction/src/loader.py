"""Load a document into per-page text. Supports PDF (pymupdf) and plain text.

We keep page boundaries because evidence sentences and (later) citations are
more useful when attributable to a page. Relation/sentence segmentation is left
to the extractor, which already runs a spaCy pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A standalone "References"/"Bibliography" line marks the start of back-matter in a
# paper. Everything after it (citation lists, appendices) is dense co-occurrence
# noise that swamps the real entities, so we cut there by default.
_BACK_MATTER = re.compile(r"(?im)^\s*(references|bibliography)\s*$")

# The "Abstract" heading marks the start of real content. Everything before it — the
# title, author list, and affiliation block — extracts as garbled, glued-together
# tokens (superscript digits, run-together names) that form a noise clique. We cut it
# off, but only when "Abstract" appears early on page 1 (guards plain-text docs that
# merely use the word "abstract" mid-body).
_FRONT_MATTER = re.compile(r"(?i)\babstract\b")
_FRONT_MATTER_MAX_OFFSET = 2000


@dataclass
class Page:
    page_num: int          # 1-indexed
    text: str


def load_pages(path: str) -> list[Page]:
    """Return the document as a list of Page. Raises FileNotFoundError if absent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"document not found: {path}")

    if p.suffix.lower() in {".txt", ".md"}:
        return [Page(page_num=1, text=p.read_text(encoding="utf-8", errors="replace"))]

    # default: treat as PDF
    import fitz  # pymupdf; imported lazily so .txt paths don't need it

    doc = fitz.open(str(p))
    try:
        return [Page(page_num=i + 1, text=doc[i].get_text()) for i in range(doc.page_count)]
    finally:
        doc.close()


def full_text(pages: list[Page]) -> str:
    """Concatenate page texts in reading order."""
    return "\n".join(pg.text for pg in pages)


def main_body_pages(pages: list[Page]) -> list[Page]:
    """Return only the main body: from the Abstract up to (not incl.) References.

    Trims the title/author front-matter on page 1 (when "Abstract" appears early) and
    drops everything from the References/Bibliography heading onward. If neither marker
    is found (e.g. a plain-text doc), pages pass through unchanged.
    """
    out: list[Page] = []
    for idx, pg in enumerate(pages):
        text = pg.text
        if idx == 0:
            fm = _FRONT_MATTER.search(text)
            if fm and fm.start() < _FRONT_MATTER_MAX_OFFSET:
                text = text[fm.start():]          # keep "Abstract ..." onward
        bm = _BACK_MATTER.search(text)
        if bm:
            head = text[: bm.start()].strip()
            if head:
                out.append(Page(page_num=pg.page_num, text=head))
            break
        out.append(Page(page_num=pg.page_num, text=text))
    return out
