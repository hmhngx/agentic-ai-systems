"""
region_chunker.py — Type-aware chunking for each document region.

Why different chunking strategies per region type?

  TITLE/HEADING: do NOT chunk. Store as metadata only, attached to child regions.
    Reason: embedding a single heading "Introduction" produces a very generic vector
    that matches every query loosely. Headings are context, not content.

  TEXT/LIST: chunk with recursive sentence-aware splitter (300 tokens, 10% overlap).
    Reason: prose benefits from overlap to prevent context loss at boundaries.
    Every chunk stores its heading_path as metadata for section-aware retrieval.

  TABLE: do NOT chunk with text splitter. Use table_serializer.
    Reason: splitting a table at an arbitrary token count destroys the row/column
    alignment. TABLE regions get their own serialization and splitting logic.

  FIGURE: store as metadata-only chunk with no text embedding.
    Reason: we cannot embed image content without a Vision LLM call.
    Store bounding box + page + heading_path so it can be found by section context.
    In production, a Vision LLM would caption this and the caption would be embedded.

  CAPTION: attach to the preceding TABLE or FIGURE region as supplementary text.
    Do not embed captions as standalone chunks.

  FALLBACK (region_type="text" from pdfplumber): use same strategy as TEXT.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any

from .table_serializer import extract_table_title, serialize_table


CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE_TOKENS", "300"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "30"))

# Minimum characters for a prose chunk to be worth embedding. Anything
# shorter is almost always a stray fragment (page number, dangling label).
_MIN_PROSE_CHARS: int = 50

# Sentence boundary: a lowercase-ending word + terminal punctuation, followed
# by whitespace and a capitalized next word. Conservative on purpose so we do
# not split on abbreviations like "Inc." mid-sentence.
_SENTENCE_SPLIT: re.Pattern[str] = re.compile(r"(?<=[a-z]{2}[.!?])\s+(?=[A-Z])")


def _approx_tokens(text: str) -> int:
    """Approximate token count as ``len(text) // 4``.

    Rationale: 4 chars/token is the well-known rule of thumb for English
    text with BPE tokenizers. We avoid importing a real tokenizer here to
    keep chunking dependency-free and fast; exact counts are not required
    because CHUNK_SIZE is itself a soft target.
    """
    return max(1, len(text) // 4)


def _chunk_prose(region: dict[str, Any], chunk_type: str = "prose") -> list[dict[str, Any]]:
    """Split prose/list text into overlapping, sentence-aware chunks.

    Greedily packs sentences up to ``CHUNK_SIZE`` tokens, then carries a
    ``CHUNK_OVERLAP``-token tail into the next chunk so context is not lost
    at boundaries. Chunks shorter than ``_MIN_PROSE_CHARS`` are discarded.
    """
    text: str = region["text"]
    sentences: list[str] = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    chunk_texts: list[str] = []
    current: list[str] = []
    current_tokens: int = 0

    for sentence in sentences:
        sentence_tokens: int = _approx_tokens(sentence)
        if current and current_tokens + sentence_tokens > CHUNK_SIZE:
            chunk_texts.append(" ".join(current).strip())
            # Back up from the tail to build the overlap window for the next chunk.
            overlap_sentences: list[str] = []
            overlap_tokens: int = 0
            for prev_sentence in reversed(current):
                if overlap_tokens >= CHUNK_OVERLAP:
                    break
                overlap_sentences.insert(0, prev_sentence)
                overlap_tokens += _approx_tokens(prev_sentence)
            current = overlap_sentences
            current_tokens = sum(_approx_tokens(s) for s in current)
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunk_texts.append(" ".join(current).strip())

    chunks: list[dict[str, Any]] = []
    chunk_index: int = 0
    for chunk_text in chunk_texts:
        if len(chunk_text.strip()) < _MIN_PROSE_CHARS:
            continue
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "chunk_type": chunk_type,
                "region_type": region["region_type"],
                "page_num": region["page_num"],
                "heading_path": list(region.get("heading_path", [])),
                "source_pdf": region["source_pdf"],
                "reading_order": region["reading_order"],
                "bbox": list(region.get("bbox", [0.0, 0.0, 1.0, 1.0])),
                "table_title": None,
                "chunk_index": chunk_index,
                "token_count": _approx_tokens(chunk_text),
            }
        )
        chunk_index += 1
    return chunks


def _chunk_table(region: dict[str, Any], all_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize and chunk a table region using table_serializer.

    The ``[TABLE: {title}]`` prefix injected by the serializer is what makes
    a table chunk retrievable by "find the table about X" queries even when
    the word X only appears in the caption/heading, not in the cells.
    """
    # Regions strictly before this table in reading order, used to source the
    # table title from a preceding caption or section heading.
    prev_regions: list[dict[str, Any]] = [
        r for r in all_regions if r["reading_order"] < region["reading_order"]
    ]
    table_title: str = extract_table_title(region, prev_regions)

    serialized: list[dict[str, Any]] = serialize_table(region, table_title)

    chunks: list[dict[str, Any]] = []
    for chunk_index, piece in enumerate(serialized):
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "text": piece["text"],
                "chunk_type": "table",
                "region_type": region["region_type"],
                "page_num": region["page_num"],
                "heading_path": list(region.get("heading_path", [])),
                "source_pdf": region["source_pdf"],
                "reading_order": region["reading_order"],
                "bbox": list(region.get("bbox", [0.0, 0.0, 1.0, 1.0])),
                # table_title is stored both in the text prefix AND as metadata
                # so it is searchable by content and filterable by payload.
                "table_title": piece.get("table_title", table_title),
                "chunk_index": chunk_index,
                "token_count": _approx_tokens(piece["text"]),
            }
        )
    return chunks


def _chunk_figure(region: dict[str, Any]) -> list[dict[str, Any]]:
    """Create a single metadata-only chunk for a figure region.

    The figure has no extractable text, so we synthesize a locator string
    from its page and heading context. This chunk IS embedded so the figure
    can be retrieved by section context. In production, a Vision LLM would
    caption the figure bbox and that caption would be embedded instead.
    """
    heading_path: list[str] = list(region.get("heading_path", []))
    heading_path_str: str = " > ".join(heading_path) if heading_path else "unknown section"
    text: str = f"[FIGURE on page {region['page_num']}] {heading_path_str}"
    return [
        {
            "chunk_id": str(uuid.uuid4()),
            "text": text,
            "chunk_type": "figure_meta",
            "region_type": region["region_type"],
            "page_num": region["page_num"],
            "heading_path": heading_path,
            "source_pdf": region["source_pdf"],
            "reading_order": region["reading_order"],
            "bbox": list(region.get("bbox", [0.0, 0.0, 1.0, 1.0])),
            "table_title": None,
            "chunk_index": 0,
            "token_count": _approx_tokens(text),
        }
    ]


def chunk_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Route each region to its type-specific chunker and flatten the result.

    TITLE/HEADING and CAPTION regions produce no chunks of their own: headings
    already live in every child chunk's ``heading_path`` (set during parsing),
    and captions are folded into their adjacent table's title.
    """
    chunks: list[dict[str, Any]] = []
    for region in regions:
        region_type: str = region["region_type"]
        if region_type in ("title", "heading", "caption", "header", "footer"):
            continue  # context/boilerplate — never embedded as standalone chunks
        if region_type == "text":
            chunks.extend(_chunk_prose(region, chunk_type="prose"))
        elif region_type == "list":
            chunks.extend(_chunk_prose(region, chunk_type="list"))
        elif region_type == "table":
            chunks.extend(_chunk_table(region, regions))
        elif region_type == "figure":
            chunks.extend(_chunk_figure(region))
        else:
            # Unknown/fallback region types are treated as prose.
            chunks.extend(_chunk_prose(region, chunk_type="prose"))

    n_prose: int = sum(1 for c in chunks if c["chunk_type"] in ("prose", "list"))
    n_table: int = sum(1 for c in chunks if c["chunk_type"] == "table")
    n_figure: int = sum(1 for c in chunks if c["chunk_type"] == "figure_meta")
    print(f"Chunked {len(regions)} regions -> {len(chunks)} chunks")
    print(f"{n_prose} prose, {n_table} table, {n_figure} figure")
    return chunks
