"""
table_serializer.py — Convert Docling table structures to embedded-ready text.

Why tables cannot be embedded as flat strings:
  A table like:
    | Product | Q1 | Q2 | Q3 |
    | Widget  | 10 | 20 | 30 |
    | Gadget  | 15 | 25 | 35 |

  If flattened: "Product Q1 Q2 Q3 Widget 10 20 30 Gadget 15 25 35"
  A query for "Widget Q3 revenue" cannot match "30" to "Widget" and "Q3"
  because the relational structure is destroyed.

  Serialized as Markdown: the LLM can read the grid and answer correctly.
  The embedding also captures the tabular structure better because
  header terms appear close to data values in the token sequence.

Why split large tables?
  A 200-row table embedded as one chunk exceeds context limits and
  creates a vector that averages across all rows — reducing precision.
  Splitting into groups of TABLE_MAX_ROWS_PER_CHUNK rows with the
  header row repeated on each chunk preserves per-row retrievability.

Serialization format (Markdown table):
  - First chunk always includes the column header row
  - Each subsequent chunk repeats the header row (for standalone context)
  - Chunk text: "[TABLE: {table_title}]\n| col1 | col2 | ... |\n| --- |...\n| val | val |..."
  - Prefix "[TABLE: {title}]" makes the chunk retrievable by "find table about X" queries
"""

from __future__ import annotations

import os
from typing import Any, Optional


TABLE_MAX_ROWS: int = int(os.environ.get("TABLE_MAX_ROWS_PER_CHUNK", "20"))


def _escape_cell(value: Any) -> str:
    """Render a single table cell as Markdown-safe inline text.

    Pipes would terminate a Markdown column early and newlines would break
    the single-line-per-row grid, so both are neutralized. Empty/NaN-like
    values become an empty string rather than the literal "nan".
    """
    if value is None:
        return ""
    text: str = str(value)
    if text.lower() == "nan":
        return ""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _build_markdown(header: list[str], rows: list[list[Any]]) -> str:
    """Assemble a GitHub-flavored Markdown table from header + data rows.

    Used when splitting large tables so each sub-chunk can carry the header
    row verbatim. The separator row (``| --- |``) is what tells Markdown
    renderers (and the embedding model's learned table priors) that this is
    a grid rather than free text.
    """
    header_cells: list[str] = [_escape_cell(col) for col in header]
    header_line: str = "| " + " | ".join(header_cells) + " |"
    separator_line: str = "| " + " | ".join("---" for _ in header_cells) + " |"

    body_lines: list[str] = []
    for row in rows:
        cells: list[str] = [_escape_cell(cell) for cell in row]
        # Pad/truncate so ragged rows still align to the header width.
        if len(cells) < len(header_cells):
            cells = cells + [""] * (len(header_cells) - len(cells))
        elif len(cells) > len(header_cells):
            cells = cells[: len(header_cells)]
        body_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_line, separator_line, *body_lines])


def _export_dataframe(table_data: dict[str, Any]) -> Optional[Any]:
    """Export the Docling table node to a pandas DataFrame, or None.

    ``export_to_dataframe`` now expects the owning ``DoclingDocument`` but
    accepted no argument in older docling-core; we try both signatures.
    Returns None if there is no live Docling node (e.g. fallback path) or
    the export raises — callers degrade gracefully to Markdown-only output.
    """
    docling_item: Any = table_data.get("docling_item")
    doc: Any = table_data.get("doc")
    if docling_item is None:
        return None
    try:
        return docling_item.export_to_dataframe(doc=doc)
    except TypeError:
        try:
            return docling_item.export_to_dataframe()
        except Exception as exc:  # noqa: BLE001 - degrade to Markdown-only
            print(f"WARNING: table export_to_dataframe failed: {exc}")
            return None
    except Exception as exc:  # noqa: BLE001 - degrade to Markdown-only
        print(f"WARNING: table export_to_dataframe failed: {exc}")
        return None


def serialize_table(table_region: dict[str, Any], table_title: str = "") -> list[dict[str, Any]]:
    """Convert a table region into one or more serialized chunk dicts.

    Each returned dict has: ``text`` (Markdown with ``[TABLE: ...]`` prefix),
    ``chunk_type`` ("table"), ``table_title``, and ``row_range`` (start, end).
    Returns an empty list (with a warning) if the table cannot be serialized,
    rather than crashing — some Docling detections yield empty bounding boxes.
    """
    table_data: Optional[dict[str, Any]] = table_region.get("table_data")
    if not table_data:
        print("WARNING: table region has no table_data; skipping serialization.")
        return []

    markdown_full: str = str(table_data.get("markdown", "") or "")
    dataframe: Optional[Any] = _export_dataframe(table_data)

    # Path A: no cell-level frame available. Emit the Markdown we already have
    # as a single chunk if it is non-empty; otherwise we cannot serialize.
    if dataframe is None or getattr(dataframe, "empty", True):
        if not markdown_full.strip():
            print("WARNING: table has no extractable content; skipping.")
            return []
        return [
            {
                "text": f"[TABLE: {table_title}]\n{markdown_full}",
                "chunk_type": "table",
                "table_title": table_title,
                "row_range": (0, markdown_full.count("\n")),
            }
        ]

    header: list[str] = [str(col) for col in dataframe.columns.tolist()]
    rows: list[list[Any]] = dataframe.values.tolist()
    n_rows: int = len(rows)

    # Path B: small table — one self-contained chunk using the full Markdown.
    if n_rows <= TABLE_MAX_ROWS:
        return [
            {
                "text": f"[TABLE: {table_title}]\n{markdown_full or _build_markdown(header, rows)}",
                "chunk_type": "table",
                "table_title": table_title,
                "row_range": (0, n_rows),
            }
        ]

    # Path C: large table — split into row-groups, repeating the header on
    # every chunk so each sub-chunk is independently answerable.
    chunks: list[dict[str, Any]] = []
    for start in range(0, n_rows, TABLE_MAX_ROWS):
        end: int = min(start + TABLE_MAX_ROWS, n_rows)
        group_markdown: str = _build_markdown(header, rows[start:end])
        chunks.append(
            {
                # 1-indexed, inclusive row labels are friendlier for humans
                # reading the chunk text; row_range stays 0-indexed exclusive.
                "text": f"[TABLE: {table_title} (rows {start + 1}-{end})]\n{group_markdown}",
                "chunk_type": "table",
                "table_title": table_title,
                "row_range": (start, end),
            }
        )
    return chunks


def extract_table_title(region: dict[str, Any], prev_regions: list[dict[str, Any]]) -> str:
    """Determine a meaningful title for a table.

    Why table titles matter for retrieval:
      A query "find the accuracy comparison table" needs the word "accuracy"
      to appear in the chunk. If the table only contains numbers, no row/column
      header contains "accuracy", and the query fails. The title derived from
      the caption or preceding heading provides this signal.

    Strategy (in priority order):
      1. Immediately preceding region is a CAPTION → use its text.
      2. Closest preceding non-empty region is a HEADING → use its text.
      3. ``heading_path[-1]`` (nearest ancestor section title).
      4. Default: "Table (page {page_num})".
    """
    if prev_regions:
        immediate: dict[str, Any] = prev_regions[-1]
        if immediate.get("region_type") == "caption" and immediate.get("text", "").strip():
            return immediate["text"].strip()

        for candidate in reversed(prev_regions):
            if not candidate.get("text", "").strip():
                continue
            if candidate.get("region_type") == "heading":
                return candidate["text"].strip()
            break  # closest non-empty region is not a heading → stop looking

    heading_path: list[str] = region.get("heading_path", []) or []
    if heading_path:
        return heading_path[-1]

    return f"Table (page {region.get('page_num', 1)})"
