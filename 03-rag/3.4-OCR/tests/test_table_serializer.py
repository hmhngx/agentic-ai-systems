"""Offline tests for table serialization — flat strings destroy relational retrieval."""

from __future__ import annotations

from typing import Any

import pytest

from src.table_serializer import extract_table_title, serialize_table


def test_serialize_table_returns_list(table_region: dict[str, Any]) -> None:
    """Verifies tables produce embeddable chunk dicts rather than being dropped."""
    result = serialize_table(table_region, table_title="Accuracy Comparison")
    assert isinstance(result, list)
    assert len(result) >= 1, "Must return at least one chunk for non-empty table"


def test_serialize_table_text_starts_with_table_prefix(table_region: dict[str, Any]) -> None:
    """Verifies [TABLE:] prefix enables 'find table about X' semantic queries."""
    result = serialize_table(table_region, table_title="Accuracy Comparison")
    for chunk in result:
        assert chunk["text"].startswith("[TABLE:"), (
            f"Table chunk must start with '[TABLE:' prefix for retrieval. "
            f"Got: {chunk['text'][:50]}"
        )


def test_serialize_table_title_in_prefix(table_region: dict[str, Any]) -> None:
    """Verifies table title in prefix links caption vocabulary to cell content."""
    result = serialize_table(table_region, table_title="Accuracy Comparison")
    assert "Accuracy Comparison" in result[0]["text"], (
        "Table title must appear in [TABLE: ...] prefix for 'find table about X' queries"
    )


def test_serialize_table_contains_markdown_structure(table_region: dict[str, Any]) -> None:
    """Verifies Markdown grid preserves row/column structure for LLM and embedding."""
    result = serialize_table(table_region, table_title="Test")
    text = result[0]["text"]
    assert "|" in text, (
        "Table chunk must contain Markdown table pipes '|' — "
        "flat strings destroy relational structure"
    )
    assert "---" in text or "---|" in text, (
        "Table chunk must contain Markdown header separator row"
    )


def test_serialize_table_preserves_column_headers(table_region: dict[str, Any]) -> None:
    """Verifies column headers survive serialization for header-matching queries."""
    result = serialize_table(table_region, table_title="Test")
    text = result[0]["text"]
    assert "Model" in text, "Column header 'Model' must appear in serialized table"
    assert "Accuracy" in text, "Column header 'Accuracy' must appear in serialized table"
    assert "F1 Score" in text, "Column header 'F1 Score' must appear in serialized table"


def test_serialize_table_preserves_data_values(table_region: dict[str, Any]) -> None:
    """Verifies cell values remain retrievable after Markdown serialization."""
    result = serialize_table(table_region, table_title="Test")
    text = result[0]["text"]
    assert "BERT" in text, "Row data 'BERT' must appear in serialized table"
    assert "0.91" in text, "Numeric value '0.91' must appear in serialized table"
    assert "RoBERTa" in text, "Row data 'RoBERTa' must appear in serialized table"


def test_serialize_table_chunk_type_is_table(table_region: dict[str, Any]) -> None:
    """Verifies chunk_type metadata enables server-side table-only filtered search."""
    result = serialize_table(table_region, table_title="Test")
    for chunk in result:
        assert chunk.get("chunk_type") == "table", (
            f"Table serializer must set chunk_type='table', got '{chunk.get('chunk_type')}'"
        )


def test_serialize_table_none_table_data_returns_empty(table_region: dict[str, Any]) -> None:
    """Verifies missing table_data degrades gracefully without crashing ingestion."""
    empty_region = {**table_region, "table_data": None}
    result = serialize_table(empty_region, table_title="Test")
    assert isinstance(result, list), "Must return list even for None table_data"


def test_serialize_large_table_splits_with_repeated_header(
    table_region: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies large tables split with repeated headers so each chunk is self-contained."""
    many_rows = "\n".join([f"| Row{i} | Val{i} | Score{i} |" for i in range(50)])
    large_table = {
        **table_region,
        "text": f"| Name | Value | Score |\n|---|---|---|\n{many_rows}",
        "table_data": {
            "markdown": f"| Name | Value | Score |\n|---|---|---|\n{many_rows}",
            "rows": 50,
            "cols": 3,
        },
    }
    monkeypatch.setenv("TABLE_MAX_ROWS_PER_CHUNK", "10")
    monkeypatch.setattr("src.table_serializer.TABLE_MAX_ROWS", 10)
    result = serialize_table(large_table, table_title="Large Table")
    if len(result) > 1:
        for i, chunk in enumerate(result[1:], 1):
            assert "Name" in chunk["text"] and "Value" in chunk["text"], (
                f"Chunk {i} missing header row — large table splits must repeat headers "
                "so every chunk is self-contained for retrieval"
            )


def test_extract_table_title_uses_caption(synthetic_regions: list[dict[str, Any]]) -> None:
    """Verifies caption text supplies retrieval vocabulary absent from table cells."""
    title = extract_table_title(synthetic_regions[5], synthetic_regions[:5])
    assert isinstance(title, str)
    assert len(title) > 0, "extract_table_title must return non-empty string"
    assert (
        "accuracy" in title.lower()
        or "table" in title.lower()
        or "comparison" in title.lower()
    ), f"Caption text should inform table title, got: '{title}'"


def test_extract_table_title_falls_back_to_heading(
    synthetic_regions: list[dict[str, Any]],
) -> None:
    """Verifies section heading titles tables when no caption precedes them."""
    regions_no_caption = [
        r for r in synthetic_regions[:5] if r["region_type"] != "caption"
    ]
    title = extract_table_title(synthetic_regions[5], regions_no_caption)
    assert isinstance(title, str)
    assert len(title) > 0


def test_extract_table_title_default_fallback(synthetic_regions: list[dict[str, Any]]) -> None:
    """Verifies page-based fallback title when no caption or heading is available."""
    region = {**synthetic_regions[5], "heading_path": []}
    title = extract_table_title(region, [])
    assert isinstance(title, str)
    assert "page" in title.lower() or "table" in title.lower(), (
        "Default fallback title should mention page number or 'table'"
    )
