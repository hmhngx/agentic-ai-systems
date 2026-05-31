"""Offline tests for region chunking — structure metadata drives section-aware retrieval."""

from __future__ import annotations

from typing import Any

from src.region_chunker import CHUNK_OVERLAP, CHUNK_SIZE


def test_chunk_regions_returns_list(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies chunk_regions produces embeddable units from typed regions."""
    assert isinstance(sample_chunks, list)
    assert len(sample_chunks) > 0, "chunk_regions must return at least one chunk"


def test_chunk_regions_all_have_required_keys(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies every chunk carries Qdrant payload fields for filtered retrieval."""
    required = {
        "chunk_id",
        "text",
        "chunk_type",
        "region_type",
        "page_num",
        "heading_path",
        "source_pdf",
        "reading_order",
        "bbox",
        "table_title",
        "chunk_index",
        "token_count",
    }
    for chunk in sample_chunks:
        assert required.issubset(set(chunk.keys())), (
            f"Chunk missing keys: {required - set(chunk.keys())} — "
            "incomplete payloads break section and type filters"
        )


def test_chunk_regions_chunk_types_are_valid(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies chunk_type values match vector_store filter vocabulary."""
    valid_types = {"prose", "table", "figure_meta", "list"}
    for chunk in sample_chunks:
        assert chunk["chunk_type"] in valid_types, (
            f"Invalid chunk_type '{chunk['chunk_type']}' — must be one of {valid_types}"
        )


def test_chunk_regions_heading_path_on_every_chunk(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies heading_path on every chunk enables section-scoped retrieval."""
    for chunk in sample_chunks:
        assert "heading_path" in chunk, (
            f"heading_path missing from chunk {chunk.get('chunk_id', 'unknown')}"
        )
        assert isinstance(chunk["heading_path"], list), (
            f"heading_path must be a list, got {type(chunk['heading_path'])}"
        )


def test_chunk_regions_prose_chunks_inherit_heading_path(
    sample_chunks: list[dict[str, Any]],
) -> None:
    """Verifies prose chunks inherit ancestor headings for section-aware search."""
    prose_chunks = [c for c in sample_chunks if c["chunk_type"] == "prose"]
    assert len(prose_chunks) > 0, "Must produce at least one prose chunk"
    for chunk in prose_chunks:
        assert isinstance(chunk["heading_path"], list)
        if chunk["page_num"] == 2:
            assert len(chunk["heading_path"]) > 0, (
                f"Prose chunk on page 2 must have non-empty heading_path, "
                f"got {chunk['heading_path']}"
            )


def test_chunk_regions_table_chunks_have_table_prefix(
    sample_chunks: list[dict[str, Any]],
) -> None:
    """Verifies table chunks retain [TABLE:] prefix for table-specific queries."""
    table_chunks = [c for c in sample_chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) > 0, (
        "Must produce at least one table chunk — tables must not be discarded"
    )
    for chunk in table_chunks:
        assert chunk["text"].startswith("[TABLE:"), (
            f"Table chunk text must start with '[TABLE:' prefix. "
            f"Got: {chunk['text'][:50]}"
        )


def test_chunk_regions_table_chunks_have_table_title(
    sample_chunks: list[dict[str, Any]],
) -> None:
    """Verifies table_title metadata supports caption-driven table lookup."""
    table_chunks = [c for c in sample_chunks if c["chunk_type"] == "table"]
    for chunk in table_chunks:
        assert chunk.get("table_title") is not None, (
            "table_title must be set on table chunks for metadata-based retrieval"
        )
        assert isinstance(chunk["table_title"], str)


def test_chunk_regions_header_footer_discarded(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies headers/footers are excluded — repeated boilerplate degrades precision."""
    all_text = " ".join(c["text"] for c in sample_chunks)
    assert "Confidential — Do Not Distribute" not in all_text, (
        "Footer text must be DISCARDED — page footers pollute the vector index "
        "with repeated content"
    )
    assert "Machine Learning Report — Page 2" not in all_text, (
        "Header text must be DISCARDED — repeating headers cause precision degradation"
    )


def test_chunk_regions_titles_headings_not_embedded_standalone(
    sample_chunks: list[dict[str, Any]],
) -> None:
    """Verifies headings stay as metadata only — standalone headings match every query."""
    standalone_heading_texts = {
        "Machine Learning Report",
        "1. Introduction",
        "2. Results",
        "3. Conclusion",
    }
    for chunk in sample_chunks:
        assert chunk["text"].strip() not in standalone_heading_texts, (
            f"Heading '{chunk['text'].strip()}' must not be embedded as a standalone chunk. "
            "Headings are context metadata, not content — standalone heading chunks match "
            "every query loosely and degrade precision."
        )


def test_chunk_regions_figure_chunks_are_meta_type(
    sample_chunks: list[dict[str, Any]],
) -> None:
    """Verifies figure chunks use [FIGURE] locator text for section-context retrieval."""
    figure_chunks = [c for c in sample_chunks if c["chunk_type"] == "figure_meta"]
    for chunk in figure_chunks:
        assert "[FIGURE" in chunk["text"], (
            f"Figure meta chunk must contain [FIGURE marker, got: {chunk['text'][:50]}"
        )


def test_chunk_regions_chunk_ids_are_unique(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies unique chunk_ids prevent Qdrant point collisions and dedup errors."""
    ids = [c["chunk_id"] for c in sample_chunks]
    assert len(ids) == len(set(ids)), (
        f"Duplicate chunk_ids found — {len(ids) - len(set(ids))} duplicates"
    )


def test_chunk_regions_token_counts_positive(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies positive token_count flags abnormally sized chunks in the payload."""
    for chunk in sample_chunks:
        assert chunk["token_count"] > 0, (
            f"token_count must be positive, got {chunk['token_count']} "
            f"for chunk {chunk['chunk_id']}"
        )


def test_chunk_regions_no_empty_text_chunks(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies short fragments are filtered — tiny chunks add noise without retrieval value."""
    for chunk in sample_chunks:
        assert len(chunk["text"].strip()) >= 50, (
            f"Empty/short chunk produced: '{chunk['text'][:30]}...' — "
            "filter must remove these"
        )


def test_chunk_regions_prose_text_is_coherent(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies prose chunks contain meaningful sentences for semantic embedding."""
    prose_chunks = [c for c in sample_chunks if c["chunk_type"] == "prose"]
    for chunk in prose_chunks:
        words = chunk["text"].split()
        assert len(words) >= 5, (
            f"Prose chunk has only {len(words)} words — too short to be meaningful"
        )


def test_chunk_regions_reading_order_preserved(sample_chunks: list[dict[str, Any]]) -> None:
    """Verifies reading_order metadata enables document-order result sorting."""
    table_chunks = [c for c in sample_chunks if c["chunk_type"] == "table"]
    intro_chunks = [
        c for c in sample_chunks if c["chunk_type"] == "prose" and c["page_num"] == 1
    ]
    if table_chunks and intro_chunks:
        assert table_chunks[0]["reading_order"] > intro_chunks[0]["reading_order"], (
            "Table (page 2) must have higher reading_order than intro prose (page 1)"
        )


def test_chunk_constants() -> None:
    """Verifies chunk size constants match RAG overlap strategy for context preservation."""
    assert CHUNK_SIZE == 300, f"CHUNK_SIZE must be 300, got {CHUNK_SIZE}"
    assert CHUNK_OVERLAP == 30, f"CHUNK_OVERLAP must be 30, got {CHUNK_OVERLAP}"
    assert CHUNK_OVERLAP < CHUNK_SIZE, "Overlap must be less than chunk size"
