"""Offline tests for ``src/pdf_loader.py``.

Every test in this file runs without Qdrant or any API key. The
``sample_pdf`` fixture generates a real multi-page PDF in ``tmp_path``
via reportlab so we exercise the production extract + chunk path end
to end, not a stubbed version of it.

RAG correctness rationale: if pdf_loader returns malformed chunks,
every downstream component (embeddings, retrieval, citations, the
LLM prompt) is corrupted in ways that are very hard to diagnose
after the fact. The defects must be caught here at the source.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

import pytest

from src.pdf_loader import chunk_pages, extract_pages, load_and_chunk


def test_extract_pages_returns_correct_structure(sample_pdf: pathlib.Path) -> None:
    """Extracted pages must expose page_num/text/char_count for citation lookup."""
    result = extract_pages(str(sample_pdf))

    assert isinstance(result, list), \
        f"extract_pages must return a list, got {type(result).__name__} - " \
        "callers iterate the result so a non-list return shape will crash ingest."
    assert len(result) >= 3, \
        f"Expected >=3 pages from the test PDF, got {len(result)} - " \
        "the chunker needs page boundaries to attach citations."

    required_keys = {"page_num", "text", "char_count"}
    for page in result:
        assert set(page.keys()) >= required_keys, \
            f"Page dict missing required keys: " \
            f"{required_keys - set(page.keys())}. " \
            "Every downstream chunk needs these fields for citations."
        assert page["page_num"] >= 1, \
            f"page_num={page['page_num']} must be 1-indexed for human-readable " \
            f"citations - users count pages starting at 1."
        assert page["char_count"] == len(page["text"]), \
            f"char_count={page['char_count']} != len(text)={len(page['text'])} - " \
            "drift here breaks chunk-size analytics and stats reporting."


def test_extract_pages_page_nums_are_sequential(sample_pdf: pathlib.Path) -> None:
    """Page numbers must be 1..N with no gaps so citations stay verifiable."""
    result = extract_pages(str(sample_pdf))
    expected = list(range(1, len(result) + 1))
    actual = [p["page_num"] for p in result]
    assert actual == expected, \
        f"page_num sequence broken: expected {expected}, got {actual}. " \
        "Non-sequential page numbers mean a [Doc N] citation pointing to a " \
        "page the user cannot find in the source PDF."


def test_extract_pages_text_is_nonempty(sample_pdf: pathlib.Path) -> None:
    """Blank pages must be filtered - they produce useless empty chunks."""
    result = extract_pages(str(sample_pdf))
    for page in result:
        assert len(page["text"].strip()) > 0, \
            f"Page {page['page_num']} text is blank after extraction. " \
            "Blank pages must be filtered upstream or they become " \
            "zero-information chunks that pollute retrieval scores."


def test_extract_pages_missing_file_raises(tmp_path: pathlib.Path) -> None:
    """Missing PDF path must raise FileNotFoundError, not a deep traceback."""
    bogus = str(tmp_path / "nonexistent" / "file.pdf")
    with pytest.raises(FileNotFoundError) as excinfo:
        extract_pages(bogus)
    assert bogus in str(excinfo.value), \
        f"Exception message missing path '{bogus}'. " \
        "Users must see which file the loader could not find - a bare " \
        "FileNotFoundError without the path is unactionable."


def test_extract_pages_text_not_excessive_whitespace(sample_pdf: pathlib.Path) -> None:
    """Whitespace runs must be collapsed so embeddings don't tokenize blanks."""
    result = extract_pages(str(sample_pdf))
    for page in result:
        assert "\n\n\n" not in page["text"], \
            f"Page {page['page_num']} contains triple-newline whitespace " \
            "run. Excessive whitespace wastes embedding tokens and can " \
            "shift sentence boundaries during chunking."


def test_chunk_pages_returns_correct_structure(sample_pdf: pathlib.Path) -> None:
    """Chunks must carry the full payload schema upsert_chunks expects."""
    chunks = chunk_pages(extract_pages(str(sample_pdf)))

    assert isinstance(chunks, list), \
        f"chunk_pages must return a list, got {type(chunks).__name__}."
    assert len(chunks) >= 3, \
        f"Expected >=3 chunks from a 3-page PDF, got {len(chunks)}. " \
        "Too few chunks means retrieval has too little to choose from."

    required_keys = {
        "chunk_id", "text", "page_num", "chunk_index",
        "token_count", "char_count", "source_pdf",
    }
    for chunk in chunks:
        missing = required_keys - set(chunk.keys())
        assert not missing, \
            f"Chunk missing required keys {missing}. " \
            "Each missing key breaks a downstream invariant: chunk_id -> " \
            "Qdrant point id, page_num -> citation, source_pdf -> footer."


def test_chunk_pages_chunk_ids_are_unique(sample_pdf: pathlib.Path) -> None:
    """Duplicate chunk_ids would collide on the Qdrant point-id namespace."""
    chunks = chunk_pages(extract_pages(str(sample_pdf)))
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), \
        f"Duplicate chunk_ids found ({len(ids) - len(set(ids))} dupes). " \
        "vector_store._uuid_to_point_id derives point ids from these - " \
        "collisions silently overwrite previously-upserted vectors."


def test_chunk_pages_page_nums_valid(sample_pdf: pathlib.Path) -> None:
    """Every chunk must point at a page that actually exists in the PDF."""
    pages = extract_pages(str(sample_pdf))
    page_nums_in_pages = {p["page_num"] for p in pages}
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert chunk["page_num"] in page_nums_in_pages, \
            f"chunk page_num {chunk['page_num']} not in source pages " \
            f"{sorted(page_nums_in_pages)}. A citation pointing to a " \
            "non-existent page is worse than no citation at all - users " \
            "cannot verify and lose trust in the system."


def test_chunk_pages_no_empty_chunks(sample_pdf: pathlib.Path) -> None:
    """Chunks <50 chars must be filtered - signal-to-noise too low to embed."""
    chunks = chunk_pages(extract_pages(str(sample_pdf)))
    for chunk in chunks:
        text = chunk["text"]
        assert len(text.strip()) >= 50, \
            f"Chunk too short ({len(text)} chars): {text[:30]!r}. " \
            "Sub-50-char chunks produce unreliable voyage-3 embeddings " \
            "(mostly stop words / page artifacts) and pollute retrieval."


def test_chunk_pages_token_counts_reasonable(sample_pdf: pathlib.Path) -> None:
    """Token counts must be in 10..400 range to avoid pathological chunks."""
    chunks = chunk_pages(extract_pages(str(sample_pdf)))
    for chunk in chunks:
        n = chunk["token_count"]
        assert 10 <= n <= 400, \
            f"Token count {n} out of [10, 400] range for chunk " \
            f"{chunk['text'][:30]!r}. " \
            "Tiny chunks under-embed; >400-token chunks dilute attention."


def test_chunk_pages_source_pdf_is_basename(sample_pdf: pathlib.Path) -> None:
    """source_pdf must be the basename so citations don't leak local paths."""
    chunks, _ = load_and_chunk(str(sample_pdf))
    for chunk in chunks:
        src = chunk["source_pdf"]
        assert "/" not in src, \
            f"source_pdf '{src}' contains '/' - must be basename. " \
            "Full paths leak the dev's filesystem into shipped citations."
        assert "\\" not in src, \
            f"source_pdf '{src}' contains '\\' - must be basename. " \
            "Windows-style paths must be stripped before display."


def test_chunk_pages_chunk_index_per_page(sample_pdf: pathlib.Path) -> None:
    """chunk_index must be 0..N-1 contiguous within each page for debug logs."""
    chunks = chunk_pages(extract_pages(str(sample_pdf)))
    by_page: dict[int, list[int]] = defaultdict(list)
    for chunk in chunks:
        by_page[chunk["page_num"]].append(chunk["chunk_index"])
    for page_num, indices in by_page.items():
        expected = list(range(len(indices)))
        assert indices == expected, \
            f"Page {page_num}: chunk_index not sequential: " \
            f"got {indices}, expected {expected}. " \
            "Non-sequential indices break debug output that maps " \
            "[Doc N] back to a specific position on the page."


def test_load_and_chunk_returns_tuple(sample_pdf: pathlib.Path) -> None:
    """load_and_chunk must return (chunks, stats) with health metrics."""
    result = load_and_chunk(str(sample_pdf))
    assert isinstance(result, tuple), \
        f"load_and_chunk must return a tuple, got {type(result).__name__}. " \
        "The CLI unpacks it as (chunks, stats) - any other shape crashes ingest."
    assert len(result) == 2, \
        f"Tuple must have 2 elements, got {len(result)}."

    chunks, stats = result
    required_stat_keys = {
        "pdf_path", "total_pages", "pages_with_text", "total_chunks",
        "avg_tokens_per_chunk", "min_tokens", "max_tokens",
    }
    assert set(stats.keys()) == required_stat_keys, \
        f"stats key drift - missing: {required_stat_keys - set(stats.keys())}; " \
        f"unexpected: {set(stats.keys()) - required_stat_keys}. " \
        "These metrics are how the operator decides if ingestion looks healthy."

    assert stats["total_chunks"] == len(chunks), \
        f"stats['total_chunks']={stats['total_chunks']} != len(chunks)={len(chunks)}. " \
        "Stats must reflect what was actually returned or operators distrust the dashboard."
    assert stats["total_pages"] >= 3, \
        f"Expected >=3 pages in stats, got {stats['total_pages']}."
    assert stats["min_tokens"] <= stats["avg_tokens_per_chunk"] <= stats["max_tokens"], \
        f"min={stats['min_tokens']}, avg={stats['avg_tokens_per_chunk']}, " \
        f"max={stats['max_tokens']} - violates min <= avg <= max ordering."


def test_chunk_pages_coverage(sample_pdf: pathlib.Path) -> None:
    """Chunks must cover >=80% of source text - lower means silent data loss."""
    pages = extract_pages(str(sample_pdf))
    chunks = chunk_pages(pages)
    all_chunk_text = " ".join(c["text"] for c in chunks)
    all_page_text = " ".join(p["text"] for p in pages)
    coverage = len(all_chunk_text) / len(all_page_text)
    assert coverage >= 0.80, \
        f"Coverage {coverage:.2%} below 80% - chunking is dropping source " \
        "text. The user could ask about a fact in the dropped span and " \
        "the retriever would always say NO_RESULTS without any hint why."
