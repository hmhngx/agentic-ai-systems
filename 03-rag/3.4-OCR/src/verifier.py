"""
verifier.py — Retrieval verification for the ingestion pipeline.

Purpose: verify that the pipeline achieved its goals:
  1. Tables stored as structured text (not lost) — verified by type-filtered retrieval
  2. Headings preserved as chunk metadata — verified by payload inspection
  3. "find the table about X" queries work — verified by semantic search on table chunks

By default queries are derived from payloads already in Qdrant so any PDF with
at least one table and one prose chunk can pass the checklist. Set
VERIFICATION_ADAPTIVE=false to use the legacy fixed queries for sample_pdfs/.
"""

from __future__ import annotations

import os
import re
from typing import Any

from qdrant_client import QdrantClient

from .embedder import embed_queries
from .vector_store import count_by_type, sample_payloads_by_type, search, search_by_type


_RULE: str = "─" * 62
_TOP_K: int = 5

# Legacy fixed queries (sample_pdfs/ only) when VERIFICATION_ADAPTIVE=false.
_LEGACY_Q1: str = "revenue earnings financial results"
_LEGACY_Q2: str = "accuracy performance comparison results"
_LEGACY_Q3: str = "find the table about methodology"
_LEGACY_Q4: str = "introduction background motivation"
_LEGACY_Q5: str = "product specifications benchmark comparison"


def _heading_str(heading_path: list[str]) -> str:
    """Render a heading_path list as a readable ' > '-joined breadcrumb."""
    return " > ".join(heading_path) if heading_path else "(none)"


def _adaptive_enabled() -> bool:
    return os.environ.get("VERIFICATION_ADAPTIVE", "true").lower() in ("1", "true", "yes")


def _table_topic(chunk: dict[str, Any]) -> str:
    """Extract a short topic string for 'find the table about X' queries."""
    title: str = str(chunk.get("table_title") or "").strip()
    if not title:
        text: str = str(chunk.get("text", ""))
        if text.startswith("[TABLE:"):
            title = text.split("]", 1)[0].replace("[TABLE:", "").strip()
    if title.lower().startswith("table ") and ":" in title:
        title = title.split(":", 1)[1].strip()
    return title or "document tables"


def _table_distinctive_words(chunk: dict[str, Any]) -> str:
    """Words from table caption/body that rarely appear in reference lists."""
    text: str = str(chunk.get("text", ""))
    stop: set[str] = {"table", "figure", "about", "find", "the", "with", "from", "that", "this"}
    words: list[str] = []
    for line in text.splitlines()[:6]:
        if line.strip().startswith("|") and "---" in line:
            continue
        for word in re.findall(r"[A-Za-z]{4,}", line):
            if word.lower() not in stop:
                words.append(word)
    seen: set[str] = set()
    unique: list[str] = []
    for word in words:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            unique.append(word)
    return " ".join(unique[:6])


def _prose_heading_topic(chunk: dict[str, Any]) -> str:
    """Extract section words from a prose chunk for Q4-style queries."""
    path: list[str] = list(chunk.get("heading_path", []) or [])
    if path:
        return path[-1]
    text: str = str(chunk.get("text", ""))
    words: list[str] = re.findall(r"[A-Za-z]{4,}", text)
    return " ".join(words[:6]) if words else "introduction"


def _prose_in_table_section(
    table_chunk: dict[str, Any],
    prose_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick prose from the same section as the anchor table (shared heading_path)."""
    table_headings: set[str] = set(table_chunk.get("heading_path", []) or [])
    if not table_headings:
        return prose_candidates[0] if prose_candidates else None
    for candidate in prose_candidates:
        if set(candidate.get("heading_path", []) or []) & table_headings:
            return candidate
    return prose_candidates[0] if prose_candidates else None


def _build_verification_queries(
    client: QdrantClient,
) -> tuple[str, str, str, str, str]:
    """Return (q1, q2, q3, q4, q5) texts for the smoke-test suite."""
    if not _adaptive_enabled():
        q3_override = os.environ.get("VERIFICATION_FIND_TABLE_QUERY")
        q3 = q3_override if q3_override else _LEGACY_Q3
        return _LEGACY_Q1, _LEGACY_Q2, q3, _LEGACY_Q4, _LEGACY_Q5

    n_table: int = count_by_type(client, "table")
    n_prose: int = count_by_type(client, "prose")

    table_samples: list[dict[str, Any]] = (
        sample_payloads_by_type(client, "table", limit=1) if n_table else []
    )
    prose_samples: list[dict[str, Any]] = (
        sample_payloads_by_type(client, "prose", limit=8) if n_prose else []
    )

    if table_samples:
        table_chunk: dict[str, Any] = table_samples[0]
        topic: str = _table_topic(table_chunk)
        hints: str = _table_distinctive_words(table_chunk)
        q1 = f"{topic} {hints} table data values".strip()
        q2 = f"{topic} {hints} comparison results metrics".strip()
        q3 = f"find the table about {topic} {hints}".strip()
    else:
        q1, q2, q3 = _LEGACY_Q1, _LEGACY_Q2, _LEGACY_Q3

    if prose_samples:
        if table_samples:
            section_chunk = _prose_in_table_section(table_samples[0], prose_samples)
            section = _prose_heading_topic(section_chunk or prose_samples[0])
        else:
            section = _prose_heading_topic(prose_samples[0])
        q4 = f"{section} introduction background motivation"
    else:
        q4 = _LEGACY_Q4

    if table_samples and prose_samples:
        table_chunk = table_samples[0]
        topic = _table_topic(table_chunk)
        hints = _table_distinctive_words(table_chunk)
        section_chunk = _prose_in_table_section(table_chunk, prose_samples)
        section = _prose_heading_topic(section_chunk or prose_samples[0])
        q5 = f"{topic} {hints} {section} benchmark comparison".strip()
    elif table_samples:
        topic = _table_topic(table_samples[0])
        hints = _table_distinctive_words(table_samples[0])
        q5 = f"{topic} {hints} specifications comparison".strip()
    else:
        q5 = _LEGACY_Q5

    return q1, q2, q3, q4, q5


def _print_results(
    query_text: str,
    filter_desc: str,
    results: list[dict[str, Any]],
    top_k: int,
) -> int:
    """Print one query's results in the standard verification format.

    Returns the number of table chunks present in ``results`` so the caller
    can aggregate per-query table-hit counts for the summary.
    """
    n_table_chunks: int = sum(1 for r in results if r["chunk_type"] == "table")

    print(_RULE)
    print(f'Query: "{query_text}" [filter: {filter_desc}]')
    print(_RULE)
    for result in results:
        text_preview: str = result["text"].replace("\n", " ")[:200]
        print(
            f"#{result['rank']} [{result['chunk_type']}] "
            f"page {result['page_num']} score={result['score']:.4f}"
        )
        print(f"   Heading: {_heading_str(result['heading_path'])}")
        print(f"   Table: {result['table_title'] or 'N/A'}")
        print(f"   Text: {text_preview}...")
    print(f"Result: {n_table_chunks}/{len(results)} table chunks in top-{top_k}")
    print(_RULE)
    return n_table_chunks


def run_verification(client: QdrantClient, pdf_names: list[str]) -> dict[str, Any]:
    """Run the verification query suite and return a summary dict."""
    print()
    print("Running retrieval verification queries...")
    print(f"(corpus PDFs: {', '.join(pdf_names) if pdf_names else 'collection contents'})")
    if _adaptive_enabled():
        print("(adaptive queries derived from collection payloads)")
    print()

    q1_text, q2_text, q3_text, q4_text, q5_text = _build_verification_queries(client)

    queries_with_table_chunks: int = 0
    headings_in_metadata: bool = False

    def _note_headings(results: list[dict[str, Any]]) -> None:
        nonlocal headings_in_metadata
        if any(r["heading_path"] for r in results):
            headings_in_metadata = True

    query_texts: list[str] = [q1_text, q2_text, q3_text, q4_text, q5_text]
    query_vectors = embed_queries(query_texts)

    q1_results: list[dict[str, Any]] = search_by_type(
        client, query_vectors[0], "table", top_k=_TOP_K
    )
    q1_tables: int = _print_results(q1_text, "chunk_type=table", q1_results, _TOP_K)
    _note_headings(q1_results)
    q1_filter_ok: bool = all(r["chunk_type"] == "table" for r in q1_results)
    if q1_tables > 0:
        queries_with_table_chunks += 1

    q2_results: list[dict[str, Any]] = search_by_type(
        client, query_vectors[1], "table", top_k=_TOP_K
    )
    q2_tables: int = _print_results(q2_text, "chunk_type=table", q2_results, _TOP_K)
    _note_headings(q2_results)
    q2_filter_ok: bool = all(r["chunk_type"] == "table" for r in q2_results)
    if q2_tables > 0:
        queries_with_table_chunks += 1

    q3_results: list[dict[str, Any]] = search(client, query_vectors[2], top_k=_TOP_K)
    q3_tables: int = _print_results(q3_text, "none", q3_results, _TOP_K)
    _note_headings(q3_results)
    if q3_tables > 0:
        queries_with_table_chunks += 1

    q4_results: list[dict[str, Any]] = search_by_type(
        client, query_vectors[3], "prose", top_k=_TOP_K
    )
    _print_results(q4_text, "chunk_type=prose", q4_results, _TOP_K)
    _note_headings(q4_results)
    q4_filter_ok: bool = all(r["chunk_type"] == "prose" for r in q4_results)

    q5_results: list[dict[str, Any]] = search(client, query_vectors[4], top_k=_TOP_K)
    q5_tables: int = _print_results(q5_text, "none", q5_results, _TOP_K)
    _note_headings(q5_results)
    if q5_tables > 0:
        queries_with_table_chunks += 1
    q5_has_prose: bool = any(r["chunk_type"] in ("prose", "list") for r in q5_results)
    q5_cross_region_ok: bool = q5_tables > 0 and q5_has_prose

    all_type_filters_correct: bool = q1_filter_ok and q2_filter_ok and q4_filter_ok
    table_retrieval_works: bool = q1_tables > 0 and q2_tables > 0

    summary: dict[str, Any] = {
        "total_queries": 5,
        "queries_with_table_chunks": queries_with_table_chunks,
        "all_type_filters_correct": all_type_filters_correct,
        "headings_in_metadata": headings_in_metadata,
        "table_retrieval_works": table_retrieval_works,
        "find_table_query_works": q3_tables > 0,
        "cross_region_query_works": q5_cross_region_ok,
    }

    print()
    print("Verification summary:")
    print(f"  Queries returning >=1 table chunk: {queries_with_table_chunks}/5")
    print(f"  Type filters returned correct types: {all_type_filters_correct}")
    print(f"  Headings present in chunk metadata: {headings_in_metadata}")
    print(f"  Table retrieval works (Q1 and Q2): {table_retrieval_works}")
    print(f"  Cross-region query mixes table+prose (Q5): {q5_cross_region_ok}")
    print()

    return summary
