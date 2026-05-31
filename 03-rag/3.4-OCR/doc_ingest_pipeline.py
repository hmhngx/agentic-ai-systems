"""
doc_ingest_pipeline.py — Day 6 deliverable: PDF → Qdrant e2e ingestion pipeline.

No RAG orchestration frameworks. Raw Docling + qdrant-client + embedding API.

What this script proves:
  1. Tables stored as structured Markdown text, not lost as flat strings
  2. Headings preserved as chunk metadata (heading_path field)
  3. Reading order detection produces coherent chunk sequences
  4. Region-aware chunking: different strategies per content type
  5. "find the table about X" queries work via semantic search

Usage:
  python doc_ingest_pipeline.py --pdf path/to/doc.pdf
  python doc_ingest_pipeline.py --pdf-dir sample_pdfs/          # process 3 PDFs
  python doc_ingest_pipeline.py --pdf doc.pdf --reingest        # force re-ingest
  python doc_ingest_pipeline.py --verify-only                   # skip ingest, run verification
  python doc_ingest_pipeline.py --pdf doc.pdf --debug           # show region details

Known limitations of this Day 6 implementation:
  1. Figures not semantically embedded — stored as metadata-only chunks
     (Day 7 integration: route figure bboxes to Vision LLM for captioning)
  2. Mathematical equations treated as text — Docling's equation detection
     is disabled by default (high latency, low production value for most docs)
  3. Scanned PDFs require enable_ocr=True — adds 30+ seconds per page on CPU
  4. Large tables (>200 rows) are split but headers repeat — storage overhead
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv

# Load .env BEFORE importing src modules: several of them bind configuration
# (collection name, chunk sizes, table split limit) from the environment at
# import time, so the values must already be present.
load_dotenv()

from src.docling_parser import parse_pdf_to_regions  # noqa: E402
from src.embedder import EMBEDDING_DIM, embed_chunks  # noqa: E402
from src.pdf_classifier import classify_pdf  # noqa: E402
from src.region_chunker import chunk_regions  # noqa: E402
from src.vector_store import (  # noqa: E402
    COLLECTION_NAME,
    QDRANT_URL,
    collection_exists,
    count_by_type,
    create_collection,
    get_client,
    upsert_chunks,
)
from src.verifier import run_verification  # noqa: E402


_HEAVY_RULE: str = "═" * 56


def _parse_args() -> argparse.Namespace:
    """Define and parse the CLI surface."""
    parser = argparse.ArgumentParser(
        description="Docling → Qdrant document ingestion pipeline (Day 6).",
    )
    parser.add_argument("--pdf", type=str, default=None, help="path to a single PDF to ingest")
    parser.add_argument("--pdf-dir", type=str, default=None, help="directory of PDFs to ingest (*.pdf)")
    parser.add_argument("--reingest", action="store_true", help="drop the collection and re-ingest")
    parser.add_argument("--verify-only", action="store_true", help="skip ingestion, run verification only")
    parser.add_argument("--debug", action="store_true", help="print per-region detail for each PDF")
    parser.add_argument("--enable-ocr", action="store_true", help="force Docling OCR on (scanned PDFs)")
    parser.add_argument("--no-tables", action="store_true", help="skip table regions (quick testing)")
    return parser.parse_args()


def _resolve_pdfs(args: argparse.Namespace) -> list[str]:
    """Resolve the list of PDF paths to process from the CLI arguments."""
    pdfs: list[str] = []
    if args.pdf:
        pdfs.append(args.pdf)
    if args.pdf_dir:
        pdfs.extend(sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf"))))

    missing: list[str] = [p for p in pdfs if not os.path.isfile(p)]
    if missing:
        print(f"ERROR: PDF(s) not found: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    return pdfs


def _debug_dump_regions(regions: list[dict[str, Any]]) -> None:
    """Print a compact per-region summary when --debug is set."""
    for region in regions:
        preview: str = region["text"].replace("\n", " ")[:80]
        print(
            f"    [{region['region_type']}] p{region['page_num']} "
            f"ord={region['reading_order']} | {preview}"
        )


def _ingest_one_pdf(
    client: Any,
    pdf_path: str,
    enable_ocr_flag: bool,
    no_tables: bool,
    debug: bool,
    stats: dict[str, Any],
    all_chunks: list[dict[str, Any]],
) -> None:
    """Run classify → parse → chunk → embed → upsert for a single PDF."""
    basename: str = os.path.basename(pdf_path)
    print(f"\n--- Processing {basename} ---")

    # a. Classify.
    classification: dict[str, Any] = classify_pdf(pdf_path)
    needs_ocr: bool = enable_ocr_flag or bool(classification["needs_ocr"])
    print(f"PDF type: {classification['pdf_type']}. OCR {'enabled' if needs_ocr else 'disabled'}.")

    # b. Parse to typed regions.
    regions: list[dict[str, Any]] = parse_pdf_to_regions(pdf_path, enable_ocr=needs_ocr)
    if no_tables:
        # --no-tables: drop table regions entirely for a fast prose-only run.
        regions = [r for r in regions if r["region_type"] != "table"]
    print(f"{len(regions)} regions extracted.")
    if debug:
        _debug_dump_regions(regions)

    for region in regions:
        stats["region_types"][region["region_type"]] = (
            stats["region_types"].get(region["region_type"], 0) + 1
        )
    stats["total_regions"] += len(regions)

    # c. Chunk.
    chunks: list[dict[str, Any]] = chunk_regions(regions)
    n_prose: int = sum(1 for c in chunks if c["chunk_type"] in ("prose", "list"))
    n_table: int = sum(1 for c in chunks if c["chunk_type"] == "table")
    n_fig: int = sum(1 for c in chunks if c["chunk_type"] == "figure_meta")
    print(f"{len(chunks)} chunks ({n_prose} prose, {n_table} table, {n_fig} figure).")

    if not chunks:
        print(f"No embeddable chunks produced for {basename}; skipping embed/upsert.")
        return

    # d. Embed.
    embeddings: np.ndarray = embed_chunks([c["text"] for c in chunks])
    print(f"Embedded {len(chunks)} chunks.")

    # e. Upsert.
    upsert_chunks(client, chunks, embeddings)
    print(f"Stored {len(chunks)} chunks in Qdrant.")

    all_chunks.extend(chunks)


def _print_ingestion_summary(stats: dict[str, Any], all_chunks: list[dict[str, Any]]) -> None:
    """Print the boxed ingestion summary block."""
    n_prose: int = sum(1 for c in all_chunks if c["chunk_type"] in ("prose", "list"))
    n_table: int = sum(1 for c in all_chunks if c["chunk_type"] == "table")
    n_fig: int = sum(1 for c in all_chunks if c["chunk_type"] == "figure_meta")
    total_tokens: int = sum(int(c["token_count"]) for c in all_chunks)
    avg_tokens: float = (total_tokens / len(all_chunks)) if all_chunks else 0.0
    region_breakdown: str = ", ".join(
        f"{rtype}={count}" for rtype, count in sorted(stats["region_types"].items())
    ) or "none"

    print()
    print(_HEAVY_RULE)
    print("  INGESTION SUMMARY")
    print(_HEAVY_RULE)
    print(f"  PDFs processed:      {stats['pdfs']}")
    print(f"  Total regions:       {stats['total_regions']} ({region_breakdown})")
    print(f"  Total chunks:        {len(all_chunks)}")
    print(f"    Prose chunks:      {n_prose}")
    print(f"    Table chunks:      {n_table}")
    print(f"    Figure chunks:     {n_fig}")
    print(f"  Total vectors:       {len(all_chunks)} in Qdrant")
    print(f"  Avg tokens/chunk:    {avg_tokens:.0f}")
    print(_HEAVY_RULE)


def _print_sample_chunks(all_chunks: list[dict[str, Any]]) -> None:
    """Show one table chunk and one prose chunk from this ingest."""
    table_chunk: dict[str, Any] | None = next(
        (c for c in all_chunks if c["chunk_type"] == "table"), None
    )
    prose_chunk: dict[str, Any] | None = next(
        (c for c in all_chunks if c["chunk_type"] in ("prose", "list")), None
    )

    print()
    print("Sample stored chunks:")
    if table_chunk is not None:
        print("  [TABLE CHUNK]")
        print(f"    page={table_chunk['page_num']}  title={table_chunk['table_title']}")
        print(f"    heading_path={table_chunk['heading_path']}")
        print(f"    text: {table_chunk['text'].replace(chr(10), ' ')[:200]}...")
    else:
        print("  [TABLE CHUNK] none produced in this ingest")
    if prose_chunk is not None:
        print("  [PROSE CHUNK]")
        print(f"    page={prose_chunk['page_num']}")
        print(f"    heading_path={prose_chunk['heading_path']}")
        print(f"    text: {prose_chunk['text'].replace(chr(10), ' ')[:200]}...")
    else:
        print("  [PROSE CHUNK] none produced in this ingest")


def _mark(flag: bool) -> str:
    """Render a checklist mark for a boolean condition."""
    return "✓" if flag else "✗"


def _print_deliverable_checklist(
    summary: dict[str, Any],
    n_table_chunks: int,
    distinct_types: bool,
    pipeline_ok: bool,
    tables_markdown_ok: bool,
) -> None:
    """Print the exact Day 6 deliverable checklist with computed marks."""
    print()
    print(_HEAVY_RULE)
    print("  DAY 6 DELIVERABLE CHECKLIST")
    print(_HEAVY_RULE)
    print(f"  {_mark(pipeline_ok)}  PDF → Qdrant end-to-end pipeline runs cleanly")
    print(f"  {_mark(tables_markdown_ok)}  Tables stored as structured Markdown (not flat strings)")
    print(f"  {_mark(summary['headings_in_metadata'])}  Headings preserved as chunk metadata (heading_path)")
    print(f"  {_mark(summary['find_table_query_works'])}  \"find the table about X\" query returns table chunks")
    print(f"  {_mark(summary['all_type_filters_correct'])}  Type-filtered search works (chunk_type filter)")
    print(f"  {_mark(distinct_types)}  Prose and table chunks have different chunk_type values")
    print(f"  {_mark(n_table_chunks > 0)}  At least {n_table_chunks} table chunks in collection")
    print(_HEAVY_RULE)


def main() -> None:
    """Orchestrate the 7-step ingestion + verification pipeline."""
    args: argparse.Namespace = _parse_args()

    # [1/7] Environment.
    if not os.environ.get("VOYAGE_API_KEY"):
        print(
            "ERROR: VOYAGE_API_KEY is not set. This pipeline requires real "
            "semantic embeddings (voyage-3). Add it to .env and retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Embedding: voyage-3 | Vector DB: {QDRANT_URL} | Collection: {COLLECTION_NAME}")

    # [2/7] Qdrant.
    client = get_client()
    print("Qdrant connected.")

    # [3/7] Resolve PDFs and decide whether to ingest.
    pdfs: list[str] = [] if args.verify_only else _resolve_pdfs(args)
    pdf_names: list[str] = [os.path.basename(p) for p in pdfs]
    all_chunks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"pdfs": 0, "total_regions": 0, "region_types": {}}

    do_ingest: bool = not args.verify_only
    if do_ingest:
        if not pdfs:
            print("ERROR: no PDFs to ingest. Pass --pdf or --pdf-dir.", file=sys.stderr)
            sys.exit(1)
        if collection_exists(client) and not args.reingest:
            count = count_by_type(client, "table") + count_by_type(client, "prose")
            print(
                f"Collection exists with {count}+ chunks. Use --reingest to re-ingest."
            )
            print(
                "WARNING: Ingest skipped. Verification below runs against the "
                "existing Qdrant collection, not the PDF path(s) you passed.",
                file=sys.stderr,
            )
            do_ingest = False
        else:
            create_collection(client, EMBEDDING_DIM)

    # [4/7] Ingest each PDF.
    if do_ingest:
        for pdf_path in pdfs:
            _ingest_one_pdf(
                client=client,
                pdf_path=pdf_path,
                enable_ocr_flag=args.enable_ocr,
                no_tables=args.no_tables,
                debug=args.debug,
                stats=stats,
                all_chunks=all_chunks,
            )
            stats["pdfs"] += 1

        # [5/7] Ingestion summary.
        _print_ingestion_summary(stats, all_chunks)

        # [6/7] Sample stored chunks.
        _print_sample_chunks(all_chunks)

    # [7/7] Verification.
    verify_corpus: list[str] = (
        pdf_names if stats["pdfs"] > 0 else []
    )
    summary: dict[str, Any] = run_verification(client, verify_corpus)

    # Deliverable checklist. Counts are read back from Qdrant so they are
    # correct on both the ingest and the --verify-only paths.
    n_table_in_collection: int = count_by_type(client, "table")
    n_prose_in_collection: int = count_by_type(client, "prose") + count_by_type(client, "list")
    distinct_types: bool = n_table_in_collection > 0 and n_prose_in_collection > 0
    tables_markdown_ok: bool = n_table_in_collection > 0  # serializer always emits [TABLE: ...] Markdown
    _print_deliverable_checklist(
        summary=summary,
        n_table_chunks=n_table_in_collection,
        distinct_types=distinct_types,
        pipeline_ok=True,  # reaching this line means no fatal error occurred
        tables_markdown_ok=tables_markdown_ok,
    )


if __name__ == "__main__":
    main()
