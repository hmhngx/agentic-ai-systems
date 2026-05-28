"""
naive_rag.py - Day 4 deliverable: Naive RAG CLI over a PDF.

Framework-free implementation. OpenRouter API (OpenAI-compatible) only.

What this implements:
  The 6-step naive RAG loop from scratch:
  Query -> Embed -> Retrieve -> Construct Prompt -> Generate -> Return

What this deliberately does NOT implement:
  - Hybrid search (BM25 + dense) - that's Day 5
  - Reranking - that's Day 5
  - Query decomposition for multi-hop - that's Day 14
  - Self-correction / reflection - that's Day 14

Known failure modes of this implementation:
  1. Multi-hop questions will fail - single query vector cannot chain hops
  2. Exact keyword matches may miss - pure dense search, no BM25
  3. Answers are only as good as retrieval - no fallback if chunks are wrong
  4. No conversation memory - each question is independent

Usage:
  python naive_rag.py --pdf path/to/document.pdf
  python naive_rag.py --pdf path/to/document.pdf --reingest
  python naive_rag.py --pdf path/to/document.pdf --query "what is X?"
  python naive_rag.py --pdf path/to/document.pdf --debug
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make ``src/...`` importable when running this script directly from any
# working directory. ``Path(__file__).parent`` is the module root, which
# already contains the ``src`` package - we just need it on sys.path.
_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from qdrant_client import QdrantClient  # noqa: E402 - sys.path adjusted above
from src.citation import format_citation_footer  # noqa: E402
from src.embedder import EMBEDDING_DIM, embed_documents  # noqa: E402
from src.generator import MODEL, TEMPERATURE, generate_answer  # noqa: E402
from src.pdf_loader import load_and_chunk  # noqa: E402
from src.retriever import retrieve  # noqa: E402
from src.vector_store import (  # noqa: E402
    collection_exists,
    create_collection,
    get_client,
    upsert_chunks,
)


# UTF-8 stdout/stderr on Windows so ASCII box characters render correctly
# even in non-UTF-8 code pages. ``reconfigure`` is a no-op if the stream
# was already utf-8, so it is safe to call unconditionally.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    ``--top-k`` uses ``choices=[3, 4, 5]`` to enforce the topK pathology
    bound at argparse time. Anything outside this range is rejected
    before any code runs. ``vector_store.search`` also caps at 5 as
    defense-in-depth.
    """
    parser = argparse.ArgumentParser(
        prog="naive_rag",
        description=(
            "Naive RAG CLI over a PDF. OpenRouter API + qdrant-client. "
            "No orchestration framework wrappers."
        ),
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF file to ingest and query.",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Force re-ingestion even if the Qdrant collection already exists.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        choices=[3, 4, 5],   # RAG best practice; >5 triggers topK pathology
        help="Number of chunks to retrieve. Allowed: 3, 4, 5.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieval scores, chunk previews, and citation warnings.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output during ingestion.",
    )
    return parser.parse_args()


def _validate_environment(pdf_path: str) -> None:
    """Stage [1/5]: load .env and validate required keys + the PDF path."""
    load_dotenv()  # loads .env from the current working directory if present

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY is not set. "
            "Add it to .env or export it in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("OPENROUTER_EMBEDDING_MODEL"):
        # A default embedding model is configured in src.embedder, but we
        # surface this as a warning rather than a hard exit so the user
        # can still make the configuration explicit when desired.
        print(
            "WARNING: OPENROUTER_EMBEDDING_MODEL is not set. "
            "Using default embedding model from src/embedder.py.",
            file=sys.stderr,
        )

    if not os.path.isfile(pdf_path):
        # Raised here (with sys.exit) rather than letting pdf_loader raise
        # so the user sees stage [1/5] context rather than a deep traceback.
        print(f"ERROR: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)


def _ingest_pdf(client: QdrantClient, pdf_path: str, reingest: bool, quiet: bool) -> int:
    """Stage [3/5]: load + chunk + embed + upsert. Returns the chunk count.

    If a populated collection already exists and ``reingest`` is False
    we skip the ingest entirely - the on-disk Qdrant data is reused.
    """
    if collection_exists(client) and not reingest:
        # Read back the current vector count from Qdrant so the welcome
        # banner shows accurate "N chunks indexed" even when we skipped.
        info = client.get_collection(collection_name=os.environ.get(
            "COLLECTION_NAME", "naive_rag_chunks"
        ))
        existing: int = int(getattr(info, "points_count", 0) or getattr(info, "vectors_count", 0) or 0)
        print(
            f"Collection exists with {existing} vectors. "
            "Skipping ingestion. Use --reingest to force."
        )
        return existing

    chunks, stats = load_and_chunk(pdf_path)
    if not chunks:
        print(
            "ERROR: PDF produced zero chunks after filtering. "
            "Document may be too short or extraction failed.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not quiet:
        print(
            f"Stats: {stats['total_pages']} pages, "
            f"{stats['total_chunks']} chunks, "
            f"avg {stats['avg_tokens_per_chunk']:.0f} tokens/chunk "
            f"(min {stats['min_tokens']}, max {stats['max_tokens']})."
        )

    texts: list[str] = [c["text"] for c in chunks]
    embeddings = embed_documents(texts)

    create_collection(client, EMBEDDING_DIM)
    upsert_chunks(client, chunks, embeddings)

    print(f"Ingestion complete. {len(chunks)} chunks indexed.")
    return len(chunks)


def _print_welcome_banner(pdf_path: str, chunk_count: int, top_k: int) -> None:
    """Stage [4/5]: pretty banner so the user sees the live config."""
    bar: str = "=" * 47
    print(bar)
    print(f"  NAIVE RAG - {os.path.basename(pdf_path)}")
    print(f"  {chunk_count} chunks indexed | Model: {MODEL}")
    print(f"  Retrieval: top-{top_k} chunks | Temp: {TEMPERATURE}")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'debug' to toggle debug mode.")
    print(bar)


def _print_no_results_box(question: str) -> None:
    """Print the no-results message. We MUST NOT call Claude in this path.

    Why short-circuit?
        Sending an empty context to a temperature=0 LLM still produces
        an answer - it just comes from parametric memory rather than
        retrieved evidence. That is the textbook hallucination case for
        RAG systems. Refusing to answer is always better than confidently
        wrong.
    """
    rule: str = "+" + "-" * 41 + "+"
    print(rule)
    print("| No relevant content found".ljust(42) + " |")
    # Truncate the query for the box; ljust ensures consistent box width.
    query_display: str = question if len(question) <= 35 else question[:32] + "..."
    print(f"| Query: \"{query_display}\"".ljust(42) + " |")
    print("|".ljust(42) + " |")
    print("| The document does not appear to contain".ljust(42) + " |")
    print("| information relevant to this query.".ljust(42) + " |")
    print("| Try rephrasing or ask about a different".ljust(42) + " |")
    print("| topic covered in the document.".ljust(42) + " |")
    print(rule)


def _print_answer_block(answer: str, results: list[dict]) -> None:
    """Print the answer + unconditional citation footer."""
    rule: str = "+" + "-" * 41 + "+"
    print(rule)
    print("| Answer".ljust(42) + " |")
    print(rule)
    print(answer)
    print()
    print(format_citation_footer(results))


def _print_debug_chunks(results: list[dict]) -> None:
    """Show retrieved chunks with scores when --debug is active."""
    print("--- DEBUG: retrieved chunks ---")
    for result in results:
        preview: str = result["text"][:120].replace("\n", " ")
        if len(result["text"]) > 120:
            preview += "..."
        print(
            f"  [Doc {result['rank']}] page={result['page_num']} "
            f"score={result['score']:.4f}"
        )
        print(f"      {preview}")
    print("-------------------------------")


def _answer_one(client: QdrantClient, question: str, top_k: int, debug: bool) -> None:
    """Run one full RAG turn for ``question`` and print the result.

    This is the 6-step naive RAG loop in concrete form:
        embed -> search -> threshold -> (NO_RESULTS short-circuit) ->
        format context -> generate -> validate citations -> print.
    """
    results, status = retrieve(client, question, top_k=top_k, quiet=not debug)

    if status == "NO_RESULTS":
        _print_no_results_box(question)
        return

    if debug:
        _print_debug_chunks(results)

    answer, warnings = generate_answer(question, results)
    _print_answer_block(answer, results)

    if debug and warnings:
        # Warnings go to stderr so they do not pollute machine-parsable
        # answer output but still appear above the prompt in a terminal.
        for warning in warnings:
            print(warning, file=sys.stderr)


def _query_loop(client: QdrantClient, top_k: int, debug: bool) -> None:
    """Stage [5/5]: interactive REPL.

    Recognized commands inside the loop:
      - 'quit' / 'exit': stop the program
      - 'debug': toggle debug mode without restarting
    Anything else is treated as a question.
    """
    while True:
        try:
            question: str = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C exits cleanly; we explicitly catch these
            # rather than using bare ``except`` to satisfy the no-bare-
            # except rule from the verification gate.
            print()  # newline so the next shell prompt starts on its own line
            return

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            return
        if question.lower() == "debug":
            debug = not debug
            print(f"Debug mode: {'ON' if debug else 'OFF'}")
            continue

        _answer_one(client, question, top_k=top_k, debug=debug)


def main() -> None:
    """Top-level entry point: orchestrates the five pipeline stages."""
    args = _parse_args()

    # ----- Stage [1/5]: env + path validation -----
    _validate_environment(args.pdf)

    # ----- Stage [2/5]: Qdrant connection -----
    client = get_client()

    # ----- Stage [3/5]: ingest (or skip if collection already populated) -----
    chunk_count: int = _ingest_pdf(client, args.pdf, args.reingest, args.quiet)

    # ----- Stage [4/5]: welcome banner -----
    if args.query is None:
        # Banner is only useful for the interactive loop; skip in --query
        # mode so single-shot CLI calls have clean machine-readable output.
        _print_welcome_banner(args.pdf, chunk_count, args.top_k)

    # ----- Stage [5/5]: query loop or single query -----
    if args.query is not None:
        _answer_one(client, args.query, top_k=args.top_k, debug=args.debug)
        return

    _query_loop(client, top_k=args.top_k, debug=args.debug)


if __name__ == "__main__":
    main()
