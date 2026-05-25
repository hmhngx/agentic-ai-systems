"""CLI orchestrator for the chunking benchmark suite."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

# Prefer locally cached embedding weights (no Hub warnings during CLI runs).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking benchmark suite")
    parser.add_argument(
        "--pdf_path",
        type=str,
        required=True,
        help="Path to the PDF document to chunk and evaluate",
    )
    args = parser.parse_args()

    import pandas as pd

    from src.chunkers import naive_chunk, recursive_chunk, semantic_chunk, sentence_chunk
    from src.evaluator import evaluate_chunks
    from src.pdf_loader import load_pdf_as_text

    try:
        text = load_pdf_as_text(args.pdf_path)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    print(f"Loaded {len(text)} characters from {args.pdf_path}.")

    strategies: list[tuple[str, Callable[[str], list[str]]]] = [
        ("naive", naive_chunk),
        ("recursive", recursive_chunk),
        ("sentence", sentence_chunk),
        ("semantic", semantic_chunk),
    ]

    results: list[dict] = []
    for strategy_name, chunker in strategies:
        print(f"Running {strategy_name}...")
        chunks = chunker(text)
        results.append(evaluate_chunks(chunks, strategy_name))

    df = pd.DataFrame(results)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))

    winner = max(results, key=lambda r: r["ICC Score"])
    print(
        f"Best ICC: {winner['Strategy']} ({winner['ICC Score']:.4f})"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
