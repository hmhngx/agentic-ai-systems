"""agentic_rag.py — Day 11 (Week 2 Review): a LangGraph agent that uses RAG as a
tool, decides retrieve-vs-direct, and self-corrects via a RAGAS-faithfulness-
gated Reflexion loop (refine if <0.70, max 2 retries, then graceful fallback).

RUN:
  python agentic_rag.py                 # offline demo: 4 paths
  python agentic_rag.py "your question" # single question
  python agentic_rag.py --eval          # full-pipeline RAGAS eval table

Offline & deterministic by default (Qdrant :memory:, tf-idf, extractive
generator, offline faithfulness proxy). USE_LLM=1 / USE_RAGAS=1 swap in real
backends without changing the graph.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Make `src` and this file importable when run directly from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph import build_graph, run_agent          # noqa: E402
from src.ragas_eval import evaluate_dataset, format_table  # noqa: E402


def _rule(title: str = "", width: int = 74) -> str:
    return "-" * width if not title else f"-- {title} " + "-" * max(width - len(title) - 4, 0)


def visualize(mermaid_path: Optional[str] = None) -> None:
    drawable = build_graph().get_graph()
    print(_rule("GRAPH (ASCII)"))
    try:
        print(drawable.draw_ascii())
    except Exception:  # noqa: BLE001 - grandalf can't place the refine self-loop
        print("(ASCII layout can't render the reflexion back-edge; see Mermaid below.)")
    print(_rule("GRAPH (Mermaid)"))
    print(drawable.draw_mermaid().rstrip())
    if mermaid_path:
        with open(mermaid_path, "w", encoding="utf-8") as fh:
            fh.write(drawable.draw_mermaid())


def _print_run(question: str) -> None:
    final = run_agent(question)
    print(_rule(f"Q: {question}"))
    for line in final["log"]:
        print("  " + line)
    print(f"  route={final['route']} status={final['status']} served={final['served']} "
          f"faithfulness={final['faithfulness']:.2f} attempts={len(final['attempts_log'])}")
    print(f"  ANSWER: {final['final_answer']}")


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--eval":
        print(_rule("FULL-PIPELINE RAGAS EVAL"))
        print(format_table(evaluate_dataset()))
        return

    if argv:
        _print_run(" ".join(argv))
        return

    print(_rule("DAY 11 — AGENTIC RAG"))
    here = os.path.dirname(os.path.abspath(__file__))
    visualize(mermaid_path=os.path.join(here, "graph.mmd"))
    print()
    print(_rule("DEMO 1 — first-try faithful (no reflexion)"))
    _print_run("What is the Helios query quota per day?")
    print(_rule("DEMO 2 — reflexion loop (low faithfulness -> refine -> faithful)"))
    _print_run("How long does Helios keep documents before deleting them?")
    print(_rule("DEMO 3 — direct (answer from context, no retrieval)"))
    _print_run("Given that the pro tier allows 100000 queries per day, is 80000 within that limit?")
    print(_rule("DEMO 4 — graceful fallback (unanswerable, retries exhausted)"))
    _print_run("What is the Helios carbon footprint per query?")
    print()
    print(_rule("EVAL"))
    print(format_table(evaluate_dataset()))


if __name__ == "__main__":
    main()
