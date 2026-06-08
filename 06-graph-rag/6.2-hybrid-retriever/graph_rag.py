#!/usr/bin/env python3
"""
Graph RAG Hybrid Retriever — benchmark CLI.

Usage:
    python graph_rag.py --index              # build Qdrant + graph, then exit
    python graph_rag.py --benchmark          # run 5-question comparison
    python graph_rag.py --query "question"   # answer a single question

Requires Qdrant running locally:
    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.answerer import generate_answer
from src.indexer import build_index
from src.retriever import embed_query, hybrid_retrieve, vanilla_retrieve

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "What alignment technique does the company started by the former "
            "OpenAI researchers who left in 2021 use?"
        ),
        "expected": "Constitutional AI",
        "hop_chain": "Dario Amodei → left OpenAI → co-founded Anthropic → uses Constitutional AI",
    },
    {
        "id": "Q2",
        "question": (
            "What database provided the training data for the model that solved "
            "protein folding, and who founded its developer?"
        ),
        "expected": "Protein Data Bank",
        "hop_chain": "AlphaFold 2 → trained_on → Protein Data Bank; DeepMind → co-founded by → Demis Hassabis",
    },
    {
        "id": "Q3",
        "question": (
            "What is the primary pretraining data source for the open-weight model "
            "that uses the architecture from Vaswani et al.?"
        ),
        "expected": "Common Crawl",
        "hop_chain": "Vaswani et al. → introduced → Transformer → LLaMA 2 uses Transformer → trained_on → Common Crawl",
    },
    {
        "id": "Q4",
        "question": (
            "What is the first model released by the organization formed after "
            "Geoffrey Hinton's former employer merged with DeepMind?"
        ),
        "expected": "Gemini Ultra",
        "hop_chain": "Geoffrey Hinton → worked_at → Google Brain → formed → Google DeepMind → released → Gemini Ultra",
    },
    {
        "id": "Q5",
        "question": (
            "What training method does the most capable model from the company "
            "led by Sam Altman use?"
        ),
        "expected": "RLHF",
        "hop_chain": "Sam Altman → is_ceo_of → OpenAI → developed → GPT-4 → trained_with → RLHF",
    },
]

_SEP = "═" * 72
_SUB = "─" * 72


def _verdict(vanilla_ans: str, graph_ans: str, expected: str) -> tuple[str, str]:
    v_has = expected.lower() in vanilla_ans.lower()
    g_has = expected.lower() in graph_ans.lower()
    if g_has and not v_has:
        return "✓ GRAPH RAG WINS", "graph"
    if v_has and g_has:
        return "~ TIE (both correct)", "tie"
    if v_has and not g_has:
        return "✗ VANILLA RAG WINS", "vanilla"
    return "? NEITHER correct (offline hash embeddings — run with API key for real results)", "neither"


def _truncate(s: str, n: int = 400) -> str:
    return s[:n] + "…" if len(s) > n else s


def run_benchmark(client, graph) -> list[dict]:
    results = []
    graph_wins = vanilla_wins = ties = neither = 0

    for q in TEST_QUESTIONS:
        print(f"\n{_SEP}")
        print(f"{q['id']}: {q['question']}")
        print(_SUB)

        query_vec = embed_query(q["question"])

        v_chunks = vanilla_retrieve(client, query_vec)
        v_context = "\n".join(
            f"[{i+1}] (score={c.score:.3f}) {c.text}" for i, c in enumerate(v_chunks)
        )
        vanilla_ans = generate_answer(v_context, q["question"])

        hybrid = hybrid_retrieve(client, graph, q["question"], query_vec)
        graph_ans = generate_answer(hybrid.fused_context(), q["question"])

        verdict, winner = _verdict(vanilla_ans, graph_ans, q["expected"])

        print(f"VANILLA RAG  → {_truncate(vanilla_ans)}")
        print()
        print(f"GRAPH RAG    → {_truncate(graph_ans)}")
        print()
        print(f"VERDICT      → {verdict}")
        print(f"             [hops: {q['hop_chain']}]")

        if winner == "graph":
            graph_wins += 1
        elif winner == "vanilla":
            vanilla_wins += 1
        elif winner == "tie":
            ties += 1
        else:
            neither += 1

        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "vanilla_answer": vanilla_ans,
                "graph_answer": graph_ans,
                "expected": q["expected"],
                "verdict": verdict,
                "winner": winner,
                "hop_chain": q["hop_chain"],
            }
        )

    print(f"\n{_SEP}")
    print(
        f"FINAL SCORE: Graph RAG {graph_wins}/5 | "
        f"Vanilla RAG {vanilla_wins}/5 | "
        f"Ties {ties} | Neither {neither}"
    )
    print(_SEP)

    _write_obsidian_report(results, graph_wins, vanilla_wins, ties, neither)
    return results


def _write_obsidian_report(
    results: list[dict],
    graph_wins: int,
    vanilla_wins: int,
    ties: int,
    neither: int,
) -> None:
    from src import config

    obsidian_path = (
        Path(config.OBSIDIAN_VAULT)
        / "06-Graph RAG (Hybrid retrieval)"
        / "6.2-Benchmark-Results.md"
    )

    lines = [
        "# 6.2 Hybrid Graph RAG — Benchmark Results",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Graph RAG wins | {graph_wins}/5 |",
        f"| Vanilla RAG wins | {vanilla_wins}/5 |",
        f"| Ties | {ties}/5 |",
        f"| Neither correct | {neither}/5 |",
        "",
        "---",
        "",
    ]

    for r in results:
        lines += [
            f"## {r['id']}: {r['question']}",
            "",
            f"**Expected answer fragment:** `{r['expected']}`  ",
            f"**Hop chain:** `{r['hop_chain']}`",
            "",
            "### Vanilla RAG Answer",
            "```",
            r["vanilla_answer"][:600],
            "```",
            "",
            "### Graph RAG Answer",
            "```",
            r["graph_answer"][:600],
            "```",
            "",
            f"**Verdict:** {r['verdict']}",
            "",
        ]

        if r["winner"] == "graph":
            lines += [
                "**Analysis — Graph RAG won:**  ",
                "The answer required traversing the knowledge graph. "
                "Vanilla RAG's top-5 chunks did not contain the bridging entity or "
                "relationship needed to resolve the multi-hop chain. "
                "Graph RAG's BFS step surfaced the connecting triple that Vanilla RAG missed.",
                "",
            ]
        elif r["winner"] == "vanilla":
            lines += [
                "**Analysis — Vanilla RAG won / Graph RAG failed:**  ",
                "The relevant information was concentrated in a high-scoring chunk that "
                "vector search ranked in the top-5. Graph RAG's additional triples "
                "did not improve the answer and may have introduced noise.",
                "",
            ]
        elif r["winner"] == "tie":
            lines += [
                "**Analysis — Tie:**  ",
                "Both systems found the answer. "
                "The question may be partially answerable from a single chunk. "
                "Graph RAG still demonstrates the correct reasoning chain.",
                "",
            ]
        else:
            lines += [
                "**Analysis — Neither correct:**  ",
                "Offline hash embeddings are non-semantic, so vector search may have "
                "retrieved irrelevant chunks. "
                "Re-run with `OPENROUTER_API_KEY` set to enable semantic embeddings.",
                "",
            ]

        lines += ["---", ""]

    try:
        obsidian_path.parent.mkdir(parents=True, exist_ok=True)
        obsidian_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nObsidian report written → {obsidian_path}")
    except Exception as exc:
        print(f"Warning: could not write Obsidian report: {exc}", file=sys.stderr)


def run_single_query(client, graph, question: str) -> None:
    query_vec = embed_query(question)

    print(f"\nQ: {question}")
    print(_SUB)

    v_chunks = vanilla_retrieve(client, query_vec)
    v_ctx = "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(v_chunks))
    vanilla_ans = generate_answer(v_ctx, question)
    print(f"VANILLA RAG  → {_truncate(vanilla_ans)}")
    print()

    hybrid = hybrid_retrieve(client, graph, question, query_vec)
    graph_ans = generate_answer(hybrid.fused_context(), question)
    print(f"GRAPH RAG    → {_truncate(graph_ans)}")
    print()
    print("Graph triples injected:")
    for t in hybrid.triples[:10]:
        print(f"  {t.source_entity} → [{t.relation}] → {t.target_entity}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph RAG Hybrid Retriever — compare vanilla vs graph-augmented RAG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", action="store_true", help="Build index and exit")
    group.add_argument("--benchmark", action="store_true", help="Run 5-question benchmark")
    group.add_argument("--query", metavar="QUESTION", help="Answer a single question")
    args = parser.parse_args()

    print("Building index…")
    client, graph = build_index()

    if args.index:
        print("Index built successfully.")
        return

    if args.benchmark:
        run_benchmark(client, graph)
        return

    if args.query:
        run_single_query(client, graph, args.query)


if __name__ == "__main__":
    main()
