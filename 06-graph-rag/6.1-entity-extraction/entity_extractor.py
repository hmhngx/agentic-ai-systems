"""entity_extractor.py — document -> knowledge graph (Graph RAG, Day 6.1).

Pipeline: load -> extract (offline spaCy by default; OpenRouter if USE_LLM=1)
-> disambiguate -> build NetworkX graph -> save to disk -> embed entities ->
upsert to Qdrant -> visualize clusters -> print top-5 most-connected.

Usage:
  python entity_extractor.py --doc path/to/doc.pdf
  python entity_extractor.py --doc doc.txt --no-qdrant --no-viz --top 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout so entity names with non-ASCII render on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()

from src import config  # noqa: E402
from src.disambiguation import resolve  # noqa: E402
from src.embeddings import embed_entities, embedding_dim  # noqa: E402
from src.extractor import extract  # noqa: E402
from src.graph import build_graph, save_graph, to_undirected_weighted  # noqa: E402
from src.loader import load_pages, main_body_pages  # noqa: E402
from src.report import format_top_connected  # noqa: E402
from src.visualize import draw_graph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract a knowledge graph from a document.")
    ap.add_argument("--doc", required=True, help="path to a PDF or .txt/.md document")
    ap.add_argument("--out-dir", default="out", help="directory for graph + image outputs")
    ap.add_argument("--top", type=int, default=5, help="how many top-connected entities to print")
    ap.add_argument("--no-qdrant", action="store_true", help="skip Qdrant upsert")
    ap.add_argument("--no-viz", action="store_true", help="skip PNG visualization")
    ap.add_argument(
        "--include-back-matter",
        action="store_true",
        help="keep References/Appendix (default: extract only the main body)",
    )
    args = ap.parse_args()

    print(f"Loading {args.doc} ...")
    pages = load_pages(args.doc)
    print(f"  {len(pages)} page(s).")
    if not args.include_back_matter:
        pages = main_body_pages(pages)
        print(f"  using main body: {len(pages)} page(s) (References/Appendix dropped).")

    mode = "OpenRouter LLM" if config.use_llm() else "offline spaCy"
    print(f"Extracting entities & relations ({mode}) ...")
    result = extract(pages)
    print(f"  raw: {len(result.entities)} entities, {len(result.triples)} triples")

    result = resolve(result)
    print(f"  after disambiguation: {len(result.entities)} entities, {len(result.triples)} triples")

    G = build_graph(result)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    paths = save_graph(G, args.out_dir)
    print(f"Graph saved: {paths['graphml']} | {paths['json']}")

    if not args.no_qdrant:
        from src import vector_store

        client = vector_store.get_client()
        vectors = embed_entities(result.entities)
        U = to_undirected_weighted(G)
        degrees = {n: U.degree(n) for n in U.nodes()}
        vector_store.create_collection(client, dim=embedding_dim())
        vector_store.upsert_entities(client, result.entities, vectors, degrees=degrees)

    if not args.no_viz:
        img = draw_graph(G, str(Path(args.out_dir) / "graph.png"))
        print(f"Visualization saved: {img}")

    print()
    print(format_top_connected(G, n=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
