"""Human-readable summaries of graph structure."""
from __future__ import annotations

import networkx as nx

from src.graph import top_connected


def format_top_connected(G: nx.MultiDiGraph, n: int = 5) -> str:
    rows = top_connected(G, n=n)
    width = max((len(name) for name, _ in rows), default=4)
    lines = [
        f"Top {len(rows)} most-connected entities:",
        f"  {'#':>2}  {'entity':<{width}}  {'type':<8}  degree  mentions",
    ]
    for i, (name, degree) in enumerate(rows, start=1):
        etype = G.nodes[name].get("type", "?")
        mentions = G.nodes[name].get("mentions", 0)
        lines.append(f"  {i:>2}  {name:<{width}}  {etype:<8}  {degree:>6}  {mentions:>8}")
    return "\n".join(lines)
