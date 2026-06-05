"""Render the knowledge graph to PNG with clear, color-coded communities.

Communities (NetworkX Louvain on the undirected projection) drive node color, so
densely-connected groups — authors, models, institutions — pop out as clusters.
Node size scales with degree; the most-connected entities are labeled. matplotlib
runs on the non-interactive Agg backend so it works headless / in tests.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from src.graph import to_undirected_weighted, top_connected  # noqa: E402


def draw_graph(G: nx.MultiDiGraph, out_path: str, *, label_top: int = 12, seed: int = 42) -> str:
    U = to_undirected_weighted(G)
    if U.number_of_nodes() == 0:
        raise ValueError("cannot visualize an empty graph")

    # community detection -> color index per node
    communities = nx.community.louvain_communities(U, weight="weight", seed=seed)
    color_of = {node: i for i, comm in enumerate(communities) for node in comm}
    node_colors = [color_of.get(n, 0) for n in U.nodes()]

    degrees = dict(U.degree())
    node_sizes = [300 + 400 * degrees[n] for n in U.nodes()]

    pos = nx.spring_layout(U, seed=seed, k=0.6, iterations=100, weight="weight")

    top_names = {name for name, _ in top_connected(G, n=label_top)}
    labels = {n: n for n in U.nodes() if n in top_names}

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(U, pos, alpha=0.25, width=1.0)
    nx.draw_networkx_nodes(
        U, pos, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.tab20, alpha=0.9
    )
    nx.draw_networkx_labels(U, pos, labels=labels, font_size=9, font_weight="bold")
    plt.title(
        f"Knowledge Graph — {U.number_of_nodes()} entities, {len(communities)} communities",
        fontsize=14,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
