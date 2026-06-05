"""NetworkX knowledge graph: build from triples, persist, and query.

Storage choices:
  - MultiDiGraph preserves direction AND parallel edges (two entities can have
    several distinct relations).
  - GraphML for interoperability (Gephi/yEd) — list attrs (aliases) are joined to
    a string because GraphML only stores scalars.
  - JSON node-link for a lossless round-trip used by load_graph().
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.schema import ExtractionResult


def build_graph(result: ExtractionResult) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for e in result.entities:
        G.add_node(e.name, type=e.type.value, mentions=e.mentions, aliases=list(e.aliases))
    for t in result.triples:
        # only connect entities that exist as nodes (guards against stray names)
        if t.source in G and t.target in G:
            G.add_edge(t.source, t.target, relation=t.relation, evidence=t.evidence)
    return G


def to_undirected_weighted(G: nx.MultiDiGraph) -> nx.Graph:
    """Collapse to an undirected simple graph; weight = number of parallel relations.
    Used for community detection and layout."""
    U = nx.Graph()
    U.add_nodes_from(G.nodes(data=True))
    for u, v in G.edges():
        if U.has_edge(u, v):
            U[u][v]["weight"] += 1
        else:
            U.add_edge(u, v, weight=1)
    return U


def top_connected(G: nx.MultiDiGraph, n: int = 5) -> list[tuple[str, int]]:
    """Return [(node, degree)] for the n most-connected nodes (undirected degree).
    Ties broken by mention count then name for determinism."""
    U = to_undirected_weighted(G)
    ranked = sorted(
        U.nodes(),
        key=lambda x: (-U.degree(x), -G.nodes[x].get("mentions", 0), x),
    )
    return [(node, U.degree(node)) for node in ranked[:n]]


def save_graph(G: nx.MultiDiGraph, out_dir: str) -> dict[str, str]:
    """Write GraphML + JSON node-link. Returns the paths written."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    graphml_path = str(Path(out_dir) / "graph.graphml")
    json_path = str(Path(out_dir) / "graph.json")

    # GraphML can't hold lists: serialize aliases to a string on a copy.
    H = G.copy()
    for _, data in H.nodes(data=True):
        data["aliases"] = "; ".join(data.get("aliases", []))
    nx.write_graphml(H, graphml_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G, edges="links"), f, ensure_ascii=False, indent=2)

    return {"graphml": graphml_path, "json": json_path}


def load_graph(json_path: str) -> nx.MultiDiGraph:
    """Reload the lossless JSON node-link graph."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, multigraph=True, directed=True, edges="links")
