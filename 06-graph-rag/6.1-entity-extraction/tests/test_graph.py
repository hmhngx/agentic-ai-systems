from pathlib import Path

from src.graph import (
    build_graph,
    save_graph,
    load_graph,
    top_connected,
    to_undirected_weighted,
)
from src.schema import Entity, EntityType, ExtractionResult, Triple


def _sample():
    return ExtractionResult(
        entities=[
            Entity(name="Stanford University", type=EntityType.ORG, mentions=3),
            Entity(name="Nelson Liu", type=EntityType.PERSON, mentions=2),
            Entity(name="Percy Liang", type=EntityType.PERSON, mentions=1),
            Entity(name="Claude", type=EntityType.MODEL, mentions=1),
        ],
        triples=[
            Triple(source="Nelson Liu", relation="works_at", target="Stanford University"),
            Triple(source="Percy Liang", relation="works_at", target="Stanford University"),
            Triple(source="Nelson Liu", relation="collaborates_with", target="Percy Liang"),
        ],
    )


def test_nodes_are_entities_edges_are_relations():
    G = build_graph(_sample())
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert G.nodes["Stanford University"]["type"] == "ORG"
    rels = {d["relation"] for _, _, d in G.edges(data=True)}
    assert "works_at" in rels


def test_top_connected_orders_by_degree():
    G = build_graph(_sample())
    top = top_connected(G, n=2)
    assert top[0][0] == "Stanford University"   # degree 2, highest
    assert len(top) == 2


def test_save_and_load_roundtrip(tmp_path: Path):
    G = build_graph(_sample())
    out = save_graph(G, str(tmp_path))
    assert Path(out["graphml"]).exists()
    assert Path(out["json"]).exists()
    G2 = load_graph(out["json"])
    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()
    assert G2.nodes["Claude"]["type"] == "MODEL"


def test_undirected_projection_weights_parallel_edges():
    G = build_graph(_sample())
    U = to_undirected_weighted(G)
    assert U.number_of_nodes() == 4
    assert U.has_edge("Nelson Liu", "Stanford University")
