import numpy as np
import pytest
from unittest.mock import MagicMock

from src.retriever import embed_query, vanilla_retrieve, RetrievedChunk


def _mock_qdrant_point(chunk_id: str, text: str, score: float, entities: list[str]):
    p = MagicMock()
    p.score = score
    p.payload = {"chunk_id": chunk_id, "text": text, "entities": entities}
    return p


def test_embed_query_returns_normalized_vector():
    vec = embed_query("What is the training method for GPT-4?")
    assert vec.ndim == 1
    assert vec.dtype == np.float32
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_query_different_queries_differ():
    v1 = embed_query("OpenAI CEO")
    v2 = embed_query("protein folding database")
    assert not np.allclose(v1, v2)


def test_vanilla_retrieve_returns_retrieved_chunks():
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _mock_qdrant_point("gpt4", "GPT-4 text", 0.91, ["GPT-4"]),
        _mock_qdrant_point("altman_ceo", "Sam Altman text", 0.85, ["Sam Altman", "OpenAI"]),
    ]
    chunks = vanilla_retrieve(mock_client, np.zeros(256, dtype=np.float32))
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "gpt4"
    assert chunks[0].score == pytest.approx(0.91)
    assert chunks[0].source == "vector"
    assert isinstance(chunks[0], RetrievedChunk)


def test_vanilla_retrieve_top_k_passed_to_qdrant():
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = []
    vanilla_retrieve(mock_client, np.zeros(256, dtype=np.float32), top_k=3)
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["limit"] == 3


import networkx as nx
from src.retriever import bfs_1hop, GraphTriple


@pytest.fixture
def small_graph():
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        G.add_node(n)
    G.add_edge("A", "B", relation="knows", weight=1)
    G.add_edge("B", "C", relation="works_at", weight=2)
    G.add_edge("D", "A", relation="manages", weight=1)
    return G


def test_bfs_1hop_returns_outgoing_edges(small_graph):
    triples = bfs_1hop(small_graph, ["A"])
    keys = {(t.source_entity, t.relation, t.target_entity) for t in triples}
    assert ("A", "knows", "B") in keys


def test_bfs_1hop_returns_incoming_edges(small_graph):
    triples = bfs_1hop(small_graph, ["A"])
    keys = {(t.source_entity, t.relation, t.target_entity) for t in triples}
    assert ("D", "manages", "A") in keys


def test_bfs_does_not_go_two_hops(small_graph):
    triples = bfs_1hop(small_graph, ["A"])
    entities_reached = {t.target_entity for t in triples} | {t.source_entity for t in triples}
    assert "C" not in entities_reached  # C is 2 hops from A


def test_bfs_unknown_entity_returns_empty(small_graph):
    assert bfs_1hop(small_graph, ["NONEXISTENT"]) == []


def test_bfs_max_neighbors_caps_outgoing(small_graph):
    G = small_graph
    for i in range(15):
        G.add_node(f"X{i}")
        G.add_edge("A", f"X{i}", relation="rel", weight=1)
    triples = bfs_1hop(G, ["A"], max_neighbors=3)
    outgoing_from_a = [t for t in triples if t.source_entity == "A"]
    assert len(outgoing_from_a) <= 3


def test_bfs_no_duplicate_triples(small_graph):
    triples = bfs_1hop(small_graph, ["A", "A"])  # seed repeated
    keys = [(t.source_entity, t.relation, t.target_entity) for t in triples]
    assert len(keys) == len(set(keys))
