import numpy as np
import networkx as nx
import pytest
from src.retriever import (
    merge_results,
    hybrid_retrieve,
    RetrievedChunk,
    GraphTriple,
    HybridResult,
)
from unittest.mock import MagicMock


@pytest.fixture
def tiny_graph():
    G = nx.MultiDiGraph()
    G.add_node("X"); G.add_node("Y"); G.add_node("Z")
    G.add_edge("X", "Y", relation="connects_to", weight=1)
    G.add_edge("Y", "Z", relation="part_of", weight=2)
    return G


def test_fused_context_has_documents_section(tiny_graph):
    r = HybridResult(
        chunks=[RetrievedChunk("c1", "Doc about X", 0.9, ["X"], "vector")],
        triples=[GraphTriple("X", "connects_to", "Y", 1.0)],
    )
    ctx = r.fused_context()
    assert "[Retrieved Documents]" in ctx
    assert "Doc about X" in ctx


def test_fused_context_has_relationships_section(tiny_graph):
    r = HybridResult(
        chunks=[],
        triples=[GraphTriple("X", "connects_to", "Y", 1.0)],
    )
    ctx = r.fused_context()
    assert "[Entity Relationships]" in ctx
    assert "X → [connects_to] → Y" in ctx


def test_fused_context_graph_boosted_tag():
    r = HybridResult(
        chunks=[RetrievedChunk("c2", "extra text", 0.5, [], "graph_boosted")],
        triples=[],
    )
    assert "[graph-boosted]" in r.fused_context()


def test_merge_deduplicates_triples(tiny_graph):
    chunks = [RetrievedChunk("c1", "text", 0.9, ["X"], "vector")]
    triples = [
        GraphTriple("X", "connects_to", "Y", 1.0),
        GraphTriple("X", "connects_to", "Y", 1.0),  # exact duplicate
        GraphTriple("X", "connects_to", "Y", 0.5),  # same key, different score
    ]
    result = merge_results(chunks, triples, tiny_graph, np.zeros(256))
    assert len(result.triples) == 1


def test_merge_preserves_all_chunks(tiny_graph):
    chunks = [
        RetrievedChunk("c1", "text1", 0.9, [], "vector"),
        RetrievedChunk("c2", "text2", 0.8, [], "vector"),
    ]
    result = merge_results(chunks, [], tiny_graph, np.zeros(256))
    assert len(result.chunks) == 2


def test_hybrid_retrieve_returns_hybrid_result():
    mock_client = MagicMock()
    p = MagicMock()
    p.score = 0.8
    p.payload = {
        "chunk_id": "altman_ceo",
        "text": "Sam Altman is CEO of OpenAI.",
        "entities": ["Sam Altman", "OpenAI"],
    }
    mock_client.query_points.return_value.points = [p]

    G = nx.MultiDiGraph()
    G.add_node("Sam Altman"); G.add_node("OpenAI"); G.add_node("GPT-4")
    G.add_edge("Sam Altman", "OpenAI", relation="is_ceo_of", weight=1)
    G.add_edge("OpenAI", "GPT-4", relation="developed", weight=1)

    result = hybrid_retrieve(
        mock_client, G, "What company does Sam Altman lead?", np.zeros(256)
    )
    assert isinstance(result, HybridResult)
    assert len(result.chunks) >= 1
    triple_entities = {t.source_entity for t in result.triples} | {t.target_entity for t in result.triples}
    assert "Sam Altman" in triple_entities or "OpenAI" in triple_entities


def test_hybrid_retrieve_includes_graph_triples_for_seed_entities():
    mock_client = MagicMock()
    p = MagicMock()
    p.score = 0.75
    p.payload = {
        "chunk_id": "dario_left",
        "text": "Dario Amodei left OpenAI in 2021.",
        "entities": ["Dario Amodei", "OpenAI"],
    }
    mock_client.query_points.return_value.points = [p]

    G = nx.MultiDiGraph()
    G.add_node("Dario Amodei"); G.add_node("OpenAI"); G.add_node("Anthropic")
    G.add_edge("Dario Amodei", "OpenAI", relation="left", weight=1)
    G.add_edge("Dario Amodei", "Anthropic", relation="co-founded", weight=1)

    result = hybrid_retrieve(
        mock_client, G, "What did Dario Amodei found?", np.zeros(256)
    )
    relation_targets = {t.target_entity for t in result.triples}
    assert "Anthropic" in relation_targets
