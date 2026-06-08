import networkx as nx
from src.knowledge_base import build_curated_graph


def test_graph_is_multidigraph():
    G = build_curated_graph()
    assert isinstance(G, nx.MultiDiGraph)


def test_q1_chain_present():
    G = build_curated_graph()
    # Dario → co-founded → Anthropic → uses → Constitutional AI
    assert G.has_node("Dario Amodei")
    assert G.has_node("Anthropic")
    assert G.has_node("Constitutional AI")
    edges_da = {d["relation"] for _, _, d in G.out_edges("Dario Amodei", data=True)}
    assert "co-founded" in edges_da
    edges_ant = {d["relation"] for _, _, d in G.out_edges("Anthropic", data=True)}
    assert "uses" in edges_ant


def test_q2_chain_present():
    G = build_curated_graph()
    assert G.has_node("AlphaFold 2")
    assert G.has_node("Protein Data Bank")
    assert G.has_node("Demis Hassabis")
    pdb_edges = {d["relation"] for _, _, d in G.out_edges("AlphaFold 2", data=True)}
    assert "trained_on" in pdb_edges


def test_q3_chain_present():
    G = build_curated_graph()
    assert G.has_node("Vaswani et al.")
    assert G.has_node("Transformer")
    assert G.has_node("LLaMA 2")
    assert G.has_node("Common Crawl")
    t_edges = {d["relation"] for _, _, d in G.out_edges("LLaMA 2", data=True)}
    assert "trained_on" in t_edges


def test_q4_chain_present():
    G = build_curated_graph()
    assert G.has_node("Geoffrey Hinton")
    assert G.has_node("Google Brain")
    assert G.has_node("Google DeepMind")
    assert G.has_node("Gemini Ultra")
    gb_edges = {d["relation"] for _, _, d in G.out_edges("Google Brain", data=True)}
    assert "merged_with" in gb_edges or "formed" in gb_edges


def test_q5_chain_present():
    G = build_curated_graph()
    assert G.has_node("Sam Altman")
    assert G.has_node("OpenAI")
    assert G.has_node("GPT-4")
    assert G.has_node("RLHF")
    oa_edges = {d["relation"] for _, _, d in G.out_edges("OpenAI", data=True)}
    assert "developed" in oa_edges


def test_all_edges_have_relation_and_weight():
    G = build_curated_graph()
    for u, v, data in G.edges(data=True):
        assert "relation" in data, f"Edge ({u},{v}) missing 'relation'"
        assert "weight" in data, f"Edge ({u},{v}) missing 'weight'"
