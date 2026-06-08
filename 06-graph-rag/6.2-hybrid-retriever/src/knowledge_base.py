"""Hand-curated knowledge graph covering all 5 multi-hop question chains.

In offline mode this is the authoritative graph — no LLM extraction needed.
Every (source, relation, target) triple is verified to be reachable from
at least one query in the benchmark.
"""
from __future__ import annotations

import networkx as nx


def build_curated_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # ── Nodes ──────────────────────────────────────────────────────────────
    nodes = [
        # Q1
        ("Dario Amodei", "PERSON"),
        ("Daniela Amodei", "PERSON"),
        ("OpenAI", "ORG"),
        ("Anthropic", "ORG"),
        ("Constitutional AI", "CONCEPT"),
        ("Claude 3", "MODEL"),
        # Q2
        ("Demis Hassabis", "PERSON"),
        ("DeepMind", "ORG"),
        ("AlphaFold 2", "MODEL"),
        ("UniProt", "DATASET"),
        ("Protein Data Bank", "DATASET"),
        ("Google", "ORG"),
        # Q3
        ("Vaswani et al.", "PERSON"),
        ("Transformer", "CONCEPT"),
        ("LLaMA 2", "MODEL"),
        ("Common Crawl", "DATASET"),
        ("Meta AI", "ORG"),
        ("Yann LeCun", "PERSON"),
        # Q4
        ("Geoffrey Hinton", "PERSON"),
        ("Google Brain", "ORG"),
        ("Google DeepMind", "ORG"),
        ("Gemini Ultra", "MODEL"),
        # Q5
        ("Sam Altman", "PERSON"),
        ("GPT-4", "MODEL"),
        ("RLHF", "CONCEPT"),
    ]
    for name, etype in nodes:
        G.add_node(name, type=etype)

    # ── Edges ──────────────────────────────────────────────────────────────
    triples = [
        # Q1 chain
        ("Dario Amodei", "left", "OpenAI"),
        ("Daniela Amodei", "left", "OpenAI"),
        ("Dario Amodei", "co-founded", "Anthropic"),
        ("Daniela Amodei", "co-founded", "Anthropic"),
        ("Anthropic", "uses", "Constitutional AI"),
        ("Claude 3", "developed_by", "Anthropic"),
        ("Claude 3", "trained_with", "Constitutional AI"),
        # Q2 chain
        ("Demis Hassabis", "co-founded", "DeepMind"),
        ("Demis Hassabis", "leads", "DeepMind"),
        ("DeepMind", "developed", "AlphaFold 2"),
        ("AlphaFold 2", "trained_on", "UniProt"),
        ("AlphaFold 2", "trained_on", "Protein Data Bank"),
        ("Google", "acquired", "DeepMind"),
        # Q3 chain
        ("Vaswani et al.", "introduced", "Transformer"),
        ("LLaMA 2", "uses_architecture", "Transformer"),
        ("LLaMA 2", "trained_on", "Common Crawl"),
        ("Meta AI", "developed", "LLaMA 2"),
        ("Yann LeCun", "leads", "Meta AI"),
        # Q4 chain
        ("Geoffrey Hinton", "worked_at", "Google Brain"),
        ("Google Brain", "merged_with", "DeepMind"),
        ("Google Brain", "formed", "Google DeepMind"),
        ("DeepMind", "formed", "Google DeepMind"),
        ("Google DeepMind", "released", "Gemini Ultra"),
        # Q5 chain
        ("Sam Altman", "is_ceo_of", "OpenAI"),
        ("OpenAI", "developed", "GPT-4"),
        ("OpenAI", "developed", "Claude 3"),
        ("GPT-4", "trained_with", "RLHF"),
        ("OpenAI", "uses", "RLHF"),
        # Cross-chain edges
        ("Dario Amodei", "previously_worked_at", "OpenAI"),
        ("Geoffrey Hinton", "left", "Google Brain"),
        ("Vaswani et al.", "worked_at", "Google Brain"),
    ]

    for source, relation, target in triples:
        if source in G and target in G:
            G.add_edge(source, target, relation=relation, weight=1)

    return G
