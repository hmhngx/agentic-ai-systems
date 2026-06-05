# 06 — Graph RAG

## Objective

This module covers knowledge-graph-augmented retrieval—entity extraction, graph construction, community summarization, and hybrid vector-plus-graph search following the Microsoft GraphRAG architecture. After completing this module, you will be able to extract entities and relationships from unstructured text, build and query knowledge graphs with NetworkX, generate community-level summaries for global queries, and combine graph traversal with vector search for questions that require multi-hop reasoning.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| Entity extraction | LLM identifies named entities and typed relationships from text chunks | Converts unstructured prose into structured graph edges queryable by traversal |
| Knowledge graph | Directed graph G=(V,E) where nodes are entities and edges are relationships | Enables multi-hop reasoning that flat vector search cannot perform ("Who reports to the CEO of X?") |
| Community detection | Clustering densely connected subgraphs and summarizing each cluster | Answers global/summary questions ("What are the main themes?") that local chunk retrieval misses |
| Hybrid retrieval | Combining vector similarity on chunk text with graph traversal on entity neighborhoods | Local queries use vectors; global queries use community summaries; relational queries use graph paths |
| GraphRAG pipeline | Index-time graph construction + query-time routing between local, global, and DRIFT search modes | Microsoft's production architecture for corpus-level reasoning at scale |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| `entity_extractor.py` | LLM-powered entity and relationship extraction with Pydantic schema validation | ✅ Done — see [`6.1-entity-extraction/`](6.1-entity-extraction/) |
| `graph_builder.py` | NetworkX graph construction, deduplication, and persistence from extracted triples | 🔲 Pending |
| `community_summaries.py` | Leiden clustering and LLM-generated community-level summaries | 🔲 Pending |
| `hybrid_graph_retriever.py` | Query router selecting vector, graph traversal, or community summary retrieval | 🔲 Pending |
| `graphrag_pipeline.py` | End-to-end GraphRAG index and query pipeline following Microsoft architecture | 🔲 Pending |

## Key decisions & tradeoffs

- NetworkX will be used for graph storage and traversal in development; the architecture will document Neo4j migration path for production scale beyond 100K entities.
- Entity extraction will use structured Pydantic output with typed relationships (WORKS_AT, REPORTS_TO, LOCATED_IN) rather than free-text triples—enabling typed graph queries.
- Leiden algorithm (via graspologic) will be chosen over Louvain for community detection due to better handling of disconnected components.
- Community summaries will be generated at index time and cached; query-time community selection will use embedding similarity against the query.
- Hybrid retrieval will default to vector search for entity-specific queries and community summaries for thematic/global queries, with a classifier routing between modes.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| Graph retrieval recall@5 (multi-hop queries) | ≥ 0.75 | |
| Hybrid lift vs. vector-only (multi-hop subset) | ≥ +10% | |
| Community summary relevance (LLM-judged) | ≥ 0.80 | |
| Index construction time (1K documents) | ≤ 30min | |

## References

- Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130. https://arxiv.org/abs/2404.16130
- Microsoft GraphRAG repository: https://github.com/microsoft/graphrag
- NetworkX documentation: https://networkx.org/documentation/stable/
- Traag, V. A., et al. (2019). *From Louvain to Leiden: guaranteeing well-connected communities*. Scientific Reports. https://www.nature.com/articles/s41598-019-41665-z
- LlamaIndex Knowledge Graph Index: https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/
