# 03 — RAG

## Objective

This module covers retrieval-augmented generation from naive baseline through production-grade hybrid pipelines. After completing this module, you will be able to build a naive RAG system, upgrade it with BM25 + dense hybrid search and reciprocal rank fusion, apply HyDE and multi-query retrieval for recall improvement, integrate Cohere reranking, and implement agentic RAG where the LLM decides when and how to retrieve.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| Naive RAG | Embed query → top-k cosine search → stuff context into prompt → generate | The baseline every system starts from; establishes the quality floor to beat with measured deltas |
| Hybrid search | Combining dense vector retrieval with sparse BM25 lexical matching via RRF fusion | Catches exact keyword matches (SKUs, error codes, names) that pure semantic search misses |
| HyDE | Generating a hypothetical answer document, embedding it, and using that vector for retrieval | Bridges the vocabulary gap between short queries and long corpus passages |
| Multi-query retrieval | LLM generates query variants, retrieves for each, deduplicates results | Improves recall on ambiguous queries without manual query engineering |
| Agentic RAG | LLM agent decides whether to retrieve, reformulates queries, and iterates until satisfied | Handles multi-hop questions that single-shot retrieval cannot answer |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| `naive_rag.py` | Baseline embed-query-retrieve-generate pipeline with RAGAS evaluation | 🔲 Pending |
| `hybrid_rag.py` | BM25 + dense search with RRF fusion and Cohere cross-encoder reranking | 🔲 Pending |
| `hyde_retrieval.py` | Hypothetical Document Embedding query transformation before retrieval | 🔲 Pending |
| `multi_query_retrieval.py` | LLM-generated query variants with result deduplication and fusion | 🔲 Pending |
| `agentic_rag.py` | ReAct-style agent that iteratively retrieves, evaluates, and reformulates | 🔲 Pending |

## Key decisions & tradeoffs

- Hybrid search with RRF (k=60) will replace pure dense retrieval as the default after measuring a ≥15% context recall improvement on the evaluation set.
- Cohere Rerank v3 will be applied to the top-20 fused candidates, returning top-5 to the LLM—trading 200ms latency for measurable precision gains.
- HyDE will be enabled only for queries under 10 words where vocabulary mismatch is highest; longer queries will use direct embedding to avoid hallucinated document drift.
- Multi-query retrieval will generate 3 variants maximum to control cost; diminishing returns beyond 3 variants will be measured and documented.
- Agentic RAG will cap retrieval iterations at 3 to prevent runaway token costs, with a fallback to single-shot retrieval on timeout.
- Configure `QDRANT_URL` (default `http://localhost:6333`), `QDRANT_API_KEY` (blank for local Docker), and `COHERE_API_KEY` via the root `.env.example` before running hybrid retrieval pipelines.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| RAGAS faithfulness | ≥ 0.85 | |
| RAGAS context precision | ≥ 0.80 | |
| RAGAS context recall | ≥ 0.75 | |
| Retrieval MRR@10 (hybrid vs. naive) | ≥ +15% lift | |
| End-to-end latency p95 | ≤ 3s | |

## References

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
- Es, S., et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217. https://arxiv.org/abs/2309.15217
- RAGAS documentation: https://docs.ragas.io/
- Gao, Y., et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997. https://arxiv.org/abs/2312.10997
- Gao, L., et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE). arXiv:2212.10496. https://arxiv.org/abs/2212.10496
- Cormack, G. V., et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR 2009. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Cohere Rerank documentation: https://docs.cohere.com/docs/rerank
- Qdrant Hybrid Search documentation: https://qdrant.tech/documentation/concepts/hybrid-queries/
