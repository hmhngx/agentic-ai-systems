# 01 — Embeddings

## Objective

This module covers the mathematical foundations and practical implementation of text embeddings—the atomic unit of every retrieval system. After completing this module, you will be able to select embedding models based on measured retrieval quality, implement cosine similarity search from scratch, compare model families (Voyage-3, text-embedding-3-large, BGE-M3) on your own corpus, and apply matryoshka and late chunking techniques to optimize latency without sacrificing recall.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| Vector embeddings | Fixed-dimension dense vectors where semantic similarity maps to geometric proximity in ℝⁿ | Every retrieval pipeline quality ceiling is set by embedding geometry—bad embeddings cannot be fixed downstream |
| Cosine similarity | Normalized dot product measuring the angle between two vectors, invariant to magnitude | Standard retrieval metric; magnitude-invariant comparison across differently scaled embedding models |
| Matryoshka embeddings | Truncating high-dimensional embeddings to lower dimensions with minimal recall loss | Enables tiered retrieval: fast coarse search on 256-dim, precise rerank on full 1024-dim |
| Late chunking | Embedding the full document first, then pooling token-level vectors per chunk boundary | Preserves cross-chunk context that pre-chunk embedding destroys—critical for long documents |
| Model selection | Benchmarking multiple embedding APIs on domain-specific queries with MRR and nDCG | Public leaderboard scores on MS MARCO do not predict performance on legal, medical, or code corpora |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| `embedder.py` | Unified embedding interface supporting Voyage-3, OpenAI, and BGE-M3 with batching and retry logic | 🔲 Pending |
| `compare_models.py` | Side-by-side model comparison on a fixed query-document set with MRR and rank correlation | 🔲 Pending |
| `similarity_benchmark.py` | Cosine similarity search from scratch using NumPy, validated against library implementations | 🔲 Pending |
| `matryoshka_demo.py` | Truncation experiment measuring recall at 256, 512, and 1024 dimensions | 🔲 Pending |
| `late_chunking_demo.py` | Late chunking pipeline vs. pre-chunk embedding with measured context preservation delta | 🔲 Pending |

## Key decisions & tradeoffs

- Voyage-3 will be chosen as the primary embedding model for code and technical documents based on domain-specific MRR benchmarks against OpenAI and BGE-M3.
- Matryoshka truncation at 512 dimensions will be adopted as the default retrieval dimension, reserving full 1024-dim vectors for reranking only—trading 3% recall for 40% latency reduction.
- Late chunking will be implemented for documents exceeding 8K tokens where cross-paragraph context is critical; fixed pre-chunk embedding will remain the default for shorter documents to avoid pooling overhead.
- Batch size will be capped at 128 texts per API call to balance throughput against rate-limit headroom on cloud embedding endpoints.
- Normalization (L2) will be applied to all vectors before storage to ensure cosine similarity is equivalent to dot product at query time.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| Retrieval MRR@10 | ≥ 0.85 | |
| Cosine rank correlation (vs. full-dim) | ≥ 0.95 | |
| Embedding latency p95 (per 1K tokens) | ≤ 200ms | |
| Matryoshka recall@10 at 512-dim (vs. full) | ≥ 0.97 | |

## References

- Chen, J., et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216. https://arxiv.org/abs/2402.03216
- Kusupati, A., et al. (2022). *Matryoshka Representation Learning*. NeurIPS 2022. https://arxiv.org/abs/2205.13147
- Günther, M., et al. (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. arXiv:2409.04701. https://arxiv.org/abs/2409.04701
- OpenAI Embeddings documentation: https://platform.openai.com/docs/guides/embeddings
- Voyage AI Embeddings documentation: https://docs.voyageai.com/docs/embeddings
