# Capstone — Production RAG Application

## Overview

The capstone project is a full production RAG application integrating all eight modules—`01-embeddings`, `02-chunking`, `03-rag`, `04-agents-mcp`, `05-agentic-systems`, `06-graph-rag`, `07-guardrails`, and `08-evaluation`—with hybrid retrieval and reranking, guardrail enforcement, RAGAS evaluation, and Docker-based deployment. It serves a Streamlit chat UI backed by a FastAPI API, designed as the portfolio centerpiece demonstrating end-to-end AI systems engineering from ingestion to monitored production serving.

## Architecture

```
[User browser]
       |
       v
[Streamlit UI] ---> HTTP ---> [FastAPI Backend]
                                    |
                    +---------------+---------------+
                    |               |               |
                    v               v               v
            [Guardrail Layer] [RAG Pipeline]  [LangSmith Tracer]
                    |               |
                    |       +-------+-------+
                    |       |               |
                    |       v               v
                    |  [Hybrid Retriever] [LLM Generator]
                    |       |
                    |   +---+---+---+
                    |   |       |   |
                    |   v       v   v
                    | [Qdrant] [BM25] [Cohere Rerank]
                    |
                    v
            [Block / Allow / Redact]
                    |
                    v
            [Response to user]

[Docker Compose]
  ├── fastapi service
  ├── streamlit service
  ├── qdrant service
  └── (optional) langsmith
```

## Tech stack

| Layer | Tool | Why this tool |
|-------|------|---------------|
| Frontend | Streamlit | Rapid chat UI prototyping with session state and file upload |
| API | FastAPI | Async REST API with OpenAPI docs and Pydantic request validation |
| Vector store | Qdrant | Production hybrid search with sparse + dense vectors |
| Sparse retrieval | rank_bm25 | In-memory BM25 index for lexical matching |
| Reranking | Cohere Rerank v3 | Cross-encoder precision on fused candidates |
| Embeddings | Voyage-3 | Domain-tuned retrieval quality from module 01 |
| Chunking | Semantic + parent-child | Best measured chunking strategy from module 02 |
| Guardrails | Custom Pydantic + Presidio | Input/output validation and PII redaction from module 07 |
| Evaluation | RAGAS + LangSmith | Automated quality metrics and production tracing from module 08 |
| Deployment | Docker Compose | Reproducible multi-service local and cloud deployment |

## Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | FastAPI backend with document upload and ingestion pipeline | 🔲 Pending |
| M2 | Hybrid retrieval (Qdrant dense + BM25 + RRF + Cohere rerank) | 🔲 Pending |
| M3 | Guardrail layer (input classifier, PII detection, output validator) | 🔲 Pending |
| M4 | Streamlit chat UI with conversation history | 🔲 Pending |
| M5 | RAGAS evaluation suite with CI-integrated scoring | 🔲 Pending |
| M6 | Docker Compose deployment with health checks | 🔲 Pending |

## How to run

```powershell
cd projects\capstone
copy ..\..\.env.example .env
docker compose up --build -d
```

Open the UI:

```powershell
start http://localhost:8501
```

Run evaluation:

```powershell
docker compose exec fastapi python -m evaluation.ragas_eval
```

Tear down:

```powershell
docker compose down -v
```

## Evaluation targets

| Metric | Target |
|--------|--------|
| RAGAS faithfulness | ≥ 0.85 |
| RAGAS answer relevancy | ≥ 0.80 |
| RAGAS context precision | ≥ 0.80 |
| RAGAS context recall | ≥ 0.75 |
| PII leak rate | 0% |
| API latency p95 | ≤ 3s |
| Uptime (Docker health checks) | ≥ 99.9% |

## Known limitations

- Single-tenant deployment; no multi-user authentication or per-user document isolation in this POC.
- Qdrant runs as a single-node instance; no clustering or sharding for large corpora beyond 100K chunks.
- Streamlit UI is functional but not production-polished; a React frontend would be the next iteration for public deployment.
