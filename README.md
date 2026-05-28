# Agentic AI Systems

I built this repository as a structured, implementation-first learning record documenting my mastery of production-grade AI engineering. Every module pairs working code with measurable evaluation—covering embeddings, chunking, RAG pipelines, agents, MCP, Graph RAG, OCR document pipelines, hallucination reduction, guardrails, and agentic system design. This is not a tutorial collection; it is an engineering portfolio with RAGAS scores, architectural tradeoffs, and deployable reference implementations.

## Tech Stack

| Tool | Purpose | Used in |
|------|---------|---------|
| Python | Core ML/RAG pipelines, agents, evaluation | 02–08, research-agent, capstone |
| TypeScript | Local-first doc agent, MCP server | doc-agent |
| OpenRouter (OpenAI-compatible) | LLM inference + embeddings via a single API key | 03-rag, doc-agent, research-agent, capstone |
| LangGraph | Stateful agent workflows, checkpointing, multi-agent routing | 05-agentic-systems, research-agent |
| Qdrant | Production vector store with hybrid search | 03-rag, capstone |
| sqlite-vec | Embedded vector search for local-first agents | doc-agent |
| Docling | Document parsing, layout-aware extraction | doc-agent, 03-rag |
| RAGAS | RAG evaluation metrics (faithfulness, precision, recall) | 08-evaluation, all modules |
| Pydantic | Schema validation, structured LLM outputs | 04-agents-mcp, 07-guardrails, research-agent |
| FastAPI | REST API backend for production RAG serving | capstone |
| MCP SDK | Model Context Protocol server/client implementation | 04-agents-mcp, doc-agent |
| NetworkX | Knowledge graph construction and traversal | 06-graph-rag |
| BM25 (rank_bm25) | Sparse lexical retrieval for hybrid search | 03-rag, capstone |
| Cohere Rerank | Cross-encoder reranking for retrieval quality | 03-rag, capstone |

## Repository Structure

```
agentic-ai-systems/
├── .github/                    # GitHub templates and CI configuration
├── 01-embeddings/              # Vector math, embedding models, similarity search
├── 02-chunking/                # Chunking strategies, overlap, parent-child retrieval
├── 03-rag/                     # Naive RAG through hybrid search, HyDE, agentic RAG
├── 04-agents-mcp/              # ReAct agents, tool schemas, MCP architecture
├── 05-agentic-systems/       # LangGraph workflows, checkpointing, multi-agent routing
├── 06-graph-rag/               # Entity extraction, knowledge graphs, GraphRAG
├── 07-guardrails/              # Input/output validation, PII detection, topic restriction
├── 08-evaluation/              # RAGAS metrics, LLM-as-judge, hallucination measurement
├── projects/
│   ├── doc-agent/              # Local-first document understanding agent (TypeScript)
│   ├── research-agent/         # Autonomous research system (LangGraph)
│   └── capstone/               # Full production RAG application with UI and Docker
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```

## Learning Philosophy

- Every module ships with a RAGAS evaluation dataset. If it cannot be measured, it does not ship.
- I implement the naive version first, benchmark it, then add complexity only when metrics justify the cost.
- Architectural decisions are documented as tradeoffs with measured deltas—not as preferences or opinions.
- Local-first by default: sqlite-vec and on-device inference before cloud vector stores, unless scale demands otherwise.
- Guardrails and evaluation are first-class modules, not afterthoughts bolted onto a demo.

## Progress

| Module | Status | Key artifact | RAGAS score |
|--------|--------|--------------|-------------|
| 01-embeddings | 🔲 Not started | | |
| 02-chunking | 🔲 Not started | | |
| 03-rag | 🔲 Not started | | |
| 04-agents-mcp | 🔲 Not started | | |
| 05-agentic-systems | 🔲 Not started | | |
| 06-graph-rag | 🔲 Not started | | |
| 07-guardrails | 🔲 Not started | | |
| 08-evaluation | 🔲 Not started | | |
| doc-agent | 🔲 Not started | | |
| research-agent | 🔲 Not started | | |
| capstone | 🔲 Not started | | |

## Environment

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` (and other keys per module) before running any project or pipeline.

## Contact

- LinkedIn: [Your Name](https://linkedin.com/in/your-profile-placeholder)
- GitHub: [Your Name](https://github.com/your-username-placeholder)
