# Doc Agent

## Overview

Doc Agent is a local-first document understanding system that indexes files on your machine and answers questions about them through an MCP-connected LLM agent. Built for developers and knowledge workers who need semantic search across PDFs, Markdown, and code without uploading documents to a cloud vector store. The system runs entirely on-device using sqlite-vec for vector storage and exposes its capabilities via the Model Context Protocol.

## Architecture

```
[Local filesystem]
       |
       v  (scan & watch)
[Folder Scanner] ---> [Docling Parser] ---> [AST Chunker]
                                                  |
                                                  v
                                          [Embedding API]
                                                  |
                                                  v
                                          [sqlite-vec DB]
                                                  |
                                                  v
                                          [MCP Server (stdio)]
                                                  |
                                                  v  (tool calls)
                                          [Claude Agent (Anthropic SDK)]
                                                  |
                                                  v
                                          [Answer to user]
```

## Tech stack

| Layer | Tool | Why this tool |
|-------|------|---------------|
| Runtime | Node.js + TypeScript | Type-safe MCP server with strong ecosystem for file I/O |
| Document parsing | Docling | Layout-aware PDF and Office extraction preserving structure |
| Chunking | tree-sitter (AST) | Syntax-aware code chunking; respects function and class boundaries |
| Embeddings | Voyage-3 via API | Highest measured MRR on technical document corpora |
| Vector store | sqlite-vec | Embedded vector search with zero external dependencies |
| Agent protocol | MCP SDK (TypeScript) | Standard interface compatible with Claude Desktop and Cursor |
| LLM | Anthropic SDK (Claude) | Tool use, long context, structured outputs |

## Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Folder scanner with file-type detection and incremental indexing | 🔲 Pending |
| M2 | Docling integration for PDF and Markdown parsing | 🔲 Pending |
| M3 | AST chunker for Python and TypeScript files | 🔲 Pending |
| M4 | sqlite-vec embedding storage and cosine search | 🔲 Pending |
| M5 | MCP server exposing search, read, and metadata tools | 🔲 Pending |
| M6 | Claude agent with multi-turn document Q&A | 🔲 Pending |

## How to run

```powershell
cd projects\doc-agent
copy ..\..\.env.example .env
# Set ANTHROPIC_API_KEY in .env
npm install
npm run build
npm run index -- --dir C:\path\to\your\documents
npm run mcp-server
```

In a separate terminal (Claude Desktop or custom client):

```powershell
npm run agent -- --query "What does the authentication module do?"
```

## Evaluation targets

| Metric | Target |
|--------|--------|
| Retrieval MRR@5 | ≥ 0.85 |
| Answer faithfulness (RAGAS) | ≥ 0.85 |
| Indexing throughput | ≥ 100 files/minute |
| End-to-end query latency p95 | ≤ 2s |

## Known limitations

- Indexes only files accessible on the local filesystem; no cloud storage connectors (S3, Google Drive) in this POC.
- Embedding requires an API call to Voyage-3; fully offline embedding is out of scope.
- AST chunking supports Python and TypeScript only; other languages fall back to fixed-size chunking.
