# 03.2 - Naive RAG (PDF + Qdrant + Claude)

Day 4 module implementing a framework-free naive RAG pipeline over a single PDF:
embed query -> retrieve top-k chunks from Qdrant -> prompt Claude with context only ->
return answer plus page-level citations.

- **LLM:** Anthropic `claude-sonnet-4-20250514`
- **Embeddings:** Voyage `voyage-3` (1024-d)
- **Vector DB:** Qdrant (`COSINE`, HNSW configured, exact auto-fallback for small corpora)
- **No orchestration frameworks:** no LangChain, no LlamaIndex

## Folder layout

```text
3.2-NaiveRAG/
├── naive_rag.py
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── pdf_loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── generator.py
│   └── citation.py
└── tests/
```

## Prerequisites

- Docker Desktop (Qdrant on `localhost:6333`)
- Python 3.11+
- `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY`

## Setup (Windows PowerShell)

```powershell
cd "03-rag\3.2-NaiveRAG"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Copy-Item ".env.example" ".env"
# Fill ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env

docker run -d --name qdrant_naive_rag --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
curl.exe -sf http://localhost:6333/healthz
```

Stop DB: `docker stop qdrant_naive_rag`

## Usage

```powershell
# Ingest + interactive REPL
python naive_rag.py --pdf path\to\document.pdf

# Force re-ingest
python naive_rag.py --pdf path\to\document.pdf --reingest

# One-shot query
python naive_rag.py --pdf path\to\document.pdf --query "What is the main claim?"

# Debug retrieval traces
python naive_rag.py --pdf path\to\document.pdf --debug
```

`--top-k` is intentionally restricted to `3`, `4`, or `5`.

## Pipeline summary

1. `src/pdf_loader.py`: extract text, sentence-aware chunking (`300` tokens with `30` overlap).
2. `src/embedder.py`: embed chunks (`input_type="document"`) and query (`input_type="query"`), L2-normalized.
3. `src/vector_store.py`: create/use Qdrant collection, upsert chunk payloads with page metadata.
4. `src/retriever.py`: retrieve top-k, apply score filter (`MIN_SCORE_THRESHOLD = 0.40`), deduplicate overlaps.
5. `src/generator.py`: build strict context-only prompt, call Claude with `temperature = 0`.
6. `src/citation.py`: validate `[Doc N]` references and always print `Sources consulted`.

If retrieval yields no usable chunks, generation is skipped to avoid empty-context hallucinations.

## Tests

```powershell
# All tests
python -m pytest tests\ -v

# Skip slow/integration tests
python -m pytest tests\ -v -m "not slow and not integration"
```

## Known limits (intentional for naive baseline)

- No sparse/BM25 hybrid retrieval
- No reranker
- No multi-hop query planning
- No conversation memory

These are addressed in later modules.
