# 02 — Chunking

## What this module shows

This folder benchmarks four chunking strategies on the same PDF text and scores each run with **ICC (Intrachunk Cohesion)**—mean pairwise cosine similarity between sentences inside a chunk. Higher ICC usually means chunks stay on one topic; the CLI prints token stats and picks the best strategy for the sample document.

| Strategy | How it splits | Defaults |
|----------|---------------|----------|
| **naive** | Fixed character windows | 1000 chars, no overlap |
| **recursive** | LangChain hierarchy (`\n\n` → `\n` → space) | 1000 chars, 100-char overlap |
| **sentence** | spaCy sentences, merged up to a char budget | ~1000 chars per chunk |
| **semantic** | Embedding breakpoints between sentences (adjacent sim below 0.4), then char merge | ~1000 chars per chunk |

## Results
![Chunking strategies ICC results](02-chunking/results/image.png)

## Layout

```
02-chunking/
├── benchmark.py          # CLI: load PDF → chunk → evaluate → table + winner
├── sample_doc.pdf        # Multi-chapter sample (regenerate with script below)
├── src/
│   ├── chunkers.py       # naive, recursive, sentence, semantic
│   ├── evaluator.py      # ICC + token statistics
│   └── pdf_loader.py     # PyMuPDF text extraction
├── scripts/
│   └── generate_sample_pdf.py
├── tests/
├── requirements.txt
├── .gitignore
├── Makefile              # Unix/Git Bash shortcuts (CI uses this)
└── pytest.ini
```

## Install

From this directory (venv recommended):

```powershell
cd "02-chunking"
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first semantic run downloads `all-MiniLM-L6-v2` via `sentence-transformers` unless models are already cached. `benchmark.py` sets `HF_HUB_OFFLINE=1` by default for quieter runs; remove those env vars if you need a fresh Hub download.

## Run the benchmark

```powershell
python benchmark.py --pdf_path sample_doc.pdf
```

Example output columns: `Strategy`, `Total Chunks`, `Avg Tokens`, `Min Tokens`, `Max Tokens`, `Std Tokens`, `ICC Score`, plus a line naming the best ICC strategy.

Regenerate the sample PDF:

```powershell
python scripts/generate_sample_pdf.py
```

## Tests

```powershell
# Fast suite (skips @pytest.mark.slow semantic/model tests)
python -m pytest tests/ -v -m "not slow"

# Full suite
python -m pytest tests/ -v

# Coverage gate (≥80% on src/)
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80
```

On WSL or Git Bash you can use `make install`, `make test-fast`, `make benchmark`, etc. from the `Makefile`.

GitHub Actions (`.github/workflows/ci.yml`) runs lint, fast tests, and an 80% coverage gate on pushes that touch `02-chunking/`.

## Concepts

| Concept | In this code | Why it matters |
|---------|--------------|----------------|
| Fixed-size chunking | `naive_chunk`, `recursive_chunk` | Fast and predictable; can cut mid-thought without overlap or hierarchy |
| Sentence boundaries | `sentence_chunk` + spaCy | Avoids splitting mid-sentence when merging to a size budget |
| Semantic boundaries | `semantic_chunk` + MiniLM similarities | Groups sentences that embed similarly before size caps |
| ICC | `evaluate_chunks` samples up to 30 chunks | Cheap proxy for “does this chunk read like one topic?” without human labels |
| Overlap | `recursive_chunk` only (100 chars) | Helps answers that span naive window edges |

## References

- [LangChain text splitters](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Quivr semantic chunker](https://github.com/QuivrHQ/quivr/tree/main/core/quivr_core/semantic_chunker) (similar embedding-breakpoint idea)
