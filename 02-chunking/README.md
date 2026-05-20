# 02 — Chunking

## Objective

This module covers document segmentation strategies—the step that most silently destroys RAG quality when done wrong. After completing this module, you will be able to implement fixed-size, sentence-boundary, semantic, AST-based, and agentic chunking strategies; configure overlap for context continuity; and build parent-child retrieval indexes that return precise child chunks while providing broad parent context to the LLM.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| Fixed-size chunking | Splitting text every N tokens with a sliding window overlap | Fast and predictable, but cuts mid-sentence and mid-paragraph—acceptable for homogeneous prose |
| Semantic chunking | Splitting at embedding-detected topic boundaries where adjacent sentences diverge | Preserves topical coherence; chunks align with how humans structure arguments |
| AST-based chunking | Parsing source code or structured documents into syntactic units (functions, classes, sections) | Code RAG fails without respecting syntax boundaries—a 512-token window splits functions in half |
| Parent-child retrieval | Indexing small child chunks for precision, returning parent chunks for LLM context | Solves the precision-vs-context tradeoff: retrieve precisely, generate with breadth |
| Chunk overlap | Duplicating N tokens across adjacent chunk boundaries | Prevents answers that span chunk boundaries from being unreachable by any single retrieval hit |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| `fixed_chunker.py` | Token-based fixed-size chunker with configurable size and overlap | 🔲 Pending |
| `sentence_chunker.py` | Sentence-boundary-aware chunker using NLTK or spaCy sentence segmentation | 🔲 Pending |
| `semantic_chunker.py` | Embedding-based breakpoint detection using cosine distance thresholds | 🔲 Pending |
| `ast_chunker.py` | Tree-sitter AST parser for Python/TypeScript code chunking by function and class | 🔲 Pending |
| `agentic_chunker.py` | LLM-guided chunking that proposes boundaries based on document structure | 🔲 Pending |
| `parent_child_store.py` | Dual-index store linking child chunks to parent documents in Qdrant | 🔲 Pending |

## Key decisions & tradeoffs

- Semantic chunking will be chosen as the default for prose documents, with a cosine distance threshold of 0.25 tuned on a held-out evaluation set—fixed-size chunking reserved for speed-critical paths.
- A 512-token cap will be enforced on all semantic chunks to prevent oversized segments that dilute embedding signal, accepting variable chunk sizes below the cap.
- AST-based chunking via tree-sitter will be used exclusively for code files (.py, .ts, .js), never mixing code and prose chunkers in the same pipeline.
- Parent-child retrieval will use 128-token child chunks indexed for search and 1024-token parent chunks injected into the LLM prompt—trading storage cost for retrieval precision.
- 10% token overlap will be applied to all fixed-size chunks; semantic chunks will rely on boundary detection instead of overlap to avoid redundant storage.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| Chunk boundary F1 (vs. human-annotated) | ≥ 0.80 | |
| Parent-child retrieval recall@5 | ≥ 0.90 | |
| Context preservation score (answer spans chunk boundary) | ≥ 0.85 | |
| Chunking throughput (docs/minute) | ≥ 50 | |

## References

- LangChain Text Splitters documentation: https://python.langchain.com/docs/how_to/recursive_text_splitter/
- LlamaIndex Node Parser documentation: https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/
- Anthropic (2024). *Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
- tree-sitter documentation (AST-based chunking for code): https://tree-sitter.github.io/tree-sitter/
- Quivr semantic chunking implementation: https://github.com/QuivrHQ/quivr/tree/main/core/quivr_core/semantic_chunker
