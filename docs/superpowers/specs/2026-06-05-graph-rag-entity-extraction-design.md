# Design — Graph RAG: Entity Extraction (Day 6.1)

**Date:** 2026-06-05
**Module:** `06-graph-rag/6.1-entity-extraction/`
**Status:** Approved (design), pending implementation

## 1. Goal

Build `entity_extractor.py`: takes a document, extracts entities and
`(entity, relation, entity)` triples, builds a NetworkX knowledge graph
(nodes = entities, edges = relations), saves the graph to disk, stores entity
embeddings in Qdrant, visualizes the graph with matplotlib showing clear entity
clusters, and prints the top-5 most-connected entities.

This is **Day 6.1 only** — the first artifact of the `06-graph-rag` module.
`graph_builder.py`, `community_summaries.py`, `hybrid_graph_retriever.py`, and
`graphrag_pipeline.py` are explicitly out of scope for this day.

### Source document

`C:\Users\minhh\Downloads\Lost in the Middle.pdf` — Liu et al. (2023),
"Lost in the Middle: How Language Models Use Long Contexts". 18 pages,
~65K chars (~16K tokens). Entity-rich with natural clusters: authors, institutions
(Stanford, UC Berkeley, Samaya AI), models (GPT-3.5-Turbo, Claude, MPT-30B,
LongChat), datasets (NaturalQuestions), and concepts (multi-document QA, key-value
retrieval, positional bias). These groupings drive a visually clustered graph.

### Learning concepts the README must answer

- WHY a graph beats vanilla RAG on multi-hop questions
- What a "community summary" is and why it helps
- Trade-off: graph construction cost vs query quality
- When to use Graph RAG vs. stick with vanilla
- How entity disambiguation works in practice

## 2. Architecture

Self-contained day folder matching the repo convention (see `05-agentic-systems/5.3`,
`03-rag/3.4-OCR`): heavily-documented `src/` modules + a top-level CLI orchestrator
+ `tests/` + `README.md` + `requirements.txt` + `.env.example` + `.gitignore` +
`conftest.py`.

```
6.1-entity-extraction/
├── entity_extractor.py        # top-level CLI orchestrator (the deliverable)
├── src/
│   ├── config.py              # env resolution, USE_LLM / key gating (mirrors 5.3)
│   ├── loader.py              # PDF/.txt → pages[] + sentence segmentation (pymupdf)
│   ├── schema.py              # Pydantic: Entity, Triple, ExtractionResult (typed labels)
│   ├── extractor_offline.py   # DEFAULT: spaCy NER + dependency-SVO + co-occurrence
│   ├── extractor_llm.py       # opt-in: OpenRouter typed-triple extraction (prompt lives here)
│   ├── extractor.py           # façade: routes offline vs LLM by config
│   ├── disambiguation.py      # canonicalize + alias-merge + rapidfuzz fuzzy resolution
│   ├── graph.py               # NetworkX build / save (GraphML+JSON) / load / top-N
│   ├── embeddings.py          # entity embeddings: OpenRouter if key, else deterministic offline
│   ├── vector_store.py        # Qdrant (local Docker) create/upsert/search entity points
│   ├── visualize.py           # matplotlib spring layout, color by community, size by degree
│   └── report.py              # top-5 most-connected, pretty table
├── tests/
├── conftest.py · requirements.txt · .env.example · .gitignore
└── README.md
```

### Decisions

- **Offline-by-default, opt-in LLM** (matches every prior day). Tests and grading
  are hermetic. The offline extractor is spaCy NER (nodes) + dependency
  subject–verb–object parse + intra-sentence co-occurrence (edges, labeled by the
  connecting verb, evidence = sentence). `USE_LLM=1` + a key swaps in OpenRouter
  typed-triple extraction. Topology of the pipeline is identical either way.
- **Typed entities, not free-text triples**: `type ∈ {PERSON, ORG, LOCATION,
  MODEL, DATASET, CONCEPT, WORK}`. Enables type-aware disambiguation and queryable
  edges (consistent with the module README's stated intent).
- **Local-Docker Qdrant** at `localhost:6333` (matches days 3.x); clear actionable
  error if the server is down.
- **Embeddings**: OpenRouter `text-embedding-3-small` when `OPENROUTER_API_KEY`
  is present; otherwise a **deterministic offline embedding** (clearly labeled
  non-semantic) so the full pipeline and tests run with no key. This is the one
  deviation from the "fail loudly" embedder pattern (3.4), chosen so the
  educational pipeline is runnable end-to-end offline.
- **Community detection** via NetworkX built-in `community.louvain_communities`
  (no extra dependency); drives node coloring → "clear clusters".

## 3. Data flow

```
load(doc)
  → extract(pages) → ExtractionResult{entities, triples}
  → disambiguate()  (merge aliases / surface forms, type-aware)
  → build_graph()   (NetworkX MultiDiGraph; nodes=entities, edges=relations)
  → save to disk    (graph.graphml + graph.json node-link)
  → embed_entities() → upsert to Qdrant (skippable)
  → visualize()     (graph.png, colored by community) (skippable)
  → report top-5 most-connected
```

CLI flags: `--doc PATH`, `--out-dir DIR`, `--no-qdrant`, `--no-viz`, `--top N`
(default 5).

## 4. Component contracts

| Module | Responsibility | Key interface | Depends on |
|--------|----------------|---------------|-----------|
| `config.py` | env resolution, gating | `use_llm()`, `use_embeddings_api()`, model/url constants | os |
| `loader.py` | doc → text | `load_pages(path) -> list[Page]`, `iter_sentences(pages)` | pymupdf |
| `schema.py` | typed data models | `Entity`, `Triple`, `ExtractionResult` | pydantic |
| `extractor_offline.py` | deterministic extraction | `extract_offline(pages) -> ExtractionResult` | spacy |
| `extractor_llm.py` | LLM extraction + prompt | `extract_llm(pages) -> ExtractionResult` | openai, schema |
| `extractor.py` | route offline/LLM | `extract(pages) -> ExtractionResult` | config, the two extractors |
| `disambiguation.py` | entity resolution | `resolve(result) -> ExtractionResult` | rapidfuzz |
| `graph.py` | build/save/load/top-N | `build_graph`, `save_graph`, `load_graph`, `top_connected` | networkx |
| `embeddings.py` | entity vectors | `embed_entities(entities) -> np.ndarray`, `EMBEDDING_DIM` | openai/numpy |
| `vector_store.py` | Qdrant ops | `get_client`, `create_collection`, `upsert_entities`, `search` | qdrant-client |
| `visualize.py` | render PNG | `draw_graph(G, path)` | matplotlib, networkx |
| `report.py` | top-5 table | `format_top_connected(G, n)` | networkx |

Each module has one clear purpose, a small documented interface, and is testable
in isolation.

## 5. Disambiguation (made concrete)

1. **Normalize** surface forms (case, whitespace, trailing punctuation/footnote marks).
2. **Block** candidate pairs by entity type (never merge across types).
3. **Score** pairs with `rapidfuzz` token-set ratio + acronym/alias seeds
   (e.g. "UC Berkeley" ≡ "University of California, Berkeley";
   "GPT-3.5-Turbo" ≡ "gpt-3.5").
4. **Cluster** matches above threshold via union-find.
5. **Canonicalize**: pick most-frequent / longest surface form per cluster; record
   the rest as `aliases`.

README maps this to production entity resolution: blocking → candidate generation
→ pairwise scoring → clustering.

## 6. Graph & visualization

- `MultiDiGraph` preserves every relation (parallel edges for multiple relations
  between the same pair).
- For clustering/viz, project to an undirected weighted `Graph`
  (weight = number of relations) and run `louvain_communities`.
- Nodes colored by community (the "clear clusters"), sized by degree; type legend.
- Top-5 = highest-degree nodes. Saved as `graph.png`.

## 7. Embeddings + Qdrant

- Entity embedding text = `name + type + aliases + representative evidence sentence`.
- `text-embedding-3-small` (1536-dim) via OpenRouter when key present; else a
  deterministic hashed embedding (stable across runs, labeled non-semantic).
- Local-Docker Qdrant, COSINE HNSW; payload `{name, type, degree, aliases, mentions}`.

## 8. Tests & verification

pytest goal-tests, all offline/deterministic:

- **G1** extraction returns non-empty entities + triples
- **G2** graph nodes == entities and edges == relations
- **G3** graph saves and reloads identically (round-trip)
- **G4** top-5 most-connected computed correctly (degree order)
- **G5** disambiguation merges a known alias pair into one canonical node
- **G6** visualization file is produced
- **Determinism** offline run is reproducible across two invocations

Qdrant tests skip unless a server is reachable; LLM tests skip unless a key is set.

## 9. Dependencies

`pymupdf, spacy (+en_core_web_sm), networkx, matplotlib, numpy, qdrant-client,
openai, pydantic, rapidfuzz, python-dotenv, pytest`. Community detection uses the
NetworkX built-in Louvain (no extra dependency).
