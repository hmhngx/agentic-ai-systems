# Design — `5.3-Agentic-RAG`: LangGraph Agentic RAG with a RAGAS-gated Reflexion Loop

- **Date:** 2026-06-04
- **Module:** `05-agentic-systems/5.3-Agentic-RAG/`
- **Context:** Week 2 Review — Agentic RAG. Capstone tying together RAG (module 03),
  agents/LangGraph (module 05), and evaluation (module 08).
- **Status:** Approved (design); spec under review.

## 1. Goal

Deliver `agentic_rag.py`: a LangGraph agent that uses RAG as a **tool it decides
whether to call**, then **self-corrects** when its answer is not grounded, with the
self-correction gated on a RAGAS-faithfulness score. The artifact must:

- Wire `rag_v1_complete` (the v1 RAG pipeline) as a tool in LangGraph.
- Let the agent decide per query: **retrieve** vs **answer from context** (direct).
- Add **Reflexion**: if RAGAS faithfulness `< 0.70`, refine the query and re-query.
- Cap at **2 retry attempts**, then **graceful fallback** (safe abstention).
- **Run a RAGAS eval on the full agentic pipeline.**

### Success criteria (machine-checked)

1. A compiled LangGraph `StateGraph` runs end-to-end and the RAG tool is registered
   as a LangGraph/LangChain tool.
2. The decision gate routes `retrieve` vs `direct` correctly on the eval dataset.
3. The Reflexion loop fires deterministically when faithfulness `< 0.70` and is
   bounded to **≤ 2 retries** (≤ 3 generate cycles), enforced in the router **and**
   by a `recursion_limit` backstop.
4. Graceful fallback returns the **best-scoring attempt** (not the last), as a safe
   abstention message.
5. **`ragas_eval.py` reports mean faithfulness ≥ 0.75** over served answers on
   `eval_dataset.json`, asserted in a test.
6. Runs **fully offline and deterministically by default** (no keys, no Docker, no
   network). `USE_LLM=1` / `USE_RAGAS=1` swap in real backends without changing
   topology.

## 2. Conventions inherited from the repo (non-negotiable)

- Self-contained day folder: `README.md`, runnable script, pinned
  `requirements.txt`, machine-checked `tests/`, `.env.example`, `.gitignore`.
- **Offline & deterministic by default**; optional real backends behind env flags.
- LangGraph `StateGraph` with typed `TypedDict` state; `Annotated[..., operator.add]`
  reducers for accumulating fields.
- Nodes are pure-ish functions `state -> partial-state dict`; **routers read state and
  return a key, never mutate**; **nodes never raise — they trap errors into the state
  bus** and let the router degrade to graceful failure.
- Compile with `MemorySaver()`; `thread_id`-isolated checkpointing.
- Force UTF-8 stdout/stderr (Windows cp1252 box-drawing fix).
- Each Day goal maps to an explicit test assertion; the source-note "common mistakes"
  get their own tests.

## 3. Graph topology

```
START → decide ──(direct)──→ direct_answer ─────────────────────────┐
          │                                                          │
       (retrieve)                                                    │
          ▼                                                          │
   retrieve_and_generate ◀───────────────┐                          │
          │                              │ (refined query)          │
          ▼                              │                          │
       evaluate  ── route_after_eval ──→ refine_query  (attempt<max)│
          │            │                                            │
          │            ├─ faithful (≥ 0.70) ───→ finalize ──────────┼─→ END
          │            └─ unfaithful & attempts≥max → fallback ─────┘
```

Nodes:

- **`decide`** — the "SHOULD I retrieve?" gate. Returns route `retrieve` (factual KB
  question) or `direct` (query carries its own context, or is meta/trivial/chitchat).
  Deterministic heuristic by default; LLM classifier under `USE_LLM`.
- **`retrieve_and_generate`** — calls the `rag_v1_complete` tool: retrieve → score
  threshold → grounded answer with `[Doc N]` citations. Overwrites `chunks`, `answer`,
  `citations` for the current attempt.
- **`evaluate`** — computes faithfulness of `answer` vs *its retrieved context*, plus a
  diagnosis (which claims are unsupported, the active failure mode). Writes
  `faithfulness` + `eval_report`; appends to `attempts_log`.
- **`refine_query`** (Reflexion) — reads accumulated `attempts_log` diagnoses and
  rewrites `query` to retrieve better-grounded context; increments `attempt`.
- **`direct_answer`** — answers without retrieval (from query-provided context or
  trivial). Terminal-ish → finalize/END.
- **`fallback`** — graceful safe abstention; selects and reports the **best-scoring
  attempt** across all tries.
- **`finalize`** — assembles `final_answer`, `served`, `status`, and the trace.

Routers (pure functions returning a key):

- **`route_decision`**: `state["route"]` → `{"retrieve", "direct"}`.
- **`route_after_eval`**:
  - `faithfulness ≥ threshold` → `finalize`
  - `faithfulness < threshold` **and** `attempt < max_retries` → `refine`
  - `faithfulness < threshold` **and** `attempt ≥ max_retries` → `fallback`
  - any node error in state → `fallback`

## 4. Two thresholds, reconciled

- **`faithfulness_threshold = 0.70`** — reflexion trigger / serve gate ("re-query if
  faithfulness < 0.7").
- **Dataset target `0.75`** — the **mean** faithfulness over served answers on the
  eval dataset (the headline goal). The loop raises per-query faithfulness so the
  served set averages ≥ 0.75. Both values are configurable; both documented in code
  and README.

## 5. State schema

```python
class AgenticRAGState(TypedDict):
    # input
    question: str
    max_retries: int                 # default 2
    faithfulness_threshold: float    # default 0.70

    # decision
    route: str                       # "retrieve" | "direct"
    decision_reason: str

    # working query (refined across attempts)
    query: str
    attempt: int                     # count of retrieve+generate cycles done

    # retrieval + generation (overwritten each attempt)
    chunks: list[dict]               # [{id:"Doc 1", text, score, ...}]
    answer: str
    citations: list[str]

    # evaluation (overwritten each attempt)
    faithfulness: float
    eval_report: dict                # {supported_ratio, unsupported_claims, failure_mode, ...}

    # reflexion memory (accumulates)
    attempts_log: Annotated[list[dict], operator.add]  # {query, answer, faithfulness, diagnosis}

    # terminal
    status: str                      # "answered" | "fallback" | "direct"
    final_answer: str
    served: bool

    # transcript
    log: Annotated[list[str], operator.add]
```

`initial_state(question, max_retries=2, faithfulness_threshold=0.70)` seeds reducer
fields to `[]`.

## 6. The `rag_v1_complete` tool (offline, in-memory)

Wired as **both** a plain callable `rag_v1_complete(query) -> {answer, chunks,
citations}` **and** a LangGraph/LangChain `@tool` wrapper (satisfies "wire as a tool").

- **Vector store:** Qdrant `:memory:` (`QdrantClient(":memory:")`) — no server, fully
  offline, deterministic.
- **Corpus** (`corpus.py`): a small bundled deterministic corpus on one coherent topic,
  **engineered so**:
  - a vague/underspecified initial query retrieves **distractor** chunks → low
    faithfulness → fires reflexion;
  - a refined query (with the key terms) retrieves the **right** chunk → high
    faithfulness → serves.
  This makes the self-correction loop deterministically demonstrable and testable with
  zero keys.
- **Embedder** (`embedder.py`): deterministic offline embedding (hashing/bag-of-words →
  fixed-dim vector) by default; optional real OpenRouter embeddings under `USE_LLM`.
- **Retriever** (`retriever.py`): embed → Qdrant `:memory:` search → score threshold →
  top-k chunks labelled `[Doc 1..N]`.
- **Generator** (`generator.py`): offline default = **deterministic extractive grounded
  generator** that emits only sentences supported by retrieved chunks with `[Doc N]`
  citations (so faithfulness is high when retrieval is good and low when retrieval is
  wrong — exactly what should trigger reflexion). Optional `USE_LLM` → real OpenRouter
  grounded generation reusing the Day-4 context-only system prompt.

## 7. Faithfulness scoring (`faithfulness.py`)

- **Offline proxy (default):** claim segmentation → per-claim support against retrieved
  chunks (token/lexical overlap with a support threshold) → `supported_ratio` in
  `[0,1]` = faithfulness. Returns score **and** diagnosis: the unsupported claims and
  the active **failure mode** (e.g. `NO_RETRIEVAL`, `WRONG_CHUNKS`, `UNSUPPORTED`,
  `UNCITED`), reusing the spirit of Day 10's grounding logic. The diagnosis feeds
  reflexion and the RAGAS table.
- **Real RAGAS (optional, `USE_RAGAS=1` + key):** swap in the `ragas` library's
  `faithfulness` (and, where cheap, `answer_relevancy` / `context_precision`). Same
  return contract so the graph is unchanged.

## 8. RAGAS eval (`ragas_eval.py` + `eval_dataset.json`)

- **`eval_dataset.json`** — a curated ~10-query set covering all four behaviors:
  1. retrieve → grounded answer on first try (faithful);
  2. retrieve → low faithfulness → reflexion refine → faithful (the headline loop);
  3. direct (query carries context / meta) → no retrieval;
  4. unanswerable from corpus → reflexion exhausts → graceful fallback.
  Each entry: `{id, question, expected_route, expects_reflexion?, expects_fallback?,
  ground_truth?}`.
- **`ragas_eval.py`** — runs the **whole agent** per query and prints a RAGAS-style
  table (faithfulness; plus context-precision/answer-relevancy in the proxy), with the
  headline **mean faithfulness over served answers**. A test asserts the mean
  **≥ 0.75**.
- **Honest caveat (in README):** this is a *day-scale* curated set (~10 queries), not
  the 50-pair production set module 08 mandates for final scoring.

## 9. File layout

```
05-agentic-systems/5.3-Agentic-RAG/
  agentic_rag.py          # entrypoint: build graph, 3-run demo, CLI, ASCII/Mermaid viz
  src/
    __init__.py
    state.py              # AgenticRAGState + initial_state
    corpus.py             # bundled deterministic corpus (+ distractors)
    embedder.py           # offline deterministic embedder + optional OpenRouter
    rag_tool.py           # rag_v1_complete callable + @tool wrapper (Qdrant :memory:)
    retriever.py          # embed → search → threshold → top-k
    generator.py          # grounded answer + [Doc N] citations (offline + optional LLM)
    decision.py           # "should I retrieve?" classifier (decide-node logic)
    faithfulness.py       # offline RAGAS-faithfulness proxy + optional real ragas
    reflexion.py          # query refinement from accumulated diagnoses
    nodes.py              # node functions
    graph.py              # StateGraph wiring + compile + routers
    ragas_eval.py         # eval runner over eval_dataset.json on the full pipeline
  eval_dataset.json
  tests/
    __init__.py
    conftest.py
    test_decision.py      # SHOULD vs SHOULDN'T retrieve
    test_rag_tool.py      # tool wiring + retrieval/generation
    test_faithfulness.py  # scorer correctness + diagnosis / failure mode
    test_reflexion.py     # refine improves query; loop fires on low faithfulness
    test_graph.py         # topology, routing, ≤2 retries, fallback, best-attempt selection
    test_ragas_eval.py    # full-pipeline mean faithfulness ≥ 0.75
    test_concepts.py      # the 4 concept goals + the common mistakes
  README.md
  requirements.txt        # langgraph, langchain-core, langchain-openai, qdrant-client,
                          # ragas (optional path), pydantic, typing_extensions, grandalf, pytest
  .env.example
  .gitignore
```

The README's **planned** rows are renumbered: `5.3-Checkpointing-HITL → 5.4`,
`5.4-Supervisor-Agents → 5.5`, `5.5-Reflexion-Loop → 5.6` (or fold Reflexion notes into
this artifact's "see also"). `05-agentic-systems/README.md` gets a new ✅ row for 5.3.

## 10. Error handling

- Every node wraps risky work in `try/except <specific>` and returns `{"error": ...}`
  into the state bus; the graph never crashes on an agent failure.
- `route_after_eval` treats a present `error` as a route to `fallback`.
- `recursion_limit` is a backstop only; the `attempt < max_retries` guard is the real
  bound. `GraphRecursionError` is caught and the last checkpoint returned.

## 11. Determinism & optional real backends

| Flag | Default | Effect when set |
|------|---------|-----------------|
| (none) | offline | mock embedder, extractive grounded generator, offline faithfulness proxy, Qdrant `:memory:`, heuristic decision/refine |
| `USE_LLM=1` (+`OPENROUTER_API_KEY`) | off | real OpenRouter generation + LLM decision/refine |
| `USE_RAGAS=1` (+key) | off | real `ragas` faithfulness instead of the proxy |

## 12. Testing (each goal → an assertion)

~45–60 tests:

- **Decision:** retrieve vs direct on every dataset query.
- **RAG tool:** registered as a LangGraph/LangChain tool; retrieval threshold;
  grounded citations.
- **Faithfulness:** high on grounded answer, low on ungrounded; correct failure-mode
  diagnosis; refusal scored as abstention, **not** hallucination.
- **Reflexion:** low faithfulness → refine fires; refined query improves retrieval;
  loop bounded to ≤ 2 retries; **best** attempt selected, not last.
- **Graph:** topology (nodes/edges), routing keys, graceful fallback path, no crash on
  injected node error.
- **RAGAS eval:** mean faithfulness over served answers **≥ 0.75**.
- **Concepts (the four "understand" goals):** when to retrieve vs answer from context;
  reading a RAGAS score table → what to fix; tracing a hallucination to its root failure
  mode; RAG-as-a-tool (not a fixed pipeline).
- **Common mistakes:** unbounded retry loop; serving last instead of best attempt;
  refusal mis-scored as hallucination.

## 13. Obsidian notes (Week 2 = OCR, ReAct, MCP, LangGraph, Multi-agent, Hallucination)

Vault: `C:\Users\minhh\OneDrive\Documents\Obsidian Vault`.

New folder **`Week 2 Review - Agentic RAG/`** containing:

- `2026-06-04 task.md` — the review task/log note.
- `5 Concepts from Week 2 Review - Agentic RAG.md` — index of the five concept notes.
- Concept notes:
  - `Agentic RAG — RAG as a Tool, Not a Pipeline.md`
  - `When to Retrieve vs Answer from Context.md`
  - `Reflexion Loop — Self-Correction Gated on RAGAS.md`
  - `Reading a RAGAS Score Table — What to Fix.md`
  - `Tracing a Hallucination to its Root Failure Mode.md`

Each note `[[wikilink]]`-cross-links into the six Week-2 folders' concept notes (e.g.
LangGraph "Conditional Edges", Multi-agent "Error Recovery", Hallucination "Citation
Grounding — The 3-Check Pattern" and "Five Failure Modes"). A backlink line
(`→ see [[...Agentic RAG...]]`) is added into the most relevant existing Week-2 concept
notes so the review is reachable from them.

## 14. Out of scope (YAGNI)

- Durable checkpointing (SqliteSaver) and HITL gates — that's the renumbered 5.4.
- Multi-document / multi-hop decomposition — separate artifact.
- The full 50-pair production eval set — module 08's responsibility.
- Real Qdrant server / PDF ingestion — the offline `:memory:` corpus is the default;
  the real 3.2 pipeline already covers server-backed ingestion.

## 15. References

- Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement
  Learning.* NeurIPS 2023. https://arxiv.org/abs/2303.11366
- Es, S., et al. (2023). *RAGAS: Automated Evaluation of RAG.* arXiv:2309.15217.
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Repo precedents: `05-agentic-systems/5.2-Multi-Agent-Research/`,
  `07-guardrails/7.1-Hallucination-Detection/`, `03-rag/3.2-NaiveRAG/`.
