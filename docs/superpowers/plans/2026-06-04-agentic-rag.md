# Agentic RAG (5.3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `05-agentic-systems/5.3-Agentic-RAG/agentic_rag.py` — a LangGraph agent that wires `rag_v1_complete` as a tool, decides retrieve-vs-direct, runs a RAGAS-faithfulness-gated Reflexion loop (refine if <0.70, max 2 retries, graceful fallback), and ships a full-pipeline RAGAS eval scoring mean faithfulness ≥0.75.

**Architecture:** A compiled LangGraph `StateGraph` with nodes `decide → {retrieve_and_generate → evaluate → (refine ↺ | fallback | finalize)} | direct_answer`. Offline & deterministic by default: tf-idf retrieval over a corpus-built vocabulary in Qdrant `:memory:`, an extractive grounded generator that *simulates* parametric-memory fabrication when retrieval grounds weakly, and an offline faithfulness proxy. `USE_LLM=1` / `USE_RAGAS=1` swap in real OpenRouter generation and the real `ragas` library without changing topology.

**Tech Stack:** Python, LangGraph, langchain-core (`@tool`), qdrant-client (`:memory:`), numpy, pydantic, pytest. Optional: langchain-openai (OpenRouter), ragas.

**The deterministic Reflexion mechanism (read first):** The corpus uses *formal* vocabulary (`retention`, `purged`, `quota`, `pricing`); some eval questions use *casual* vocabulary (`keep`, `deleting`). tf-idf with `idf = log(N/df)` makes corpus-ubiquitous terms (`helios`, in every doc) contribute 0, so grounding is driven by *distinctive* terms. A casual question grounds below `GROUND_MIN` → the generator emits a "guess" sentence (reusing the question's own words, which aren't in any chunk) → the faithfulness proxy flags unsupported claims → `refine_query` expands casual→formal via a generic `SYNONYM_MAP` → retrieval sharpens onto the one owning document → faithful. This is principled (vocabulary mismatch is *the* classic retrieval failure) and fully deterministic.

**Conventions (inherited, non-negotiable):** self-contained day folder; offline/deterministic by default with optional real backends behind env flags; typed `TypedDict` state with `Annotated[..., operator.add]` reducers; nodes return partial-state dicts and **never raise** (trap errors into the state bus); routers read state and return a key, never mutate; compile with `MemorySaver()`; force UTF-8 stdout/stderr; **each goal → an explicit test assertion**, plus tests for the source-note "common mistakes".

**Note on code blocks:** the code below is complete and correct. When implementing, expand docstrings/comments to match the repo's annotation density (see Day 9 `research_agent_langgraph.py` and Day 10 `src/`). Do **not** change function/type names or signatures — later tasks depend on them.

**Tests assert behaviors, not exact float scores** (cosine of tf-idf vectors is deterministic but brittle to hand-predict): assert faithful ≥ threshold, route equality, "reflexion fired" (`len(attempts_log) > 1`), rank-1 document identity, retry-count bounds, best-attempt selection.

---

## Locked shared names (used across tasks)

- `src/config.py`: `use_llm()`, `use_ragas()`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`
- `src/text.py`: `STOPWORDS`, `tokenize(text)`, `content_tokens(text)`
- `src/state.py`: `DEFAULT_MAX_RETRIES=2`, `FAITHFULNESS_THRESHOLD=0.70`, `DATASET_TARGET=0.75`, `AgenticRAGState`, `initial_state(question, max_retries=2, faithfulness_threshold=0.70)`
- `src/corpus.py`: `DOCUMENTS`, `documents()`
- `src/embedder.py`: `TfidfSpace` (attrs: `.vocab`, `.idf`, `.dim`, `.ubiquitous`; methods: `.embed(text)->np.ndarray`, `.idf_mass(terms)->float`), `build_space(docs)`
- `src/retriever.py`: `TOP_K=4`, `MIN_SCORE=0.0`, `get_space()`, `retrieve(query, top_k=TOP_K)->list[dict]` (chunk = `{"doc_id","text","score","rank","label"}`)
- `src/generator.py`: `GROUND_MIN=1.6`, `REFUSAL`, `generate(question, query, chunks)->dict` (`{"answer","citations","confidence"}`)
- `src/faithfulness.py`: `SUPPORT_FRAC=0.5`, `FaithReport`, `score(question, answer, chunks)->FaithReport`
- `src/rag_tool.py`: `RagResult`, `rag_v1_complete(question, query=None)->RagResult`, `rag_tool` (a `StructuredTool`)
- `src/decision.py`: `DIRECT_MARKERS`, `Decision`, `decide(question)->Decision`
- `src/reflexion.py`: `SYNONYM_MAP`, `refine(question, attempts_log)->str`
- `src/nodes.py`: `decide_node`, `retrieve_and_generate_node`, `evaluate_node`, `refine_node`, `direct_node`, `fallback_node`, `finalize_node`, `route_decision`, `route_after_eval`
- `src/graph.py`: `build_graph(checkpointer=None)`, `run_agent(question, *, max_retries=2, faithfulness_threshold=0.70, thread_id=None, recursion_limit=25)->AgenticRAGState`
- `src/ragas_eval.py`: `load_dataset(path)`, `evaluate_dataset(path)->dict`, `format_table(metrics)->str`

---

## Task 0: Scaffold the day folder

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/requirements.txt`
- Create: `05-agentic-systems/5.3-Agentic-RAG/.env.example`
- Create: `05-agentic-systems/5.3-Agentic-RAG/.gitignore`
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/__init__.py`
- Create: `05-agentic-systems/5.3-Agentic-RAG/tests/__init__.py`
- Create: `05-agentic-systems/5.3-Agentic-RAG/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
# Day 11 — Agentic RAG (Week 2 Review): LangGraph agent + RAGAS reflexion loop.
# Runs FULLY OFFLINE and deterministically by default (tf-idf retrieval in
# Qdrant :memory:, extractive generator, offline faithfulness proxy). Set the
# env flags in .env.example to switch any component to its real backend.
langgraph>=0.2.50          # StateGraph, conditional edges, MemorySaver, recursion guard
langchain-core>=0.3.0      # StructuredTool — wiring rag_v1_complete as a tool
qdrant-client>=1.7.0       # in-memory vector store (:memory:), no server needed
numpy>=1.26.0              # tf-idf vectors
pydantic>=2.7.0            # typed reports / structured outputs
typing_extensions>=4.11.0  # TypedDict + Annotated reducers
grandalf>=0.8             # ASCII graph rendering
python-dotenv>=1.0.0       # load OPENROUTER_*/USE_* from .env
pytest>=8.0.0             # machine-checked validation of every goal
# --- optional real backends (NOT needed for the offline default) ---
# langchain-openai>=0.2.0  # real OpenRouter generation / decision / refine (USE_LLM=1)
# ragas>=0.2.0             # real RAGAS faithfulness (USE_RAGAS=1)
```

- [ ] **Step 2: Create `.env.example`**

```
# All optional. With NONE of these set, the agent runs fully offline & deterministic.
# --- real generation + LLM decision/refine ---
USE_LLM=0
OPENROUTER_API_KEY=
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# --- real RAGAS faithfulness instead of the offline proxy ---
USE_RAGAS=0
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
graph.mmd
.env
```

- [ ] **Step 4: Create empty `src/__init__.py` and `tests/__init__.py`**

Both files: a single line.

```python
"""Day 11 — Agentic RAG (Week 2 Review)."""
```

- [ ] **Step 5: Create `conftest.py`** (makes `src` importable when running pytest from the folder)

```python
"""Put the module root on sys.path so `import src.foo` works under pytest."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

- [ ] **Step 6: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/
git commit -m "chore(5.3-agentic-rag): scaffold day folder (reqs, env, gitignore, conftest)"
```

---

## Task 1: `config.py` + `text.py` (shared utilities)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/config.py`
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/text.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_text.py`

- [ ] **Step 1: Write the failing test** (`tests/test_text.py`)

```python
from src.text import tokenize, content_tokens, STOPWORDS


def test_tokenize_lowercases_and_splits_alnum():
    assert tokenize("Helios keeps 90 Documents!") == ["helios", "keeps", "90", "documents"]


def test_content_tokens_drops_stopwords_keeps_oov_content():
    # "how", "does", "the" are stopwords; "keep"/"documents" are content (kept even if OOV).
    assert content_tokens("How does the Helios keep documents") == ["helios", "keep", "documents"]


def test_stopwords_contains_function_words():
    for w in ("the", "is", "a", "per", "what", "how", "does"):
        assert w in STOPWORDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.text'`)

- [ ] **Step 3: Write `src/config.py`**

```python
"""Resolve which backend each component uses, from env. Offline by default."""
from __future__ import annotations

import os

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def use_llm() -> bool:
    """True only when explicitly enabled AND a key is present."""
    return os.environ.get("USE_LLM", "0") == "1" and bool(os.environ.get("OPENROUTER_API_KEY"))


def use_ragas() -> bool:
    """True only when explicitly enabled AND a key is present (ragas needs an LLM)."""
    return os.environ.get("USE_RAGAS", "0") == "1" and bool(os.environ.get("OPENROUTER_API_KEY"))
```

- [ ] **Step 4: Write `src/text.py`**

```python
"""Tokenization + stopwords shared by retrieval, generation, scoring, routing."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "how", "what",
    "which", "who", "whom", "that", "this", "these", "those", "it", "its", "they",
    "them", "their", "i", "we", "us", "you", "your", "my", "me", "per", "by", "at",
    "with", "as", "if", "then", "than", "so", "such", "can", "could", "would",
    "will", "shall", "should", "may", "might", "must", "more", "less", "within",
    "before", "after", "every", "each", "any", "some", "about", "into", "from",
    "up", "out", "over", "under", "again", "there", "here", "when", "where", "why",
    "use", "used", "using", "get", "got", "make", "made", "given", "based",
})


def tokenize(text: str) -> list[str]:
    """Lowercase, keep [a-z0-9]+ runs, drop everything else."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """tokenize() minus stopwords. OOV content words are kept on purpose:
    they are the signal an answer drifted off the retrieved context."""
    return [t for t in tokenize(text) if t not in STOPWORDS]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_text.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/config.py 05-agentic-systems/5.3-Agentic-RAG/src/text.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_text.py
git commit -m "feat(5.3-agentic-rag): config + text tokenization utilities"
```

---

## Task 2: `state.py` (typed graph state)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/state.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_state.py`

- [ ] **Step 1: Write the failing test** (`tests/test_state.py`)

```python
import operator

from src.state import (
    AgenticRAGState,
    DEFAULT_MAX_RETRIES,
    FAITHFULNESS_THRESHOLD,
    DATASET_TARGET,
    initial_state,
)


def test_constants():
    assert DEFAULT_MAX_RETRIES == 2
    assert FAITHFULNESS_THRESHOLD == 0.70
    assert DATASET_TARGET == 0.75


def test_initial_state_defaults():
    s = initial_state("How long does Helios keep documents?")
    assert s["question"] == "How long does Helios keep documents?"
    assert s["query"] == s["question"]          # first query == the question verbatim
    assert s["attempt"] == 0
    assert s["max_retries"] == 2
    assert s["faithfulness_threshold"] == 0.70
    assert s["attempts_log"] == []              # reducer field seeded empty
    assert s["log"] == []
    assert s["served"] is False


def test_reducer_fields_are_annotated_lists():
    # The two accumulating fields must use operator.add so node outputs append.
    hints = AgenticRAGState.__annotations__
    for field in ("attempts_log", "log"):
        meta = getattr(hints[field], "__metadata__", ())
        assert operator.add in meta, f"{field} must use operator.add reducer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.state'`)

- [ ] **Step 3: Write `src/state.py`**

```python
"""The shared typed state — the communication bus for every node."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

DEFAULT_MAX_RETRIES = 2          # max reflexion retries (=> <= 3 generate cycles)
FAITHFULNESS_THRESHOLD = 0.70    # per-query reflexion trigger / serve gate
DATASET_TARGET = 0.75            # headline: mean faithfulness over served retrieve answers


class AgenticRAGState(TypedDict):
    # input
    question: str
    max_retries: int
    faithfulness_threshold: float
    # decision
    route: str                   # "retrieve" | "direct"
    decision_reason: str
    # working query (refined across attempts)
    query: str
    attempt: int                 # count of retrieve+generate cycles done
    # retrieval + generation (overwritten each attempt)
    chunks: list[dict]
    answer: str
    citations: list[str]
    # evaluation (overwritten each attempt)
    faithfulness: float
    eval_report: dict
    # reflexion memory (accumulates)
    attempts_log: Annotated[list[dict], operator.add]
    # terminal
    status: str                  # "answered" | "fallback" | "direct" | ""
    final_answer: str
    served: bool
    # transcript (accumulates)
    log: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
) -> AgenticRAGState:
    """Seed a fresh state. Reducer fields start [] so operator.add can append."""
    q = question.strip()
    return AgenticRAGState(
        question=q,
        max_retries=max_retries,
        faithfulness_threshold=faithfulness_threshold,
        route="",
        decision_reason="",
        query=q,
        attempt=0,
        chunks=[],
        answer="",
        citations=[],
        faithfulness=0.0,
        eval_report={},
        attempts_log=[],
        status="",
        final_answer="",
        served=False,
        log=[],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/state.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_state.py
git commit -m "feat(5.3-agentic-rag): typed AgenticRAGState + initial_state"
```

---

## Task 3: `corpus.py` (the engineered deterministic corpus)

The corpus is deliberately designed (see plan intro). Each "owning" document holds distinctive formal terms (`retention`, `quota`, `pricing`, `embedding`); `helios` is in every doc (idf 0); `documents` is shared by D1 and D5 so a casual retention question grounds ambiguously until Reflexion adds the formal terms.

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/corpus.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_corpus.py`

- [ ] **Step 1: Write the failing test** (`tests/test_corpus.py`)

```python
from src.corpus import DOCUMENTS, documents
from src.text import tokenize


def test_seven_documents_with_unique_ids():
    assert len(DOCUMENTS) == 7
    ids = [d["doc_id"] for d in DOCUMENTS]
    assert ids == [f"D{i}" for i in range(1, 8)]


def test_documents_returns_a_copy():
    a = documents()
    a.append({"doc_id": "X", "text": "x"})
    assert len(documents()) == 7  # mutating the returned list must not affect the source


def test_helios_is_ubiquitous_distinctive_terms_are_owned():
    toks = {d["doc_id"]: set(tokenize(d["text"])) for d in DOCUMENTS}
    # 'helios' in every doc => idf 0 => non-discriminative
    assert all("helios" in t for t in toks.values())
    # distinctive terms each owned by exactly one doc
    owned = {"retention": "D1", "purged": "D1", "quota": "D3",
             "pricing": "D2", "embedding": "D4"}
    for term, owner in owned.items():
        holders = [doc_id for doc_id, t in toks.items() if term in t]
        assert holders == [owner], f"{term} should be owned only by {owner}, got {holders}"
    # 'documents' is shared by D1 and D5 (the ambiguity that motivates reflexion)
    holders = sorted(doc_id for doc_id, t in toks.items() if "documents" in t)
    assert holders == ["D1", "D5"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.corpus'`)

- [ ] **Step 3: Write `src/corpus.py`**

```python
"""A tiny, fully-controlled corpus about a fictional product ("Helios").

Fictional on purpose: ground truth is entirely in these texts, so no parametric
knowledge can leak into "faithful" answers. The vocabulary is engineered so the
reflexion loop fires deterministically (see the plan intro / README)."""
from __future__ import annotations

DOCUMENTS: list[dict] = [
    {"doc_id": "D1", "text": (
        "Helios retention: indexed documents are retained for 90 days, after "
        "which they are automatically purged. Retention can be extended to 365 "
        "days per collection by an administrator.")},
    {"doc_id": "D2", "text": (
        "Helios pricing: the free tier is free of charge, while the pro tier is "
        "billed at ten cents for every thousand queries processed.")},
    {"doc_id": "D3", "text": (
        "Helios query quota: the free tier permits one thousand queries each day, "
        "and the pro tier raises the quota to one hundred thousand queries daily.")},
    {"doc_id": "D4", "text": (
        "Helios embedding model: collections use the voyage-3 embedding model, "
        "which produces 1024-dimensional vectors and is fixed at creation.")},
    {"doc_id": "D5", "text": (
        "Helios stores collection documents metadata in a managed Postgres "
        "database, and backups run nightly.")},
    {"doc_id": "D6", "text": (
        "The Helios dashboard displays query latency and recall charts that "
        "refresh every five minutes.")},
    {"doc_id": "D7", "text": (
        "Helios is SOC 2 compliant and supports single sign-on for enterprise "
        "organizations.")},
]


def documents() -> list[dict]:
    """Return a shallow copy so callers cannot mutate the source corpus."""
    return [dict(d) for d in DOCUMENTS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/corpus.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_corpus.py
git commit -m "feat(5.3-agentic-rag): engineered deterministic Helios corpus"
```

---

## Task 4: `embedder.py` (tf-idf vector space)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/embedder.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_embedder.py`

- [ ] **Step 1: Write the failing test** (`tests/test_embedder.py`)

```python
import math

import numpy as np

from src.corpus import documents
from src.embedder import TfidfSpace, build_space


def test_idf_zero_for_ubiquitous_term():
    space = build_space(documents())
    # 'helios' appears in all 7 docs => idf = log(N/df) = log(1) = 0
    assert space.idf.get("helios", 0.0) == 0.0
    # 'retention' appears in 1 doc => idf = log(7) > 0
    assert space.idf["retention"] == math.log(7 / 1)


def test_ubiquitous_set_contains_helios_only():
    space = build_space(documents())
    assert space.ubiquitous == {"helios"}


def test_embed_is_unit_norm_and_deterministic():
    space = build_space(documents())
    v1 = space.embed("Helios query quota per day")
    v2 = space.embed("Helios query quota per day")
    assert np.allclose(v1, v2)
    assert v1.shape == (space.dim,)
    assert math.isclose(float(np.linalg.norm(v1)), 1.0, rel_tol=1e-6)


def test_oov_only_query_is_zero_vector():
    space = build_space(documents())
    v = space.embed("carbon footprint")  # neither token in vocab
    assert float(np.linalg.norm(v)) == 0.0


def test_idf_mass_sums_distinctive_terms_in_vocab():
    space = build_space(documents())
    # 'quota' (idf>0) counts; 'helios' (idf 0) and 'carbon' (OOV) do not.
    expected = space.idf["quota"]
    assert math.isclose(space.idf_mass(["quota", "helios", "carbon"]), expected, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.embedder'`)

- [ ] **Step 3: Write `src/embedder.py`**

```python
"""Deterministic tf-idf vector space over the corpus vocabulary.

idf = log(N/df): corpus-ubiquitous terms (df == N, e.g. 'helios') get idf 0 and
drop out of both ranking and grounding — exactly what we want, since a term in
every document is non-discriminative. With USE_LLM=1 a real embedding model can
be substituted, but the offline tf-idf space is what makes the reflexion loop
deterministic and assertable."""
from __future__ import annotations

import math

import numpy as np

from src.text import tokenize


class TfidfSpace:
    def __init__(self, vocab: list[str], idf: dict[str, float]):
        self.vocab = vocab
        self.idf = idf
        self.dim = len(vocab)
        self._index = {t: i for i, t in enumerate(vocab)}
        self.ubiquitous = {t for t, w in idf.items() if w == 0.0}

    def embed(self, text: str) -> np.ndarray:
        """tf-idf vector, L2-normalized. Zero vector if no in-vocab term hits."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            i = self._index.get(tok)
            if i is not None:
                vec[i] += self.idf[tok]   # tf (count) * idf
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def idf_mass(self, terms) -> float:
        """Sum of idf over the given terms that are discriminative (idf>0)."""
        return float(sum(self.idf.get(t, 0.0) for t in terms))


def build_space(docs: list[dict]) -> TfidfSpace:
    """Build vocab + idf from the documents' token sets."""
    n = len(docs)
    df: dict[str, int] = {}
    for d in docs:
        for tok in set(tokenize(d["text"])):
            df[tok] = df.get(tok, 0) + 1
    vocab = sorted(df)
    idf = {t: math.log(n / df[t]) for t in vocab}   # df==n -> log(1)==0
    return TfidfSpace(vocab, idf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/embedder.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_embedder.py
git commit -m "feat(5.3-agentic-rag): deterministic tf-idf embedding space"
```

---

## Task 5: `retriever.py` (Qdrant `:memory:` retrieval)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/retriever.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_retriever.py`

- [ ] **Step 1: Write the failing test** (`tests/test_retriever.py`)

```python
from src.retriever import retrieve, get_space, TOP_K


def test_distinctive_query_ranks_owner_first():
    chunks = retrieve("Helios query quota per day")
    assert chunks[0]["doc_id"] == "D3"          # 'quota' is owned by D3
    assert chunks[0]["label"] == "Doc 1"
    assert chunks[0]["rank"] == 1


def test_embedding_query_ranks_d4_first():
    chunks = retrieve("Which embedding model does Helios use for collections?")
    assert chunks[0]["doc_id"] == "D4"


def test_top_k_limits_results_and_labels_are_sequential():
    chunks = retrieve("Helios documents")
    assert len(chunks) <= TOP_K
    assert [c["label"] for c in chunks] == [f"Doc {i}" for i in range(1, len(chunks) + 1)]


def test_oov_only_query_returns_empty():
    assert retrieve("carbon footprint emissions") == []


def test_get_space_is_singleton():
    assert get_space() is get_space()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.retriever'`)

- [ ] **Step 3: Write `src/retriever.py`**

```python
"""Retrieval over an in-memory Qdrant collection of tf-idf vectors.

Qdrant runs in :memory: mode — no server, no Docker, fully offline. The space
and the index are built once (lazy singletons) from the corpus. Below MIN_SCORE
(default 0.0, i.e. any in-vocab overlap) nothing is returned, so an OOV-only
query yields [] -> the NO_RETRIEVAL failure mode."""
from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.corpus import documents
from src.embedder import TfidfSpace, build_space

TOP_K = 4
MIN_SCORE = 0.0
_COLLECTION = "helios_chunks"

_space: Optional[TfidfSpace] = None
_client: Optional[QdrantClient] = None


def get_space() -> TfidfSpace:
    global _space
    if _space is None:
        _space = build_space(documents())
    return _space


def _get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client
    space = get_space()
    docs = documents()
    client = QdrantClient(location=":memory:")
    client.recreate_collection(
        collection_name=_COLLECTION,
        vectors_config=VectorParams(size=space.dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=i, vector=space.embed(d["text"]).tolist(),
                    payload={"doc_id": d["doc_id"], "text": d["text"]})
        for i, d in enumerate(docs)
    ]
    client.upsert(collection_name=_COLLECTION, points=points, wait=True)
    _client = client
    return _client


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed the query, search, threshold, and label results [Doc 1..N]."""
    space = get_space()
    qvec = space.embed(query)
    if float((qvec != 0).sum()) == 0:     # zero vector => no in-vocab signal
        return []
    client = _get_client()
    hits = client.query_points(
        collection_name=_COLLECTION, query=qvec.tolist(), limit=top_k,
    ).points
    chunks: list[dict] = []
    rank = 0
    for h in hits:
        score = float(h.score) if h.score is not None else 0.0
        if score <= MIN_SCORE:
            continue
        rank += 1
        payload = h.payload or {}
        chunks.append({
            "doc_id": payload.get("doc_id", ""),
            "text": payload.get("text", ""),
            "score": score,
            "rank": rank,
            "label": f"Doc {rank}",
        })
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: PASS (5 passed)

Note: if `recreate_collection` emits a deprecation warning on your qdrant-client version, it still works; leave it (matches the repo's existing usage style).

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/retriever.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_retriever.py
git commit -m "feat(5.3-agentic-rag): Qdrant :memory: tf-idf retriever"
```

---

## Task 6: `generator.py` (grounded extractive generator + simulated drift)

The generator's behavior is the heart of the offline lesson: when the best chunk grounds the query strongly (idf mass ≥ `GROUND_MIN`) it **extracts** the best-matching sentence verbatim and cites it (faithful). When grounding is weak it **guesses** — it composes a sentence from the *question's own content words* (which aren't in any chunk), simulating parametric-memory fabrication. That guess is what the faithfulness proxy later flags.

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/generator.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_generator.py`

- [ ] **Step 1: Write the failing test** (`tests/test_generator.py`)

```python
from src.generator import generate, GROUND_MIN, REFUSAL
from src.retriever import retrieve


def test_strong_grounding_extracts_verbatim_sentence_with_citation():
    chunks = retrieve("Helios query quota per day")
    out = generate("What is the Helios query quota per day?", "Helios query quota per day", chunks)
    assert out["confidence"] >= GROUND_MIN
    assert "quota" in out["answer"].lower()
    assert out["citations"] == ["Doc 1"]
    # the answer sentence is taken from the chunk text (grounded)
    assert out["answer"].split(" [Doc")[0].strip().rstrip(".") in chunks[0]["text"]


def test_weak_grounding_guesses_using_question_words():
    # casual question; query == question; grounds weakly -> guess
    chunks = retrieve("How long does Helios keep documents before deleting them?")
    out = generate(
        "How long does Helios keep documents before deleting them?",
        "How long does Helios keep documents before deleting them?",
        chunks,
    )
    assert out["confidence"] < GROUND_MIN
    # guess reuses casual question words that are NOT in the corpus
    assert "keep" in out["answer"].lower() or "deleting" in out["answer"].lower()


def test_no_chunks_returns_refusal():
    out = generate("What is the Helios carbon footprint?", "carbon footprint", [])
    assert out["answer"] == REFUSAL
    assert out["citations"] == []
    assert out["confidence"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generator.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.generator'`)

- [ ] **Step 3: Write `src/generator.py`**

```python
"""Grounded answer generation.

Offline default: extract the best chunk sentence when grounding is strong;
otherwise simulate the canonical RAG failure (answer from the question itself).
With USE_LLM=1 a real OpenRouter model generates instead, but the topology and
the {answer, citations, confidence} contract are unchanged."""
from __future__ import annotations

import re

from src import config
from src.retriever import get_space
from src.text import content_tokens

GROUND_MIN = 1.6   # min idf mass (query ∩ best chunk) to trust extraction
REFUSAL = ("I do not have enough information in the retrieved documents to "
           "answer this question.")

_SENT_RE = re.compile(r"[^.;]+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _best_sentence(query: str, chunk_text: str, space) -> str:
    """Sentence in the chunk with the highest idf overlap with the query."""
    qterms = set(content_tokens(query))
    best, best_mass = chunk_text.strip(), -1.0
    for sent in _sentences(chunk_text):
        mass = space.idf_mass(t for t in content_tokens(sent) if t in qterms)
        if mass > best_mass:
            best, best_mass = sent, mass
    return best


def _guess(question: str) -> str:
    """Simulated parametric-memory fabrication: assert the question's own content
    words as if answered. These words are absent from the chunks -> unsupported."""
    core = " ".join(content_tokens(question))
    return f"{core} is described in the retrieved context"


def generate(question: str, query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": REFUSAL, "citations": [], "confidence": 0.0}

    if config.use_llm():
        return _generate_llm(question, query, chunks)

    space = get_space()
    best = chunks[0]
    qterms = set(content_tokens(query))
    chunk_terms = set(content_tokens(best["text"]))
    confidence = space.idf_mass(qterms & chunk_terms)

    if confidence >= GROUND_MIN:
        sentence = _best_sentence(query, best["text"], space)
        answer = f"{sentence} [{best['label']}]."
        return {"answer": answer, "citations": [best["label"]], "confidence": confidence}

    # weak grounding -> guess (cites the best chunk, but the claim isn't in it)
    answer = f"{_guess(question)} [{best['label']}]."
    return {"answer": answer, "citations": [best["label"]], "confidence": confidence}


def _generate_llm(question: str, query: str, chunks: list[dict]) -> dict:
    """Real OpenRouter generation (only when USE_LLM=1). Context-only prompt."""
    from langchain_openai import ChatOpenAI

    context = "\n".join(f"[{c['label']}] {c['text']}" for c in chunks)
    system = (
        "Answer ONLY from the CONTEXT. Cite every claim inline as [Doc N]. "
        "If the context lacks the answer, reply exactly: " + REFUSAL)
    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=__import__("os").environ["OPENROUTER_API_KEY"],
                     temperature=0, max_tokens=400, timeout=45)
    text = str(llm.invoke([("system", system),
                           ("human", f"CONTEXT:\n{context}\n\nQUESTION:\n{question}")]).content).strip()
    cites = re.findall(r"Doc \d+", text)
    return {"answer": text, "citations": sorted(set(cites)), "confidence": float(len(cites))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/generator.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_generator.py
git commit -m "feat(5.3-agentic-rag): grounded extractive generator with simulated drift"
```

---

## Task 7: `faithfulness.py` (offline RAGAS-faithfulness proxy + failure mode)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/faithfulness.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_faithfulness.py`

- [ ] **Step 1: Write the failing test** (`tests/test_faithfulness.py`)

```python
from src.faithfulness import score, FaithReport, SUPPORT_FRAC
from src.generator import REFUSAL


def _chunks():
    return [{"doc_id": "D1", "label": "Doc 1", "score": 0.9, "rank": 1,
             "text": "Helios retention: indexed documents are retained for 90 days, "
                     "after which they are automatically purged."}]


def test_extracted_answer_is_fully_faithful():
    ans = "indexed documents are retained for 90 days [Doc 1]."
    rep = score("How long are documents kept?", ans, _chunks())
    assert isinstance(rep, FaithReport)
    assert rep.score == 1.0
    assert rep.failure_mode == "GROUNDED"
    assert rep.unsupported_claims == []


def test_guess_answer_is_unfaithful_and_diagnosed():
    # 'keep'/'deleting' are not in the chunk -> unsupported claim
    ans = "helios keep documents deleting is described in the retrieved context [Doc 1]."
    rep = score("How long does Helios keep documents before deleting them?", ans, _chunks())
    assert rep.score < 0.70
    assert rep.failure_mode in ("WRONG_CHUNKS", "UNSUPPORTED_GENERATION")
    assert rep.unsupported_claims


def test_no_chunks_is_no_retrieval():
    rep = score("anything", "some answer", [])
    assert rep.score == 0.0
    assert rep.failure_mode == "NO_RETRIEVAL"


def test_refusal_is_abstained_not_hallucination():
    rep = score("unanswerable", REFUSAL, _chunks())
    assert rep.failure_mode == "ABSTAINED"
    assert rep.is_refusal is True


def test_support_fraction_threshold_constant():
    assert SUPPORT_FRAC == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_faithfulness.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.faithfulness'`)

- [ ] **Step 3: Write `src/faithfulness.py`**

```python
"""Offline RAGAS-faithfulness proxy + root-failure-mode diagnosis.

faithfulness = supported_claims / total_claims, where a claim (a sentence of the
answer) is supported iff >= SUPPORT_FRAC of its DISCRIMINATIVE content tokens
appear in the union of retrieved chunks. Corpus-ubiquitous terms (idf 0, e.g.
'helios') are excluded so they neither prop up nor sink a claim; OOV words are
KEPT (they are the fabrication signal). With USE_RAGAS=1 the real `ragas`
faithfulness metric is used instead, behind the same return contract.

Failure mode (the 'trace a hallucination to its root cause' goal):
  NO_RETRIEVAL          - nothing retrieved
  ABSTAINED             - answer is the system's own refusal (safe, not a bug)
  GROUNDED              - every claim supported
  WRONG_CHUNKS          - unsupported AND retrieval grounded weakly (retrieval bug)
  UNSUPPORTED_GENERATION- unsupported despite the answer terms being retrievable
                          (generation drifted off otherwise-adequate context)"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config
from src.generator import REFUSAL, GROUND_MIN
from src.retriever import get_space
from src.text import content_tokens

SUPPORT_FRAC = 0.5
_CITE_RE = re.compile(r"\[doc \d+\]", re.IGNORECASE)
_SENT_RE = re.compile(r"[^.;]+")


@dataclass
class FaithReport:
    score: float
    supported: int
    total: int
    unsupported_claims: list[str]
    failure_mode: str
    is_refusal: bool = False
    extra: dict = field(default_factory=dict)

    def asdict(self) -> dict:
        return {
            "score": self.score, "supported": self.supported, "total": self.total,
            "unsupported_claims": self.unsupported_claims,
            "failure_mode": self.failure_mode, "is_refusal": self.is_refusal,
        }


def _claims(answer: str) -> list[str]:
    stripped = _CITE_RE.sub("", answer)
    return [s.strip() for s in _SENT_RE.findall(stripped) if s.strip()]


def score(question: str, answer: str, chunks: list[dict]) -> FaithReport:
    if config.use_ragas():
        return _score_ragas(question, answer, chunks)

    if not chunks:
        return FaithReport(0.0, 0, 0, [], "NO_RETRIEVAL")

    if REFUSAL.lower() in answer.lower():
        return FaithReport(1.0, 0, 0, [], "ABSTAINED", is_refusal=True)

    space = get_space()
    chunk_tokens: set[str] = set()
    for c in chunks:
        chunk_tokens |= set(content_tokens(c["text"]))

    claims = _claims(answer)
    if not claims:
        return FaithReport(1.0, 0, 0, [], "GROUNDED")

    supported, unsupported = 0, []
    for claim in claims:
        # discriminative denominator: drop corpus-ubiquitous terms, keep OOV
        terms = [t for t in content_tokens(claim) if t not in space.ubiquitous]
        if not terms:
            supported += 1
            continue
        hit = sum(1 for t in terms if t in chunk_tokens)
        if hit / len(terms) >= SUPPORT_FRAC:
            supported += 1
        else:
            unsupported.append(claim)

    total = len(claims)
    faith = supported / total

    if not unsupported:
        mode = "GROUNDED"
    else:
        # root cause: did retrieval even ground the question?
        qterms = set(content_tokens(question))
        best_mass = space.idf_mass(qterms & set(content_tokens(chunks[0]["text"])))
        mode = "WRONG_CHUNKS" if best_mass < GROUND_MIN else "UNSUPPORTED_GENERATION"

    return FaithReport(faith, supported, total, unsupported, mode)


def _score_ragas(question: str, answer: str, chunks: list[dict]) -> FaithReport:
    """Real RAGAS faithfulness (only when USE_RAGAS=1)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness as ragas_faithfulness

    ds = Dataset.from_dict({
        "question": [question], "answer": [answer],
        "contexts": [[c["text"] for c in chunks]],
    })
    result = evaluate(ds, metrics=[ragas_faithfulness])
    val = float(result["faithfulness"])
    mode = "GROUNDED" if val >= 0.70 else "UNSUPPORTED_GENERATION"
    return FaithReport(val, 0, 1, [] if val >= 0.70 else [answer], mode,
                       extra={"ragas": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_faithfulness.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/faithfulness.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_faithfulness.py
git commit -m "feat(5.3-agentic-rag): offline RAGAS-faithfulness proxy + failure-mode diagnosis"
```

---

## Task 8: `rag_tool.py` (wire `rag_v1_complete` as a LangChain tool)

This is the "Wire rag_v1_complete as a tool in LangGraph" requirement. `rag_v1_complete` is the plain callable (retrieve → generate); `rag_tool` is the `StructuredTool` wrapper an LLM-driven agent would call.

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/rag_tool.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_rag_tool.py`

- [ ] **Step 1: Write the failing test** (`tests/test_rag_tool.py`)

```python
import json

from langchain_core.tools import BaseTool

from src.rag_tool import rag_v1_complete, rag_tool, RagResult


def test_rag_v1_complete_returns_result_with_chunks_and_answer():
    res = rag_v1_complete("What is the Helios query quota per day?")
    assert isinstance(res, RagResult)
    assert res.chunks and res.chunks[0]["doc_id"] == "D3"
    assert "quota" in res.answer.lower()
    assert res.citations == ["Doc 1"]


def test_query_defaults_to_question_but_can_differ():
    res = rag_v1_complete("How long does Helios keep documents?",
                          query="Helios documents retained retention purged")
    assert res.chunks[0]["doc_id"] == "D1"   # refined query sharpens onto D1


def test_rag_tool_is_a_langchain_tool():
    assert isinstance(rag_tool, BaseTool)
    assert rag_tool.name == "rag_v1_complete"
    assert "retriev" in rag_tool.description.lower()


def test_rag_tool_invoke_returns_json_string():
    out = rag_tool.invoke({"query": "What is the Helios query quota per day?"})
    payload = json.loads(out)
    assert payload["citations"] == ["Doc 1"]
    assert "quota" in payload["answer"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rag_tool.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.rag_tool'`)

- [ ] **Step 3: Write `src/rag_tool.py`**

```python
"""rag_v1_complete: the v1 RAG pipeline (retrieve -> generate) exposed BOTH as a
plain callable (rich RagResult, used by the graph node) and as a LangChain
StructuredTool (rag_tool, the LLM-facing interface a tool-calling agent invokes)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from langchain_core.tools import StructuredTool

from src.generator import generate
from src.retriever import retrieve


@dataclass
class RagResult:
    answer: str
    chunks: list[dict]
    citations: list[str]
    confidence: float

    def asdict(self) -> dict:
        return {"answer": self.answer, "chunks": self.chunks,
                "citations": self.citations, "confidence": self.confidence}


def rag_v1_complete(question: str, query: Optional[str] = None) -> RagResult:
    """Retrieve for `query` (defaults to `question`), then generate a grounded,
    cited answer to `question`."""
    search = query if query is not None else question
    chunks = retrieve(search)
    out = generate(question, search, chunks)
    return RagResult(answer=out["answer"], chunks=chunks,
                     citations=out["citations"], confidence=out["confidence"])


def _rag_tool_fn(query: str) -> str:
    """LLM-facing entry: returns a compact JSON string (answer + citations + doc ids)."""
    res = rag_v1_complete(query)
    return json.dumps({
        "answer": res.answer,
        "citations": res.citations,
        "doc_ids": [c["doc_id"] for c in res.chunks],
    })


rag_tool = StructuredTool.from_function(
    func=_rag_tool_fn,
    name="rag_v1_complete",
    description=("Retrieve grounded, cited answers from the Helios knowledge base. "
                 "Input: a search query string. Use when a question needs facts "
                 "from the knowledge base."),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rag_tool.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/rag_tool.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_rag_tool.py
git commit -m "feat(5.3-agentic-rag): wire rag_v1_complete as a LangChain StructuredTool"
```

---

## Task 9: `decision.py` (retrieve vs answer-from-context gate)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/decision.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_decision.py`

- [ ] **Step 1: Write the failing test** (`tests/test_decision.py`)

```python
from src.decision import decide, Decision, DIRECT_MARKERS


def test_factual_kb_question_routes_retrieve():
    d = decide("What is the Helios query quota per day?")
    assert isinstance(d, Decision)
    assert d.route == "retrieve"


def test_self_contained_context_routes_direct():
    d = decide("Given that the pro tier allows 100000 queries per day, is 80000 within that limit?")
    assert d.route == "direct"
    assert d.reason


def test_meta_question_routes_direct():
    assert decide("Could you restate your previous answer more concisely?").route == "direct"


def test_markers_present():
    assert "given that" in DIRECT_MARKERS
    assert "previous answer" in DIRECT_MARKERS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decision.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.decision'`)

- [ ] **Step 3: Write `src/decision.py`**

```python
"""The 'should I retrieve?' gate.

Route `direct` when the question carries its own context (a self-contained
premise) or is meta/about a prior turn — retrieval would add nothing. Otherwise
route `retrieve`. Offline: a deterministic marker heuristic. With USE_LLM=1 a
classifier LLM decides instead (same Decision contract)."""
from __future__ import annotations

from dataclasses import dataclass

from src import config

DIRECT_MARKERS: tuple[str, ...] = (
    "given that", "given the", "assuming", "based on the following",
    "based on the above", "as shown above", "previous answer", "previously",
    "restate", "rephrase", "summarize your", "more concisely", "translate your",
    "you just said", "earlier you", "your last answer",
)


@dataclass
class Decision:
    route: str   # "retrieve" | "direct"
    reason: str


def decide(question: str) -> Decision:
    if config.use_llm():
        return _decide_llm(question)
    low = question.lower()
    for marker in DIRECT_MARKERS:
        if marker in low:
            return Decision("direct", f"matched self-contained/meta marker '{marker}'")
    return Decision("retrieve", "factual question with no self-contained context")


def _decide_llm(question: str) -> Decision:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=__import__("os").environ["OPENROUTER_API_KEY"],
                     temperature=0, max_tokens=5, timeout=30)
    system = ("Reply with one word: RETRIEVE if answering needs external knowledge-base "
              "facts, or DIRECT if the question is self-contained or about a prior turn.")
    verdict = str(llm.invoke([("system", system), ("human", question)]).content).strip().upper()
    route = "direct" if "DIRECT" in verdict else "retrieve"
    return Decision(route, f"LLM classifier -> {verdict}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decision.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/decision.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_decision.py
git commit -m "feat(5.3-agentic-rag): retrieve-vs-direct decision gate"
```

---

## Task 10: `reflexion.py` (query refinement from failure diagnoses)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/reflexion.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_reflexion.py`

- [ ] **Step 1: Write the failing test** (`tests/test_reflexion.py`)

```python
from src.reflexion import refine, SYNONYM_MAP
from src.retriever import retrieve


def test_refine_expands_casual_terms_to_formal_synonyms():
    q = "How long does Helios keep documents before deleting them?"
    refined = refine(q, attempts_log=[{"query": q, "faithfulness": 0.0,
                                       "diagnosis": "WRONG_CHUNKS"}])
    low = refined.lower()
    assert "retention" in low or "retained" in low   # keep -> retain/retention
    assert "purged" in low                            # deleting -> purged


def test_refined_query_sharpens_retrieval_onto_owner_doc():
    q = "How long does Helios keep documents before deleting them?"
    refined = refine(q, attempts_log=[{"query": q, "faithfulness": 0.0,
                                       "diagnosis": "WRONG_CHUNKS"}])
    assert retrieve(refined)[0]["doc_id"] == "D1"


def test_refine_differs_from_previous_query():
    q = "What is the Helios carbon footprint per query?"  # no synonyms available
    prev = [{"query": q, "faithfulness": 0.0, "diagnosis": "WRONG_CHUNKS"}]
    refined = refine(q, attempts_log=prev)
    assert refined != q   # must change so the loop makes progress (even if unhelpful)


def test_synonym_map_is_generic_vocabulary_bridging():
    assert "keep" in SYNONYM_MAP and "retention" in SYNONYM_MAP["keep"]
    assert "cost" in SYNONYM_MAP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reflexion.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.reflexion'`)

- [ ] **Step 3: Write `src/reflexion.py`**

```python
"""Reflexion: rewrite the search query after an ungrounded attempt.

Offline: a generic casual->formal SYNONYM_MAP bridges the vocabulary gap that
defeats tf-idf retrieval (the documented root cause). It is NOT keyed to answers
— it maps everyday verbs to the formal register documentation tends to use. The
refined query is guaranteed to differ from the last one so the loop progresses;
when no synonym applies it broadens with a generic widening token. With USE_LLM=1
an LLM rewrites the query instead."""
from __future__ import annotations

from src import config
from src.text import content_tokens

SYNONYM_MAP: dict[str, list[str]] = {
    "keep": ["retained", "retention"], "kept": ["retained", "retention"],
    "store": ["retained", "retention"], "stored": ["retained", "retention"],
    "hold": ["retained", "retention"], "retain": ["retention"],
    "delete": ["purged"], "deleting": ["purged"], "deleted": ["purged"],
    "remove": ["purged"], "erase": ["purged"], "expire": ["purged"],
    "cost": ["pricing", "billed"], "costs": ["pricing", "billed"],
    "price": ["pricing", "billed"], "fee": ["pricing", "billed"],
    "expensive": ["pricing", "billed"], "much": ["pricing"],
    "limit": ["quota"], "cap": ["quota"], "allowance": ["quota"],
    "allow": ["quota"], "allows": ["quota"], "many": ["quota"],
    "embed": ["embedding", "model"], "vector": ["embedding", "vectors"],
}
_WIDEN = "details overview"   # generic broadening when no synonym applies


def refine(question: str, attempts_log: list[dict]) -> str:
    if config.use_llm():
        return _refine_llm(question, attempts_log)

    terms = content_tokens(question)
    expanded: list[str] = list(terms)
    added = False
    for t in terms:
        for syn in SYNONYM_MAP.get(t, []):
            if syn not in expanded:
                expanded.append(syn)
                added = True

    refined = " ".join(expanded)
    last = attempts_log[-1]["query"] if attempts_log else question
    if not added or refined == last:
        # guarantee progress even when no synonym helps
        refined = f"{refined} {_WIDEN}".strip()
    return refined


def _refine_llm(question: str, attempts_log: list[dict]) -> str:
    from langchain_openai import ChatOpenAI

    history = "\n".join(
        f"- tried '{a['query']}' -> faithfulness {a.get('faithfulness'):.2f} "
        f"({a.get('diagnosis')})" for a in attempts_log)
    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=__import__("os").environ["OPENROUTER_API_KEY"],
                     temperature=0.2, max_tokens=40, timeout=30)
    system = ("Rewrite the user's question into a better keyword search query that "
              "would retrieve grounding evidence. Avoid the failed queries below. "
              "Reply with ONLY the query.\n" + history)
    return str(llm.invoke([("system", system), ("human", question)]).content).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reflexion.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/reflexion.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_reflexion.py
git commit -m "feat(5.3-agentic-rag): reflexion query refinement (casual->formal expansion)"
```

---

## Task 11: `nodes.py` (node functions + routers)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/nodes.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_nodes.py`

- [ ] **Step 1: Write the failing test** (`tests/test_nodes.py`)

```python
from src.state import initial_state
from src import nodes


def test_decide_node_sets_route_and_query():
    out = nodes.decide_node(initial_state("What is the Helios query quota per day?"))
    assert out["route"] == "retrieve"
    assert out["query"] == "What is the Helios query quota per day?"


def test_retrieve_and_generate_increments_attempt_and_fills_answer():
    s = initial_state("What is the Helios query quota per day?")
    out = nodes.retrieve_and_generate_node(s)
    assert out["attempt"] == 1
    assert out["chunks"][0]["doc_id"] == "D3"
    assert "quota" in out["answer"].lower()


def test_evaluate_node_appends_attempt_and_scores():
    s = initial_state("What is the Helios query quota per day?")
    s.update(nodes.retrieve_and_generate_node(s))
    out = nodes.evaluate_node(s)
    assert out["faithfulness"] >= 0.70
    assert len(out["attempts_log"]) == 1
    assert out["attempts_log"][0]["diagnosis"] == "GROUNDED"


def test_route_after_eval_finalizes_when_faithful():
    s = initial_state("q")
    s["faithfulness"] = 0.9
    s["attempt"] = 1
    assert nodes.route_after_eval(s) == "finalize"


def test_route_after_eval_refines_then_falls_back():
    s = initial_state("q", max_retries=2)
    s["faithfulness"] = 0.0
    s["attempt"] = 1
    assert nodes.route_after_eval(s) == "refine"
    s["attempt"] = 2
    assert nodes.route_after_eval(s) == "refine"
    s["attempt"] = 3
    assert nodes.route_after_eval(s) == "fallback"


def test_fallback_selects_best_attempt_not_last():
    s = initial_state("q")
    s["attempts_log"] = [
        {"query": "a", "answer": "AAA", "faithfulness": 0.2, "diagnosis": "WRONG_CHUNKS"},
        {"query": "b", "answer": "BBB", "faithfulness": 0.6, "diagnosis": "UNSUPPORTED_GENERATION"},
        {"query": "c", "answer": "CCC", "faithfulness": 0.1, "diagnosis": "WRONG_CHUNKS"},
    ]
    out = nodes.fallback_node(s)
    assert out["status"] == "fallback"
    assert out["served"] is False
    assert out["faithfulness"] == 0.6        # best, not the last (0.1)


def test_finalize_selects_best_attempt_answer():
    s = initial_state("q")
    s["attempts_log"] = [
        {"query": "a", "answer": "low", "faithfulness": 0.5, "diagnosis": "UNSUPPORTED_GENERATION"},
        {"query": "b", "answer": "high", "faithfulness": 0.9, "diagnosis": "GROUNDED"},
    ]
    out = nodes.finalize_node(s)
    assert out["status"] == "answered"
    assert out["served"] is True
    assert out["final_answer"] == "high"


def test_direct_node_serves_without_retrieval():
    out = nodes.direct_node(initial_state("Given that X, is Y true?"))
    assert out["status"] == "direct"
    assert out["served"] is True
    assert out["final_answer"]


def test_nodes_never_raise_on_bad_state():
    # a malformed state must degrade, not crash (defensive node contract)
    out = nodes.retrieve_and_generate_node({"question": "q", "query": None, "attempt": 0})
    assert out["attempt"] == 1            # still advances; answer degrades to refusal/empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nodes.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.nodes'`)

- [ ] **Step 3: Write `src/nodes.py`**

```python
"""Graph nodes + routers.

Every node returns a partial-state dict and NEVER raises — risky work is trapped
so a failure degrades into the graceful-fallback path instead of crashing the
graph. Routers are pure: they read state and return an edge key."""
from __future__ import annotations

import logging

from src import config
from src.decision import decide
from src.faithfulness import score
from src.generator import REFUSAL
from src.rag_tool import rag_v1_complete
from src.reflexion import refine
from src.state import AgenticRAGState

log = logging.getLogger("agentic_rag")


def decide_node(state: AgenticRAGState) -> dict:
    question = state["question"]
    try:
        d = decide(question)
        return {"route": d.route, "decision_reason": d.reason, "query": question,
                "log": [f"decide: {d.route} ({d.reason})"]}
    except Exception as exc:  # noqa: BLE001 - degrade to retrieve
        return {"route": "retrieve", "decision_reason": f"decide error: {exc}",
                "query": question, "log": [f"decide: ERROR {exc} -> retrieve"]}


def retrieve_and_generate_node(state: AgenticRAGState) -> dict:
    attempt = state.get("attempt", 0) + 1
    question = state["question"]
    query = state.get("query") or question
    try:
        res = rag_v1_complete(question, query)
        return {"attempt": attempt, "chunks": res.chunks, "answer": res.answer,
                "citations": res.citations,
                "log": [f"retrieve+generate: attempt {attempt}, query={query!r}, "
                        f"{len(res.chunks)} chunk(s), conf={res.confidence:.2f}"]}
    except Exception as exc:  # noqa: BLE001 - degrade to empty retrieval
        return {"attempt": attempt, "chunks": [], "answer": REFUSAL, "citations": [],
                "log": [f"retrieve+generate: ERROR {exc} (attempt {attempt})"]}


def evaluate_node(state: AgenticRAGState) -> dict:
    question, answer, chunks = state["question"], state["answer"], state["chunks"]
    try:
        rep = score(question, answer, chunks)
    except Exception as exc:  # noqa: BLE001 - treat as worst case
        from src.faithfulness import FaithReport
        rep = FaithReport(0.0, 0, 0, [answer], "WRONG_CHUNKS")
        log.info("evaluate error: %s", exc)
    entry = {"query": state.get("query", question), "answer": answer,
             "faithfulness": rep.score, "diagnosis": rep.failure_mode,
             "unsupported": rep.unsupported_claims}
    return {"faithfulness": rep.score, "eval_report": rep.asdict(),
            "attempts_log": [entry],
            "log": [f"evaluate: faithfulness={rep.score:.2f} mode={rep.failure_mode}"]}


def refine_node(state: AgenticRAGState) -> dict:
    question = state["question"]
    try:
        nq = refine(question, state.get("attempts_log", []))
        return {"query": nq, "log": [f"refine: new query={nq!r}"]}
    except Exception as exc:  # noqa: BLE001
        return {"query": question, "log": [f"refine: ERROR {exc}"]}


def direct_node(state: AgenticRAGState) -> dict:
    question = state["question"]
    try:
        if config.use_llm():
            ans = _direct_llm(question)
        else:
            ans = ("Answered directly from the context contained in your question; "
                   "no knowledge-base retrieval was required.")
        return {"answer": ans, "final_answer": ans, "status": "direct", "served": True,
                "log": ["direct: answered without retrieval"]}
    except Exception as exc:  # noqa: BLE001
        ans = "I could not answer this directly."
        return {"answer": ans, "final_answer": ans, "status": "direct", "served": True,
                "log": [f"direct: ERROR {exc}"]}


def _best(attempts_log: list[dict]) -> dict | None:
    return max(attempts_log, key=lambda a: a.get("faithfulness", 0.0)) if attempts_log else None


def finalize_node(state: AgenticRAGState) -> dict:
    best = _best(state.get("attempts_log", [])) or {"answer": state.get("answer", ""),
                                                     "faithfulness": state.get("faithfulness", 0.0)}
    return {"final_answer": best["answer"], "answer": best["answer"], "status": "answered",
            "served": True, "faithfulness": best["faithfulness"],
            "log": [f"finalize: served best attempt (faithfulness={best['faithfulness']:.2f})"]}


def fallback_node(state: AgenticRAGState) -> dict:
    best = _best(state.get("attempts_log", []))
    if best is None:
        msg = REFUSAL
        faith, mode = 0.0, "NO_RETRIEVAL"
    else:
        faith, mode = best["faithfulness"], best.get("diagnosis", "WRONG_CHUNKS")
        msg = (f"{REFUSAL} After {len(state.get('attempts_log', []))} attempt(s) the "
               f"best answer scored faithfulness {faith:.2f} (failure mode: {mode}), "
               f"below the {state['faithfulness_threshold']:.2f} bar — abstaining.")
    return {"final_answer": msg, "status": "fallback", "served": False,
            "faithfulness": faith,
            "eval_report": {"best_attempt": best, "failure_mode": mode},
            "log": [f"fallback: abstained (best faithfulness={faith:.2f}, mode={mode})"]}


def _direct_llm(question: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=__import__("os").environ["OPENROUTER_API_KEY"],
                     temperature=0, max_tokens=300, timeout=45)
    return str(llm.invoke([("system", "Answer the question directly and concisely."),
                           ("human", question)]).content).strip()


# ---- routers (pure: read state, return an edge key) ----

def route_decision(state: AgenticRAGState) -> str:
    return state["route"]


def route_after_eval(state: AgenticRAGState) -> str:
    if state["faithfulness"] >= state["faithfulness_threshold"]:
        return "finalize"
    if state["attempt"] <= state["max_retries"]:
        return "refine"
    return "fallback"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_nodes.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/nodes.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_nodes.py
git commit -m "feat(5.3-agentic-rag): graph nodes + pure routers (best-attempt selection)"
```

---

## Task 12: `graph.py` (assemble + compile the StateGraph)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/graph.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_graph.py`

- [ ] **Step 1: Write the failing test** (`tests/test_graph.py`)

```python
from src.graph import build_graph, run_agent


def test_graph_has_expected_nodes():
    g = build_graph().get_graph()
    names = set(g.nodes)
    for n in ("decide", "retrieve_and_generate", "evaluate", "refine",
              "direct_answer", "fallback", "finalize"):
        assert n in names


def test_first_try_faithful_no_reflexion():
    final = run_agent("What is the Helios query quota per day?")
    assert final["status"] == "answered"
    assert final["served"] is True
    assert final["faithfulness"] >= 0.70
    assert len(final["attempts_log"]) == 1          # no reflexion needed


def test_reflexion_loop_fires_then_succeeds():
    final = run_agent("How long does Helios keep documents before deleting them?")
    assert len(final["attempts_log"]) >= 2          # reflexion fired
    assert final["status"] == "answered"
    assert final["served"] is True
    assert final["faithfulness"] >= 0.70
    assert final["attempts_log"][0]["faithfulness"] < final["faithfulness"]


def test_direct_route_skips_retrieval():
    final = run_agent("Given that the pro tier allows 100000 queries per day, is 80000 within that limit?")
    assert final["status"] == "direct"
    assert final["attempts_log"] == []              # never retrieved


def test_unanswerable_exhausts_retries_then_falls_back():
    final = run_agent("What is the Helios carbon footprint per query?")
    assert final["status"] == "fallback"
    assert final["served"] is False
    assert final["attempt"] == 3                    # 1 initial + exactly 2 retries
    assert len(final["attempts_log"]) == 3


def test_fallback_reports_best_attempt():
    final = run_agent("What is the Helios carbon footprint per query?")
    best = max(a["faithfulness"] for a in final["attempts_log"])
    assert final["faithfulness"] == best            # best, not last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.graph'`)

- [ ] **Step 3: Write `src/graph.py`**

```python
"""Assemble the agentic-RAG StateGraph and compile it.

  START -> decide --(retrieve)--> retrieve_and_generate -> evaluate
                                       ^                       |
                                       | refine  <------------ route_after_eval --(finalize)--> finalize -> END
                                       |                       |--(fallback)--> fallback -> END
        decide --(direct)--> direct_answer -> END

The refine -> retrieve_and_generate back-edge makes this a Directed *Cyclic*
Graph; the `attempt <= max_retries` guard in route_after_eval keeps the cycle
finite (recursion_limit is only a backstop)."""
from __future__ import annotations

import uuid
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from src import nodes
from src.state import (
    AgenticRAGState,
    DEFAULT_MAX_RETRIES,
    FAITHFULNESS_THRESHOLD,
    initial_state,
)


def build_graph(checkpointer: Optional[MemorySaver] = None):
    builder = StateGraph(AgenticRAGState)
    builder.add_node("decide", nodes.decide_node)
    builder.add_node("retrieve_and_generate", nodes.retrieve_and_generate_node)
    builder.add_node("evaluate", nodes.evaluate_node)
    builder.add_node("refine", nodes.refine_node)
    builder.add_node("direct_answer", nodes.direct_node)
    builder.add_node("finalize", nodes.finalize_node)
    builder.add_node("fallback", nodes.fallback_node)

    builder.add_edge(START, "decide")
    builder.add_conditional_edges("decide", nodes.route_decision,
                                  {"retrieve": "retrieve_and_generate",
                                   "direct": "direct_answer"})
    builder.add_edge("retrieve_and_generate", "evaluate")
    builder.add_conditional_edges("evaluate", nodes.route_after_eval,
                                  {"finalize": "finalize", "refine": "refine",
                                   "fallback": "fallback"})
    builder.add_edge("refine", "retrieve_and_generate")
    builder.add_edge("direct_answer", END)
    builder.add_edge("finalize", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def run_agent(
    question: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
    thread_id: Optional[str] = None,
    recursion_limit: int = 25,
) -> AgenticRAGState:
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or f"arag-{uuid.uuid4().hex[:8]}"},
              "recursion_limit": recursion_limit}
    seed = initial_state(question, max_retries, faithfulness_threshold)
    try:
        return graph.invoke(seed, config)  # type: ignore[return-value]
    except GraphRecursionError:
        return graph.get_state(config).values  # type: ignore[return-value]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph.py -v`
Expected: PASS (6 passed)

If `test_reflexion_loop_fires_then_succeeds` fails because the retention question grounded on the first try (D1 ranked first and confidence ≥ GROUND_MIN), it means `documents` alone cleared the bar — verify D5 also contains `documents` (Task 3) so the first query is ambiguous; the design keeps first-attempt confidence below `GROUND_MIN=1.6` because only `documents` (idf `log(7/2)≈1.25`) matches. Do not lower the threshold to force it; the corpus is the control.

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/graph.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_graph.py
git commit -m "feat(5.3-agentic-rag): compile the agentic-RAG StateGraph + run_agent"
```

---

## Task 13: `eval_dataset.json` + `ragas_eval.py` (full-pipeline RAGAS eval)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/eval_dataset.json`
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/ragas_eval.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_ragas_eval.py`

- [ ] **Step 1: Create `eval_dataset.json`**

```json
[
  {"id": "q1", "question": "What is the Helios query quota per day?", "expected_route": "retrieve", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q2", "question": "Which embedding model does Helios use for collections?", "expected_route": "retrieve", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q3", "question": "What is the Helios pro tier pricing?", "expected_route": "retrieve", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q4", "question": "What is the Helios retention period in days?", "expected_route": "retrieve", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q5", "question": "What embedding vectors does Helios produce at creation?", "expected_route": "retrieve", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q6", "question": "How long does Helios keep documents before deleting them?", "expected_route": "retrieve", "expects_reflexion": true, "expects_fallback": false},
  {"id": "q7", "question": "Given that the pro tier allows 100000 queries per day, is 80000 within that limit?", "expected_route": "direct", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q8", "question": "Could you restate your previous answer more concisely?", "expected_route": "direct", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q9", "question": "Assuming retention is 90 days, when is a day-1 document purged?", "expected_route": "direct", "expects_reflexion": false, "expects_fallback": false},
  {"id": "q10", "question": "What is the Helios carbon footprint per query?", "expected_route": "retrieve", "expects_reflexion": true, "expects_fallback": true}
]
```

- [ ] **Step 2: Write the failing test** (`tests/test_ragas_eval.py`)

```python
from src.ragas_eval import evaluate_dataset, format_table
from src.state import DATASET_TARGET


def test_full_pipeline_mean_faithfulness_meets_target():
    m = evaluate_dataset()
    assert m["mean_faithfulness"] >= DATASET_TARGET   # headline goal: >= 0.75


def test_route_accuracy_is_perfect_on_curated_set():
    assert evaluate_dataset()["route_accuracy"] == 1.0


def test_reflexion_and_fallback_rows_match_expectations():
    rows = {r["id"]: r for r in evaluate_dataset()["rows"]}
    assert rows["q6"]["reflexion_fired"] is True
    assert rows["q6"]["status"] == "answered"
    assert rows["q10"]["status"] == "fallback"
    assert rows["q10"]["served"] is False


def test_format_table_is_readable_string():
    table = format_table(evaluate_dataset())
    assert "faithfulness" in table.lower()
    assert "q6" in table
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_ragas_eval.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.ragas_eval'`)

- [ ] **Step 4: Write `src/ragas_eval.py`**

```python
"""Run the FULL agentic pipeline over eval_dataset.json and report a RAGAS-style
table. Headline metric: mean faithfulness over served, retrieve-routed answers
(direct answers have no retrieved context to be faithful to; fallbacks abstain)."""
from __future__ import annotations

import json
from pathlib import Path

from src.graph import run_agent

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "eval_dataset.json"


def load_dataset(path: Path | str = _DEFAULT_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_dataset(path: Path | str = _DEFAULT_PATH) -> dict:
    data = load_dataset(path)
    rows: list[dict] = []
    faiths: list[float] = []
    route_hits = 0
    for item in data:
        final = run_agent(item["question"])
        reflexion_fired = len(final["attempts_log"]) > 1
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_route": item["expected_route"],
            "route": final["route"],
            "status": final["status"],
            "served": final["served"],
            "faithfulness": round(float(final["faithfulness"]), 3),
            "reflexion_fired": reflexion_fired,
            "attempts": len(final["attempts_log"]),
        }
        rows.append(row)
        if final["route"] == item["expected_route"]:
            route_hits += 1
        if final["status"] == "answered":      # served, retrieve-routed answers only
            faiths.append(float(final["faithfulness"]))

    n_fallback = sum(1 for r in rows if r["status"] == "fallback")
    n_direct = sum(1 for r in rows if r["status"] == "direct")
    return {
        "rows": rows,
        "mean_faithfulness": round(sum(faiths) / len(faiths), 3) if faiths else 0.0,
        "route_accuracy": round(route_hits / len(rows), 3) if rows else 0.0,
        "n_answered": len(faiths),
        "n_direct": n_direct,
        "n_fallback": n_fallback,
        "reflexion_count": sum(1 for r in rows if r["reflexion_fired"]),
    }


def format_table(metrics: dict) -> str:
    lines = []
    header = f"{'id':<4} {'route':<9} {'status':<9} {'faith':>6} {'refl':>5} {'tries':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in metrics["rows"]:
        lines.append(f"{r['id']:<4} {r['route']:<9} {r['status']:<9} "
                     f"{r['faithfulness']:>6.2f} {('Y' if r['reflexion_fired'] else '.'):>5} "
                     f"{r['attempts']:>5}")
    lines.append("-" * len(header))
    lines.append(f"mean faithfulness (served retrieve answers): {metrics['mean_faithfulness']:.3f}  "
                 f"(target >= 0.75)")
    lines.append(f"route accuracy: {metrics['route_accuracy']:.3f}  | "
                 f"answered={metrics['n_answered']} direct={metrics['n_direct']} "
                 f"fallback={metrics['n_fallback']} reflexion={metrics['reflexion_count']}")
    return "\n".join(lines)


def main() -> None:
    print(format_table(evaluate_dataset()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_ragas_eval.py -v`
Expected: PASS (4 passed)

If `mean_faithfulness` is below 0.75, inspect the table (`python -m src.ragas_eval` from the folder): every `retrieve` row that reached `answered` should show `faith ≈ 1.00`. A sub-0.75 mean means a query meant to ground first-try did not (vocabulary mismatch) — fix the *question wording* or the *corpus term*, never the threshold.

- [ ] **Step 6: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/eval_dataset.json 05-agentic-systems/5.3-Agentic-RAG/src/ragas_eval.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_ragas_eval.py
git commit -m "feat(5.3-agentic-rag): full-pipeline RAGAS eval + curated dataset (mean faithfulness >=0.75)"
```

---

## Task 14: `agentic_rag.py` (entrypoint: demo, CLI, visualization)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/agentic_rag.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_cli.py`

- [ ] **Step 1: Write the failing test** (`tests/test_cli.py`)

```python
from agentic_rag import main


def test_single_question_prints_answer(capsys):
    main(["What is the Helios query quota per day?"])
    out = capsys.readouterr().out.lower()
    assert "quota" in out
    assert "answered" in out or "served" in out


def test_eval_flag_prints_table_and_target(capsys):
    main(["--eval"])
    out = capsys.readouterr().out.lower()
    assert "faithfulness" in out
    assert "0.75" in out


def test_default_demo_runs_all_four_paths(capsys):
    main([])
    out = capsys.readouterr().out.lower()
    for marker in ("reflexion", "direct", "fallback"):
        assert marker in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'agentic_rag'`)

- [ ] **Step 3: Write `agentic_rag.py`**

```python
"""agentic_rag.py — Day 11 (Week 2 Review): a LangGraph agent that uses RAG as a
tool, decides retrieve-vs-direct, and self-corrects via a RAGAS-faithfulness-
gated Reflexion loop (refine if <0.70, max 2 retries, then graceful fallback).

RUN:
  python agentic_rag.py                 # offline demo: 4 paths
  python agentic_rag.py "your question" # single question
  python agentic_rag.py --eval          # full-pipeline RAGAS eval table

Offline & deterministic by default (Qdrant :memory:, tf-idf, extractive
generator, offline faithfulness proxy). USE_LLM=1 / USE_RAGAS=1 swap in real
backends without changing the graph."""
from __future__ import annotations

import os
import sys
from typing import Optional

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Make `src` and this file importable when run directly from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph import build_graph, run_agent          # noqa: E402
from src.ragas_eval import evaluate_dataset, format_table  # noqa: E402


def _rule(title: str = "", width: int = 74) -> str:
    return "-" * width if not title else f"-- {title} " + "-" * max(width - len(title) - 4, 0)


def visualize(mermaid_path: Optional[str] = None) -> None:
    drawable = build_graph().get_graph()
    print(_rule("GRAPH (ASCII)"))
    try:
        print(drawable.draw_ascii())
    except Exception:  # noqa: BLE001 - grandalf can't place the refine self-loop
        print("(ASCII layout can't render the reflexion back-edge; see Mermaid below.)")
    print(_rule("GRAPH (Mermaid)"))
    print(drawable.draw_mermaid().rstrip())
    if mermaid_path:
        with open(mermaid_path, "w", encoding="utf-8") as fh:
            fh.write(drawable.draw_mermaid())


def _print_run(question: str) -> None:
    final = run_agent(question)
    print(_rule(f"Q: {question}"))
    for line in final["log"]:
        print("  " + line)
    print(f"  route={final['route']} status={final['status']} served={final['served']} "
          f"faithfulness={final['faithfulness']:.2f} attempts={len(final['attempts_log'])}")
    print(f"  ANSWER: {final['final_answer']}")


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--eval":
        print(_rule("FULL-PIPELINE RAGAS EVAL"))
        print(format_table(evaluate_dataset()))
        return

    if argv:
        _print_run(" ".join(argv))
        return

    print(_rule("DAY 11 — AGENTIC RAG"))
    here = os.path.dirname(os.path.abspath(__file__))
    visualize(mermaid_path=os.path.join(here, "graph.mmd"))
    print()
    print(_rule("DEMO 1 — first-try faithful (no reflexion)"))
    _print_run("What is the Helios query quota per day?")
    print(_rule("DEMO 2 — reflexion loop (low faithfulness -> refine -> faithful)"))
    _print_run("How long does Helios keep documents before deleting them?")
    print(_rule("DEMO 3 — direct (answer from context, no retrieval)"))
    _print_run("Given that the pro tier allows 100000 queries per day, is 80000 within that limit?")
    print(_rule("DEMO 4 — graceful fallback (unanswerable, retries exhausted)"))
    _print_run("What is the Helios carbon footprint per query?")
    print()
    print(_rule("EVAL"))
    print(format_table(evaluate_dataset()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/agentic_rag.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_cli.py
git commit -m "feat(5.3-agentic-rag): entrypoint (demo, CLI, ASCII/Mermaid viz)"
```

---

## Task 15: `remediation.py` + `test_concepts.py` (the 4 concept goals + common mistakes)

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/src/remediation.py`
- Test: `05-agentic-systems/5.3-Agentic-RAG/tests/test_concepts.py`

- [ ] **Step 1: Write the failing test** (`tests/test_concepts.py`)

```python
"""Each Week-2-Review concept goal -> an explicit assertion, plus the source-note
common mistakes."""
from src.decision import decide
from src.faithfulness import score
from src.generator import REFUSAL
from src.graph import run_agent
from src.rag_tool import rag_tool
from src.remediation import advise, REMEDIATION
from langchain_core.tools import BaseTool


# --- Concept 1: RAG as a TOOL (not a fixed pipeline) ---
def test_concept_rag_is_a_tool():
    assert isinstance(rag_tool, BaseTool) and rag_tool.name == "rag_v1_complete"


# --- Concept 2: when the agent SHOULD vs SHOULDN'T retrieve ---
def test_concept_should_vs_shouldnt_retrieve():
    assert decide("What is the Helios query quota?").route == "retrieve"      # SHOULD
    assert decide("Given that quota is 1000, is 500 under it?").route == "direct"  # SHOULDN'T
    # and the graph honors it: a direct route never touches retrieval
    assert run_agent("Could you rephrase your previous answer?")["attempts_log"] == []


# --- Concept 3: read a RAGAS score table and know what to fix ---
def test_concept_read_ragas_table_to_know_what_to_fix():
    # every failure mode maps to a concrete remediation
    for mode in ("NO_RETRIEVAL", "WRONG_CHUNKS", "UNSUPPORTED_GENERATION", "GROUNDED", "ABSTAINED"):
        assert mode in REMEDIATION
    assert "retriev" in advise("WRONG_CHUNKS").lower()
    assert "generat" in advise("UNSUPPORTED_GENERATION").lower()


# --- Concept 4: trace a hallucination back to its ROOT failure mode ---
def test_concept_trace_retrieval_failure_vs_generation_failure():
    d3 = [{"doc_id": "D3", "label": "Doc 1", "score": 0.9, "rank": 1,
           "text": "Helios query quota: the free tier permits one thousand queries each day."}]
    # retrieval root cause: question grounds weakly, answer guesses
    weak = [{"doc_id": "D6", "label": "Doc 1", "score": 0.3, "rank": 1,
             "text": "The Helios dashboard displays latency charts."}]
    r_retrieval = score("How long does Helios keep documents?",
                        "keep documents deleting is described in the retrieved context [Doc 1].", weak)
    assert r_retrieval.failure_mode == "WRONG_CHUNKS"
    # generation root cause: chunk grounds the question, but answer adds an unsupported claim
    r_generation = score("What is the Helios query quota each day?",
                         "The quota is one thousand queries each day but unlimited on weekends [Doc 1].",
                         d3)
    assert r_generation.failure_mode == "UNSUPPORTED_GENERATION"


# --- Common mistake 1: unbounded retry loop ---
def test_mistake_retry_loop_is_bounded():
    final = run_agent("What is the Helios carbon footprint per query?")
    assert final["attempt"] <= 3            # 1 initial + max 2 retries


# --- Common mistake 2: serving the LAST attempt instead of the BEST ---
def test_mistake_serve_best_not_last():
    final = run_agent("What is the Helios carbon footprint per query?")
    assert final["faithfulness"] == max(a["faithfulness"] for a in final["attempts_log"])


# --- Common mistake 3: a correct refusal mis-scored as a hallucination ---
def test_mistake_refusal_is_abstained_not_hallucination():
    chunks = [{"doc_id": "D1", "label": "Doc 1", "score": 0.9, "rank": 1, "text": "Helios retention."}]
    assert score("q", REFUSAL, chunks).failure_mode == "ABSTAINED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_concepts.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.remediation'`)

- [ ] **Step 3: Write `src/remediation.py`**

```python
"""Map each RAGAS faithfulness failure mode to the concrete fix — the 'read a
RAGAS score table and know what to fix' skill, encoded."""
from __future__ import annotations

REMEDIATION: dict[str, str] = {
    "GROUNDED": "No action — every claim is supported by the retrieved context.",
    "ABSTAINED": "No action — a correct refusal is the safety valve working, not a bug.",
    "NO_RETRIEVAL": ("Fix RETRIEVAL: the query produced no in-vocabulary signal. Expand/"
                     "rewrite the query (reflexion), or ingest the missing documents."),
    "WRONG_CHUNKS": ("Fix RETRIEVAL: the retrieved chunks do not ground the question "
                     "(vocabulary mismatch / poor recall). Improve the query, embeddings, "
                     "or index — this is what the reflexion loop targets."),
    "UNSUPPORTED_GENERATION": ("Fix GENERATION: the chunks contain the answer but the model "
                               "added claims not in the context. Tighten the context-only "
                               "prompt and keep temperature at 0."),
    "UNCITED": "Fix GENERATION: require inline [Doc N] citations for every claim.",
}


def advise(failure_mode: str) -> str:
    return REMEDIATION.get(failure_mode, "Unknown failure mode — inspect the eval report.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_concepts.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/src/remediation.py 05-agentic-systems/5.3-Agentic-RAG/tests/test_concepts.py
git commit -m "feat(5.3-agentic-rag): failure-mode remediation map + concept/mistake tests"
```

---

## Task 16: `README.md` + update module `05-agentic-systems/README.md`

**Files:**
- Create: `05-agentic-systems/5.3-Agentic-RAG/README.md`
- Modify: `05-agentic-systems/README.md` (add ✅ 5.3 row; renumber planned 5.3→5.4, 5.4→5.5, 5.5→5.6)

- [ ] **Step 1: Write `05-agentic-systems/5.3-Agentic-RAG/README.md`**

Write the README modeled on Day 9/10 (objective, architecture diagram, the five-ish concepts, goal→evidence mapping, run instructions, offline/optional-backend note, verification table). Include the exact diagram below and a goal→evidence table.

````markdown
# Day 11 — Agentic RAG (Week 2 Review)

A LangGraph agent that uses **RAG as a tool it decides whether to call**, then
**self-corrects** when its answer isn't grounded — gated on a RAGAS faithfulness
score. Offline & deterministic by default; `USE_LLM=1` / `USE_RAGAS=1` swap in
real OpenRouter generation and the real `ragas` library.

## Architecture

```
START -> decide --(retrieve)--> retrieve_and_generate -> evaluate
                                      ^                       |
                              refine  |        route_after_eval
                                      |         |       |        |
                                      +--<------(refine)|        (finalize)--> finalize -> END
                                                        (fallback)----------> fallback -> END
        decide --(direct)--> direct_answer ------------------------------------------> END
```

- **decide** — SHOULD I retrieve? (factual KB question) vs answer **direct**
  (self-contained premise / meta question).
- **retrieve_and_generate** — calls the `rag_v1_complete` tool (tf-idf retrieval
  in Qdrant `:memory:` → grounded, cited answer).
- **evaluate** — RAGAS faithfulness of the answer vs its retrieved context, plus a
  root **failure-mode** diagnosis.
- **refine** (Reflexion) — rewrites the query from accumulated failure diagnoses.
- **fallback** — graceful abstention that reports the **best-scoring** attempt.

## Two thresholds

- `0.70` — reflexion trigger / serve gate (re-query if faithfulness < 0.70).
- `0.75` — the **mean** faithfulness over served, retrieve-routed answers on the
  eval dataset (the headline goal).

## The deterministic reflexion mechanism

The corpus uses *formal* vocabulary (`retention`, `purged`, `quota`); some
questions use *casual* vocabulary (`keep`, `deleting`). With `idf = log(N/df)`,
corpus-ubiquitous terms (`helios`) contribute 0, so a casually-worded question
grounds below `GROUND_MIN`, the generator emits an unsupported guess, faithfulness
drops, and `refine` expands casual→formal — sharpening retrieval onto the owning
document. Vocabulary mismatch is *the* classic retrieval failure; query expansion
is the classic fix. Fully deterministic, zero keys.

## Run

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python agentic_rag.py            # offline demo: 4 paths + eval table
python agentic_rag.py --eval     # full-pipeline RAGAS eval table
python -m pytest -v              # machine-checked goals
```

Optional real backends: copy `.env.example` to `.env`, set `OPENROUTER_API_KEY`
and `USE_LLM=1` (real generation/decision/refine) and/or `USE_RAGAS=1` (real
RAGAS). Topology is identical.

## Goal → evidence

| Goal / task | Where | Test |
|---|---|---|
| Wire `rag_v1_complete` as a tool | `src/rag_tool.py` | `tests/test_rag_tool.py` |
| Agent decides retrieve vs answer-from-context | `src/decision.py`, `decide_node` | `tests/test_decision.py`, `test_concepts.py` |
| Reflexion: refine when faithfulness < 0.70 | `src/reflexion.py`, `route_after_eval` | `tests/test_reflexion.py`, `test_graph.py` |
| Max 2 retries, then graceful fallback | `route_after_eval`, `fallback_node` | `tests/test_graph.py`, `test_concepts.py` |
| Run RAGAS eval on the full pipeline | `src/ragas_eval.py` | `tests/test_ragas_eval.py` |
| Mean faithfulness ≥ 0.75 | `eval_dataset.json` | `tests/test_ragas_eval.py` |
| Read a RAGAS table → what to fix | `src/remediation.py` | `tests/test_concepts.py` |
| Trace hallucination → root failure mode | `src/faithfulness.py` | `tests/test_faithfulness.py`, `test_concepts.py` |

## Honest scope

The eval set is a **day-scale curated set (10 queries)** to exercise every path,
not the 50-pair production set module 08 mandates for final scoring.
````

- [ ] **Step 2: Update `05-agentic-systems/README.md`**

In the "Implementation artifacts" table, change the Day-10/11/12 planned rows and insert Day-11 as done. Replace the three planned rows:

```
| 10 | `5.3-Checkpointing-HITL/` *(planned)* | Durable persistence (SqliteSaver) with `interrupt_before` approval gates and resume-from-interrupt | 🔲 Planned |
| 11 | `5.4-Supervisor-Agents/` *(planned)* | Supervisor/star topology: a router LLM delegating to specialized worker agents that report back | 🔲 Planned |
| 12 | `5.5-Reflexion-Loop/` *(planned)* | Generate → critique → revise loop with episodic memory of past failures | 🔲 Planned |
```

with:

```
| 11 | [`5.3-Agentic-RAG/`](5.3-Agentic-RAG/README.md) | LangGraph agent using RAG as a tool: retrieve-vs-direct decision, a RAGAS-faithfulness-gated Reflexion loop (refine <0.70, max 2 retries), graceful fallback, and a full-pipeline RAGAS eval (mean faithfulness ≥0.75) | ✅ Done |
| 12 | `5.4-Checkpointing-HITL/` *(planned)* | Durable persistence (SqliteSaver) with `interrupt_before` approval gates and resume-from-interrupt | 🔲 Planned |
| 13 | `5.5-Supervisor-Agents/` *(planned)* | Supervisor/star topology: a router LLM delegating to specialized worker agents that report back | 🔲 Planned |
```

Then in the "Verification" table add a row:

```
| 5.3 — Agentic RAG | (run `pytest -v`) | all Day-11 goals/tasks + the 4 review concepts + 3 common mistakes |
```

- [ ] **Step 3: Commit**

```bash
git add 05-agentic-systems/5.3-Agentic-RAG/README.md 05-agentic-systems/README.md
git commit -m "docs(5.3-agentic-rag): day README + module index (renumber planned days)"
```

---

## Task 17: Update the Obsidian vault (Week 2 Review notes)

Vault root: `C:\Users\minhh\OneDrive\Documents\Obsidian Vault`. Create a new folder `Week 2 Review - Agentic RAG/` with the notes below, then add backlinks into existing Week-2 notes. These files are outside the git repo — **do not `git add` them**; this task has no commit step.

> Cross-links use Obsidian `[[Basename]]` wikilinks (resolved by filename). The referenced basenames exist in the vault (verified): `Conditional Edges — Routing Logic`, `Error Recovery — What Happens When a Node Fails`, `Citation Grounding — The 3-Check Pattern`, `Five Failure Modes — The Hallucination Taxonomy`, `Tool Schema — Name, Description, Parameters`, `Multi-Hop Failure — The Naive RAG Reasoning Gap`.

- [ ] **Step 1: Create `Week 2 Review - Agentic RAG/2026-06-04 task.md`**

```markdown
# 2026-06-04 — Week 2 Review: Agentic RAG

**Deliverable:** `05-agentic-systems/5.3-Agentic-RAG/agentic_rag.py` — a LangGraph
agent that wires `rag_v1_complete` as a tool, decides retrieve-vs-direct, and
self-corrects via a RAGAS-faithfulness-gated Reflexion loop.

## Tasks
- [x] Wire `rag_v1_complete` as a tool in LangGraph
- [x] Agent decides: retrieve? or answer from context?
- [x] Reflexion: if RAGAS faithfulness < 0.70, re-query with a refined question
- [x] Max 2 retry attempts, then graceful fallback
- [x] Run RAGAS eval on the full agentic pipeline (mean faithfulness ≥ 0.75)

## Concepts
- [[Agentic RAG — RAG as a Tool, Not a Pipeline]]
- [[When to Retrieve vs Answer from Context]]
- [[Reflexion Loop — Self-Correction Gated on RAGAS]]
- [[Reading a RAGAS Score Table — What to Fix]]
- [[Tracing a Hallucination to its Root Failure Mode]]

Builds on Week 2: [[Conditional Edges — Routing Logic]],
[[Error Recovery — What Happens When a Node Fails]],
[[Citation Grounding — The 3-Check Pattern]],
[[Five Failure Modes — The Hallucination Taxonomy]].
```

- [ ] **Step 2: Create `Week 2 Review - Agentic RAG/5 Concepts from Week 2 Review - Agentic RAG.md`**

```markdown
# 5 Concepts from Week 2 Review — Agentic RAG

1. [[Agentic RAG — RAG as a Tool, Not a Pipeline]] — retrieval becomes a tool the
   agent *chooses* to call, inside a graph with cycles.
2. [[When to Retrieve vs Answer from Context]] — retrieval is a cost; skip it for
   self-contained or meta questions.
3. [[Reflexion Loop — Self-Correction Gated on RAGAS]] — measure faithfulness,
   refine the query on failure, bound the retries, abstain gracefully.
4. [[Reading a RAGAS Score Table — What to Fix]] — faithfulness/precision/recall
   each point at a different fix.
5. [[Tracing a Hallucination to its Root Failure Mode]] — retrieval failure vs
   generation drift demand opposite fixes.
```

- [ ] **Step 3: Create `Week 2 Review - Agentic RAG/Agentic RAG — RAG as a Tool, Not a Pipeline.md`**

```markdown
# Agentic RAG — RAG as a Tool, Not a Pipeline

Naive RAG is a fixed line: `query → retrieve → generate`. **Agentic RAG** wraps
retrieval as a **tool** inside a [[StateGraph API — Building the Graph Structure|LangGraph state machine]],
so the agent decides *whether*, *with what query*, and *how many times* to call it.

- The RAG call is a LangChain `StructuredTool` (`rag_v1_complete`) — same interface
  an LLM tool-caller would use. See [[Tool Schema — Name, Description, Parameters]].
- The graph has **cycles** ([[DAG vs Cyclic Graphs — Why LangGraph Allows Cycles]]):
  a `refine → retrieve` back-edge enables self-correction a linear pipeline can't.
- Control flow is explicit and testable via [[Conditional Edges — Routing Logic]].

**Why it matters:** retrieval quality is variable; making it a tool lets the agent
react to bad retrieval (re-query) instead of faithfully generating from garbage.
```

- [ ] **Step 4: Create `Week 2 Review - Agentic RAG/When to Retrieve vs Answer from Context.md`**

```markdown
# When to Retrieve vs Answer from Context

Retrieval is not free: latency, tokens, and the risk of injecting distractor
chunks that *lower* faithfulness. The agent's first decision is whether to call
the RAG tool at all.

**Retrieve when:** the question needs facts from the knowledge base.

**Answer directly (don't retrieve) when:**
- the question carries its own premise ("Given that X, is Y…") — it's a
  computation/inference, not a lookup;
- it's about a prior turn ("rephrase your previous answer");
- it's trivial/meta.

Mis-retrieving on a self-contained question can *introduce* a distractor and cause
the very hallucination retrieval was meant to prevent. Related:
[[TopK Pathology — Why Larger ≠ Better]], [[Parametric vs. Retrieved Memory]].
```

- [ ] **Step 5: Create `Week 2 Review - Agentic RAG/Reflexion Loop — Self-Correction Gated on RAGAS.md`**

```markdown
# Reflexion Loop — Self-Correction Gated on RAGAS

**Reflexion** (Shinn et al., 2023): generate → evaluate → on failure, retry with
*verbal* memory of what failed. Here the evaluator is **RAGAS faithfulness**.

```
retrieve_and_generate → evaluate(faithfulness)
   faithfulness ≥ 0.70 → finalize (serve best attempt)
   faithfulness < 0.70 & retries left → refine query → retrieve again
   faithfulness < 0.70 & retries exhausted → graceful fallback (abstain)
```

Three non-negotiables (the common mistakes):
1. **Bound the loop** — max 2 retries, enforced in the router *and* by a recursion
   limit. An unbounded reflexion loop is a runaway cost bug.
2. **Serve the BEST attempt, not the last** — keep episodic memory of every
   attempt's score and return the max.
3. **Abstain, don't serve garbage** — below threshold after retries, return the
   safety valve ([[Abstain vs Answer — The Production Safety Valve]]).

Builds on [[Error Recovery — What Happens When a Node Fails]] and
[[Shared State — How State Accumulates Across Agent Calls]] (the `attempts_log`
reducer is the verbal memory).
```

- [ ] **Step 6: Create `Week 2 Review - Agentic RAG/Reading a RAGAS Score Table — What to Fix.md`**

```markdown
# Reading a RAGAS Score Table — What to Fix

Each RAGAS metric isolates a different stage, so the table tells you *where* to
look:

| Metric | Low means | Fix |
|---|---|---|
| **Faithfulness** | answer not grounded in context (hallucination) | tighten the context-only prompt; lower temperature; add citations |
| **Context precision** | retrieved chunks are irrelevant | better query / reranking / threshold |
| **Context recall** | needed info wasn't retrieved | better embeddings, chunking, or query expansion |
| **Answer relevancy** | answer dodges the question | prompt for directness |

Faithfulness is the trust metric: **hallucination rate ≈ 1 − faithfulness**. Don't
"fix" a low score by lowering the threshold — fix the stage the metric points at.
See [[Confidence Scoring — Multi-Signal Aggregation in Production]] and
[[Measuring Retrieval Improvement]].
```

- [ ] **Step 7: Create `Week 2 Review - Agentic RAG/Tracing a Hallucination to its Root Failure Mode.md`**

```markdown
# Tracing a Hallucination to its Root Failure Mode

A low faithfulness score is a *symptom*. Trace it to one of these roots — they
need opposite fixes:

- **NO_RETRIEVAL** — nothing came back. Query has no signal / corpus gap.
- **WRONG_CHUNKS (retrieval failure)** — chunks returned, but they don't ground
  the question (vocabulary mismatch / poor recall). *Fix retrieval* — this is what
  the reflexion query-rewrite targets.
- **UNSUPPORTED_GENERATION (generation drift)** — the chunks *do* contain the
  answer, but the model added claims beyond them. *Fix generation* — context-only
  prompt, temperature 0.
- **FABRICATED_CITATION** — cites a chunk that doesn't exist. Distrust signal.
- **ABSTAINED** — a correct refusal. Not a hallucination; the system working.

The tell: compare the question's terms against the *retrieved* chunks. If the
answer isn't even retrievable → retrieval bug. If it is, but the answer drifts →
generation bug. Connects [[Five Failure Modes — The Hallucination Taxonomy]] and
[[Diagnosing Which Failure Mode is Active]].
```

- [ ] **Step 8: Add backlinks into existing Week-2 notes**

Append the exact line below to the **end** of each listed note (read the file first, then append). Use the same line for all:

```markdown

> **Week 2 Review:** applied in [[Agentic RAG — RAG as a Tool, Not a Pipeline]] — see [[5 Concepts from Week 2 Review - Agentic RAG]].
```

Files to append to:
- `05-LangGraph (State machines)/Conditional Edges — Routing Logic.md`
- `05-LangGraph Multi-agent/Error Recovery — What Happens When a Node Fails.md`
- `08-Hallucination (Causes + mitigation)/Citation Grounding — The 3-Check Pattern.md`
- `08-Hallucination (Causes + mitigation)/Five Failure Modes — The Hallucination Taxonomy.md`
- `04-Agents ReAct from scratch/Tool Schema — Name, Description, Parameters.md`
- `03-RAG Naive RAG/Multi-Hop Failure — The Naive RAG Reasoning Gap.md`

- [ ] **Step 9: No commit** (vault is outside the repo). Confirm the 7 new files exist and the 6 backlinks were appended.

---

## Task 18: Full-suite verification + branch wrap-up

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite from the day folder**

Run (from `05-agentic-systems/5.3-Agentic-RAG/`): `python -m pytest -v`
Expected: **all tests pass** (~55+), zero failures. Count should cover:
text(3), state(3), corpus(3), embedder(5), retriever(5), generator(3),
faithfulness(5), rag_tool(4), decision(4), reflexion(4), nodes(9), graph(6),
ragas_eval(4), cli(3), concepts(8).

- [ ] **Step 2: Run the entrypoint end-to-end (smoke test)**

Run: `python agentic_rag.py`
Expected: prints the graph (ASCII/Mermaid), four demo runs (faithful / reflexion /
direct / fallback), and the eval table showing `mean faithfulness … >= 0.75` and
`route accuracy: 1.000`. Confirm `graph.mmd` was written.

Run: `python agentic_rag.py --eval`
Expected: just the eval table; mean faithfulness ≥ 0.75.

- [ ] **Step 3: Confirm no stray artifacts are tracked**

Run: `git status`
Expected: clean working tree on `day11-agentic-rag` (no `.venv/`, `__pycache__/`,
`graph.mmd`, or `.env` staged — all covered by `.gitignore`).

- [ ] **Step 4: Verify the goals are met (manual checklist)**

- [ ] `agentic_rag.py` is a LangGraph agent with a RAG tool + self-correction loop.
- [ ] RAGAS faithfulness ≥ 0.75 on the eval dataset (`test_ragas_eval.py` green).
- [ ] All Week 2 Obsidian notes updated (Task 17 done).

- [ ] **Step 5: Finish the branch**

Invoke the `superpowers:finishing-a-development-branch` skill to choose how to
integrate `day11-agentic-rag` (merge to `main`, open a PR, or keep the branch).

---

## Plan self-review

- **Spec coverage:** every spec section maps to a task — tool wiring (T8), decision
  gate (T9), reflexion + thresholds (T10/T11/T12), max-2-retries + fallback (T11/T12),
  RAGAS eval + ≥0.75 (T13), offline determinism (T1–T7), error handling (T11),
  testing incl. concepts + common mistakes (T15), Obsidian (T17), README + renumber
  (T16). ✓
- **Placeholder scan:** no TBD/TODO; every code step has complete code; every test
  step has real assertions. ✓
- **Type/name consistency:** `AgenticRAGState`, `initial_state`, `TfidfSpace`,
  `retrieve`, `generate`, `score`/`FaithReport`, `rag_v1_complete`/`rag_tool`,
  `decide`/`Decision`, `refine`, node names, `route_after_eval`, `run_agent`,
  `evaluate_dataset`/`format_table`, `advise`/`REMEDIATION` — used identically across
  tasks. State keys referenced by nodes all exist in `state.py`. ✓
- **Known risk:** the deterministic thresholds (`GROUND_MIN=1.6`, idf weights) are
  tuned to the corpus; Task 12/13 notes say to fix the *corpus/wording*, not the
  threshold, if a behavior misfires. ✓

