# Graph RAG Entity Extraction (Day 6.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `entity_extractor.py` that takes a document, extracts typed entities and `(entity, relation, entity)` triples, builds and saves a NetworkX knowledge graph, stores entity embeddings in Qdrant, visualizes clear entity clusters, and prints the top-5 most-connected entities.

**Architecture:** Self-contained day folder (`06-graph-rag/6.1-entity-extraction/`) of small, single-responsibility `src/` modules behind a top-level CLI orchestrator. Offline-deterministic by default (spaCy NER + dependency/co-occurrence relations); `USE_LLM=1` swaps in OpenRouter typed-triple extraction. Graph in NetworkX `MultiDiGraph`, communities via NetworkX Louvain, embeddings in local-Docker Qdrant.

**Tech Stack:** Python 3.12, spaCy (`en_core_web_sm`), NetworkX, matplotlib, qdrant-client, openai (OpenRouter), pydantic, rapidfuzz, pymupdf, pytest.

**Working dir for all commands:** `06-graph-rag/6.1-entity-extraction/` unless stated. Use the folder's own `.venv`. On Windows the venv python is `.venv/Scripts/python.exe`; `python` below means that interpreter. Set `PYTHONIOENCODING=utf-8` when running anything that prints entity text.

---

### Task 0: Scaffold the day folder

**Files:**
- Create: `06-graph-rag/6.1-entity-extraction/requirements.txt`
- Create: `06-graph-rag/6.1-entity-extraction/.env.example`
- Create: `06-graph-rag/6.1-entity-extraction/.gitignore`
- Create: `06-graph-rag/6.1-entity-extraction/conftest.py`
- Create: `06-graph-rag/6.1-entity-extraction/src/__init__.py` (empty)
- Create: `06-graph-rag/6.1-entity-extraction/tests/__init__.py` (empty)

- [ ] **Step 1: Create `requirements.txt`**

```
pymupdf==1.27.2.3
spacy==3.8.14
networkx>=3.3
matplotlib>=3.9
numpy>=2.0
qdrant-client>=1.12
openai>=1.50
pydantic>=2.8
rapidfuzz>=3.9
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 2: Create `.env.example`**

```
# All optional. With NONE set, extraction + embeddings run fully offline & deterministic.
# --- real LLM triple extraction instead of the spaCy offline extractor ---
USE_LLM=0
OPENROUTER_API_KEY=
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# --- real semantic entity embeddings (used iff OPENROUTER_API_KEY is set) ---
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
# --- Qdrant ---
QDRANT_URL=http://localhost:6333
ENTITY_COLLECTION=graph_rag_entities
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
# generated artifacts
out/
*.png
!docs/**/*.png
```

- [ ] **Step 4: Create `conftest.py`** (makes `src` importable and provides shared fixtures)

```python
"""Pytest config: put the day folder on sys.path and provide shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SAMPLE_TEXT = (
    "Nelson Liu and John Hewitt study language models at Stanford University. "
    "Percy Liang advises the project at Stanford University. "
    "GPT-3.5-Turbo was evaluated on the NaturalQuestions dataset. "
    "Claude was also evaluated on NaturalQuestions. "
    "Kevin Lin works at the University of California, Berkeley. "
    "Samaya AI collaborated with Stanford University on multi-document question answering."
)


@pytest.fixture
def sample_text() -> str:
    """A few sentences with known, clustered entities (people/orgs/models/datasets)."""
    return SAMPLE_TEXT
```

- [ ] **Step 5: Create empty `src/__init__.py` and `tests/__init__.py`**

```python
```

- [ ] **Step 6: Create the venv and install**

Run (from the day folder):
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -U pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m spacy download en_core_web_sm
```
Expected: installs succeed; `en_core_web_sm` downloads.

- [ ] **Step 7: Commit**

```bash
git add 06-graph-rag/6.1-entity-extraction/requirements.txt 06-graph-rag/6.1-entity-extraction/.env.example 06-graph-rag/6.1-entity-extraction/.gitignore 06-graph-rag/6.1-entity-extraction/conftest.py 06-graph-rag/6.1-entity-extraction/src/__init__.py 06-graph-rag/6.1-entity-extraction/tests/__init__.py
git commit -m "chore(6.1-entity-extraction): scaffold day folder (reqs, env, gitignore, conftest)"
```

---

### Task 1: `schema.py` — typed Pydantic models

**Files:**
- Create: `src/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
from src.schema import Entity, EntityType, Triple, ExtractionResult


def test_entity_normalizes_and_defaults():
    e = Entity(name="  Stanford University  ", type=EntityType.ORG)
    assert e.name == "Stanford University"   # whitespace stripped
    assert e.type == EntityType.ORG
    assert e.aliases == []
    assert e.mentions == 1


def test_triple_holds_relation_and_evidence():
    t = Triple(source="Nelson Liu", relation="works_at", target="Stanford University",
               evidence="Nelson Liu ... at Stanford University.")
    assert t.source == "Nelson Liu"
    assert t.relation == "works_at"
    assert t.target == "Stanford University"
    assert "Stanford" in t.evidence


def test_extraction_result_roundtrips_json():
    r = ExtractionResult(
        entities=[Entity(name="Claude", type=EntityType.MODEL)],
        triples=[Triple(source="Claude", relation="evaluated_on", target="NaturalQuestions")],
    )
    again = ExtractionResult.model_validate_json(r.model_dump_json())
    assert again.entities[0].name == "Claude"
    assert again.triples[0].relation == "evaluated_on"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.schema'`

- [ ] **Step 3: Write `src/schema.py`**

```python
"""Typed data models for entity extraction.

Typed entity labels (not free-text triples) are deliberate: they make the graph
queryable ("all MODELs evaluated on a DATASET") and let disambiguation be
type-aware (never merge a PERSON into an ORG). Every extractor — offline or LLM —
returns the same ExtractionResult, so the rest of the pipeline is backend-agnostic.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    LOCATION = "LOCATION"
    MODEL = "MODEL"
    DATASET = "DATASET"
    CONCEPT = "CONCEPT"
    WORK = "WORK"


class Entity(BaseModel):
    name: str                                   # canonical surface form
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    mentions: int = 1                           # how many times seen in the doc

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("entity name must be non-empty")
        return v


class Triple(BaseModel):
    source: str                                 # entity name
    relation: str                               # verb/relation label
    target: str                                 # entity name
    evidence: str = ""                          # the sentence the triple came from

    @field_validator("source", "target", "relation")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("triple fields must be non-empty")
        return v


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_schema.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/schema.py tests/test_schema.py
git commit -m "feat(6.1): typed Pydantic schema for entities and triples"
```

---

### Task 2: `config.py` — backend gating

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib

from src import config


def test_use_llm_requires_flag_and_key(monkeypatch):
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    importlib.reload(config)
    assert config.use_llm() is False          # flag without key -> offline

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert config.use_llm() is True


def test_use_embeddings_api_tracks_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    importlib.reload(config)
    assert config.use_embeddings_api() is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert config.use_embeddings_api() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Write `src/config.py`**

```python
"""Single source of truth for "real backend vs offline deterministic path".

Components ask here instead of reading os.environ directly, so the offline default
is enforced in exactly one place. Mirrors the convention in 05-agentic-systems/5.3.
"""
from __future__ import annotations

import os

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_EMBEDDING_MODEL = os.environ.get(
    "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
ENTITY_COLLECTION = os.environ.get("ENTITY_COLLECTION", "graph_rag_entities")


def _key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "")


def use_llm() -> bool:
    """True only when explicitly enabled AND a key is present."""
    return os.environ.get("USE_LLM", "0") == "1" and bool(_key())


def use_embeddings_api() -> bool:
    """True when a key is present: real semantic embeddings are then available
    regardless of which extractor produced the entities."""
    return bool(_key())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(6.1): config module gating LLM/embeddings backends"
```

---

### Task 3: `loader.py` — document → pages

**Files:**
- Create: `src/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from src.loader import Page, load_pages, full_text


def test_load_txt(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("Hello world.\nSecond line.", encoding="utf-8")
    pages = load_pages(str(p))
    assert len(pages) == 1
    assert isinstance(pages[0], Page)
    assert pages[0].page_num == 1
    assert "Hello world" in pages[0].text


def test_full_text_joins_pages():
    pages = [Page(page_num=1, text="alpha"), Page(page_num=2, text="beta")]
    text = full_text(pages)
    assert "alpha" in text and "beta" in text


def test_missing_file_raises(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_pages(str(tmp_path / "nope.pdf"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.loader'`

- [ ] **Step 3: Write `src/loader.py`**

```python
"""Load a document into per-page text. Supports PDF (pymupdf) and plain text.

We keep page boundaries because evidence sentences and (later) citations are
more useful when attributable to a page. Relation/sentence segmentation is left
to the extractor, which already runs a spaCy pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    page_num: int          # 1-indexed
    text: str


def load_pages(path: str) -> list[Page]:
    """Return the document as a list of Page. Raises FileNotFoundError if absent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"document not found: {path}")

    if p.suffix.lower() in {".txt", ".md"}:
        return [Page(page_num=1, text=p.read_text(encoding="utf-8", errors="replace"))]

    # default: treat as PDF
    import fitz  # pymupdf; imported lazily so .txt paths don't need it

    doc = fitz.open(str(p))
    try:
        return [Page(page_num=i + 1, text=doc[i].get_text()) for i in range(doc.page_count)]
    finally:
        doc.close()


def full_text(pages: list[Page]) -> str:
    """Concatenate page texts in reading order."""
    return "\n".join(pg.text for pg in pages)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loader.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/loader.py tests/test_loader.py
git commit -m "feat(6.1): document loader (PDF via pymupdf, plain text)"
```

---

### Task 4: `extractor_offline.py` — deterministic spaCy extraction

**Files:**
- Create: `src/extractor_offline.py`
- Test: `tests/test_extractor_offline.py`

**Design note:** spaCy NER provides PERSON/ORG/GPE spans. A small domain lexicon adds
MODEL/DATASET/CONCEPT terms spaCy won't tag (e.g. "GPT-3.5-Turbo", "NaturalQuestions",
"multi-document question answering"). Entities are the union; relations come from
intra-sentence co-occurrence labeled by the sentence's root-verb lemma (falling back
to `mentioned_with`). This is fully deterministic and guarantees connected clusters.

- [ ] **Step 1: Write the failing test**

```python
from src.extractor_offline import extract_offline
from src.loader import Page
from src.schema import EntityType


def test_extracts_known_entities(sample_text):
    result = extract_offline([Page(page_num=1, text=sample_text)])
    names = {e.name for e in result.entities}
    # people and orgs from spaCy NER
    assert any("Stanford" in n for n in names)
    assert any("Nelson Liu" == n or "Nelson" in n for n in names)
    # domain-lexicon types
    types = {e.type for e in result.entities}
    assert EntityType.MODEL in types        # GPT-3.5-Turbo / Claude
    assert EntityType.DATASET in types      # NaturalQuestions


def test_emits_triples_between_cooccurring_entities(sample_text):
    result = extract_offline([Page(page_num=1, text=sample_text)])
    assert len(result.triples) > 0
    # every triple references real entity names
    names = {e.name for e in result.entities}
    for t in result.triples:
        assert t.source in names and t.target in names
        assert t.evidence  # carries the sentence


def test_is_deterministic(sample_text):
    r1 = extract_offline([Page(page_num=1, text=sample_text)])
    r2 = extract_offline([Page(page_num=1, text=sample_text)])
    assert [e.name for e in r1.entities] == [e.name for e in r2.entities]
    assert [(t.source, t.relation, t.target) for t in r1.triples] == \
           [(t.source, t.relation, t.target) for t in r2.triples]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extractor_offline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.extractor_offline'`

- [ ] **Step 3: Write `src/extractor_offline.py`**

```python
"""Deterministic, offline entity + relation extraction with spaCy.

Pipeline:
  1. Run en_core_web_sm over each page's text.
  2. Entities = spaCy NER spans (mapped to our typed labels) UNION a domain
     lexicon for MODEL/DATASET/CONCEPT terms NER misses.
  3. Relations = for each sentence, every unordered pair of distinct entities that
     co-occur becomes a directed Triple in document order, labeled by the sentence's
     root-verb lemma (fallback "mentioned_with"), with the sentence as evidence.

No randomness anywhere -> identical output across runs (asserted in tests).
"""
from __future__ import annotations

import functools
import re

from src.loader import Page
from src.schema import Entity, EntityType, ExtractionResult, Triple

# spaCy NER label -> our typed label. Labels not listed are ignored.
_SPACY_MAP = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORG,
    "GPE": EntityType.LOCATION,
    "LOC": EntityType.LOCATION,
    "FAC": EntityType.LOCATION,
    "PRODUCT": EntityType.MODEL,
    "WORK_OF_ART": EntityType.WORK,
}

# Domain lexicon: phrases spaCy won't tag, matched case-insensitively as whole words.
# Order longest-first so "GPT-3.5-Turbo" wins over a hypothetical "GPT".
_LEXICON: dict[str, EntityType] = {
    "GPT-3.5-Turbo": EntityType.MODEL,
    "GPT-3.5": EntityType.MODEL,
    "GPT-4": EntityType.MODEL,
    "Claude": EntityType.MODEL,
    "MPT-30B": EntityType.MODEL,
    "MPT-30B-Instruct": EntityType.MODEL,
    "LongChat": EntityType.MODEL,
    "LongChat-13B": EntityType.MODEL,
    "NaturalQuestions": EntityType.DATASET,
    "Natural Questions": EntityType.DATASET,
    "multi-document question answering": EntityType.CONCEPT,
    "key-value retrieval": EntityType.CONCEPT,
    "positional bias": EntityType.CONCEPT,
    "in-context learning": EntityType.CONCEPT,
}


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy

    # parser needed for sentence boundaries + root verb; ner for entities.
    return spacy.load("en_core_web_sm")


def _clean(name: str) -> str:
    # collapse whitespace/newlines that PDF extraction injects mid-name
    return re.sub(r"\s+", " ", name).strip(" .,;:")


def _lexicon_hits(text: str) -> list[tuple[int, int, str, EntityType]]:
    """Return (start, end, surface, type) for each lexicon phrase found in text."""
    hits: list[tuple[int, int, str, EntityType]] = []
    for phrase in sorted(_LEXICON, key=len, reverse=True):
        for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE):
            hits.append((m.start(), m.end(), text[m.start():m.end()], _LEXICON[phrase]))
    return hits


def extract_offline(pages: list[Page]) -> ExtractionResult:
    nlp = _nlp()
    # name -> Entity (accumulates mentions); first-seen order preserved by dict
    entities: dict[str, Entity] = {}
    triples: list[Triple] = []
    seen_triples: set[tuple[str, str, str]] = set()

    def _register(name: str, etype: EntityType) -> str | None:
        name = _clean(name)
        if len(name) < 2:
            return None
        if name in entities:
            entities[name].mentions += 1
        else:
            entities[name] = Entity(name=name, type=etype)
        return name

    for page in pages:
        doc = nlp(page.text)
        for sent in doc.sents:
            sent_names: list[str] = []

            # spaCy NER spans in this sentence
            for ent in sent.ents:
                etype = _SPACY_MAP.get(ent.label_)
                if etype is None:
                    continue
                n = _register(ent.text, etype)
                if n:
                    sent_names.append(n)

            # domain lexicon spans in this sentence
            for _s, _e, surface, etype in _lexicon_hits(sent.text):
                n = _register(surface, etype)
                if n:
                    sent_names.append(n)

            # relation label = root verb lemma of the sentence
            root_verbs = [t.lemma_.lower() for t in sent if t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"}]
            relation = root_verbs[0] if root_verbs else "mentioned_with"

            # connect every unordered pair of distinct entities in the sentence
            uniq = list(dict.fromkeys(sent_names))  # de-dup, keep order
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    key = (a, relation, b)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    triples.append(Triple(source=a, relation=relation, target=b,
                                          evidence=_clean(sent.text)))

    return ExtractionResult(entities=list(entities.values()), triples=triples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_extractor_offline.py -v`
Expected: PASS (3 tests). (First run loads the spaCy model; allow a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add src/extractor_offline.py tests/test_extractor_offline.py
git commit -m "feat(6.1): deterministic offline extractor (spaCy NER + co-occurrence)"
```

---

### Task 5: `extractor_llm.py` — OpenRouter typed-triple extraction

**Files:**
- Create: `src/extractor_llm.py`
- Test: `tests/test_extractor_llm.py`

**Design note (the extraction prompt):** The prompt is the heart of LLM Graph RAG.
Design choices, all encoded in `_SYSTEM_PROMPT` below: (1) a fixed typed label set so
the graph is queryable; (2) strict JSON-only output matching our schema so parsing is
deterministic; (3) "extract relations only between entities you also list" to prevent
dangling edges; (4) a one-shot example to anchor format and granularity; (5) an
instruction to use canonical names + collect aliases, pushing first-pass disambiguation
into the model. These map 1:1 to the README's "explain the extraction prompt design".

- [ ] **Step 1: Write the failing test** (no key required — we parse a canned response)

```python
import json

from src.extractor_llm import build_messages, parse_response
from src.schema import EntityType


def test_build_messages_includes_schema_and_text():
    msgs = build_messages("Some document text about Claude.")
    assert msgs[0]["role"] == "system"
    assert "JSON" in msgs[0]["content"]
    assert "Claude" in msgs[1]["content"]


def test_parse_response_extracts_entities_and_triples():
    raw = json.dumps({
        "entities": [
            {"name": "Claude", "type": "MODEL", "aliases": []},
            {"name": "NaturalQuestions", "type": "DATASET", "aliases": ["NQ"]},
        ],
        "triples": [
            {"source": "Claude", "relation": "evaluated_on",
             "target": "NaturalQuestions", "evidence": "Claude was evaluated on NQ."}
        ],
    })
    result = parse_response(raw)
    assert {e.name for e in result.entities} == {"Claude", "NaturalQuestions"}
    assert result.entities[1].type == EntityType.DATASET
    assert result.triples[0].relation == "evaluated_on"


def test_parse_response_tolerates_code_fence():
    raw = "```json\n{\"entities\": [], \"triples\": []}\n```"
    result = parse_response(raw)
    assert result.entities == [] and result.triples == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extractor_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.extractor_llm'`

- [ ] **Step 3: Write `src/extractor_llm.py`**

```python
"""LLM entity + relation extraction via OpenRouter (OpenAI-compatible API).

Used only when USE_LLM=1 and a key is set. The offline extractor is the default.
This module is split into pure helpers (build_messages, parse_response) — unit
testable without any network — and a thin extract_llm() that does the I/O.
"""
from __future__ import annotations

import json
import re

from src import config
from src.loader import Page, full_text
from src.schema import Entity, EntityType, ExtractionResult, Triple

_LABELS = ", ".join(t.value for t in EntityType)

_SYSTEM_PROMPT = f"""You are a precise information-extraction engine that turns prose \
into a knowledge graph.

Extract ENTITIES and RELATIONSHIPS and return STRICT JSON only (no prose, no code \
fence) with this exact shape:
{{"entities": [{{"name": str, "type": str, "aliases": [str]}}],
 "triples": [{{"source": str, "relation": str, "target": str, "evidence": str}}]}}

Rules:
- "type" MUST be one of: {_LABELS}.
- Use a single CANONICAL name per entity; put other surface forms in "aliases"
  (e.g. name "University of California, Berkeley", aliases ["UC Berkeley"]).
- "relation" is a short snake_case verb phrase (e.g. works_at, evaluated_on,
  affiliated_with, authored).
- Only emit a triple whose "source" and "target" both appear in "entities".
- "evidence" is the verbatim sentence the relationship came from.
- Do not invent facts; extract only what the text states.

Example:
Text: "Percy Liang advises the project at Stanford University."
Output: {{"entities": [{{"name": "Percy Liang", "type": "PERSON", "aliases": []}},
{{"name": "Stanford University", "type": "ORG", "aliases": []}}],
"triples": [{{"source": "Percy Liang", "relation": "affiliated_with",
"target": "Stanford University", "evidence": "Percy Liang advises the project at Stanford University."}}]}}
"""


def build_messages(text: str) -> list[dict[str, str]]:
    """Construct the chat messages for one extraction call."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract the knowledge graph from this text:\n\n{text}"},
    ]


def parse_response(raw: str) -> ExtractionResult:
    """Parse the model's JSON (tolerating a ```json fence) into an ExtractionResult.

    Unknown entity types are coerced to CONCEPT rather than crashing the run.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)

    entities: list[Entity] = []
    for e in data.get("entities", []):
        try:
            etype = EntityType(e["type"])
        except (KeyError, ValueError):
            etype = EntityType.CONCEPT
        entities.append(Entity(name=e["name"], type=etype,
                               aliases=list(e.get("aliases", []))))

    triples: list[Triple] = []
    for t in data.get("triples", []):
        triples.append(Triple(source=t["source"], relation=t.get("relation", "related_to"),
                              target=t["target"], evidence=t.get("evidence", "")))

    return ExtractionResult(entities=entities, triples=triples)


def _chunk_pages(pages: list[Page], max_chars: int = 12000) -> list[str]:
    """Group page text into <=max_chars chunks so very long docs stay within context."""
    chunks: list[str] = []
    buf = ""
    for pg in pages:
        if buf and len(buf) + len(pg.text) > max_chars:
            chunks.append(buf)
            buf = ""
        buf += ("\n" + pg.text) if buf else pg.text
    if buf:
        chunks.append(buf)
    return chunks or [full_text(pages)]


def extract_llm(pages: list[Page]) -> ExtractionResult:
    """Run real OpenRouter extraction over the document, merging chunk results."""
    from openai import OpenAI

    import os
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=config.OPENROUTER_BASE_URL)

    merged_entities: dict[str, Entity] = {}
    merged_triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    for i, chunk in enumerate(_chunk_pages(pages), start=1):
        print(f"LLM extraction chunk {i}...")
        resp = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            messages=build_messages(chunk),
            temperature=0,
        )
        part = parse_response(resp.choices[0].message.content or "{}")
        for e in part.entities:
            if e.name in merged_entities:
                merged_entities[e.name].mentions += 1
                for a in e.aliases:
                    if a not in merged_entities[e.name].aliases:
                        merged_entities[e.name].aliases.append(a)
            else:
                merged_entities[e.name] = e
        for t in part.triples:
            key = (t.source, t.relation, t.target)
            if key not in seen:
                seen.add(key)
                merged_triples.append(t)

    return ExtractionResult(entities=list(merged_entities.values()), triples=merged_triples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extractor_llm.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/extractor_llm.py tests/test_extractor_llm.py
git commit -m "feat(6.1): OpenRouter LLM extractor with documented prompt design"
```

---

### Task 6: `extractor.py` — façade routing offline/LLM

**Files:**
- Create: `src/extractor.py`
- Test: `tests/test_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
from src import extractor
from src.loader import Page
from src.schema import Entity, EntityType, ExtractionResult


def test_routes_to_offline_by_default(monkeypatch, sample_text):
    monkeypatch.setattr(extractor.config, "use_llm", lambda: False)
    called = {"offline": False}

    def fake_offline(pages):
        called["offline"] = True
        return ExtractionResult(entities=[Entity(name="X", type=EntityType.CONCEPT)], triples=[])

    monkeypatch.setattr(extractor, "extract_offline", fake_offline)
    extractor.extract([Page(page_num=1, text=sample_text)])
    assert called["offline"] is True


def test_routes_to_llm_when_enabled(monkeypatch, sample_text):
    monkeypatch.setattr(extractor.config, "use_llm", lambda: True)
    called = {"llm": False}

    def fake_llm(pages):
        called["llm"] = True
        return ExtractionResult(entities=[Entity(name="Y", type=EntityType.MODEL)], triples=[])

    monkeypatch.setattr(extractor, "extract_llm", fake_llm)
    extractor.extract([Page(page_num=1, text=sample_text)])
    assert called["llm"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.extractor'`

- [ ] **Step 3: Write `src/extractor.py`**

```python
"""Extraction façade: pick the backend, return a uniform ExtractionResult."""
from __future__ import annotations

from src import config
from src.extractor_llm import extract_llm
from src.extractor_offline import extract_offline
from src.loader import Page
from src.schema import ExtractionResult


def extract(pages: list[Page]) -> ExtractionResult:
    """Route to the LLM extractor when enabled, else the deterministic offline one."""
    if config.use_llm():
        return extract_llm(pages)
    return extract_offline(pages)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extractor.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/extractor.py tests/test_extractor.py
git commit -m "feat(6.1): extractor facade routing offline vs LLM"
```

---

### Task 7: `disambiguation.py` — entity resolution

**Files:**
- Create: `src/disambiguation.py`
- Test: `tests/test_disambiguation.py`

- [ ] **Step 1: Write the failing test**

```python
from src.disambiguation import resolve
from src.schema import Entity, EntityType, ExtractionResult, Triple


def test_merges_alias_pair_and_rewrites_triples():
    result = ExtractionResult(
        entities=[
            Entity(name="University of California, Berkeley", type=EntityType.ORG, mentions=2),
            Entity(name="UC Berkeley", type=EntityType.ORG, mentions=1),
            Entity(name="Kevin Lin", type=EntityType.PERSON),
        ],
        triples=[Triple(source="Kevin Lin", relation="works_at", target="UC Berkeley")],
    )
    out = resolve(result)
    names = {e.name for e in out.entities}
    # the two Berkeley forms collapse to one canonical node
    assert sum(1 for n in names if "Berkeley" in n) == 1
    canonical = next(e for e in out.entities if "Berkeley" in e.name)
    assert "UC Berkeley" in canonical.aliases or canonical.name == "UC Berkeley"
    # the triple was rewritten to the canonical target
    assert out.triples[0].target == canonical.name


def test_does_not_merge_across_types():
    result = ExtractionResult(
        entities=[
            Entity(name="Claude", type=EntityType.MODEL),
            Entity(name="Claude", type=EntityType.PERSON),
        ],
        triples=[],
    )
    out = resolve(result)
    assert len(out.entities) == 2   # same string, different type -> kept apart
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_disambiguation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.disambiguation'`

- [ ] **Step 3: Write `src/disambiguation.py`**

```python
"""Entity resolution: collapse surface-form variants of the same real entity.

Production entity resolution is: blocking -> candidate generation -> pairwise
scoring -> clustering. This is a compact, deterministic version of exactly that:
  - BLOCK by entity type (never merge a PERSON with an ORG).
  - SEED known acronym/alias equivalences.
  - SCORE remaining pairs with rapidfuzz token_set_ratio.
  - CLUSTER matches with union-find.
  - CANONICALIZE: highest mentions, then longest name; the rest become aliases.
Triples are rewritten to canonical names so no edge points at a merged-away node.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from src.schema import Entity, ExtractionResult, Triple

_THRESHOLD = 90.0  # token_set_ratio above this -> same entity

# Hand-seeded equivalences (case-insensitive substring match on either side).
_ALIAS_SEEDS = [
    ("university of california, berkeley", "uc berkeley"),
    ("gpt-3.5-turbo", "gpt-3.5"),
]


def _seeded_same(a: str, b: str) -> bool:
    la, lb = a.lower(), b.lower()
    for x, y in _ALIAS_SEEDS:
        if (x in la and y in lb) or (y in la and x in lb):
            return True
    return False


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)  # deterministic: keep lower index


def resolve(result: ExtractionResult) -> ExtractionResult:
    ents = list(result.entities)
    n = len(ents)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if ents[i].type != ents[j].type:        # blocking by type
                continue
            same = _seeded_same(ents[i].name, ents[j].name) or \
                fuzz.token_set_ratio(ents[i].name, ents[j].name) >= _THRESHOLD
            if same:
                uf.union(i, j)

    # group indices by cluster root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    canonical_of: dict[str, str] = {}     # old name -> canonical name
    merged: list[Entity] = []
    for members in clusters.values():
        group = [ents[i] for i in members]
        # canonical: most mentions, then longest name, then alphabetical (stable)
        canon = sorted(group, key=lambda e: (-e.mentions, -len(e.name), e.name))[0]
        aliases = sorted({a for e in group for a in e.aliases} |
                         {e.name for e in group if e.name != canon.name})
        merged.append(Entity(name=canon.name, type=canon.type,
                             aliases=aliases, mentions=sum(e.mentions for e in group)))
        for e in group:
            canonical_of[e.name] = canon.name

    new_triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for t in result.triples:
        s = canonical_of.get(t.source, t.source)
        o = canonical_of.get(t.target, t.target)
        if s == o:                       # self-loop created by merge -> drop
            continue
        key = (s, t.relation, o)
        if key in seen:
            continue
        seen.add(key)
        new_triples.append(Triple(source=s, relation=t.relation, target=o, evidence=t.evidence))

    return ExtractionResult(entities=merged, triples=new_triples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_disambiguation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/disambiguation.py tests/test_disambiguation.py
git commit -m "feat(6.1): type-aware entity disambiguation (rapidfuzz + union-find)"
```

---

### Task 8: `graph.py` — NetworkX build / save / load / top-N

**Files:**
- Create: `src/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from src.graph import build_graph, save_graph, load_graph, top_connected, to_undirected_weighted
from src.schema import Entity, EntityType, ExtractionResult, Triple


def _sample():
    return ExtractionResult(
        entities=[
            Entity(name="Stanford University", type=EntityType.ORG, mentions=3),
            Entity(name="Nelson Liu", type=EntityType.PERSON, mentions=2),
            Entity(name="Percy Liang", type=EntityType.PERSON, mentions=1),
            Entity(name="Claude", type=EntityType.MODEL, mentions=1),
        ],
        triples=[
            Triple(source="Nelson Liu", relation="works_at", target="Stanford University"),
            Triple(source="Percy Liang", relation="works_at", target="Stanford University"),
            Triple(source="Nelson Liu", relation="collaborates_with", target="Percy Liang"),
        ],
    )


def test_nodes_are_entities_edges_are_relations():
    G = build_graph(_sample())
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert G.nodes["Stanford University"]["type"] == "ORG"
    rels = {d["relation"] for _, _, d in G.edges(data=True)}
    assert "works_at" in rels


def test_top_connected_orders_by_degree():
    G = build_graph(_sample())
    top = top_connected(G, n=2)
    assert top[0][0] == "Stanford University"   # degree 2, highest
    assert len(top) == 2


def test_save_and_load_roundtrip(tmp_path: Path):
    G = build_graph(_sample())
    out = save_graph(G, str(tmp_path))
    assert Path(out["graphml"]).exists()
    assert Path(out["json"]).exists()
    G2 = load_graph(out["json"])
    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()
    assert G2.nodes["Claude"]["type"] == "MODEL"


def test_undirected_projection_weights_parallel_edges():
    G = build_graph(_sample())
    U = to_undirected_weighted(G)
    assert U.number_of_nodes() == 4
    assert U.has_edge("Nelson Liu", "Stanford University")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph'`

- [ ] **Step 3: Write `src/graph.py`**

```python
"""NetworkX knowledge graph: build from triples, persist, and query.

Storage choices:
  - MultiDiGraph preserves direction AND parallel edges (two entities can have
    several distinct relations).
  - GraphML for interoperability (Gephi/yEd) — list attrs (aliases) are joined to
    a string because GraphML only stores scalars.
  - JSON node-link for a lossless round-trip used by load_graph().
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.schema import ExtractionResult


def build_graph(result: ExtractionResult) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for e in result.entities:
        G.add_node(e.name, type=e.type.value, mentions=e.mentions, aliases=list(e.aliases))
    for t in result.triples:
        # only connect entities that exist as nodes (guards against stray names)
        if t.source in G and t.target in G:
            G.add_edge(t.source, t.target, relation=t.relation, evidence=t.evidence)
    return G


def to_undirected_weighted(G: nx.MultiDiGraph) -> nx.Graph:
    """Collapse to an undirected simple graph; weight = number of parallel relations.
    Used for community detection and layout."""
    U = nx.Graph()
    U.add_nodes_from(G.nodes(data=True))
    for u, v in G.edges():
        if U.has_edge(u, v):
            U[u][v]["weight"] += 1
        else:
            U.add_edge(u, v, weight=1)
    return U


def top_connected(G: nx.MultiDiGraph, n: int = 5) -> list[tuple[str, int]]:
    """Return [(node, degree)] for the n most-connected nodes (undirected degree).
    Ties broken by mention count then name for determinism."""
    U = to_undirected_weighted(G)
    ranked = sorted(
        U.nodes(),
        key=lambda x: (-U.degree(x), -G.nodes[x].get("mentions", 0), x),
    )
    return [(node, U.degree(node)) for node in ranked[:n]]


def save_graph(G: nx.MultiDiGraph, out_dir: str) -> dict[str, str]:
    """Write GraphML + JSON node-link. Returns the paths written."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    graphml_path = str(Path(out_dir) / "graph.graphml")
    json_path = str(Path(out_dir) / "graph.json")

    # GraphML can't hold lists: serialize aliases to a string on a copy.
    H = G.copy()
    for _, data in H.nodes(data=True):
        data["aliases"] = "; ".join(data.get("aliases", []))
    nx.write_graphml(H, graphml_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G, edges="links"), f, ensure_ascii=False, indent=2)

    return {"graphml": graphml_path, "json": json_path}


def load_graph(json_path: str) -> nx.MultiDiGraph:
    """Reload the lossless JSON node-link graph."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, multigraph=True, directed=True, edges="links")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph.py -v`
Expected: PASS (4 tests)

> Note on `edges=` kwarg: NetworkX 3.x emits a FutureWarning if `edges` is omitted; passing `edges="links"` to both `node_link_data`/`node_link_graph` keeps writer and reader consistent. If your installed NetworkX rejects the kwarg, drop it from both calls together.

- [ ] **Step 5: Commit**

```bash
git add src/graph.py tests/test_graph.py
git commit -m "feat(6.1): NetworkX graph build/save/load/top-N + undirected projection"
```

---

### Task 9: `embeddings.py` — entity vectors (API or deterministic offline)

**Files:**
- Create: `src/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from src import embeddings
from src.schema import Entity, EntityType


def test_offline_embeddings_are_deterministic_and_normalized(monkeypatch):
    monkeypatch.setattr(embeddings.config, "use_embeddings_api", lambda: False)
    ents = [Entity(name="Claude", type=EntityType.MODEL),
            Entity(name="Stanford University", type=EntityType.ORG)]
    v1 = embeddings.embed_entities(ents)
    v2 = embeddings.embed_entities(ents)
    assert v1.shape == (2, embeddings.embedding_dim())
    assert np.allclose(v1, v2)                       # deterministic
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)  # L2-normalized


def test_embedding_text_includes_name_type_aliases():
    e = Entity(name="UC Berkeley", type=EntityType.ORG, aliases=["University of California, Berkeley"])
    text = embeddings.entity_text(e)
    assert "UC Berkeley" in text and "ORG" in text and "University of California" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.embeddings'`

- [ ] **Step 3: Write `src/embeddings.py`**

```python
"""Entity embeddings for Qdrant.

Real path: OpenRouter `text-embedding-3-small` (1536-dim) when a key is set.
Offline path: a deterministic hashed bag-of-tokens vector (256-dim) so the whole
pipeline — including Qdrant upsert and search — runs with no key or network.
The offline vectors are NON-SEMANTIC placeholders (stable across runs); we print a
clear notice when they are used.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from src import config
from src.schema import Entity

_API_DIM = 1536
_OFFLINE_DIM = 256


def embedding_dim() -> int:
    return _API_DIM if config.use_embeddings_api() else _OFFLINE_DIM


def entity_text(e: Entity) -> str:
    """The string we embed for an entity: name + type + aliases."""
    parts = [e.name, f"type: {e.type.value}"]
    if e.aliases:
        parts.append("aka " + ", ".join(e.aliases))
    return " | ".join(parts)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (m / safe).astype(np.float32)


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic hashed bag-of-tokens embedding (offline fallback)."""
    rows = np.zeros((len(texts), _OFFLINE_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            rows[i, h % _OFFLINE_DIM] += 1.0
    return _normalize(rows)


def _api_embed(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=config.OPENROUTER_BASE_URL)
    result = client.embeddings.create(model=config.OPENROUTER_EMBEDDING_MODEL, input=texts)
    return _normalize(np.asarray([d.embedding for d in result.data], dtype=np.float32))


def embed_entities(entities: list[Entity]) -> np.ndarray:
    """Return an (n, embedding_dim()) L2-normalized float32 matrix for the entities."""
    if not entities:
        return np.zeros((0, embedding_dim()), dtype=np.float32)
    texts = [entity_text(e) for e in entities]
    if config.use_embeddings_api():
        return _api_embed(texts)
    print("NOTE: no OPENROUTER_API_KEY — using deterministic offline (non-semantic) embeddings.")
    return _hash_embed(texts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat(6.1): entity embeddings (OpenRouter API or deterministic offline)"
```

---

### Task 10: `vector_store.py` — Qdrant entity storage

**Files:**
- Create: `src/vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing test** (logic test for the pure helper; live Qdrant gated)

```python
import numpy as np
import pytest

from src import vector_store
from src.schema import Entity, EntityType


def test_point_id_is_stable_and_int():
    a = vector_store.point_id("Stanford University")
    b = vector_store.point_id("Stanford University")
    assert a == b and isinstance(a, int) and a >= 0


def _qdrant_up() -> bool:
    try:
        import requests
        return requests.get("http://localhost:6333/healthz", timeout=2).ok
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_up(), reason="local Qdrant not running")
def test_upsert_and_search_roundtrip():
    from src import config
    client = vector_store.get_client()
    ents = [Entity(name="Claude", type=EntityType.MODEL, mentions=2),
            Entity(name="Stanford University", type=EntityType.ORG, mentions=3)]
    vecs = np.eye(2, vector_store_dim := 4, dtype=np.float32)  # tiny dummy vectors
    vector_store.create_collection(client, dim=vector_store_dim)
    vector_store.upsert_entities(client, ents, vecs)
    hits = vector_store.search(client, np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
    assert hits and hits[0]["name"] == "Claude"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.vector_store'` (the live test skips if Qdrant is down)

- [ ] **Step 3: Write `src/vector_store.py`**

```python
"""Qdrant storage for entity embeddings (local Docker by default).

Mirrors the day-3.4 vector store but the unit of storage is an ENTITY, not a chunk.
Payload carries graph attributes (type, degree, mentions, aliases) so a future
hybrid retriever can filter ("MODEL entities only") without touching the graph.
"""
from __future__ import annotations

import hashlib
import sys
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from src import config
from src.schema import Entity


def get_client() -> QdrantClient:
    try:
        client = QdrantClient(url=config.QDRANT_URL, timeout=30)
        client.get_collections()
        return client
    except (ConnectionError, ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        print(
            f"ERROR: Could not connect to Qdrant at {config.QDRANT_URL} "
            f"({type(exc).__name__}: {exc}).\nStart it with:\n"
            "  docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest",
            file=sys.stderr,
        )
        sys.exit(1)


def point_id(name: str) -> int:
    """Stable uint63 id derived from the canonical entity name (idempotent upserts)."""
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (2**63)


def create_collection(client: QdrantClient, dim: int) -> None:
    if client.collection_exists(config.ENTITY_COLLECTION):
        client.delete_collection(config.ENTITY_COLLECTION)
    client.create_collection(
        collection_name=config.ENTITY_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Collection '{config.ENTITY_COLLECTION}' created ({dim}-dim COSINE).")


def upsert_entities(
    client: QdrantClient,
    entities: list[Entity],
    vectors: np.ndarray,
    degrees: dict[str, int] | None = None,
) -> None:
    if len(entities) != vectors.shape[0]:
        raise ValueError(f"entities ({len(entities)}) != vectors ({vectors.shape[0]})")
    degrees = degrees or {}
    points = [
        PointStruct(
            id=point_id(e.name),
            vector=vectors[i].tolist(),
            payload={
                "name": e.name,
                "type": e.type.value,
                "mentions": e.mentions,
                "aliases": e.aliases,
                "degree": degrees.get(e.name, 0),
            },
        )
        for i, e in enumerate(entities)
    ]
    client.upsert(collection_name=config.ENTITY_COLLECTION, points=points, wait=True)
    count = client.count(collection_name=config.ENTITY_COLLECTION, exact=True).count
    print(f"Upserted {len(points)} entities; collection now holds {int(count)}.")


def search(client: QdrantClient, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    resp = client.query_points(
        collection_name=config.ENTITY_COLLECTION,
        query=query_vector.astype(np.float32).tolist(),
        limit=top_k,
    )
    out = []
    for rank, p in enumerate(resp.points, start=1):
        payload = p.payload or {}
        out.append({"rank": rank, "score": float(p.score or 0.0),
                    "name": payload.get("name", ""), "type": payload.get("type", ""),
                    "degree": int(payload.get("degree", 0))})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vector_store.py -v`
Expected: PASS (`test_point_id_is_stable_and_int` passes; the roundtrip test passes if Qdrant is up, else SKIPPED).

To exercise the live path, first start Qdrant:
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

- [ ] **Step 5: Commit**

```bash
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat(6.1): Qdrant entity vector store (local Docker)"
```

---

### Task 11: `visualize.py` — clustered matplotlib render

**Files:**
- Create: `src/visualize.py`
- Test: `tests/test_visualize.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from src.graph import build_graph
from src.visualize import draw_graph
from src.schema import Entity, EntityType, ExtractionResult, Triple


def test_draw_graph_writes_png(tmp_path: Path):
    result = ExtractionResult(
        entities=[Entity(name="A", type=EntityType.PERSON, mentions=2),
                  Entity(name="B", type=EntityType.ORG),
                  Entity(name="C", type=EntityType.MODEL)],
        triples=[Triple(source="A", relation="r", target="B"),
                 Triple(source="A", relation="r", target="C")],
    )
    G = build_graph(result)
    out = tmp_path / "graph.png"
    path = draw_graph(G, str(out))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.visualize'`

- [ ] **Step 3: Write `src/visualize.py`**

```python
"""Render the knowledge graph to PNG with clear, color-coded communities.

Communities (NetworkX Louvain on the undirected projection) drive node color, so
densely-connected groups — authors, models, institutions — pop out as clusters.
Node size scales with degree; the most-connected entities are labeled. matplotlib
runs on the non-interactive Agg backend so it works headless / in tests.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt
import networkx as nx

from src.graph import to_undirected_weighted, top_connected


def draw_graph(G: nx.MultiDiGraph, out_path: str, *, label_top: int = 12, seed: int = 42) -> str:
    U = to_undirected_weighted(G)
    if U.number_of_nodes() == 0:
        raise ValueError("cannot visualize an empty graph")

    # community detection -> color index per node
    communities = nx.community.louvain_communities(U, weight="weight", seed=seed)
    color_of = {node: i for i, comm in enumerate(communities) for node in comm}
    node_colors = [color_of.get(n, 0) for n in U.nodes()]

    degrees = dict(U.degree())
    node_sizes = [300 + 400 * degrees[n] for n in U.nodes()]

    pos = nx.spring_layout(U, seed=seed, k=0.6, iterations=100, weight="weight")

    top_names = {name for name, _ in top_connected(G, n=label_top)}
    labels = {n: n for n in U.nodes() if n in top_names}

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(U, pos, alpha=0.25, width=1.0)
    nx.draw_networkx_nodes(U, pos, node_color=node_colors, node_size=node_sizes,
                           cmap=plt.cm.tab20, alpha=0.9)
    nx.draw_networkx_labels(U, pos, labels=labels, font_size=9, font_weight="bold")
    plt.title(f"Knowledge Graph — {U.number_of_nodes()} entities, "
              f"{len(communities)} communities", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_visualize.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/visualize.py tests/test_visualize.py
git commit -m "feat(6.1): matplotlib graph visualization colored by community"
```

---

### Task 12: `report.py` — top-5 most-connected table

**Files:**
- Create: `src/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
from src.graph import build_graph
from src.report import format_top_connected
from src.schema import Entity, EntityType, ExtractionResult, Triple


def test_format_top_connected_lists_highest_degree():
    result = ExtractionResult(
        entities=[Entity(name="Hub", type=EntityType.ORG),
                  Entity(name="A", type=EntityType.PERSON),
                  Entity(name="B", type=EntityType.PERSON)],
        triples=[Triple(source="A", relation="r", target="Hub"),
                 Triple(source="B", relation="r", target="Hub")],
    )
    G = build_graph(result)
    text = format_top_connected(G, n=2)
    assert "Hub" in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].lower().startswith("top") or "Hub" in lines[0] or "Hub" in lines[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.report'`

- [ ] **Step 3: Write `src/report.py`**

```python
"""Human-readable summaries of graph structure."""
from __future__ import annotations

import networkx as nx

from src.graph import top_connected


def format_top_connected(G: nx.MultiDiGraph, n: int = 5) -> str:
    rows = top_connected(G, n=n)
    width = max((len(name) for name, _ in rows), default=4)
    lines = [f"Top {len(rows)} most-connected entities:",
             f"  {'#':>2}  {'entity':<{width}}  {'type':<8}  degree  mentions"]
    for i, (name, degree) in enumerate(rows, start=1):
        etype = G.nodes[name].get("type", "?")
        mentions = G.nodes[name].get("mentions", 0)
        lines.append(f"  {i:>2}  {name:<{width}}  {etype:<8}  {degree:>6}  {mentions:>8}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report.py
git commit -m "feat(6.1): top-5 most-connected entities report"
```

---

### Task 13: `entity_extractor.py` — CLI orchestrator

**Files:**
- Create: `entity_extractor.py` (top level of the day folder)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import sys
from pathlib import Path


def test_cli_runs_end_to_end_offline(tmp_path: Path):
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Nelson Liu and Percy Liang work at Stanford University. "
        "Claude was evaluated on NaturalQuestions.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "entity_extractor.py"),
         "--doc", str(doc), "--out-dir", str(out_dir), "--no-qdrant"],
        capture_output=True, text=True, encoding="utf-8",
        env={"PYTHONIOENCODING": "utf-8", **_clean_env()},
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "graph.json").exists()
    assert (out_dir / "graph.graphml").exists()
    assert (out_dir / "graph.png").exists()
    assert "most-connected" in proc.stdout


def _clean_env():
    import os
    e = dict(os.environ)
    e["USE_LLM"] = "0"
    return e
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL (entity_extractor.py does not exist yet → non-zero return / file missing)

- [ ] **Step 3: Write `entity_extractor.py`**

```python
"""entity_extractor.py — document -> knowledge graph (Graph RAG, Day 6.1).

Pipeline: load -> extract (offline spaCy by default; OpenRouter if USE_LLM=1)
-> disambiguate -> build NetworkX graph -> save to disk -> embed entities ->
upsert to Qdrant -> visualize clusters -> print top-5 most-connected.

Usage:
  python entity_extractor.py --doc path/to/doc.pdf
  python entity_extractor.py --doc doc.txt --no-qdrant --no-viz --top 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout so entity names with non-ASCII render on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()

from src import config
from src.disambiguation import resolve
from src.embeddings import embed_entities, embedding_dim
from src.extractor import extract
from src.graph import build_graph, save_graph, to_undirected_weighted
from src.loader import load_pages
from src.report import format_top_connected
from src.visualize import draw_graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract a knowledge graph from a document.")
    ap.add_argument("--doc", required=True, help="path to a PDF or .txt/.md document")
    ap.add_argument("--out-dir", default="out", help="directory for graph + image outputs")
    ap.add_argument("--top", type=int, default=5, help="how many top-connected entities to print")
    ap.add_argument("--no-qdrant", action="store_true", help="skip Qdrant upsert")
    ap.add_argument("--no-viz", action="store_true", help="skip PNG visualization")
    args = ap.parse_args()

    print(f"Loading {args.doc} ...")
    pages = load_pages(args.doc)
    print(f"  {len(pages)} page(s).")

    mode = "OpenRouter LLM" if config.use_llm() else "offline spaCy"
    print(f"Extracting entities & relations ({mode}) ...")
    result = extract(pages)
    print(f"  raw: {len(result.entities)} entities, {len(result.triples)} triples")

    result = resolve(result)
    print(f"  after disambiguation: {len(result.entities)} entities, {len(result.triples)} triples")

    G = build_graph(result)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    paths = save_graph(G, args.out_dir)
    print(f"Graph saved: {paths['graphml']} | {paths['json']}")

    if not args.no_qdrant:
        from src import vector_store
        client = vector_store.get_client()
        vectors = embed_entities(result.entities)
        U = to_undirected_weighted(G)
        degrees = {n: U.degree(n) for n in U.nodes()}
        vector_store.create_collection(client, dim=embedding_dim())
        vector_store.upsert_entities(client, result.entities, vectors, degrees=degrees)

    if not args.no_viz:
        img = draw_graph(G, str(Path(args.out_dir) / "graph.png"))
        print(f"Visualization saved: {img}")

    print()
    print(format_top_connected(G, n=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_cli.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add entity_extractor.py tests/test_cli.py
git commit -m "feat(6.1): entity_extractor CLI orchestrating the full pipeline"
```

---

### Task 14: Full suite + real run on the paper

**Files:** none new (produces `out/` artifacts)

- [ ] **Step 1: Run the full test suite**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -v`
Expected: all tests PASS (Qdrant live test SKIPPED unless server running).

- [ ] **Step 2: Start Qdrant and run the real document end-to-end**

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
PYTHONIOENCODING=utf-8 python entity_extractor.py --doc "C:/Users/minhh/Downloads/Lost in the Middle.pdf" --out-dir out
```
Expected: prints page count, entity/triple counts, "Graph saved", "Collection created", "Upserted N entities", "Visualization saved", and the top-5 table. `out/graph.png` shows visually separated clusters (authors / institutions / models / datasets / concepts).

- [ ] **Step 3: Eyeball `out/graph.png`** — confirm clusters are visually distinct and the most-connected hubs (e.g. an institution or the dataset) are labeled. If clusters look tangled, increase `k` in `spring_layout` or lower it; if too many tiny nodes, that's expected for a real paper.

- [ ] **Step 4: Commit the sample outputs**

```bash
git add -f out/graph.png out/graph.json out/graph.graphml
git commit -m "chore(6.1): sample outputs from Lost in the Middle paper"
```
(`-f` because `.gitignore` excludes `out/` and `*.png`; we intentionally keep these as evidence.)

---

### Task 15: `README.md` — concepts + prompt design

**Files:**
- Create: `06-graph-rag/6.1-entity-extraction/README.md`
- Modify: `06-graph-rag/README.md` (flip `entity_extractor.py` status to ✅ and link the day)

- [ ] **Step 1: Write `6.1-entity-extraction/README.md`** covering, with real content (no placeholders):
  - **Overview** + how to run (offline default, `USE_LLM=1`, Qdrant docker line).
  - **The five concepts**, answered concretely:
    1. *Why graph beats vanilla RAG on multi-hop Qs* — vanilla retrieves chunks by surface similarity; a 2-hop question ("which dataset did the model advised-by Percy Liang's group get evaluated on?") needs to traverse PERSON→ORG→MODEL→DATASET edges that no single chunk contains. The graph encodes those edges explicitly, so traversal answers it; vector search would need every hop to be lexically close to the query, which it isn't.
    2. *Community summary* — a short LLM-written summary of a densely-connected sub-cluster (a Louvain community). It lets "global" questions ("what are the main themes?") be answered from a handful of cluster summaries instead of stuffing the whole corpus into context.
    3. *Construction cost vs query quality* — building the graph costs an LLM pass per chunk (extraction) + clustering + summarization at index time; you pay once, up front, for cheaper, higher-quality multi-hop/global answers at query time. Vanilla RAG has ~zero index cost but caps out on relational/global questions.
    4. *When to use Graph RAG vs vanilla* — use Graph RAG when questions are multi-hop, relational, or global ("summarize across documents"), and the corpus is stable enough to amortize indexing. Stick with vanilla for single-fact lookups, rapidly-changing corpora, or tight indexing budgets.
    5. *Entity disambiguation in practice* — surface-form variants ("UC Berkeley" vs the full name) must collapse to one node or the graph fragments. The pipeline does blocking-by-type → seeded aliases + rapidfuzz scoring → union-find clustering → canonical-name selection; production systems add embedding similarity and human review for hard cases.
  - **Extraction prompt design** — walk through the five choices in `extractor_llm._SYSTEM_PROMPT` (fixed typed label set, strict JSON shape, "only edges between listed entities", one-shot example, canonical-name+aliases) and why each prevents a specific failure mode (untyped sprawl, parse errors, dangling edges, wrong granularity, duplicate nodes).
  - **Goal → evidence** table mapping each task/goal to the file + test that proves it.
  - **Architecture diagram** (hand-drawable ASCII of the pipeline).

- [ ] **Step 2: Update `06-graph-rag/README.md`** — change the `entity_extractor.py` row Status from `🔲 Pending` to `✅ Done — see 6.1-entity-extraction/` and add a one-line pointer.

- [ ] **Step 3: Commit**

```bash
git add 06-graph-rag/6.1-entity-extraction/README.md 06-graph-rag/README.md
git commit -m "docs(6.1-entity-extraction): README — 5 concepts, prompt design, goal→evidence"
```

---

## Self-Review

**Spec coverage:**
- Run LLM over 10-page doc → Task 5 (`extractor_llm`) + Task 14 real run. ✅
- Extract (entity, relation, entity) triples → Tasks 4/5 (`Triple`). ✅
- NetworkX graph nodes=entities edges=rels → Task 8. ✅
- Store entity embeddings in Qdrant → Tasks 9 + 10. ✅
- Visualize with matplotlib/networkx → Task 11. ✅
- Top-5 most-connected → Tasks 8 (`top_connected`) + 12 (`report`). ✅
- Graph saved to disk → Task 8 (`save_graph`). ✅
- Visualization shows clusters → Task 11 (Louvain coloring). ✅
- Explain extraction prompt design → Task 5 docstring + Task 15 README. ✅
- 5 learning concepts → Task 15 README. ✅
- Entity disambiguation → Task 7. ✅
- Offline-default + opt-in OpenRouter → Tasks 2/4/5/6/9. ✅
- Local Docker Qdrant → Task 10. ✅

**Placeholder scan:** README content (Task 15) is specified as bullet substance, not "TBD". All code steps contain full code. ✅

**Type consistency:** `ExtractionResult{entities, triples}`, `Entity{name,type,aliases,mentions}`, `Triple{source,relation,target,evidence}` used identically across Tasks 1,4,5,6,7,8,9,10. `build_graph/save_graph/load_graph/top_connected/to_undirected_weighted` names consistent across Tasks 8,11,12,13. `embed_entities/embedding_dim/entity_text` consistent across 9,13. `get_client/create_collection/upsert_entities/search/point_id` consistent across 10,13. ✅
