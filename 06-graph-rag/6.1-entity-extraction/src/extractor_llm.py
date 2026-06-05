"""LLM entity + relation extraction via OpenRouter (OpenAI-compatible API).

Used only when USE_LLM=1 and a key is set. The offline extractor is the default.
This module is split into pure helpers (build_messages, parse_response) — unit
testable without any network — and a thin extract_llm() that does the I/O.

The extraction prompt design (see _SYSTEM_PROMPT) is the heart of LLM Graph RAG.
Five deliberate choices, each preventing a specific failure mode:
  1. Fixed typed label set     -> prevents untyped entity sprawl; keeps the graph queryable.
  2. Strict JSON-only shape    -> deterministic parsing; no prose to strip.
  3. "Only edges between listed entities" -> no dangling edges pointing at non-nodes.
  4. One-shot example          -> anchors output format AND relation granularity.
  5. Canonical name + aliases  -> pushes first-pass disambiguation into the model.
"""
from __future__ import annotations

import json
import os
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
        entities.append(Entity(name=e["name"], type=etype, aliases=list(e.get("aliases", []))))

    triples: list[Triple] = []
    for t in data.get("triples", []):
        triples.append(
            Triple(
                source=t["source"],
                relation=t.get("relation", "related_to"),
                target=t["target"],
                evidence=t.get("evidence", ""),
            )
        )

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
