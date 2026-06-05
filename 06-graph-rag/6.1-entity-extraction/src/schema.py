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
