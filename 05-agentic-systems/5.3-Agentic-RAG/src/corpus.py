"""A tiny, fully-controlled corpus about a fictional product ("Helios").

Fictional on purpose: ground truth is entirely in these texts, so no parametric
knowledge can leak into "faithful" answers. The vocabulary is engineered so the
reflexion loop fires deterministically (see the plan intro / README):

  - 'helios' is in EVERY doc  -> idf 0  -> non-discriminative (ignored).
  - each topic's formal term is OWNED by one doc (retention=D1, pricing=D2,
    quota=D3, embedding=D4), so a query carrying that term lands decisively.
  - 'documents' is shared by D1 and D5, so a casually-worded retention question
    grounds ambiguously on the first try -> the generator guesses -> faithfulness
    drops -> Reflexion expands keep->retained/retention, delete->purged -> the
    refined query now lands on D1.
"""
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
