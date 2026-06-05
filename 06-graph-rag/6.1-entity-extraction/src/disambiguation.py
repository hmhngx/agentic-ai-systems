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
        aliases = sorted(
            {a for e in group for a in e.aliases}
            | {e.name for e in group if e.name != canon.name}
        )
        merged.append(
            Entity(
                name=canon.name,
                type=canon.type,
                aliases=aliases,
                mentions=sum(e.mentions for e in group),
            )
        )
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
