"""
text.py - Deterministic, offline text normalisation primitives.

These are the only place strings get turned into the tokens and sentences the
overlap checks operate on, so the normalisation policy is centralised and
testable. No network, no model - pure functions over ``str``.

Design notes:
  - We tokenise to *content tokens* (lowercased word characters, stopwords
    removed). Stopwords are dropped so that generic filler ("the", "is", "of")
    cannot inflate overlap between an answer and an off-topic chunk. Numbers are
    kept - dates and quantities ("1969", "100") are exactly the facts we want to
    verify are grounded.
  - Inline ``[Doc N]`` citation markers are stripped before tokenising a claim,
    so the citation label itself never counts as "overlap" with a chunk.
"""

from __future__ import annotations

import re


# Inline citation marker, tolerant of spacing variations the LLM may emit
# ("[Doc 2]", "[ Doc 2 ]"). Mirrors the pattern used by the naive-RAG citation
# validator so the two layers agree on what a citation looks like.
DOC_CITATION_RE = re.compile(r"\[\s*Doc\s+(\d+)\s*\]", re.IGNORECASE)

# Sentence boundary: terminal punctuation runs, or newlines. Intentionally
# simple - claims do not need linguistic perfection, only consistent splitting.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Word token: unicode word characters. ``re.UNICODE`` is the default in Py3.
_WORD_RE = re.compile(r"\w+")

# A compact, high-frequency English stopword set. Kept small on purpose: an
# overlarge list risks stripping content words. This is good enough to stop
# function words from dominating Jaccard/containment on short claims.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from
    had has have having he her hers him his how i if in into is it its itself me
    my of on or our ours out over own she should so some such than that the their
    theirs them then there these they this those to too under until up very was we
    were what when where which while who whom why will with would you your yours
    """.split()
)


def strip_citations(text: str) -> str:
    """Remove inline ``[Doc N]`` markers so they do not count as overlap."""
    return DOC_CITATION_RE.sub(" ", text)


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences (claims)."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def content_tokens(text: str) -> list[str]:
    """Lowercased word tokens with stopwords removed, order preserved.

    Order is preserved (a list, not a set) because ``difflib`` needs the token
    sequence to measure contiguous runs. Callers that want set semantics build
    a ``set(...)`` from the result.
    """
    return [
        tok
        for tok in (m.group(0).lower() for m in _WORD_RE.finditer(text))
        if tok not in STOPWORDS
    ]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace - used for refusal-phrase matching."""
    return re.sub(r"\s+", " ", text.strip().lower())
