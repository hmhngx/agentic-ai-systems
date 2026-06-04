"""Tokenization + stopwords shared by retrieval, generation, scoring, routing.

One tokenizer, used everywhere, so the notion of a "token" is identical in the
embedder, the grounding-confidence check, and the faithfulness scorer. If these
ever disagreed, faithfulness numbers would silently stop matching retrieval.
"""
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
