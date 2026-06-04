"""Deterministic tf-idf vector space over the corpus vocabulary.

idf = log(N/df): corpus-ubiquitous terms (df == N, e.g. 'helios') get idf 0 and
drop out of both ranking and grounding — exactly what we want, since a term in
every document is non-discriminative. With USE_LLM=1 a real embedding model could
be substituted, but the offline tf-idf space is what makes the reflexion loop
deterministic and assertable.
"""
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
