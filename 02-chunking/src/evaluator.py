"""Chunk evaluation metrics including ICC (Intrachunk Cohesion)."""

from __future__ import annotations

import itertools
import random
from typing import Any

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

_SPACY_MODEL: spacy.Language | None = None
_EMBEDDER: SentenceTransformer | None = None

_SPACY_INSTALL_MSG = (
    "spaCy model 'en_core_web_sm' is not installed. "
    "Run: python -m spacy download en_core_web_sm"
)


def _load_nlp() -> spacy.Language:
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        try:
            _SPACY_MODEL = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise OSError(_SPACY_INSTALL_MSG) from exc
    return _SPACY_MODEL


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _chunk_sentences(text: str) -> list[str]:
    nlp = _load_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_chunk_icc(chunk: str, embedder: SentenceTransformer) -> float:
    sentences = _chunk_sentences(chunk)
    if len(sentences) < 2:
        return 1.0

    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    pairwise_scores: list[float] = []
    for i, j in itertools.combinations(range(len(sentences)), 2):
        pairwise_scores.append(_cosine_similarity(embeddings[i], embeddings[j]))

    if not pairwise_scores:
        return 1.0
    return float(np.mean(pairwise_scores))


def evaluate_chunks(chunks: list[str], strategy_name: str) -> dict[str, Any]:
    empty_result: dict[str, Any] = {
        "Strategy": strategy_name,
        "Total Chunks": 0,
        "Avg Tokens": 0,
        "Min Tokens": 0,
        "Max Tokens": 0,
        "Std Tokens": 0.0,
        "ICC Score": 0.0,
    }

    if len(chunks) == 0:
        return empty_result

    token_counts = [_count_tokens(chunk) for chunk in chunks]
    token_array = np.array(token_counts, dtype=float)

    random.seed(42)
    sample_size = min(30, len(chunks))
    sampled_chunks = random.sample(chunks, sample_size)

    embedder = _get_embedder()
    chunk_iccs: list[float] = []
    for chunk in sampled_chunks:
        chunk_iccs.append(_compute_chunk_icc(chunk, embedder))

    overall_icc = round(float(np.mean(chunk_iccs)), 4) if chunk_iccs else 0.0

    return {
        "Strategy": strategy_name,
        "Total Chunks": len(chunks),
        "Avg Tokens": float(np.mean(token_array)),
        "Min Tokens": int(np.min(token_array)),
        "Max Tokens": int(np.max(token_array)),
        "Std Tokens": float(np.std(token_array)),
        "ICC Score": overall_icc,
    }
