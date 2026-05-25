"""Four chunking strategies for the benchmark suite."""

from __future__ import annotations

import numpy as np
import spacy
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

_SPACY_MODEL: spacy.Language | None = None
_EMBEDDER: SentenceTransformer | None = None

_SPACY_INSTALL_MSG = (
    "spaCy model 'en_core_web_sm' is not installed. "
    "Run: python -m spacy download en_core_web_sm"
)


def _filter_chunks(chunks: list[str]) -> list[str]:
    return [c.strip() for c in chunks if c.strip()]


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


def _sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    nlp = _load_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _coalesce_single_sentence_groups(groups: list[list[str]]) -> list[list[str]]:
    """Merge lone-sentence groups into the previous group to avoid micro-chunks."""
    if len(groups) <= 1:
        return groups

    merged: list[list[str]] = [list(groups[0])]
    for group in groups[1:]:
        if len(group) == 1 and len(merged[-1]) == 1:
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    return merged


def _greedy_merge_by_char_limit(
    parts: list[str],
    max_chars: int = 1000,
    sep: str = " ",
) -> list[str]:
    if not parts:
        return []

    chunks: list[str] = []
    current = parts[0]

    for part in parts[1:]:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def naive_chunk(text: str) -> list[str]:
    if not text.strip():
        return []
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=1000,
        chunk_overlap=0,
    )
    return _filter_chunks(splitter.split_text(text))


def recursive_chunk(text: str) -> list[str]:
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    return _filter_chunks(splitter.split_text(text))


def sentence_chunk(text: str) -> list[str]:
    if not text.strip():
        return []
    sentences = _sentences(text)
    if not sentences:
        return []
    return _filter_chunks(_greedy_merge_by_char_limit(sentences, max_chars=1000))


def semantic_chunk(text: str) -> list[str]:
    if not text.strip():
        return []

    sentences = _sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return _filter_chunks([sentences[0]])

    embedder = _get_embedder()
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    boundaries: list[int] = [0]
    for i in range(len(sentences) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < 0.4:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))

    semantic_groups: list[list[str]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        semantic_groups.append(sentences[start:end])
    semantic_groups = _coalesce_single_sentence_groups(semantic_groups)

    all_chunks: list[str] = []
    for group in semantic_groups:
        group_chunks = _greedy_merge_by_char_limit(group, max_chars=1000)
        all_chunks.extend(group_chunks)

    return _filter_chunks(all_chunks)
