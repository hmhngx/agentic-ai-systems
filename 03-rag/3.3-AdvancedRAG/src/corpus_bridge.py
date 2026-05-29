"""corpus_bridge.py - Reuses Day 3 corpus and query set without duplication.

Design decision: do NOT copy corpus.py from Day 3 into this module.
Instead, dynamically locate and import it from the sibling directory.
This enforces DRY - one source of truth for the 500-chunk corpus.

If Day 3 corpus is not importable (path issues), fall back to regenerating
it inline using the same TOPIC_SENTENCES and generation logic.
In either case, the corpus is ALWAYS deterministic with seed=42.

This matters because the BM25 index is built from the same 500 chunks
that are stored in Qdrant. The BM25 corpus and the Qdrant corpus must
be identical - any mismatch causes chunk_id lookups to fail during RRF fusion.

Why we also normalize chunk_id to str(chunk["id"]):
    Day 3 stores the chunk identifier as an int (the Qdrant point ID).
    BM25 results, RRF accumulator keys, and Cohere candidate lookups
    all need a single hashable identifier shared across systems. Strings
    survive JSON round-trips through Qdrant payload cleanly while ints
    sometimes get coerced into floats by intermediary serializers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


_DAY3_VECTORDBS_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "3.1-VectorDBs"
)

# Inline fallback constants - kept BIT-IDENTICAL to
# 03-rag/3.1-VectorDBs/src/corpus.py. The dict ordering matters because
# Day 3 iterates topics in insertion order to assign point IDs 0..499;
# any reordering would break recall@5 reproducibility across runs.
_FALLBACK_TOPIC_SENTENCES: dict[str, list[str]] = {
    "machine_learning": [
        "Neural networks learn hierarchical representations through backpropagation.",
        "Gradient descent iteratively adjusts weights to minimize the loss function.",
        "Overfitting occurs when a model memorizes training data rather than generalizing.",
        "Transformers use self-attention to model long-range token dependencies.",
        "Regularization techniques like dropout prevent co-adaptation of neurons.",
        "Cross-entropy loss measures the divergence between predicted and true distributions.",
        "Batch normalization stabilizes training by normalizing layer inputs.",
        "Convolutional layers exploit spatial locality and translation invariance.",
        "Reinforcement learning trains agents through reward signals from an environment.",
        "Embedding spaces encode semantic relationships as geometric distances.",
    ],
    "ocean_biology": [
        "Coral reefs support approximately twenty-five percent of all marine species.",
        "Bioluminescent organisms produce light through chemical reactions involving luciferin.",
        "Deep-sea hydrothermal vents sustain ecosystems independent of solar energy.",
        "Ocean currents redistribute heat and regulate global climate patterns.",
        "Whale migration routes span thousands of miles between feeding and breeding grounds.",
        "Phytoplankton produce roughly half of Earth's atmospheric oxygen through photosynthesis.",
        "Cephalopods like octopuses demonstrate remarkable problem-solving and camouflage abilities.",
        "The abyssal zone receives no sunlight and supports unique pressure-adapted organisms.",
        "Mangrove forests serve as nurseries for juvenile fish and buffer coastlines from erosion.",
        "Jellyfish blooms are increasing globally as ocean temperatures rise.",
    ],
    "ancient_history": [
        "The Roman Republic transitioned to Empire following Julius Caesar's assassination in 44 BCE.",
        "Egyptian hieroglyphs combined logographic and alphabetic elements in a complex writing system.",
        "The Silk Road facilitated trade and cultural exchange between East Asia and the Mediterranean.",
        "Greek city-states developed diverse governance systems ranging from democracy to oligarchy.",
        "The construction of the Great Wall spanned multiple Chinese dynasties over centuries.",
        "Mesopotamian civilizations developed early legal codes including the Code of Hammurabi.",
        "The Library of Alexandria was the ancient world's largest repository of knowledge.",
        "Phoenician traders spread their alphabet throughout the Mediterranean basin.",
        "The Persian Empire under Cyrus the Great practiced religious tolerance in conquered territories.",
        "Mayan astronomers developed a calendar system of remarkable precision without telescopes.",
    ],
    "cooking": [
        "Maillard reaction between amino acids and reducing sugars creates browning and complex flavors.",
        "Emulsification combines immiscible liquids like oil and water through mechanical agitation.",
        "Fermentation by microorganisms transforms sugars into alcohol, acids, and carbon dioxide.",
        "Braising uses moist heat to break down collagen in tough cuts into gelatin.",
        "Salt draws moisture from vegetables through osmosis and seasons from within.",
        "The smoke point of an oil determines the maximum safe temperature for frying.",
        "Gluten development through kneading gives bread dough elasticity and structure.",
        "Blanching vegetables in boiling water then shocking in ice water preserves color and texture.",
        "Resting meat after cooking allows juices to redistribute throughout the fibers.",
        "Acid ingredients like lemon juice denature proteins and balance rich flavors.",
    ],
    "urban_architecture": [
        "Modernist architecture rejected ornament in favor of functional form and industrial materials.",
        "Green building standards measure energy efficiency, water use, and indoor air quality.",
        "Transit-oriented development concentrates density and mixed uses around public transport nodes.",
        "Brutalist structures express raw concrete as both structural and aesthetic material.",
        "Adaptive reuse converts obsolete industrial buildings into residential or commercial spaces.",
        "Parametric design uses computational algorithms to generate complex building geometries.",
        "Urban heat islands form when impervious surfaces replace vegetation in dense areas.",
        "Zoning laws separate land uses to manage density, noise, and traffic flows.",
        "Passive solar design orients buildings and openings to maximize natural heating and lighting.",
        "Mixed-use developments reduce car dependency by placing housing near jobs and services.",
    ],
}

_FALLBACK_TOPIC_SOURCES: dict[str, str] = {
    "machine_learning": "ml_textbook_ch3.pdf",
    "ocean_biology": "marine_biology_primer.pdf",
    "ancient_history": "world_history_vol2.pdf",
    "cooking": "culinary_science.pdf",
    "urban_architecture": "city_planning_handbook.pdf",
}


def _try_import_day3() -> tuple[Any, Any] | None:
    """Attempt to import generate_corpus and get_query_set from Day 3.

    Returns None if Day 3 is not present on disk, in which case the
    caller activates the inline fallback. We deliberately catch only
    ImportError + FileNotFoundError - any other failure surfaces as a
    crash, because it means Day 3's corpus.py is broken (not just
    missing).
    """
    if not _DAY3_VECTORDBS_DIR.is_dir():
        return None
    day3_str: str = str(_DAY3_VECTORDBS_DIR)
    if day3_str not in sys.path:
        sys.path.insert(0, day3_str)
    try:
        from src.corpus import generate_corpus, get_query_set  # noqa: WPS433
    except (ImportError, FileNotFoundError):
        return None
    return generate_corpus, get_query_set


def _fallback_build_chunk_text(rng: np.random.Generator, topic: str) -> str:
    """Reproduce Day 3's _build_chunk_text logic byte-for-byte."""
    pool: list[str] = _FALLBACK_TOPIC_SENTENCES[topic]
    indices: np.ndarray = rng.integers(low=0, high=len(pool), size=4)
    return " ".join(pool[int(i)] for i in indices)


def _fallback_generate_corpus(n_chunks: int = 500, seed: int = 42) -> list[dict]:
    """Fallback: identical corpus generation to Day 3.

    Seed=42 guarantees same 500 chunks. The chunk dict schema matches
    Day 3 exactly: ``id`` (int), ``text``, ``topic``, ``source``, ``page``,
    ``chunk_index``.
    """
    if n_chunks <= 0 or n_chunks % len(_FALLBACK_TOPIC_SENTENCES) != 0:
        raise ValueError(
            f"n_chunks must be a positive multiple of "
            f"{len(_FALLBACK_TOPIC_SENTENCES)} (got {n_chunks})"
        )

    chunks_per_topic: int = n_chunks // len(_FALLBACK_TOPIC_SENTENCES)
    rng: np.random.Generator = np.random.default_rng(seed)
    topics: list[str] = list(_FALLBACK_TOPIC_SENTENCES.keys())

    corpus: list[dict] = []
    chunk_id: int = 0
    for topic in topics:
        source: str = _FALLBACK_TOPIC_SOURCES[topic]
        for local_index in range(chunks_per_topic):
            corpus.append(
                {
                    "id": chunk_id,
                    "text": _fallback_build_chunk_text(rng, topic),
                    "topic": topic,
                    "source": source,
                    "page": int(rng.integers(low=1, high=51)),
                    "chunk_index": local_index,
                }
            )
            chunk_id += 1
    return corpus


def _fallback_first_sentence(text: str) -> str:
    head: str = text.split(". ", 1)[0]
    if not head.endswith("."):
        head = head + "."
    return head


def _fallback_get_query_set(
    corpus: list[dict],
    n_queries: int = 20,
    seed: int = 99,
) -> list[dict]:
    """Fallback: identical stratified query set to Day 3.

    Seed=99 (different from corpus seed=42) ensures queries are not
    biased toward the first chunks generated.
    """
    n_topics: int = len(_FALLBACK_TOPIC_SENTENCES)
    if n_queries <= 0 or n_queries % n_topics != 0:
        raise ValueError(
            f"n_queries must be a positive multiple of {n_topics} (got {n_queries})"
        )

    per_topic: int = n_queries // n_topics
    rng: np.random.Generator = np.random.default_rng(seed)

    by_topic: dict[str, list[dict]] = {topic: [] for topic in _FALLBACK_TOPIC_SENTENCES}
    for chunk in corpus:
        by_topic[chunk["topic"]].append(chunk)

    queries: list[dict] = []
    for topic in _FALLBACK_TOPIC_SENTENCES:
        candidates: list[dict] = by_topic[topic]
        if len(candidates) < per_topic:
            raise ValueError(
                f"Topic '{topic}' has only {len(candidates)} chunks, "
                f"need {per_topic} for stratified queries"
            )
        chosen: np.ndarray = rng.choice(
            len(candidates), size=per_topic, replace=False
        )
        for i in chosen:
            chunk = candidates[int(i)]
            queries.append(
                {
                    "query_text": _fallback_first_sentence(chunk["text"]),
                    "source_chunk_id": chunk["id"],
                    "topic": topic,
                }
            )
    return queries


def _annotate_chunk_ids(corpus: list[dict]) -> None:
    """Add a string ``chunk_id`` field to every chunk dict in place.

    This is the canonical cross-system identifier used by BM25 results,
    RRF accumulator keys, Qdrant payload (after re-ingest), and the
    Cohere candidate-text lookup table. Stringifying once here avoids
    repeated str() casts in every downstream module.
    """
    for chunk in corpus:
        chunk["chunk_id"] = str(chunk["id"])


def get_corpus_and_queries() -> tuple[list[dict], list[dict]]:
    """Return ``(corpus, queries)`` deterministically.

    Attempts to import Day 3's ``generate_corpus`` / ``get_query_set``;
    falls back to an identical inline implementation if Day 3 is not on
    disk. Both paths produce the same 500 chunks and 20 queries because
    the seeds (42, 99) and topic-sentence pools are pinned.
    """
    imported = _try_import_day3()
    if imported is not None:
        generate_corpus, get_query_set = imported
        corpus: list[dict] = generate_corpus(n_chunks=500, seed=42)
        queries: list[dict] = get_query_set(corpus, n_queries=20, seed=99)
    else:
        corpus = _fallback_generate_corpus(n_chunks=500, seed=42)
        queries = _fallback_get_query_set(corpus, n_queries=20, seed=99)

    assert len(corpus) == 500, f"corpus must be 500 chunks (got {len(corpus)})"
    assert len(queries) == 20, f"queries must be 20 (got {len(queries)})"

    _annotate_chunk_ids(corpus)

    print("Corpus loaded: 500 chunks, 20 test queries (seed=42, seed=99)")
    return corpus, queries
