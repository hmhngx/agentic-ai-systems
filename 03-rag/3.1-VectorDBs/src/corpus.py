"""Synthetic corpus generation for the Qdrant recall@5 benchmark.

This module has zero dependencies beyond stdlib and NumPy. The corpus is
fully deterministic given a seed: the same seed always produces the same
500 chunks in the same order, which is what allows recall@5 to be
reproducible across runs and across the HNSW / FLAT collections.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Topic sentence pools.
#
# Five topics, ten seed sentences each. Each generated chunk samples four
# sentences (with replacement) from one topic pool, so chunks within a topic
# share vocabulary and cluster tightly in embedding space while chunks across
# topics are well-separated. That separation is what makes recall@5 a
# meaningful signal even on only 500 vectors.
# ---------------------------------------------------------------------------
TOPIC_SENTENCES: dict[str, list[str]] = {
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

# Stable per-topic source filenames so the payload looks like a real RAG
# corpus (real systems track which document a chunk came from). The ordering
# of this list is fixed; do not change it -- chunk_index is computed against
# this order.
_TOPIC_SOURCES: dict[str, str] = {
    "machine_learning": "ml_textbook_ch3.pdf",
    "ocean_biology": "marine_biology_primer.pdf",
    "ancient_history": "world_history_vol2.pdf",
    "cooking": "culinary_science.pdf",
    "urban_architecture": "city_planning_handbook.pdf",
}

# Number of chunks generated per topic. Five topics * 100 = 500 total.
_CHUNKS_PER_TOPIC: int = 100


def _build_chunk_text(rng: np.random.Generator, topic: str) -> str:
    """Sample four sentences (with replacement) from a topic pool and join.

    Sampling with replacement is intentional: it keeps the distribution
    over sentences uniform while letting individual chunks repeat or omit
    sentences, which mirrors how real-world chunked documents have uneven
    sentence reuse across pages.
    """
    pool: list[str] = TOPIC_SENTENCES[topic]
    indices: np.ndarray = rng.integers(low=0, high=len(pool), size=4)
    return " ".join(pool[int(i)] for i in indices)


def generate_corpus(n_chunks: int = 500, seed: int = 42) -> list[dict]:
    """Generate a deterministic synthetic corpus of ``n_chunks`` chunks.

    Each chunk dict contains:
      - ``id``           : int  (0-indexed, used as the Qdrant point ID)
      - ``text``         : str  (four-sentence body, ~60-120 words)
      - ``topic``        : str  (one of the five topic keys)
      - ``source``       : str  (simulated filename, e.g. ``ml_textbook_ch3.pdf``)
      - ``page``         : int  (simulated page number in 1..50)
      - ``chunk_index``  : int  (position within the source document, 0-indexed)

    Distribution: chunks are split evenly across the five topics so that
    payload filtering can demonstrate ~80% search-space reduction
    (100 / 500 = 20%) per topic.

    A fixed ``seed`` guarantees identical corpora between the HNSW and FLAT
    collections, which is what makes recall@5 ground truth meaningful.

    Raises ``ValueError`` if ``n_chunks`` is not a positive multiple of 5.
    """
    if n_chunks <= 0 or n_chunks % len(TOPIC_SENTENCES) != 0:
        raise ValueError(
            f"n_chunks must be a positive multiple of {len(TOPIC_SENTENCES)} "
            f"(got {n_chunks})"
        )

    chunks_per_topic: int = n_chunks // len(TOPIC_SENTENCES)
    rng: np.random.Generator = np.random.default_rng(seed)
    topics: list[str] = list(TOPIC_SENTENCES.keys())

    corpus: list[dict] = []
    chunk_id: int = 0
    for topic in topics:
        source: str = _TOPIC_SOURCES[topic]
        for local_index in range(chunks_per_topic):
            chunks: dict = {
                "id": chunk_id,
                "text": _build_chunk_text(rng, topic),
                "topic": topic,
                "source": source,
                "page": int(rng.integers(low=1, high=51)),
                "chunk_index": local_index,
            }
            corpus.append(chunks)
            chunk_id += 1

    return corpus


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text``, restoring a trailing period."""
    head: str = text.split(". ", 1)[0]
    if not head.endswith("."):
        head = head + "."
    return head


def _pick_topic_queries(
    candidates: list[dict],
    per_topic: int,
    rng: np.random.Generator,
    topic: str,
) -> list[dict]:
    """Sample ``per_topic`` chunks from one topic and shape them as queries."""
    if len(candidates) < per_topic:
        raise ValueError(
            f"Topic '{topic}' has only {len(candidates)} chunks, "
            f"need {per_topic} for stratified queries"
        )
    chosen: np.ndarray = rng.choice(len(candidates), size=per_topic, replace=False)
    out: list[dict] = []
    for i in chosen:
        chunk = candidates[int(i)]
        out.append(
            {
                "query_text": _first_sentence(chunk["text"]),
                "source_chunk_id": chunk["id"],
                "topic": topic,
            }
        )
    return out


def get_query_set(
    corpus: list[dict],
    n_queries: int = 20,
    seed: int = 99,
) -> list[dict]:
    """Build a stratified query set of ``n_queries`` queries.

    Stratification: ``n_queries / 5`` queries are drawn from each topic so
    that the filtered-search benchmark always has the same number of
    in-topic queries per topic. With ``n_queries=20`` that yields 4 per
    topic.

    Each entry contains:
      - ``query_text``      : the first sentence of the source chunk
      - ``source_chunk_id`` : the integer ID of the chunk the query came from
      - ``topic``           : the chunk's topic

    Why the first sentence? It is a partial signal: the query exercises
    the retriever the same way a real user query would, by asking the
    index to recover the full surrounding chunk from a fragment.

    A different seed from ``generate_corpus`` ensures the queries are not
    biased toward the first chunks generated.

    Raises ``ValueError`` if ``n_queries`` is not a positive multiple of
    the number of topics, or if any topic has fewer chunks than the
    per-topic query count.
    """
    n_topics: int = len(TOPIC_SENTENCES)
    if n_queries <= 0 or n_queries % n_topics != 0:
        raise ValueError(
            f"n_queries must be a positive multiple of {n_topics} (got {n_queries})"
        )

    per_topic: int = n_queries // n_topics
    rng: np.random.Generator = np.random.default_rng(seed)

    by_topic: dict[str, list[dict]] = {topic: [] for topic in TOPIC_SENTENCES}
    for chunk in corpus:
        by_topic[chunk["topic"]].append(chunk)

    queries: list[dict] = []
    for topic in TOPIC_SENTENCES:
        queries.extend(_pick_topic_queries(by_topic[topic], per_topic, rng, topic))
    return queries
