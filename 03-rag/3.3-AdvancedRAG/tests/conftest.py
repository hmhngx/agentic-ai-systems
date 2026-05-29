"""Shared fixtures for Advanced RAG offline and integration tests."""

from __future__ import annotations

import os

import pytest

from src.bm25_retriever import build_bm25_index
from src.dense_retriever import COLLECTION_NAME


@pytest.fixture(scope="session")
def tiny_corpus() -> list[dict]:
    """Ten-chunk synthetic corpus (two per topic) for fast BM25/RRF tests."""
    chunks: list[dict] = [
        {
            "id": 0,
            "chunk_id": "11111111-1111-4111-8111-111111110000",
            "text": (
                "Neural networks learn hierarchical representations through backpropagation. "
                "Gradient descent iteratively adjusts weights to minimize the loss function. "
                "Regularization techniques like dropout prevent co-adaptation of neurons."
            ),
            "page_num": 1,
            "chunk_index": 0,
            "token_count": 42,
            "char_count": 210,
            "source_pdf": "test.pdf",
            "topic": "machine_learning",
        },
        {
            "id": 1,
            "chunk_id": "22222222-2222-4222-8222-222222220001",
            "text": (
                "Transformers use self-attention to model long-range token dependencies. "
                "Cross-entropy loss measures divergence between predicted and true distributions. "
                "Embedding spaces encode semantic relationships as geometric distances."
            ),
            "page_num": 1,
            "chunk_index": 1,
            "token_count": 40,
            "char_count": 200,
            "source_pdf": "test.pdf",
            "topic": "machine_learning",
        },
        {
            "id": 2,
            "chunk_id": "33333333-3333-4333-8333-333333330002",
            "text": (
                "Coral reefs support approximately twenty-five percent of all marine species. "
                "Bioluminescent organisms produce light through chemical reactions involving luciferin. "
                "Phytoplankton produce roughly half of Earth's atmospheric oxygen through photosynthesis."
            ),
            "page_num": 2,
            "chunk_index": 0,
            "token_count": 38,
            "char_count": 195,
            "source_pdf": "test.pdf",
            "topic": "ocean_biology",
        },
        {
            "id": 3,
            "chunk_id": "44444444-4444-4444-8444-444444440003",
            "text": (
                "Deep-sea hydrothermal vents sustain ecosystems independent of solar energy. "
                "Whale migration routes span thousands of miles between feeding and breeding grounds. "
                "The abyssal zone receives no sunlight and supports unique pressure-adapted organisms."
            ),
            "page_num": 2,
            "chunk_index": 1,
            "token_count": 36,
            "char_count": 188,
            "source_pdf": "test.pdf",
            "topic": "ocean_biology",
        },
        {
            "id": 4,
            "chunk_id": "55555555-5555-4555-8555-555555550004",
            "text": (
                "The Roman Republic transitioned to Empire following Julius Caesar's assassination. "
                "Egyptian hieroglyphs combined logographic and alphabetic elements in a complex system. "
                "The Silk Road facilitated trade and cultural exchange between East Asia and the Mediterranean."
            ),
            "page_num": 3,
            "chunk_index": 0,
            "token_count": 41,
            "char_count": 205,
            "source_pdf": "test.pdf",
            "topic": "ancient_history",
        },
        {
            "id": 5,
            "chunk_id": "66666666-6666-4666-8666-666666660005",
            "text": (
                "Greek city-states developed diverse governance systems ranging from democracy to oligarchy. "
                "The construction of the Great Wall spanned multiple Chinese dynasties over centuries. "
                "Mesopotamian civilizations developed early legal codes including the Code of Hammurabi."
            ),
            "page_num": 3,
            "chunk_index": 1,
            "token_count": 39,
            "char_count": 198,
            "source_pdf": "test.pdf",
            "topic": "ancient_history",
        },
        {
            "id": 6,
            "chunk_id": "77777777-7777-4777-8777-777777770006",
            "text": (
                "Maillard reaction between amino acids and reducing sugars creates browning and complex flavors. "
                "Emulsification combines immiscible liquids like oil and water through mechanical agitation. "
                "Braising uses moist heat to break down collagen in tough cuts into gelatin."
            ),
            "page_num": 4,
            "chunk_index": 0,
            "token_count": 37,
            "char_count": 192,
            "source_pdf": "test.pdf",
            "topic": "cooking",
        },
        {
            "id": 7,
            "chunk_id": "88888888-8888-4888-8888-888888880007",
            "text": (
                "Fermentation by microorganisms transforms sugars into alcohol, acids, and carbon dioxide. "
                "Salt draws moisture from vegetables through osmosis and seasons from within. "
                "Acid ingredients like lemon juice denature proteins and balance rich flavors."
            ),
            "page_num": 4,
            "chunk_index": 1,
            "token_count": 35,
            "char_count": 185,
            "source_pdf": "test.pdf",
            "topic": "cooking",
        },
        {
            "id": 8,
            "chunk_id": "99999999-9999-4999-8999-999999990008",
            "text": (
                "Modernist architecture rejected ornament in favor of functional form and industrial materials. "
                "Brutalist structures express raw concrete as both structural and aesthetic material. "
                "Parametric design uses computational algorithms to generate complex building geometries."
            ),
            "page_num": 5,
            "chunk_index": 0,
            "token_count": 34,
            "char_count": 180,
            "source_pdf": "test.pdf",
            "topic": "urban_architecture",
        },
        {
            "id": 9,
            "chunk_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaa0009",
            "text": (
                "Green building standards measure energy efficiency, water use, and indoor air quality. "
                "Transit-oriented development concentrates density around public transport nodes. "
                "Mixed-use developments reduce car dependency by placing housing near jobs and services."
            ),
            "page_num": 5,
            "chunk_index": 1,
            "token_count": 33,
            "char_count": 175,
            "source_pdf": "test.pdf",
            "topic": "urban_architecture",
        },
    ]
    return chunks


@pytest.fixture(scope="session")
def tiny_corpus_texts(tiny_corpus: list[dict]) -> list[str]:
    """Plain-text bodies from ``tiny_corpus`` for tokenizer assertions."""
    return [c["text"] for c in tiny_corpus]


@pytest.fixture(scope="session")
def tiny_chunk_ids(tiny_corpus: list[dict]) -> list[str]:
    """Chunk UUID strings from ``tiny_corpus`` in corpus order."""
    return [c["chunk_id"] for c in tiny_corpus]


@pytest.fixture(scope="session")
def bm25_index_and_ids(tiny_corpus: list[dict]):
    """Session BM25 index aligned with ``tiny_corpus`` chunk order."""
    return build_bm25_index(tiny_corpus)


@pytest.fixture(scope="session")
def sample_dense_results(tiny_chunk_ids: list[str]) -> list[dict]:
    """Hardcoded dense top-5 with even-index chunk_ids only."""
    return [
        {"chunk_id": tiny_chunk_ids[0], "dense_score": 0.92, "dense_rank": 1},
        {"chunk_id": tiny_chunk_ids[2], "dense_score": 0.85, "dense_rank": 2},
        {"chunk_id": tiny_chunk_ids[4], "dense_score": 0.78, "dense_rank": 3},
        {"chunk_id": tiny_chunk_ids[6], "dense_score": 0.71, "dense_rank": 4},
        {"chunk_id": tiny_chunk_ids[8], "dense_score": 0.65, "dense_rank": 5},
    ]


@pytest.fixture(scope="session")
def sample_bm25_results(tiny_chunk_ids: list[str]) -> list[dict]:
    """Hardcoded BM25 top-5 partially overlapping dense (ids 1, 3, 9 BM25-only)."""
    return [
        {"chunk_id": tiny_chunk_ids[0], "bm25_score": 8.3, "bm25_rank": 1},
        {"chunk_id": tiny_chunk_ids[1], "bm25_score": 7.1, "bm25_rank": 2},
        {"chunk_id": tiny_chunk_ids[2], "bm25_score": 6.4, "bm25_rank": 3},
        {"chunk_id": tiny_chunk_ids[3], "bm25_score": 5.8, "bm25_rank": 4},
        {"chunk_id": tiny_chunk_ids[9], "bm25_score": 4.2, "bm25_rank": 5},
    ]


@pytest.fixture
def qdrant_client():
    """Live Qdrant client; skip if Day 3 collection is missing or empty."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant-client not installed")

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=url, timeout=5)
        client.get_collections()
    except Exception:
        pytest.skip("Qdrant not running — integration test skipped")

    try:
        info = client.get_collection(CLECTION_NAME)
    except Exception:
        pytest.skip("Day 3 collection not found — run Day 3 first")

    points_count = int(getattr(info, "points_count", 0) or 0)
    vectors_count = int(getattr(info, "vectors_count", 0) or 0)
    if max(points_count, vectors_count) <= 0:
        pytest.skip("Day 3 collection not found — run Day 3 first")

    return client
