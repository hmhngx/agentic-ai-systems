import numpy as np
import pytest
from unittest.mock import MagicMock

from src.retriever import embed_query, vanilla_retrieve, RetrievedChunk


def _mock_qdrant_point(chunk_id: str, text: str, score: float, entities: list[str]):
    p = MagicMock()
    p.score = score
    p.payload = {"chunk_id": chunk_id, "text": text, "entities": entities}
    return p


def test_embed_query_returns_normalized_vector():
    vec = embed_query("What is the training method for GPT-4?")
    assert vec.ndim == 1
    assert vec.dtype == np.float32
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_query_different_queries_differ():
    v1 = embed_query("OpenAI CEO")
    v2 = embed_query("protein folding database")
    assert not np.allclose(v1, v2)


def test_vanilla_retrieve_returns_retrieved_chunks():
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        _mock_qdrant_point("gpt4", "GPT-4 text", 0.91, ["GPT-4"]),
        _mock_qdrant_point("altman_ceo", "Sam Altman text", 0.85, ["Sam Altman", "OpenAI"]),
    ]
    chunks = vanilla_retrieve(mock_client, np.zeros(256, dtype=np.float32))
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "gpt4"
    assert chunks[0].score == pytest.approx(0.91)
    assert chunks[0].source == "vector"
    assert isinstance(chunks[0], RetrievedChunk)


def test_vanilla_retrieve_top_k_passed_to_qdrant():
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = []
    vanilla_retrieve(mock_client, np.zeros(256, dtype=np.float32), top_k=3)
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["limit"] == 3
