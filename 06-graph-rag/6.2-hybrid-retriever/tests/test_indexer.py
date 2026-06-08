import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from src.indexer import _hash_embed, _link_entities_to_chunks, _chunk_id


def test_hash_embed_shape_and_normalized():
    vecs = _hash_embed(["hello world", "foo bar"])
    assert vecs.shape == (2, 256)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hash_embed_deterministic():
    a = _hash_embed(["test text"])
    b = _hash_embed(["test text"])
    assert np.allclose(a, b)


def test_hash_embed_different_texts_differ():
    vecs = _hash_embed(["Anthropic uses Constitutional AI", "DeepMind built AlphaFold"])
    assert not np.allclose(vecs[0], vecs[1])


def test_chunk_id_stable():
    assert _chunk_id("altman_ceo") == _chunk_id("altman_ceo")
    assert _chunk_id("altman_ceo") != _chunk_id("gpt4")


def test_link_entities_finds_known_entities():
    import networkx as nx
    G = nx.MultiDiGraph()
    G.add_node("OpenAI"); G.add_node("GPT-4")
    mapping = _link_entities_to_chunks(G)
    # altman_ceo passage contains "OpenAI"
    assert "OpenAI" in mapping["altman_ceo"]
    # gpt4 passage contains "GPT-4" and "OpenAI"
    assert "GPT-4" in mapping["gpt4"]
