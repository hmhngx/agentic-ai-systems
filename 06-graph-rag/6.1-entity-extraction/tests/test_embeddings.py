import numpy as np

from src import embeddings
from src.schema import Entity, EntityType


def test_offline_embeddings_are_deterministic_and_normalized(monkeypatch):
    monkeypatch.setattr(embeddings.config, "use_embeddings_api", lambda: False)
    ents = [
        Entity(name="Claude", type=EntityType.MODEL),
        Entity(name="Stanford University", type=EntityType.ORG),
    ]
    v1 = embeddings.embed_entities(ents)
    v2 = embeddings.embed_entities(ents)
    assert v1.shape == (2, embeddings.embedding_dim())
    assert np.allclose(v1, v2)                       # deterministic
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)  # L2-normalized


def test_embedding_text_includes_name_type_aliases():
    e = Entity(
        name="UC Berkeley",
        type=EntityType.ORG,
        aliases=["University of California, Berkeley"],
    )
    text = embeddings.entity_text(e)
    assert "UC Berkeley" in text and "ORG" in text and "University of California" in text
