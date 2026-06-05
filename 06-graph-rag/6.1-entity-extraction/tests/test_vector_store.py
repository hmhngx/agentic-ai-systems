import numpy as np
import pytest

from src import vector_store
from src.schema import Entity, EntityType


def test_point_id_is_stable_and_int():
    a = vector_store.point_id("Stanford University")
    b = vector_store.point_id("Stanford University")
    assert a == b and isinstance(a, int) and a >= 0


def _qdrant_up() -> bool:
    try:
        import requests

        return requests.get("http://localhost:6333/healthz", timeout=2).ok
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_up(), reason="local Qdrant not running")
def test_upsert_and_search_roundtrip():
    client = vector_store.get_client()
    ents = [
        Entity(name="Claude", type=EntityType.MODEL, mentions=2),
        Entity(name="Stanford University", type=EntityType.ORG, mentions=3),
    ]
    dim = 4
    vecs = np.eye(2, dim, dtype=np.float32)  # tiny dummy vectors
    vector_store.create_collection(client, dim=dim)
    vector_store.upsert_entities(client, ents, vecs)
    hits = vector_store.search(client, np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
    assert hits and hits[0]["name"] == "Claude"
