import math

import numpy as np

from src.corpus import documents
from src.embedder import TfidfSpace, build_space


def test_idf_zero_for_ubiquitous_term():
    space = build_space(documents())
    # 'helios' appears in all 7 docs => idf = log(N/df) = log(1) = 0
    assert space.idf.get("helios", 0.0) == 0.0
    # 'retention' appears in 1 doc => idf = log(7) > 0
    assert space.idf["retention"] == math.log(7 / 1)


def test_ubiquitous_set_contains_helios_only():
    space = build_space(documents())
    assert space.ubiquitous == {"helios"}


def test_embed_is_unit_norm_and_deterministic():
    space = build_space(documents())
    v1 = space.embed("Helios query quota per day")
    v2 = space.embed("Helios query quota per day")
    assert np.allclose(v1, v2)
    assert v1.shape == (space.dim,)
    assert math.isclose(float(np.linalg.norm(v1)), 1.0, rel_tol=1e-6)


def test_oov_only_query_is_zero_vector():
    space = build_space(documents())
    v = space.embed("carbon footprint")  # neither token in vocab
    assert float(np.linalg.norm(v)) == 0.0


def test_idf_mass_sums_distinctive_terms_in_vocab():
    space = build_space(documents())
    # 'quota' (idf>0) counts; 'helios' (idf 0) and 'carbon' (OOV) do not.
    expected = space.idf["quota"]
    assert math.isclose(space.idf_mass(["quota", "helios", "carbon"]), expected, rel_tol=1e-9)
