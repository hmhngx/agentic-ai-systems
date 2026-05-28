"""Shared fixtures for the NaiveRAG test suite.

Fixtures are split into two tiers:

  * Pure-data fixtures (``sample_chunks``, ``sample_results``,
    ``low_score_results``) - session-scoped, fully offline.
  * Live-resource fixtures (``sample_pdf``, ``qdrant_client``,
    ``api_keys_present``) - either function-scoped or guarded by
    ``pytest.skip`` so the offline tier still passes without Qdrant
    or OpenRouter credentials.

All fixed UUIDs and scores are hardcoded for determinism: two test
runs must produce identical bytes, so re-running on flake never
hides a real regression.
"""

from __future__ import annotations

import os
import pathlib
from typing import Iterator

import pytest


# -----------------------------------------------------------------------------
# Topic corpus
# -----------------------------------------------------------------------------
# Three disjoint topics (machine learning, ocean biology, cooking) with three
# paragraphs each. Disjoint topics make retrieval-relevance assertions
# meaningful: an ML query MUST retrieve page 1 chunks, not page 2 or page 3.
# Each paragraph is 80+ words across 4-6 sentences so chunk_pages produces
# multiple non-trivial chunks per page.
# -----------------------------------------------------------------------------

_ML_PARAGRAPHS = [
    (
        "Backpropagation is the fundamental algorithm used to train deep "
        "neural networks across modern machine learning systems. The method "
        "works by computing the gradient of the loss function with respect "
        "to each weight using the chain rule of calculus. During the "
        "backward pass, errors are propagated from the output layer back "
        "through every hidden layer in turn. Each weight is then adjusted "
        "in the direction that reduces the overall loss on the training "
        "examples. This process is repeated across many training iterations "
        "until the network converges to a useful solution that generalises "
        "to new inputs."
    ),
    (
        "Gradient descent is the optimisation workhorse that powers nearly "
        "every machine learning model in production today. The algorithm "
        "iteratively adjusts model parameters by moving them in the "
        "direction opposite to the gradient of the loss function. "
        "Stochastic gradient descent variants use small batches of training "
        "data to estimate the gradient more cheaply at each step. Learning "
        "rate scheduling controls how aggressively the parameters move "
        "during each update across the training run. Momentum-based "
        "variants such as Adam accelerate convergence by accumulating past "
        "gradient information over time."
    ),
    (
        "Loss functions quantify the gap between a model's predictions and "
        "the ground truth labels in the training set. Cross-entropy loss "
        "is the standard choice for classification problems because it "
        "heavily penalises confident but incorrect predictions. Mean "
        "squared error remains the default loss for regression tasks "
        "whose outputs are continuous quantities. The choice of loss "
        "function shapes the entire training trajectory and determines "
        "which mistakes the model learns to avoid first. A well-designed "
        "loss function aligns the optimisation target with the real-world "
        "objective the system is meant to achieve."
    ),
]

_OCEAN_PARAGRAPHS = [
    (
        "Coral reefs are among the most biologically diverse ecosystems on "
        "Earth, supporting roughly a quarter of all marine species. These "
        "underwater structures are built over thousands of years by tiny "
        "coral polyps that secrete calcium carbonate skeletons. Healthy "
        "reefs depend on a delicate symbiotic relationship between coral "
        "animals and photosynthetic algae called zooxanthellae. Rising "
        "ocean temperatures cause the algae to be expelled in a stress "
        "response widely known as coral bleaching. Without their algal "
        "partners, the corals lose their primary food source and often "
        "die within weeks of bleaching events."
    ),
    (
        "Bioluminescence is the production of light by living organisms "
        "through chemical reactions inside specialised cells. In the deep "
        "ocean, more than three quarters of animals are capable of "
        "producing their own light at depth. Most marine bioluminescence "
        "appears blue-green in colour because those wavelengths penetrate "
        "seawater the most efficiently. Species use light for predator "
        "defence, prey attraction, mate signalling, and camouflage through "
        "counter-illumination from below. The chemistry typically involves "
        "a substrate called luciferin reacting with oxygen under the "
        "catalysis of an enzyme called luciferase."
    ),
    (
        "Ocean currents are continuous flows of seawater driven by wind "
        "patterns, temperature gradients, and differences in salinity "
        "across the globe. The thermohaline circulation moves heat between "
        "the equator and the poles on timescales of many centuries. "
        "Surface currents like the Gulf Stream transport warm tropical "
        "water northward and profoundly influence European weather and "
        "climate. Upwelling zones along continental margins bring "
        "nutrient-rich deep water to the sunlit surface, fuelling some of "
        "the planet's most productive fisheries. Disruptions to these "
        "circulation patterns from climate change could reshape global "
        "weather systems for generations to come."
    ),
]

_COOK_PARAGRAPHS = [
    (
        "The Maillard reaction is a complex chemical browning process that "
        "gives roasted meats, baked breads, and seared vegetables their "
        "distinctive flavour and brown colour. The reaction occurs between "
        "amino acids and reducing sugars at temperatures above roughly one "
        "hundred and forty degrees Celsius. Hundreds of new aroma "
        "compounds are produced as the reaction progresses, each "
        "contributing to the deep savoury character of cooked food. "
        "Moisture must be driven off before the surface can brown "
        "effectively, which is why patting meat dry before searing "
        "dramatically improves the resulting crust. Skilled cooks learn "
        "to manage heat and timing to maximise Maillard browning without "
        "burning the food."
    ),
    (
        "Emulsification is the technique of stably combining two liquids "
        "that normally refuse to mix together, such as oil and "
        "water-based ingredients. Classic culinary emulsions include "
        "mayonnaise, vinaigrette, and hollandaise sauce, all of which "
        "suspend tiny droplets of one liquid throughout another. "
        "Emulsifiers like egg yolks contain molecules that have both "
        "water-loving and oil-loving regions and so can bridge the "
        "interface between the two phases. Vigorous mixing breaks one "
        "liquid into microscopic droplets while the emulsifier stabilises "
        "the newly created surface area. A broken emulsion can often be "
        "rescued by slowly whisking the broken mixture back into a fresh "
        "emulsifier base."
    ),
    (
        "Braising is a combination cooking method that uses both dry heat "
        "for initial browning and moist heat for the long simmer that "
        "follows. The technique is ideal for tough cuts of meat that are "
        "rich in collagen but would be inedible if cooked quickly at high "
        "temperatures. A long, gentle simmer in flavourful liquid "
        "converts that collagen into gelatin, yielding silky, fork-tender "
        "results no quick cooking method can replicate. The braising "
        "liquid simultaneously absorbs the rich flavours released by the "
        "meat and aromatics, transforming into a deeply savoury sauce by "
        "the time the dish is finished. Patience is the only real "
        "ingredient that cannot be substituted in a great braise."
    ),
]


# -----------------------------------------------------------------------------
# Fixed text used by the in-memory chunk / result fixtures
# -----------------------------------------------------------------------------
# These strings are intentionally distinct in their first 20 characters so
# tests can assert "first_20 in formatted_output" without ambiguity.
# -----------------------------------------------------------------------------

_ML_TEXT_1 = (
    "Backpropagation is the fundamental algorithm used to train deep "
    "neural networks. The method computes the gradient of the loss "
    "function with respect to each weight via the chain rule. Errors "
    "propagate from output to input layer during the backward pass."
)
_ML_TEXT_2 = (
    "Gradient descent iteratively adjusts model parameters by moving "
    "them opposite to the loss gradient. Variants include stochastic "
    "gradient descent and Adam. Learning rate schedules control update "
    "magnitudes during training."
)
_OCEAN_TEXT_1 = (
    "Coral reefs support roughly a quarter of all marine species "
    "despite covering a tiny fraction of the ocean. Reefs are built "
    "over thousands of years by polyps secreting calcium carbonate. "
    "Rising temperatures cause coral bleaching when symbiotic algae "
    "are expelled."
)
_OCEAN_TEXT_2 = (
    "Bioluminescence is the production of light by living organisms "
    "through chemical reactions. Most marine bioluminescence appears "
    "blue-green for efficient transmission through water. Animals use "
    "it for defence, predation, mating, and counter-illumination."
)
_COOK_TEXT_1 = (
    "The Maillard reaction is a complex chemical browning between "
    "amino acids and reducing sugars at high heat. It produces "
    "hundreds of new aroma compounds in roasted meats and breads. "
    "Moisture must be driven off before browning begins on a hot "
    "surface."
)
_COOK_TEXT_2 = (
    "Emulsification combines two normally immiscible liquids by "
    "stabilising droplet interfaces with surfactants. Mayonnaise and "
    "vinaigrette are classic culinary emulsions. Broken emulsions can "
    "sometimes be rescued by whisking into a fresh emulsifier base."
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a deterministic 3-page test PDF with disjoint per-page topics.

    Page 1 = ML, page 2 = ocean biology, page 3 = cooking. This layout
    lets retrieval tests verify topic-specific queries route to the
    correct page rather than dragging in chunks from unrelated pages.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed - cannot generate offline test PDF")

    pdf_path = tmp_path / "test_doc.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    _width, height = letter
    max_chars_per_line = 85  # visually fits inside 1-inch margins at 11pt

    def render_page(paragraphs: list[str]) -> None:
        c.setFont("Helvetica", 11)
        y = height - 72
        for para in paragraphs:
            line = ""
            for word in para.split():
                trial = (line + " " + word).strip() if line else word
                if len(trial) > max_chars_per_line:
                    c.drawString(72, y, line)
                    y -= 14
                    line = word
                else:
                    line = trial
            if line:
                c.drawString(72, y, line)
                y -= 14
            y -= 10
        c.showPage()

    render_page(_ML_PARAGRAPHS)
    render_page(_OCEAN_PARAGRAPHS)
    render_page(_COOK_PARAGRAPHS)
    c.save()

    return pdf_path


def _make_chunk(chunk_id: str, text: str, page_num: int, chunk_index: int) -> dict:
    """Build a chunk dict matching pdf_loader._build_chunk's schema."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "page_num": page_num,
        "chunk_index": chunk_index,
        "token_count": len(text) // 4,
        "char_count": len(text),
        "source_pdf": "test_doc.pdf",
    }


@pytest.fixture(scope="session")
def sample_chunks() -> list[dict]:
    """Six chunks (two per topic) with fixed UUIDs for deterministic upsert.

    Fixed UUIDs are essential: ``vector_store._uuid_to_point_id`` derives
    Qdrant point IDs from these strings, so any drift would shift the
    point-id namespace and silently invalidate upsert/retrieve tests.
    """
    return [
        _make_chunk("a1b2c3d4-0000-0000-0000-000000000001", _ML_TEXT_1, 1, 0),
        _make_chunk("a1b2c3d4-0000-0000-0000-000000000002", _ML_TEXT_2, 1, 1),
        _make_chunk("a1b2c3d4-0000-0000-0000-000000000003", _OCEAN_TEXT_1, 2, 0),
        _make_chunk("a1b2c3d4-0000-0000-0000-000000000004", _OCEAN_TEXT_2, 2, 1),
        _make_chunk("a1b2c3d4-0000-0000-0000-000000000005", _COOK_TEXT_1, 3, 0),
        _make_chunk("a1b2c3d4-0000-0000-0000-000000000006", _COOK_TEXT_2, 3, 1),
    ]


def _make_result(
    text: str, page_num: int, chunk_id: str, score: float, rank: int
) -> dict:
    """Build a search-result dict matching vector_store.search's schema."""
    return {
        "text": text,
        "page_num": page_num,
        "source_pdf": "test_doc.pdf",
        "chunk_id": chunk_id,
        "score": score,
        "rank": rank,
    }


@pytest.fixture(scope="session")
def sample_results() -> list[dict]:
    """Three high-confidence results - all above MIN_SCORE_THRESHOLD = 0.25.

    Scores 0.92 / 0.78 / 0.61 simulate a clean retrieval where the top
    result is a near-perfect match and the lowest is still confidently
    above the floor. One result per topic so context-block tests can
    verify per-doc formatting independently.
    """
    return [
        _make_result(_ML_TEXT_1,    1, "a1b2c3d4-0000-0000-0000-000000000001", 0.92, 1),
        _make_result(_OCEAN_TEXT_1, 2, "a1b2c3d4-0000-0000-0000-000000000003", 0.78, 2),
        _make_result(_COOK_TEXT_1,  3, "a1b2c3d4-0000-0000-0000-000000000005", 0.61, 3),
    ]


@pytest.fixture(scope="session")
def low_score_results() -> list[dict]:
    """Three results all below MIN_SCORE_THRESHOLD - simulates failed retrieval.

    The retriever must return NO_RESULTS in this case; sending these to
    the LLM with status="OK" would be the textbook hallucination path.
    """
    return [
        _make_result(_ML_TEXT_1,    1, "a1b2c3d4-0000-0000-0000-000000000001", 0.24, 1),
        _make_result(_OCEAN_TEXT_1, 2, "a1b2c3d4-0000-0000-0000-000000000003", 0.21, 2),
        _make_result(_COOK_TEXT_1,  3, "a1b2c3d4-0000-0000-0000-000000000005", 0.22, 3),
    ]


@pytest.fixture
def qdrant_client() -> Iterator[object]:
    """Live Qdrant client; auto-skip if the server is unreachable.

    Cleans up any collection created during the test by diffing the
    collection set on entry and exit. This keeps the dev Qdrant
    instance free of stray ``test_*`` collections even when a test
    crashes mid-flight before its own cleanup runs.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("qdrant-client not installed")

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=url, timeout=5)
        before = {c.name for c in client.get_collections().collections}
    except Exception as exc:  # noqa: BLE001 - any connect failure -> skip
        pytest.skip(f"Qdrant not running at {url} - integration test skipped ({exc})")

    yield client

    try:
        after = {c.name for c in client.get_collections().collections}
        for name in after - before:
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="session")
def api_keys_present() -> None:
    """Skip the calling test when OpenRouter key is missing.

    OpenRouter powers both embedding and generation in this module.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set - integration test skipped")
