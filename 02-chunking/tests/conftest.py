"""Shared pytest fixtures for the 02-chunking test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (semantic model loading)")

# Hermetic runs: use locally cached models only (no Hub downloads during tests).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="session", autouse=True)
def _warm_nlp_and_embedder():
    """Load spaCy and the sentence embedder once per session to avoid repeated cold starts."""
    from src.chunkers import _get_embedder, _load_nlp

    _load_nlp()
    _get_embedder()


@pytest.fixture(scope="session")
def sample_plain_text() -> str:
    """Multi-topic plain text (2000+ chars) for chunker tests."""
    p1 = (
        "Machine learning is a branch of artificial intelligence focused on learning "
        "from data without explicit programming. Supervised learning maps inputs to "
        "labels using training examples. Unsupervised learning discovers structure in "
        "unlabeled data through clustering and dimensionality reduction. Reinforcement "
        "learning trains agents with rewards and penalties in simulated environments."
    )
    p2 = (
        "Deep learning uses neural networks with many layers to model complex patterns. "
        "Convolutional networks excel at images while transformers dominate language tasks. "
        "Gradient descent and backpropagation adjust weights to minimize loss on batches. "
        "Regularization, dropout, and early stopping help models generalize beyond training "
        "sets and avoid memorizing noise in small datasets."
    )
    p3 = (
        "Ocean biology studies life in marine ecosystems from plankton to whales. "
        "Phytoplankton form the base of food webs and produce much of Earth's oxygen. "
        "Coral reefs shelter diverse species but face bleaching from warming seas. "
        "Deep-sea vents host chemosynthetic bacteria that sustain unique communities "
        "without sunlight, revealing how life adapts to extreme pressure and chemistry."
    )
    p4 = (
        "Marine mammals such as dolphins use echolocation to hunt in murky water. "
        "Migratory species follow currents and temperature fronts across ocean basins. "
        "Overfishing and plastic pollution disrupt populations and food chains worldwide. "
        "Marine protected areas aim to restore habitats and allow fish stocks to recover "
        "while scientists monitor biodiversity with autonomous underwater vehicles."
    )
    p5 = (
        "Cooking transforms raw ingredients through heat, acid, salt, and time. "
        "Sautéing quickly browns vegetables while preserving texture and color. "
        "Slow braising breaks down collagen in tough cuts into tender, flavorful meat. "
        "Balancing sweet, sour, salty, bitter, and umami creates satisfying dishes that "
        "highlight seasonal produce and regional traditions passed down generations."
    )
    p6 = (
        "Baking requires precise ratios of flour, liquid, fat, and leavening agents. "
        "Yeast fermentation produces carbon dioxide that lightens bread crumb structure. "
        "Caramelization and Maillard reactions develop aroma and crust on pastries and roasts. "
        "Food safety practices such as proper storage temperatures prevent spoilage and "
        "illness while mise en place keeps professional kitchens efficient during service."
    )
    return "\n\n".join([p1, p2, p3, p4, p5, p6])


@pytest.fixture(scope="session")
def minimal_text() -> str:
    """Single short sentence for edge-case chunker input."""
    return "This is a test."


@pytest.fixture(scope="session")
def empty_text() -> str:
    """Empty string for edge-case chunker input."""
    return ""


@pytest.fixture(scope="session")
def session_pdf(tmp_path_factory) -> Path:
    """Session-scoped valid PDF reused by CLI tests (one reportlab build per run)."""
    tmp_dir = tmp_path_factory.mktemp("session_pdf")
    pdf_path = tmp_dir / "test.pdf"
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    words = (
        "Machine learning extracts patterns from data using optimization and statistics. "
        "Ocean biology explores ecosystems from plankton to whales across diverse habitats. "
        "Cooking applies heat salt acid and time to transform ingredients into nourishing meals. "
    ) * 40
    page1, page2 = words[:2500], words[2500:5000]
    width, height = letter
    margin = 50
    y = height - margin
    for line in page1.split(". "):
        if y < margin:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line[:90] + ".")
        y -= 14
    c.showPage()
    y = height - margin
    for line in page2.split(". "):
        if y < margin:
            break
        c.drawString(margin, y, line[:90] + ".")
        y -= 14
    c.save()
    return pdf_path


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """Minimal valid two-page PDF with 500+ words of readable text."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    words = (
        "Machine learning extracts patterns from data using optimization and statistics. "
        "Ocean biology explores ecosystems from plankton to whales across diverse habitats. "
        "Cooking applies heat salt acid and time to transform ingredients into nourishing meals. "
    ) * 40

    page1 = words[:2500]
    page2 = words[2500:5000]
    width, height = letter
    margin = 50
    y = height - margin

    for line in page1.split(". "):
        if y < margin:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line[:90] + ".")
        y -= 14

    c.showPage()
    y = height - margin
    for line in page2.split(". "):
        if y < margin:
            break
        c.drawString(margin, y, line[:90] + ".")
        y -= 14

    c.save()
    return pdf_path


@pytest.fixture
def tmp_invalid_file(tmp_path: Path) -> Path:
    """Non-PDF bytes saved with a .pdf extension."""
    bad = tmp_path / "not_a_pdf.pdf"
    bad.write_bytes(b"NOT A PDF")
    return bad
