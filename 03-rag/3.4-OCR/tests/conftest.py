"""Shared pytest fixtures for the 3.4-OCR test suite."""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path
from typing import Any, Generator

import pytest

from src.region_chunker import chunk_regions

OCR_DIR = Path(__file__).resolve().parent.parent

_COMMON: dict[str, Any] = {
    "source_pdf": "test.pdf",
    "bbox": [0.0, 0.0, 1.0, 1.0],
    "table_data": None,
}


def _region(
    region_type: str,
    text: str,
    page_num: int,
    reading_order: int,
    heading_path: list[str],
    *,
    bbox: list[float] | None = None,
    table_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **_COMMON,
        "region_type": region_type,
        "text": text,
        "page_num": page_num,
        "reading_order": reading_order,
        "heading_path": heading_path,
        "bbox": bbox if bbox is not None else list(_COMMON["bbox"]),
        "table_data": table_data,
    }


_TABLE_MARKDOWN = (
    "| Model | Accuracy | F1 Score |\n"
    "|---|---|---|\n"
    "| BERT | 0.91 | 0.89 |\n"
    "| GPT-2 | 0.87 | 0.85 |\n"
    "| RoBERTa | 0.93 | 0.91 |"
)


@pytest.fixture(scope="session")
def synthetic_regions() -> list[dict[str, Any]]:
    """Hardcoded regions simulating parse_pdf_to_regions() output — no Docling needed."""
    doc_title = ["Machine Learning Report"]
    intro = doc_title + ["1. Introduction"]
    results = doc_title + ["2. Results"]
    conclusion = doc_title + ["3. Conclusion"]

    return [
        _region("title", "Machine Learning Report", 1, 0, []),
        _region("heading", "1. Introduction", 1, 1, doc_title),
        _region(
            "text",
            "Neural networks learn hierarchical representations through backpropagation. "
            "Gradient descent iteratively adjusts weights to minimize the loss function. "
            "Overfitting occurs when a model memorizes training data rather than generalizing. "
            "Transformers use self-attention to model long-range token dependencies.",
            1,
            2,
            intro,
        ),
        _region("heading", "2. Results", 2, 3, doc_title),
        _region(
            "caption",
            "Table 1: Accuracy comparison across models.",
            2,
            4,
            results,
        ),
        _region(
            "table",
            _TABLE_MARKDOWN,
            2,
            5,
            results,
            bbox=[0.1, 0.3, 0.9, 0.6],
            table_data={
                "markdown": _TABLE_MARKDOWN,
                "rows": 3,
                "cols": 3,
            },
        ),
        _region(
            "text",
            "The results demonstrate that RoBERTa achieves the highest accuracy. "
            "Baseline comparisons confirm the transformer architecture advantage. "
            "All models were evaluated on the same held-out test set.",
            2,
            6,
            results,
        ),
        _region("figure", "", 2, 7, results),
        _region("heading", "3. Conclusion", 3, 8, doc_title),
        _region(
            "text",
            "This paper presented a comprehensive evaluation of transformer models. "
            "Future work will extend these benchmarks to multilingual datasets. "
            "The findings have significant implications for NLP practitioners.",
            3,
            9,
            conclusion,
        ),
        _region(
            "header",
            "Machine Learning Report — Page 2",
            2,
            10,
            results,
        ),
        _region(
            "footer",
            "Confidential — Do Not Distribute",
            2,
            11,
            results,
        ),
    ]


@pytest.fixture(scope="session")
def table_region(synthetic_regions: list[dict[str, Any]]) -> dict[str, Any]:
    """The table region with known Markdown content."""
    return synthetic_regions[5]


@pytest.fixture(scope="session")
def text_region(synthetic_regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Prose region about neural networks."""
    return synthetic_regions[2]


@pytest.fixture(scope="session")
def sample_chunks(synthetic_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic chunks from chunk_regions() — no API calls."""
    return chunk_regions(synthetic_regions)


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    """Minimal 2-page born-digital PDF for classifier tests."""
    if importlib.util.find_spec("reportlab") is None:
        pytest.skip("reportlab not installed")

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdf_path = tmp_path / "tiny.pdf"
    styles = getSampleStyleSheet()
    story: list = []

    def _build_page1(story: list) -> None:
        story.append(Paragraph("Test Document", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Section 1", styles["Heading2"]))
        story.append(
            Paragraph(
                "This is the first paragraph of section one. It contains enough "
                "extractable text to classify the page as born-digital content.",
                styles["Normal"],
            )
        )
        story.append(
            Paragraph(
                "This is the second paragraph continuing the discussion with "
                "additional sentences for a realistic character count per page.",
                styles["Normal"],
            )
        )

    def _build_page2(story: list) -> None:
        story.append(Paragraph("Section 2", styles["Heading2"]))
        table_data = [
            ["A", "B", "C"],
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        table = Table(table_data, colWidths=[80, 80, 80])
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(
            Paragraph(
                "Section two summarizes tabular data and provides concluding "
                "remarks about the values shown in the table above.",
                styles["Normal"],
            )
        )

    _build_page1(story)
    story.append(PageBreak())
    _build_page2(story)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    doc.build(story)

    return pdf_path


@pytest.fixture
def qdrant_client() -> Generator[Any, None, None]:
    """Connect to local Qdrant; skip integration tests if unavailable."""
    try:
        sock = socket.create_connection(("localhost", 6333), timeout=2)
        sock.close()
    except OSError:
        pytest.skip("Qdrant not running — integration test skipped")

    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=30)
    created: list[str] = []

    class TrackingClient:
        """Proxy that records collection names for teardown."""

        def __init__(self, inner: QdrantClient) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self._inner, name)
            if name == "create_collection":

                def _create(*args: Any, **kwargs: Any) -> Any:
                    coll = kwargs.get("collection_name") or (args[0] if args else None)
                    if coll:
                        created.append(str(coll))
                    return attr(*args, **kwargs)

                return _create
            return attr

    tracked = TrackingClient(client)
    yield tracked

    for coll_name in created:
        try:
            if client.collection_exists(collection_name=coll_name):
                client.delete_collection(collection_name=coll_name)
        except Exception:
            pass


@pytest.fixture(scope="session")
def api_key_present() -> None:
    """Ensure VOYAGE_API_KEY is set for integration tests."""
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set — integration test skipped")
