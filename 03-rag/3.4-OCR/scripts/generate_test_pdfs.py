"""
Generate 3 synthetic test PDFs with known content for pipeline verification.
Uses reportlab. Run: python scripts/generate_test_pdfs.py
Outputs: sample_pdfs/test_academic.pdf, test_report.pdf, test_tables.pdf
Each PDF has: title, section headings, prose paragraphs, and at least 2 tables.
Table content is deterministic so verification queries can be hardcoded.

Why deterministic content?
  verifier.py runs fixed semantic queries ("revenue earnings financial results",
  "accuracy performance comparison results", "find the table about methodology",
  "introduction background motivation", "product specifications benchmark
  comparison"). Those queries only return meaningful chunks if the source PDFs
  actually contain that vocabulary, so the topics below are chosen to match.
"""

from __future__ import annotations

import os
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import StyleSheet1, getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable


# A section is a heading, its prose paragraphs, and an optional table
# (caption, header row, data rows).
Section = tuple[str, list[str], Optional[tuple[str, list[str], list[list[str]]]]]


def _output_dir() -> str:
    """Resolve (and create) the ``sample_pdfs`` output directory."""
    here: str = os.path.dirname(os.path.abspath(__file__))
    out_dir: str = os.path.join(here, "..", "sample_pdfs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _build_table(
    caption: str,
    header: list[str],
    rows: list[list[str]],
    styles: StyleSheet1,
) -> list[Flowable]:
    """Build a captioned, bordered table flowable group.

    The caption is emitted as its own paragraph immediately above the table so
    Docling tags it as a CAPTION region — that is what table_serializer uses to
    title the table (and what makes "find the table about X" queries work).
    """
    caption_para: Paragraph = Paragraph(f"<b>{caption}</b>", styles["Italic"])
    data: list[list[str]] = [header, *rows]
    table: Table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    return [caption_para, Spacer(1, 4), table, Spacer(1, 12)]


def build_pdf(path: str, title: str, sections: list[Section]) -> None:
    """Render a single PDF from a title and a list of sections."""
    styles: StyleSheet1 = getSampleStyleSheet()
    doc: SimpleDocTemplate = SimpleDocTemplate(
        path,
        pagesize=letter,
        title=title,
    )
    story: list[Flowable] = [Paragraph(title, styles["Title"]), Spacer(1, 16)]

    for heading, paragraphs, table in sections:
        story.append(Paragraph(heading, styles["Heading1"]))
        story.append(Spacer(1, 6))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyText"]))
            story.append(Spacer(1, 6))
        if table is not None:
            caption, header, rows = table
            story.extend(_build_table(caption, header, rows, styles))

    doc.build(story)
    print(f"Wrote {os.path.basename(path)}")


def _academic_sections() -> list[Section]:
    """Sections for test_academic.pdf (ML, gradient descent, accuracy table)."""
    return [
        (
            "Introduction",
            [
                "This paper studies machine learning models for text classification. "
                "The introduction provides background and motivation for comparing "
                "optimization strategies under a fixed compute budget.",
                "Our motivation is to understand how gradient descent variants affect "
                "final model accuracy on a held-out evaluation set.",
            ],
            None,
        ),
        (
            "Methodology",
            [
                "We train each model with stochastic gradient descent and report the "
                "mean accuracy across five random seeds. The methodology section "
                "describes the learning rate schedule and batch size.",
            ],
            (
                "Table 1: Accuracy comparison across optimizers",
                ["Optimizer", "Accuracy", "F1", "Epochs"],
                [
                    ["SGD", "0.842", "0.831", "20"],
                    ["Adam", "0.910", "0.905", "20"],
                    ["AdamW", "0.921", "0.918", "20"],
                ],
            ),
        ),
        (
            "Results",
            [
                "The results show that AdamW achieves the highest accuracy and F1 "
                "performance in our comparison. Gradient descent with momentum "
                "trailed the adaptive optimizers.",
            ],
            (
                "Table 2: Performance comparison by dataset",
                ["Dataset", "Accuracy", "Latency_ms"],
                [
                    ["News", "0.918", "12"],
                    ["Reviews", "0.904", "11"],
                ],
            ),
        ),
    ]


def _report_sections() -> list[Section]:
    """Sections for test_report.pdf (quarterly revenue, market analysis)."""
    return [
        (
            "Executive Summary",
            [
                "This report summarizes quarterly revenue and market analysis for the "
                "fiscal year. The introduction outlines the motivation behind our "
                "regional expansion strategy.",
            ],
            None,
        ),
        (
            "Financial Results",
            [
                "Quarterly revenue grew steadily, with earnings improving each quarter. "
                "The financial results below break down revenue by quarter.",
            ],
            (
                "Table 1: Quarterly revenue and earnings",
                ["Quarter", "Revenue_M", "Earnings_M"],
                [
                    ["Q1", "120", "18"],
                    ["Q2", "135", "22"],
                    ["Q3", "150", "27"],
                    ["Q4", "168", "31"],
                ],
            ),
        ),
        (
            "Market Analysis",
            [
                "Our market analysis indicates growing demand in the enterprise "
                "segment. Financial performance is expected to track revenue growth.",
            ],
            (
                "Table 2: Market share by region",
                ["Region", "Share_pct"],
                [
                    ["North America", "42"],
                    ["Europe", "31"],
                    ["Asia", "27"],
                ],
            ),
        ),
    ]


def _tables_sections() -> list[Section]:
    """Sections for test_tables.pdf (product specs, comparison, benchmarks)."""
    return [
        (
            "Product Specifications",
            [
                "This document lists product specifications and benchmark comparison "
                "results. The introduction motivates the specification choices.",
            ],
            (
                "Table 1: Product specifications comparison",
                ["Model", "RAM_GB", "Cores", "Price_USD"],
                [
                    ["Base", "8", "4", "499"],
                    ["Pro", "16", "8", "899"],
                    ["Max", "32", "12", "1399"],
                ],
            ),
        ),
        (
            "Benchmarks",
            [
                "The benchmark comparison measures throughput and latency across the "
                "product lineup. Higher throughput indicates better performance.",
            ],
            (
                "Table 2: Benchmark comparison results",
                ["Model", "Throughput_ops", "Latency_ms"],
                [
                    ["Base", "1200", "8"],
                    ["Pro", "2400", "5"],
                    ["Max", "4100", "3"],
                ],
            ),
        ),
    ]


def main() -> None:
    """Generate all three deterministic test PDFs."""
    out_dir: str = _output_dir()
    build_pdf(
        os.path.join(out_dir, "test_academic.pdf"),
        "A Comparison of Optimizers for Text Classification",
        _academic_sections(),
    )
    build_pdf(
        os.path.join(out_dir, "test_report.pdf"),
        "Annual Financial Report",
        _report_sections(),
    )
    build_pdf(
        os.path.join(out_dir, "test_tables.pdf"),
        "Product Specifications and Benchmarks",
        _tables_sections(),
    )
    print(f"All test PDFs written to {out_dir}")


if __name__ == "__main__":
    main()
