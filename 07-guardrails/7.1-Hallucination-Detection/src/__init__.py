"""
Hallucination detection guardrail (Day 10 / 07-guardrails 7.1).

A runtime output guardrail that wraps any RAG turn, scores how well the answer
is grounded in the retrieved context (0-1), diagnoses the failure mode, and
refuses with a fallback message below the confidence threshold.

Public surface:
    from src import detect, DetectorConfig, GroundednessReport, Verdict
    report = detect(question, answer, chunks)
    print(report.served_answer, report.score, report.verdict)
"""

from __future__ import annotations

from src.detector import detect
from src.types import (
    DetectorConfig,
    FALLBACK_MESSAGE,
    GroundednessReport,
    RetrievedChunk,
    Verdict,
)

__all__ = [
    "detect",
    "DetectorConfig",
    "GroundednessReport",
    "RetrievedChunk",
    "Verdict",
    "FALLBACK_MESSAGE",
]
