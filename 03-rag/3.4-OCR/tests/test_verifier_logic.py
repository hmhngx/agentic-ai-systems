"""Offline tests for verifier output format — smoke checks catch broken retrieval before prod."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.verifier import run_verification


@pytest.fixture(autouse=True)
def legacy_verification_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use fixed sample-PDF queries so tests do not need a live Qdrant scroll."""
    monkeypatch.setenv("VERIFICATION_ADAPTIVE", "false")


@pytest.fixture
def mock_client() -> MagicMock:
    """Qdrant client stand-in — only external I/O is mocked, not verifier logic."""
    return MagicMock()


@pytest.fixture
def verification_result(mock_client: MagicMock) -> dict[str, Any]:
    """Run verification with mocked search/embed calls returning deterministic results."""
    sample_result = {
        "chunk_type": "table",
        "page_num": 2,
        "score": 0.85,
        "heading_path": ["Section 2"],
        "table_title": "Accuracy",
        "text": "[TABLE: Accuracy] | Model | Score |\n| BERT | 0.91 |",
        "source_pdf": "test.pdf",
        "rank": 1,
    }
    with patch("src.verifier.search_by_type", return_value=[sample_result]):
        with patch("src.verifier.search", return_value=[sample_result]):
            with patch(
                "src.verifier.embed_queries",
                return_value=np.zeros((5, 1024), dtype=np.float32),
            ):
                return run_verification(mock_client, ["test.pdf"])


def test_run_verification_returns_correct_keys(verification_result: dict[str, Any]) -> None:
    """Verifies summary dict exposes checklist booleans for automated pipeline gates."""
    result = verification_result
    assert isinstance(result, dict)
    required_keys = {
        "total_queries",
        "queries_with_table_chunks",
        "all_type_filters_correct",
        "headings_in_metadata",
        "table_retrieval_works",
    }
    assert required_keys.issubset(set(result.keys())), (
        f"Missing verification summary keys: {required_keys - set(result.keys())}"
    )


def test_run_verification_total_queries_is_five(verification_result: dict[str, Any]) -> None:
    """Verifies fixed query suite size — fewer queries would miss retrieval failure modes."""
    result = verification_result
    assert result["total_queries"] == 5, (
        f"Must run exactly 5 verification queries, got {result['total_queries']}"
    )


def test_run_verification_returns_bool_fields(verification_result: dict[str, Any]) -> None:
    """Verifies checklist fields are booleans suitable for pass/fail automation."""
    result = verification_result
    for key in ["all_type_filters_correct", "headings_in_metadata", "table_retrieval_works"]:
        assert isinstance(result[key], bool), (
            f"Verification field '{key}' must be bool, got {type(result[key])}"
        )


def test_verification_print_format(
    mock_client: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verifies printed output shows query, chunk_type, score, and heading for human review."""
    sample_result = {
        "chunk_type": "table",
        "page_num": 2,
        "score": 0.85,
        "heading_path": ["Section 2"],
        "table_title": "Accuracy",
        "text": "[TABLE: Accuracy] | Model | Score |\n| BERT | 0.91 |",
        "source_pdf": "test.pdf",
        "rank": 1,
    }
    with patch("src.verifier.search_by_type", return_value=[sample_result]):
        with patch("src.verifier.search", return_value=[sample_result]):
            with patch(
                "src.verifier.embed_queries",
                return_value=np.zeros((5, 1024), dtype=np.float32),
            ):
                run_verification(mock_client, ["test.pdf"])

    out = capsys.readouterr().out
    assert "Query:" in out, "Verification output must show 'Query:' label"
    assert "chunk_type" in out or "table" in out.lower(), (
        "Verification output must show chunk_type per result"
    )
    assert "score" in out.lower() or "Score" in out, (
        "Verification output must show retrieval scores"
    )
    assert "heading" in out.lower() or "Heading:" in out, (
        "Verification output must show heading_path per result"
    )
