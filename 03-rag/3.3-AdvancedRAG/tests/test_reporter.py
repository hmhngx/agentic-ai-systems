"""Offline tests for benchmark reporter stdout formatting."""

from __future__ import annotations

from src.reporter import print_recall_table, print_target_check

SAMPLE_RESULTS = [
    {
        "pipeline": "baseline (dense top-5)",
        "per_query_recall": [0.8, 0.6, 1.0, 0.8, 0.6] * 4,
        "mean_recall": 0.76,
        "per_query_latency_ms": [5.0] * 20,
        "p50_latency_ms": 5.0,
        "p95_latency_ms": 6.0,
    },
    {
        "pipeline": "hybrid (BM25+dense+RRF)",
        "per_query_recall": [1.0, 0.8, 1.0, 1.0, 0.8] * 4,
        "mean_recall": 0.92,
        "per_query_latency_ms": [8.0] * 20,
        "p50_latency_ms": 8.0,
        "p95_latency_ms": 10.0,
    },
    {
        "pipeline": "reranked (hybrid+Cohere)",
        "per_query_recall": [1.0, 1.0, 1.0, 1.0, 0.8] * 4,
        "mean_recall": 0.96,
        "per_query_latency_ms": [120.0] * 20,
        "p50_latency_ms": 120.0,
        "p95_latency_ms": 150.0,
    },
]


def test_print_recall_table_runs_without_error() -> None:
    """Recall table must print without exception — CLI depends on this path."""
    print_recall_table(SAMPLE_RESULTS)


def test_print_recall_table_output_contains_pipelines(capsys) -> None:
    """Table must name all three pipelines so operators can read the ablation."""
    print_recall_table(SAMPLE_RESULTS)
    out = capsys.readouterr().out
    assert "baseline" in out.lower(), (
        "baseline pipeline name missing — cannot compare dense-only anchor."
    )
    assert "hybrid" in out.lower(), (
        "hybrid pipeline name missing — cannot verify RRF lift."
    )
    assert "rerank" in out.lower(), (
        "reranked pipeline name missing — cannot verify cross-encoder stage."
    )


def test_print_recall_table_output_contains_recall_values(capsys) -> None:
    """Mean recall values must appear so results are human-verifiable from logs."""
    print_recall_table(SAMPLE_RESULTS)
    out = capsys.readouterr().out
    assert "0.76" in out or "0.760" in out, "baseline recall not in output"
    assert "0.92" in out or "0.920" in out, "hybrid recall not in output"
    assert "0.96" in out or "0.960" in out, "reranked recall not in output"


def test_print_target_check_achieved(capsys) -> None:
    """+15pp improvement must print ACHIEVED when target gap is met."""
    print_target_check(baseline_recall=0.70, reranked_recall=0.85)
    out = capsys.readouterr().out
    assert "ACHIEVED" in out or "achieved" in out.lower(), (
        "0.85 - 0.70 = 0.15 >= 0.10: target ACHIEVED must appear in output"
    )


def test_print_target_check_not_met(capsys) -> None:
    """Small lift must print NOT MET or diagnostics — avoids false success claims."""
    print_target_check(baseline_recall=0.94, reranked_recall=0.96)
    out = capsys.readouterr().out
    assert (
        "NOT MET" in out
        or "not met" in out.lower()
        or "diagnostic" in out.lower()
        or "500" in out
    ), (
        "0.96 - 0.94 = 0.02 < 0.10: NOT MET or diagnostic guidance must appear"
    )


def test_print_target_check_improvement_calculation(capsys) -> None:
    """Improvement banner must show the percentage-point delta."""
    print_target_check(baseline_recall=0.80, reranked_recall=0.95)
    out = capsys.readouterr().out
    assert "15" in out or "0.15" in out, (
        "15pp improvement must appear in output"
    )


def test_print_target_check_uses_hybrid_when_dense_ceiling(capsys) -> None:
    """When baseline >= 0.95, +10pp target is measured against hybrid recall."""
    print_target_check(
        baseline_recall=1.0,
        reranked_recall=1.0,
        hybrid_recall=0.64,
    )
    out = capsys.readouterr().out
    assert "ACHIEVED" in out
    assert "Hybrid recall@5" in out
    assert "Dense baseline >= 0.95" in out
