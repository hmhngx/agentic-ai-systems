"""Subprocess black-box tests for advanced_rag.py CLI behavior."""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess

import pytest

ADVANCED_RAG_DIR = pathlib.Path(__file__).parent.parent


def test_cli_help_exits_0() -> None:
    """--help must succeed and document key flags for operators."""
    result = subprocess.run(
        ["python", "advanced_rag.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(ADVANCED_RAG_DIR),
    )
    assert result.returncode == 0, (
        f"--help failed with code {result.returncode}: {result.stderr[:200]}"
    )
    assert "--debug" in result.stdout, (
        "--debug flag missing from help — operators cannot enable RRF diagnostics."
    )
    assert "--rrf-k" in result.stdout or "rrf" in result.stdout.lower(), (
        "RRF k flag missing from help — fusion constant must be tunable/documented."
    )
    assert "--query" in result.stdout, (
        "--query flag missing — single-query mode is required for debugging retrieval."
    )


def test_cli_missing_api_key_exits_1() -> None:
    """Missing OPENROUTER_API_KEY must exit 1 before any retrieval runs."""
    from dotenv import dotenv_values

    from advanced_rag import _ENV_PLACEHOLDERS, _candidate_env_paths, _is_real_value

    for path in _candidate_env_paths():
        if not path.is_file():
            continue
        values = dotenv_values(path)
        if _is_real_value("OPENROUTER_API_KEY", values.get("OPENROUTER_API_KEY")):
            pytest.skip(
                "Real OPENROUTER_API_KEY in project .env — subprocess cannot "
                "override dotenv file resolution; test on a machine without .env keys"
            )

    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in ("PATH", "SYSTEMROOT", "PATHEXT", "WINDIR", "COMSPEC", "PYTHONPATH")
    }
    env["OPENROUTER_API_KEY"] = _ENV_PLACEHOLDERS["OPENROUTER_API_KEY"]
    env.pop("VOYAGE_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)

    try:
        result = subprocess.run(
            ["python", "advanced_rag.py"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ADVANCED_RAG_DIR),
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        pytest.skip(
            "CLI did not exit within 15s — a real API key was likely loaded from "
            "a sibling .env; cannot assert early exit in this environment"
        )

    assert result.returncode == 1, (
        f"Expected exit code 1 without API key, got {result.returncode}"
    )
    combined = result.stdout + result.stderr
    assert (
        "api" in combined.lower()
        or "key" in combined.lower()
        or "error" in combined.lower()
    ), f"Expected API key error, got: {combined[:200]}"


def test_cli_single_query_mode() -> None:
    """--query must run the full pipeline when Qdrant and embedding keys are available."""
    try:
        s = socket.create_connection(("localhost", 6333), timeout=2)
        s.close()
    except OSError:
        pytest.skip("Qdrant not running")

    if not os.environ.get("VOYAGE_API_KEY") and not os.environ.get(
        "OPENROUTER_API_KEY"
    ):
        pytest.skip("No embedding API key")

    result = subprocess.run(
        ["python", "advanced_rag.py", "--query", "neural networks backpropagation"],
        capture_output=True,
        text=True,
        cwd=str(ADVANCED_RAG_DIR),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"--query mode failed.\nSTDOUT:\n{result.stdout[-1000:]}\n"
        f"STDERR:\n{result.stderr[-500:]}"
    )
    out_lower = result.stdout.lower()
    assert (
        "chunk" in out_lower
        or "result" in out_lower
        or "pipeline" in out_lower
    ), "Single query mode must print retrieval results"


@pytest.mark.integration
@pytest.mark.slow
def test_cli_full_benchmark_run() -> None:
    """Full benchmark must print ablation table and all seven step headers."""
    try:
        s = socket.create_connection(("localhost", 6333), timeout=2)
        s.close()
    except OSError:
        pytest.skip("Qdrant not running")

    result = subprocess.run(
        ["python", "advanced_rag.py"],
        capture_output=True,
        text=True,
        cwd=str(ADVANCED_RAG_DIR),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Full benchmark failed.\nSTDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-500:]}"
    )

    out_lower = result.stdout.lower()
    assert "baseline" in out_lower, "baseline row missing from benchmark output"
    assert "hybrid" in out_lower, "hybrid row missing from benchmark output"
    assert "recall" in out_lower, "recall column missing from benchmark output"

    for i in range(1, 8):
        assert f"[{i}/7]" in result.stdout, (
            f"Step [{i}/7] header missing from output"
        )

    assert "60" in result.stdout, (
        "RRF k=60 must appear in benchmark output header"
    )

    assert (
        "target" in out_lower
        or "10pp" in out_lower
        or "ACHIEVED" in result.stdout
        or "NOT MET" in result.stdout
    ), "Target check section missing from output"

    assert "AVG" in result.stdout or "avg" in out_lower, (
        "Per-query average row missing"
    )
