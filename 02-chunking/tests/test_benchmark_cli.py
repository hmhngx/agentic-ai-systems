"""Black-box CLI tests for benchmark.py (no src imports)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CHUNKING_ROOT = Path(__file__).resolve().parent.parent


def _run_benchmark(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "benchmark.py", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(CHUNKING_ROOT),
    )


def _stderr_is_clean(stderr: str) -> bool:
    """True when stderr has no hard failures (tracebacks or ERROR lines)."""
    upper = stderr.upper()
    return "TRACEBACK" not in upper and "ERROR" not in upper


@pytest.fixture(scope="module")
def cli_success_run(session_pdf: Path) -> subprocess.CompletedProcess[str]:
    """Single full benchmark subprocess reused by heavy CLI assertions."""
    proc = _run_benchmark("--pdf_path", str(session_pdf))
    assert proc.returncode == 0, f"benchmark setup failed: {proc.stderr}"
    return proc


def test_cli_runs_with_valid_pdf(cli_success_run):
    """benchmark.py exits 0 and prints strategy table for a valid PDF."""
    proc = cli_success_run
    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}: {proc.stderr}"
    assert "Strategy" in proc.stdout, "expected Strategy column in stdout table"
    assert "ICC" in proc.stdout, "expected ICC column in stdout"
    assert "Best ICC:" in proc.stdout, "expected best-strategy summary line"
    assert _stderr_is_clean(proc.stderr), (
        f"expected no ERROR or Traceback in stderr, got: {proc.stderr!r}"
    )


def test_cli_missing_pdf_exits_1():
    """Missing PDF path exits 1 with not-found style message."""
    proc = _run_benchmark("--pdf_path", "/nonexistent/file.pdf")
    assert proc.returncode == 1, "expected exit code 1 for missing PDF"
    combined = proc.stdout + proc.stderr
    assert "not found" in combined.lower() or "filenotfounderror" in combined.lower(), (
        "expected not-found message in stdout or stderr"
    )


def test_cli_missing_argument():
    """benchmark.py without --pdf_path exits 2 (argparse usage error)."""
    proc = _run_benchmark()
    assert proc.returncode == 2, "expected argparse exit code 2 when --pdf_path omitted"


def test_cli_output_has_all_four_strategies(cli_success_run):
    """CLI stdout mentions naive, recursive, sentence, and semantic strategies."""
    lower = cli_success_run.stdout.lower()
    assert "naive" in lower, "expected naive strategy in output"
    assert "recursive" in lower, "expected recursive strategy in output"
    assert "sentence" in lower, "expected sentence strategy in output"
    assert "semantic" in lower, "expected semantic strategy in output"


def test_cli_output_is_valid_markdown_table(cli_success_run):
    """CLI prints a Markdown pipe table with header, separator, and data rows."""
    lines = [line for line in cli_success_run.stdout.split("\n") if "|" in line]
    assert len(lines) >= 5, "expected header, separator, and four strategy rows"
    assert lines[0].startswith("|"), "expected Markdown table row to start with pipe"
