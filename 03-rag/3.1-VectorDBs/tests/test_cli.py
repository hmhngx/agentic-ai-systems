"""Black-box CLI tests for qdrant_bench.py via subprocess."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from src.qdrant_ops import get_client

MODULE_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _qdrant_reachable() -> bool:
    """Return True if Qdrant responds on the configured URL."""
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def _run_cli(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run qdrant_bench.py with merged stdout/stderr for reliable capture on Windows."""
    env = kwargs.pop("env", None) or os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        [PYTHON, "qdrant_bench.py", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(MODULE_ROOT),
        **kwargs,
    )


def test_cli_exits_1_if_qdrant_not_running():
    """Verify CLI exits with code 1 when Qdrant is unreachable on a bad port."""
    env = os.environ.copy()
    env["QDRANT_URL"] = "http://localhost:19999"
    result = _run_cli(["--skip-upsert"], env=env)
    combined = (result.stdout or "").lower()
    assert result.returncode == 1, (
        "expected exit code 1 because Qdrant is not running on port 19999"
    )
    assert (
        "not running" in combined
        or "connection" in combined
        or "error" in combined
    ), "expected error message in output because connection should fail"


def test_cli_help_exits_0():
    """Verify --help prints documented flags and exits successfully."""
    result = _run_cli(["--help"])
    out = result.stdout or ""
    assert result.returncode == 0, "expected exit code 0 because --help should succeed"
    assert "--skip-upsert" in out, (
        "expected --skip-upsert in help because the flag is documented"
    )
    assert "--ef-only" in out, (
        "expected --ef-only in help because the flag is documented"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_cli_full_run_exit_0():
    """Verify full CLI pipeline succeeds when Qdrant is running."""
    if not _qdrant_reachable():
        pytest.skip("Qdrant not running — integration test skipped")
    result = _run_cli([], timeout=300)
    out = result.stdout or ""
    assert result.returncode == 0, (
        f"expected exit code 0 because full run should succeed (stderr: {result.stderr})"
    )
    assert "recall@5" in out.lower(), (
        "expected recall@5 in output because benchmark table is printed"
    )
    assert "deliverable checklist" in out.lower(), (
        "expected deliverable checklist because main() prints the checklist"
    )
    for i in range(1, 8):
        assert f"[{i}/7]" in out, (
            f"expected step [{i}/7] in output because pipeline has seven steps"
        )
    assert "✓" in out, "expected checkmarks in deliverable checklist output"
    for topic in [
        "machine_learning",
        "ocean_biology",
        "ancient_history",
        "cooking",
        "urban_architecture",
    ]:
        assert topic in out, (
            f"expected topic {topic} in filtered benchmark table output"
        )


@pytest.mark.integration
def test_cli_skip_upsert_flag_accepted():
    """Verify --skip-upsert and --ef-only complete successfully against live Qdrant."""
    if not _qdrant_reachable():
        pytest.skip("Qdrant not running — integration test skipped")
    result = _run_cli(["--skip-upsert", "--ef-only", "64"], timeout=120)
    assert result.returncode == 0, (
        f"expected exit code 0 because skip-upsert run should succeed (out: {result.stdout})"
    )
