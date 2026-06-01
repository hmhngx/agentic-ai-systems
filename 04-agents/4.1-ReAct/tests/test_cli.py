"""Black-box CLI tests via subprocess — integration tests need OPENROUTER_API_KEY."""

import os
import pathlib
import subprocess

import pytest

AGENT_DIR = pathlib.Path(__file__).resolve().parent.parent

# Child prints UTF-8 box-drawing trace; Windows default cp1252 decode yields stdout=None.
_SUBPROCESS_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def test_cli_help_exits_0():
    """--help must succeed and document primary CLI flags."""
    result = subprocess.run(
        ["python", "react_agent.py", "--help"],
        capture_output=True,
        cwd=str(AGENT_DIR),
        check=False,
        **_SUBPROCESS_TEXT,
    )
    assert result.returncode == 0
    for flag in ["--query", "--max-iterations", "--quiet"]:
        assert flag in result.stdout, f"Flag '{flag}' missing from --help"


def test_cli_missing_api_key_exits_1():
    """Missing OPENROUTER_API_KEY must exit 1 before any API call."""
    env = os.environ.copy()
    # Empty string blocks load_dotenv from filling the key (override=False).
    env["OPENROUTER_API_KEY"] = ""
    result = subprocess.run(
        ["python", "react_agent.py", "--query", "test"],
        capture_output=True,
        env=env,
        cwd=str(AGENT_DIR),
        timeout=30,
        check=False,
        **_SUBPROCESS_TEXT,
    )
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "OPENROUTER_API_KEY" in combined or "api" in combined.lower()


def test_cli_max_iterations_capped_at_10():
    """argparse must reject --max-iterations > 10 at CLI parse time."""
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = env.get("OPENROUTER_API_KEY") or "test-key-for-argparse-only"
    result = subprocess.run(
        ["python", "react_agent.py", "--max-iterations", "999", "--query", "test"],
        capture_output=True,
        env=env,
        cwd=str(AGENT_DIR),
        timeout=10,
        check=False,
        **_SUBPROCESS_TEXT,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, (
        "max-iterations=999 must not run — argparse should reject values above 10"
    )
    assert result.returncode == 2 or "cannot exceed" in combined.lower(), (
        f"Expected argparse rejection, got exit {result.returncode}: {combined[:500]}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_cli_default_demo_solves_problem():
    """Default demo MUST solve revenue/headcount in ≤10 iterations using 2+ tools."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set — skipping live demo run")

    result = subprocess.run(
        ["python", "react_agent.py"],
        capture_output=True,
        cwd=str(AGENT_DIR),
        timeout=180,
        check=False,
        **_SUBPROCESS_TEXT,
    )
    out = result.stdout or ""
    assert result.returncode == 0, (
        f"Demo failed.\nSTDOUT:\n{out[-3000:]}\nSTDERR:\n{(result.stderr or '')[-500:]}"
    )
    assert "Thought" in out, "Trace must show 'Thought' header"
    assert "Action" in out, "Trace must show 'Action' header"
    assert "Observation" in out, "Trace must show 'Observation' header"
    assert "file_read" in out, "Demo must use file_read tool"
    assert "calculator" in out, "Demo must use calculator tool"
    assert "FINAL ANSWER" in out, "Demo must reach FINAL ANSWER"
    assert "50000" in out or "50,000" in out, (
        "Demo answer must be $50,000 (1,250,000 / 25). "
        f"Output end:\n{out[-1500:]}"
    )
    assert "HARD STOP" not in out, (
        "Demo should solve in ≤10 iterations, not hard-stop"
    )


@pytest.mark.integration
def test_cli_quiet_mode_suppresses_trace():
    """--quiet must suppress box-drawing trace output."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    result = subprocess.run(
        ["python", "react_agent.py", "--quiet", "--query", "what is 2+2"],
        capture_output=True,
        cwd=str(AGENT_DIR),
        timeout=60,
        check=False,
        **_SUBPROCESS_TEXT,
    )
    assert result.returncode == 0
    assert "┌─ Thought" not in (result.stdout or ""), (
        "--quiet must suppress box-drawing trace"
    )
