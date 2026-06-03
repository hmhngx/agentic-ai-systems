"""End-to-end CLI tests: run hallucination_detector.py as a real process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["USE_LLM"] = "0"  # force the deterministic offline path
    return subprocess.run(
        [sys.executable, "hallucination_detector.py", *args],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


class TestDemo:
    def test_demo_exits_zero_when_goal_met(self):
        result = _run("--demo")
        assert result.returncode == 0, result.stderr
        assert "Absent-context fallbacks: 5/5" in result.stdout
        assert "Grounded control served:  True" in result.stdout

    def test_demo_json_is_well_formed(self):
        result = _run("--demo", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert len(data) == 6  # 5 absent + 1 control
        absent = data[:5]
        assert all(entry["served"] is False for entry in absent)
        assert data[-1]["scenario"] == "grounded_control"
        assert data[-1]["served"] is True
        for entry in data:
            assert 0.0 <= entry["score"] <= 1.0


class TestAdHoc:
    def test_grounded_answer_exits_zero(self):
        result = _run(
            "--question", "When did Armstrong walk on the Moon?",
            "--answer", "Neil Armstrong walked on the Moon on July 20, 1969 [Doc 1].",
            "--chunk", "Neil Armstrong walked on the Moon on July 20, 1969.",
        )
        assert result.returncode == 0, result.stderr
        assert "SERVED" in result.stdout

    def test_unsupported_answer_exits_nonzero(self):
        result = _run(
            "--question", "What is the powerhouse of the cell?",
            "--answer", "The mitochondria is the powerhouse of the cell [Doc 1].",
            "--chunk", "Apollo 11 launched in 1969.",
        )
        assert result.returncode == 2
        assert "FALLBACK" in result.stdout
        assert "I could not find relevant context." in result.stdout

    def test_json_output_shape(self):
        result = _run(
            "--question", "q", "--answer", "a [Doc 1].",
            "--chunk", "some text", "--json",
        )
        report = json.loads(result.stdout)
        assert {"score", "verdict", "served", "served_answer"} <= report.keys()


class TestArgErrors:
    def test_adhoc_requires_both_question_and_answer(self):
        result = _run("--question", "only a question")
        assert result.returncode == 1
        assert "ad-hoc mode needs both" in result.stderr

    def test_live_requires_both_pdf_and_query(self):
        result = _run("--pdf", "x.pdf")
        assert result.returncode == 1
        assert "live mode needs both" in result.stderr
