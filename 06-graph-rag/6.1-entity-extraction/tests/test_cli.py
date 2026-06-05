import os
import subprocess
import sys
from pathlib import Path


def _clean_env():
    e = dict(os.environ)
    e["USE_LLM"] = "0"
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def test_cli_runs_end_to_end_offline(tmp_path: Path):
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Nelson Liu and Percy Liang work at Stanford University. "
        "Claude was evaluated on NaturalQuestions.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "entity_extractor.py"),
            "--doc",
            str(doc),
            "--out-dir",
            str(out_dir),
            "--no-qdrant",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_clean_env(),
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "graph.json").exists()
    assert (out_dir / "graph.graphml").exists()
    assert (out_dir / "graph.png").exists()
    assert "most-connected" in proc.stdout
