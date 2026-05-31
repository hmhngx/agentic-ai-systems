"""Black-box CLI subprocess tests — validates operator-facing pipeline entry points."""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

OCR_DIR = Path(__file__).resolve().parent.parent


def _subprocess_env() -> dict[str, str]:
    """Subprocess environment with UTF-8 stdout on Windows."""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def test_cli_help_exits_0() -> None:
    """Verifies CLI documents all ingestion flags for correct pipeline invocation."""
    result = subprocess.run(
        ["python", "doc_ingest_pipeline.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(OCR_DIR),
        env=_subprocess_env(),
    )
    assert result.returncode == 0
    for flag in ["--pdf", "--pdf-dir", "--reingest", "--verify-only", "--debug", "--enable-ocr"]:
        assert flag in result.stdout, f"Flag '{flag}' missing from --help output"


def test_cli_missing_api_key_exits_1() -> None:
    """Verifies pipeline refuses to run without embeddings — prevents silent garbage index."""
    env = _subprocess_env()
    for key in ["VOYAGE_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]:
        env.pop(key, None)
    result = subprocess.run(
        ["python", "doc_ingest_pipeline.py", "--pdf", "nonexistent.pdf"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(OCR_DIR),
        timeout=30,
    )
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert (
        "api" in combined.lower()
        or "key" in combined.lower()
        or "error" in combined.lower()
    )


def test_cli_missing_pdf_exits_nonzero() -> None:
    """Verifies missing PDF path fails before Qdrant writes corrupt partial state."""
    result = subprocess.run(
        ["python", "doc_ingest_pipeline.py", "--pdf", "/nonexistent/path/file.pdf"],
        capture_output=True,
        text=True,
        cwd=str(OCR_DIR),
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode != 0


def test_generate_test_pdfs_script_runs() -> None:
    """Verifies synthetic PDF generator produces deterministic corpus for verification queries."""
    if importlib.util.find_spec("reportlab") is None:
        pytest.skip("reportlab not installed")

    result = subprocess.run(
        ["python", "scripts/generate_test_pdfs.py"],
        capture_output=True,
        text=True,
        cwd=str(OCR_DIR),
        timeout=60,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"generate_test_pdfs.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    sample_dir = OCR_DIR / "sample_pdfs"
    pdfs = list(sample_dir.glob("*.pdf"))
    assert len(pdfs) >= 3, (
        f"Expected at least 3 generated PDFs, found {len(pdfs)}: {[p.name for p in pdfs]}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_cli_full_pipeline_with_generated_pdfs() -> None:
    """Verifies end-to-end ingest + verification against Qdrant with real embeddings."""
    try:
        sock = socket.create_connection(("localhost", 6333), timeout=2)
        sock.close()
    except OSError:
        pytest.skip("Qdrant not running")

    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    if importlib.util.find_spec("reportlab") is None:
        pytest.skip("reportlab not installed")

    subprocess.run(
        ["python", "scripts/generate_test_pdfs.py"],
        cwd=str(OCR_DIR),
        timeout=60,
        check=True,
    )

    env = _subprocess_env()
    env["COLLECTION_NAME"] = f"test_ocr_e2e_{uuid.uuid4().hex[:8]}"

    try:
        result = subprocess.run(
            [
                "python",
                "doc_ingest_pipeline.py",
                "--pdf-dir",
                "sample_pdfs/",
                "--reingest",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(OCR_DIR),
            timeout=600,
        )
        assert result.returncode == 0, (
            f"Pipeline failed.\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-500:]}"
        )

        assert "Qdrant connected" in result.stdout
        assert "Embedded" in result.stdout or "Stored" in result.stdout

        assert "table" in result.stdout.lower(), "Output must mention table chunks"
        assert "[TABLE:" in result.stdout, (
            "At least one [TABLE: prefix must appear in output — "
            "proves tables stored as structured text"
        )
        assert "heading" in result.stdout.lower() or "Heading:" in result.stdout, (
            "Output must show heading_path — proves headings preserved as metadata"
        )
        assert (
            "DELIVERABLE CHECKLIST" in result.stdout
            or "Deliverable" in result.stdout
        ), "Deliverable checklist must appear in output"
        assert "Query:" in result.stdout, "Verification queries must be printed"

    finally:
        try:
            from qdrant_client import QdrantClient

            url = env.get("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=url)
            client.delete_collection(env["COLLECTION_NAME"])
        except Exception:
            pass
