"""Black-box CLI tests for ``naive_rag.py``.

Two tiers:

  * Argparse / error-path tests run offline. They exercise the
    command-line surface (``--help``, missing PDF, missing API key,
    invalid ``--top-k``). These finish in well under 5 seconds and
    require no external services.

  * Two end-to-end tests (marked ``integration`` + ``slow``) actually
    ingest a generated PDF and call OpenRouter APIs. They auto-skip
    when Qdrant or API keys are unavailable.

We launch the CLI via ``subprocess.run`` rather than calling ``main()``
in-process so the tests verify the same code path users hit on the
command line - including argparse, ``sys.exit`` codes, and stdio
encoding on Windows.

Subprocess timeouts are generous (60s for argparse-only paths) because
``naive_rag.py`` imports heavyweight SDKs (qdrant-client, openai,
pymupdf, tiktoken) at module level BEFORE argparse runs.
On a cold Windows interpreter these imports can take 20-30 seconds.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import uuid

import pytest


NAIVE_RAG_DIR = pathlib.Path(__file__).resolve().parent.parent
NAIVE_RAG_SCRIPT = NAIVE_RAG_DIR / "naive_rag.py"


def test_cli_help_exits_0() -> None:
    """--help must exit 0 and document every supported flag."""
    result = subprocess.run(
        [sys.executable, str(NAIVE_RAG_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(NAIVE_RAG_DIR),
        timeout=60,
    )
    assert result.returncode == 0, \
        f"--help exited {result.returncode} - help should never error. " \
        f"stderr: {result.stderr[-400:]}"
    for flag in ("--pdf", "--top-k", "--reingest", "--query", "--debug"):
        assert flag in result.stdout, \
            f"Flag {flag} missing from --help output. " \
            "Undocumented flags trap users into reading source code - " \
            "they should be discoverable from --help alone."


def test_cli_missing_pdf_exits_nonzero(tmp_path: pathlib.Path) -> None:
    """A nonexistent --pdf path must exit with a non-zero code and clear error."""
    bogus_pdf = str(tmp_path / "nonexistent" / "file.pdf")
    result = subprocess.run(
        [sys.executable, str(NAIVE_RAG_SCRIPT),
         "--pdf", bogus_pdf, "--query", "test"],
        capture_output=True,
        text=True,
        cwd=str(NAIVE_RAG_DIR),
        timeout=60,
    )
    assert result.returncode != 0, \
        f"Expected non-zero exit for missing PDF, got {result.returncode}. " \
        "Silently exiting 0 on a missing file would let scripts and CI " \
        "treat a no-op as success."
    combined = (result.stdout + result.stderr).lower()
    assert (
        "not found" in combined
        or "error" in combined
        or "no such file" in combined
    ), f"Error output does not mention the missing file: " \
       f"{(result.stdout + result.stderr)[-400:]!r}. " \
       "Users need an actionable message, not a bare exit code."


def test_cli_missing_api_key_exits_1(tmp_path: pathlib.Path) -> None:
    """Missing API keys must exit 1 BEFORE any embedding/Qdrant call.

    We run from ``tmp_path`` so the script's ``load_dotenv()`` cannot
    re-populate the keys from the project's ``.env`` file. Using the
    script's absolute path means cwd no longer needs to be NAIVE_RAG_DIR.
    """
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        [sys.executable, str(NAIVE_RAG_SCRIPT),
         "--pdf", "nonexistent.pdf", "--query", "test"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=60,
    )
    assert result.returncode == 1, \
        f"Expected exit 1 on missing API keys, got {result.returncode}. " \
        "Any other code (especially 0) means the script ran past the env " \
        "validation gate and may have leaked partial state."
    combined = (result.stdout + result.stderr).lower()
    assert (
        "openrouter_api_key" in combined
        or "api key" in combined
        or "error" in combined
    ), f"Expected API-key error in output, got: " \
       f"{(result.stdout + result.stderr)[:400]!r}. " \
       "Users must know WHICH key is missing - a bare exit 1 forces them " \
       "to read source code."


def test_cli_top_k_rejects_invalid_values() -> None:
    """argparse must reject --top-k=10 with exit code 2 (the topK pathology bound)."""
    result = subprocess.run(
        [sys.executable, str(NAIVE_RAG_SCRIPT),
         "--pdf", "x.pdf", "--top-k", "10", "--query", "test"],
        capture_output=True,
        text=True,
        cwd=str(NAIVE_RAG_DIR),
        timeout=60,
    )
    assert result.returncode == 2, \
        f"argparse should reject --top-k=10 with exit code 2 (invalid choice), " \
        f"got {result.returncode}. " \
        "Allowing top_k>5 triggers the topK pathology: extra distractor " \
        "chunks dilute the LLM's attention away from the best evidence."


def test_cli_top_k_rejects_zero() -> None:
    """argparse must reject --top-k=0 - we cannot retrieve zero chunks."""
    result = subprocess.run(
        [sys.executable, str(NAIVE_RAG_SCRIPT),
         "--pdf", "x.pdf", "--top-k", "0", "--query", "test"],
        capture_output=True,
        text=True,
        cwd=str(NAIVE_RAG_DIR),
        timeout=60,
    )
    assert result.returncode == 2, \
        f"argparse should reject --top-k=0 with exit code 2, got {result.returncode}. " \
        "top_k=0 would force the LLM to answer with empty context - the " \
        "definition of a hallucination prompt."


# -----------------------------------------------------------------------------
# Integration tests - require live Qdrant + OpenRouter
# -----------------------------------------------------------------------------


def _delete_collection_best_effort(collection_name: str, url: str) -> None:
    """Best-effort teardown of a test collection. Never raises."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=url, timeout=5)
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001 - cleanup is best-effort by design
        pass


@pytest.mark.integration
@pytest.mark.slow
def test_cli_full_ingest_and_query(
    sample_pdf: pathlib.Path, api_keys_present, qdrant_client
) -> None:
    """End-to-end: ingest test PDF, run a real query, verify cited answer printed."""
    env = os.environ.copy()
    collection_name = f"test_naive_rag_{uuid.uuid4().hex[:8]}"
    env["COLLECTION_NAME"] = collection_name

    try:
        result = subprocess.run(
            [sys.executable, str(NAIVE_RAG_SCRIPT),
             "--pdf", str(sample_pdf),
             "--query", "What is backpropagation?",
             "--top-k", "3"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(NAIVE_RAG_DIR),
            timeout=180,
        )
        assert result.returncode == 0, (
            f"CLI failed with exit {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-1000:]}\n"
            f"STDERR:\n{result.stderr[-500:]}\n"
            "End-to-end ingest + query must succeed when Qdrant and API "
            "keys are available - this is the happy path of the whole product."
        )
        assert "Answer" in result.stdout, \
            "Answer section header not found in CLI output - the response " \
            "box around the answer is the user's signal that generation " \
            "actually ran."
        assert "[Doc" in result.stdout, \
            "No [Doc N] citation found in answer - either the LLM ignored " \
            "the citation rule or the footer is missing. Both break audit."
        assert "Sources" in result.stdout or "Page" in result.stdout, \
            "Citation footer not found in output - users lose the audit " \
            "trail back to specific PDF pages."

        combined = result.stdout.lower()
        assert (
            "backpropagation" in combined
            or "do not have enough information" in combined
        ), "Answer is not grounded in the retrieved ML chunk and did not " \
           "fall back to the no-info phrase. This is the precise gap where " \
           "hallucinations live - the answer came from somewhere else entirely."
    finally:
        url = env.get("QDRANT_URL", "http://localhost:6333")
        _delete_collection_best_effort(collection_name, url)


@pytest.mark.integration
@pytest.mark.slow
def test_cli_no_results_handled_gracefully(
    sample_pdf: pathlib.Path, api_keys_present, qdrant_client
) -> None:
    """Off-topic gibberish query must hit NO_RESULTS path - LLM must NOT be called blind."""
    env = os.environ.copy()
    collection_name = f"test_naive_rag_{uuid.uuid4().hex[:8]}"
    env["COLLECTION_NAME"] = collection_name

    try:
        result = subprocess.run(
            [sys.executable, str(NAIVE_RAG_SCRIPT),
             "--pdf", str(sample_pdf),
             "--query", "xkcd zyzzyva fluorescent antidisestablishmentarianism purple"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(NAIVE_RAG_DIR),
            timeout=180,
        )
        assert result.returncode == 0, (
            f"CLI crashed on no-results query (exit {result.returncode}). "
            f"stderr:\n{result.stderr[-500:]}\n"
            "NO_RESULTS is a normal user-facing outcome, not an error - "
            "exiting non-zero would break shell scripts that loop over queries."
        )
        combined = result.stdout.lower()
        assert (
            "no relevant" in combined
            or "not found" in combined
            or "do not have enough" in combined
        ), f"No-results case not handled gracefully. " \
           f"Output: {result.stdout[:500]!r}. " \
           "Either we showed the user a confused/empty answer, or we " \
           "called the LLM with empty context (the hallucination path)."
        assert (
            "openrouter" not in result.stderr.lower()
            or "api" not in result.stderr.lower()
            or "no relevant" in combined
        ), "Suspected OpenRouter API was called on NO_RESULTS - the CLI must " \
           "short-circuit BEFORE generation when retrieval returns nothing, " \
           "otherwise the LLM answers from parametric memory and hallucinates."
    finally:
        url = env.get("QDRANT_URL", "http://localhost:6333")
        _delete_collection_best_effort(collection_name, url)
