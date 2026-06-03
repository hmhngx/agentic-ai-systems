"""
rag_adapter.py - Bridge the detector onto an existing RAG ("wrap any RAG").

The detector core knows nothing about Qdrant, OpenRouter, or the naive-RAG
package. This adapter is the only place that coupling lives, kept isolated so
the core stays reusable and the offline test path never imports a vector store.

Two layers:
  - ``chunks_from_naive_rag`` - pure: map naive-RAG result dicts to
    ``RetrievedChunk`` (rank -> "Doc N"). Unit-testable without any RAG running.
  - ``run_naive_rag_and_detect`` - live: lazily import the Day-4 naive RAG,
    optionally ingest a PDF, run retrieval + generation, then score the result.
    This is the concrete "add hallucination detection to rag_v1_complete"
    integration. It requires Qdrant + an OpenRouter key (and, for ingestion,
    the naive-RAG project's own requirements) and is only reached from the CLI
    ``--pdf`` mode.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.detector import detect
from src.types import DetectorConfig, GroundednessReport, RetrievedChunk


@dataclass(frozen=True)
class _NaiveRag:
    """The slice of the naive-RAG API the bridge needs, after isolated import."""

    get_client: Callable
    retrieve: Callable
    generate_answer: Callable
    collection_exists: Callable
    create_collection: Callable
    upsert_chunks: Callable
    embed_documents: Callable
    embedding_dim: int
    load_and_chunk: Callable | None  # only imported when ingestion is needed


# Path to the Day-4 naive-RAG package, relative to this file:
#   07-guardrails/7.1-Hallucination-Detection/src/rag_adapter.py
#   -> repo_root/03-rag/3.2-NaiveRAG
_NAIVE_RAG_ROOT = (
    Path(__file__).resolve().parents[3] / "03-rag" / "3.2-NaiveRAG"
)


def _fast_tokenize_enabled() -> bool:
    """True when FAST_TOKENIZE asks the loader to skip tiktoken's download."""
    return os.environ.get("FAST_TOKENIZE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def chunks_from_naive_rag(results: list[dict]) -> list[RetrievedChunk]:
    """Map naive-RAG retrieval result dicts to detector ``RetrievedChunk``s.

    Each naive-RAG result carries ``rank`` (1-based), ``text`` and ``score``.
    The generator is instructed to cite ``[Doc N]`` where N is the rank, so the
    chunk's citable label is ``f"Doc {rank}"`` - exactly what Check 2 validates
    against.
    """
    return [
        RetrievedChunk(
            chunk_id=f"Doc {result['rank']}",
            text=result["text"],
            score=float(result["score"]) if result.get("score") is not None else None,
        )
        for result in results
    ]


def _import_naive_rag(need_ingest: bool = False) -> _NaiveRag:
    """Lazily import the naive-RAG functions the bridge needs.

    The naive-RAG project also has a top-level package named ``src`` (and its
    modules import each other as ``from src.embedder import ...``). Because THIS
    project's ``src`` is already in ``sys.modules`` by the time we get here, a
    plain ``from src.generator import ...`` would resolve against our package
    and fail with "No module named 'src.generator'".

    To load the sibling package cleanly we open an isolated window: stash and
    remove our own ``src*`` modules, put the naive-RAG root first on the path so
    ``src`` resolves to *its* package, import what we need (its internal
    cross-imports now resolve consistently), then restore our modules. The
    imported function objects keep references to the naive-RAG module globals,
    so they remain valid after the restore.

    Raises a clear ``RuntimeError`` (not a raw traceback) on any failure.
    """
    if not _NAIVE_RAG_ROOT.is_dir():
        raise RuntimeError(
            f"naive-RAG package not found at {_NAIVE_RAG_ROOT}. The live --pdf "
            "mode requires the Day-4 project (03-rag/3.2-NaiveRAG)."
        )

    import importlib

    root = str(_NAIVE_RAG_ROOT)
    stashed = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for name in stashed:
        del sys.modules[name]
    sys.path.insert(0, root)
    try:
        retriever = importlib.import_module("src.retriever")
        generator = importlib.import_module("src.generator")
        vector_store = importlib.import_module("src.vector_store")
        embedder = importlib.import_module("src.embedder")
        # pdf_loader pulls the heavy PDF-parsing deps, so only import it when a
        # caller actually needs to ingest.
        load_and_chunk = None
        if need_ingest:
            pdf_loader = importlib.import_module("src.pdf_loader")
            if _fast_tokenize_enabled():
                # The naive loader counts tokens with tiktoken, whose
                # cl100k_base BPE vocab downloads from the network on first
                # use - that download can hang for a long time on a slow or
                # restricted connection, stalling chunking before embedding
                # even starts. Pre-seed the loader's cache so it skips
                # tiktoken and uses its len//4 heuristic instead. Token counts
                # only shift chunk boundaries slightly; they never affect
                # answer correctness or the groundedness score.
                setattr(pdf_loader, "_tiktoken_attempted", True)
                setattr(pdf_loader, "_tiktoken_encoder", None)
            load_and_chunk = pdf_loader.load_and_chunk
        return _NaiveRag(
            get_client=vector_store.get_client,
            retrieve=retriever.retrieve,
            generate_answer=generator.generate_answer,
            collection_exists=vector_store.collection_exists,
            create_collection=vector_store.create_collection,
            upsert_chunks=vector_store.upsert_chunks,
            embed_documents=embedder.embed_documents,
            embedding_dim=embedder.EMBEDDING_DIM,
            load_and_chunk=load_and_chunk,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"could not import naive-RAG modules from {_NAIVE_RAG_ROOT}: {exc}. "
            "Install its requirements.txt into the active environment."
        ) from exc
    finally:
        # Drop any naive-RAG ``src*`` modules and restore ours, so the rest of
        # this process (including our own detect()) sees the right package.
        try:
            sys.path.remove(root)
        except ValueError:
            pass
        for name in [
            n for n in sys.modules if n == "src" or n.startswith("src.")
        ]:
            del sys.modules[name]
        sys.modules.update(stashed)


def _ingest_pdf(rag: _NaiveRag, client, pdf_path: str, reingest: bool) -> None:
    """Ingest ``pdf_path`` into Qdrant, mirroring naive_rag._ingest_pdf.

    Skips ingestion when a populated collection already exists (unless
    ``reingest``). Raises ``RuntimeError`` (instead of naive_rag's ``sys.exit``)
    so the CLI can report the problem without killing the interpreter abruptly.
    """
    if rag.collection_exists(client) and not reingest:
        return
    if rag.load_and_chunk is None:  # pragma: no cover - guarded by need_ingest
        raise RuntimeError("ingestion requested but pdf_loader was not imported")

    chunks, _stats = rag.load_and_chunk(pdf_path)
    if not chunks:
        raise RuntimeError(
            f"PDF produced zero chunks after filtering: {pdf_path}. "
            "The document may be too short or text extraction failed."
        )
    # Progress prints so the (network-bound) embedding step is not a silent wait.
    print(f"Embedding {len(chunks)} chunks via OpenRouter...", flush=True)
    embeddings = rag.embed_documents([c["text"] for c in chunks])
    print(f"Upserting {len(chunks)} vectors into Qdrant...", flush=True)
    rag.create_collection(client, rag.embedding_dim)
    rag.upsert_chunks(client, chunks, embeddings)
    print("Ingestion complete.", flush=True)


def run_naive_rag_and_detect(
    question: str,
    *,
    pdf_path: str | None = None,
    top_k: int = 5,
    reingest: bool = False,
    config: DetectorConfig | None = None,
) -> GroundednessReport:
    """End-to-end: (optionally ingest a PDF), retrieve, generate, then score.

    This is the live "add hallucination detection to rag_v1_complete" path. It
    runs the real Day-4 pipeline and pipes its answer + retrieved chunks through
    the guardrail.

    When retrieval yields nothing, we score with an empty chunk list and an
    empty answer - the detector returns ``NO_RETRIEVAL`` and never calls the
    generator, which is the correct (and cheapest) behaviour for absent context.

    Network/Qdrant failures are surfaced as ``RuntimeError`` so the CLI prints
    an actionable message rather than a raw traceback.
    """
    config = config or DetectorConfig.from_env()

    if pdf_path is not None and not os.path.isfile(pdf_path):
        raise RuntimeError(f"PDF file not found: {pdf_path}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; the live RAG needs it to embed and "
            "generate. Add it to .env or your shell."
        )

    rag = _import_naive_rag(need_ingest=pdf_path is not None)

    try:
        client = rag.get_client()
        if pdf_path is not None:
            _ingest_pdf(rag, client, pdf_path, reingest)
        results, status = rag.retrieve(client, question, top_k=top_k, quiet=True)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any backend failure cleanly
        raise RuntimeError(
            f"naive-RAG live run failed during ingest/retrieval "
            f"({type(exc).__name__}: {exc}). Is Qdrant running and reachable "
            "at QDRANT_URL, and is OPENROUTER_API_KEY valid?"
        ) from exc

    if status == "NO_RESULTS" or not results:
        return detect(question, "", [], config=config)

    answer, _warnings = rag.generate_answer(question, results)
    chunks = chunks_from_naive_rag(results)
    return detect(question, answer, chunks, config=config)
