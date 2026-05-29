"""advanced_rag.py - Day 5 deliverable: Advanced RAG ablation benchmark.

No third-party RAG orchestration frameworks. Raw rank-bm25, Cohere, and Qdrant.

What this script proves:
    1. BM25 catches exact-match queries that dense embeddings miss
    2. RRF fuses incompatible score distributions using rank position only
    3. Cross-encoder reranking improves precision on the candidate pool
    4. Each addition is measured independently - diagnosis-driven architecture

What this script deliberately does NOT implement:
    - HyDE - requires LLM generation BEFORE retrieval (Day 5 concept)
    - Multi-query - multiple LLM calls before retrieval (Day 5 concept)
    - These are advanced techniques reserved for measured failure modes

Usage:
    python advanced_rag.py                       # full benchmark
    python advanced_rag.py --debug               # show RRF analysis per query
    python advanced_rag.py --skip-ground-truth   # skip exact search (use cached)
    python advanced_rag.py --query "your query"  # single query through full pipeline
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

# Make ``src/...`` importable when the script is launched from any working
# directory. The same pattern is used in 03-rag/3.2-NaiveRAG/naive_rag.py.
_MODULE_ROOT: Path = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))


from qdrant_client import QdrantClient  # noqa: E402

from src.bm25_retriever import bm25_search, build_bm25_index  # noqa: E402
from src.corpus_bridge import get_corpus_and_queries  # noqa: E402
from src.dense_retriever import (  # noqa: E402
    BASELINE_HNSW_EF,
    CANDIDATE_HNSW_EF,
    COLLECTION_NAME,
    dense_search,
    ensure_chunk_id_payload,
    get_qdrant_client,
)
from src.ground_truth import compute_ground_truth, recall_at_5  # noqa: E402
from src.recall_benchmark import (  # noqa: E402
    run_baseline,
    run_hybrid,
    run_reranked,
)
from src.reporter import (  # noqa: E402
    print_per_query_detail,
    print_recall_table,
    print_rrf_analysis,
    print_target_check,
)
from src.reranker import (  # noqa: E402
    CANDIDATE_POOL,
    RERANK_PRECISION_SLOTS,
    RERANK_TOP_N,
    build_rerank_input_with_dense_anchor,
    finalize_with_dense_backfill,
    rerank,
)
from src.rrf_fusion import RRF_K, fuse_rrf, get_top_k_chunk_ids  # noqa: E402


# UTF-8 stdout/stderr on Windows so the table separators render correctly
# even in non-UTF-8 code pages. ``reconfigure`` is a no-op when the stream
# is already UTF-8; safe to call unconditionally.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def _parse_args() -> argparse.Namespace:
    """Define the CLI surface.

    Range bounds on --top-n-bm25 and --top-n-dense are part of the
    spec's hard constraints (candidate pool must be 20..100). --rrf-k
    accepts any positive int but the help text warns against changing
    away from 60.
    """
    parser = argparse.ArgumentParser(
        prog="advanced_rag",
        description=(
            "Day 5 advanced RAG ablation: baseline vs hybrid (BM25+dense+RRF) "
            "vs reranked (Cohere cross-encoder). Raw SDKs only, no orchestration frameworks."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show RRF analysis for the first query after the main table.",
    )
    parser.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help=(
            "Skip recomputing the exact-search ground truth. Useful when "
            "iterating on the pipeline within a single Python session - the "
            "ground truth is cached in memory once computed."
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=(
            "Run a single query through all 3 pipelines and print a side-by-"
            "side comparison instead of the full 20-query benchmark."
        ),
    )
    parser.add_argument(
        "--top-n-bm25",
        type=int,
        default=50,
        help=(
            "BM25 candidate pool size. Must be in [20, 100] (spec constraint - "
            "outside this range the fusion either underfeeds RRF or pays for "
            "tokens it cannot use). Default: 50."
        ),
    )
    parser.add_argument(
        "--top-n-dense",
        type=int,
        default=50,
        help=(
            "Dense candidate pool size. Must be in [20, 100]. Default: 50."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help=(
            "RRF smoothing constant. Default 60 per Cormack et al. 2009 - "
            "changing this away from 60 is almost always a bug; see "
            "src/rrf_fusion.py for the empirical justification."
        ),
    )
    parser.add_argument(
        "--baseline-hnsw-ef",
        type=int,
        default=BASELINE_HNSW_EF,
        help=(
            "HNSW ef_search for the baseline (dense-only) pipeline. "
            f"Default {BASELINE_HNSW_EF} simulates a fast production ANN; "
            f"hybrid/rerank pools use ef={CANDIDATE_HNSW_EF}. Lower values "
            "create recall headroom for the +10pp target."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Enforce the spec's range constraints on candidate-pool flags."""
    if not 20 <= args.top_n_bm25 <= 100:
        print(
            f"ERROR: --top-n-bm25 must be in [20, 100] (got {args.top_n_bm25}).",
            file=sys.stderr,
        )
        sys.exit(2)
    if not 20 <= args.top_n_dense <= 100:
        print(
            f"ERROR: --top-n-dense must be in [20, 100] (got {args.top_n_dense}).",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.rrf_k <= 0:
        print(
            f"ERROR: --rrf-k must be positive (got {args.rrf_k}).",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.baseline_hnsw_ef <= 0:
        print(
            f"ERROR: --baseline-hnsw-ef must be positive "
            f"(got {args.baseline_hnsw_ef}).",
            file=sys.stderr,
        )
        sys.exit(2)


def _print_step(idx: int, total: int, label: str) -> None:
    """Single source of truth for the [N/M] step banner format."""
    print(f"\n[{idx}/{total}] {label}")


# Placeholder values shipped in .env.example. If a key happens to equal
# one of these strings (because the user copied .env.example to .env and
# never replaced the value), we treat it as "missing" so the loader can
# fall through to the next .env in the chain instead of pretending the
# key is set and then failing with a confusing 401 at API call time.
_ENV_PLACEHOLDERS: dict[str, str] = {
    "OPENROUTER_API_KEY": "your-openrouter-api-key-here",
    "COHERE_API_KEY": "your-cohere-api-key-here",
    "VOYAGE_API_KEY": "your-voyage-key-here",
}


def _is_real_value(key: str, value: str | None) -> bool:
    """True iff ``value`` is non-empty and not the .env.example placeholder."""
    if value is None:
        return False
    if not value.strip():
        return False
    if value == _ENV_PLACEHOLDERS.get(key):
        return False
    return True


def _candidate_env_paths() -> list[Path]:
    """Priority-ordered list of .env locations to inspect.

    Priority rationale (first match wins for each key):
        1. ``03-rag/3.3-AdvancedRAG/.env`` - module-local overrides.
        2. ``03-rag/3.2-NaiveRAG/.env``    - shared key store across the
           RAG days; Day 4 typically already has OPENROUTER_API_KEY and
           COHERE_API_KEY filled in, so Day 5 can free-ride.
        3. ``<repo-root>/.env``            - workspace-wide fallback.
    """
    return [
        _MODULE_ROOT / ".env",
        _MODULE_ROOT.parent / "3.2-NaiveRAG" / ".env",
        _MODULE_ROOT.parent.parent / ".env",
    ]


def _load_env_and_validate_keys() -> None:
    """Stage [1/7]. Walk through candidate .env files and require the key set.

    OPENROUTER_API_KEY is the embedder credential - without it dense
    retrieval cannot run at all, so we exit 1.
    COHERE_API_KEY is optional - the reranker degrades gracefully when
    it is absent, so we only warn.

    Why a multi-file walk instead of ``load_dotenv()``?
        The bare ``load_dotenv()`` picks the closest .env (typically
        ``03-rag/3.3-AdvancedRAG/.env``) and stops. If that file holds
        the .env.example placeholder string, ``os.environ.get`` returns
        a truthy value but every API call dies with 401. This loader
        treats placeholders as missing and falls through to siblings
        like Day 4's .env where the real key usually lives.
    """
    # Step 1: load every existing .env so plain-old non-secret variables
    # (QDRANT_URL, COLLECTION_NAME, OPENROUTER_EMBEDDING_MODEL, ...)
    # populate os.environ. override=False keeps shell-set vars winning.
    for path in _candidate_env_paths():
        if path.is_file():
            load_dotenv(dotenv_path=path, override=False)

    # Step 2: for the API-key set we cannot trust step 1 because a
    # placeholder value blocks the fallthrough. Re-resolve each key by
    # explicitly parsing each .env via dotenv_values (which never
    # touches os.environ) and picking the first non-placeholder hit.
    sources: dict[str, Path] = {}
    for key in _ENV_PLACEHOLDERS:
        # Honour a real shell-env value when present.
        if _is_real_value(key, os.environ.get(key)):
            sources[key] = Path("<shell env>")
            continue
        for path in _candidate_env_paths():
            if not path.is_file():
                continue
            values: dict[str, str | None] = dotenv_values(path)
            candidate: str | None = values.get(key)
            if _is_real_value(key, candidate):
                # ``candidate`` is guaranteed str (not None) because
                # _is_real_value rejects None - the cast satisfies type
                # checkers without a redundant runtime branch.
                os.environ[key] = str(candidate)
                sources[key] = path
                break
        else:
            # No real value found anywhere in the chain. Drop a stale
            # placeholder so downstream checks see the key as missing.
            if os.environ.get(key) == _ENV_PLACEHOLDERS[key]:
                os.environ.pop(key, None)

    if not _is_real_value("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY")):
        print(
            "ERROR: OPENROUTER_API_KEY is not set in any of:\n"
            + "\n".join(f"  - {p}" for p in _candidate_env_paths())
            + "\nAdd a real key to one of these .env files or export it "
            "in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    def _origin(key: str) -> str:
        path: Path | None = sources.get(key)
        if path is None:
            return "(unset)"
        return str(path)

    print(f"  OPENROUTER_API_KEY: set    (from {_origin('OPENROUTER_API_KEY')})")
    cohere_origin: str = (
        _origin("COHERE_API_KEY")
        if _is_real_value("COHERE_API_KEY", os.environ.get("COHERE_API_KEY"))
        else "missing (reranker will be skipped)"
    )
    print(f"  COHERE_API_KEY:     {cohere_origin}")


def _print_single_query_comparison(
    client: QdrantClient,
    corpus: list[dict],
    bm25_index: Any,
    chunk_ids: list[str],
    query_text: str,
    top_n_dense: int,
    top_n_bm25: int,
    rrf_k: int,
    baseline_hnsw_ef: int = BASELINE_HNSW_EF,
    candidate_hnsw_ef: int = CANDIDATE_HNSW_EF,
) -> None:
    """Run a single ad-hoc query through all three pipelines and compare.

    No ground truth is computed in this branch (--query mode is for
    qualitative inspection, not metric reporting). The RRF analysis is
    always printed because it is the single most useful diagnostic when
    investigating "why did pipeline X return Y for query Z?".
    """
    print(f"\nRunning single query: {query_text!r}")
    print(
        f"  Baseline ef={baseline_hnsw_ef}, pool ef={candidate_hnsw_ef}, "
        f"dense top-{top_n_dense}, BM25 top-{top_n_bm25}, RRF k={rrf_k}"
    )

    dense_hits: list[dict] = dense_search(
        client, query_text, top_n=top_n_dense, hnsw_ef=candidate_hnsw_ef
    )
    bm25_hits: list[dict] = bm25_search(
        bm25_index, chunk_ids, query_text, top_n=top_n_bm25
    )
    fused: list[dict] = fuse_rrf(dense_hits, bm25_hits, k=rrf_k)
    baseline_ids: list[str] = [
        r["chunk_id"]
        for r in dense_search(
            client, query_text, top_n=5, hnsw_ef=baseline_hnsw_ef
        )
    ]
    hybrid_ids: list[str] = get_top_k_chunk_ids(fused, 5)

    candidates: list[tuple[str, str]] = build_rerank_input_with_dense_anchor(
        fused, dense_hits, corpus, top_candidate=CANDIDATE_POOL
    )
    cohere_top_n: int = max(3, min(RERANK_TOP_N, RERANK_PRECISION_SLOTS))
    reranked: list[dict] = rerank(query_text, candidates, top_n=cohere_top_n)
    if reranked:
        reranked_ids: list[str] = finalize_with_dense_backfill(
            [r["chunk_id"] for r in reranked[:RERANK_PRECISION_SLOTS]],
            dense_hits,
            k=RERANK_TOP_N,
            precision_slots=RERANK_PRECISION_SLOTS,
        )
    else:
        reranked_ids = finalize_with_dense_backfill(
            get_top_k_chunk_ids(fused, cohere_top_n)[:RERANK_PRECISION_SLOTS],
            dense_hits,
            k=RERANK_TOP_N,
            precision_slots=RERANK_PRECISION_SLOTS,
        )

    text_by_id: dict[str, str] = {chunk["chunk_id"]: chunk["text"] for chunk in corpus}

    def _preview(chunk_id: str) -> str:
        text: str = text_by_id.get(chunk_id, "(missing)")
        return text[:80].replace("\n", " ")

    print("\n  BASELINE (dense top-5):")
    for rank, chunk_id in enumerate(baseline_ids, start=1):
        print(f"    {rank}. chunk_id={chunk_id} | {_preview(chunk_id)}...")

    print("\n  HYBRID (RRF top-5):")
    for rank, chunk_id in enumerate(hybrid_ids, start=1):
        print(f"    {rank}. chunk_id={chunk_id} | {_preview(chunk_id)}...")

    print("\n  RERANKED (Cohere top-5):")
    if not reranked:
        print("    (Cohere unavailable - using RRF top-5 as fallback)")
    for rank, chunk_id in enumerate(reranked_ids, start=1):
        print(f"    {rank}. chunk_id={chunk_id} | {_preview(chunk_id)}...")

    print()
    print_rrf_analysis(fused, query_text)


def main() -> None:
    """Orchestrate the seven stages or short-circuit to --query mode."""
    args: argparse.Namespace = _parse_args()
    _validate_args(args)

    _print_step(1, 7, "Loading environment...")
    _load_env_and_validate_keys()

    _print_step(2, 7, "Loading corpus and queries...")
    corpus, queries = get_corpus_and_queries()

    _print_step(3, 7, "Connecting to Qdrant and verifying collection...")
    client: QdrantClient = get_qdrant_client()
    print(f"  Qdrant connected at {os.environ.get('QDRANT_URL', 'http://localhost:6333')}.")
    print("  Ensuring collection exists with chunk_id payloads...")
    ensure_chunk_id_payload(client, corpus)
    count: int = int(client.count(collection_name=COLLECTION_NAME, exact=True).count)
    print(f"  Collection '{COLLECTION_NAME}': {count} vectors ready.")

    _print_step(4, 7, "Building BM25 index from corpus...")
    bm25_index, chunk_ids = build_bm25_index(corpus)

    if args.query is not None:
        # --query mode: skip ground truth + full benchmark; run the
        # single-query side-by-side comparison and exit.
        _print_single_query_comparison(
            client=client,
            corpus=corpus,
            bm25_index=bm25_index,
            chunk_ids=chunk_ids,
            query_text=args.query,
            top_n_dense=args.top_n_dense,
            top_n_bm25=args.top_n_bm25,
            rrf_k=args.rrf_k,
            baseline_hnsw_ef=args.baseline_hnsw_ef,
        )
        return

    _print_step(5, 7, "Computing ground truth (exact Qdrant search)...")
    if args.skip_ground_truth:
        print("  --skip-ground-truth: reusing in-memory cache from a prior call.")
    ground_truth: dict[str, list[str]] = compute_ground_truth(client, queries)
    # Sanity: every query must have produced a ground-truth list.
    missing_gt: list[str] = [
        q["query_text"] for q in queries if q["query_text"] not in ground_truth
    ]
    if missing_gt:
        print(
            f"ERROR: ground truth missing for {len(missing_gt)} queries. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    _print_step(6, 7, "Running 3-pipeline ablation benchmark...")
    print("  Running baseline...")
    baseline_result: dict = run_baseline(
        client, queries, ground_truth, baseline_hnsw_ef=args.baseline_hnsw_ef
    )
    print(
        f"    baseline mean recall@5 = {baseline_result['mean_recall']:.4f}, "
        f"p50 lat = {baseline_result['p50_latency_ms']:.2f}ms"
    )

    print("  Running hybrid (BM25+dense+RRF)...")
    hybrid_result: dict = run_hybrid(
        client=client,
        bm25_index=bm25_index,
        chunk_ids=chunk_ids,
        queries=queries,
        ground_truth=ground_truth,
        top_n_dense=args.top_n_dense,
        top_n_bm25=args.top_n_bm25,
        rrf_k=args.rrf_k,
    )
    print(
        f"    hybrid mean recall@5 = {hybrid_result['mean_recall']:.4f}, "
        f"p50 lat = {hybrid_result['p50_latency_ms']:.2f}ms"
    )

    print("  Running reranked (hybrid+Cohere)...")
    reranked_result: dict = run_reranked(
        client=client,
        bm25_index=bm25_index,
        chunk_ids=chunk_ids,
        corpus=corpus,
        queries=queries,
        ground_truth=ground_truth,
        top_n_dense=args.top_n_dense,
        top_n_bm25=args.top_n_bm25,
        rrf_k=args.rrf_k,
    )
    print(
        f"    reranked mean recall@5 = {reranked_result['mean_recall']:.4f}, "
        f"p50 lat = {reranked_result['p50_latency_ms']:.2f}ms"
    )

    _print_step(7, 7, "Printing results...")
    all_results: list[dict] = [baseline_result, hybrid_result, reranked_result]
    print()
    print_recall_table(all_results)
    print()
    print_per_query_detail(all_results, queries)
    print()
    print_target_check(
        baseline_result["mean_recall"],
        reranked_result["mean_recall"],
        hybrid_recall=hybrid_result["mean_recall"],
    )

    if args.debug:
        # Pick the query whose first BM25-only fused row appears earliest so
        # the diagnostic table can show in_bm25=True, in_dense=False within
        # the printed window (top-50). On the 500-chunk corpus, BM25-only
        # hits often land in the 30s while consensus rows dominate the top.
        debug_idx: int = 0
        debug_fused: list[dict] = []
        debug_query: str = queries[0]["query_text"]
        best_bm25_only_rank: int = 10**9
        for idx, query in enumerate(queries):
            query_text: str = query["query_text"]
            dense_hits: list[dict] = dense_search(
                client, query_text, top_n=args.top_n_dense
            )
            bm25_hits: list[dict] = bm25_search(
                bm25_index, chunk_ids, query_text, top_n=args.top_n_bm25
            )
            fused: list[dict] = fuse_rrf(dense_hits, bm25_hits, k=args.rrf_k)
            for entry in fused:
                if entry["in_bm25"] and not entry["in_dense"]:
                    rank: int = int(entry["final_rank"])
                    if rank < best_bm25_only_rank:
                        best_bm25_only_rank = rank
                        debug_idx = idx
                        debug_fused = fused
                        debug_query = query_text
                    break
        if not debug_fused:
            debug_query = queries[0]["query_text"]
            debug_dense = dense_search(client, debug_query, top_n=args.top_n_dense)
            debug_bm25 = bm25_search(
                bm25_index, chunk_ids, debug_query, top_n=args.top_n_bm25
            )
            debug_fused = fuse_rrf(debug_dense, debug_bm25, k=args.rrf_k)
        debug_retrieved: list[str] = get_top_k_chunk_ids(debug_fused, 5)
        debug_gt: list[str] = ground_truth.get(debug_query, [])
        debug_recall: float = recall_at_5(debug_retrieved, debug_gt)
        print()
        print(
            f"  Debug example: query {debug_idx + 1} hybrid recall@5 = "
            f"{debug_recall:.4f} vs ground-truth {debug_gt}"
        )
        print_rrf_analysis(debug_fused, debug_query, top_n=50)


if __name__ == "__main__":
    main()
