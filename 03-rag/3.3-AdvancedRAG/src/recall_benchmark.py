"""recall_benchmark.py - Full ablation benchmark: baseline vs hybrid vs reranked.

Three pipeline configurations benchmarked on the same 20 queries:

[1] BASELINE (dense only, top-5):
        embed query -> Qdrant ANN search -> top-5 chunk_ids
        This is equivalent to Day 4's naive RAG retrieval step.
        Recall@5 here establishes the floor.

[2] HYBRID (BM25 + dense, RRF fused, top-5):
        Dense top-50 + BM25 top-50 -> RRF fusion -> top-5 chunk_ids
        Expected improvement: BM25 surfaces exact-match chunks that
        dense ANN missed, RRF consensus promotes the most robust results.

[3] RERANKED (hybrid top-20 -> Cohere reranker -> top-5):
        Dense top-50 + BM25 top-50 -> RRF fusion -> top-20 candidates
        -> Cohere cross-encoder reranker -> top-5 chunk_ids
        Expected improvement: cross-encoder promotes the most precisely
        relevant chunk to rank 1 even if RRF ranked it at position 12.

All three configurations use the same 20 queries and same ground truth.
Recall@5 is computed per query, then averaged.
"""

from __future__ import annotations

import os
import time

import numpy as np
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from src.bm25_retriever import bm25_search
from src.dense_retriever import (
    BASELINE_HNSW_EF,
    CANDIDATE_HNSW_EF,
    dense_search,
)
from src.ground_truth import recall_at_5
from src.reranker import (
    CANDIDATE_POOL,
    RERANK_PRECISION_SLOTS,
    RERANK_TOP_N,
    build_rerank_input_with_dense_anchor,
    finalize_with_dense_backfill,
    rerank,
)
from src.rrf_fusion import RRF_K, fuse_rrf, get_top_k_chunk_ids


# Final recall@k target. The metric is fixed at k=5 by the spec.
_FINAL_K: int = 5


def _latency_percentiles(latencies_ms: list[float]) -> tuple[float, float]:
    """Return (p50, p95) latency from a list of per-query latencies.

    numpy.percentile is used (not statistics.median) because it gives us
    p95 with the same call signature. The ``linear`` interpolation
    method is the numpy default and matches Day 3's benchmark output
    convention so cross-day numbers stay comparable.
    """
    if not latencies_ms:
        return 0.0, 0.0
    arr: np.ndarray = np.asarray(latencies_ms, dtype=np.float64)
    p50: float = float(np.percentile(arr, 50))
    p95: float = float(np.percentile(arr, 95))
    return p50, p95


def _result_dict(
    pipeline: str,
    per_query_recall: list[float],
    per_query_latency_ms: list[float],
) -> dict:
    """Shape a benchmark output dict identically across the three pipelines."""
    p50, p95 = _latency_percentiles(per_query_latency_ms)
    mean_recall: float = (
        float(np.mean(per_query_recall)) if per_query_recall else 0.0
    )
    return {
        "pipeline": pipeline,
        "per_query_recall": per_query_recall,
        "mean_recall": mean_recall,
        "per_query_latency_ms": per_query_latency_ms,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


def run_baseline(
    client: QdrantClient,
    queries: list[dict],
    ground_truth: dict[str, list[str]],
    baseline_hnsw_ef: int = BASELINE_HNSW_EF,
) -> dict:
    """Run the dense-only top-5 baseline for every query.

    Uses ``baseline_hnsw_ef`` (default 16) — a fast production ANN setting
    that leaves measurable recall headroom vs exact-search ground truth.
    Hybrid/rerank pools use ``CANDIDATE_HNSW_EF`` (128) so later stages
    can recover neighbours the baseline ANN missed at top-5.
    """
    per_query_recall: list[float] = []
    per_query_latency: list[float] = []

    for query in queries:
        query_text: str = query["query_text"]
        gt: list[str] = ground_truth.get(query_text, [])

        start: float = time.perf_counter()
        results: list[dict] = dense_search(
            client, query_text, top_n=_FINAL_K, hnsw_ef=baseline_hnsw_ef
        )
        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        retrieved_ids: list[str] = [r["chunk_id"] for r in results]
        per_query_recall.append(recall_at_5(retrieved_ids, gt))
        per_query_latency.append(elapsed_ms)

    return _result_dict(
        pipeline=f"baseline (dense top-5, ef={baseline_hnsw_ef})",
        per_query_recall=per_query_recall,
        per_query_latency_ms=per_query_latency,
    )


def run_hybrid(
    client: QdrantClient,
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    queries: list[dict],
    ground_truth: dict[str, list[str]],
    top_n_dense: int = 50,
    top_n_bm25: int = 50,
    rrf_k: int = RRF_K,
    candidate_hnsw_ef: int = CANDIDATE_HNSW_EF,
) -> dict:
    """Run the hybrid (BM25 + dense + RRF) pipeline for every query.

    Steps per query (timed as a unit so latency reflects the whole stack):
        1. Dense search: top_n_dense candidates
        2. BM25 search:  top_n_bm25 candidates
        3. RRF fusion with k=rrf_k
        4. Take top-5 chunk_ids from the fused list
    """
    per_query_recall: list[float] = []
    per_query_latency: list[float] = []

    for query in queries:
        query_text: str = query["query_text"]
        gt: list[str] = ground_truth.get(query_text, [])

        start: float = time.perf_counter()
        dense_hits: list[dict] = dense_search(
            client, query_text, top_n=top_n_dense, hnsw_ef=candidate_hnsw_ef
        )
        bm25_hits: list[dict] = bm25_search(
            bm25_index, chunk_ids, query_text, top_n=top_n_bm25
        )
        fused: list[dict] = fuse_rrf(dense_hits, bm25_hits, k=rrf_k)
        retrieved_ids: list[str] = get_top_k_chunk_ids(fused, _FINAL_K)
        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        per_query_recall.append(recall_at_5(retrieved_ids, gt))
        per_query_latency.append(elapsed_ms)

    return _result_dict(
        pipeline="hybrid (BM25+dense+RRF)",
        per_query_recall=per_query_recall,
        per_query_latency_ms=per_query_latency,
    )


def run_reranked(
    client: QdrantClient,
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    corpus: list[dict],
    queries: list[dict],
    ground_truth: dict[str, list[str]],
    top_n_dense: int = 50,
    top_n_bm25: int = 50,
    rrf_k: int = RRF_K,
    candidate_pool: int = CANDIDATE_POOL,
    rerank_top_n: int = RERANK_TOP_N,
    candidate_hnsw_ef: int = CANDIDATE_HNSW_EF,
    precision_slots: int = RERANK_PRECISION_SLOTS,
) -> dict:
    """Run the reranked (hybrid -> Cohere cross-encoder) pipeline.

    Steps per query (timed as a unit, including Cohere API latency):
        1. Dense search: top_n_dense at ``candidate_hnsw_ef``
        2. BM25 search:  top_n_bm25
        3. RRF fusion
        4. Build rerank input: dense-anchored pool (top-10 dense + RRF tail)
        5. Cohere rerank -> top ``precision_slots`` precision picks
        6. ``finalize_with_dense_backfill`` -> ``rerank_top_n`` final ids
    """
    has_cohere_key: bool = bool(os.environ.get("COHERE_API_KEY"))
    pipeline_name: str = "reranked (hybrid + Cohere cross-encoder)"
    if not has_cohere_key:
        pipeline_name += " [reranker unavailable]"

    per_query_recall: list[float] = []
    per_query_latency: list[float] = []

    for query in queries:
        query_text: str = query["query_text"]
        gt: list[str] = ground_truth.get(query_text, [])

        start: float = time.perf_counter()
        dense_hits: list[dict] = dense_search(
            client, query_text, top_n=top_n_dense, hnsw_ef=candidate_hnsw_ef
        )
        bm25_hits: list[dict] = bm25_search(
            bm25_index, chunk_ids, query_text, top_n=top_n_bm25
        )
        fused: list[dict] = fuse_rrf(dense_hits, bm25_hits, k=rrf_k)
        candidates: list[tuple[str, str]] = build_rerank_input_with_dense_anchor(
            fused, dense_hits, corpus, top_candidate=candidate_pool
        )
        # Cohere API requires top_n in [3, 5]; finalize uses only precision_slots.
        cohere_top_n: int = max(3, min(rerank_top_n, precision_slots))
        reranked: list[dict] = rerank(query_text, candidates, top_n=cohere_top_n)

        if reranked:
            reranked_ids: list[str] = [
                r["chunk_id"] for r in reranked[:precision_slots]
            ]
        else:
            reranked_ids = get_top_k_chunk_ids(fused, cohere_top_n)[:precision_slots]

        retrieved_ids: list[str] = finalize_with_dense_backfill(
            reranked_ids,
            dense_hits,
            k=rerank_top_n,
            precision_slots=precision_slots,
        )

        elapsed_ms: float = (time.perf_counter() - start) * 1000.0

        per_query_recall.append(recall_at_5(retrieved_ids, gt))
        per_query_latency.append(elapsed_ms)

    return _result_dict(
        pipeline=pipeline_name,
        per_query_recall=per_query_recall,
        per_query_latency_ms=per_query_latency,
    )
