"""Benchmark module. Computes recall@5 and latency for multiple ef_search values.

What recall@5 measures
----------------------
For each of the 20 queries:
  * Get ground-truth top-5 IDs from exact search (denominator = 5).
  * Get ANN top-5 IDs from HNSW search.
  * recall@5 for that query = ``|set(ANN) & set(GT)| / 5``.

Overall recall@5 is the mean across all queries.

  * recall@5 = 1.0 -> ANN returned exactly the same top-5 as brute force
  * recall@5 = 0.8 -> ANN missed on average 1 of 5 relevant results

Latency is reported as p50 (median typical case) and p95 (tail). Mean
latency is intentionally not reported -- it is skewed by JIT warmup, GC
pauses, and Docker overhead, none of which represent steady-state
behaviour your users see.
"""

from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient

from .qdrant_ops import COLLECTION_NAME, search_flat, search_hnsw


def compute_recall_at_k(
    ann_ids: list[int],
    gt_ids: list[int],
    k: int = 5,
) -> float:
    """Compute recall@k for a single query.

    Formula: ``|set(ann_ids[:k]) & set(gt_ids[:k])| / k``.

    Why intersection and not ordered comparison? ANN may return the same
    items in a different rank order than exact search; recall@k measures
    *presence*, not rank. Rank-sensitive metrics (NDCG, MAP) are out of
    scope here.
    """
    if k <= 0:
        raise ValueError(f"k must be positive (got {k})")
    overlap: int = len(set(ann_ids[:k]) & set(gt_ids[:k]))
    return overlap / k


def _percentiles(values: list[float]) -> tuple[float, float]:
    """Return ``(p50, p95)`` of a list of latency samples.

    ``np.percentile`` with linear interpolation gives stable values even
    on small samples (20 queries here).
    """
    arr: np.ndarray = np.asarray(values, dtype=np.float64)
    p50: float = float(np.percentile(arr, 50))
    p95: float = float(np.percentile(arr, 95))
    return p50, p95


def _run_single_ef(
    client: QdrantClient,
    queries: list[dict],
    query_embeddings: np.ndarray,
    ef_search: int,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Run the full 20-query loop for one ``ef_search`` value.

    Two latency vectors are accumulated:
      * ``ann_latencies``   -- one HNSW call per query at this ef_search
      * ``exact_latencies`` -- one exact call per query (ground truth)
    plus a per-query recall@k computed against the exact result.
    """
    ann_latencies: list[float] = []
    exact_latencies: list[float] = []
    recalls: list[float] = []

    for idx, query in enumerate(queries):
        qvec: np.ndarray = query_embeddings[idx]
        gt_ids, gt_ms = search_flat(
            client, qvec, top_k=k, collection_name=collection_name
        )
        ann_ids, ann_ms = search_hnsw(
            client,
            qvec,
            top_k=k,
            ef_search=ef_search,
            collection_name=collection_name,
        )
        ann_latencies.append(ann_ms)
        exact_latencies.append(gt_ms)
        recalls.append(compute_recall_at_k(ann_ids, gt_ids, k=k))

    ann_p50, ann_p95 = _percentiles(ann_latencies)
    exact_p50, exact_p95 = _percentiles(exact_latencies)
    return {
        "ef_search": ef_search,
        "recall@5": float(np.mean(recalls)),
        "ann_p50_ms": ann_p50,
        "ann_p95_ms": ann_p95,
        "exact_p50_ms": exact_p50,
        "exact_p95_ms": exact_p95,
        "queries_run": len(queries),
    }


def run_benchmark(
    client: QdrantClient,
    queries: list[dict],
    query_embeddings: np.ndarray,
    ef_search_values: list[int],
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Run the full benchmark across all queries and ef_search values.

    For each ``ef_search`` in ``ef_search_values`` the inner loop runs
    every query through both ``search_flat`` (ground truth) and
    ``search_hnsw`` at that ef, then aggregates recall and latency.

    Returns one result dict per ef_search value, in input order.
    """
    if query_embeddings.shape[0] != len(queries):
        raise ValueError(
            f"queries ({len(queries)}) and query_embeddings "
            f"({query_embeddings.shape[0]}) must have matching length"
        )
    if not ef_search_values:
        raise ValueError("ef_search_values must contain at least one value")

    results: list[dict] = []
    for ef in ef_search_values:
        print(f"  Benchmarking ef_search={ef}...")
        results.append(
            _run_single_ef(
                client, queries, query_embeddings, ef, collection_name=collection_name
            )
        )
    return results


def _group_queries_by_topic(
    queries: list[dict],
    query_embeddings: np.ndarray,
) -> dict[str, list[tuple[dict, np.ndarray]]]:
    """Index queries by their topic, preserving the embedding alignment."""
    grouped: dict[str, list[tuple[dict, np.ndarray]]] = {}
    for idx, query in enumerate(queries):
        grouped.setdefault(query["topic"], []).append((query, query_embeddings[idx]))
    return grouped


def _run_topic_pair(
    client: QdrantClient,
    topic: str,
    bundles: list[tuple[dict, np.ndarray]],
    ef_search: int,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Compare filtered vs unfiltered HNSW search for a single topic."""
    unfilt_recalls: list[float] = []
    filt_recalls: list[float] = []
    unfilt_lat: list[float] = []
    filt_lat: list[float] = []

    for _query, qvec in bundles:
        gt_ids, _ = search_flat(
            client, qvec, top_k=k, collection_name=collection_name
        )
        unfilt_ids, unfilt_ms = search_hnsw(
            client,
            qvec,
            top_k=k,
            ef_search=ef_search,
            topic_filter=None,
            collection_name=collection_name,
        )
        filt_ids, filt_ms = search_hnsw(
            client,
            qvec,
            top_k=k,
            ef_search=ef_search,
            topic_filter=topic,
            collection_name=collection_name,
        )
        unfilt_lat.append(unfilt_ms)
        filt_lat.append(filt_ms)
        unfilt_recalls.append(compute_recall_at_k(unfilt_ids, gt_ids, k=k))
        filt_recalls.append(compute_recall_at_k(filt_ids, gt_ids, k=k))

    unfilt_p50, _ = _percentiles(unfilt_lat)
    filt_p50, _ = _percentiles(filt_lat)
    return {
        "topic": topic,
        "recall_unfiltered": float(np.mean(unfilt_recalls)),
        "recall_filtered": float(np.mean(filt_recalls)),
        "latency_unfiltered_p50_ms": unfilt_p50,
        "latency_filtered_p50_ms": filt_p50,
    }


def run_filtered_benchmark(
    client: QdrantClient,
    queries: list[dict],
    query_embeddings: np.ndarray,
    ef_search: int = 64,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Run the payload-filtering benchmark across all topics in the query set.

    For each topic, every in-topic query runs twice: once with no filter
    (search across all 500 vectors) and once filtered to its own topic
    (~100 vectors after filtering). Both are compared against the
    unfiltered exact ground truth so the recall numbers are apples-to-
    apples.

    Filtered latency should be at or below unfiltered latency; recall
    should remain high because Qdrant's filterable HNSW preserves graph
    connectivity within the filtered subset.
    """
    if query_embeddings.shape[0] != len(queries):
        raise ValueError(
            f"queries ({len(queries)}) and query_embeddings "
            f"({query_embeddings.shape[0]}) must have matching length"
        )

    grouped = _group_queries_by_topic(queries, query_embeddings)
    results: list[dict] = []
    for topic in sorted(grouped):
        print(f"  Filtered benchmark for topic='{topic}'...")
        results.append(
            _run_topic_pair(
                client,
                topic,
                grouped[topic],
                ef_search,
                collection_name=collection_name,
            )
        )
    return results
