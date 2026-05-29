"""ground_truth.py - Exact search ground truth for recall@5 computation.

Design decision: ground truth is derived from Qdrant's exact=True search,
NOT from manually labeled data.

Why Qdrant exact search as ground truth?
    exact=True forces Qdrant to compute distance to every vector in the
    collection (brute force, O(N)). This gives recall@5 = 1.0 by
    definition - the returned top-5 ARE the 5 nearest neighbours in the
    embedding space.
    For this benchmark, we define "ground truth" as the dense exact
    neighbours. This is a valid definition for measuring how well our
    hybrid pipeline preserves the dense retrieval's recall while
    improving precision.

Limitation acknowledged:
    Ground truth from exact dense search measures dense recall only.
    It does not capture ground truth for BM25's exact-match advantage.
    In production, ground truth should be human-annotated relevance
    judgments.
    For this learning benchmark, exact dense search is the correct
    proxy: it is the upper bound of what an ANN-based dense retriever
    can ever surface.

The ground truth is computed ONCE and cached per process. A second call
within the same Python process returns the cached dict without paying
the 20 Qdrant round-trips again.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import SearchParams

from src.dense_retriever import COLLECTION_NAME, embed_query


# Module-level cache. Keyed by query_text so subsequent calls within the
# same process (e.g. the --debug rerun, the per-query --query branch)
# do not re-hit Qdrant. The cache is INTENTIONALLY in-memory only -
# persisting it to disk would risk staleness if the collection content
# changes between runs.
_ground_truth_cache: dict[str, list[str]] = {}

# recall@k is fixed to k=5 for this Day 5 benchmark. Hard-coding the
# constant here documents that contract and prevents accidental
# k-parameter drift between callers.
_K: int = 5


def compute_ground_truth(
    client: QdrantClient,
    queries: list[dict],
) -> dict[str, list[str]]:
    """Run exact Qdrant search for every query and cache the top-5 chunk_ids.

    Returns a dict mapping ``query["query_text"]`` to its top-5 chunk_ids
    in rank order. The order matters only for diagnostic output -
    recall@5 itself is set-based (see :func:`recall_at_5`).

    Performs ~``len(queries)`` Qdrant search calls plus 1 OpenRouter
    embedding call per query (cached internally by the embedder for
    repeated query_text). At 20 queries this is well under the free
    tier on both providers.
    """
    print(
        f"Computing ground truth for {len(queries)} queries via exact search..."
    )
    out: dict[str, list[str]] = {}
    for i, query in enumerate(queries, start=1):
        query_text: str = query["query_text"]
        if query_text in _ground_truth_cache:
            out[query_text] = _ground_truth_cache[query_text]
            continue

        qvec: np.ndarray = embed_query(query_text)
        try:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=qvec.astype(np.float32).tolist(),
                limit=_K,
                search_params=SearchParams(
                    exact=True,            # brute force; O(N) exact recall=1.0 by definition
                ),
                with_payload=True,         # need payload['chunk_id'] for the ground truth keys
                with_vectors=False,        # vectors are 1536-dim each; never needed for ground truth
            )
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise RuntimeError(
                f"Qdrant exact search failed on ground-truth query "
                f"{i}/{len(queries)} ({query_text!r}): {exc}"
            ) from exc

        chunk_ids: list[str] = []
        for point in response.points:
            payload: dict[str, Any] = point.payload or {}
            chunk_id_raw: Any = payload.get("chunk_id", str(point.id))
            chunk_ids.append(str(chunk_id_raw))

        _ground_truth_cache[query_text] = chunk_ids
        out[query_text] = chunk_ids
        snippet: str = query_text[:40]
        print(
            f"  Query {i}/{len(queries)}: '{snippet}...' -> GT: {chunk_ids}"
        )
    return out


def recall_at_5(
    retrieved_chunk_ids: list[str],
    ground_truth_chunk_ids: list[str],
) -> float:
    """Compute recall@5 for a single query.

    Formula: ``|set(retrieved[:5]) intersect set(ground_truth[:5])| / 5``

    Why set intersection, not ordered comparison?
        Recall@k measures presence, not rank order.
        If the 5 correct chunks are returned but in a different order,
        recall is 1.0. This is appropriate because RRF and reranking
        change the order DELIBERATELY (RRF promotes consensus,
        reranking promotes precision). We want to measure: did we
        retrieve the RIGHT chunks, not: did we get the order right?

    Both inputs are truncated to top-5 before the comparison so callers
    can pass longer lists without distorting the metric.
    """
    if not ground_truth_chunk_ids:
        # Defensive guard. With our Qdrant ground truth this should never
        # happen, but returning 0.0 keeps the average computation safe
        # rather than blowing up with a ZeroDivisionError.
        return 0.0

    retrieved_top: set[str] = {str(c) for c in retrieved_chunk_ids[:_K]}
    truth_top: set[str] = {str(c) for c in ground_truth_chunk_ids[:_K]}
    intersection_size: int = len(retrieved_top & truth_top)
    # Denominator is _K, not len(truth_top): the metric is "fraction of
    # top-5 ground truth that we recovered", and with k=5 fixed, the
    # denominator is always 5 by definition.
    return intersection_size / float(_K)
