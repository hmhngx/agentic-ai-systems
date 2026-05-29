"""rrf_fusion.py - Reciprocal Rank Fusion implementation from scratch.

This is the mathematical core of the hybrid search pipeline.
Every line is commented because this formula must be understood, not
trusted blindly.

RRF solves: how do you combine BM25 scores (range: 0 to 20+, query-dependent)
with dense cosine similarity scores (range: 0 to 1)?
Answer: don't use the scores at all. Use only the rank positions.

Formula:
    RRF(d) = sum_over_systems(1 / (k + rank_of_d_in_system))

k=60 WHY:
    k=60 was empirically validated by Cormack, Clarke, and Buettcher
    (SIGIR 2009, "Reciprocal Rank Fusion outperforms Condorcet and
    individual Rank Learning Methods") across multiple TREC ad-hoc
    retrieval benchmarks. It is NOT an arbitrary choice.

    The value of 60 creates a smooth decay curve where:
        - rank 1 scores: 1/61  ~= 0.01639
        - rank 2 scores: 1/62  ~= 0.01613 (1.6% less than rank 1)
        - rank 60 scores: 1/120 ~= 0.00833 (49% less than rank 1)

    This means the top ~60 results from each system are treated near-equally,
    and consensus across systems matters more than being #1 in one system.
    If k=0: rank 1 scores 1.0, rank 2 scores 0.5 - a 50% cliff.
    If k=60: top positions are smoothed - consensus wins over single-system
    dominance.

    Documents absent from a system's results receive a score of 0 for that
    system. NOT 1/(k + infinity) - literally 0. They did not appear in the
    candidate pool.
"""

from __future__ import annotations

from collections import defaultdict


# Cormack et al. 2009 empirically validated constant. Changing this away
# from 60 is almost always a bug: the smoothing curve described above is
# specific to k=60 and the paper's TREC ad-hoc evaluations - other values
# of k bias the fusion toward either extreme rank consensus (low k) or
# uniform averaging (high k).
RRF_K: int = 60


def fuse_rrf(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion.

    ``dense_results`` items must contain ``chunk_id`` and ``dense_rank``
    (1-indexed). ``bm25_results`` items must contain ``chunk_id`` and
    ``bm25_rank`` (1-indexed).

    Returns a list of dicts sorted by descending RRF score:
        {
            "chunk_id":   str,
            "rrf_score":  float,   # accumulated 1/(k+rank) across systems
            "final_rank": int,     # 1-indexed rank in the fused list
            "in_dense":   bool,
            "in_bm25":    bool,
        }

    The ``in_dense`` and ``in_bm25`` flags are diagnostic - they let you
    see which system contributed each result. Chunks in both systems are
    consensus picks, which is the strongest signal RRF can produce.

    The input lists are NEVER mutated. Empty inputs are handled
    gracefully: if both are empty, returns []. If only one is empty,
    the non-empty list survives the fusion unchanged in rank order
    (but every score is still a valid 1/(k+rank) value).
    """
    if k <= 0:
        # Defence-in-depth. The paper studies k in [0, 1000]; values <= 0
        # would either crash (k=-rank) or invert the rank ordering.
        raise ValueError(f"RRF k must be positive (got {k})")

    # Step 1: build the score accumulator. A defaultdict(float) means a
    # chunk that appears in only one system still gets its single
    # 1/(k+rank) credit, with 0 implied for the missing system.
    scores: defaultdict[str, float] = defaultdict(float)
    seen_dense: set[str] = set()
    seen_bm25: set[str] = set()

    # Step 2: process dense results.
    for result in dense_results:
        chunk_id: str = str(result["chunk_id"])
        rank: int = int(result["dense_rank"])     # 1-indexed
        # Why 1/(k+rank) and not 1/(k+rank-1)?
        # Ranks are 1-indexed. rank=1 -> 1/61. If 0-indexed, rank=0 -> 1/60.
        # The paper's k=60 was tuned with 1-indexed ranks. Passing
        # 0-indexed array positions directly here is the #1 RRF bug.
        scores[chunk_id] += 1.0 / (k + rank)
        seen_dense.add(chunk_id)

    # Step 3: process BM25 results. Documents present in BM25 but absent
    # from dense still accumulate a score here - this is the key
    # mechanism by which BM25 can surface exact-match documents that
    # dense ANN missed entirely.
    for result in bm25_results:
        chunk_id = str(result["chunk_id"])
        rank = int(result["bm25_rank"])           # 1-indexed
        scores[chunk_id] += 1.0 / (k + rank)
        seen_bm25.add(chunk_id)

    if not scores:
        return []

    # Step 4: sort by accumulated RRF score, descending. ties broken by
    # chunk_id lexicographic order so the output is deterministic.
    ordered: list[tuple[str, float]] = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )

    # Step 5 + 6: assign 1-indexed final_rank and return the dict list.
    fused: list[dict] = []
    for rank_pos, (chunk_id, score) in enumerate(ordered, start=1):
        fused.append(
            {
                "chunk_id": chunk_id,
                "rrf_score": score,
                "final_rank": rank_pos,
                "in_dense": chunk_id in seen_dense,
                "in_bm25": chunk_id in seen_bm25,
            }
        )
    return fused


def get_top_k_chunk_ids(fused_results: list[dict], k: int) -> list[str]:
    """Return the chunk_ids of the top-k fused results.

    Used in recall@k computation where we only care about presence in
    the top-k, not the RRF score itself. If fewer than k results were
    fused (very common when a query is OOV for both systems), all
    available are returned - never raises.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    if len(fused_results) < k:
        return [r["chunk_id"] for r in fused_results]
    return [r["chunk_id"] for r in fused_results[:k]]
