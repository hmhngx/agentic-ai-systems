"""reranker.py - Cross-encoder reranking via the Cohere Rerank API.

Why Cohere Rerank (not a local cross-encoder)?
    Local cross-encoders (sentence-transformers CrossEncoder) require
    downloading 300MB+ models and GPU for reasonable latency. Cohere
    Rerank provides production-grade cross-encoder reranking via API
    with no local model overhead. It uses an in-domain trained cross-
    encoder that has seen diverse text types.

Why cross-encoder beats bi-encoder for reranking:
    Bi-encoder (our dense retrieval): query and document embedded
    independently.
    Cross-encoder: query and document concatenated, processed jointly
    by the transformer.
    Joint processing enables full cross-attention between every query
    token and every document token. This captures:
        - Negation: "does NOT require restarting" != "requires restarting"
        - Attribute specificity: "gradient descent with momentum"
          != "gradient descent"
        - Relational phrases that bi-encoder flattens into averaged vectors.
    Cost: O((Lq + Ld)^2) attention vs O(Lq^2) + O(Ld^2) for bi-encoder.
    Implication: only feasible on a small candidate pool (20-100 docs),
    not the full corpus.

Why top_n=5 for the reranker output?
    topK pathology: retrieving more than 5 final chunks floods the LLM
    context with noise, diluting attention on the true top evidence.
    The reranker's job is to reduce the fused candidate pool (top-20)
    to the final answer set (top-5). The reranker is the precision
    gate - after it, no further filtering should occur.

Fallback strategy:
    If COHERE_API_KEY is not set: skip reranking, return an empty list
    so the caller can substitute the RRF top-N. Benchmark continues
    without the reranker column.
    If the Cohere call raises: catch, print the error, return the
    candidate pool truncated to ``top_n`` so the pipeline does not
    crash mid-benchmark.
"""

from __future__ import annotations

import os
import time
from typing import Any


# Cohere's production reranker (rerank-v3.5) as of 2025. v3 family is the
# multilingual cross-encoder line; rerank-v3.5 supersedes rerank-english-
# v3.0 with cross-lingual coverage and longer context windows.
RERANK_MODEL: str = "rerank-v3.5"

# Final result count after reranking. MUST stay in [3, 5] per the topK
# pathology constraint. 5 is the maximum useful count: beyond that the
# LLM context fills with weakly relevant passages that dilute attention.
RERANK_TOP_N: int = 5

# Input to the reranker: top-20 from RRF fusion.
# Cross-encoder cost is O(N^2) in sequence length per pair, and N pairs
# of cost. 20 candidates is the standard cost/quality sweet spot:
#   - 10 candidates: cheap but often misses the right chunk if RRF
#     ranked it 11-20.
#   - 50 candidates: more API tokens, marginal recall gain.
CANDIDATE_POOL: int = 20

# Cross-encoder owns the top-2 precision slots on easy corpora; slots 3-5
# are filled from high-ef dense top-5 (recall safety net).
RERANK_PRECISION_SLOTS: int = 2

# Module-level cache of the Cohere client. Constructed lazily on the
# first call so importing this module without a key does not warn.
_cohere_client: Any | None = None
_cohere_warned: bool = False
_last_rerank_call: float = 0.0
_rate_limit_hint_printed: bool = False

# Cohere Trial keys allow ~10 rerank calls/minute. Spacing requests avoids
# 429 bursts during the 20-query benchmark. Override via env or set to 0 to
# disable proactive spacing (Production keys with higher limits).
# 429 retries always wait at least _COHERE_429_MIN_BACKOFF_SEC even when
# proactive spacing is disabled.
_COHERE_MIN_INTERVAL_SEC: float = float(
    os.environ.get("COHERE_RERANK_MIN_INTERVAL_SEC", "6.5")
)
_COHERE_429_MIN_BACKOFF_SEC: float = float(
    os.environ.get("COHERE_RERANK_429_MIN_BACKOFF_SEC", "6.5")
)
_COHERE_MAX_RETRIES: int = int(os.environ.get("COHERE_RERANK_MAX_RETRIES", "3"))
_cohere_cooldown_until: float = 0.0


def _rate_limit_backoff_sec(attempt: int) -> float:
    """Seconds to wait after 429; never zero even when proactive throttle is off."""
    step: float = max(_COHERE_MIN_INTERVAL_SEC, _COHERE_429_MIN_BACKOFF_SEC)
    return step * (attempt + 1)


def _wait_cohere_cooldown() -> None:
    """Honor any cooldown set by a prior 429 (applies even when interval=0)."""
    global _cohere_cooldown_until
    now: float = time.monotonic()
    if now < _cohere_cooldown_until:
        time.sleep(_cohere_cooldown_until - now)


def _extend_cohere_cooldown(seconds: float) -> None:
    """Push back the earliest time the next rerank call may fire."""
    global _cohere_cooldown_until
    if seconds <= 0:
        return
    _cohere_cooldown_until = max(_cohere_cooldown_until, time.monotonic() + seconds)


def _get_cohere_client() -> Any | None:
    """Lazily build a cohere.Client; return None if the SDK or key is missing.

    Returning None (rather than raising) lets the caller decide whether to
    substitute the RRF top-N, log a warning, or both. The cohere SDK is an
    optional runtime dependency for this module: the benchmark must run
    even on offline machines.
    """
    global _cohere_client, _cohere_warned
    if _cohere_client is not None:
        return _cohere_client

    api_key: str | None = os.environ.get("COHERE_API_KEY")
    if not api_key:
        if not _cohere_warned:
            print(
                "  WARN: COHERE_API_KEY not set - reranker disabled. "
                "Hybrid pipeline will skip the cross-encoder stage."
            )
            _cohere_warned = True
        return None

    try:
        import cohere  # type: ignore[import-not-found]
    except ImportError:
        if not _cohere_warned:
            print(
                "  WARN: cohere package not installed - reranker disabled. "
                "Install with: pip install 'cohere>=5.0.0'"
            )
            _cohere_warned = True
        return None

    _cohere_client = cohere.Client(api_key=api_key)
    return _cohere_client


def _is_rate_limit_error(exc: Exception) -> bool:
    """True when Cohere rejected the call due to quota / rate limits."""
    if type(exc).__name__ in ("TooManyRequestsError",):
        return True
    status_code: Any = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    message: str = str(exc).lower()
    return "429" in message or "too many requests" in message or "limited to 10" in message


def _throttle_cohere() -> None:
    """Space rerank calls so Trial-tier keys stay under 10 requests/minute."""
    global _last_rerank_call, _rate_limit_hint_printed
    _wait_cohere_cooldown()
    if _COHERE_MIN_INTERVAL_SEC <= 0:
        _last_rerank_call = time.monotonic()
        return
    if not _rate_limit_hint_printed:
        print(
            f"  Cohere rerank throttle: {_COHERE_MIN_INTERVAL_SEC:.1f}s between "
            f"calls (Trial keys ≈10/min; set COHERE_RERANK_MIN_INTERVAL_SEC=0 "
            f"for Production keys)."
        )
        _rate_limit_hint_printed = True
    elapsed: float = time.monotonic() - _last_rerank_call
    if _last_rerank_call > 0.0 and elapsed < _COHERE_MIN_INTERVAL_SEC:
        time.sleep(_COHERE_MIN_INTERVAL_SEC - elapsed)
    _last_rerank_call = time.monotonic()


def _rrf_fallback(chunk_ids: list[str], top_n: int) -> list[dict]:
    """Synthesise rerank rows from the RRF-ordered candidate list."""
    fallback: list[dict] = []
    for rank_pos, chunk_id in enumerate(chunk_ids[:top_n], start=1):
        fallback.append(
            {
                "chunk_id": chunk_id,
                "rerank_score": 0.0,
                "rerank_rank": rank_pos,
                "original_rrf_rank": rank_pos,
            }
        )
    return fallback


def build_rerank_input(
    fused_results: list[dict],
    corpus: list[dict],
    top_candidate: int = CANDIDATE_POOL,
) -> list[tuple[str, str]]:
    """Prepare the (chunk_id, chunk_text) list passed to the Cohere API.

    Steps:
        1. Take the top ``top_candidate`` results from ``fused_results``
           (already sorted by final_rank ascending from fuse_rrf).
        2. For each result: look up the chunk text from ``corpus`` by
           chunk_id.
        3. Return a list of (chunk_id, chunk_text) tuples preserving
           the RRF order.

    Why look up from the in-memory corpus dict rather than fetching from
    Qdrant payload?
        Qdrant payload retrieval is one network call per result.
        The corpus dict is already in memory. Faster, zero added
        latency. They are the same text - corpus was upserted into
        Qdrant in Day 3 (and re-upserted with chunk_id payload by
        dense_retriever.ensure_chunk_id_payload).

    Raises ``KeyError`` with an informative message if a chunk_id from
    ``fused_results`` is not in ``corpus`` - that means the Qdrant
    collection and the in-memory corpus are out of sync, a corpus
    integrity error.
    """
    if top_candidate <= 0:
        raise ValueError(f"top_candidate must be positive (got {top_candidate})")

    text_by_id: dict[str, str] = {chunk["chunk_id"]: chunk["text"] for chunk in corpus}
    selected: list[dict] = fused_results[:top_candidate]
    pairs: list[tuple[str, str]] = []
    for entry in selected:
        chunk_id: str = entry["chunk_id"]
        if chunk_id not in text_by_id:
            raise KeyError(
                f"chunk_id={chunk_id!r} from fused results is not present in "
                f"corpus (size={len(corpus)}). The Qdrant collection and the "
                f"in-memory corpus are out of sync; re-run with the re-ingest "
                f"step enabled."
            )
        pairs.append((chunk_id, text_by_id[chunk_id]))
    return pairs


def build_rerank_input_with_dense_anchor(
    fused_results: list[dict],
    dense_hits: list[dict],
    corpus: list[dict],
    top_candidate: int = CANDIDATE_POOL,
    dense_anchor: int = 10,
) -> list[tuple[str, str]]:
    """Build rerank candidates: high-ef dense top-N first, then RRF tail.

    Prepends dense hits so exact neighbours are always visible to the
    cross-encoder even when RRF ranks consensus same-topic chunks above
    them. The list is capped at ``top_candidate`` pairs.
    """
    if top_candidate <= 0:
        raise ValueError(f"top_candidate must be positive (got {top_candidate})")

    text_by_id: dict[str, str] = {chunk["chunk_id"]: chunk["text"] for chunk in corpus}
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for hit in dense_hits[:dense_anchor]:
        chunk_id = str(hit["chunk_id"])
        if chunk_id in seen:
            continue
        if chunk_id not in text_by_id:
            raise KeyError(
                f"chunk_id={chunk_id!r} from dense hits is not present in corpus."
            )
        pairs.append((chunk_id, text_by_id[chunk_id]))
        seen.add(chunk_id)
        if len(pairs) >= top_candidate:
            return pairs

    for entry in fused_results:
        if len(pairs) >= top_candidate:
            break
        chunk_id = str(entry["chunk_id"])
        if chunk_id in seen:
            continue
        if chunk_id not in text_by_id:
            raise KeyError(
                f"chunk_id={chunk_id!r} from fused results is not present in corpus."
            )
        pairs.append((chunk_id, text_by_id[chunk_id]))
        seen.add(chunk_id)

    return pairs


def finalize_with_dense_backfill(
    reranked_ids: list[str],
    dense_hits: list[dict],
    k: int = RERANK_TOP_N,
    precision_slots: int = RERANK_PRECISION_SLOTS,
) -> list[str]:
    """Build final top-k: cohere may reorder only within high-ef dense top-k.

    Cohere picks outside the dense top-k are ignored for the final answer
    set so ANN recall is not traded away for cross-encoder precision on
    small corpora. Remaining slots stay in dense rank order.
    """
    if k < 3 or k > 5:
        raise ValueError(f"k must be 3..5 (got {k})")
    if precision_slots < 0 or precision_slots > k:
        raise ValueError(
            f"precision_slots must be in [0, k] (got {precision_slots}, k={k})"
        )

    dense_top: list[str] = [str(h["chunk_id"]) for h in dense_hits[:k]]
    if not dense_top:
        return [str(cid) for cid in reranked_ids[:k]]

    allowed: set[str] = set(dense_top)
    promoted: list[str] = []
    for chunk_id in reranked_ids:
        cid = str(chunk_id)
        if cid in allowed and cid not in promoted:
            promoted.append(cid)
        if len(promoted) >= precision_slots:
            break

    remainder: list[str] = [cid for cid in dense_top if cid not in promoted]
    return (promoted + remainder)[:k]


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    """Call Cohere Rerank and return the top-N (chunk_id, score) results.

    Returns a list of dicts:
        {
            "chunk_id":          str,
            "rerank_score":      float,   # Cohere cross-encoder confidence (0..1)
            "rerank_rank":       int,     # 1-indexed final rank
            "original_rrf_rank": int,     # rank within ``candidates`` before reranking
        }

    Fallback semantics:
        - No API key or SDK missing -> returns [] with a logged warning.
          Caller must treat empty as "reranker unavailable" and use the
          RRF top-N directly.
        - SDK call raises -> catches the exception, prints it, and
          returns the first ``top_n`` candidates ordered by their
          original RRF rank. Pipeline keeps running.
    """
    if top_n < 3 or top_n > 5:
        # Hard constraint from the spec: topK pathology bounds the final
        # answer set to [3, 5]. Anything else is a configuration error.
        raise ValueError(f"top_n must be 3..5 (got {top_n})")
    if not candidates:
        return []

    client: Any | None = _get_cohere_client()
    if client is None:
        return []

    chunk_ids: list[str] = [chunk_id for chunk_id, _ in candidates]
    documents: list[str] = [text for _, text in candidates]

    response: Any | None = None
    last_exc: Exception | None = None
    for attempt in range(_COHERE_MAX_RETRIES + 1):
        _throttle_cohere()
        try:
            response = client.rerank(
                query=query,
                documents=documents,
                top_n=top_n,
                model=RERANK_MODEL,
                return_documents=False,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < _COHERE_MAX_RETRIES:
                backoff: float = _rate_limit_backoff_sec(attempt)
                print(
                    f"  WARN: Cohere rate limited (429); retry "
                    f"{attempt + 1}/{_COHERE_MAX_RETRIES} in {backoff:.1f}s..."
                )
                _extend_cohere_cooldown(backoff)
                time.sleep(backoff)
                continue
            print(
                f"  WARN: Cohere rerank API failed ({type(exc).__name__}: {exc}). "
                f"Falling back to RRF top-{top_n}."
            )
            return _rrf_fallback(chunk_ids, top_n)

    if response is None:
        assert last_exc is not None
        print(
            f"  WARN: Cohere rerank exhausted retries ({type(last_exc).__name__}). "
            f"Falling back to RRF top-{top_n}."
        )
        return _rrf_fallback(chunk_ids, top_n)

    # Cohere returns a RerankResponse with .results, each carrying
    # .index (position in our ``documents`` list) and .relevance_score
    # (float 0..1, higher = more relevant).
    results: list[dict] = []
    for rank_pos, item in enumerate(response.results, start=1):
        original_idx: int = int(item.index)
        if original_idx < 0 or original_idx >= len(chunk_ids):
            # Defence in depth - the API contract says indices are
            # bounded, but skipping a malformed entry is safer than
            # raising mid-benchmark.
            continue
        results.append(
            {
                "chunk_id": chunk_ids[original_idx],
                "rerank_score": float(item.relevance_score),
                "rerank_rank": rank_pos,
                "original_rrf_rank": original_idx + 1,  # candidate list was already 0-indexed by RRF
            }
        )
    return results
