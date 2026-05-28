"""
retriever.py - Search orchestration and result validation.

Design decision: the retriever is a separate layer from the vector store.
vector_store.py handles Qdrant mechanics.
retriever.py handles RAG-specific logic:
  - score threshold filtering
  - no-results detection
  - result deduplication (same page, near-identical text)

Design decision: minimum score threshold = 0.40.
Below 0.40 cosine similarity, the retrieved chunk is unlikely to be
semantically related to the query. Returning low-confidence chunks
causes the LLM to hallucinate by trying to construct an answer
from irrelevant text. Better to return "no results" than low-quality results.
This threshold was determined empirically on voyage-3 embeddings.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from src.embedder import embed_query
from src.vector_store import search


MIN_SCORE_THRESHOLD: float = 0.40

# Jaccard token overlap above which we treat two chunks as duplicates.
# 0.95 is intentionally aggressive - we only collapse near-identical
# text from the same page (e.g. overlap regions between adjacent chunks).
# Anything below 0.95 is treated as distinct context worth keeping.
_DEDUP_OVERLAP_THRESHOLD: float = 0.95


def _jaccard(a: str, b: str) -> float:
    """Compute set-based Jaccard token overlap between two strings.

    Token = whitespace-split word. This is fast, language-agnostic, and
    good enough for spotting near-duplicate chunks that arise from the
    sentence-level overlap window in ``pdf_loader.chunk_pages``. We use
    sets (not multisets) because duplicate words inside a chunk should
    not pump up the similarity score.
    """
    tokens_a: set[str] = set(a.split())
    tokens_b: set[str] = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection: int = len(tokens_a & tokens_b)
    union: int = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return intersection / union


def _deduplicate(results: list[dict]) -> list[dict]:
    """Drop later results that are near-duplicates of an already-kept one.

    Two results are considered duplicates only if they:
      1. Share the same ``page_num`` (chunks from different pages are
         independent context even if they share boilerplate phrasing).
      2. Have Jaccard token overlap > ``_DEDUP_OVERLAP_THRESHOLD``.
    Earlier-ranked (higher-scoring) results are always kept; later
    duplicates are dropped. The remaining list preserves the original
    rank order.
    """
    kept: list[dict] = []
    for candidate in results:
        is_duplicate: bool = False
        for existing in kept:
            if candidate["page_num"] != existing["page_num"]:
                continue
            if _jaccard(candidate["text"], existing["text"]) > _DEDUP_OVERLAP_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


def retrieve(
    client: QdrantClient,
    query: str,
    top_k: int = 5,
    quiet: bool = False,
) -> tuple[list[dict], str]:
    """Run the full retrieval pipeline for a single query.

    Steps:
        1. Embed the query (``input_type="query"``).
        2. Vector search for top_k candidates.
        3. Drop candidates below ``MIN_SCORE_THRESHOLD``.
        4. Deduplicate near-identical chunks on the same page.
        5. Return ``(results, status)`` where status is "OK" or "NO_RESULTS".

    The status string is the caller's signal to either generate an
    answer (OK) or short-circuit to the no-results message (NO_RESULTS).
    The CLI MUST NOT call Claude when status is NO_RESULTS - sending an
    empty context to the LLM is the most reliable way to provoke
    hallucination.
    """
    query_vector = embed_query(query)

    raw_results: list[dict] = search(client, query_vector, top_k=top_k)

    # Threshold filter. Strictly less-than is intentional: results AT the
    # threshold (rare in practice with float scores) are kept.
    filtered: list[dict] = [r for r in raw_results if r["score"] >= MIN_SCORE_THRESHOLD]

    deduped: list[dict] = _deduplicate(filtered)

    if not quiet:
        # Diagnostic line so a user running with --debug (or just by default)
        # can see how aggressively the threshold trimmed the result set.
        dropped_threshold: int = len(raw_results) - len(filtered)
        dropped_dup: int = len(filtered) - len(deduped)
        print(
            f"Retrieved {len(deduped)} chunks "
            f"(after threshold={MIN_SCORE_THRESHOLD} filter, "
            f"dropped {dropped_threshold} below threshold, "
            f"{dropped_dup} duplicates)"
        )
        if deduped:
            top: dict = deduped[0]
            print(f"Top result: page {top['page_num']}, score {top['score']:.4f}")

    if not deduped:
        # The two empty paths (no Qdrant hits, all hits below threshold)
        # both collapse to NO_RESULTS - the caller treats them identically.
        return [], "NO_RESULTS"

    # Re-rank the deduped list 1..N so [Doc 1] in the prompt always
    # matches the order the LLM sees, even after we dropped duplicates.
    for new_rank, result in enumerate(deduped, start=1):
        result["rank"] = new_rank

    return deduped, "OK"
