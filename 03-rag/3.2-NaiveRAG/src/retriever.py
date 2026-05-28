"""
retriever.py - Search orchestration and result validation.

Design decision: the retriever is a separate layer from the vector store.
vector_store.py handles Qdrant mechanics.
retriever.py handles RAG-specific logic:
  - score threshold filtering
  - no-results detection
  - result deduplication (same page, near-identical text)

Design decision: minimum score threshold defaults to 0.25.
Below the threshold, the retrieved chunk is unlikely to be
semantically related to the query. Returning low-confidence chunks
causes the LLM to hallucinate by trying to construct an answer
from irrelevant text. Better to return "no results" than low-quality results.
This threshold should be tuned per embedding model and corpus and can
be overridden with the ``MIN_SCORE_THRESHOLD`` environment variable.
"""

from __future__ import annotations

import os
import re

from qdrant_client import QdrantClient

from src.embedder import embed_query
from src.vector_store import search


MIN_SCORE_THRESHOLD: float = float(os.environ.get("MIN_SCORE_THRESHOLD", "0.25"))
REFERENCE_PENALTY: float = float(os.environ.get("REFERENCE_PENALTY", "0.08"))
QUERY_AWARE_REFERENCE_PENALTY: float = float(
    os.environ.get("QUERY_AWARE_REFERENCE_PENALTY", "0.18")
)
EARLY_PAGE_BOOST: float = float(os.environ.get("EARLY_PAGE_BOOST", "0.06"))

# Jaccard token overlap above which we treat two chunks as duplicates.
# 0.95 is intentionally aggressive - we only collapse near-identical
# text from the same page (e.g. overlap regions between adjacent chunks).
# Anything below 0.95 is treated as distinct context worth keeping.
_DEDUP_OVERLAP_THRESHOLD: float = 0.95
_YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")


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


def _looks_reference_like(text: str) -> bool:
    """Return True for bibliography/acknowledgment-heavy chunks.

    This is a lightweight lexical heuristic used only for re-ranking.
    It intentionally does not drop chunks outright: reference sections can
    still be useful for specific queries about citations or funding.
    """
    low = text.lower()
    signals: tuple[str, ...] = (
        "references",
        "bibliography",
        "acknowledgment",
        "acknowledgement",
        "et al.",
        "doi:",
        "arxiv:",
        "in proceedings of",
        "findings of the association for computational linguistics",
        "advances in neural information processing systems",
        "association for computational linguistics",
        "new york, ny, usa",
    )
    citation_like_years: int = len(_YEAR_RE.findall(low))
    has_many_semicolons: bool = low.count(";") >= 3
    # Author list heuristic: many commas plus years usually indicates refs.
    author_list_like: bool = low.count(",") >= 6 and citation_like_years >= 2
    return (
        any(signal in low for signal in signals)
        or has_many_semicolons
        or author_list_like
    )


def _is_broad_intent_query(query: str) -> bool:
    """Detect broad "paper overview" intents that need body chunks first."""
    q = query.lower()
    cues: tuple[str, ...] = (
        "summary",
        "summarize",
        "abstract",
        "main contribution",
        "main contributions",
        "what is this paper about",
        "what is the paper about",
        "paper about",
        "main idea",
    )
    return any(cue in q for cue in cues)


def _is_reference_intent_query(query: str) -> bool:
    """Detect queries that explicitly ask about references/citations."""
    q = query.lower()
    cues: tuple[str, ...] = (
        "citation",
        "citations",
        "reference",
        "references",
        "bibliography",
        "works cited",
    )
    return any(cue in q for cue in cues)


def _rerank_reference_noise(results: list[dict], query: str) -> list[dict]:
    """Query-aware re-rank to reduce bibliography/reference dominance."""
    broad_intent: bool = _is_broad_intent_query(query)
    reference_intent: bool = _is_reference_intent_query(query)
    rescored: list[dict] = []
    for item in results:
        adjusted_score: float = item["score"]
        is_reference_like: bool = _looks_reference_like(item["text"])
        if is_reference_like and not reference_intent:
            penalty: float = QUERY_AWARE_REFERENCE_PENALTY if broad_intent else REFERENCE_PENALTY
            adjusted_score -= penalty
        if is_reference_like and reference_intent:
            # If user explicitly asks about references/citations, reverse the
            # suppression so those chunks stay near the top.
            adjusted_score += REFERENCE_PENALTY
        # For broad intents, favor early pages where title/abstract/intro
        # usually live. This is a soft boost, not a hard filter.
        if broad_intent and int(item.get("page_num", 0) or 0) <= 2:
            adjusted_score += EARLY_PAGE_BOOST
        enriched = dict(item)
        enriched["_is_reference_like"] = is_reference_like
        enriched["_adjusted_score"] = adjusted_score
        rescored.append(enriched)

    rescored.sort(key=lambda r: (r["_adjusted_score"], r["score"]), reverse=True)
    if broad_intent and not reference_intent:
        # Keep context diverse for overview questions: at most one
        # reference-like chunk in the final top-k list.
        kept: list[dict] = []
        ref_kept: int = 0
        for item in rescored:
            if item["_is_reference_like"]:
                if ref_kept >= 1:
                    continue
                ref_kept += 1
            kept.append(item)
            if len(kept) >= len(results):
                break
        rescored = kept

    for rank, item in enumerate(rescored, start=1):
        item["rank"] = rank
        item.pop("_adjusted_score", None)
        item.pop("_is_reference_like", None)
    return rescored


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
    reranked: list[dict] = _rerank_reference_noise(deduped, query=query)

    if not quiet:
        # Diagnostic line so a user running with --debug (or just by default)
        # can see how aggressively the threshold trimmed the result set.
        dropped_threshold: int = len(raw_results) - len(filtered)
        dropped_dup: int = len(filtered) - len(deduped)
        print(
            f"Retrieved {len(reranked)} chunks "
            f"(after threshold={MIN_SCORE_THRESHOLD} filter, "
            f"dropped {dropped_threshold} below threshold, "
            f"{dropped_dup} duplicates)"
        )
        if reranked:
            top: dict = reranked[0]
            print(f"Top result: page {top['page_num']}, score {top['score']:.4f}")

    if not reranked:
        # The two empty paths (no Qdrant hits, all hits below threshold)
        # both collapse to NO_RESULTS - the caller treats them identically.
        return [], "NO_RESULTS"

    # Re-rank the deduped list 1..N so [Doc 1] in the prompt always
    # matches the order the LLM sees, even after we dropped duplicates.
    for new_rank, result in enumerate(reranked, start=1):
        result["rank"] = new_rank

    return reranked, "OK"
