"""bm25_retriever.py - BM25 sparse retrieval using rank-bm25 (BM25Okapi).

Why BM25Okapi (not BM25Plus or BM25L)?
    BM25Okapi is the canonical BM25 variant with k1 and b parameters.
    It is the implementation referenced in all major IR papers and benchmarks.
    BM25Plus addresses zero-term-frequency issues but adds complexity
    without meaningful improvement for general-purpose document retrieval.

Why BM25 is mandatory alongside dense search:
    Dense embeddings are blind to exact token matches.
    A query for "XJ-99-B error code" embeds into a semantic neighborhood
    that includes "product error" and "fault code" - semantically adjacent
    but potentially missing the exact string.
    BM25 with its inverted index guarantees that any chunk containing
    "XJ-99-B" scores highly regardless of semantic context.
    For our 5-topic corpus: "gradient descent", "bioluminescent", "Maillard reaction"
    are all exact tokens that BM25 will match precisely.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


# BM25Okapi default parameters (documented in rank-bm25 source):
# k1=1.5 - TF saturation: controls how quickly additional term occurrences
#            lose marginal value. 1.5 is slightly more aggressive than
#            the BM25 paper's recommendation of 1.2-2.0.
#            We use the library default - change only if domain analysis
#            shows very short chunks (k1 closer to 1.2) or very repetitive
#            text (k1 closer to 2.0).
# b=0.75  - Document length normalization. 0.75 is the standard empirical default.
#            Partially normalizes for chunk length - longer chunks get
#            a slight penalty to prevent them from dominating by sheer volume.
#
# rank-bm25's BM25Okapi.__init__ hard-codes these defaults; surfacing the
# values here as module constants documents the contract without forcing
# us to pass them explicitly (passing them and getting them wrong would
# be silent failure - leaving them implicit lets the library default
# remain the single source of truth).
_BM25_K1: float = 1.5
_BM25_B: float = 0.75

# Word tokenizer used by BOTH the index builder and the query side.
# \b\w+\b matches Unicode word characters bounded by non-word boundaries:
#   - keeps "gradient-descent" as two tokens (good - either half matches)
#   - keeps "Maillard" as one token after .lower()
#   - drops standalone punctuation entirely
# Critical: re-using the SAME compiled pattern at query time guarantees
# the BM25 vocabulary lookup never misses due to tokenizer drift.
_WORD_RE: re.Pattern[str] = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Lowercase + word-tokenize. Used identically at index and query time.

    Stopwords are intentionally NOT removed. Reason: BM25's IDF term
    naturally down-weights tokens that appear in most documents, and
    stopword filtering can hurt recall on exact-phrase queries like
    "the great wall" where each word individually carries low IDF but
    the phrase as a whole is highly diagnostic.
    """
    return _WORD_RE.findall(text.lower())


def build_bm25_index(corpus: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """Build a BM25Okapi index from a list of chunk dicts.

    Returns ``(bm25_index, chunk_ids)`` where ``chunk_ids[i]`` is the
    ``chunk_id`` of the i-th tokenized document inside the index. This
    parallel-list contract is the ONLY way to map BM25's returned
    integer ranks back to the cross-system chunk identifier; mutating
    ``corpus`` order after this call invalidates the mapping.

    Tokenization: ``re.findall(r"\\b\\w+\\b", text.lower())``. We do NOT
    remove stopwords - see _tokenize docstring.
    """
    if not corpus:
        raise ValueError("build_bm25_index called with empty corpus")

    tokenized_corpus: list[list[str]] = [_tokenize(chunk["text"]) for chunk in corpus]
    chunk_ids: list[str] = [chunk["chunk_id"] for chunk in corpus]

    bm25_index: BM25Okapi = BM25Okapi(tokenized_corpus)

    total_tokens: int = sum(len(doc) for doc in tokenized_corpus)
    avg_tokens: float = total_tokens / len(tokenized_corpus)
    print(
        f"BM25 index built: {len(tokenized_corpus)} documents, "
        f"avg {avg_tokens:.0f} tokens/doc (k1={_BM25_K1}, b={_BM25_B})"
    )
    return bm25_index, chunk_ids


def bm25_search(
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    query: str,
    top_n: int = 50,
) -> list[dict]:
    """Run BM25 search for a single query. Returns ranked candidate results.

    ``top_n=50`` default:
        BM25 retrieves a CANDIDATE POOL, not the final results.
        50 is appropriate here - we will fuse with dense top-50 via RRF,
        then optionally rerank the fused top-20 with a cross-encoder.
        Do NOT use top_n=5 here. That is the final result count, not
        the candidate pool. Retrieving only 5 from BM25 before RRF
        defeats the purpose of fusion.

    Returns a list of dicts:
        {"chunk_id": str, "bm25_score": float, "bm25_rank": int}
    with 1-indexed ranks, in descending score order. Zero-score rows
    are filtered (zero means the query had no token overlap with the
    chunk - including it would distort RRF by giving it a free rank).

    Returns [] when the entire query is OOV (all tokens unseen at index
    time), which yields an all-zero score vector.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive (got {top_n})")
    if len(chunk_ids) == 0:
        return []

    # Symmetric tokenization with build_bm25_index. WHY: BM25 scores are
    # computed against the index's vocabulary - if the query tokenizer
    # differs from the index tokenizer, query terms may not match any
    # indexed tokens and the search silently produces zero scores.
    tokenized_query: list[str] = _tokenize(query)
    if not tokenized_query:
        return []

    scores: np.ndarray = np.asarray(
        bm25_index.get_scores(tokenized_query), dtype=np.float64
    )
    # argsort descending: highest score first. ``[::-1]`` reverses the
    # ascending argsort output. We slice to top_n BEFORE filtering zeros
    # because BM25 score==0 only happens at the tail of the ranking
    # (zero-IDF documents always rank lowest among non-zero ones).
    ranked_idx: np.ndarray = np.argsort(scores)[::-1][:top_n]

    results: list[dict] = []
    for rank_pos, idx in enumerate(ranked_idx, start=1):  # 1-indexed rank
        score: float = float(scores[int(idx)])
        if score <= 0.0:
            # Zero-score rows mean the query had no overlap with this
            # chunk's vocabulary. Drop them - feeding them into RRF
            # would give the chunk an undeserved 1/(k+rank) credit.
            continue
        results.append(
            {
                "chunk_id": chunk_ids[int(idx)],
                "bm25_score": score,
                "bm25_rank": rank_pos,
            }
        )
    return results
