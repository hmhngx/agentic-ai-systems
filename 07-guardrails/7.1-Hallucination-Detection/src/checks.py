"""
checks.py - The three independent grounding checks.

Each check is a pure function ``(answer, chunks, config) -> <Result>``. They do
not know about scoring, thresholds, or verdicts - they only measure. Combining
those measurements into a single groundedness score and a verdict is the job of
``scoring.py``. This separation is what lets each check be unit-tested in
isolation and reused by any RAG.

  Check 1  check_support   - did the answer actually use the retrieved chunks?
  Check 2  check_citations - are the cited [Doc N] real chunk IDs?
  (retrieval adequacy)     - did we retrieve a usable amount of evidence at all?
"""

from __future__ import annotations

import difflib

import numpy as np

from src.text import (
    DOC_CITATION_RE,
    content_tokens,
    split_sentences,
    strip_citations,
)
from src.types import (
    ClaimSupport,
    CitationResult,
    DetectorConfig,
    RetrievalResult,
    RetrievedChunk,
    SupportResult,
)


# ---------------------------------------------------------------------------
# Check 1: support (string-overlap method)
# ---------------------------------------------------------------------------
def _token_containment(claim_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Fraction of the claim's content tokens that appear in the chunk.

    Containment ( |claim & chunk| / |claim| ), not Jaccard, on purpose: a short
    true claim fully contained in a long chunk should score ~1.0, whereas
    Jaccard would punish it for the chunk's length. This is the
    paraphrase-tolerant signal - it ignores word order.
    """
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


def _difflib_span_ratio(claim_tokens: list[str], chunk_tokens: list[str]) -> float:
    """Fraction of claim tokens covered by contiguous matches against the chunk.

    Uses ``difflib.SequenceMatcher`` over the *token sequences* and sums the
    sizes of the matching blocks, normalised by the claim length. This is the
    verbatim-extraction signal: it rewards the claim being lifted (in order)
    from the chunk, which set-overlap alone cannot distinguish from a bag of
    coincidentally shared words. ``autojunk=False`` keeps short, common tokens
    eligible for matching on long chunks.
    """
    if not claim_tokens:
        return 0.0
    matcher = difflib.SequenceMatcher(a=claim_tokens, b=chunk_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(claim_tokens)


def check_support(
    answer: str,
    chunks: list[RetrievedChunk],
    config: DetectorConfig,
) -> SupportResult:
    """Check 1: measure how well each claim in the answer is grounded.

    Algorithm:
      1. Strip citations, split the answer into sentences ("claims").
      2. Tokenise each claim and each chunk to content tokens.
      3. Build a claims x chunks support matrix with numpy, where each cell is
         ``max(token_containment, difflib_span_ratio)`` - the best evidence the
         claim is present in that chunk, by either a lenient set measure or a
         strict contiguous measure.
      4. Per-claim support = the row max (best supporting chunk).
      5. supported_ratio = fraction of claims with support >= tau_claim;
         mean_support = mean of per-claim support (feeds the composed score).

    Claims with no content tokens (e.g. "It is.") are skipped - they carry no
    verifiable facts and would otherwise distort the ratio.
    """
    raw_claims = split_sentences(strip_citations(answer))

    # Pre-tokenise claims, dropping those with no content tokens.
    claim_token_lists: list[list[str]] = []
    kept_claims: list[str] = []
    for claim in raw_claims:
        toks = content_tokens(claim)
        if toks:
            claim_token_lists.append(toks)
            kept_claims.append(claim)

    method = "string-overlap"

    # No scorable claims, or nothing to compare against -> zero support.
    if not claim_token_lists or not chunks:
        per_claim = tuple(
            ClaimSupport(
                claim=claim,
                support=0.0,
                containment=0.0,
                span_ratio=0.0,
                best_chunk_id=None,
                supported=False,
            )
            for claim in kept_claims
        )
        return SupportResult(
            supported_ratio=0.0,
            mean_support=0.0,
            per_claim=per_claim,
            method=method,
        )

    chunk_token_lists = [content_tokens(c.text) for c in chunks]
    chunk_token_sets = [set(toks) for toks in chunk_token_lists]

    n_claims = len(claim_token_lists)
    n_chunks = len(chunks)

    # Two matrices so both signals are available for diagnostics; the support
    # matrix is their elementwise max.
    containment = np.zeros((n_claims, n_chunks), dtype=float)
    span = np.zeros((n_claims, n_chunks), dtype=float)

    for i, claim_toks in enumerate(claim_token_lists):
        claim_set = set(claim_toks)
        for j in range(n_chunks):
            containment[i, j] = _token_containment(claim_set, chunk_token_sets[j])
            span[i, j] = _difflib_span_ratio(claim_toks, chunk_token_lists[j])

    support_matrix = np.maximum(containment, span)

    # Best supporting chunk per claim.
    best_chunk_idx = support_matrix.argmax(axis=1)
    per_claim_support = support_matrix.max(axis=1)

    per_claim: list[ClaimSupport] = []
    for i, claim in enumerate(kept_claims):
        j = int(best_chunk_idx[i])
        support_val = float(per_claim_support[i])
        per_claim.append(
            ClaimSupport(
                claim=claim,
                support=support_val,
                containment=float(containment[i, j]),
                span_ratio=float(span[i, j]),
                best_chunk_id=chunks[j].chunk_id,
                supported=support_val >= config.claim_support_threshold,
            )
        )

    supported_ratio = float(
        np.mean(per_claim_support >= config.claim_support_threshold)
    )
    mean_support = float(np.mean(per_claim_support))

    return SupportResult(
        supported_ratio=supported_ratio,
        mean_support=mean_support,
        per_claim=tuple(per_claim),
        method=method,
    )


# ---------------------------------------------------------------------------
# Check 2: citation validity
# ---------------------------------------------------------------------------
def check_citations(
    answer: str,
    chunks: list[RetrievedChunk],
    config: DetectorConfig,
) -> CitationResult:
    """Check 2: verify every cited ``[Doc N]`` maps to a real chunk ID.

    The valid label space is exactly the set of ``chunk.chunk_id`` values (for
    the naive-RAG adapter, ``"Doc 1" .. "Doc N"``). A citation to a label
    outside that set - e.g. ``[Doc 9]`` when five chunks were retrieved - is a
    *fabricated* citation: the model invented a source. That is treated as a
    hard red flag by the scorer.

    ``validity`` is |cited & valid| / |cited|, or 0.0 when nothing is cited
    (an uncited answer cannot be traced, so it earns no citation credit).
    """
    valid_ids = tuple(c.chunk_id for c in chunks)
    valid_set = set(valid_ids)

    # Extract distinct cited labels, normalised to the "Doc N" form so they
    # compare directly against chunk_id values.
    cited_set: set[str] = {
        f"Doc {int(n)}" for n in DOC_CITATION_RE.findall(answer)
    }
    cited_ids = tuple(sorted(cited_set, key=lambda s: int(s.split()[1])))

    fabricated = tuple(
        sorted(cited_set - valid_set, key=lambda s: int(s.split()[1]))
    )

    if not cited_set:
        validity = 0.0
    else:
        validity = len(cited_set & valid_set) / len(cited_set)

    return CitationResult(
        cited_ids=cited_ids,
        valid_ids=valid_ids,
        fabricated_ids=fabricated,
        validity=validity,
        has_citations=bool(cited_set),
    )


# ---------------------------------------------------------------------------
# Retrieval adequacy
# ---------------------------------------------------------------------------
def check_retrieval(
    chunks: list[RetrievedChunk],
    config: DetectorConfig,
) -> RetrievalResult:
    """Measure whether a usable amount of evidence was retrieved at all.

    This is a faint prior, not a quality judgement - it is weighted lowest in
    the composed score precisely because a high retrieval score does not imply
    a grounded answer. When chunks expose retrieval scores we blend a
    count-adequacy term with the mean score; when they do not, "chunks exist"
    is treated as adequate so the detector still works on RAGs that hide scores.
    """
    n = len(chunks)
    if n == 0:
        return RetrievalResult(n_chunks=0, adequacy=0.0, mean_score=None)

    count_adequacy = min(n / max(config.target_chunk_count, 1), 1.0)

    present_scores = [c.score for c in chunks if c.score is not None]
    if present_scores:
        mean_score = float(np.mean(present_scores))
        # Clip into [0, 1]; cosine similarities already live there but other
        # backends may not.
        clipped = min(max(mean_score, 0.0), 1.0)
        adequacy = 0.5 * count_adequacy + 0.5 * clipped
    else:
        mean_score = None
        adequacy = count_adequacy

    return RetrievalResult(n_chunks=n, adequacy=adequacy, mean_score=mean_score)
