"""
citation.py - Citation formatting and inline reference injection.

Design decision: citations are formatted as [Source: {pdf}, p.{page}] inline.
Every answer chunk reference includes page number.
This is mandatory - without page citations, users cannot verify claims
against the source document. Unverifiable RAG answers are not production-ready.

Design decision: context blocks are labeled [Doc 1], [Doc 2], etc.
The LLM is instructed to cite [Doc N] inline for every claim.
This creates a two-level citation:
  - [Doc N] in the LLM answer maps to
  - [Source: {pdf}, p.{page}] in the context block
Together they form a complete audit trail from claim -> chunk -> source page.
"""

from __future__ import annotations

import re


# Regex for extracting [Doc N] inline citations from an LLM answer.
# We accept any positive integer N, then validate it against the actual
# result count in ``validate_citations``. The pattern is intentionally
# tolerant of whitespace inside the brackets ("[ Doc 2 ]") because LLMs
# occasionally insert spacing variations.
_DOC_CITATION_RE = re.compile(r"\[\s*Doc\s+(\d+)\s*\]")


def format_context_block(results: list[dict]) -> str:
    """Format retrieved chunks into the CONTEXT portion of the prompt.

    Exact output schema (one block per result, separated by a blank line):

        [Doc 1] Source: {source_pdf}, Page {page_num} | Score: {score:.3f}
        {text}

        [Doc 2] Source: {source_pdf}, Page {page_num} | Score: {score:.3f}
        {text}
        ...

    Why include the score?
        It lets the LLM weight higher-confidence sources more heavily.
        It is also visible in debug output so the user can evaluate
        retrieval quality at a glance.

    Why include [Doc N]?
        The system prompt instructs the LLM to cite [Doc N] inline.
        Programmatic citation validation (see ``validate_citations``)
        depends on this exact label format.
    """
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        header: str = (
            f"[Doc {index}] Source: {result['source_pdf']}, "
            f"Page {result['page_num']} | Score: {result['score']:.3f}"
        )
        body: str = result["text"]
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def format_citation_footer(results: list[dict]) -> str:
    """Format the audit-trail footer appended after EVERY answer.

    Exact output schema:

        -----------------------------------------
        Sources consulted:
          [Doc 1] {source_pdf} - Page {page_num} (relevance: {score:.3f})
          [Doc 2] {source_pdf} - Page {page_num} (relevance: {score:.3f})
          ...
        -----------------------------------------

    This footer is unconditional. Even if the LLM forgot to inline any
    [Doc N] references, the footer still lists every chunk that was
    fed to the prompt - so the user can manually verify the answer
    against the listed pages.
    """
    rule: str = "-" * 41
    if not results:
        # Defensive: callers should never invoke this with empty results
        # because NO_RESULTS skips generation entirely, but if they do we
        # surface that explicitly rather than producing an empty footer.
        return f"{rule}\nSources consulted: (none)\n{rule}"

    lines: list[str] = [rule, "Sources consulted:"]
    for index, result in enumerate(results, start=1):
        lines.append(
            f"  [Doc {index}] {result['source_pdf']} - "
            f"Page {result['page_num']} (relevance: {result['score']:.3f})"
        )
    lines.append(rule)
    return "\n".join(lines)


def validate_citations(answer: str, results: list[dict]) -> list[str]:
    """Verify the LLM's answer cites the retrieved documents correctly.

    Returns a list of human-readable warning strings (empty list means
    the answer passed all citation checks).

    Checks performed:
      1. The answer must contain at least one [Doc N] reference. If it
         does not, the user cannot trace the answer back to specific
         chunks - a serious audit-trail failure.
      2. Every cited N must point to a real result index (1..len(results)).
         A [Doc 7] reference when only 5 chunks were retrieved means the
         LLM invented a citation.

    Warnings are intended to be printed to stderr by the CLI (and shown
    in --debug mode) - they are diagnostic, not user-facing.
    """
    warnings: list[str] = []

    matches: list[str] = _DOC_CITATION_RE.findall(answer)
    if not matches:
        warnings.append("WARNING: Answer contains no inline citations.")
        return warnings  # no point checking ranges - there is nothing to check

    valid_range: range = range(1, len(results) + 1)
    for raw in matches:
        try:
            n: int = int(raw)
        except ValueError:
            # The regex guarantees \d+ so this branch is unreachable in
            # practice, but we keep it for defensive correctness.
            warnings.append(f"WARNING: Answer cites malformed reference [Doc {raw}].")
            continue
        if n not in valid_range:
            warnings.append(
                f"WARNING: Answer cites [Doc {n}] which does not exist "
                f"(valid range: 1..{len(results)})."
            )

    return warnings
