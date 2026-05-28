"""Offline tests for ``src/citation.py``.

Citations are the audit trail of the RAG system. If the [Doc N]
labels in the context block do not match the labels in the footer,
or if validate_citations cannot detect a missing/invented reference,
the user has no way to verify the LLM's claims against the source.
That destroys trust in the system far faster than wrong answers do.

All tests use the ``sample_results`` fixture from tests/conftest.py.
No mocking; ``citation.py`` is pure-Python and pure-functional.
"""

from __future__ import annotations

from src.citation import (
    format_citation_footer,
    format_context_block,
    validate_citations,
)


def test_format_context_block_structure(sample_results: list[dict]) -> None:
    """Context block must label each chunk with [Doc N], Page, and Score."""
    result = format_context_block(sample_results)
    assert isinstance(result, str), \
        f"format_context_block must return str, got {type(result).__name__}. " \
        "The caller concatenates this into the LLM prompt as-is."
    assert "[Doc 1]" in result, \
        "Missing [Doc 1] label - the LLM is told to cite [Doc N] and " \
        "without these labels in the context it cannot cite at all."
    assert "[Doc 2]" in result, "Missing [Doc 2] label in context block."
    assert "[Doc 3]" in result, "Missing [Doc 3] label in context block."
    assert "Page" in result, \
        "Context block missing 'Page' header - users need page numbers " \
        "to manually verify the LLM's citations against the PDF."
    assert "Score:" in result, \
        "Context block missing 'Score:' - the score lets the LLM weight " \
        "high-confidence sources more heavily during generation."


def test_format_context_block_doc_labels_sequential(sample_results: list[dict]) -> None:
    """[Doc N] labels must increment 1..N in order - LLM relies on positional cue."""
    result = format_context_block(sample_results)
    lines = result.split("\n")
    doc_headers = [l for l in lines if l.strip().startswith("[Doc")]
    assert len(doc_headers) == 3, \
        f"Expected 3 [Doc N] headers, found {len(doc_headers)}: {doc_headers}. " \
        "Missing or extra headers desynchronize labels from the actual chunks."
    for i, header in enumerate(doc_headers, start=1):
        assert f"[Doc {i}]" in header, \
            f"Expected [Doc {i}] at position {i}, found: {header!r}. " \
            "Out-of-order labels mean [Doc 1] in the answer can no longer " \
            "be assumed to refer to the highest-scoring chunk."


def test_format_context_block_contains_page_nums(sample_results: list[dict]) -> None:
    """Every chunk's page number must appear in the formatted block."""
    result = format_context_block(sample_results)
    for r in sample_results:
        assert str(r["page_num"]) in result, \
            f"Page {r['page_num']} not found in context block. " \
            "Users cannot verify a claim if the prompt doesn't show which " \
            "page the chunk came from."


def test_format_context_block_contains_chunk_text(sample_results: list[dict]) -> None:
    """The chunk's text must appear verbatim - the LLM cannot cite what it cannot see."""
    result = format_context_block(sample_results)
    for r in sample_results:
        first_20 = r["text"][:20]
        assert first_20 in result, \
            f"Chunk text prefix {first_20!r} not in context block. " \
            "Either the formatter is dropping text or mangling it - both " \
            "are catastrophic because the LLM is now grounded in nothing."


def test_format_context_block_empty_results() -> None:
    """Empty result list must not crash - NO_RESULTS path may call here defensively."""
    result = format_context_block([])
    assert isinstance(result, str), \
        f"format_context_block([]) must still return a string, " \
        f"got {type(result).__name__}."
    assert result.strip() == "" or "[Doc" not in result, \
        f"Empty input produced spurious doc labels: {result!r}. " \
        "Fake labels would mislead the LLM into citing nothing."


def test_format_citation_footer_structure(sample_results: list[dict]) -> None:
    """Footer must list every chunk with [Doc N], source filename, page, score."""
    footer = format_citation_footer(sample_results)
    assert isinstance(footer, str), \
        f"format_citation_footer must return str, got {type(footer).__name__}."
    assert "Sources" in footer or "Source" in footer, \
        f"Footer missing 'Sources' header: {footer!r}. " \
        "Without a labeled section users can't tell answer from audit trail."
    assert "[Doc 1]" in footer, \
        "Footer missing [Doc 1] - the unconditional audit trail must list " \
        "every chunk that went into the prompt, regardless of LLM behavior."
    assert "Page" in footer, \
        "Footer missing 'Page' - the user can't trace [Doc N] back to a page."
    for r in sample_results:
        assert r["source_pdf"] in footer, \
            f"source_pdf '{r['source_pdf']}' missing from footer. " \
            "Users must see which file each citation comes from when " \
            "multiple PDFs are ingested later."


def test_format_citation_footer_all_docs_listed(sample_results: list[dict]) -> None:
    """Every retrieved chunk must appear in the footer - dropping any breaks audit."""
    footer = format_citation_footer(sample_results)
    for i in range(1, len(sample_results) + 1):
        assert f"[Doc {i}]" in footer, \
            f"[Doc {i}] missing from footer - the audit trail is incomplete, " \
            "which means a chunk the LLM saw is invisible to the user."


def test_format_citation_footer_scores_present(sample_results: list[dict]) -> None:
    """Footer must show relevance scores so users see retrieval confidence."""
    footer = format_citation_footer(sample_results)
    for r in sample_results:
        score_str = f"{r['score']:.3f}"
        assert score_str in footer, \
            f"Score {score_str} not in citation footer. " \
            "Without scores users cannot tell a confident match (0.92) " \
            "from a barely-above-threshold result (0.41)."


def test_validate_citations_no_citations_warns(sample_results: list[dict]) -> None:
    """An uncited answer must produce a warning - that's a serious audit failure."""
    warnings = validate_citations(
        "This is an answer with no doc references.", sample_results
    )
    assert isinstance(warnings, list), \
        f"validate_citations must return list, got {type(warnings).__name__}."
    assert len(warnings) >= 1, \
        f"Expected >=1 warning for uncited answer, got {warnings}. " \
        "An uncited RAG answer cannot be verified - the warning is the only " \
        "signal that the LLM ignored the citation directive."
    assert any(
        "no inline citations" in w.lower() or "citation" in w.lower()
        for w in warnings
    ), f"Expected citation warning text, got: {warnings}. " \
       "The operator searches stderr for 'citation' when triaging - the " \
       "warning text must contain that token to be findable."


def test_validate_citations_valid_citation_no_warn(sample_results: list[dict]) -> None:
    """A properly cited answer must NOT trigger a false-positive warning."""
    answer = "The model uses backpropagation [Doc 1]. Coral reefs are diverse [Doc 2]."
    warnings = validate_citations(answer, sample_results)
    citation_warnings = [w for w in warnings if "no inline citations" in w.lower()]
    assert len(citation_warnings) == 0, \
        f"Valid citations triggered a false warning: {warnings}. " \
        "False positives train operators to ignore the warning channel, " \
        "which then hides real audit failures."


def test_validate_citations_invalid_doc_number_warns(sample_results: list[dict]) -> None:
    """A [Doc 99] reference must trigger a warning - the LLM invented a citation."""
    answer = "This cites a nonexistent document [Doc 99]."
    warnings = validate_citations(answer, sample_results)
    assert any("99" in w or "does not exist" in w.lower() for w in warnings), \
        f"Expected warning for [Doc 99] (out of range 1..{len(sample_results)}), " \
        f"got: {warnings}. " \
        "Invented citations are the loudest hallucination signal - if we " \
        "don't catch them we have no defense against fabricated audit trails."


def test_validate_citations_returns_list(sample_results: list[dict]) -> None:
    """Return type contract: list of strings, never None, never a single string."""
    result = validate_citations("test", sample_results)
    assert isinstance(result, list), \
        f"validate_citations must return list, got {type(result).__name__}. " \
        "Callers iterate the result; a string would iterate per-character " \
        "and produce nonsense warnings."
    assert all(isinstance(w, str) for w in result), \
        f"Every warning must be str, got types {[type(w).__name__ for w in result]}. " \
        "Non-string warnings break the CLI's print-to-stderr loop."
