"""
pdf_loader.py - Page-aware PDF extraction and chunking.

Design decision: we store page number as metadata on every chunk.
This is mandatory for citations - without page numbers, answers cannot
be verified by the user against the source document.

Design decision: we use pymupdf (fitz) not pypdf because fitz preserves
reading order more reliably and handles complex layouts (multi-column,
headers, footers) better than pypdf's text extraction.
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from typing import Optional

import fitz  # pymupdf: the import name is "fitz" by historical accident


# Cached tiktoken encoder. We try the cl100k_base encoding once; if the BPE
# vocab download fails (e.g. air-gapped CI), we permanently fall back to a
# len(text)//4 heuristic. Caching prevents repeated network retries.
_tiktoken_encoder: "Optional[object]" = None
_tiktoken_attempted: bool = False
_tiktoken_failed: bool = False


def _get_tiktoken_encoder() -> "Optional[object]":
    """Lazily resolve a tiktoken encoder, caching success or failure.

    We catch a broad Exception (not bare ``except``) because tiktoken's
    failure surface includes network errors, ssl errors, JSON decode
    errors, and ``ImportError`` when the package is missing. All of
    them mean the same thing here: fall back to the char/4 heuristic.
    """
    global _tiktoken_encoder, _tiktoken_attempted, _tiktoken_failed
    if _tiktoken_attempted:
        return _tiktoken_encoder

    _tiktoken_attempted = True
    try:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 - network/SSL/import all collapse to fallback
        # We swallow because token counting is advisory: a wrong count
        # only changes chunk boundaries slightly, never correctness.
        print(
            f"  WARN: tiktoken unavailable ({type(exc).__name__}); "
            f"using len(text)//4 fallback for token counts.",
            file=sys.stderr,
        )
        _tiktoken_encoder = None
        _tiktoken_failed = True
    return _tiktoken_encoder


def _count_tokens(text: str) -> int:
    """Return an approximate token count for ``text``.

    Tries tiktoken cl100k_base first (Anthropic's tokenizer is close enough
    to OpenAI's for chunk-sizing purposes - we are not billing tokens here).
    Falls back to ``len(text)//4`` which is the standard "English averages
    ~4 chars per token" rule of thumb.
    """
    encoder = _get_tiktoken_encoder()
    if encoder is None:
        return max(1, len(text) // 4)
    try:
        return len(encoder.encode(text))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - any failure here also collapses to fallback
        return max(1, len(text) // 4)


# Sentence splitter. Splits on [.!?] followed by whitespace + an uppercase
# letter, but ONLY when the punctuation is preceded by 2+ lowercase chars.
# That extra lookbehind prevents splits inside common abbreviations like
# "U.S. Army" (single uppercase before the period) or "e.g. The" (only one
# lowercase). It is a heuristic - true sentence segmentation needs a model
# (spaCy/NLTK) - but it is good enough for retrieval-quality chunking and
# stays in the stdlib.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[a-z]{2}[.!?])\s+(?=[A-Z])")

# Whitespace normalizer: collapses runs of whitespace (including newlines
# inside paragraphs) into single spaces. We keep this conservative so we do
# not destroy paragraph structure that fitz already gave us.
_WHITESPACE_RE = re.compile(r"\s+")


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from every page of ``pdf_path``.

    Returns a list of ``{"page_num": int, "text": str, "char_count": int}``
    dicts. ``page_num`` is **1-indexed** because citations are shown to
    humans, and humans count pages starting at 1.

    Raises:
        FileNotFoundError: if ``pdf_path`` does not exist.
        ValueError: if the PDF appears to be image-only (total extracted
            text under 100 chars) - those PDFs need OCR, which is out of
            scope for this naive RAG.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: list[dict] = []
    total_chars: int = 0

    try:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            raw_text: str = page.get_text("text")  # plain reading-order text
            normalized: str = _WHITESPACE_RE.sub(" ", raw_text).strip()
            if not normalized:
                # Skip blank pages so they do not become empty chunks.
                continue
            pages.append(
                {
                    "page_num": index + 1,  # 1-indexed for human-readable citations
                    "text": normalized,
                    "char_count": len(normalized),
                }
            )
            total_chars += len(normalized)
    finally:
        doc.close()

    if total_chars < 100:
        # Heuristic for image-only PDFs (scanned docs without OCR). Anything
        # under 100 chars across the whole document is almost certainly
        # extraction failure, not a tiny PDF.
        raise ValueError(
            f"PDF appears to be image-only or unreadable "
            f"(extracted {total_chars} chars total from {doc.page_count} pages). "
            f"OCR is required and is out of scope for this naive RAG."
        )

    return pages


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence-ish units using the module regex.

    If no boundary is found (single long sentence or non-prose page), we
    return ``[text]`` so the caller still gets one chunk per page rather
    than silently dropping content.
    """
    parts: list[str] = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _sentences_for_overlap(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Take whole sentences from the tail until we have ~``overlap_tokens``.

    We work backward from the end of the just-emitted chunk so the next
    chunk starts with the most recent sentences. Sentence-level (not
    char-level) overlap keeps thoughts intact across chunk boundaries -
    splitting mid-sentence at character N would create two chunks each
    containing half-formed claims, which kills retrieval relevance.
    """
    accumulated: list[str] = []
    running: int = 0
    for sentence in reversed(sentences):
        accumulated.insert(0, sentence)
        running += _count_tokens(sentence)
        if running >= overlap_tokens:
            break
    return accumulated


def chunk_pages(
    pages: list[dict],
    chunk_size_tokens: int = 300,
    overlap_tokens: int = 30,
) -> list[dict]:
    """Split per-page text into overlapping chunks.

    Why 300 tokens?
        Sweet spot for voyage-3 embedding quality.
        Too small (<100 tokens): chunks lack context for semantic matching.
        Too large (>500 tokens): chunks mix topics, degrading embedding
        precision.

    Why 30 token overlap?
        ~10% of chunk size. Prevents context loss at sentence boundaries.
        Sentence-level overlap (not character-level) ensures complete
        thoughts are not split between the tail of one chunk and the head
        of the next.

    Returns a list of chunk dicts with the schema documented in
    ``load_and_chunk``. Chunks whose stripped text is shorter than 50
    characters are dropped (signal-to-noise too low for embedding).
    """
    chunks: list[dict] = []
    # We need the source filename on each chunk for the citation footer.
    # We grab it from the first page (all pages share the same pdf).
    # If pages is empty this never executes, so we do not need a guard.
    for page in pages:
        page_num: int = page["page_num"]
        page_text: str = page["text"]
        source_pdf: str = page.get("source_pdf", "")

        sentences: list[str] = _split_sentences(page_text)
        if not sentences:
            continue

        current: list[str] = []     # sentences being accumulated into the current chunk
        current_tokens: int = 0     # running token count for the current chunk
        chunk_index: int = 0        # 0-indexed position of the chunk within this page

        i: int = 0
        while i < len(sentences):
            sentence: str = sentences[i]
            sent_tokens: int = _count_tokens(sentence)

            # Case 1: a single sentence alone exceeds the chunk budget.
            # Emit any accumulated text first, then emit this sentence as
            # its own oversized chunk. Flag it so downstream callers can
            # log/inspect oversized chunks if they care.
            if sent_tokens > chunk_size_tokens:
                if current:
                    chunks.append(
                        _build_chunk(
                            text=" ".join(current),
                            page_num=page_num,
                            chunk_index=chunk_index,
                            source_pdf=source_pdf,
                            token_count=current_tokens,
                            oversized=False,
                        )
                    )
                    chunk_index += 1
                    current = []
                    current_tokens = 0
                chunks.append(
                    _build_chunk(
                        text=sentence,
                        page_num=page_num,
                        chunk_index=chunk_index,
                        source_pdf=source_pdf,
                        token_count=sent_tokens,
                        oversized=True,  # explicit flag for downstream visibility
                    )
                )
                chunk_index += 1
                i += 1
                continue

            # Case 2: adding this sentence would tip us over budget. Emit
            # the accumulated chunk, then back up by ~overlap_tokens worth
            # of sentences for the next chunk's starting context.
            if current_tokens + sent_tokens > chunk_size_tokens and current:
                chunks.append(
                    _build_chunk(
                        text=" ".join(current),
                        page_num=page_num,
                        chunk_index=chunk_index,
                        source_pdf=source_pdf,
                        token_count=current_tokens,
                        oversized=False,
                    )
                )
                chunk_index += 1
                overlap_sentences: list[str] = _sentences_for_overlap(current, overlap_tokens)
                current = list(overlap_sentences)
                current_tokens = sum(_count_tokens(s) for s in current)
                # Note: do NOT advance i - we re-evaluate the same sentence
                # against the new (smaller) chunk buffer.
                continue

            # Case 3: normal case - sentence fits, accumulate it.
            current.append(sentence)
            current_tokens += sent_tokens
            i += 1

        # Flush any trailing accumulated sentences at end-of-page.
        if current:
            chunks.append(
                _build_chunk(
                    text=" ".join(current),
                    page_num=page_num,
                    chunk_index=chunk_index,
                    source_pdf=source_pdf,
                    token_count=current_tokens,
                    oversized=False,
                )
            )

    # Drop chunks that are too short to embed usefully. 50 chars is below
    # the floor where voyage-3 produces a reliable semantic embedding -
    # below this you are mostly embedding stop words or page artifacts.
    filtered: list[dict] = [c for c in chunks if len(c["text"].strip()) >= 50]
    return filtered


def _build_chunk(
    text: str,
    page_num: int,
    chunk_index: int,
    source_pdf: str,
    token_count: int,
    oversized: bool,
) -> dict:
    """Construct a chunk dict with the canonical schema.

    ``chunk_id`` is a UUID4 string. Downstream ``vector_store.upsert_chunks``
    converts it to a uint64 for the Qdrant point ID; the original UUID is
    preserved in the payload so we keep a stable, globally-unique handle
    on every chunk for logging and debugging.
    """
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "page_num": page_num,
        "chunk_index": chunk_index,
        "token_count": token_count,
        "char_count": len(text),
        "source_pdf": source_pdf,
        "oversized": oversized,
    }


def load_and_chunk(pdf_path: str) -> tuple[list[dict], dict]:
    """Extract + chunk a PDF in one call. Returns ``(chunks, stats)``.

    The ``stats`` dict gives the caller enough information to decide if
    the ingestion looks healthy (no zero-chunk pages, reasonable token
    distribution) without having to re-iterate the chunk list.
    """
    pages: list[dict] = extract_pages(pdf_path)
    source_pdf: str = os.path.basename(pdf_path)
    # Stamp the source filename on each page dict so chunk_pages can
    # propagate it onto every chunk without a second arg.
    for page in pages:
        page["source_pdf"] = source_pdf

    print(f"Extracting text from {len(pages)} pages...")

    chunks: list[dict] = chunk_pages(pages)

    if chunks:
        token_counts: list[int] = [c["token_count"] for c in chunks]
        avg_tokens: float = sum(token_counts) / len(token_counts)
        min_tokens: int = min(token_counts)
        max_tokens: int = max(token_counts)
    else:
        avg_tokens = 0.0
        min_tokens = 0
        max_tokens = 0

    print(f"Generated {len(chunks)} chunks. Avg {avg_tokens:.0f} tokens/chunk.")

    stats: dict = {
        "pdf_path": pdf_path,
        "total_pages": len(pages),
        "pages_with_text": len(pages),  # blank pages are already filtered in extract_pages
        "total_chunks": len(chunks),
        "avg_tokens_per_chunk": avg_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
    }
    return chunks, stats
