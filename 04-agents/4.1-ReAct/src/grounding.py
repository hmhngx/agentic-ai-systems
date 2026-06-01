"""
grounding.py — Reject FINAL ANSWER claims not supported by tool Observations.

Evidence is collected from:
  - web_search JSON observations (result snippets)
  - prose observations (numbered lists from web_mock, file_read, calculator)
"""

from __future__ import annotations

import json
import re

_SKIP_TERMS = frozenset(
    {
        "thought",
        "action",
        "observation",
        "final",
        "answer",
        "error",
        "the",
        "this",
        "that",
        "with",
        "from",
        "when",
        "what",
        "which",
        "there",
        "their",
        "about",
        "after",
        "before",
        "because",
        "however",
        "given",
        "united",
        "manchester",
        "player",
        "award",
        "puskas",
        "fifa",
    }
)

_LATEST_QUERY = re.compile(r"\b(latest|most recent|newest|last)\b", re.IGNORECASE)

_DENIAL_PHRASES = (
    "unclear if",
    "not certain",
    "not confirm",
    "did not win",
    "has not won",
    "have not won",
    "was nominated",
    "nominated for",
    "none of them",
    "uncertain if",
)


def _observation_bodies(history: list[dict[str, str]]) -> list[str]:
    """Raw observation bodies (without the 'Observation:' prefix)."""
    bodies: list[str] = []
    for message in history:
        content = message.get("content", "")
        if message.get("role") == "user" and content.startswith("Observation:"):
            bodies.append(content[len("Observation:") :].strip())
    return bodies


def _chunks_from_body(body: str) -> list[str]:
    """Split one observation into evidence chunks (JSON snippets or numbered list items)."""
    stripped = body.strip()
    if not stripped:
        return []

    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            chunks: list[str] = []
            for row in payload["results"]:
                if not isinstance(row, dict):
                    continue
                parts = [str(row.get("title") or ""), str(row.get("snippet") or "")]
                text = " ".join(p for p in parts if p).strip()
                if text:
                    chunks.append(text)
            if chunks:
                return chunks

    text = stripped.lower()
    parts = re.split(r"\s+(?=\d+\.\s)", stripped)
    chunks = []
    for part in parts:
        cleaned = re.sub(r"^\d+\.\s*", "", part.strip())
        if cleaned:
            chunks.append(cleaned)
    if chunks:
        return chunks
    return [stripped]


def _evidence_chunks(history: list[dict[str, str]]) -> list[str]:
    chunks: list[str] = []
    for body in _observation_bodies(history):
        chunks.extend(_chunks_from_body(body))
    return chunks


def _observation_blob(history: list[dict[str, str]]) -> str:
    return "\n".join(c.lower() for c in _evidence_chunks(history))


def _entity_keywords(user_query: str) -> tuple[str, ...]:
    query_lower = user_query.lower()
    keywords: list[str] = []
    if "manchester united" in query_lower or "man united" in query_lower:
        keywords.extend(["manchester united", "man united"])
    return tuple(keywords)


def _claim_terms(final_answer: str) -> list[str]:
    claims: list[str] = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", final_answer):
        phrase = match.group(1).strip()
        if phrase.lower() not in _SKIP_TERMS:
            claims.append(phrase)
    for match in re.finditer(r"\b([A-Z][a-z]{4,})\b", final_answer):
        word = match.group(1).strip()
        if word.lower() not in _SKIP_TERMS and not any(word in c for c in claims):
            claims.append(word)
    return claims


def _years_in(text: str) -> list[int]:
    return [int(year) for year in re.findall(r"20\d{2}", text)]


def _chunk_is_denial(chunk: str) -> bool:
    low = chunk.lower()
    return any(phrase in low for phrase in _DENIAL_PHRASES)


def _chunk_matches_entity(chunk: str, entity_keywords: tuple[str, ...]) -> bool:
    if not entity_keywords:
        return True
    low = chunk.lower()
    return any(keyword in low for keyword in entity_keywords)


def _scoped_evidence_chunks(
    chunks: list[str],
    entity_keywords: tuple[str, ...],
) -> list[str]:
    if not entity_keywords:
        return [c for c in chunks if not _chunk_is_denial(c)]
    return [
        c
        for c in chunks
        if _chunk_matches_entity(c, entity_keywords) and not _chunk_is_denial(c)
    ]


def _term_supported(term: str, observation_blob: str) -> bool:
    parts = term.split()
    if len(parts) > 1:
        return all(part.lower() in observation_blob for part in parts)
    return term.lower() in observation_blob


def _query_wants_latest(user_query: str) -> bool:
    return bool(_LATEST_QUERY.search(user_query))


def _cannot_determine_answer(final_answer: str) -> bool:
    low = final_answer.lower()
    return (
        "cannot determine" in low
        or "not in the observations" in low
        or "insufficient" in low
    )


def _answer_supported_in_chunks(
    final_answer: str,
    chunks: list[str],
) -> bool:
    terms = _claim_terms(final_answer)
    if not terms:
        return False
    answer_years = _years_in(final_answer)
    for chunk in chunks:
        low = chunk.lower()
        for term in terms:
            parts = term.split()
            if not all(part.lower() in low for part in parts):
                continue
            if answer_years and not any(year in answer_years for year in _years_in(chunk)):
                continue
            return True
    return False


def _search_tool_hint() -> str:
    return "web_search or web_mock"


def _check_latest_year_grounding(
    user_query: str,
    final_answer: str,
    history: list[dict[str, str]],
) -> tuple[bool, str | None]:
    if not _query_wants_latest(user_query):
        return True, None

    chunks = _evidence_chunks(history)
    entity_keywords = _entity_keywords(user_query)
    answer_years = _years_in(final_answer)
    scoped = _scoped_evidence_chunks(chunks, entity_keywords)

    if entity_keywords:
        evidence_years: list[int] = []
        for chunk in scoped:
            evidence_years.extend(_years_in(chunk))
        if not evidence_years:
            if _cannot_determine_answer(final_answer):
                return True, None
            return (
                False,
                "Observations do not contain enough scoped evidence for this question. "
                f"Call {_search_tool_hint()} with a narrower query, or FINAL ANSWER: "
                "I cannot determine this from the observations.",
            )

        latest_year = max(evidence_years)
        if not answer_years:
            return (
                False,
                "This question asks for the LATEST match in Observations. "
                f"Evidence includes year {latest_year}. "
                "FINAL ANSWER must name the person/event and include that year, "
                "or say you cannot determine this from the observations.",
            )
        if not _answer_supported_in_chunks(final_answer, scoped):
            return (
                False,
                "FINAL ANSWER must name someone who appears in a supporting Observation "
                "snippet for this team or scope — not an unrelated global result.",
            )
        if max(answer_years) < latest_year:
            return (
                False,
                f"FINAL ANSWER uses year {max(answer_years)} but the latest year in "
                f"scoped Observations is {latest_year}.",
            )
        return True, None

    obs_years: list[int] = []
    for chunk in chunks:
        if not _chunk_is_denial(chunk):
            obs_years.extend(_years_in(chunk))
    if not obs_years:
        return True, None
    latest_obs = max(obs_years)
    if not answer_years:
        return (
            False,
            "This question asks for the LATEST result. "
            f"Observations mention years up to {latest_obs}. "
            "FINAL ANSWER must include the year from Observations.",
        )
    if max(answer_years) < latest_obs:
        return (
            False,
            f"FINAL ANSWER uses year {max(answer_years)} but Observations include {latest_obs}. "
            f"Use the newest year or call {_search_tool_hint()} again.",
        )
    return True, None


def check_final_answer_grounded(
    final_answer: str,
    history: list[dict[str, str]],
    user_query: str = "",
) -> tuple[bool, str | None]:
    """
    Return (True, None) if grounded, else (False, rejection reason for Observation).
    """
    observation_blob = _observation_blob(history)
    if not observation_blob.strip():
        return True, None

    ok, temporal_err = _check_latest_year_grounding(user_query, final_answer, history)
    if not ok:
        return False, temporal_err

    unsupported: list[str] = []
    for term in _claim_terms(final_answer):
        if not _term_supported(term, observation_blob):
            unsupported.append(term)

    if not unsupported:
        return True, None

    listed = ", ".join(repr(t) for t in unsupported)
    return (
        False,
        "FINAL ANSWER rejected (grounding): "
        f"{listed} not found in any prior Observation. "
        "You may only state names, dates, and facts that appear in Observations. "
        f"Call {_search_tool_hint()} again, or give FINAL ANSWER: "
        "I cannot determine this from the observations.",
    )
