"""
web_search.py — Real web search for the ReAct agent (structured JSON observations).

Uses DuckDuckGo text search (no API key). Set REACT_WEB_MODE=mock to use web_mock instead.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field, ValidationError

WEB_SEARCH_DESCRIPTION = (
    "Runs a real web search via DuckDuckGo and returns structured JSON with title, url, "
    "and snippet fields per result. Use for factual or time-sensitive questions such as "
    "award winners, capitals, or current events. Does NOT return full page HTML — only "
    "short snippets; your FINAL ANSWER must cite facts that appear in those snippets."
)


class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description=(
            "A natural-language search query of 3 to 200 characters. "
            "Example: 'latest Manchester United Puskas Award winner'."
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=8,
        description="Number of search results to return, between 1 and 8.",
    )


def _duckduckgo_timeout_seconds() -> float:
    raw = os.environ.get("WEB_SEARCH_TIMEOUT_SECONDS", "12")
    try:
        return max(3.0, float(raw))
    except ValueError:
        return 12.0


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Query DuckDuckGo and return normalized result dicts.
    Raises RuntimeError on network or library failures.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        raise RuntimeError(
            "duckduckgo-search is not installed. Run: python -m pip install -r requirements.txt"
        ) from e

    timeout = _duckduckgo_timeout_seconds()
    rows: list[dict[str, str]] = []
    try:
        with DDGS(timeout=timeout) as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                rows.append(
                    {
                        "title": str(item.get("title") or ""),
                        "url": str(item.get("href") or item.get("link") or ""),
                        "snippet": str(item.get("body") or item.get("snippet") or ""),
                    }
                )
    except Exception as e:
        raise RuntimeError(f"DuckDuckGo search failed: {type(e).__name__}: {e}") from e

    return [r for r in rows if r.get("snippet") or r.get("title")]


def format_search_payload(
    query: str,
    results: list[dict[str, str]],
    *,
    provider: str = "duckduckgo",
    error: str | None = None,
) -> dict[str, Any]:
    """Build the JSON object returned as an Observation."""
    payload: dict[str, Any] = {
        "query": query,
        "provider": provider,
        "results": results,
    }
    if error:
        payload["error"] = error
    return payload


def format_search_observation(payload: dict[str, Any]) -> str:
    """Compact JSON string for the ReAct Observation line."""
    return json.dumps(payload, ensure_ascii=False)


def execute_web_search(input_dict: dict[str, object]) -> str:
    """
    Execute web_search tool. Returns JSON string (not prose).
    Never raises — errors are encoded in the JSON payload or as Error: lines.
    """
    try:
        validated = WebSearchInput(**input_dict)
    except ValidationError as e:
        return f"Error: ValidationError: {e}"

    try:
        results = search_duckduckgo(validated.query, validated.max_results)
    except RuntimeError as e:
        payload = format_search_payload(
            validated.query,
            [],
            error=str(e),
        )
        return format_search_observation(payload)

    if not results:
        payload = format_search_payload(
            validated.query,
            [],
            error="No results returned. Try a shorter or more specific query.",
        )
        return format_search_observation(payload)

    return format_search_observation(
        format_search_payload(validated.query, results),
    )
