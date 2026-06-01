"""Offline tests for web_search (DuckDuckGo mocked)."""

import json
from unittest.mock import patch

from src.grounding import check_final_answer_grounded
from src.web_search import (
    execute_web_search,
    format_search_observation,
    format_search_payload,
)


def test_format_search_payload_roundtrip():
    payload = format_search_payload(
        "test query",
        [{"title": "A", "url": "https://a", "snippet": "Fact about Oslo."}],
    )
    text = format_search_observation(payload)
    parsed = json.loads(text)
    assert parsed["query"] == "test query"
    assert parsed["results"][0]["snippet"] == "Fact about Oslo."


def test_execute_web_search_validation_no_network():
    result = execute_web_search({"query": "ab"})
    assert result.startswith("Error:")


@patch("src.web_search.search_duckduckgo")
def test_execute_web_search_returns_json(mock_search):
    mock_search.return_value = [
        {
            "title": "Garnacho wins Puskas",
            "url": "https://example.com",
            "snippet": (
                "Alejandro Garnacho won the 2024 FIFA Puskas Award for Manchester United."
            ),
        }
    ]
    raw = execute_web_search({"query": "Man United Puskas winner", "max_results": 3})
    data = json.loads(raw)
    assert data["results"][0]["snippet"].startswith("Alejandro")
    mock_search.assert_called_once()


@patch("src.web_search.search_duckduckgo")
def test_grounding_accepts_json_web_search_observation(mock_search):
    mock_search.return_value = [
        {
            "title": "FIFA",
            "url": "https://fifa.com",
            "snippet": (
                "Alejandro Garnacho won the 2024 FIFA Puskas Award for Manchester United."
            ),
        },
        {
            "title": "BBC",
            "url": "https://bbc.com",
            "snippet": (
                "Cristiano Ronaldo won the FIFA Puskas Award in 2009 for Manchester United."
            ),
        },
    ]
    observation = execute_web_search({"query": "latest Man United Puskas winner"})
    history = [{"role": "user", "content": f"Observation: {observation}"}]
    ok, err = check_final_answer_grounded(
        "Alejandro Garnacho in 2024",
        history,
        user_query="Latest Man United player to win Puskas?",
    )
    assert ok is True, err


@patch("src.web_search.search_duckduckgo", side_effect=RuntimeError("network down"))
def test_execute_web_search_encodes_errors_in_json(mock_search):
    raw = execute_web_search({"query": "capital of Norway"})
    data = json.loads(raw)
    assert data["results"] == []
    assert "error" in data
    mock_search.assert_called_once()
