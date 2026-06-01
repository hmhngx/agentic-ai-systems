"""Offline unit tests for system prompt and conversation assembly."""

from src.prompt_builder import build_conversation, build_system_prompt


def test_system_prompt_contains_format_spec():
    """System prompt must document Thought, Action, and FINAL ANSWER format."""
    prompt = build_system_prompt()
    assert "Thought:" in prompt, "Format spec must include 'Thought:'"
    assert "Action:" in prompt, "Format spec must include 'Action:'"
    assert "FINAL ANSWER" in prompt, "Format spec must include FINAL ANSWER"


def test_system_prompt_forbids_self_observation():
    """System prompt must forbid the model from generating its own Observation."""
    prompt = build_system_prompt()
    assert "Observation" in prompt, "Must mention Observation in format"
    forbid_phrases = [
        "not generate Observation",
        "do not generate observation",
        "Observations come from",
        "not generate observation",
        "do not write observation",
        "from the observations",
    ]
    found = any(phrase.lower() in prompt.lower() for phrase in forbid_phrases)
    assert found, (
        "System prompt MUST forbid model from generating its own Observation. "
        "Without this, model hallucinates tool results."
    )


def test_system_prompt_lists_all_tools():
    """Every registered tool name must appear in the system prompt."""
    prompt = build_system_prompt()
    for tool in ["calculator", "file_read"]:
        assert tool in prompt, f"Tool '{tool}' must appear in system prompt"
    assert "web_search" in prompt or "web_mock" in prompt


def test_system_prompt_contains_schemas():
    """Schema field names must appear so the model knows valid JSON keys."""
    prompt = build_system_prompt()
    assert "expression" in prompt, "calculator schema field must be in prompt"
    assert "filename" in prompt, "file_read schema field must be in prompt"
    assert "query" in prompt, "web tool schema field 'query' must be in prompt"


def test_build_conversation_structure():
    """Conversation must be system, user query, then history in order."""
    history = [
        {"role": "assistant", "content": "Thought: X\nAction: {...}"},
        {"role": "user", "content": "Observation: Result: 4"},
    ]
    messages = build_conversation("solve this", history)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "solve this"
    assert messages[2] == history[0]
    assert messages[3] == history[1]


def test_build_conversation_empty_history():
    """Empty history yields only system and user messages."""
    messages = build_conversation("query", [])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
