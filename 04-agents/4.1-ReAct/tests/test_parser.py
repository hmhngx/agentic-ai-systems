"""Offline unit tests for ReAct output parsing — critical for loop self-correction."""

from src.parser import parse_llm_output


def test_parse_thought_action_basic():
    """Valid Thought/Action JSON must parse into tool name and input dict."""
    output = (
        'Thought: I need to add numbers.\n'
        'Action: {"tool":"calculator","input":{"expression":"2+2"}}'
    )
    result = parse_llm_output(output)
    assert result.parse_error is None, f"Should parse cleanly. Error: {result.parse_error}"
    assert "add numbers" in result.thought
    assert result.action_name == "calculator"
    assert result.action_input == {"expression": "2+2"}
    assert result.is_final is False


def test_parse_final_answer():
    """FINAL ANSWER must terminate parsing with no action fields."""
    output = "Thought: I have the answer.\nFINAL ANSWER: The result is 50000."
    result = parse_llm_output(output)
    assert result.is_final is True
    assert "50000" in result.final_answer
    assert result.action_name is None
    assert result.action_input is None


def test_parse_handles_markdown_fences():
    """Models often wrap JSON in ```json fences — parser must strip them."""
    output = """Thought: Calculate.
Action: ```json
{"tool":"calculator","input":{"expression":"3*4"}}
```"""
    result = parse_llm_output(output)
    assert result.parse_error is None or result.action_name == "calculator", (
        "Parser must handle markdown code fences around JSON"
    )


def test_parse_handles_uppercase_final_answer():
    """FINAL ANSWER label must be recognized in standard casing."""
    output = "Thought: Done.\nFINAL ANSWER: 42"
    result = parse_llm_output(output)
    assert result.is_final is True


def test_parse_handles_lowercase_final_answer():
    """Parser must be case-insensitive on FINAL ANSWER for model variance."""
    output = "Thought: Done.\nfinal answer: 42"
    result = parse_llm_output(output)
    assert result.is_final is True, (
        "Parser must be case-insensitive on FINAL ANSWER"
    )


def test_parse_invalid_json_returns_parse_error():
    """Invalid JSON must produce parse_error so loop can inject as Observation."""
    output = "Thought: Try.\nAction: {tool: calculator, input: invalid}"
    result = parse_llm_output(output)
    assert result.parse_error is not None, (
        "Invalid JSON must produce parse_error so loop can inject as Observation"
    )
    assert result.is_final is False


def test_parse_missing_tool_key_returns_error():
    """Action JSON without 'tool' must yield parse_error mentioning tool."""
    output = 'Thought: ok.\nAction: {"input":{"x":1}}'
    result = parse_llm_output(output)
    assert result.parse_error is not None
    assert "tool" in result.parse_error.lower()


def test_parse_missing_input_key_returns_error():
    """Action JSON without 'input' must yield parse_error mentioning input."""
    output = 'Thought: ok.\nAction: {"tool":"calculator"}'
    result = parse_llm_output(output)
    assert result.parse_error is not None
    assert "input" in result.parse_error.lower()


def test_parse_empty_output_returns_error():
    """Empty LLM output must not crash the parser."""
    result = parse_llm_output("")
    assert result.parse_error is not None


def test_parse_thought_only_returns_actionable_error():
    """Thought without Action must hint truncation and brief thoughts."""
    output = "Thought: I need to check one more thing but forgot the action line."
    result = parse_llm_output(output)
    assert result.parse_error is not None
    assert "Action" in result.parse_error
    assert "FINAL ANSWER" in result.parse_error


def test_parse_no_format_at_all_returns_error():
    """Raw answer without ReAct format — must produce parse_error for self-correction."""
    result = parse_llm_output("The answer is 42.")
    assert result.parse_error is not None, (
        "Raw text without Thought/Action/FINAL ANSWER must trigger parse_error"
    )


def test_parse_never_raises():
    """Parser must NEVER raise — errors must be returned as parse_error."""
    pathological_inputs = [
        "",
        "   ",
        "{{{}}",
        "Thought:",
        "Action: null",
        "Thought:\nAction: }",
        "\x00\x01garbage",
    ]
    for bad in pathological_inputs:
        try:
            result = parse_llm_output(bad)
            assert (
                result.parse_error is not None
                or result.is_final
                or result.action_name
            )
        except Exception as e:
            assert False, (
                f"Parser raised {type(e).__name__} on input {bad!r}. Must never raise."
            )


def test_parsed_turn_preserves_raw_output():
    """ParsedTurn.raw_output must be exactly the original LLM response for debugging."""
    original = "Thought: X\nFINAL ANSWER: Y"
    result = parse_llm_output(original)
    assert result.raw_output == original, (
        "raw_output must preserve exact original"
    )
