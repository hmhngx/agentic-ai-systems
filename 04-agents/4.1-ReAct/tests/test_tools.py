"""Offline unit tests for tool executors and Pydantic schemas."""

import re

from src.tools import (
    CALCULATOR_DESCRIPTION,
    FILE_READ_DESCRIPTION,
    WEB_MOCK_DESCRIPTION,
    CalculatorInput,
    FileReadInput,
    build_web_mock_prompt,
    execute_calculator,
    execute_file_read,
    execute_web_mock,
)


# --- CALCULATOR ---


def test_calculator_valid_expression():
    """Verify calculator returns correct float for valid arithmetic."""
    result = execute_calculator({"expression": "(3 + 4) * 2"})
    assert "Result: 14" in result, (
        f"Calculator must compute (3+4)*2=14. Got: {result}"
    )


def test_calculator_decimal_expression():
    """Verify demo division 1250000/25 yields 50000 for revenue-per-employee problems."""
    result = execute_calculator({"expression": "1250000 / 25"})
    assert "Result: 50000" in result, (
        f"Calculator must compute 1250000/25=50000 (the demo problem). Got: {result}"
    )


def test_calculator_rejects_letters():
    """Verify calculator security validator blocks code injection via eval()."""
    result = execute_calculator({"expression": "__import__('os').system('ls')"})
    assert "Error" in result, (
        "Calculator must reject expressions with letters/imports. "
        "Without this gate, eval() executes arbitrary code from LLM output."
    )
    assert "ValidationError" in result or "disallowed" in result.lower(), (
        "Rejected expression should cite validation failure"
    )


def test_calculator_rejects_function_calls():
    """Verify function names are rejected so eval() cannot call imported helpers."""
    result = execute_calculator({"expression": "sqrt(16)"})
    assert "Error" in result, (
        "Calculator must reject function names — they require imports."
    )


def test_calculator_rejects_variables():
    """Verify variable names are rejected to keep expressions literal-only."""
    result = execute_calculator({"expression": "x + 1"})
    assert "Error" in result, "Calculator must reject variable names"


def test_calculator_missing_expression():
    """Missing expression must return Error string for the agent Observation."""
    result = execute_calculator({})
    assert "Error" in result, (
        "Missing 'expression' field must produce Error (fed back to LLM as Observation)"
    )


def test_calculator_division_by_zero_returns_error():
    """Division by zero must return Error string — never raise to agent loop."""
    result = execute_calculator({"expression": "1 / 0"})
    assert "Error" in result, (
        "Division by zero must return Error string — never raise to agent loop"
    )


# --- FILE_READ ---


def test_file_read_existing_file(sample_notes_file):
    """Verify file_read returns full contents of notes.txt for the demo workflow."""
    result = execute_file_read({"filename": "notes.txt"})
    assert "Error" not in result[:10], f"file_read failed: {result[:100]}"
    assert "1,250,000" in result or "1250000" in result, (
        "notes.txt must contain the revenue figure for the demo to work"
    )
    assert "25" in result, "notes.txt must contain headcount=25 for demo problem"


def test_file_read_rejects_path_traversal_slash():
    """Path traversal '..' must be blocked to prevent arbitrary file reads."""
    result = execute_file_read({"filename": "../etc/passwd"})
    assert "Error" in result, (
        "Path traversal '..' must be blocked. Without this, LLM can exfiltrate any file."
    )


def test_file_read_rejects_absolute_path():
    """Absolute paths must be blocked because filename is joined under sample_data/."""
    result = execute_file_read({"filename": "/etc/passwd"})
    assert "Error" in result, "Absolute paths must be blocked"


def test_file_read_rejects_backslash():
    """Backslash paths must be blocked to prevent Windows path escape."""
    result = execute_file_read({"filename": "subdir\\file.txt"})
    assert "Error" in result, "Backslash paths must be blocked"


def test_file_read_rejects_tilde():
    """Home directory expansion via ~ must be blocked."""
    result = execute_file_read({"filename": "~/secrets"})
    assert "Error" in result, "Home directory expansion must be blocked"


def test_file_read_missing_file():
    """Missing files must return Error, not crash the agent loop."""
    result = execute_file_read({"filename": "nonexistent_file.txt"})
    assert "Error" in result, "Missing files must return Error, not crash"
    assert "not found" in result.lower()


def test_file_read_missing_filename():
    """Missing filename returns validation error for self-correction."""
    result = execute_file_read({})
    assert "Error" in result, "Missing filename returns validation error"


# --- WEB_MOCK ---


def test_web_mock_pydantic_validation_no_api_call():
    """Verify Pydantic validation runs BEFORE the LLM call."""
    result = execute_web_mock({"query": "ab"})
    assert "Error" in result, (
        "Query <3 chars must fail Pydantic validation (no network call needed)"
    )


def test_web_mock_max_results_bounds():
    """max_results above 5 must fail validation without calling the API."""
    result = execute_web_mock({"query": "test query", "max_results": 10})
    assert "Error" in result, "max_results > 5 must be rejected by Pydantic le=5"


def test_web_mock_max_results_zero():
    """max_results below 1 must fail validation without calling the API."""
    result = execute_web_mock({"query": "test query", "max_results": 0})
    assert "Error" in result, "max_results < 1 must be rejected by Pydantic ge=1"


def test_web_mock_prompt_temporal_orders_newest_first():
    """Latest queries must instruct descending years and grounding-friendly win phrasing."""
    prompt = build_web_mock_prompt("latest population of Tokyo", max_results=3)
    assert "LATEST" in prompt or "latest" in prompt.lower()
    assert "highest-year" in prompt.lower() or "highest year" in prompt.lower()


def test_web_mock_prompt_temporal_does_not_hardcode_domain_facts():
    """Mock prompts should not embed hardcoded answer keys (use real web_search for facts)."""
    prompt = build_web_mock_prompt(
        "Latest Man United player to win Puskas?",
        max_results=3,
    )
    assert "LATEST" in prompt or "latest" in prompt.lower()
    assert "Garnacho" not in prompt


# --- PYDANTIC SCHEMA EXPORT ---


def test_calculator_schema_json_schema():
    """Calculator JSON schema must expose expression for prompt injection."""
    schema = CalculatorInput.model_json_schema()
    assert "properties" in schema
    assert "expression" in schema["properties"]
    assert "required" in schema
    assert "expression" in schema["required"]


def test_file_read_schema_has_validation_description():
    """filename field description must guide the LLM away from invalid paths."""
    schema = FileReadInput.model_json_schema()
    desc = schema["properties"]["filename"].get("description", "")
    assert len(desc) > 30, (
        f"filename field needs detailed description for LLM (got {len(desc)} chars). "
        "Without this, LLM passes wrong paths."
    )


def test_tool_descriptions_at_least_3_sentences():
    """Each tool description needs what/when/what-not for reliable tool selection."""
    for name, desc in [
        ("calc", CALCULATOR_DESCRIPTION),
        ("web", WEB_MOCK_DESCRIPTION),
        ("file", FILE_READ_DESCRIPTION),
    ]:
        sentences = [s for s in re.split(r"\.(?:\s|$)", desc) if s.strip()]
        assert len(sentences) >= 3, (
            f"{name} description must be ≥3 sentences (what/when/what-not). "
            f"Got {len(sentences)}: '{desc[:100]}'"
        )
