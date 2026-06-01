"""
tools.py — Three tools with strict Pydantic input schemas.

Why Pydantic schemas instead of plain dict validation?
  Pydantic does three things at once:
  1. Type coercion: "3.14" → 3.14 if field is float
  2. Validation: required fields must be present, types must match
  3. Self-documentation: schema can be auto-converted to JSON Schema for the prompt

  Without Pydantic, every tool function would need manual validation:
    if "expression" not in input_dict: raise...
    if not isinstance(input_dict["expression"], str): raise...
  Pydantic replaces all of that with `Calculator(**input_dict)`.

Why each tool needs a description ≥ 3 sentences?
  The LLM selects tools by semantic similarity between query and description.
  A one-line description like "does math" matches arithmetic, statistics,
  geometry, currency conversion, and unit conversion — all simultaneously.
  Three sentences allow:
    1. What the tool does (one sentence)
    2. Concrete examples of valid inputs (one sentence)
    3. What it does NOT do — explicit failure modes (one sentence)
  This eliminates a class of tool-selection errors before they happen.

Why temperature=0.7 inside web_mock but 0 in the agent loop?
  The web_mock tool simulates a web search by asking the LLM to generate
  plausible-looking search results. Real search engines return varied results,
  so we use temperature=0.7 to introduce variation. The AGENT's tool-selection
  logic still runs at temperature=0 — only the internal simulation varies.
"""

import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(
        ...,
        description=(
            "A single arithmetic expression containing only digits, decimal points, "
            "the operators + - * / and parentheses. Example: '(3 + 4) * 2.5'. "
            "Do NOT include variables, function names, units, or words."
        ),
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """
        Block dangerous characters. Only allow digits, operators, decimals, parens, spaces.
        This prevents code injection via eval().
        """
        if not re.fullmatch(r"[0-9+\-*/().\s]+", v):
            raise ValueError(
                f"Expression contains disallowed characters. Got: '{v}'. "
                "Only digits, + - * / ( ) . and spaces are permitted."
            )
        return v.strip()


CALCULATOR_DESCRIPTION = (
    "Evaluates a single arithmetic expression and returns the numeric result as a float. "
    "Accepts only digits, decimal points, the operators + - * /, and parentheses — "
    "for example '(3 + 4) * 2.5' returns 17.5. "
    "Does NOT handle variables, function names like sqrt() or sin(), unit conversions, "
    "string operations, or any expression containing letters."
)


def execute_calculator(input_dict: dict[str, object]) -> str:
    """
    Execute calculator tool. Returns string representation of the result.

    Validation pipeline:
      1. Pydantic validates the input dict against CalculatorInput schema.
         Raises ValidationError on invalid input.
      2. The validator regex blocks anything other than digits/operators/parens.
         This is the security gate against eval() injection.
      3. eval() is called inside a restricted namespace.

    Why eval()? Because we have already validated the expression against a strict regex.
    This is the standard pattern for "calculator tools" in agent systems.
    For production, use a math expression parser like asteval or simpleeval.

    Returns: f"Result: {value}" on success, or "Error: {message}" on any failure.
    """
    try:
        validated = CalculatorInput(**input_dict)
        result = eval(validated.expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except ValidationError as e:
        return f"Error: ValidationError: {e}"
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError) as e:
        return f"Error: {type(e).__name__}: {e}"


class WebMockInput(BaseModel):
    """Input schema for the web_mock search tool."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description=(
            "A natural-language search query of 3 to 200 characters. "
            "Example: 'population of Tokyo 2024' or 'speed of light in vacuum'. "
            "Use this for factual lookups that would normally require a web search."
        ),
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of mock search results to return, between 1 and 5.",
    )


WEB_MOCK_DESCRIPTION = (
    "Simulates a web search by generating search-result snippets via a secondary LLM call. "
    "Use for factual lookups like 'capital of Norway' or 'Puskas Award Manchester United winner'. "
    "Snippets aim to include specific names, years, and teams relevant to the query. "
    "Still verify: your FINAL ANSWER must only cite facts that appear in the Observation text."
)

_TEMPORAL_WORDS = ("latest", "most recent", "newest", "last ")


def _query_is_temporal(query_lower: str) -> bool:
    return any(word in query_lower for word in _TEMPORAL_WORDS)


def _web_mock_search_hints(query_lower: str) -> str:
    """Extra instructions for the mock-search LLM (testable via build_web_mock_prompt)."""
    if not _query_is_temporal(query_lower):
        return ""
    return (
        "The user wants the LATEST / MOST RECENT item: put the highest-year fact "
        "first in snippet 1 with an explicit 4-digit year. When multiple winners exist, "
        "list them in descending year order. Do not claim someone is the 'only' winner "
        "if a newer winner exists. Include full names and 4-digit years."
    )


def build_web_mock_prompt(query: str, max_results: int = 3) -> str:
    """Build the internal LLM prompt for web_mock (testable without an API call)."""
    extra = _web_mock_search_hints(query.lower())
    extra_block = f" {extra}" if extra else ""
    return (
        f"Generate {max_results} web search result snippets for the query: "
        f"\"{query}\".{extra_block} "
        "Format as a numbered list. Each snippet is 1-2 sentences. "
        "Use accurate, widely-known public facts when possible (award winners, capitals, dates, teams). "
        "Include specific proper names and years the user would need to answer the query. "
        "If uncertain, state uncertainty in the snippet. No commentary outside the numbered list."
    )


def execute_web_mock(input_dict: dict[str, object]) -> str:
    """
    Generate fake search results via a secondary LLM call.

    Why is this a separate LLM call?
      This tool exists to test the agent's ability to integrate retrieved information
      into its reasoning without depending on a real web service. The fake results
      are deterministic-ish (temperature=0.7) and contain enough plausible information
      for the agent to chain through multi-step problems.

    The internal LLM call uses temperature=0.7 (varied results, like a real search).
    The agent loop using this tool runs at temperature=0 (deterministic reasoning).
    """
    from src.llm_client import WEB_MOCK_TEMPERATURE, call_llm

    try:
        validated = WebMockInput(**input_dict)
    except ValidationError as e:
        return f"Error: ValidationError: {e}"

    prompt = build_web_mock_prompt(validated.query, validated.max_results)
    try:
        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=WEB_MOCK_TEMPERATURE,
        )
        return response.strip()
    except RuntimeError as e:
        return f"Error: RuntimeError: {e}"


class FileReadInput(BaseModel):
    """Input schema for the file_read tool."""

    filename: str = Field(
        ...,
        description=(
            "The name of a file inside the sample_data/ directory. "
            "Example: 'notes.txt'. Do NOT include subdirectories, absolute paths, "
            "or '..' — only a plain filename."
        ),
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """
        Block path traversal. Filename must not contain / \\ .. or absolute path indicators.
        """
        if "/" in v or "\\" in v or ".." in v or v.startswith("~"):
            raise ValueError(
                f"Filename must be a plain filename inside sample_data/. Got: '{v}'. "
                "Path traversal characters (/, \\, .., ~) are not allowed."
            )
        return v


FILE_READ_DESCRIPTION = (
    "Reads the full contents of a text file located inside the sample_data/ directory. "
    "Provide only the filename (e.g., 'notes.txt'), never a path with slashes. "
    "Returns the complete file contents as a string, or an error message if the file does not exist."
)


def execute_file_read(input_dict: dict[str, object]) -> str:
    """
    Read a file from sample_data/. Returns file contents or error message.

    Security: filename is validated against path traversal by the Pydantic schema.
    The file path is then constructed by joining a fixed base directory with the
    validated filename. No way for the model to escape sample_data/.
    """
    try:
        validated = FileReadInput(**input_dict)
    except ValidationError as e:
        return f"Error: ValidationError: {e}"

    base_dir = Path(__file__).resolve().parent.parent / "sample_data"
    file_path = base_dir / validated.filename

    if not file_path.exists():
        return f"Error: File '{validated.filename}' not found in sample_data/."
    if not file_path.is_file():
        return f"Error: '{validated.filename}' is not a regular file."

    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading file: {type(e).__name__}: {e}"
