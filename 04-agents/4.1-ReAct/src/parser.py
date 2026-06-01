"""
parser.py — Extract Thought, Action, FINAL ANSWER from LLM output.

Why a parser module instead of inline regex?
  The ReAct format is:
    Thought: <reasoning>
    Action: {"tool": "name", "input": {...}}

  OR

    Thought: <reasoning>
    FINAL ANSWER: <answer>

  Models occasionally violate the format:
    - Add markdown code fences around the JSON
    - Put extra text before "Thought:"
    - Use "Final Answer" or "final answer" with different casing
    - Omit "Thought:" entirely

  The parser handles these gracefully and returns a structured result.
  If parsing fails completely, the loop injects an error Observation
  telling the model to follow the format on the next turn.
"""

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedTurn:
    """Result of parsing one LLM response."""

    thought: str  # extracted reasoning, may be empty
    is_final: bool  # True if model emitted FINAL ANSWER
    final_answer: str | None  # the answer text if is_final, else None
    action_name: str | None  # tool name if is_final=False, else None
    action_input: dict[str, object] | None  # tool input dict if is_final=False, else None
    parse_error: str | None  # error message if parsing failed, else None
    raw_output: str  # original LLM response for debugging


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that models sometimes wrap around JSON."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def parse_llm_output(text: str) -> ParsedTurn:
    """
    Parse the model's response into a ParsedTurn.

    Never raise. Always return a ParsedTurn — the loop reads parse_error and injects
    it as an Observation so the model can self-correct on the next iteration.
    """
    raw_output = text
    cleaned = _strip_code_fences(text)

    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\n(?:Action|FINAL ANSWER|$))",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = re.search(
        r"FINAL ANSWER:\s*(.+)$",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        return ParsedTurn(
            thought=thought,
            is_final=True,
            final_answer=final_match.group(1).strip(),
            action_name=None,
            action_input=None,
            parse_error=None,
            raw_output=raw_output,
        )

    action_match = re.search(
        r"Action:\s*(\{.+?\})(?=\n|$)",
        cleaned,
        re.IGNORECASE | re.DOTALL,
    )
    if not action_match:
        if not thought:
            return ParsedTurn(
                thought="",
                is_final=False,
                final_answer=None,
                action_name=None,
                action_input=None,
                parse_error="No Thought/Action/FINAL ANSWER block found in output.",
                raw_output=raw_output,
            )
        return ParsedTurn(
            thought=thought,
            is_final=False,
            final_answer=None,
            action_name=None,
            action_input=None,
            parse_error=(
                "Thought present but missing Action or FINAL ANSWER. "
                "Keep Thought to 1-2 short sentences, then on the next line output either "
                'Action: {"tool": "...", "input": {...}} or FINAL ANSWER: <answer>. '
                "If your reply was cut off, raise OPENROUTER_MAX_TOKENS in .env."
            ),
            raw_output=raw_output,
        )

    action_json_str = action_match.group(1).strip()
    try:
        action_data = json.loads(action_json_str)
    except json.JSONDecodeError as e:
        return ParsedTurn(
            thought=thought,
            is_final=False,
            final_answer=None,
            action_name=None,
            action_input=None,
            parse_error=f"Action JSON could not be parsed: {e}",
            raw_output=raw_output,
        )

    if not isinstance(action_data, dict):
        return ParsedTurn(
            thought=thought,
            is_final=False,
            final_answer=None,
            action_name=None,
            action_input=None,
            parse_error="Action JSON could not be parsed: expected a JSON object",
            raw_output=raw_output,
        )

    if "tool" not in action_data:
        return ParsedTurn(
            thought=thought,
            is_final=False,
            final_answer=None,
            action_name=None,
            action_input=None,
            parse_error="Action JSON missing required key: tool",
            raw_output=raw_output,
        )

    if "input" not in action_data:
        return ParsedTurn(
            thought=thought,
            is_final=False,
            final_answer=None,
            action_name=None,
            action_input=None,
            parse_error="Action JSON missing required key: input",
            raw_output=raw_output,
        )

    action_input = action_data["input"]
    if not isinstance(action_input, dict):
        return ParsedTurn(
            thought=thought,
            is_final=False,
            final_answer=None,
            action_name=None,
            action_input=None,
            parse_error="Action JSON missing required key: input must be a dict",
            raw_output=raw_output,
        )

    action_name = action_data["tool"]
    if not isinstance(action_name, str):
        return ParsedTurn(
            thought=thought,
            is_final=False,
            final_answer=None,
            action_name=None,
            action_input=None,
            parse_error="Action JSON missing required key: tool must be a string",
            raw_output=raw_output,
        )

    return ParsedTurn(
        thought=thought,
        is_final=False,
        final_answer=None,
        action_name=action_name,
        action_input=action_input,
        parse_error=None,
        raw_output=raw_output,
    )
