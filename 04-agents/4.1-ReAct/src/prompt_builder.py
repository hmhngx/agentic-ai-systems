"""
prompt_builder.py — Construct the system prompt and conversation messages.

Why the system prompt is the most important file in this module:
  The system prompt defines the entire behavior contract. It must include:
  1. Role definition (you are a reasoning agent)
  2. The Thought/Action/Observation format spec (exact format with examples)
  3. The list of available tools with full descriptions and schemas
  4. The FINAL ANSWER escape hatch (how to conclude)
  5. The hard-stop guidance (do not call tools beyond what is needed)

  The format spec must be exact. The model follows the format it was shown.
  Vague instructions like "respond in JSON" lead to malformed output.
  Concrete examples lock in the format.
"""

import json
import os

from src.tool_registry import get_tools_description


def _use_compact_prompt() -> bool:
    return os.environ.get("OPENROUTER_COMPACT_PROMPT", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def build_system_prompt() -> str:
    """
    Returns the full system prompt with role, format spec, tools, and rules.

    The tool descriptions block is injected from get_tools_description() so
    adding a new tool automatically updates the prompt without editing this file.
    """
    tool_descriptions = get_tools_description()

    if _use_compact_prompt():
        return (
            "ReAct agent. Each turn:\n"
            "Thought: ...\n"
            'Action: {"tool":"name","input":{...}}\n'
            "OR Thought: ... / FINAL ANSWER: ...\n"
            "Stop after Action (no Observation).\n"
            f"{tool_descriptions}\n"
            "Rules: facts only from Observations; latest = max year in scoped snippets; "
            "else web_search/web_mock or FINAL ANSWER: I cannot determine this from the observations."
        )

    return f"""You are a ReAct reasoning agent. Solve problems by thinking step by step
and using tools when needed.

Use this exact response format on every turn:

Thought: <your reasoning about what to do next>
Action: {{"tool": "<tool_name>", "input": {{<schema-conforming input dict>}}}}

The Action MUST be valid JSON on a single line. After Action, stop generating —
the system will execute the tool and provide the Observation on the next turn.

When you have enough information to answer the original question, respond with:

Thought: <final reasoning>
FINAL ANSWER: <your answer>

Do NOT call any tool after FINAL ANSWER. Do NOT generate Observation yourself —
Observations come from tool execution, not your output.

{tool_descriptions}

Rules:
- Always start with Thought: (keep it to 1-2 short sentences — you must still fit Action or FINAL ANSWER in the same reply)
- Action must be valid JSON with keys "tool" and "input"
- Use FINAL ANSWER as soon as you have the answer — do not call unnecessary tools
- If a tool returns an Error, read it and adjust your input on the next turn
- Grounding: FINAL ANSWER may only include names, numbers, and dates that appear verbatim in a prior Observation (copy from the tool result — do not invent)
- If the question asks for "latest" or "most recent", use the highest year appearing in Observations and name that person/event
- If Observations do not contain enough information, call web_search (or web_mock) again with a more specific query instead of guessing
- If still insufficient, output FINAL ANSWER: I cannot determine this from the observations."""


def _max_history_turns() -> int:
    """How many prior assistant+observation pairs to send (limits prompt tokens)."""
    return int(os.environ.get("OPENROUTER_MAX_HISTORY_TURNS", "1"))


def _max_observation_chars() -> int:
    return int(os.environ.get("OPENROUTER_MAX_OBS_CHARS", "280"))


def _truncate_json_results(payload: dict[str, object], cap: int) -> str | None:
    """Shrink web_search JSON by dropping trailing results until it fits."""
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    trimmed = dict(payload)
    items = list(results)
    while items:
        trimmed["results"] = items
        encoded = json.dumps(trimmed, ensure_ascii=False)
        if len(encoded) <= cap:
            if len(items) < len(results):
                trimmed["_truncated"] = f"kept {len(items)} of {len(results)} results for context limit"
            return encoded
        items.pop()
    return json.dumps({**trimmed, "results": []}, ensure_ascii=False)


def truncate_for_context(text: str, limit: int | None = None) -> str:
    """Truncate long tool output before it is stored in LLM conversation history."""
    cap = limit if limit is not None else _max_observation_chars()
    if len(text) <= cap:
        return text

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "results" in payload:
            shrunk = _truncate_json_results(payload, cap)
            if shrunk is not None:
                return shrunk

    omitted = len(text) - cap
    return f"{text[:cap]}\n(truncated {omitted} chars for context limit)"


def trim_history_for_prompt(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the most recent turns so prompt size stays within free-tier limits."""
    max_turns = _max_history_turns()
    if max_turns <= 0:
        return []
    keep = max_turns * 2
    return history[-keep:]


def build_conversation(
    user_query: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Construct the messages list for the LLM call.

    Structure:
      [
        {"role": "system",    "content": build_system_prompt()},
        {"role": "user",      "content": user_query},
        ... history entries ...
      ]

    history is a list of {"role": ..., "content": ...} entries built up by the loop.
    Each iteration appends:
      - assistant turn: the LLM's previous Thought + Action
      - user turn:      "Observation: <tool result>"
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_query},
    ]
    messages.extend(trim_history_for_prompt(history))
    return messages
