"""
tracer.py — Visible reasoning trace printer.

The whole point of ReAct is that the reasoning is VISIBLE. The tracer prints
every Thought/Action/Observation cycle so you can see exactly what the agent is doing.

Compare to tool-calling (function calling) APIs:
  The reasoning is hidden inside the model's internal state.
  You see only the final answer.
  Debugging a failure means re-running with logging enabled.

ReAct's trace is the primary debugging surface. Make it readable.
"""

import json
import textwrap


def _prefix_lines(text: str, prefix: str = "│  ") -> str:
    """Wrap text to 80 chars and prefix each line for box-drawing output."""
    wrapped = textwrap.fill(text, width=80)
    return "\n".join(f"{prefix}{line}" for line in wrapped.splitlines())


def print_iteration_header(iteration: int, max_iterations: int) -> None:
    """Print iteration banner showing current step and hard cap."""
    print(f"═══ Iteration {iteration}/{max_iterations} ═══")


def print_thought(thought: str) -> None:
    """Print the model's reasoning block."""
    print("┌─ Thought")
    print(_prefix_lines(thought if thought else "(empty)"))
    print("└─")


def print_action(tool_name: str, input_dict: dict[str, object]) -> None:
    """Print the tool name and formatted input JSON."""
    print(f"┌─ Action: {tool_name}")
    input_json = json.dumps(input_dict, indent=2)
    for line in input_json.splitlines():
        print(f"│  {line}")
    print("└─")


def print_observation(observation: str) -> None:
    """
    Print tool result, truncated to 500 chars if longer.

    Truncation keeps the trace readable while still showing enough context
    to debug tool failures without flooding the terminal.
    """
    max_len = 500
    truncated = observation
    omitted = 0
    if len(observation) > max_len:
        truncated = observation[:max_len]
        omitted = len(observation) - max_len

    print("┌─ Observation")
    print(_prefix_lines(truncated))
    if omitted > 0:
        print(f"│  (truncated, {omitted} chars omitted)")
    print("└─")


def print_final_answer(answer: str, iterations_used: int) -> None:
    """Print the converged final answer with iteration count."""
    print("════════════════════════════════════════════════════════════")
    print(f"  FINAL ANSWER  (after {iterations_used} iterations)")
    print("════════════════════════════════════════════════════════════")
    print(answer)
    print("════════════════════════════════════════════════════════════")


def print_fatal_api_error(message: str) -> None:
    """Print a non-recoverable OpenRouter error (credits, auth) and stop immediately."""
    print("════════════════════════════════════════════════════════════")
    print("  API ERROR — cannot continue (not retried)")
    print("════════════════════════════════════════════════════════════")
    print(message[:800])
    if "402" in message or "credits" in message.lower():
        print()
        print("Fix: add credits at https://openrouter.ai/settings/credits")
        if "prompt tokens limit exceeded" in message.lower():
            print("  Prompt too large for your balance — OPENROUTER_COMPACT_PROMPT=1 (default)")
            print("  is already on; add credits or use a model with a higher free prompt limit.")
        else:
            print("  or lower OPENROUTER_MAX_TOKENS in .env below the 'can only afford N' value.")
            print("  or switch OPENROUTER_MODEL (meta-llama/llama-3.3-70b-instruct).")


def print_parse_loop_stop(streak: int) -> None:
    """Stop early when the model repeats the same parse failure."""
    print("════════════════════════════════════════════════════════════")
    print(f"  PARSE LOOP STOP — {streak} consecutive format errors")
    print("════════════════════════════════════════════════════════════")
    print("The model kept replying with Thought only (no Action / FINAL ANSWER).")
    print("Try: shorter thoughts, OPENROUTER_MAX_TOKENS=160+ in .env, or a stronger model.")


def print_hard_stop(max_iterations: int) -> None:
    """Print hard-stop message when the loop cap is reached without convergence."""
    print("════════════════════════════════════════════════════════════")
    print(f"  HARD STOP — exceeded {max_iterations} iterations")
    print("════════════════════════════════════════════════════════════")
    print("The agent did not converge. Inspect the trace above for the failure mode.")
    print("Common causes:")
    print("  - Tool returning unhelpful results, model retrying with same input")
    print("  - Model alternating between two tools without converging")
    print("  - Query is too vague for the available tools")
