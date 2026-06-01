"""
react_loop.py — The ReAct loop engine.

Why max_iterations is enforced in CODE not in PROMPT:
  Prompt-level instructions like "use at most 10 steps" are SUGGESTIONS.
  The model can ignore them, especially when it is confused or stuck.
  The HARD CAP is a safety contract enforced in the loop control structure.

  Without the cap, three failure modes cause infinite loops:
    1. Tool errors that the model retries identically forever
    2. Two tools alternating without ever converging
    3. Model misreading observations and looping back instead of concluding

  In production, an uncapped agent can burn hundreds of dollars in API costs
  before a human notices. The hard cap is non-negotiable.
"""

from dataclasses import dataclass, field

from src.grounding import check_final_answer_grounded
from src.llm_client import call_llm, is_fatal_api_error
from src.parser import parse_llm_output
from src.prompt_builder import build_conversation, truncate_for_context
from src.tool_registry import dispatch_tool
from src.tracer import (
    print_action,
    print_fatal_api_error,
    print_final_answer,
    print_hard_stop,
    print_iteration_header,
    print_observation,
    print_parse_loop_stop,
    print_thought,
)

PARSE_ERROR_STREAK_LIMIT = 3

MAX_ITERATIONS = 10  # HARD STOP — enforced in code, not just prompt
STOP_SEQUENCES = ["Observation:"]  # see llm_client docstring for why


@dataclass
class AgentResult:
    """Final result returned by the agent."""

    final_answer: str | None  # the answer text, or None if hard-stopped
    iterations_used: int  # how many iterations actually ran
    hard_stopped: bool  # True if hit max_iterations without FINAL ANSWER
    api_fatal: bool = False  # True if OpenRouter auth/billing error (401/402/403)
    trace: list[dict[str, str]] = field(default_factory=list)  # full message history


def run_agent(
    query: str,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = True,
) -> AgentResult:
    """
    Execute the ReAct loop with a hard iteration cap enforced in Python.

    The for-loop boundary `for iteration in range(1, max_iterations + 1)` is
    the hard cap. It is enforced by Python's iteration protocol, not by the model.
    Even if the model keeps requesting tools, the loop terminates at iteration 10.
    """
    history: list[dict[str, str]] = []
    iterations_used = 0
    consecutive_parse_errors = 0

    # HARD CAP: range(1, max_iterations + 1) enforces the iteration limit in code.
    # The model cannot override this — Python's for-loop terminates unconditionally.
    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration

        if verbose:
            print_iteration_header(iteration, max_iterations)

        messages = build_conversation(query, history)

        try:
            llm_output = call_llm(messages=messages, stop=STOP_SEQUENCES)
        except RuntimeError as e:
            if is_fatal_api_error(e):
                if verbose:
                    print_thought("")
                    print_fatal_api_error(str(e))
                return AgentResult(
                    final_answer=None,
                    iterations_used=iterations_used,
                    hard_stopped=False,
                    api_fatal=True,
                    trace=history,
                )
            error_observation = f"Error: RuntimeError: {e}"
            if verbose:
                print_thought("")
                print_action("llm_error", {"error": str(e)})
                print_observation(error_observation)
            history.append({"role": "assistant", "content": f"Thought: LLM call failed.\nAction: {{\"tool\": \"llm_error\", \"input\": {{}}}}"})
            history.append(
                {
                    "role": "user",
                    "content": truncate_for_context(f"Observation: {error_observation}"),
                }
            )
            continue

        parsed = parse_llm_output(llm_output)

        if verbose:
            print_thought(parsed.thought)

        if parsed.parse_error:
            consecutive_parse_errors += 1
            error_observation = (
                f"Error parsing your response: {parsed.parse_error} "
                "Please follow the format exactly."
            )
            if verbose:
                print_action("parse_error", {"error": parsed.parse_error})
                print_observation(error_observation)
            history.append({"role": "assistant", "content": parsed.raw_output})
            history.append(
                {
                    "role": "user",
                    "content": truncate_for_context(f"Observation: {error_observation}"),
                }
            )
            if consecutive_parse_errors >= PARSE_ERROR_STREAK_LIMIT:
                if verbose:
                    print_parse_loop_stop(consecutive_parse_errors)
                return AgentResult(
                    final_answer=None,
                    iterations_used=iterations_used,
                    hard_stopped=True,
                    trace=history,
                )
            continue

        consecutive_parse_errors = 0

        if parsed.is_final:
            grounded, rejection = check_final_answer_grounded(
                parsed.final_answer or "",
                history,
                user_query=query,
            )
            if not grounded and rejection:
                error_observation = rejection
                if verbose:
                    print_action("grounding_reject", {"reason": rejection})
                    print_observation(error_observation)
                history.append({"role": "assistant", "content": parsed.raw_output})
                history.append(
                    {
                        "role": "user",
                        "content": truncate_for_context(f"Observation: {error_observation}"),
                    }
                )
                continue
            if verbose:
                print_final_answer(parsed.final_answer or "", iterations_used)
            history.append({"role": "assistant", "content": parsed.raw_output})
            return AgentResult(
                final_answer=parsed.final_answer,
                iterations_used=iterations_used,
                hard_stopped=False,
                trace=history,
            )

        action_name = parsed.action_name or ""
        action_input = parsed.action_input or {}

        if verbose:
            print_action(action_name, action_input)

        observation = dispatch_tool(action_name, action_input)

        if verbose:
            print_observation(observation)

        history.append({"role": "assistant", "content": parsed.raw_output})
        history.append(
            {
                "role": "user",
                "content": truncate_for_context(f"Observation: {observation}"),
            }
        )

    if verbose:
        print_hard_stop(max_iterations)

    return AgentResult(
        final_answer=None,
        iterations_used=iterations_used,
        hard_stopped=True,
        trace=history,
    )
