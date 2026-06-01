"""Offline ReAct loop tests — only call_llm may be mocked."""

import inspect
from unittest.mock import patch

from src import react_loop
from src.llm_client import OpenRouterAPIError
from src.react_loop import MAX_ITERATIONS, run_agent


def test_max_iterations_is_10():
    """MAX_ITERATIONS=10 is the safety contract for hard-stop behavior."""
    assert MAX_ITERATIONS == 10, (
        f"MAX_ITERATIONS must be 10, got {MAX_ITERATIONS}. "
        "This is the safety contract enforced in loop control code."
    )


def test_max_iterations_in_for_loop_range():
    """Verify the hard cap is in the FOR loop bound, not just the prompt."""
    source = inspect.getsource(react_loop.run_agent)
    assert "range(1, max_iterations" in source or "range(max_iterations" in source, (
        "run_agent MUST use range(...) over max_iterations for the hard cap. "
        "Without this, the model can request unlimited iterations via prompting."
    )


def test_loop_terminates_on_final_answer():
    """Verify loop exits as soon as model emits FINAL ANSWER."""
    with patch("src.react_loop.call_llm") as mock:
        mock.return_value = "Thought: I know this.\nFINAL ANSWER: 42"
        result = run_agent("trivial question", verbose=False)
    assert result.final_answer is not None
    assert "42" in result.final_answer
    assert result.iterations_used == 1, (
        f"Trivial question must terminate in 1 iteration, used {result.iterations_used}"
    )
    assert result.hard_stopped is False
    assert mock.call_count == 1, "Only one LLM call needed for trivial answer"


def test_loop_executes_tool_then_terminates():
    """Multi-step flow: tool action then FINAL ANSWER on the next turn."""
    responses = [
        (
            'Thought: Need to read file.\n'
            'Action: {"tool":"file_read","input":{"filename":"notes.txt"}}'
        ),
        "Thought: Got it.\nFINAL ANSWER: Done",
    ]
    call_iter = iter(responses)
    with patch(
        "src.react_loop.call_llm",
        side_effect=lambda *a, **kw: next(call_iter),
    ):
        result = run_agent("read notes", verbose=False)
    assert result.final_answer == "Done"
    assert result.iterations_used == 2


def test_loop_hard_stops_at_max_iterations():
    """Infinite tool calls must stop at MAX_ITERATIONS without FINAL ANSWER."""
    infinite_response = (
        'Thought: Keep going.\n'
        'Action: {"tool":"calculator","input":{"expression":"1+1"}}'
    )
    with patch("src.react_loop.call_llm", return_value=infinite_response):
        result = run_agent("impossible", verbose=False)
    assert result.hard_stopped is True, (
        "Without FINAL ANSWER, loop MUST hard-stop. This is the safety contract."
    )
    assert result.final_answer is None
    assert result.iterations_used == 10, (
        f"Must hit exactly MAX_ITERATIONS=10, got {result.iterations_used}"
    )


def test_loop_custom_max_iterations():
    """Custom max_iterations must cap the for-loop before MAX_ITERATIONS."""
    infinite = (
        'Thought: X.\n'
        'Action: {"tool":"calculator","input":{"expression":"1+1"}}'
    )
    with patch("src.react_loop.call_llm", return_value=infinite):
        result = run_agent("X", max_iterations=3, verbose=False)
    assert result.iterations_used == 3, (
        f"max_iterations=3 must cap at 3, got {result.iterations_used}"
    )
    assert result.hard_stopped is True


def test_parse_error_injected_as_observation():
    """Malformed LLM output must be injected as Observation for self-correction."""
    responses = [
        "I don't follow ReAct format at all.",
        "Thought: OK I'll follow format.\nFINAL ANSWER: corrected",
    ]
    call_iter = iter(responses)
    with patch(
        "src.react_loop.call_llm",
        side_effect=lambda *a, **kw: next(call_iter),
    ):
        result = run_agent("recover from parse error", verbose=False)
    assert result.final_answer == "corrected", (
        "Model must self-correct after parse error injected as Observation"
    )
    assert result.iterations_used == 2


def test_call_llm_invoked_with_stop_sequence():
    """Every call_llm invocation must pass stop=['Observation:']."""
    captured_kwargs = []

    def capture_call(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return "Thought: done.\nFINAL ANSWER: x"

    with patch("src.react_loop.call_llm", side_effect=capture_call):
        run_agent("test", verbose=False)
    assert len(captured_kwargs) >= 1
    for kw in captured_kwargs:
        stop = kw.get("stop", [])
        assert "Observation:" in stop, (
            f"call_llm must use stop=['Observation:'], got stop={stop}. "
            "Without this, model hallucinates fake observations."
        )


def test_loop_rejects_ungrounded_final_answer():
    """Hallucinated FINAL ANSWER must not exit — model gets grounding Observation."""
    responses = [
        'Thought: ok.\nAction: {"tool":"calculator","input":{"expression":"1+1"}}',
        "Thought: done.\nFINAL ANSWER: Bruno Fernandes",
        "Thought: ok.\nFINAL ANSWER: 2",
    ]
    call_iter = iter(responses)
    with patch(
        "src.react_loop.call_llm",
        side_effect=lambda *a, **kw: next(call_iter),
    ):
        result = run_agent("test", verbose=False)
    assert result.final_answer == "2"
    assert result.iterations_used == 3


def test_loop_stops_after_three_consecutive_parse_errors():
    """Repeated format failures must not burn all 10 iterations."""
    bad = "Thought: only thinking, no action line."
    with patch("src.react_loop.call_llm", return_value=bad):
        result = run_agent("stuck", verbose=False)
    assert result.hard_stopped is True
    assert result.iterations_used == 3


def test_loop_fails_fast_on_fatal_api_error():
    """402/401 must stop immediately — retrying cannot fix billing or auth."""
    with patch(
        "src.react_loop.call_llm",
        side_effect=OpenRouterAPIError(402, "insufficient credits"),
    ):
        result = run_agent("test", verbose=False)
    assert result.api_fatal is True, "Billing errors must set api_fatal"
    assert result.iterations_used == 1, "Must not burn all 10 iterations on 402"
    assert result.hard_stopped is False


def test_loop_actually_executes_calculator():
    """Loop must dispatch to real tool execution, not mocked dispatch."""
    responses = [
        (
            'Thought: Multiply.\n'
            'Action: {"tool":"calculator","input":{"expression":"7*8"}}'
        ),
        "Thought: The answer is 56.\nFINAL ANSWER: 56",
    ]
    call_iter = iter(responses)
    with patch(
        "src.react_loop.call_llm",
        side_effect=lambda *a, **kw: next(call_iter),
    ):
        result = run_agent("multiply 7 and 8", verbose=False)

    trace_text = " ".join(m["content"] for m in result.trace)
    assert "56" in trace_text, (
        "Real calculator result must appear in trace observation"
    )
    assert result.final_answer == "56"
