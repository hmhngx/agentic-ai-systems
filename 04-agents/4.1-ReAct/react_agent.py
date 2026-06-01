"""
react_agent.py — Day 7 deliverable: ReAct agent from scratch via raw OpenRouter.

Zero agent frameworks (no LC/LG/LlamaIndex). Raw HTTP + Pydantic only.

Usage:
  python react_agent.py                              # runs default demo
  python react_agent.py --query "your problem"       # custom query
  python react_agent.py --max-iterations 5           # tighter cap
  python react_agent.py --quiet                      # suppress trace
  python react_agent.py --mock-web                   # LLM-simulated search (no DuckDuckGo)
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _apply_early_cli_flags() -> None:
    """Set REACT_WEB_MODE before importing src (registry picks tools at import time)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mock-web", action="store_true")
    pre_args, _ = pre.parse_known_args()
    if pre_args.mock_web:
        os.environ["REACT_WEB_MODE"] = "mock"


_apply_early_cli_flags()

from dotenv import load_dotenv

_preserve_empty_api_key = os.environ.get("OPENROUTER_API_KEY") == ""
load_dotenv(override=True)
if _preserve_empty_api_key:
    os.environ["OPENROUTER_API_KEY"] = ""

from src.react_loop import MAX_ITERATIONS, run_agent
from src.tool_registry import list_tool_names, use_web_mock

DEFAULT_DEMO_QUERY = (
    "Read sample_data/notes.txt to find the total revenue and headcount, "
    "then calculate the average revenue per employee."
)


def _positive_int_max_10(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("max-iterations must be at least 1")
    if parsed > MAX_ITERATIONS:
        raise argparse.ArgumentTypeError(
            f"max-iterations cannot exceed {MAX_ITERATIONS} (hard cap enforced in code)"
        )
    return parsed


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "Copy .env.example to .env and add your OpenRouter API key.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="ReAct agent — raw OpenRouter, zero frameworks",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_DEMO_QUERY,
        help="Problem for the agent to solve (default: demo query)",
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_int_max_10,
        default=MAX_ITERATIONS,
        help=f"Maximum ReAct iterations (1-{MAX_ITERATIONS}, hard cap in code)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress reasoning trace; print only final answer",
    )
    parser.add_argument(
        "--mock-web",
        action="store_true",
        help="Use web_mock (LLM simulated search) instead of real web_search",
    )
    args = parser.parse_args()

    model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
    max_tokens = os.environ.get("OPENROUTER_MAX_TOKENS", "128")
    print(f"Model: {model} | Max iterations: {args.max_iterations} | Max tokens: {max_tokens}")

    print("════════════════════════════════════════════════")
    print("  ReAct Agent")
    print(f"  Problem: {args.query}")
    web_mode = "web_mock (simulated)" if use_web_mock() else "web_search (DuckDuckGo)"
    print(f"  Tools: {', '.join(list_tool_names())}")
    print(f"  Web: {web_mode}")
    print("════════════════════════════════════════════════")
    print()

    result = run_agent(
        query=args.query,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )

    if result.api_fatal:
        print("OpenRouter API error (auth/billing). Exit code 3.")
        sys.exit(3)

    if result.hard_stopped:
        print("Agent did not converge. Exit code 2.")
        sys.exit(2)

    print(f"Agent solved the problem in {result.iterations_used} iterations. Exit 0.")
    sys.exit(0)


if __name__ == "__main__":
    main()
