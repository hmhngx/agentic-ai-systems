"""
hallucination_detector.py - Day 10 deliverable: a groundedness guardrail.

Wraps ANY RAG answer and returns a groundedness score (0-1) + a verdict, then
refuses (serves a fallback) when the answer cannot be trusted. The scoring core
is deterministic and offline; an optional OpenRouter LLM judge (USE_LLM=1) can
replace the string-overlap support signal.

The 3-check pattern:
    Check 1  retrieved?  did the answer actually USE the retrieved chunks?
    Check 2  cited?      are the [Doc N] citations REAL chunk IDs?
    Check 3  faithful?   refuse if the composed confidence score is < 0.5.

Usage:
    # Prove the Day-10 goal: 5 absent-context questions all fall back.
    python hallucination_detector.py --demo

    # Score an ad-hoc (question, answer, chunks) triple.
    python hallucination_detector.py \\
        --question "What is the capital of Australia?" \\
        --answer "The capital of Australia is Canberra [Doc 1]." \\
        --chunk "Apollo 11 launched on July 16, 1969." \\
        --chunk "Neil Armstrong walked on the Moon in 1969."

    # Live: wrap the Day-4 naive RAG over a real PDF (needs Qdrant + key).
    python hallucination_detector.py --pdf paper.pdf --query "what is X?"

Flags: --threshold FLOAT, --use-llm, --top-k {3,4,5}, --json, --debug.
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from src.demo_corpus import ABSENT_CONTEXT_SCENARIOS, GROUNDED_CONTROL, Scenario
from src.detector import detect
from src.types import DetectorConfig, GroundednessReport, RetrievedChunk


# UTF-8 stdout/stderr on Windows so box-drawing/arrow characters render even in
# a non-UTF-8 code page. ``reconfigure`` is a no-op if already utf-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


_RULE = "+" + "-" * 64 + "+"


def _config_from_args(args: argparse.Namespace) -> DetectorConfig:
    """Build the detector config from env, then apply CLI overrides."""
    base = DetectorConfig.from_env()
    overrides: dict = {}
    if args.threshold is not None:
        overrides["threshold"] = args.threshold
    if args.use_llm:
        overrides["use_llm"] = True
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)


def _print_report(report: GroundednessReport, *, debug: bool) -> None:
    """Human-readable rendering of one report."""
    decision = "SERVED" if report.served else "FALLBACK"
    print(_RULE)
    print(f"| Q: {report.question[:60]}")
    print(_RULE)
    print(f"  Groundedness: {report.score:.2f}   "
          f"Verdict: {report.verdict.value}   -> {decision}")
    print()
    print("  Served to user:")
    print(f"    {report.served_answer}")

    if debug:
        s = report.support
        c = report.citations
        r = report.retrieval
        print()
        print(f"  [Check 1 retrieved?] method={s.method} "
              f"supported_ratio={s.supported_ratio:.2f} "
              f"mean_support={s.mean_support:.2f}")
        for claim in s.per_claim:
            mark = "OK " if claim.supported else "-- "
            print(f"      {mark}({claim.support:.2f} via {claim.best_chunk_id}) "
                  f"{claim.claim[:50]}")
        print(f"  [Check 2 cited?]     cited={list(c.cited_ids)} "
              f"valid={list(c.valid_ids)} fabricated={list(c.fabricated_ids)} "
              f"validity={c.validity:.2f}")
        print(f"  [Check 3 faithful?]  score {report.score:.2f} "
              f"{'>=' if report.served else '<'} threshold; "
              f"retrieval_adequacy={r.adequacy:.2f} (n={r.n_chunks})")

    for warning in report.warnings:
        print(f"  ! {warning}", file=sys.stderr)
    print()


def _run_demo(config: DetectorConfig, *, as_json: bool, debug: bool) -> int:
    """Score the 5 absent-context scenarios + the grounded control.

    Returns process exit code: 0 iff the Day-10 guarantee holds (all 5 absent
    scenarios fall back) AND the control is served (the detector discriminates).
    """
    scenarios: list[Scenario] = list(ABSENT_CONTEXT_SCENARIOS) + [GROUNDED_CONTROL]
    reports: list[tuple[Scenario, GroundednessReport]] = [
        (sc, detect(sc.question, sc.answer, sc.chunks, config=config))
        for sc in scenarios
    ]

    if as_json:
        print(json.dumps(
            [{"scenario": sc.name, **rep.to_dict()} for sc, rep in reports],
            indent=2,
        ))
    else:
        print("=" * 66)
        print("  HALLUCINATION DETECTOR - absent-context demo")
        print(f"  threshold={config.threshold}  support_signal="
              f"{'llm-judge' if config.use_llm else 'string-overlap'}")
        print("=" * 66)
        for sc, rep in reports:
            print(f"\n### {sc.name}")
            print(f"    why: {sc.note}")
            _print_report(rep, debug=debug)

    absent_reports = reports[: len(ABSENT_CONTEXT_SCENARIOS)]
    fell_back = sum(1 for _, rep in absent_reports if not rep.served)
    control_report = reports[-1][1]

    if not as_json:
        print("=" * 66)
        print(f"  Absent-context fallbacks: {fell_back}/"
              f"{len(absent_reports)} (target: all)")
        print(f"  Grounded control served:  {control_report.served} "
              f"(verdict {control_report.verdict.value})")
        print("=" * 66)

    goal_met = fell_back == len(absent_reports) and control_report.served
    return 0 if goal_met else 1


def _run_adhoc(args: argparse.Namespace, config: DetectorConfig) -> int:
    """Score a single (question, answer, chunks) triple supplied on the CLI."""
    chunks = [
        RetrievedChunk(chunk_id=f"Doc {i}", text=text)
        for i, text in enumerate(args.chunk or [], start=1)
    ]
    report = detect(args.question, args.answer, chunks, config=config)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report, debug=args.debug)
    return 0 if report.served else 2


def _run_live(args: argparse.Namespace, config: DetectorConfig) -> int:
    """Wrap the Day-4 naive RAG over a real PDF, then score the answer."""
    # Imported lazily: only the live path needs qdrant-client / the naive RAG.
    from src.rag_adapter import run_naive_rag_and_detect

    try:
        report = run_naive_rag_and_detect(
            args.query,
            pdf_path=args.pdf,
            top_k=args.top_k,
            reingest=args.reingest,
            config=config,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report, debug=args.debug)
    return 0 if report.served else 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hallucination_detector",
        description="Groundedness guardrail that wraps any RAG answer.",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run the 5 absent-context scenarios + control.")
    parser.add_argument("--question", type=str, default=None,
                        help="Question, for ad-hoc scoring.")
    parser.add_argument("--answer", type=str, default=None,
                        help="Answer to score, for ad-hoc scoring.")
    parser.add_argument("--chunk", action="append", default=None,
                        help="A retrieved chunk's text (repeatable).")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Live mode: PDF to ingest via the Day-4 naive RAG.")
    parser.add_argument("--query", type=str, default=None,
                        help="Live mode: question to ask the naive RAG.")
    parser.add_argument("--top-k", type=int, default=5, choices=[3, 4, 5],
                        help="Live mode: chunks to retrieve (default 5).")
    parser.add_argument("--reingest", action="store_true",
                        help="Live mode: force re-ingestion of the PDF.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the refuse-below confidence threshold.")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use the OpenRouter NLI judge for Check 1.")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    parser.add_argument("--debug", action="store_true",
                        help="Show per-check diagnostics.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    config = _config_from_args(args)

    # Mode selection. --demo is the default when nothing else is specified.
    if args.pdf is not None or args.query is not None:
        if args.pdf is None or args.query is None:
            print("ERROR: live mode needs both --pdf and --query.", file=sys.stderr)
            return 1
        return _run_live(args, config)

    if args.question is not None or args.answer is not None:
        if args.question is None or args.answer is None:
            print("ERROR: ad-hoc mode needs both --question and --answer.",
                  file=sys.stderr)
            return 1
        return _run_adhoc(args, config)

    return _run_demo(config, as_json=args.json, debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
