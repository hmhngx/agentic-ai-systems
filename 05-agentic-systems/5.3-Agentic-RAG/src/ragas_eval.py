"""Run the FULL agentic pipeline over eval_dataset.json and report a RAGAS-style
table. Headline metric: mean faithfulness over served, retrieve-routed answers
(direct answers have no retrieved context to be faithful to; fallbacks abstain).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.graph import run_agent

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "eval_dataset.json"


def load_dataset(path: "Path | str" = _DEFAULT_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_dataset(path: "Path | str" = _DEFAULT_PATH) -> dict:
    data = load_dataset(path)
    rows: list[dict] = []
    faiths: list[float] = []
    route_hits = 0
    for item in data:
        final = run_agent(item["question"])
        reflexion_fired = len(final["attempts_log"]) > 1
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_route": item["expected_route"],
            "route": final["route"],
            "status": final["status"],
            "served": final["served"],
            "faithfulness": round(float(final["faithfulness"]), 3),
            "reflexion_fired": reflexion_fired,
            "attempts": len(final["attempts_log"]),
        }
        rows.append(row)
        if final["route"] == item["expected_route"]:
            route_hits += 1
        if final["status"] == "answered":      # served, retrieve-routed answers only
            faiths.append(float(final["faithfulness"]))

    n_fallback = sum(1 for r in rows if r["status"] == "fallback")
    n_direct = sum(1 for r in rows if r["status"] == "direct")
    return {
        "rows": rows,
        "mean_faithfulness": round(sum(faiths) / len(faiths), 3) if faiths else 0.0,
        "route_accuracy": round(route_hits / len(rows), 3) if rows else 0.0,
        "n_answered": len(faiths),
        "n_direct": n_direct,
        "n_fallback": n_fallback,
        "reflexion_count": sum(1 for r in rows if r["reflexion_fired"]),
    }


def format_table(metrics: dict) -> str:
    lines = []
    header = f"{'id':<4} {'route':<9} {'status':<9} {'faith':>6} {'refl':>5} {'tries':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in metrics["rows"]:
        lines.append(f"{r['id']:<4} {r['route']:<9} {r['status']:<9} "
                     f"{r['faithfulness']:>6.2f} {('Y' if r['reflexion_fired'] else '.'):>5} "
                     f"{r['attempts']:>5}")
    lines.append("-" * len(header))
    lines.append(f"mean faithfulness (served retrieve answers): {metrics['mean_faithfulness']:.3f}  "
                 f"(target >= 0.75)")
    lines.append(f"route accuracy: {metrics['route_accuracy']:.3f}  | "
                 f"answered={metrics['n_answered']} direct={metrics['n_direct']} "
                 f"fallback={metrics['n_fallback']} reflexion={metrics['reflexion_count']}")
    return "\n".join(lines)


def main() -> None:
    print(format_table(evaluate_dataset()))


if __name__ == "__main__":
    main()
