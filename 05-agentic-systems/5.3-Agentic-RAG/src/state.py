"""The shared typed state — the communication bus for every node.

Nodes never call each other; they read and write fields on this single dict.
`attempts_log` and `log` use an operator.add reducer so a node's returned list is
APPENDED to the running value rather than overwriting it — that append-only
attempts_log is the Reflexion "verbal memory" of every prior try.

NOTE: this module intentionally does NOT use ``from __future__ import
annotations``. The ``Annotated[..., operator.add]`` reducers must stay real
typing objects at runtime so both LangGraph and ``__annotations__`` introspection
can read the reducer metadata; stringized annotations would hide it.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict

DEFAULT_MAX_RETRIES = 2          # max reflexion retries (=> <= 3 generate cycles)
FAITHFULNESS_THRESHOLD = 0.70    # per-query reflexion trigger / serve gate
DATASET_TARGET = 0.75            # headline: mean faithfulness over served retrieve answers


class AgenticRAGState(TypedDict):
    # input
    question: str
    max_retries: int
    faithfulness_threshold: float
    # decision
    route: str                   # "retrieve" | "direct"
    decision_reason: str
    # working query (refined across attempts)
    query: str
    attempt: int                 # count of retrieve+generate cycles done
    # retrieval + generation (overwritten each attempt)
    chunks: list[dict]
    answer: str
    citations: list[str]
    # evaluation (overwritten each attempt)
    faithfulness: float
    eval_report: dict
    # reflexion memory (accumulates)
    attempts_log: Annotated[list[dict], operator.add]
    # terminal
    status: str                  # "answered" | "fallback" | "direct" | ""
    final_answer: str
    served: bool
    # transcript (accumulates)
    log: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
) -> AgenticRAGState:
    """Seed a fresh state. Reducer fields start [] so operator.add can append."""
    q = question.strip()
    return AgenticRAGState(
        question=q,
        max_retries=max_retries,
        faithfulness_threshold=faithfulness_threshold,
        route="",
        decision_reason="",
        query=q,
        attempt=0,
        chunks=[],
        answer="",
        citations=[],
        faithfulness=0.0,
        eval_report={},
        attempts_log=[],
        status="",
        final_answer="",
        served=False,
        log=[],
    )
