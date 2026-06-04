"""Assemble the agentic-RAG StateGraph and compile it.

  START -> decide --(retrieve)--> retrieve_and_generate -> evaluate
                                       ^                       |
                                       | refine  <------------ route_after_eval --(finalize)--> finalize -> END
                                       |                       |--(fallback)--> fallback -> END
        decide --(direct)--> direct_answer -> END

The refine -> retrieve_and_generate back-edge makes this a Directed *Cyclic*
Graph; the `attempt <= max_retries` guard in route_after_eval keeps the cycle
finite (recursion_limit is only a backstop).
"""
from __future__ import annotations

import uuid
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from src import nodes
from src.state import (
    AgenticRAGState,
    DEFAULT_MAX_RETRIES,
    FAITHFULNESS_THRESHOLD,
    initial_state,
)


def build_graph(checkpointer: Optional[MemorySaver] = None):
    builder = StateGraph(AgenticRAGState)
    builder.add_node("decide", nodes.decide_node)
    builder.add_node("retrieve_and_generate", nodes.retrieve_and_generate_node)
    builder.add_node("evaluate", nodes.evaluate_node)
    builder.add_node("refine", nodes.refine_node)
    builder.add_node("direct_answer", nodes.direct_node)
    builder.add_node("finalize", nodes.finalize_node)
    builder.add_node("fallback", nodes.fallback_node)

    builder.add_edge(START, "decide")
    builder.add_conditional_edges("decide", nodes.route_decision,
                                  {"retrieve": "retrieve_and_generate",
                                   "direct": "direct_answer"})
    builder.add_edge("retrieve_and_generate", "evaluate")
    builder.add_conditional_edges("evaluate", nodes.route_after_eval,
                                  {"finalize": "finalize", "refine": "refine",
                                   "fallback": "fallback"})
    builder.add_edge("refine", "retrieve_and_generate")
    builder.add_edge("direct_answer", END)
    builder.add_edge("finalize", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def run_agent(
    question: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
    thread_id: Optional[str] = None,
    recursion_limit: int = 25,
) -> AgenticRAGState:
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or f"arag-{uuid.uuid4().hex[:8]}"},
              "recursion_limit": recursion_limit}
    seed = initial_state(question, max_retries, faithfulness_threshold)
    try:
        return graph.invoke(seed, config)  # type: ignore[return-value]
    except GraphRecursionError:
        return graph.get_state(config).values  # type: ignore[return-value]
