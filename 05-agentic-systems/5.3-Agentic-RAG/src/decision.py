"""The 'should I retrieve?' gate.

Route `direct` when the question carries its own context (a self-contained
premise) or is meta/about a prior turn — retrieval would add nothing and could
inject a distractor. Otherwise route `retrieve`. Offline: a deterministic marker
heuristic. With USE_LLM=1 a classifier LLM decides instead (same Decision
contract).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src import config

DIRECT_MARKERS: tuple[str, ...] = (
    "given that", "given the", "assuming", "based on the following",
    "based on the above", "as shown above", "previous answer", "previously",
    "restate", "rephrase", "summarize your", "more concisely", "translate your",
    "you just said", "earlier you", "your last answer",
)


@dataclass
class Decision:
    route: str   # "retrieve" | "direct"
    reason: str


def decide(question: str) -> Decision:
    if config.use_llm():
        return _decide_llm(question)
    low = question.lower()
    for marker in DIRECT_MARKERS:
        if marker in low:
            return Decision("direct", f"matched self-contained/meta marker '{marker}'")
    return Decision("retrieve", "factual question with no self-contained context")


def _decide_llm(question: str) -> Decision:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=os.environ["OPENROUTER_API_KEY"],
                     temperature=0, max_tokens=5, timeout=30)
    system = ("Reply with one word: RETRIEVE if answering needs external knowledge-base "
              "facts, or DIRECT if the question is self-contained or about a prior turn.")
    verdict = str(llm.invoke([("system", system), ("human", question)]).content).strip().upper()
    route = "direct" if "DIRECT" in verdict else "retrieve"
    return Decision(route, f"LLM classifier -> {verdict}")
