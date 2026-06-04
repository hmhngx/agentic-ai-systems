"""Reflexion: rewrite the search query after an ungrounded attempt.

Offline: a generic casual->formal SYNONYM_MAP bridges the vocabulary gap that
defeats tf-idf retrieval (the documented root cause). It is NOT keyed to answers
— it maps everyday verbs to the formal register documentation tends to use. The
refined query is guaranteed to differ from the last one so the loop progresses;
when no synonym applies it broadens with a generic widening token. With USE_LLM=1
an LLM rewrites the query instead, conditioning on the recorded failures.
"""
from __future__ import annotations

import os

from src import config
from src.text import content_tokens

SYNONYM_MAP: dict[str, list[str]] = {
    "keep": ["retained", "retention"], "kept": ["retained", "retention"],
    "store": ["retained", "retention"], "stored": ["retained", "retention"],
    "hold": ["retained", "retention"], "retain": ["retention"],
    "delete": ["purged"], "deleting": ["purged"], "deleted": ["purged"],
    "remove": ["purged"], "erase": ["purged"], "expire": ["purged"],
    "cost": ["pricing", "billed"], "costs": ["pricing", "billed"],
    "price": ["pricing", "billed"], "fee": ["pricing", "billed"],
    "expensive": ["pricing", "billed"], "much": ["pricing"],
    "limit": ["quota"], "cap": ["quota"], "allowance": ["quota"],
    "allow": ["quota"], "allows": ["quota"], "many": ["quota"],
    "embed": ["embedding", "model"], "vector": ["embedding", "vectors"],
}
_WIDEN = "details overview"   # generic broadening when no synonym applies


def refine(question: str, attempts_log: list[dict]) -> str:
    if config.use_llm():
        return _refine_llm(question, attempts_log)

    terms = content_tokens(question)
    expanded: list[str] = list(terms)
    added = False
    for t in terms:
        for syn in SYNONYM_MAP.get(t, []):
            if syn not in expanded:
                expanded.append(syn)
                added = True

    refined = " ".join(expanded)
    last = attempts_log[-1]["query"] if attempts_log else question
    if not added or refined == last:
        # guarantee progress even when no synonym helps
        refined = f"{refined} {_WIDEN}".strip()
    return refined


def _refine_llm(question: str, attempts_log: list[dict]) -> str:
    from langchain_openai import ChatOpenAI

    history = "\n".join(
        f"- tried '{a['query']}' -> faithfulness {a.get('faithfulness'):.2f} "
        f"({a.get('diagnosis')})" for a in attempts_log)
    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=os.environ["OPENROUTER_API_KEY"],
                     temperature=0.2, max_tokens=40, timeout=30)
    system = ("Rewrite the user's question into a better keyword search query that "
              "would retrieve grounding evidence. Avoid the failed queries below. "
              "Reply with ONLY the query.\n" + history)
    return str(llm.invoke([("system", system), ("human", question)]).content).strip()
