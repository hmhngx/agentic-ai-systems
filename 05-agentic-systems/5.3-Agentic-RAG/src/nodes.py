"""Graph nodes + routers.

Every node returns a partial-state dict and NEVER raises — risky work is trapped
so a failure degrades into the graceful-fallback path instead of crashing the
graph. Routers are pure: they read state and return an edge key.
"""
from __future__ import annotations

import logging
import os

from src import config
from src.decision import decide
from src.faithfulness import score
from src.generator import REFUSAL
from src.rag_tool import rag_v1_complete
from src.reflexion import refine
from src.state import AgenticRAGState

log = logging.getLogger("agentic_rag")


def decide_node(state: AgenticRAGState) -> dict:
    question = state["question"]
    try:
        d = decide(question)
        return {"route": d.route, "decision_reason": d.reason, "query": question,
                "log": [f"decide: {d.route} ({d.reason})"]}
    except Exception as exc:  # noqa: BLE001 - degrade to retrieve
        return {"route": "retrieve", "decision_reason": f"decide error: {exc}",
                "query": question, "log": [f"decide: ERROR {exc} -> retrieve"]}


def retrieve_and_generate_node(state: AgenticRAGState) -> dict:
    attempt = state.get("attempt", 0) + 1
    question = state["question"]
    query = state.get("query") or question
    try:
        res = rag_v1_complete(question, query)
        return {"attempt": attempt, "chunks": res.chunks, "answer": res.answer,
                "citations": res.citations,
                "log": [f"retrieve+generate: attempt {attempt}, query={query!r}, "
                        f"{len(res.chunks)} chunk(s), conf={res.confidence:.2f}"]}
    except Exception as exc:  # noqa: BLE001 - degrade to empty retrieval
        return {"attempt": attempt, "chunks": [], "answer": REFUSAL, "citations": [],
                "log": [f"retrieve+generate: ERROR {exc} (attempt {attempt})"]}


def evaluate_node(state: AgenticRAGState) -> dict:
    question, answer, chunks = state["question"], state["answer"], state["chunks"]
    try:
        rep = score(question, answer, chunks)
    except Exception as exc:  # noqa: BLE001 - treat as worst case
        from src.faithfulness import FaithReport
        rep = FaithReport(0.0, 0, 0, [answer], "WRONG_CHUNKS")
        log.info("evaluate error: %s", exc)
    entry = {"query": state.get("query", question), "answer": answer,
             "faithfulness": rep.score, "diagnosis": rep.failure_mode,
             "unsupported": rep.unsupported_claims}
    return {"faithfulness": rep.score, "eval_report": rep.asdict(),
            "attempts_log": [entry],
            "log": [f"evaluate: faithfulness={rep.score:.2f} mode={rep.failure_mode}"]}


def refine_node(state: AgenticRAGState) -> dict:
    question = state["question"]
    try:
        nq = refine(question, state.get("attempts_log", []))
        return {"query": nq, "log": [f"refine: new query={nq!r}"]}
    except Exception as exc:  # noqa: BLE001
        return {"query": question, "log": [f"refine: ERROR {exc}"]}


def direct_node(state: AgenticRAGState) -> dict:
    question = state["question"]
    try:
        if config.use_llm():
            ans = _direct_llm(question)
        else:
            ans = ("Answered directly from the context contained in your question; "
                   "no knowledge-base retrieval was required.")
        return {"answer": ans, "final_answer": ans, "status": "direct", "served": True,
                "log": ["direct: answered without retrieval"]}
    except Exception as exc:  # noqa: BLE001
        ans = "I could not answer this directly."
        return {"answer": ans, "final_answer": ans, "status": "direct", "served": True,
                "log": [f"direct: ERROR {exc}"]}


def _best(attempts_log: list[dict]):
    return max(attempts_log, key=lambda a: a.get("faithfulness", 0.0)) if attempts_log else None


def finalize_node(state: AgenticRAGState) -> dict:
    best = _best(state.get("attempts_log", [])) or {"answer": state.get("answer", ""),
                                                     "faithfulness": state.get("faithfulness", 0.0)}
    return {"final_answer": best["answer"], "answer": best["answer"], "status": "answered",
            "served": True, "faithfulness": best["faithfulness"],
            "log": [f"finalize: served best attempt (faithfulness={best['faithfulness']:.2f})"]}


def fallback_node(state: AgenticRAGState) -> dict:
    best = _best(state.get("attempts_log", []))
    if best is None:
        msg = REFUSAL
        faith, mode = 0.0, "NO_RETRIEVAL"
    else:
        faith, mode = best["faithfulness"], best.get("diagnosis", "WRONG_CHUNKS")
        msg = (f"{REFUSAL} After {len(state.get('attempts_log', []))} attempt(s) the "
               f"best answer scored faithfulness {faith:.2f} (failure mode: {mode}), "
               f"below the {state['faithfulness_threshold']:.2f} bar — abstaining.")
    return {"final_answer": msg, "status": "fallback", "served": False,
            "faithfulness": faith,
            "eval_report": {"best_attempt": best, "failure_mode": mode},
            "log": [f"fallback: abstained (best faithfulness={faith:.2f}, mode={mode})"]}


def _direct_llm(question: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=config.OPENROUTER_MODEL, base_url=config.OPENROUTER_BASE_URL,
                     api_key=os.environ["OPENROUTER_API_KEY"],
                     temperature=0, max_tokens=300, timeout=45)
    return str(llm.invoke([("system", "Answer the question directly and concisely."),
                           ("human", question)]).content).strip()


# ---- routers (pure: read state, return an edge key) ----

def route_decision(state: AgenticRAGState) -> str:
    return state["route"]


def route_after_eval(state: AgenticRAGState) -> str:
    if state["faithfulness"] >= state["faithfulness_threshold"]:
        return "finalize"
    if state["attempt"] <= state["max_retries"]:
        return "refine"
    return "fallback"
