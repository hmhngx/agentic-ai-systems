"""
research_agent_langgraph.py — Day 9: a multi-agent research pipeline in LangGraph.

Given a topic, three specialized agents collaborate to produce a structured,
3-section report:

        ┌─────────┐     ┌────────────┐   more sections & under cap   ┌────────┐
  START →│ planner │ ──→ │ researcher │ ─────────────────────────────→ │ writer │─→ END
        └─────────┘     └────────────┘ ◀──────┐                       └────────┘
             LLM #1          LLM #2  + search  │ loop (one section / visit)   LLM #3
                                  │            │
                                  └── route_after_research(state) ──┐
                                       findings == 0  → fail ────────┴─→ ┌──────┐
                                                                         │ fail │ → END
                                                                         └──────┘

WHY THIS SHAPE (the five Day-9 concepts, grounded in this file):

  1. EACH AGENT IS A NODE, NOT A SEPARATE GRAPH.
     planner/researcher/writer are three nodes in ONE StateGraph sharing ONE
     typed state object. They never call each other and never serialize data over
     a network — they read and write the same in-process dict. (Separate graphs
     would force JSON-over-the-wire handoffs and fragment the memory.)

  2. STATE ACCUMULATES ACROSS AGENT CALLS (the "shared whiteboard" / state bus).
     `findings` and `log` are Annotated[..., operator.add] — every researcher
     visit APPENDS its result instead of overwriting. The writer later reads the
     whole accumulated `findings` list. State *is* the inter-agent communication
     bus; there are no direct agent-to-agent calls.

  3. HITL vs AUTONOMOUS LOOP.
     This pipeline is AUTONOMOUS: the researcher loops itself (one section per
     visit) until the plan is exhausted or the `max_iterations` guard trips — no
     human gate, because nothing here has a destructive side effect. The hooks to
     make it Human-in-the-Loop are documented in the README (compile with
     `interrupt_before=["writer"]` to require sign-off before publishing).

  4. TOOL RESULTS PASS BETWEEN AGENTS VIA STATE.
     The researcher runs web search (Tavily or mock), summarizes the hits, and
     writes them into `findings`. The writer consumes `findings`. The search
     result reaches the writer purely by sitting in shared state — never a direct
     handoff.

  5. ERROR RECOVERY: A NODE THAT FAILS RETURNS, IT DOES NOT RAISE.
     Every node wraps its risky work in try/except and, on failure, returns
     `{"error": ...}` into the state instead of crashing the graph. The router
     reads that error and routes to graceful failure. If the researcher finds
     nothing at all, the graph still terminates cleanly with a clear report.

RUN:  python research_agent_langgraph.py            # offline demo (mock LLM+search)
      python research_agent_langgraph.py "your topic here"

By default everything is mocked and DETERMINISTIC (no keys, no network), so the
graph topology, state transitions, and graceful-failure path are fully testable.
Provide OPENROUTER_API_KEY (+ USE_LLM=1) and/or TAVILY_API_KEY to switch the
planner/researcher-summarizer/writer and the search tool to their real backends.
"""

from __future__ import annotations

import json
import logging
import operator
import os
import re
import sys
import uuid
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

# Windows consoles default to cp1252 and choke on box-drawing glyphs; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

try:  # .env is convenience-only; the offline demo runs without it.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# ════════════════════════════════════════════════════════════════════════════
# Logging — so every state transition is VISIBLE (a Day-9 goal)
# ════════════════════════════════════════════════════════════════════════════

log = logging.getLogger("research_agent")


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent logger setup: one clean handler, node-tagged lines to stdout."""
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s │ %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(level)
    log.propagate = False


# ════════════════════════════════════════════════════════════════════════════
# Configuration — resolve which implementation each component uses, and say so
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_ITERATIONS = 6   # backstop on the researcher loop (3 sections + slack)
NUM_SECTIONS = 3             # the report always has exactly three sections


def _llm_enabled() -> bool:
    return os.environ.get("USE_LLM", "0") == "1" and bool(os.environ.get("OPENROUTER_API_KEY"))


def _search_enabled() -> bool:
    return os.environ.get("USE_TAVILY", "1") != "0" and bool(os.environ.get("TAVILY_API_KEY"))


# ════════════════════════════════════════════════════════════════════════════
# Structured plan — pydantic validates the Planner's JSON (and catches bad JSON)
# ════════════════════════════════════════════════════════════════════════════


class PlanSection(BaseModel):
    """One section of the report: a human-readable title + a web-search query."""

    title: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)

    @field_validator("title", "query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must be non-empty")
        return v


class ResearchPlan(BaseModel):
    """The Planner's structured output: exactly NUM_SECTIONS sections."""

    sections: list[PlanSection]

    @field_validator("sections")
    @classmethod
    def _exactly_n(cls, v: list[PlanSection]) -> list[PlanSection]:
        if not v:
            raise ValueError("plan must contain at least one section")
        return v


# ════════════════════════════════════════════════════════════════════════════
# THE SHARED STATE — the whiteboard / communication bus for all three agents
# ════════════════════════════════════════════════════════════════════════════


class ResearchState(TypedDict):
    """Global state for the whole team. Grouped by who writes each field.

    Reducers (Annotated[..., operator.add]) make `findings` and `log` ACCUMULATE
    — every researcher visit appends, nothing is clobbered. All other fields are
    plain overwrites that a node sets to an absolute new value.
    """

    # input
    topic: str
    max_iterations: int
    simulate_empty_search: bool   # force every search to return 0 hits (demo/test hook)
    # written by planner
    plan: list[dict]            # [{"title","query"}, ...]
    # accumulated by researcher (reducer = append)
    findings: Annotated[list[dict], operator.add]   # [{"title","query","content","sources"}]
    # control / loop bookkeeping (overwrite)
    cursor: int                 # index of the next plan section to research
    iterations: int             # researcher visits so far (loop guard)
    status: str                 # planning|researching|writing|done|failed
    error: Optional[str]        # error bus — set on failure, cleared on success
    # written by writer / fail
    report: str
    # global transcript (reducer = append) — the visible state-transition trail
    log: Annotated[list[str], operator.add]


def initial_state(
    topic: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    simulate_empty_search: bool = False,
) -> ResearchState:
    """Seed a fresh state. Reducer fields start as [] so operator.add can append."""
    return ResearchState(
        topic=topic.strip(),
        max_iterations=max_iterations,
        simulate_empty_search=simulate_empty_search,
        plan=[],
        findings=[],
        cursor=0,
        iterations=0,
        status="planning",
        error=None,
        report="",
        log=[],
    )


# ════════════════════════════════════════════════════════════════════════════
# LLM helper — one ChatOpenAI call through OpenRouter (used by the 3 LLM nodes)
# ════════════════════════════════════════════════════════════════════════════


def _chat(system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 700) -> str:
    """Single chat completion via OpenRouter. Raises on any failure so the caller
    can fall back to its mock implementation."""
    from langchain_openai import ChatOpenAI  # imported lazily — only the real path

    llm = ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=45,
    )
    reply = llm.invoke([("system", system), ("human", user)])
    return str(reply.content).strip()


def _extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of an LLM reply (handles ``` fences)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


# ════════════════════════════════════════════════════════════════════════════
# NODE 1 — PLANNER (LLM #1): topic -> structured JSON plan of 3 sections
# ════════════════════════════════════════════════════════════════════════════

_PLANNER_SYSTEM = (
    "You are a meticulous research planner. Given a topic, design a report outline "
    f"of EXACTLY {NUM_SECTIONS} complementary sections that together give a complete "
    "picture (e.g. background, current state, challenges/outlook). "
    'Reply with ONLY JSON of the form: '
    '{"sections":[{"title":"...","query":"..."}]}  '
    "where `title` is the section heading and `query` is a focused web-search query."
)


def _mock_plan(topic: str) -> ResearchPlan:
    """Deterministic offline plan. Embeds the topic so the no-results path is
    reproducible (a topic containing 'noresult' propagates into every query)."""
    return ResearchPlan(
        sections=[
            PlanSection(title="Background and Definition",
                        query=f"{topic} overview definition background"),
            PlanSection(title="Current State and Key Developments",
                        query=f"{topic} recent developments state of the art"),
            PlanSection(title="Challenges and Future Outlook",
                        query=f"{topic} challenges limitations future outlook"),
        ]
    )


def _make_plan(topic: str) -> ResearchPlan:
    """Real LLM plan with one repair retry, falling back to the mock plan."""
    if not _llm_enabled():
        return _mock_plan(topic)
    for attempt in range(2):
        try:
            raw = _chat(_PLANNER_SYSTEM, f"Topic: {topic}", temperature=0.3, max_tokens=400)
            return ResearchPlan.model_validate(_extract_json(raw))
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            log.info("PLANNER     │ JSON/validation error (attempt %d): %s", attempt + 1, exc)
    log.info("PLANNER     │ falling back to deterministic plan")
    return _mock_plan(topic)


def planner_node(state: ResearchState) -> dict:
    """LLM #1. Defensive: any failure becomes a graceful fallback, never a crash."""
    topic = state["topic"]
    log.info("PLANNER     │ status: planning │ topic=%r", topic)
    try:
        plan = _make_plan(topic)
        sections = [s.model_dump() for s in plan.sections][:NUM_SECTIONS]
        # If a (real) model under-produced, pad deterministically to NUM_SECTIONS.
        while len(sections) < NUM_SECTIONS:
            sections.append(_mock_plan(topic).sections[len(sections)].model_dump())
        titles = [s["title"] for s in sections]
        log.info("PLANNER     │ → %d sections: %s", len(sections), titles)
        return {
            "plan": sections,
            "status": "researching",
            "error": None,
            "log": [f"planner: produced {len(sections)}-section plan {titles}"],
        }
    except Exception as exc:  # noqa: BLE001 — defensive node: return, don't raise
        log.info("PLANNER     │ FATAL: %s", exc)
        return {"status": "failed", "error": f"planner failed: {exc}",
                "log": [f"planner: FATAL {exc}"]}


# ════════════════════════════════════════════════════════════════════════════
# Search tool — Tavily (real) or a deterministic mock, used by the Researcher
# ════════════════════════════════════════════════════════════════════════════

# Token that forces the mock to return zero hits — drives the graceful-failure demo.
_NO_RESULTS_TOKEN = "noresult"


def _mock_search(query: str, max_results: int = 3) -> list[dict]:
    """Deterministic offline search. Returns [] for queries containing the
    sentinel token, otherwise synthesizes plausible, query-derived hits."""
    if _NO_RESULTS_TOKEN in query.lower():
        return []
    topic = re.sub(r"\s+(overview|definition|background|recent|developments|state|of|the|"
                   r"art|challenges|limitations|future|outlook)\b", "", query, flags=re.I).strip()
    topic = topic or query
    return [
        {
            "title": f"{topic.title()} — Source {i + 1}",
            "url": f"https://example.org/{re.sub(r'[^a-z0-9]+', '-', topic.lower())}/{i + 1}",
            "content": (
                f"Authoritative overview of {topic}. Key point {i + 1}: {topic} is "
                f"characterized by well-documented properties relevant to the query "
                f"'{query}', supported by reproducible evidence and expert consensus."
            ),
        }
        for i in range(max_results)
    ]


def _search(query: str, max_results: int = 3) -> list[dict]:
    """Dispatch to Tavily when configured, else the mock. Tavily failures degrade
    to [] (a transient empty result), not an exception."""
    if not _search_enabled():
        return _mock_search(query, max_results)
    try:
        from tavily import TavilyClient  # imported lazily — only the real path

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        resp = client.search(query=query, max_results=max_results)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in resp.get("results", [])
        ]
    except Exception as exc:  # noqa: BLE001 — degrade to empty, let routing decide
        log.info("RESEARCHER  │ search backend error: %s", exc)
        return []


# ════════════════════════════════════════════════════════════════════════════
# NODE 2 — RESEARCHER (LLM #2 + search): one section per visit, loops itself
# ════════════════════════════════════════════════════════════════════════════

_RESEARCH_SYSTEM = (
    "You are a precise research analyst. Using ONLY the provided search results, "
    "write a concise, factual 3-5 sentence summary for the given report section. "
    "Do not invent facts. If the results are thin, say what is known and note the gaps."
)


def _summarize(title: str, results: list[dict]) -> str:
    """LLM #2 (or deterministic mock) — condense raw hits into a section finding."""
    snippets = "\n".join(f"- {r['title']}: {r['content']}" for r in results)
    if _llm_enabled():
        try:
            return _chat(_RESEARCH_SYSTEM, f"Section: {title}\n\nSearch results:\n{snippets}",
                         temperature=0.2, max_tokens=350)
        except Exception as exc:  # noqa: BLE001 — fall back to extractive mock
            log.info("RESEARCHER  │ summarizer LLM error: %s", exc)
    # Deterministic extractive fallback (USE_LLM=1 produces real prose instead).
    # Dedupe overlapping hit snippets, collapse whitespace, and cap the length so
    # raw search dumps stay tidy rather than repetitive.
    seen: set[str] = set()
    parts: list[str] = []
    for r in results:
        snippet = re.sub(r"\s+", " ", r.get("content", "")).strip()
        if snippet and snippet not in seen:
            seen.add(snippet)
            parts.append(snippet)
    joined = " ".join(parts)
    if len(joined) > 700:
        joined = joined[:700].rsplit(" ", 1)[0] + " …"
    return joined


def researcher_node(state: ResearchState) -> dict:
    """LLM #2 + search. Researches the section at `cursor`, appends a finding,
    advances the cursor, and increments the iteration counter (the loop guard)."""
    plan = state["plan"]
    cursor = state["cursor"]
    iteration = state["iterations"] + 1

    # Guard: nothing left to do (router normally prevents reaching here).
    if cursor >= len(plan):
        return {"iterations": iteration, "log": ["researcher: no section at cursor"]}

    section = plan[cursor]
    title, query = section["title"], section["query"]
    log.info("RESEARCHER  │ iter=%d │ section %d/%d: %r", iteration, cursor + 1, len(plan), title)

    try:
        # The simulate flag forces the "found nothing" condition deterministically,
        # independent of the search backend — this is what makes the graceful-failure
        # path demonstrable even when a real Tavily key would return hits.
        results = [] if state.get("simulate_empty_search") else _search(query, max_results=3)
        log.info("RESEARCHER  │ search %r → %d hit(s)", query, len(results))

        if not results:
            # No hits for this section: advance, record the gap, do NOT append a finding.
            return {
                "cursor": cursor + 1,
                "iterations": iteration,
                "error": None,
                "log": [f"researcher: section {cursor + 1} '{title}' → 0 results"],
            }

        content = _summarize(title, results)
        finding = {
            "title": title,
            "query": query,
            "content": content,
            "sources": [r["url"] for r in results if r.get("url")],
        }
        log.info("RESEARCHER  │ section %d wrote %d chars + %d source(s)",
                 cursor + 1, len(content), len(finding["sources"]))
        return {
            "findings": [finding],           # reducer APPENDS to the shared whiteboard
            "cursor": cursor + 1,
            "iterations": iteration,
            "error": None,
            "log": [f"researcher: section {cursor + 1} '{title}' → {len(finding['sources'])} sources"],
        }
    except Exception as exc:  # noqa: BLE001 — defensive: trap into the error bus
        log.info("RESEARCHER  │ ERROR on section %d: %s", cursor + 1, exc)
        return {
            "cursor": cursor + 1,
            "iterations": iteration,
            "error": f"researcher error on '{title}': {exc}",
            "log": [f"researcher: section {cursor + 1} ERROR {exc}"],
        }


# ════════════════════════════════════════════════════════════════════════════
# THE ROUTER — conditional edge after the researcher (loop / write / fail)
# ════════════════════════════════════════════════════════════════════════════


def route_after_research(state: ResearchState) -> str:
    """Pure routing function (reads state, returns a key — never mutates).

      "research" → loop back to the researcher: sections remain AND the
                   max_iterations guard has not tripped.
      "fail"     → graceful failure: research is finished but NOTHING was found.
      "write"    → synthesize the report from the accumulated findings.
    """
    cursor = state["cursor"]
    plan_len = len(state["plan"])
    iterations = state["iterations"]
    max_iter = state["max_iterations"]
    found = len(state["findings"])

    more_sections = cursor < plan_len
    under_cap = iterations < max_iter

    if more_sections and under_cap:
        decision = "research"
    elif found == 0:
        decision = "fail"
    else:
        decision = "write"

    if more_sections and not under_cap:
        log.info("ROUTER      │ ⚠ max_iterations guard tripped (iter=%d ≥ %d) — stopping loop",
                 iterations, max_iter)
    log.info("ROUTER      │ cursor=%d/%d iter=%d/%d findings=%d → %s",
             cursor, plan_len, iterations, max_iter, found, decision)
    return decision


# ════════════════════════════════════════════════════════════════════════════
# NODE 3 — WRITER (LLM #3): synthesize findings into a formatted 3-section report
# ════════════════════════════════════════════════════════════════════════════

_WRITER_SYSTEM = (
    "You are an expert technical writer. Using ONLY the provided research findings, "
    "write a clear, well-structured report in Markdown. Begin with a one-paragraph "
    "introduction, then one '## ' section per finding using its exact title, then a "
    "one-sentence conclusion. Do not invent facts beyond the findings."
)


def _render_report_mock(topic: str, findings: list[dict]) -> str:
    """Deterministic Markdown report from accumulated findings."""
    lines = [f"# {topic} — Research Report", ""]
    lines.append(
        f"This report synthesizes {len(findings)} researched section(s) on **{topic}**, "
        f"covering its background, current state, and outlook."
    )
    lines.append("")
    for f in findings:
        lines.append(f"## {f['title']}")
        lines.append(f["content"] or "_No substantive findings were available for this section._")
        if f.get("sources"):
            lines.append("")
            lines.append("**Sources:** " + ", ".join(f["sources"]))
        lines.append("")
    lines.append("## Conclusion")
    lines.append(
        f"The findings above outline the essential picture of {topic}; deeper analysis "
        f"would extend each section with primary sources."
    )
    return "\n".join(lines).strip()


def writer_node(state: ResearchState) -> dict:
    """LLM #3. Synthesize the accumulated findings into the final report."""
    topic = state["topic"]
    findings = state["findings"]
    log.info("WRITER      │ status: writing │ synthesizing %d finding(s)", len(findings))
    try:
        if _llm_enabled():
            payload = json.dumps({"topic": topic, "findings": findings}, ensure_ascii=False)
            report = _chat(_WRITER_SYSTEM, f"Findings JSON:\n{payload}",
                           temperature=0.4, max_tokens=900)
        else:
            report = _render_report_mock(topic, findings)
        log.info("WRITER      │ → report ready (%d chars) │ status: done", len(report))
        return {"report": report, "status": "done", "error": None,
                "log": [f"writer: wrote {len(report)}-char report from {len(findings)} findings"]}
    except Exception as exc:  # noqa: BLE001 — even the writer fails gracefully
        log.info("WRITER      │ ERROR: %s — emitting fallback report", exc)
        report = _render_report_mock(topic, findings)
        return {"report": report, "status": "done", "error": f"writer degraded: {exc}",
                "log": [f"writer: degraded ({exc}), used deterministic renderer"]}


def fail_node(state: ResearchState) -> dict:
    """Graceful terminal node when the researcher found nothing. Still emits a
    STRUCTURED report (planned headings, each noting the gap) — never a crash."""
    topic = state["topic"]
    titles = [s["title"] for s in state["plan"]] or ["Background", "Current State", "Outlook"]
    log.info("FAIL        │ status: failed │ no findings for %r — writing graceful report", topic)
    lines = [f"# {topic} — Research Report (Incomplete)", ""]
    lines.append(
        f"> ⚠️ The researcher could not find sufficient information on **{topic}**. "
        f"The intended structure is preserved below; each section awaits sources."
    )
    lines.append("")
    for t in titles:
        lines.append(f"## {t}")
        lines.append("_No information found._")
        lines.append("")
    report = "\n".join(lines).strip()
    return {"report": report, "status": "failed",
            "log": [f"fail: graceful no-results report for {topic!r}"]}


# ════════════════════════════════════════════════════════════════════════════
# COMPILE — assemble the three agents into ONE graph and freeze it
# ════════════════════════════════════════════════════════════════════════════


def build_graph(checkpointer: Optional[MemorySaver] = None):
    """Wire planner -> researcher -(loop)-> {writer|fail} -> END and compile.

    The researcher's self-loop (via the conditional edge) makes this a Directed
    *Cyclic* Graph; the max_iterations guard inside the router is what keeps the
    cycle finite. Compile attaches a MemorySaver so every super-step is
    checkpointed (and the run can be resumed/inspected)."""
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("fail", fail_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_conditional_edges(
        "researcher",
        route_after_research,
        {"research": "researcher", "write": "writer", "fail": "fail"},
    )
    builder.add_edge("writer", END)
    builder.add_edge("fail", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


# ════════════════════════════════════════════════════════════════════════════
# Visualization + orchestration
# ════════════════════════════════════════════════════════════════════════════


def _rule(title: str = "", width: int = 78) -> str:
    if not title:
        return "─" * width
    return f"── {title} " + "─" * max(width - len(title) - 4, 0)


def visualize(graph, mermaid_path: Optional[str] = None) -> None:
    drawable = graph.get_graph()
    print(_rule("GRAPH (ASCII)"))
    try:
        print(drawable.draw_ascii())
    except Exception:  # noqa: BLE001 — grandalf's layout can't place self-loops
        print("(ASCII layout can't render the researcher self-loop; see Mermaid below.)")
    print(_rule("GRAPH (Mermaid)"))
    print(drawable.draw_mermaid().rstrip())
    if mermaid_path:
        with open(mermaid_path, "w", encoding="utf-8") as fh:
            fh.write(drawable.draw_mermaid())


def run_research(
    topic: str,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    thread_id: Optional[str] = None,
    recursion_limit: int = 25,
    simulate_empty_search: bool = False,
) -> ResearchState:
    """Invoke the graph for one topic and return the final state. Logs the full
    state-transition trail; never raises on agent failure (returns failed state).

    `simulate_empty_search=True` forces every search to return nothing, so the
    graceful-failure path can be demonstrated regardless of the search backend."""
    graph = build_graph()
    config = {
        "configurable": {"thread_id": thread_id or f"research-{uuid.uuid4().hex[:8]}"},
        "recursion_limit": recursion_limit,
    }
    log.info(_rule(f"RUN │ topic={topic!r} │ max_iterations={max_iterations}"))
    search_mode = "forced-empty" if simulate_empty_search else (
        "Tavily" if _search_enabled() else "mock")
    log.info("MODE        │ LLM=%s │ search=%s",
             "OpenRouter" if _llm_enabled() else "mock", search_mode)
    try:
        final = graph.invoke(
            initial_state(topic, max_iterations, simulate_empty_search), config)
    except GraphRecursionError:
        # Backstop only — the max_iterations guard should prevent ever reaching this.
        log.info("RUN         │ recursion limit hit — returning last checkpoint")
        final = graph.get_state(config).values  # type: ignore[assignment]
    log.info("RUN         │ DONE │ status=%s │ findings=%d │ report=%d chars",
             final.get("status"), len(final.get("findings", [])), len(final.get("report", "")))
    return final  # type: ignore[return-value]


def _print_report(state: ResearchState) -> None:
    print("\n" + _rule(f"REPORT │ status={state['status']}"))
    print(state["report"])
    print(_rule())


def main(argv: Optional[list[str]] = None) -> None:
    configure_logging()
    argv = sys.argv[1:] if argv is None else argv
    here = os.path.dirname(os.path.abspath(__file__))

    print(_rule("DAY 9 — MULTI-AGENT RESEARCH PIPELINE"))
    visualize(build_graph(), mermaid_path=os.path.join(here, "graph.mmd"))
    print()

    if argv:
        state = run_research(argv[0])
        _print_report(state)
        return

    # --- Demo 1: a normal topic → full 3-section report -----------------------
    print("\n" + _rule("DEMO 1 — successful research"))
    _print_report(run_research("Vector databases for semantic search"))

    # --- Demo 2: researcher finds nothing → graceful failure ------------------
    # simulate_empty_search forces zero hits regardless of the search backend, so
    # this always exercises the failure path (a live key cannot accidentally
    # return unrelated results and mask it).
    print("\n" + _rule("DEMO 2 — graceful failure (researcher finds nothing)"))
    _print_report(run_research(
        "An impossibly obscure subject with zero coverage",
        simulate_empty_search=True,
    ))

    # --- Demo 3: tiny max_iterations → loop guard trips, partial report --------
    print("\n" + _rule("DEMO 3 — max_iterations guard (cap below section count)"))
    _print_report(run_research("Quantum error correction", max_iterations=2))


if __name__ == "__main__":
    main()
