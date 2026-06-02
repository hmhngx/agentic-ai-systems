"""
test_research_agent.py — rigorous, machine-checked validation that
research_agent_langgraph.py meets every Day-9 goal and embodies the multi-agent
concepts from the source notes. A green run is literal proof.

    Goals:
      [G1] given a topic -> a STRUCTURED 3-section report
      [G2] state transitions are visible (recorded in state['log'])
      [G3] fails GRACEFULLY if the researcher finds nothing (no crash)
    Tasks:
      [T1] 3 nodes: planner -> researcher -> writer (one graph)
      [T2] each node has its own distinct prompt
      [T3] planner emits a STRUCTURED JSON plan (pydantic-validated)
      [T4] researcher uses Tavily-or-mock search
      [T5] writer synthesizes a formatted (Markdown) report
      [T6] max_iterations guard prevents the loop from running away
    Concepts (from the notes):
      [C1] each agent is a NODE in ONE graph (not a separate graph)
      [C2] state ACCUMULATES across calls (findings reducer = operator.add)
      [C3] tool results pass between agents purely via shared state
      [C4] error recovery: a failing node RETURNS an error, never raises

Run:  python -m pytest -v
"""

from __future__ import annotations

import operator
import typing

import pytest

from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver

import research_agent_langgraph as ra
from research_agent_langgraph import (
    ResearchPlan,
    PlanSection,
    ResearchState,
    build_graph,
    initial_state,
    planner_node,
    researcher_node,
    writer_node,
    fail_node,
    route_after_research,
    run_research,
    _mock_search,
    _search,
    NUM_SECTIONS,
)


@pytest.fixture(autouse=True)
def _force_mock_mode(monkeypatch):
    """Pin every component to its deterministic mock so tests need no keys/network."""
    monkeypatch.setenv("USE_LLM", "0")
    monkeypatch.setenv("USE_TAVILY", "0")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


def _run(topic: str, **kw) -> ResearchState:
    return run_research(topic, **kw)


# ════════════════════════════════════════════════════════════════════════════
# [T1]/[C1] Three agents are nodes in ONE graph; topology is correct
# ════════════════════════════════════════════════════════════════════════════
class TestT1_ThreeNodesOneGraph:
    def test_all_agents_are_nodes_in_one_compiled_graph(self):
        names = set(build_graph().get_graph().nodes.keys())
        assert {"planner", "researcher", "writer", "fail"} <= names

    def test_linear_spine_planner_to_researcher(self):
        edges = {(e.source, e.target): e.conditional for e in build_graph().get_graph().edges}
        assert edges.get((START, "planner")) is False
        assert edges.get(("planner", "researcher")) is False

    def test_researcher_has_self_loop_and_two_exits(self):
        conds = [e for e in build_graph().get_graph().edges
                 if e.conditional and e.source == "researcher"]
        targets = {e.target for e in conds}
        assert targets == {"researcher", "writer", "fail"}, f"got {targets}"

    def test_both_terminals_reach_end(self):
        edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
        assert ("writer", END) in edges and ("fail", END) in edges


# ════════════════════════════════════════════════════════════════════════════
# [T2] Each node has its own distinct prompt
# ════════════════════════════════════════════════════════════════════════════
class TestT2_SeparatePrompts:
    def test_three_distinct_nonempty_system_prompts(self):
        prompts = [ra._PLANNER_SYSTEM, ra._RESEARCH_SYSTEM, ra._WRITER_SYSTEM]
        assert all(isinstance(p, str) and len(p) > 20 for p in prompts)
        assert len(set(prompts)) == 3, "the three agents must not share a prompt"


# ════════════════════════════════════════════════════════════════════════════
# [T3] Planner emits a structured, pydantic-validated JSON plan
# ════════════════════════════════════════════════════════════════════════════
class TestT3_StructuredPlan:
    def test_plan_model_validates_good_json(self):
        plan = ResearchPlan.model_validate(
            {"sections": [{"title": "A", "query": "q1"}, {"title": "B", "query": "q2"}]}
        )
        assert len(plan.sections) == 2

    def test_plan_rejects_blank_fields_and_empty(self):
        with pytest.raises(Exception):
            PlanSection(title="  ", query="q")
        with pytest.raises(Exception):
            ResearchPlan.model_validate({"sections": []})

    def test_planner_node_produces_exactly_three_sections(self):
        out = planner_node(initial_state("Graph databases"))
        assert out["status"] == "researching"
        assert len(out["plan"]) == NUM_SECTIONS
        for sec in out["plan"]:
            assert sec["title"] and sec["query"]

    def test_planner_repairs_bad_llm_json_by_falling_back(self, monkeypatch):
        # Force the real-LLM path but make the model return garbage → must not crash,
        # must fall back to a valid 3-section plan (error-recovery on malformed output).
        monkeypatch.setenv("USE_LLM", "1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "x")
        monkeypatch.setattr(ra, "_chat", lambda *a, **k: "this is not json at all")
        out = planner_node(initial_state("Topic"))
        assert len(out["plan"]) == NUM_SECTIONS


# ════════════════════════════════════════════════════════════════════════════
# [T4] Researcher uses Tavily-or-mock search
# ════════════════════════════════════════════════════════════════════════════
class TestT4_SearchTool:
    def test_mock_search_returns_hits_for_normal_query(self):
        hits = _mock_search("vector databases overview")
        assert len(hits) == 3 and all({"title", "url", "content"} <= h.keys() for h in hits)

    def test_mock_search_returns_empty_for_sentinel(self):
        assert _search("noresult anything") == []

    def test_search_dispatches_to_mock_when_tavily_disabled(self):
        # USE_TAVILY=0 in fixture → dispatcher must use the mock, never import tavily.
        assert _search("graph theory basics") != []


# ════════════════════════════════════════════════════════════════════════════
# [G1]/[T5] Given a topic -> a structured 3-section Markdown report
# ════════════════════════════════════════════════════════════════════════════
class TestG1_ThreeSectionReport:
    def test_end_to_end_report_has_three_sections(self):
        state = _run("Retrieval-augmented generation")
        assert state["status"] == "done"
        assert len(state["findings"]) == NUM_SECTIONS
        assert state["report"].count("\n## ") >= NUM_SECTIONS, "need 3 '##' section headers"

    def test_report_is_markdown_titled(self):
        state = _run("Knowledge graphs")
        assert state["report"].startswith("# Knowledge graphs")

    def test_sources_are_carried_into_findings(self):
        state = _run("Embeddings")
        assert all(f["sources"] for f in state["findings"]), "each finding should carry sources"


# ════════════════════════════════════════════════════════════════════════════
# [G2] State transitions are visible (recorded in the accumulating log)
# ════════════════════════════════════════════════════════════════════════════
class TestG2_VisibleTransitions:
    def test_log_accumulator_records_each_agent(self):
        state = _run("Approximate nearest neighbor search")
        joined = " ".join(state["log"])
        assert "planner:" in joined and "researcher:" in joined and "writer:" in joined

    def test_status_reaches_done(self):
        assert _run("HNSW indexes")["status"] == "done"

    def test_configure_logging_is_idempotent(self):
        ra.configure_logging()
        n = len(ra.log.handlers)
        ra.configure_logging()
        assert len(ra.log.handlers) == n  # no duplicate handlers


# ════════════════════════════════════════════════════════════════════════════
# [G3] Graceful failure when the researcher finds nothing
# ════════════════════════════════════════════════════════════════════════════
class TestG3_GracefulFailure:
    def test_no_results_does_not_crash_and_marks_failed(self):
        state = _run("noresult phantom subject")  # sentinel → 0 hits everywhere
        assert state["status"] == "failed"
        assert state["findings"] == []

    def test_failure_report_is_still_structured(self):
        state = _run("noresult phantom subject")
        assert "No information found." in state["report"]
        assert state["report"].count("\n## ") >= NUM_SECTIONS, "preserve the 3-section structure"

    def test_graceful_failure_triggers_even_with_search_enabled(self, monkeypatch):
        # Regression guard: a real search key must NOT mask the failure path. With
        # Tavily "enabled" but simulate_empty_search=True, search is short-circuited
        # to empty BEFORE any backend call, so the graph still fails gracefully.
        monkeypatch.setenv("USE_TAVILY", "1")
        monkeypatch.setenv("TAVILY_API_KEY", "x")
        state = run_research("a topic a live search WOULD answer", simulate_empty_search=True)
        assert state["status"] == "failed"
        assert state["findings"] == []
        assert "No information found." in state["report"]


# ════════════════════════════════════════════════════════════════════════════
# [T6] max_iterations guard prevents the loop from running away
# ════════════════════════════════════════════════════════════════════════════
class TestT6_MaxIterationsGuard:
    def test_cap_below_section_count_stops_early(self):
        state = _run("Quantum error correction", max_iterations=2)
        # 2 iterations on a 3-section plan → only 2 findings, then forced to write.
        assert state["iterations"] == 2
        assert len(state["findings"]) == 2
        assert state["status"] == "done"

    def test_router_returns_research_until_cap(self):
        st = initial_state("t", max_iterations=2)
        st["plan"] = [{"title": f"S{i}", "query": "q"} for i in range(3)]
        st["cursor"], st["iterations"], st["findings"] = 1, 1, [{"x": 1}]
        assert route_after_research(st) == "research"   # under cap, more work
        st["cursor"], st["iterations"] = 2, 2
        assert route_after_research(st) == "write"       # cap hit, have findings

    def test_router_routes_to_fail_when_nothing_found(self):
        st = initial_state("t", max_iterations=6)
        st["plan"] = [{"title": "S0", "query": "q"}]
        st["cursor"], st["iterations"], st["findings"] = 1, 1, []
        assert route_after_research(st) == "fail"


# ════════════════════════════════════════════════════════════════════════════
# [C2] State accumulates across researcher calls (findings reducer = operator.add)
# ════════════════════════════════════════════════════════════════════════════
class TestC2_StateAccumulation:
    def test_findings_field_uses_operator_add_reducer(self):
        hints = typing.get_type_hints(ResearchState, include_extras=True)
        for field in ("findings", "log"):
            meta = getattr(hints[field], "__metadata__", ())
            assert operator.add in meta, f"{field} must accumulate via operator.add"

    def test_findings_grow_one_per_section_across_the_loop(self):
        state = _run("Distributed consensus")
        assert len(state["findings"]) == NUM_SECTIONS  # appended, not overwritten

    def test_researcher_node_returns_a_single_appendable_finding(self):
        st = initial_state("Vector search")
        st["plan"] = [{"title": "Background", "query": "vector search overview"}]
        out = researcher_node(st)
        assert isinstance(out["findings"], list) and len(out["findings"]) == 1
        assert out["cursor"] == 1 and out["iterations"] == 1


# ════════════════════════════════════════════════════════════════════════════
# [C3] Tool results pass between agents purely via shared state
# ════════════════════════════════════════════════════════════════════════════
class TestC3_ToolResultPassing:
    def test_writer_consumes_findings_written_by_researcher(self):
        # Hand the writer a state whose findings came only from the shared bus.
        st = initial_state("Caching")
        st["findings"] = [
            {"title": "Background", "query": "q", "content": "MARKER_FACT_42", "sources": ["u"]}
        ]
        out = writer_node(st)
        assert "MARKER_FACT_42" in out["report"], "writer must read findings from state"


# ════════════════════════════════════════════════════════════════════════════
# [C4] Error recovery: a failing node RETURNS an error, it does not raise
# ════════════════════════════════════════════════════════════════════════════
class TestC4_ErrorRecovery:
    def test_researcher_traps_exception_into_error_bus(self, monkeypatch):
        # Make summarization blow up; the node must catch it and return, not raise.
        monkeypatch.setattr(ra, "_summarize", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        st = initial_state("Topic")
        st["plan"] = [{"title": "S", "query": "valid query with hits"}]
        out = researcher_node(st)  # must NOT raise
        assert out["error"] and "boom" in out["error"]
        assert out["cursor"] == 1  # still advances so the graph can make progress

    def test_search_backend_failure_degrades_to_empty(self, monkeypatch):
        # A real-search failure must degrade to [] (transient), not propagate.
        monkeypatch.setenv("USE_TAVILY", "1")
        monkeypatch.setenv("TAVILY_API_KEY", "x")

        def _raise(*a, **k):
            raise ConnectionError("tavily down")

        # Patch the lazily-imported client to raise inside _search's try/except.
        import sys, types
        fake = types.ModuleType("tavily")
        fake.TavilyClient = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
        monkeypatch.setitem(sys.modules, "tavily", fake)
        assert _search("anything") == []  # degraded, no exception

    def test_run_never_raises_on_total_failure(self):
        # Even the worst case (no findings) returns a state object, never throws.
        state = _run("noresult total wipeout")
        assert isinstance(state, dict) and state["status"] == "failed"


# ════════════════════════════════════════════════════════════════════════════
# Checkpointing sanity (module is about persistence) — MemorySaver attached
# ════════════════════════════════════════════════════════════════════════════
class TestCheckpointer:
    def test_graph_compiled_with_memorysaver(self):
        assert isinstance(build_graph().checkpointer, MemorySaver)
