"""
test_langgraph_basics.py — brutally rigorous, machine-checked validation that
langgraph_basics.py meets EVERY Day-8 goal. Each test class maps to one goal so a
green run is literal proof, not a vibe.

    Goal checklist (from the assignment):
      [G1] 2-node graph: Input -> Processor -> End, compiles and runs
      [G2] Typed state with >= 4 fields
      [G3] A conditional edge that routes based on state
      [G4] MemorySaver checkpointing (state saved between runs)
      [G5] Visualize the graph (ascii AND mermaid)
      [G6] Invoke with 3 different inputs, observe distinct state diffs

It also enforces the three "Common Mistakes" the source notes warn about:
      [M1] nodes/routers must be PURE (never mutate the input state)
      [M2] invoking a checkpointed graph without a thread_id must error
      [M3] the accumulating field must use a reducer (append, not overwrite)

Run:  python -m pytest -v
"""

from __future__ import annotations

import operator
import typing

import pytest

from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver

import langgraph_basics as lgb
from langgraph_basics import (
    GraphInput,
    PipelineState,
    build_graph,
    input_node,
    processor_node,
    review_node,
    finalize_node,
    route_after_processor,
    LONG_TEXT_THRESHOLD,
)


# Fresh compiled graph per test → no checkpoint bleed-through between tests.
@pytest.fixture()
def graph():
    return build_graph()


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ════════════════════════════════════════════════════════════════════════════
# [G1] 2-node spine compiles and runs: START -> input -> processor -> ... -> END
# ════════════════════════════════════════════════════════════════════════════
class TestG1_CompilesAndRuns:
    def test_compile_returns_executable(self, graph):
        # A compiled graph exposes the execution interface; a blueprint does not.
        for method in ("invoke", "stream", "get_state", "get_graph"):
            assert hasattr(graph, method), f"compiled graph missing .{method}()"

    def test_backbone_nodes_present(self, graph):
        names = set(graph.get_graph().nodes.keys())
        assert {"input", "processor"} <= names, "missing the Input->Processor spine"

    def test_spine_edges_are_unconditional(self, graph):
        edges = graph.get_graph().edges
        pairs = {(e.source, e.target): e.conditional for e in edges}
        assert pairs.get((START, "input")) is False, "START must flow into input"
        assert pairs.get(("input", "processor")) is False, "input must flow into processor"

    def test_both_branches_terminate_at_end(self, graph):
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("finalize", END) in edges
        assert ("review", END) in edges

    def test_runs_end_to_end(self, graph):
        out = graph.invoke(
            GraphInput(text="works great", request_id="r").to_state(), _cfg("g1")
        )
        assert out["decision"], "graph produced no terminal decision"


# ════════════════════════════════════════════════════════════════════════════
# [G2] Typed state with >= 4 fields
# ════════════════════════════════════════════════════════════════════════════
class TestG2_TypedState:
    def test_state_has_at_least_four_typed_fields(self):
        hints = typing.get_type_hints(PipelineState, include_extras=True)
        assert len(hints) >= 4, f"need >= 4 typed fields, found {len(hints)}"
        # The fields the goals + routing actually rely on must all be declared.
        for field in ("text", "request_id", "word_count", "sentiment", "route", "decision", "trace"):
            assert field in hints, f"state missing typed field: {field}"

    def test_pydantic_input_rejects_blank_text(self):
        with pytest.raises(Exception):
            GraphInput(text="   ", request_id="r")
        with pytest.raises(Exception):
            GraphInput(text="ok", request_id="")

    def test_to_state_seeds_a_valid_initial_state(self):
        s = GraphInput(text="hello world", request_id="r").to_state()
        assert s["trace"] == [], "trace must start empty so the reducer can append"
        assert s["word_count"] == 0 and s["decision"] == ""


# ════════════════════════════════════════════════════════════════════════════
# [G3] Conditional edge that routes based on state
# ════════════════════════════════════════════════════════════════════════════
class TestG3_ConditionalRouting:
    def test_a_conditional_edge_exists_off_processor(self, graph):
        conds = [e for e in graph.get_graph().edges if e.conditional and e.source == "processor"]
        targets = {e.target for e in conds}
        assert targets == {"review", "finalize"}, f"unexpected branch targets: {targets}"

    def test_positive_short_routes_to_auto_finalize(self, graph):
        out = graph.invoke(
            GraphInput(text="I love this, it works great!", request_id="r").to_state(),
            _cfg("g3a"),
        )
        assert out["sentiment"] == "positive"
        assert out["route"] == "auto"
        assert "AUTO-APPROVED" in out["decision"]

    def test_negative_routes_to_escalate_review(self, graph):
        out = graph.invoke(
            GraphInput(text="terrible and broken, I want a refund", request_id="r").to_state(),
            _cfg("g3b"),
        )
        assert out["sentiment"] == "negative"
        assert out["route"] == "escalate"
        assert "ESCALATED" in out["decision"]

    def test_long_neutral_routes_to_escalate_by_length(self, graph):
        long_text = " ".join(["word"] * (LONG_TEXT_THRESHOLD + 2))
        out = graph.invoke(
            GraphInput(text=long_text, request_id="r").to_state(), _cfg("g3c")
        )
        assert out["word_count"] >= LONG_TEXT_THRESHOLD
        assert out["route"] == "escalate", "long text should escalate regardless of sentiment"

    def test_router_returns_a_string_key(self):
        key = route_after_processor({"route": "auto"})  # type: ignore[arg-type]
        assert key in {"auto", "escalate"} and isinstance(key, str)


# ════════════════════════════════════════════════════════════════════════════
# [G4] MemorySaver checkpointing — state persists between runs
# ════════════════════════════════════════════════════════════════════════════
class TestG4_Checkpointing:
    def test_compiled_with_memorysaver(self, graph):
        assert isinstance(graph.checkpointer, MemorySaver)

    def test_same_thread_accumulates_state(self, graph):
        c = _cfg("session-A")
        first = graph.invoke(GraphInput(text="works great", request_id="1").to_state(), c)
        second = graph.invoke(GraphInput(text="works great again", request_id="2").to_state(), c)
        assert len(second["trace"]) > len(first["trace"]), (
            "second run on same thread must RESUME from checkpoint and grow the trace"
        )

    def test_fresh_thread_starts_clean(self, graph):
        graph.invoke(GraphInput(text="works great", request_id="1").to_state(), _cfg("session-A"))
        fresh = graph.invoke(
            GraphInput(text="brand new", request_id="2").to_state(), _cfg("session-B")
        )
        assert len(fresh["trace"]) == 3, "a new thread_id must not inherit another thread's state"

    def test_get_state_reads_back_the_saved_snapshot(self, graph):
        c = _cfg("session-A")
        graph.invoke(GraphInput(text="works great", request_id="1").to_state(), c)
        snap = graph.get_state(c)
        assert snap.values["decision"], "checkpointer did not persist the final state"
        assert snap.next == (), "a completed run should have no pending next node"


# ════════════════════════════════════════════════════════════════════════════
# [G5] Visualization — ascii AND mermaid
# ════════════════════════════════════════════════════════════════════════════
class TestG5_Visualization:
    def test_mermaid_contains_all_nodes_and_the_branch(self, graph):
        m = graph.get_graph().draw_mermaid()
        for node in ("input", "processor", "review", "finalize"):
            assert node in m, f"mermaid missing node {node}"
        assert "escalate" in m and "auto" in m, "mermaid missing the conditional branch labels"

    def test_ascii_renders(self, graph):
        ascii_art = graph.get_graph().draw_ascii()  # requires grandalf
        assert "input" in ascii_art and "processor" in ascii_art


# ════════════════════════════════════════════════════════════════════════════
# [G6] Three distinct inputs produce three distinct state diffs, deterministically
# ════════════════════════════════════════════════════════════════════════════
class TestG6_ThreeInputsDistinctDiffs:
    CASES = [
        ("I absolutely love this product, it works great!", "positive", "auto"),
        ("This app is terrible and broken, I want a refund now.", "negative", "escalate"),
        (
            "The quarterly report summarizes regional performance across every "
            "market segment in considerable detail for the board.",
            "neutral",
            "escalate",
        ),
    ]

    def test_each_case_has_expected_outcome(self, graph):
        seen_routes = set()
        for i, (text, sentiment, route) in enumerate(self.CASES):
            out = graph.invoke(GraphInput(text=text, request_id=f"r{i}").to_state(), _cfg(f"g6-{i}"))
            assert out["sentiment"] == sentiment, f"case {i}: sentiment"
            assert out["route"] == route, f"case {i}: route"
            seen_routes.add(route)
        assert seen_routes == {"auto", "escalate"}, "the 3 inputs must exercise BOTH branches"

    def test_results_are_deterministic(self, graph):
        # Same input on two fresh threads → identical computed fields (offline path).
        text = "I love it, fantastic work!"
        a = graph.invoke(GraphInput(text=text, request_id="x").to_state(), _cfg("det-1"))
        b = graph.invoke(GraphInput(text=text, request_id="x").to_state(), _cfg("det-2"))
        for k in ("word_count", "sentiment", "route", "decision"):
            assert a[k] == b[k], f"non-deterministic field: {k}"


# ════════════════════════════════════════════════════════════════════════════
# [M1] Nodes and routers are PURE — they never mutate the input state
# ════════════════════════════════════════════════════════════════════════════
class TestM1_Purity:
    def test_input_node_does_not_mutate_input(self):
        state = GraphInput(text="  spaced   out  ", request_id="r").to_state()
        snapshot = dict(state)
        snapshot_trace = list(state["trace"])
        patch = input_node(state)
        assert state == {**snapshot, "trace": snapshot_trace}, "input_node mutated its input!"
        assert "text" in patch and isinstance(patch, dict)

    def test_processor_node_does_not_mutate_input(self):
        state = GraphInput(text="great great great", request_id="r").to_state()
        before = dict(state)
        processor_node(state)
        assert state == before, "processor_node mutated its input!"

    def test_leaf_nodes_return_dict_and_dont_mutate(self):
        state = GraphInput(text="x", request_id="r").to_state()
        before = dict(state)
        assert isinstance(review_node(state), dict)
        assert isinstance(finalize_node(state), dict)
        assert state == before, "a leaf node mutated its input!"

    def test_router_does_not_mutate_state(self):
        state = {"route": "auto"}
        route_after_processor(state)  # type: ignore[arg-type]
        assert state == {"route": "auto"}, "router must be read-only"


# ════════════════════════════════════════════════════════════════════════════
# [M2] Invoking a checkpointed graph without a thread_id must error
# ════════════════════════════════════════════════════════════════════════════
class TestM2_MissingThreadIdErrors:
    def test_invoke_without_config_raises(self, graph):
        with pytest.raises(Exception):
            graph.invoke(GraphInput(text="hi", request_id="r").to_state())


# ════════════════════════════════════════════════════════════════════════════
# [M3] The accumulating field uses a reducer (append, not overwrite)
# ════════════════════════════════════════════════════════════════════════════
class TestM3_ReducerSemantics:
    def test_trace_field_is_annotated_with_operator_add(self):
        hints = typing.get_type_hints(PipelineState, include_extras=True)
        meta = getattr(hints["trace"], "__metadata__", ())
        assert operator.add in meta, "trace must use the operator.add reducer to APPEND"

    def test_trace_appends_across_one_run(self, graph):
        # A single run passes through 3 nodes (input, processor, one leaf) → 3 entries.
        out = graph.invoke(GraphInput(text="works great", request_id="r").to_state(), _cfg("m3"))
        assert len(out["trace"]) == 3, "reducer should have appended one entry per node"
