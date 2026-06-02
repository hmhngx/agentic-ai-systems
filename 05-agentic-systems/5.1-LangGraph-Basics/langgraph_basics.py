"""
langgraph_basics.py — Day 8: LangGraph state machines, from first principles.

This single file is a runnable, self-explaining tour of the five concepts that
everything else in LangGraph is built on. Read the docstrings top-to-bottom and
you will understand *why* the framework is shaped the way it is — not just the API.

────────────────────────────────────────────────────────────────────────────────
THE FIVE CONCEPTS (the "why", not just the "what")
────────────────────────────────────────────────────────────────────────────────

1. STATE OBJECT — the memory of the graph.
   A LangGraph graph has no hidden globals. The ONLY thing that flows between
   nodes is one shared, typed `state` object. Every node reads from it and writes
   to it; the next node sees those writes. That is the entire memory model. If a
   value is not in the state, it does not exist downstream. Typing the state
   (here: a TypedDict) makes that memory a *contract* — every field, its type,
   and how concurrent writes merge (its "reducer") is declared in one place.

2. NODES ARE PURE FUNCTIONS — `state -> partial state`.
   A node is just `f(state) -> dict`. It receives the current state and returns a
   dict of the fields it wants to change. It must NOT mutate the state in place and
   must NOT depend on anything but its inputs. Why pure?
     • Testable      — call the function with a dict, assert on the returned dict.
     • Resumable     — re-running a node from a checkpoint yields the same result.
     • Parallelizable— two pure nodes can run on the same state without races,
                       because each returns a patch and the reducers merge them.
   LangGraph *merges* the returned partial into the state (overwrite by default,
   or via a reducer like `operator.add` for fields that accumulate).

3. CONDITIONAL EDGES — routing logic.
   A normal edge is unconditional: "after A, always go to B". A *conditional* edge
   runs a small router function `state -> next_node_key` and the graph jumps to
   whichever node that key maps to. This is how a graph makes decisions: branching,
   early exit, loops, retries — all of it is "look at the state, return where to go
   next". The router only *reads* state; it never mutates it.

4. COMPILE — turning a blueprint into a runnable machine.
   `StateGraph(...)` is a mutable *builder*: you add nodes and edges to it. Calling
   `.compile()` validates the topology (every node reachable, no dangling edges,
   START/END wired) and freezes it into an immutable, executable `Pregel` object
   with `.invoke()/.stream()/.get_state()`. Compile is also where you attach a
   `checkpointer`. Before compile you have a drawing; after compile you have an engine.

5. invoke vs stream vs astream — three ways to run the same machine.
     • invoke(state, config)        → run to completion, return the FINAL state.
                                       Simplest; you only care about the answer.
     • stream(state, config)        → yield intermediate updates as each node runs.
                                       Lets you show progress / inspect every step.
     • astream(state, config)       → the async version of stream (`async for`).
                                       Same semantics, for asyncio servers where you
                                       must not block the event loop while the LLM
                                       streams tokens. (`ainvoke` is async `invoke`.)
   Same graph, same state, same checkpoints — only HOW results are delivered differs.

────────────────────────────────────────────────────────────────────────────────
THE GRAPH WE BUILD (you should be able to redraw this by hand)
────────────────────────────────────────────────────────────────────────────────

        ┌─────────┐      ┌───────────┐   route == "escalate"   ┌────────┐
  START →│  input  │ ───→ │ processor │ ───────────────────────→ │ review │─→ END
        └─────────┘      └───────────┘                          └────────┘
                                │  route == "auto"
                                ▼
                          ┌──────────┐
                          │ finalize │ ──────────────────────────────────→ END
                          └──────────┘

   • input      (node)  normalizes the raw text.
   • processor  (node)  computes word_count + sentiment, then DECIDES `route`.
   • a CONDITIONAL EDGE off `processor` dispatches on `route`:
       "escalate" → review   (negative sentiment, or long text → wants a human)
       "auto"     → finalize  (short & non-negative → auto-approve)
   • review / finalize each write the terminal `decision`, then go to END.

Run it:  python langgraph_basics.py
The demo runs FULLY OFFLINE and deterministically — no API key needed. Set
USE_LLM=1 (with OPENROUTER_API_KEY) to swap the keyword sentiment for a real
LLM call via OpenRouter; the graph topology is identical either way.
"""

from __future__ import annotations

import operator
import os
import re
import sys
from typing import Annotated

# Windows consoles default to cp1252, which cannot encode the box-drawing glyphs
# used in this demo's output. Force UTF-8 so the visuals render everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass

from typing_extensions import TypedDict

from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:  # .env is convenience-only; the offline demo runs without it.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass


# ════════════════════════════════════════════════════════════════════════════
# 1. THE STATE — the typed memory of the graph
# ════════════════════════════════════════════════════════════════════════════

# Threshold (in words) above which text is considered "long" enough to deserve a
# human look. Kept as a constant so the routing rule is named, not a magic number.
LONG_TEXT_THRESHOLD = 12


class PipelineState(TypedDict):
    """The shared memory that flows through every node.

    Every field is typed — this is the *contract* of the graph. Seven fields,
    grouped by who writes them:

      provided at invoke time
        text        the text to process (mutated in place by `input` -> normalized)
        request_id  caller-supplied correlation id, carried through untouched

      written by `processor`
        word_count  number of words after normalization
        sentiment   "positive" | "negative" | "neutral"
        route       routing decision the conditional edge dispatches on:
                    "escalate" | "auto"

      written by a leaf node (`review` or `finalize`)
        decision    the terminal outcome string

      accumulated by EVERY node (note the reducer)
        trace       Annotated[..., operator.add] — instead of OVERWRITING, each
                    node's returned `trace` list is APPENDED via operator.add.
                    This is the clearest demonstration of a "reducer": the channel
                    merges concurrent/sequential writes instead of clobbering them,
                    and it is what makes checkpoint accumulation visible below.
    """

    text: str
    request_id: str
    word_count: int
    sentiment: str
    route: str
    decision: str
    trace: Annotated[list[str], operator.add]


# ════════════════════════════════════════════════════════════════════════════
# Input validation — pydantic guards the boundary BEFORE state ever exists
# ════════════════════════════════════════════════════════════════════════════


class GraphInput(BaseModel):
    """Validate/normalize raw caller input before it becomes graph state.

    The graph's TypedDict gives *static* typing; this pydantic model gives
    *runtime* validation at the trust boundary — empty text is rejected here, not
    discovered three nodes deep. `to_state()` produces the initial PipelineState.
    """

    text: str = Field(..., min_length=1, description="Raw text to process")
    request_id: str = Field(..., min_length=1, description="Caller correlation id")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v

    def to_state(self) -> PipelineState:
        """Seed a fresh PipelineState. `trace` starts as [] so the operator.add
        reducer has an initial value to append onto."""
        return PipelineState(
            text=self.text,
            request_id=self.request_id,
            word_count=0,
            sentiment="",
            route="",
            decision="",
            trace=[],
        )


# ════════════════════════════════════════════════════════════════════════════
# Optional LLM sentiment (OpenRouter via langchain-openai) — gated, never required
# ════════════════════════════════════════════════════════════════════════════

_POSITIVE_WORDS = frozenset(
    "love great good excellent amazing awesome fantastic wonderful happy "
    "perfect best brilliant delightful superb glad pleased works".split()
)
_NEGATIVE_WORDS = frozenset(
    "hate terrible bad awful broken horrible worst angry refund disappointed "
    "useless slow buggy crash fail failed wrong annoyed frustrated".split()
)


def _keyword_sentiment(text: str) -> str:
    """Deterministic, offline sentiment via word-list scoring.

    Deterministic on purpose: the demo's state diffs must be reproducible run to
    run. An LLM would make the output vary; that is what USE_LLM=1 is for.
    """
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in _POSITIVE_WORDS for w in words)
    neg = sum(w in _NEGATIVE_WORDS for w in words)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def _llm_sentiment(text: str) -> str | None:
    """Classify sentiment with a real LLM through OpenRouter.

    Returns None (so the caller falls back to the keyword classifier) when the
    LLM path is disabled or anything goes wrong — the graph must never crash just
    because an optional dependency or network call is unavailable.
    """
    if os.environ.get("USE_LLM", "0") != "1":
        return None
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI  # imported lazily — optional path

        llm = ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
            temperature=0,
            max_tokens=4,
            timeout=30,
        )
        reply = llm.invoke(
            [
                (
                    "system",
                    "You are a sentiment classifier. Reply with exactly one word: "
                    "positive, negative, or neutral.",
                ),
                ("human", text),
            ]
        )
        label = str(reply.content).strip().lower()
        return label if label in {"positive", "negative", "neutral"} else None
    except Exception:  # noqa: BLE001 — optional path: degrade, never crash
        return None


# ════════════════════════════════════════════════════════════════════════════
# 2. THE NODES — each a pure function: state -> partial state
# ════════════════════════════════════════════════════════════════════════════


def input_node(state: PipelineState) -> dict:
    """Normalize the raw text: strip and collapse internal whitespace.

    Pure: reads state['text'], returns a patch. It does NOT touch the input dict.
    The returned `trace` is a one-element list; the operator.add reducer appends it.
    """
    normalized = re.sub(r"\s+", " ", state["text"]).strip()
    return {
        "text": normalized,
        "trace": [f"input: normalized text to {len(normalized)} chars"],
    }


def processor_node(state: PipelineState) -> dict:
    """Analyze the text and DECIDE the route. The heart of the graph.

    Computes word_count + sentiment, then derives `route` from BOTH signals:
    escalate to a human when the text is long OR the sentiment is negative;
    otherwise let it auto-approve. The conditional edge dispatches on the `route`
    this node writes — decision (here) and dispatch (the edge) are kept separate.
    """
    text = state["text"]
    word_count = len(text.split())
    sentiment = _llm_sentiment(text) or _keyword_sentiment(text)

    if word_count >= LONG_TEXT_THRESHOLD or sentiment == "negative":
        route = "escalate"
        reason = "long" if word_count >= LONG_TEXT_THRESHOLD else "negative sentiment"
    else:
        route = "auto"
        reason = "short & non-negative"

    return {
        "word_count": word_count,
        "sentiment": sentiment,
        "route": route,
        "trace": [
            f"processor: {word_count} words, sentiment={sentiment} "
            f"-> route={route} ({reason})"
        ],
    }


def review_node(state: PipelineState) -> dict:
    """Leaf node for the 'escalate' branch: hand off to a (notional) human."""
    return {
        "decision": f"ESCALATED for human review (request {state['request_id']})",
        "trace": ["review: queued for human review"],
    }


def finalize_node(state: PipelineState) -> dict:
    """Leaf node for the 'auto' branch: auto-approve without a human."""
    return {
        "decision": f"AUTO-APPROVED (request {state['request_id']})",
        "trace": ["finalize: auto-approved"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. THE ROUTER — a pure function: state -> next node key
# ════════════════════════════════════════════════════════════════════════════


def route_after_processor(state: PipelineState) -> str:
    """Conditional-edge function. Reads the decision `processor` already made and
    returns the routing KEY. It only reads state — routers never mutate.

    The returned key is mapped to a real node name in `add_conditional_edges`.
    """
    return state["route"]  # "escalate" or "auto"


# ════════════════════════════════════════════════════════════════════════════
# 4. COMPILE — assemble the blueprint, then freeze it into a runnable machine
# ════════════════════════════════════════════════════════════════════════════


def build_graph(checkpointer: MemorySaver | None = None):
    """Construct, wire, and COMPILE the StateGraph.

    Build phase (mutable blueprint):
      • StateGraph(PipelineState) binds the state schema + its reducers.
      • add_node registers each pure function under a name.
      • add_edge draws unconditional transitions (incl. from START / to END).
      • add_conditional_edges draws the branch: router -> {key: node} mapping.

    Compile phase (frozen engine):
      • .compile(checkpointer=...) validates topology and returns the executable
        graph. Passing a MemorySaver here is what turns on checkpointing — state
        is snapshotted after every super-step, keyed by the config's thread_id.
    """
    builder = StateGraph(PipelineState)

    # Register the two backbone nodes + the two branch leaves.
    builder.add_node("input", input_node)
    builder.add_node("processor", processor_node)
    builder.add_node("review", review_node)
    builder.add_node("finalize", finalize_node)

    # Unconditional spine: START -> input -> processor.
    builder.add_edge(START, "input")
    builder.add_edge("input", "processor")

    # THE conditional edge: after `processor`, run the router and jump to the
    # node its returned key maps to.
    builder.add_conditional_edges(
        "processor",
        route_after_processor,
        {"escalate": "review", "auto": "finalize"},
    )

    # Both leaves terminate the graph.
    builder.add_edge("review", END)
    builder.add_edge("finalize", END)

    # MemorySaver = in-process checkpoint store. Same thread_id resumes prior state.
    return builder.compile(checkpointer=checkpointer or MemorySaver())


# ════════════════════════════════════════════════════════════════════════════
# Presentation helpers — visualization + state diffing
# ════════════════════════════════════════════════════════════════════════════


def _rule(title: str = "", width: int = 78) -> str:
    if not title:
        return "─" * width
    pad = width - len(title) - 3
    return f"── {title} " + "─" * max(pad, 0)


def visualize(graph, mermaid_path: str | None = None) -> None:
    """Print the graph as ASCII + Mermaid, and optionally save the Mermaid source.

    `graph.get_graph()` returns the topology object; it can render itself as
    Mermaid (always available) or ASCII (needs the `grandalf` package).
    """
    drawable = graph.get_graph()

    print(_rule("GRAPH (ASCII)"))
    try:
        print(drawable.draw_ascii())
    except Exception as exc:  # noqa: BLE001 — grandalf missing or layout failure
        print(f"(ASCII unavailable: {exc} — `pip install grandalf`)")

    mermaid = drawable.draw_mermaid()
    print(_rule("GRAPH (Mermaid)"))
    print(mermaid.rstrip())

    if mermaid_path:
        with open(mermaid_path, "w", encoding="utf-8") as fh:
            fh.write(mermaid)
        print(f"\n(Mermaid source written to {mermaid_path} — paste into mermaid.live)")


# Fields whose values are large/structural; diffed by summary, not full repr.
_DIFF_KEYS_ORDER = ("text", "word_count", "sentiment", "route", "decision")


def print_state_diff(before: dict, after: dict) -> None:
    """Show exactly what the run changed: before -> after per field, plus the
    trace that accumulated. This is the payoff of typed state — every mutation is
    visible and attributable."""
    print(_rule("STATE DIFF (initial -> final)"))
    for key in _DIFF_KEYS_ORDER:
        old = before.get(key, "∅")
        new = after.get(key, "∅")
        if old != new:
            print(f"  {key:<11}: {old!r}  ->  {new!r}")
        else:
            print(f"  {key:<11}: {new!r}  (unchanged)")

    old_trace = before.get("trace", [])
    new_trace = after.get("trace", [])
    print(f"  {'trace':<11}: {len(old_trace)} -> {len(new_trace)} entries")
    for line in new_trace:
        print(f"               · {line}")


# ════════════════════════════════════════════════════════════════════════════
# 5. invoke / stream / checkpointing — drive the compiled machine
# ════════════════════════════════════════════════════════════════════════════


def run_one(graph, raw_text: str, request_id: str, thread_id: str) -> dict:
    """Validate -> invoke -> print the state diff. Returns the final state.

    The `config` carries the thread_id: the checkpointer namespaces all snapshots
    by it, so two invokes with the same thread_id form one continuing session.
    """
    initial = GraphInput(text=raw_text, request_id=request_id).to_state()
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + _rule(f"INVOKE  ·  thread={thread_id}  ·  {request_id}"))
    print(f"  input.text: {raw_text!r}")

    final_state = graph.invoke(initial, config)  # run to completion -> final state
    print_state_diff(initial, final_state)
    return final_state


def demo_stream(graph, raw_text: str, request_id: str, thread_id: str) -> None:
    """Same graph, run with .stream() to expose node-by-node updates.

    stream_mode='updates' yields {node_name: partial_state} after EACH node — the
    difference from invoke() is delivery, not computation. (astream is the async
    twin: `async for chunk in graph.astream(...)`.)
    """
    initial = GraphInput(text=raw_text, request_id=request_id).to_state()
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + _rule(f"STREAM  ·  thread={thread_id}  (per-node updates)"))
    for chunk in graph.stream(initial, config, stream_mode="updates"):
        for node_name, update in chunk.items():
            keys = ", ".join(k for k in update if k != "trace")
            note = update.get("trace", [""])[-1]
            print(f"  ▸ {node_name:<10} wrote [{keys}]  | {note}")


def demo_checkpointing(graph) -> None:
    """Prove the checkpointer persists state BETWEEN runs on the same thread.

    MemorySaver keeps a snapshot per thread_id. We invoke twice on one thread;
    because `trace` uses an operator.add reducer, the second run RESUMES from the
    first run's saved state and APPENDS to it — so the trace grows across runs.
    A fresh thread_id starts from nothing. That divergence is the checkpoint.
    """
    print("\n" + _rule("CHECKPOINTING  ·  state persists between runs"))
    cfg = {"configurable": {"thread_id": "session-A"}}

    first = graph.invoke(
        GraphInput(text="works great", request_id="A-turn-1").to_state(), cfg
    )
    print(f"  run 1 (session-A): trace has {len(first['trace'])} entries")

    # get_state reads the SAVED snapshot back out of the checkpointer.
    snap = graph.get_state(cfg)
    print(f"  get_state(session-A): {len(snap.values['trace'])} entries saved, "
          f"next={snap.next!r}")

    second = graph.invoke(
        GraphInput(text="still works great", request_id="A-turn-2").to_state(), cfg
    )
    print(f"  run 2 (session-A): trace GREW to {len(second['trace'])} entries "
          f"(resumed from checkpoint)")

    fresh = graph.invoke(
        GraphInput(text="brand new", request_id="B-turn-1").to_state(),
        {"configurable": {"thread_id": "session-B"}},
    )
    print(f"  run 3 (session-B): fresh thread, only {len(fresh['trace'])} entries")

    assert len(second["trace"]) > len(first["trace"]), "checkpoint did not accumulate"
    assert len(fresh["trace"]) == len(first["trace"]), "fresh thread should not inherit"
    print("  ✓ same thread accumulates; new thread starts clean — checkpointing works.")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    graph = build_graph()

    print(_rule("DAY 8 — LANGGRAPH STATE MACHINE"))
    if os.environ.get("USE_LLM") == "1" and os.environ.get("OPENROUTER_API_KEY"):
        print("sentiment engine: OpenRouter LLM (USE_LLM=1)")
    else:
        print("sentiment engine: offline keyword classifier (set USE_LLM=1 for LLM)")
    print()

    # --- Visualize the architecture --------------------------------------------
    visualize(graph, mermaid_path=os.path.join(here, "graph.mmd"))

    # --- Invoke with 3 different inputs, printing each state diff ---------------
    # Chosen to exercise BOTH branches and all three sentiments:
    cases = [
        ("I absolutely love this product, it works great!", "req-001", "demo-1"),  # +, short -> auto
        ("This app is terrible and broken, I want a refund now.", "req-002", "demo-2"),  # -, -> escalate
        (
            "The quarterly report summarizes regional performance across every "
            "market segment in considerable detail for the board.",
            "req-003",
            "demo-3",
        ),  # neutral but long -> escalate
    ]
    for raw, rid, thread in cases:
        run_one(graph, raw, rid, thread)

    # --- Show the same machine under .stream() ---------------------------------
    demo_stream(graph, "I love it, fantastic work!", "req-stream", "demo-stream")

    # --- Prove checkpoint persistence ------------------------------------------
    demo_checkpointing(graph)

    print("\n" + _rule("DONE"))


if __name__ == "__main__":
    main()
