# 05 — Agentic Systems

## Objective

This module covers multi-step agent orchestration with LangGraph—state machines, directed graphs *with cycles*, checkpointing, multi-agent routing, Reflexion loops, and human-in-the-loop gates. After completing this module, you will be able to design agent workflows as explicit graphs with typed state, persist and resume execution across failures, route tasks between specialized sub-agents, and implement self-correction loops that improve output quality without human intervention.

> **Each day is a self-contained folder** (`5.x-Name/`) with its own `README.md`, runnable script, pinned `requirements.txt`, machine-checked test suite, and `.env.example`. Every artifact **runs fully offline and deterministically by default** (mock LLM/search), so the graphs, logs, and failure paths are reproducible without API keys or spend. To run one: `cd` into its folder, create a venv, `pip install -r requirements.txt`, then run the script or `python -m pytest -v`.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| LangGraph state machine | A directed graph where nodes are functions and edges are conditional transitions on shared typed state | Makes agent logic explicit, testable, and debuggable—unlike opaque prompt chains |
| Checkpointing | Persisting graph state to a store after each node, enabling resume from any point | Production agents crash; checkpointing prevents re-running expensive LLM calls from scratch |
| Multi-agent routing | A supervisor node that classifies input and delegates to specialized worker agents | Monolithic agents degrade on diverse tasks; routing keeps each agent's context window focused |
| Reflexion | Agent generates output, a critic evaluates it, failures trigger revised attempts with memory of past mistakes | Self-correction without human feedback—critical for autonomous research and code generation |
| Human-in-the-loop (HITL) | Graph interrupt points that pause execution and await human approval before continuing | Required for high-stakes actions (deployments, financial transactions, external API calls) |

## Implementation artifacts

| Day | Artifact | What it demonstrates | Status |
|-----|----------|----------------------|--------|
| 8 | [`5.1-LangGraph-Basics/`](5.1-LangGraph-Basics/README.md) | 2-node StateGraph (`input → processor`) with typed state, a conditional edge, MemorySaver checkpointing, ASCII/Mermaid visualization, and per-input state diffs | ✅ Done |
| 9 | [`5.2-Multi-Agent-Research/`](5.2-Multi-Agent-Research/README.md) | 3-agent pipeline (`planner → researcher → writer`) in one graph: structured-JSON plan, Tavily/mock search, accumulating findings, a self-looping researcher with a `max_iterations` guard, and graceful no-results failure | ✅ Done |
| 11 | [`5.3-Agentic-RAG/`](5.3-Agentic-RAG/README.md) | LangGraph agent using RAG as a tool: retrieve-vs-direct decision, a RAGAS-faithfulness-gated Reflexion loop (refine <0.70, max 2 retries), graceful fallback, and a full-pipeline RAGAS eval (mean faithfulness ≥0.75) | ✅ Done |
| 12 | `5.4-Checkpointing-HITL/` *(planned)* | Durable persistence (SqliteSaver) with `interrupt_before` approval gates and resume-from-interrupt | 🔲 Planned |
| 13 | `5.5-Supervisor-Agents/` *(planned)* | Supervisor/star topology: a router LLM delegating to specialized worker agents that report back | 🔲 Planned |

## Key decisions & tradeoffs

**Decisions made so far (5.1–5.2):**

- **StateGraph over LangChain `AgentExecutor`** for every workflow — explicit graph topology makes individual nodes unit-testable and the control flow inspectable.
- **`MemorySaver` as the checkpoint backend** for these foundational days (in-process, zero setup). Durable `SqliteSaver`/`PostgresSaver` is deferred to the persistence day (5.3).
- **Offline, deterministic by default.** Each artifact mocks the LLM and web search unless real keys are provided, so graphs, logs, and graceful-failure paths are reproducible and testable without spend.
- **Structured outputs validated with pydantic** (with a repair-retry + fallback), rather than trusting raw model JSON.
- **Day 9 uses a linear `Planner → Researcher → Writer` pipeline** (fixed task order) instead of the supervisor pattern; the supervisor/dynamic-routing approach is the subject of 5.4.

**Forward-looking (planned days):**

- The supervisor routing pattern will use structured-output classification rather than free-text routing to eliminate misrouting on ambiguous inputs.
- Reflexion loops will cap at a fixed number of revision attempts and return the best-scoring output across attempts, not the last one.
- HITL gates will sit exclusively before nodes with external side effects (API calls, file writes), never before read-only retrieval or generation.

## Verification

Every delivered artifact ships a machine-checked test suite in which each Day goal maps to an explicit assertion. Run a day's suite from its folder with `python -m pytest -v`.

| Artifact | Tests | Coverage |
|----------|-------|----------|
| 5.1 — LangGraph basics | 28 passing | all Day-8 goals + the 3 "common mistakes" (node purity, missing `thread_id`, reducer semantics) |
| 5.2 — Multi-agent research | 32 passing | all Day-9 goals/tasks/concepts incl. graceful failure under a live search backend |
| 5.3 — Agentic RAG | 68 passing | all Day-11 goals/tasks + the 4 review concepts + 3 common mistakes (bounded retries, serve-best-not-last, refusal≠hallucination) |

## References

- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- LangGraph Checkpointing guide: https://langchain-ai.github.io/langgraph/concepts/persistence/
- Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. https://arxiv.org/abs/2303.11366
- LangGraph Multi-Agent documentation: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
- LangGraph Human-in-the-Loop guide: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
