# 05 — Agentic Systems

## Objective

This module covers multi-step agent orchestration with LangGraph—state machines, directed acyclic graphs, checkpointing, multi-agent routing, Reflexion loops, and human-in-the-loop gates. After completing this module, you will be able to design agent workflows as explicit graphs with typed state, persist and resume execution across failures, route tasks between specialized sub-agents, and implement self-correction loops that improve output quality without human intervention.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| LangGraph state machine | A directed graph where nodes are functions and edges are conditional transitions on shared typed state | Makes agent logic explicit, testable, and debuggable—unlike opaque prompt chains |
| Checkpointing | Persisting graph state to a store after each node, enabling resume from any point | Production agents crash; checkpointing prevents re-running expensive LLM calls from scratch |
| Multi-agent routing | A supervisor node that classifies input and delegates to specialized worker agents | Monolithic agents degrade on diverse tasks; routing keeps each agent's context window focused |
| Reflexion | Agent generates output, a critic evaluates it, failures trigger revised attempts with memory of past mistakes | Self-correction without human feedback—critical for autonomous research and code generation |
| Human-in-the-loop (HITL) | Graph interrupt points that pause execution and await human approval before continuing | Required for high-stakes actions (deployments, financial transactions, external API calls) |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| `langgraph_workflow.py` | Core StateGraph with typed state, conditional edges, and node functions | 🔲 Pending |
| `checkpoint_store.py` | SqliteSaver checkpoint persistence with resume-from-interrupt | 🔲 Pending |
| `multi_agent_router.py` | Supervisor agent routing tasks to Researcher, Writer, and Critic sub-agents | 🔲 Pending |
| `reflexion_loop.py` | Generate-critique-revise loop with episodic memory of past failures | 🔲 Pending |
| `hitl_gate.py` | Interrupt-before-node pattern with approval/rejection routing | 🔲 Pending |

## Key decisions & tradeoffs

- LangGraph StateGraph will be used over LangChain AgentExecutor for all multi-step workflows—explicit graph topology enables unit testing of individual nodes.
- SqliteSaver will be the default checkpoint backend for local development; PostgresSaver will be documented for production deployments requiring concurrent access.
- The supervisor routing pattern will use structured output classification rather than free-text routing to eliminate misrouting on ambiguous inputs.
- Reflexion loops will cap at 3 revision attempts; the best-scoring output across attempts will be returned rather than the last attempt.
- HITL gates will be placed exclusively before nodes with external side effects (API calls, file writes)—not before read-only retrieval or generation steps.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| Task completion rate (multi-step workflows) | ≥ 90% | |
| Checkpoint recovery success rate | 100% | |
| Reflexion quality improvement (critic score delta) | ≥ +0.15 | |
| HITL gate response latency (human approval) | ≤ 30s p95 | |

## References

- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- LangGraph Checkpointing guide: https://langchain-ai.github.io/langgraph/concepts/persistence/
- Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. https://arxiv.org/abs/2303.11366
- LangGraph Multi-Agent documentation: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
- LangGraph Human-in-the-Loop guide: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
