# 04 — Agents & MCP

## Objective

This module covers LLM agents and the Model Context Protocol—the interface layer between language models and the external world. After completing this module, you will be able to implement a ReAct reasoning loop, design tool schemas that models invoke reliably, enforce structured outputs with Pydantic, and build MCP servers and clients that expose resources, tools, and prompts to any compatible host.

## Core concepts

| Concept | Mental model (one sentence) | Why it matters in production |
|---------|----------------------------|------------------------------|
| ReAct loop | Interleaving Reasoning (thought) and Acting (tool call) until the task is complete | The foundational agent pattern—every production agent framework implements this or a variant |
| Tool schema design | JSON Schema definitions that constrain what arguments the LLM can pass to each tool | Ambiguous schemas cause malformed tool calls; strict schemas with enums and descriptions reduce failure rates |
| Structured outputs | Forcing LLM responses into validated Pydantic models via tool use or response format | Eliminates regex parsing of free-text; enables type-safe downstream processing |
| MCP server | A process exposing tools, resources, and prompts over stdio or HTTP to any MCP-compatible client | Decouples capability implementation from the LLM host—write once, use in Claude Desktop, Cursor, or custom agents |
| MCP resources vs. tools | Resources are read-only data URIs; tools are callable functions with side effects | Correct separation prevents agents from "calling" data and accidentally mutating state |

## Implementation artifacts

| File | Description | Status |
|------|-------------|--------|
| `react_agent.py` | ReAct agent loop with thought-action-observation cycle and termination conditions | 🔲 Pending |
| `tool_schemas.py` | Pydantic-defined tool schemas with validation, descriptions, and enum constraints | 🔲 Pending |
| `mcp_server.py` | MCP server exposing document search, file read, and metadata tools over stdio | 🔲 Pending |
| `mcp_client.py` | MCP client that discovers and invokes tools from a running MCP server | 🔲 Pending |
| `structured_output_demo.py` | Pydantic model extraction from LLM responses via Anthropic tool use | 🔲 Pending |

## Key decisions & tradeoffs

- Tool schemas will use Pydantic models with `Field(description=...)` on every parameter—models without descriptions show measurably higher tool call failure rates.
- MCP stdio transport will be used for local tool servers; SSE/HTTP transport will be reserved for remote or containerized servers requiring network access.
- The ReAct loop will enforce a maximum of 10 iterations with a forced final-answer tool call on the last iteration to prevent infinite loops.
- Structured outputs will be implemented via Anthropic's tool use API rather than prompt-based JSON extraction—eliminating parse failures on malformed output.
- MCP resources will expose document metadata as read-only URIs; all mutations (indexing, deletion) will be exposed exclusively as tools with explicit confirmation.

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| Tool call success rate (valid JSON, correct tool) | ≥ 95% | |
| Structured output validity (Pydantic parse) | 100% | |
| Task completion rate (multi-step) | ≥ 85% | |
| MCP tool discovery latency | ≤ 500ms | |

## References

- Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629. https://arxiv.org/abs/2210.03629
- Model Context Protocol specification: https://modelcontextprotocol.io/specification/2024-11-05
- Anthropic Tool Use documentation: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
