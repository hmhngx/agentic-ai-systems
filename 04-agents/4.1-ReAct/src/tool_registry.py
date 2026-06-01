"""
tool_registry.py — Centralized tool registration and dispatch.

Why a registry pattern instead of if/elif chains?
  Registry pattern: register {"name": (schema, executor, description)} once.
  Adding a new tool requires only tools.py + one line here — no loop changes.
"""

import json
import os
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError

from src.tools import (
    CALCULATOR_DESCRIPTION,
    FILE_READ_DESCRIPTION,
    WEB_MOCK_DESCRIPTION,
    CalculatorInput,
    FileReadInput,
    WebMockInput,
    execute_calculator,
    execute_file_read,
    execute_web_mock,
)
from src.web_search import WEB_SEARCH_DESCRIPTION, WebSearchInput, execute_web_search


@dataclass
class ToolSpec:
    """A tool's complete specification."""

    name: str
    description: str
    input_schema: type[BaseModel]
    executor: Callable[[dict[str, object]], str]


def use_web_mock() -> bool:
    """
    REACT_WEB_MODE=mock → LLM-simulated search (offline-friendly, no DuckDuckGo).
    Default (search or unset) → real web_search tool.
    """
    return os.environ.get("REACT_WEB_MODE", "search").lower() == "mock"


def build_tools_registry() -> dict[str, ToolSpec]:
    """Build the active tool map from environment (search vs mock web)."""
    tools: dict[str, ToolSpec] = {
        "calculator": ToolSpec(
            name="calculator",
            description=CALCULATOR_DESCRIPTION,
            input_schema=CalculatorInput,
            executor=execute_calculator,
        ),
        "file_read": ToolSpec(
            name="file_read",
            description=FILE_READ_DESCRIPTION,
            input_schema=FileReadInput,
            executor=execute_file_read,
        ),
    }
    if use_web_mock():
        tools["web_mock"] = ToolSpec(
            name="web_mock",
            description=WEB_MOCK_DESCRIPTION,
            input_schema=WebMockInput,
            executor=execute_web_mock,
        )
    else:
        tools["web_search"] = ToolSpec(
            name="web_search",
            description=WEB_SEARCH_DESCRIPTION,
            input_schema=WebSearchInput,
            executor=execute_web_search,
        )
    return tools


TOOLS: dict[str, ToolSpec] = build_tools_registry()


def list_tool_names() -> list[str]:
    return list(TOOLS.keys())


def dispatch_tool(name: str, input_dict: dict[str, object]) -> str:
    """
    Look up the tool by name, validate input via Pydantic, execute, return result.
    Never raises. All errors are returned as observation strings.
    """
    if name not in TOOLS:
        return f"Error: Unknown tool '{name}'. Available tools: {list_tool_names()}"

    tool_spec = TOOLS[name]
    try:
        tool_spec.input_schema(**input_dict)
    except ValidationError as e:
        return f"Error: ValidationError: {e}"

    return tool_spec.executor(input_dict)


def _use_compact_prompt() -> bool:
    return os.environ.get("OPENROUTER_COMPACT_PROMPT", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def get_tools_description(compact: bool | None = None) -> str:
    """Build the tool descriptions block for the system prompt."""
    if compact is None:
        compact = _use_compact_prompt()

    if compact:
        parts: list[str] = []
        for tool_name, tool_spec in TOOLS.items():
            schema = tool_spec.input_schema.model_json_schema()
            required = schema.get("required", [])
            props = list(schema.get("properties", {}).keys())
            keys = ", ".join(required or props)
            parts.append(f"{tool_name}({keys})")
        return "Tools: " + ", ".join(parts)

    lines: list[str] = ["Available tools:", ""]
    for tool_name, tool_spec in TOOLS.items():
        full_schema = tool_spec.input_schema.model_json_schema()
        trimmed_schema: dict[str, object] = {}
        if "properties" in full_schema:
            trimmed_schema["properties"] = full_schema["properties"]
        if "required" in full_schema:
            trimmed_schema["required"] = full_schema["required"]

        schema_json = json.dumps(trimmed_schema, indent=2)
        lines.append(f"- {tool_name}: {tool_spec.description}")
        lines.append("  Input schema (JSON):")
        for schema_line in schema_json.splitlines():
            lines.append(f"  {schema_line}")
        lines.append("")

    return "\n".join(lines).rstrip()
