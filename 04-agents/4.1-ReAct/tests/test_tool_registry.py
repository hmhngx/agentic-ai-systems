"""Offline unit tests for tool registration and dispatch."""

from pydantic import BaseModel, Field

from src.tool_registry import TOOLS, ToolSpec, dispatch_tool, get_tools_description


def test_three_tools_registered():
    """Calculator, file_read, and one web tool (search or mock) must be registered."""
    names = set(TOOLS.keys())
    assert {"calculator", "file_read"}.issubset(names)
    assert "web_search" in names or "web_mock" in names, (
        f"Need web_search or web_mock. Got: {names}"
    )


def test_each_tool_has_required_fields():
    """Every ToolSpec must have name, executor, schema, and a substantial description."""
    for name, spec in TOOLS.items():
        assert spec.name == name, f"Tool name mismatch: {spec.name} vs {name}"
        assert callable(spec.executor), f"{name} executor not callable"
        assert spec.input_schema is not None, f"{name} missing input_schema"
        assert len(spec.description) > 50, f"{name} description too short"


def test_dispatch_unknown_tool_returns_error():
    """Unknown tool must return Error listing available tools for self-correction."""
    result = dispatch_tool("nonexistent_tool", {})
    assert "Error" in result, "Unknown tool must return Error string"
    assert "calculator" in result or "available" in result.lower(), (
        "Error message should list available tools so LLM can self-correct"
    )


def test_dispatch_calculator_success():
    """Dispatch must route to calculator and return numeric result string."""
    result = dispatch_tool("calculator", {"expression": "2 + 2"})
    assert "Result: 4" in result


def test_dispatch_never_raises():
    """Dispatch must never raise — errors become Observations for self-correction."""
    for bad_input in [{}, {"wrong_key": "value"}, {"expression": None}]:
        try:
            result = dispatch_tool("calculator", bad_input)
            assert isinstance(result, str)
        except Exception as e:
            assert False, (
                f"dispatch_tool raised {type(e).__name__} on {bad_input}. "
                "Must never raise — errors must become Observations."
            )


def test_get_tools_description_format():
    """Tool description block must list tools and JSON schema fields for the prompt."""
    desc = get_tools_description()
    assert "calculator" in desc
    assert "web_search" in desc or "web_mock" in desc
    assert "file_read" in desc
    assert "expression" in desc, "JSON schema fields must appear in description"
    assert "filename" in desc, "JSON schema fields must appear in description"


def test_add_new_tool_workflow_simulated():
    """Simulate the <5 min add-tool workflow described in README."""
    class TestToolInput(BaseModel):
        x: int = Field(description="A test integer")

    def test_executor(d: dict) -> str:
        return f"Result: {TestToolInput(**d).x * 2}"

    original_count = len(TOOLS)
    TOOLS["test_tool"] = ToolSpec(
        name="test_tool",
        description="A test tool. It doubles an integer. Do not use for non-integers.",
        input_schema=TestToolInput,
        executor=test_executor,
    )
    try:
        assert "test_tool" in TOOLS
        result = dispatch_tool("test_tool", {"x": 5})
        assert "Result: 10" in result, "New tool dispatch must work immediately"
    finally:
        del TOOLS["test_tool"]
        assert len(TOOLS) == original_count
