from src.graph import build_graph
from src.report import format_top_connected
from src.schema import Entity, EntityType, ExtractionResult, Triple


def test_format_top_connected_lists_highest_degree():
    result = ExtractionResult(
        entities=[
            Entity(name="Hub", type=EntityType.ORG),
            Entity(name="A", type=EntityType.PERSON),
            Entity(name="B", type=EntityType.PERSON),
        ],
        triples=[
            Triple(source="A", relation="r", target="Hub"),
            Triple(source="B", relation="r", target="Hub"),
        ],
    )
    G = build_graph(result)
    text = format_top_connected(G, n=2)
    assert "Hub" in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].lower().startswith("top") or "Hub" in lines[0] or "Hub" in lines[1]
