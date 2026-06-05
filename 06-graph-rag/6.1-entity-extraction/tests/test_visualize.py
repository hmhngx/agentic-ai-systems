from pathlib import Path

from src.graph import build_graph
from src.visualize import draw_graph
from src.schema import Entity, EntityType, ExtractionResult, Triple


def test_draw_graph_writes_png(tmp_path: Path):
    result = ExtractionResult(
        entities=[
            Entity(name="A", type=EntityType.PERSON, mentions=2),
            Entity(name="B", type=EntityType.ORG),
            Entity(name="C", type=EntityType.MODEL),
        ],
        triples=[
            Triple(source="A", relation="r", target="B"),
            Triple(source="A", relation="r", target="C"),
        ],
    )
    G = build_graph(result)
    out = tmp_path / "graph.png"
    path = draw_graph(G, str(out))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
