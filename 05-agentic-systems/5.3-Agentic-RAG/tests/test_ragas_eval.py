from src.ragas_eval import evaluate_dataset, format_table
from src.state import DATASET_TARGET


def test_full_pipeline_mean_faithfulness_meets_target():
    m = evaluate_dataset()
    assert m["mean_faithfulness"] >= DATASET_TARGET   # headline goal: >= 0.75


def test_route_accuracy_is_perfect_on_curated_set():
    assert evaluate_dataset()["route_accuracy"] == 1.0


def test_reflexion_and_fallback_rows_match_expectations():
    rows = {r["id"]: r for r in evaluate_dataset()["rows"]}
    assert rows["q6"]["reflexion_fired"] is True
    assert rows["q6"]["status"] == "answered"
    assert rows["q10"]["status"] == "fallback"
    assert rows["q10"]["served"] is False


def test_format_table_is_readable_string():
    table = format_table(evaluate_dataset())
    assert "faithfulness" in table.lower()
    assert "q6" in table
