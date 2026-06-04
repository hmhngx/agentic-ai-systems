import operator

from src.state import (
    AgenticRAGState,
    DEFAULT_MAX_RETRIES,
    FAITHFULNESS_THRESHOLD,
    DATASET_TARGET,
    initial_state,
)


def test_constants():
    assert DEFAULT_MAX_RETRIES == 2
    assert FAITHFULNESS_THRESHOLD == 0.70
    assert DATASET_TARGET == 0.75


def test_initial_state_defaults():
    s = initial_state("How long does Helios keep documents?")
    assert s["question"] == "How long does Helios keep documents?"
    assert s["query"] == s["question"]          # first query == the question verbatim
    assert s["attempt"] == 0
    assert s["max_retries"] == 2
    assert s["faithfulness_threshold"] == 0.70
    assert s["attempts_log"] == []              # reducer field seeded empty
    assert s["log"] == []
    assert s["served"] is False


def test_reducer_fields_are_annotated_lists():
    # The two accumulating fields must use operator.add so node outputs append.
    hints = AgenticRAGState.__annotations__
    for field in ("attempts_log", "log"):
        meta = getattr(hints[field], "__metadata__", ())
        assert operator.add in meta, f"{field} must use operator.add reducer"
