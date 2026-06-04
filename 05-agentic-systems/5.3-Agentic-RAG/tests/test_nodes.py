from src.state import initial_state
from src import nodes


def test_decide_node_sets_route_and_query():
    out = nodes.decide_node(initial_state("What is the Helios query quota per day?"))
    assert out["route"] == "retrieve"
    assert out["query"] == "What is the Helios query quota per day?"


def test_retrieve_and_generate_increments_attempt_and_fills_answer():
    s = initial_state("What is the Helios query quota per day?")
    out = nodes.retrieve_and_generate_node(s)
    assert out["attempt"] == 1
    assert out["chunks"][0]["doc_id"] == "D3"
    assert "quota" in out["answer"].lower()


def test_evaluate_node_appends_attempt_and_scores():
    s = initial_state("What is the Helios query quota per day?")
    s.update(nodes.retrieve_and_generate_node(s))
    out = nodes.evaluate_node(s)
    assert out["faithfulness"] >= 0.70
    assert len(out["attempts_log"]) == 1
    assert out["attempts_log"][0]["diagnosis"] == "GROUNDED"


def test_route_after_eval_finalizes_when_faithful():
    s = initial_state("q")
    s["faithfulness"] = 0.9
    s["attempt"] = 1
    assert nodes.route_after_eval(s) == "finalize"


def test_route_after_eval_refines_then_falls_back():
    s = initial_state("q", max_retries=2)
    s["faithfulness"] = 0.0
    s["attempt"] = 1
    assert nodes.route_after_eval(s) == "refine"
    s["attempt"] = 2
    assert nodes.route_after_eval(s) == "refine"
    s["attempt"] = 3
    assert nodes.route_after_eval(s) == "fallback"


def test_fallback_selects_best_attempt_not_last():
    s = initial_state("q")
    s["attempts_log"] = [
        {"query": "a", "answer": "AAA", "faithfulness": 0.2, "diagnosis": "WRONG_CHUNKS"},
        {"query": "b", "answer": "BBB", "faithfulness": 0.6, "diagnosis": "UNSUPPORTED_GENERATION"},
        {"query": "c", "answer": "CCC", "faithfulness": 0.1, "diagnosis": "WRONG_CHUNKS"},
    ]
    out = nodes.fallback_node(s)
    assert out["status"] == "fallback"
    assert out["served"] is False
    assert out["faithfulness"] == 0.6        # best, not the last (0.1)


def test_finalize_selects_best_attempt_answer():
    s = initial_state("q")
    s["attempts_log"] = [
        {"query": "a", "answer": "low", "faithfulness": 0.5, "diagnosis": "UNSUPPORTED_GENERATION"},
        {"query": "b", "answer": "high", "faithfulness": 0.9, "diagnosis": "GROUNDED"},
    ]
    out = nodes.finalize_node(s)
    assert out["status"] == "answered"
    assert out["served"] is True
    assert out["final_answer"] == "high"


def test_direct_node_serves_without_retrieval():
    out = nodes.direct_node(initial_state("Given that X, is Y true?"))
    assert out["status"] == "direct"
    assert out["served"] is True
    assert out["final_answer"]


def test_nodes_never_raise_on_bad_state():
    # a malformed state must degrade, not crash (defensive node contract)
    out = nodes.retrieve_and_generate_node({"question": "q", "query": None, "attempt": 0})
    assert out["attempt"] == 1            # still advances; answer degrades to refusal/empty
