from src.graph import build_graph, run_agent


def test_graph_has_expected_nodes():
    g = build_graph().get_graph()
    names = set(g.nodes)
    for n in ("decide", "retrieve_and_generate", "evaluate", "refine",
              "direct_answer", "fallback", "finalize"):
        assert n in names


def test_first_try_faithful_no_reflexion():
    final = run_agent("What is the Helios query quota per day?")
    assert final["status"] == "answered"
    assert final["served"] is True
    assert final["faithfulness"] >= 0.70
    assert len(final["attempts_log"]) == 1          # no reflexion needed


def test_reflexion_loop_fires_then_succeeds():
    final = run_agent("How long does Helios keep documents before deleting them?")
    assert len(final["attempts_log"]) >= 2          # reflexion fired
    assert final["status"] == "answered"
    assert final["served"] is True
    assert final["faithfulness"] >= 0.70
    assert final["attempts_log"][0]["faithfulness"] < final["faithfulness"]


def test_direct_route_skips_retrieval():
    final = run_agent("Given that the pro tier allows 100000 queries per day, is 80000 within that limit?")
    assert final["status"] == "direct"
    assert final["attempts_log"] == []              # never retrieved


def test_unanswerable_exhausts_retries_then_falls_back():
    final = run_agent("What is the Helios carbon footprint per query?")
    assert final["status"] == "fallback"
    assert final["served"] is False
    assert final["attempt"] == 3                    # 1 initial + exactly 2 retries
    assert len(final["attempts_log"]) == 3


def test_fallback_reports_best_attempt():
    final = run_agent("What is the Helios carbon footprint per query?")
    best = max(a["faithfulness"] for a in final["attempts_log"])
    assert final["faithfulness"] == best            # best, not last
