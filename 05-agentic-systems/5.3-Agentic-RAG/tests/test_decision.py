from src.decision import decide, Decision, DIRECT_MARKERS


def test_factual_kb_question_routes_retrieve():
    d = decide("What is the Helios query quota per day?")
    assert isinstance(d, Decision)
    assert d.route == "retrieve"


def test_self_contained_context_routes_direct():
    d = decide("Given that the pro tier allows 100000 queries per day, is 80000 within that limit?")
    assert d.route == "direct"
    assert d.reason


def test_meta_question_routes_direct():
    assert decide("Could you restate your previous answer more concisely?").route == "direct"


def test_markers_present():
    assert "given that" in DIRECT_MARKERS
    assert "previous answer" in DIRECT_MARKERS
