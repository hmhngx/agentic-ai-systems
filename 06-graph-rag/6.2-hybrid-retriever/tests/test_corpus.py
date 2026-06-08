from src.corpus import PASSAGES


def test_passage_count():
    assert len(PASSAGES) == 20


def test_passage_schema():
    for p in PASSAGES:
        assert "id" in p, f"missing 'id' in {p}"
        assert "text" in p, f"missing 'text' in {p}"
        assert "entities" in p, f"missing 'entities' in {p}"
        assert isinstance(p["entities"], list)
        assert len(p["text"]) > 20, f"text too short in {p['id']}"


def test_all_question_chain_passages_present():
    ids = {p["id"] for p in PASSAGES}
    required = {
        # Q1 chain
        "dario_left", "anthropic_founded", "constitutional_ai",
        # Q2 chain
        "hassabis_deepmind", "alphafold", "alphafold_data",
        # Q3 chain
        "vaswani", "llama2_arch", "llama2_data",
        # Q4 chain
        "hinton_google", "brain_deepmind_merger", "gemini",
        # Q5 chain
        "altman_ceo", "gpt4", "rlhf",
    }
    assert required <= ids, f"Missing passages: {required - ids}"
