from src.generator import generate, GROUND_MIN, REFUSAL
from src.retriever import retrieve


def test_strong_grounding_extracts_verbatim_sentence_with_citation():
    chunks = retrieve("Helios query quota per day")
    out = generate("What is the Helios query quota per day?", "Helios query quota per day", chunks)
    assert out["confidence"] >= GROUND_MIN
    assert "quota" in out["answer"].lower()
    assert out["citations"] == ["Doc 1"]
    # the answer sentence is taken from the chunk text (grounded)
    assert out["answer"].split(" [Doc")[0].strip().rstrip(".") in chunks[0]["text"]


def test_weak_grounding_guesses_using_question_words():
    # casual question; query == question; grounds weakly -> guess
    chunks = retrieve("How long does Helios keep documents before deleting them?")
    out = generate(
        "How long does Helios keep documents before deleting them?",
        "How long does Helios keep documents before deleting them?",
        chunks,
    )
    assert out["confidence"] < GROUND_MIN
    # guess reuses casual question words that are NOT in the corpus
    assert "keep" in out["answer"].lower() or "deleting" in out["answer"].lower()


def test_no_chunks_returns_refusal():
    out = generate("What is the Helios carbon footprint?", "carbon footprint", [])
    assert out["answer"] == REFUSAL
    assert out["citations"] == []
    assert out["confidence"] == 0.0
