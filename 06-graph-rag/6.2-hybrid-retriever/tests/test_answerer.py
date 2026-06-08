from src.answerer import generate_answer


def test_offline_answer_contains_context():
    ctx = "[Retrieved Documents]\n[1] (score=0.9) Sam Altman is CEO of OpenAI."
    answer = generate_answer(ctx, "Who leads OpenAI?")
    assert "CONTEXT PREVIEW" in answer
    assert "Sam Altman" in answer


def test_offline_answer_contains_question():
    ctx = "some context"
    answer = generate_answer(ctx, "What is RLHF?")
    assert "What is RLHF?" in answer


def test_offline_answer_is_string():
    answer = generate_answer("ctx", "question?")
    assert isinstance(answer, str)
    assert len(answer) > 0
