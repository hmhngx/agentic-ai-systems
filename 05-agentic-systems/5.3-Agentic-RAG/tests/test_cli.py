from agentic_rag import main


def test_single_question_prints_answer(capsys):
    main(["What is the Helios query quota per day?"])
    out = capsys.readouterr().out.lower()
    assert "quota" in out
    assert "answered" in out or "served" in out


def test_eval_flag_prints_table_and_target(capsys):
    main(["--eval"])
    out = capsys.readouterr().out.lower()
    assert "faithfulness" in out
    assert "0.75" in out


def test_default_demo_runs_all_four_paths(capsys):
    main([])
    out = capsys.readouterr().out.lower()
    for marker in ("reflexion", "direct", "fallback"):
        assert marker in out
