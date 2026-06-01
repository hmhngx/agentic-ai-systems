"""Offline tests for FINAL ANSWER observation grounding."""

from src.grounding import check_final_answer_grounded


def test_grounding_allows_facts_in_observation():
    """Names present in observations must pass grounding."""
    history = [
        {
            "role": "user",
            "content": "Observation: Cristiano Ronaldo won the Puskas Award in 2009 for Manchester United.",
        }
    ]
    ok, err = check_final_answer_grounded("Cristiano Ronaldo in 2009", history)
    assert ok is True, err
    assert err is None


def test_grounding_rejects_hallucinated_name():
    """Names not in any observation must be rejected (Bruno Fernandes case)."""
    history = [
        {
            "role": "user",
            "content": "Observation: Cristiano Ronaldo won in 2009. Mason Greenwood was nominated in 2020.",
        }
    ]
    ok, err = check_final_answer_grounded("Bruno Fernandes", history)
    assert ok is False
    assert err is not None
    assert "Bruno Fernandes" in err or "Fernandes" in err


def test_grounding_latest_requires_year_in_answer():
    """Latest queries must cite a year; bare names are rejected."""
    history = [
        {
            "role": "user",
            "content": (
                "Observation: 1. Cristiano Ronaldo, a Manchester United player, won the "
                "Puskas Award in 2009."
            ),
        }
    ]
    ok, err = check_final_answer_grounded(
        "Cristiano Ronaldo",
        history,
        user_query="Latest Man United player to win Puskas?",
    )
    assert ok is False
    assert err is not None
    assert "year" in err.lower()


def test_grounding_latest_rejects_stale_year():
    """MU latest must use the highest year among United winner snippets, not 2009."""
    history = [
        {
            "role": "user",
            "content": (
                "Observation: 1. Cristiano Ronaldo, a Manchester United player, won the "
                "Puskas Award in 2009. "
                "2. Alejandro Garnacho of Manchester United won the Puskas Award in 2024."
            ),
        }
    ]
    ok, err = check_final_answer_grounded(
        "Cristiano Ronaldo in 2009",
        history,
        user_query="Latest Man United Puskas winner?",
    )
    assert ok is False
    assert err is not None


def test_grounding_latest_accepts_ronaldo_2009_when_only_affirmative_win():
    """Vague 'in 2022 ... not certain' must not block Ronaldo 2009 as latest confirmed MU winner."""
    history = [
        {
            "role": "user",
            "content": (
                "Observation: 1. In 2022, the latest Manchester United player to win the "
                "Puskas Award is not certain. "
                "2. Cristiano Ronaldo, a Manchester United player, won the Puskas Award in 2009."
            ),
        }
    ]
    ok, err = check_final_answer_grounded(
        "Cristiano Ronaldo in 2009",
        history,
        user_query="Latest Man United player to win Puskas?",
    )
    assert ok is True, err


def test_grounding_latest_accepts_newest_year():
    """Latest answer must match the highest United winner year in observations."""
    history = [
        {
            "role": "user",
            "content": (
                "Observation: 1. Cristiano Ronaldo, a Manchester United player, won the "
                "Puskas Award in 2009. "
                "2. Alejandro Garnacho of Manchester United won the Puskas Award in 2024."
            ),
        }
    ]
    ok, err = check_final_answer_grounded(
        "Alejandro Garnacho (2024)",
        history,
        user_query="Latest Man United Puskas winner?",
    )
    assert ok is True
    assert err is None


def test_grounding_latest_rejects_global_winner_not_man_utd():
    """Global Puskas winner (Oleksy 2022) must not satisfy a Man United latest query."""
    history = [
        {
            "role": "user",
            "content": (
                "Observation: 1. The 2022 Puskas Award was given to Marcin Oleksy, but it is "
                "unclear if any Manchester United player won in 2022. "
                "2. Cristiano Ronaldo, a Manchester United player, won the Puskas Award in 2009."
            ),
        }
    ]
    ok, err = check_final_answer_grounded(
        "Marcin Oleksy in 2022",
        history,
        user_query="Latest Man United player to win Puskas?",
    )
    assert ok is False
    assert err is not None
    assert "Manchester United" in err or "global" in err.lower() or "WON" in err


def test_grounding_latest_accepts_fifa_puskas_phrasing_from_web_mock():
    """web_mock snippets use 'FIFA Puskas Award' — must still extract winner years."""
    history = [
        {
            "role": "user",
            "content": (
                "Observation: 1. Cristiano Ronaldo won the FIFA Puskas Award in 2009 for his "
                "long-range strike against Porto while playing for Manchester United."
            ),
        }
    ]
    ok, err = check_final_answer_grounded(
        "Cristiano Ronaldo in 2009",
        history,
        user_query="Latest Man United player to win Puskas?",
    )
    assert ok is True, err


def test_grounding_latest_accepts_year_before_fifa_puskas():
    """web_mock often writes 'won the 2024 FIFA Puskas Award' (year before FIFA)."""
    history = [
        {
            "role": "user",
            "content": (
                "Observation: 1. Alejandro Garnacho won the 2024 FIFA Puskas Award for his "
                "overhead kick for Manchester United against Everton. "
                "2. Cristiano Ronaldo won the inaugural FIFA Puskas Award in 2009 for "
                "Manchester United against Porto."
            ),
        }
    ]
    ok, err = check_final_answer_grounded(
        "Alejandro Garnacho in 2024",
        history,
        user_query="Latest Man United player to win Puskas?",
    )
    assert ok is True, err


def test_grounding_allows_numeric_calculator_result():
    """Pure numeric answers without unsupported proper nouns pass."""
    history = [{"role": "user", "content": "Observation: Result: 50000.0"}]
    ok, err = check_final_answer_grounded("50000", history)
    assert ok is True
