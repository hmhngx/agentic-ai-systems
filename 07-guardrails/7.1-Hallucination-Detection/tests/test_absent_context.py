"""
The Day-10 goal, asserted directly:

    Test on 5 questions where context is absent: all 5 should trigger the
    fallback response.

Plus the discrimination guard: a grounded control answer MUST be served, so
"all five fall back" reflects judgement, not a detector that always refuses.
"""

from __future__ import annotations

import pytest

from src.demo_corpus import ABSENT_CONTEXT_SCENARIOS, GROUNDED_CONTROL
from src.detector import detect
from src.types import DetectorConfig, FALLBACK_MESSAGE, Verdict


CONFIG = DetectorConfig()  # explicit defaults: offline, threshold 0.5


def _detect(scenario):
    return detect(scenario.question, scenario.answer, scenario.chunks, config=CONFIG)


def test_exactly_five_absent_context_scenarios():
    """The suite must exercise five distinct absent-context questions."""
    assert len(ABSENT_CONTEXT_SCENARIOS) == 5


def test_all_five_absent_scenarios_fall_back():
    """The headline guarantee: none of the five serve their answer."""
    for scenario in ABSENT_CONTEXT_SCENARIOS:
        report = _detect(scenario)
        assert report.served is False, f"{scenario.name} was served unexpectedly"
        assert report.served_answer == FALLBACK_MESSAGE


@pytest.mark.parametrize(
    "scenario", ABSENT_CONTEXT_SCENARIOS, ids=lambda s: s.name
)
def test_each_absent_scenario_diagnoses_its_failure_mode(scenario):
    """Each scenario must be diagnosed with its specific, distinct verdict."""
    report = _detect(scenario)
    assert report.verdict == scenario.expected_verdict
    assert report.served == scenario.expected_served


def test_five_scenarios_cover_four_distinct_failure_modes():
    """The five scenarios are not the same trivial path repeated."""
    verdicts = {_detect(s).verdict for s in ABSENT_CONTEXT_SCENARIOS}
    assert verdicts == {
        Verdict.NO_RETRIEVAL,
        Verdict.UNSUPPORTED,        # two scenarios, different causes
        Verdict.FABRICATED_CITATION,
        Verdict.ABSTAINED,
    }


def test_absent_scenario_scores_are_below_threshold():
    """A fallback must be backed by a sub-threshold score, in [0, 1]."""
    for scenario in ABSENT_CONTEXT_SCENARIOS:
        report = _detect(scenario)
        assert 0.0 <= report.score < CONFIG.threshold


def test_grounded_control_is_served():
    """The detector discriminates: a genuinely grounded answer is served."""
    report = detect(
        GROUNDED_CONTROL.question,
        GROUNDED_CONTROL.answer,
        GROUNDED_CONTROL.chunks,
        config=CONFIG,
    )
    assert report.served is True
    assert report.verdict == Verdict.GROUNDED
    assert report.served_answer == GROUNDED_CONTROL.answer
    assert report.score >= CONFIG.threshold
