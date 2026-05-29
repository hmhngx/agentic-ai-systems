"""Offline tests for recall@5 — set-based overlap metric used in the benchmark."""

from __future__ import annotations

import random

from src.ground_truth import recall_at_5


def test_recall_at_5_perfect() -> None:
    """Perfect top-5 overlap must score 1.0 — upper bound for pipeline comparison."""
    result = recall_at_5(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"])
    assert result == 1.0, f"Perfect overlap: expected 1.0, got {result}"


def test_recall_at_5_zero() -> None:
    """No overlap must score 0.0 — complete retrieval failure on ground truth."""
    result = recall_at_5(["f", "g", "h", "i", "j"], ["a", "b", "c", "d", "e"])
    assert result == 0.0, f"No overlap: expected 0.0, got {result}"


def test_recall_at_5_partial() -> None:
    """Two of five matches must score 0.4 — standard recall@5 definition."""
    result = recall_at_5(["a", "b", "x", "y", "z"], ["a", "b", "c", "d", "e"])
    assert abs(result - 0.4) < 1e-9, (
        f"2/5 overlap: expected 0.4, got {result} — wrong formula breaks ablation."
    )


def test_recall_at_5_order_independent() -> None:
    """Recall@k is set-based — order changes must not change the score."""
    result = recall_at_5(["e", "d", "c", "b", "a"], ["a", "b", "c", "d", "e"])
    assert result == 1.0, (
        "Recall@k is order-independent — reversed list must still give 1.0"
    )


def test_recall_at_5_only_uses_first_5() -> None:
    """Only the first five retrieved and GT ids count — extras must not inflate recall."""
    retrieved = ["a", "b", "c", "d", "e", "BONUS_MATCH"]
    gt = ["a", "b", "c", "d", "e", "BONUS_MATCH"]
    result = recall_at_5(retrieved, gt)
    assert result == 1.0, (
        "Only first 5 considered — extra items beyond k=5 must not affect score"
    )


def test_recall_at_5_returns_float() -> None:
    """Return type must be float for mean recall aggregation in the reporter."""
    result = recall_at_5(["a"], ["a"])
    assert isinstance(result, float), (
        f"recall_at_5 must return float, got {type(result)}"
    )


def test_recall_at_5_range() -> None:
    """recall@5 must always lie in [0, 1] for any random id lists."""
    for _ in range(100):
        retrieved = [str(random.randint(0, 9)) for _ in range(5)]
        gt = [str(random.randint(0, 9)) for _ in range(5)]
        result = recall_at_5(retrieved, gt)
        assert 0.0 <= result <= 1.0, (
            f"recall_at_5 must be in [0,1], got {result} — out-of-range breaks averages."
        )
