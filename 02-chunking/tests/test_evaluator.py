"""Unit tests for src.evaluator."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.evaluator import _cosine_similarity, _count_tokens, evaluate_chunks

# Lines 27-28 in src/evaluator.py: intentionally uncovered — defensive branch
# (spaCy model missing OSError with install message).
# Line 74 in src/evaluator.py: intentionally uncovered — defensive branch
# (empty pairwise_scores after combinations; unreachable when len(sentences) >= 2).

EXPECTED_KEYS = {
    "Strategy",
    "Total Chunks",
    "Avg Tokens",
    "Min Tokens",
    "Max Tokens",
    "Std Tokens",
    "ICC Score",
}


@pytest.fixture
def five_chunks() -> list[str]:
    """Five same-topic machine-learning chunks for ICC coherence tests."""
    return [
        (
            "Machine learning models learn patterns from labeled training data. "
            "Gradient descent minimizes loss by updating weights on each batch. "
            "Cross-validation estimates how well models generalize to unseen examples."
        ),
        (
            "Deep neural networks stack layers to represent hierarchical features. "
            "Convolutional architectures excel at image classification and detection. "
            "Transformers use self-attention for state-of-the-art language modeling."
        ),
        (
            "Regularization techniques like dropout reduce overfitting on small datasets. "
            "Early stopping halts training when validation metrics stop improving. "
            "Hyperparameter search explores learning rates and model capacities."
        ),
        (
            "Supervised learning maps inputs to targets using annotated examples. "
            "Unsupervised learning discovers clusters without explicit labels. "
            "Reinforcement learning optimizes policies through reward signals."
        ),
        (
            "Feature engineering selects informative variables for classical algorithms. "
            "Embeddings map sparse categories into dense vector spaces. "
            "Ensemble methods combine weak learners into stronger predictors."
        ),
    ]


@pytest.fixture
def mixed_chunks() -> list[str]:
    """Five chunks each mixing unrelated topics within the same chunk (low ICC)."""
    return [
        (
            "Machine learning pipelines preprocess data before model training begins. "
            "Cooking pasta requires salting water and tasting for al dente texture. "
            "Ocean currents transport nutrients that sustain plankton and fish larvae."
        ),
        (
            "Gradient descent minimizes loss by updating neural network weights. "
            "Tomato sauces benefit from slow reduction and fresh basil at the end. "
            "Coral reefs provide shelter for diverse species in shallow tropical seas."
        ),
        (
            "Transformers use self-attention for state-of-the-art language modeling. "
            "Baking bread depends on yeast fermentation and gluten development. "
            "Marine mammals navigate using echolocation and seasonal migrations."
        ),
        (
            "Cross-validation estimates generalization on held-out labeled examples. "
            "Knife skills affect surface area and therefore cooking time in kitchens. "
            "Phytoplankton blooms color satellite images during spring upwelling events."
        ),
        (
            "Random forests average many trees to reduce variance and overfitting. "
            "Caramelization develops aroma and crust on pastries and roasted meats. "
            "Deep-sea vents host chemosynthetic bacteria without sunlight."
        ),
    ]


@pytest.fixture
def eval_result(five_chunks: list[str]) -> dict:
    """Baseline evaluation dict for five_chunks with strategy name test."""
    return evaluate_chunks(five_chunks, "test")


def test_returns_correct_keys(eval_result):
    """evaluate_chunks returns a dict with the expected metric keys."""
    assert isinstance(eval_result, dict), "expected dict result"
    assert set(eval_result.keys()) == EXPECTED_KEYS, "expected standard benchmark columns"


def test_strategy_name_preserved(eval_result):
    """Strategy column echoes the supplied strategy name."""
    assert eval_result["Strategy"] == "test", "expected Strategy field to match input name"


def test_total_chunks_correct(eval_result):
    """Total Chunks equals the number of input chunks."""
    assert eval_result["Total Chunks"] == 5, "expected five chunks counted"


def test_avg_tokens_positive(eval_result):
    """Avg Tokens is a positive float."""
    assert eval_result["Avg Tokens"] > 0, "expected positive average token count"
    assert isinstance(eval_result["Avg Tokens"], float), "expected Avg Tokens as float"


def test_icc_score_range(eval_result):
    """ICC Score lies in the closed interval [0, 1]."""
    score = eval_result["ICC Score"]
    assert 0.0 <= score <= 1.0, "expected ICC between 0 and 1 inclusive"
    assert isinstance(score, float), "expected ICC Score as float"


def test_icc_rounded_to_4_places(eval_result):
    """ICC Score is rounded to four decimal places."""
    score = eval_result["ICC Score"]
    assert score == round(score, 4), "expected ICC rounded to four decimals"


def test_same_topic_higher_icc_than_mixed(five_chunks, mixed_chunks):
    """Coherent same-topic chunks score higher ICC than mixed-topic chunks."""
    score_same = evaluate_chunks(five_chunks, "same")["ICC Score"]
    score_mixed = evaluate_chunks(mixed_chunks, "mixed")["ICC Score"]
    assert score_same > score_mixed, (
        "expected same-topic ICC to exceed mixed-topic ICC"
    )


def test_empty_input():
    """Empty chunk list returns zeros without raising."""
    result = evaluate_chunks([], "empty")
    assert result["Total Chunks"] == 0, "expected zero chunks for empty input"
    assert result["ICC Score"] == 0.0, "expected zero ICC for empty input"


def test_single_chunk_single_sentence():
    """Single-sentence chunk has ICC 1.0 as trivially coherent."""
    result = evaluate_chunks(["This is one sentence."], "single")
    assert result["Total Chunks"] == 1, "expected one chunk counted"
    assert result["ICC Score"] == 1.0, "expected ICC 1.0 for single sentence"


def test_deterministic(five_chunks):
    """Repeated evaluation yields identical results with fixed random seed."""
    first = evaluate_chunks(five_chunks, "test")
    second = evaluate_chunks(five_chunks, "test")
    assert first == second, "expected deterministic ICC and token statistics"


def test_count_tokens_falls_back_when_tiktoken_unavailable():
    """_count_tokens uses char/4 estimate when tiktoken encoding fails."""
    import tiktoken

    with patch.object(tiktoken, "get_encoding", side_effect=RuntimeError("no tiktoken")):
        count = _count_tokens("abcd")
    assert count == 1, "expected fallback token estimate of len(text)//4"


def test_evaluator_cosine_similarity_zero_vector():
    """_cosine_similarity returns 0.0 when either vector has zero norm."""
    zero = np.zeros(3)
    other = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(zero, other) == 0.0, "expected zero similarity for zero vector"
