"""Unit tests for reward functions and verifiers."""

import pytest
import torch
from slm_rl.rewards.rule_based import (
    ReasoningFormatReward,
    MathCorrectnessReward,
    CompositeReward,
)


def test_reasoning_format_reward():
    reward_fn = ReasoningFormatReward()

    prompts = ["Solve this:"] * 4
    completions = [
    # 1. Perfect structure (with real thinking >= 10 chars)
        "<think>First 2+2 is 4.</think><answer>4</answer>",
        # 2. Only think
        "<think>Just thinking here without final answer.</think>",
        # 3. Only answer
        "<answer>42</answer>",
        # 4. Inverted structure
        "<answer>42</answer><think>Thinking backwards</think>",
        # 5. Empty thought hack
        "<think></think><answer>42</answer>",
    ]

    scores = reward_fn(prompts + ["Solve:"], completions)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (5,)

    # Perfect structure gets full reward (0.4 + 0.4 + 0.2 = 1.0)
    assert pytest.approx(scores[0].item(), 0.01) == 1.0
    # Only think gets 0.4
    assert pytest.approx(scores[1].item(), 0.01) == 0.4
    # Only answer gets 0.4
    assert pytest.approx(scores[2].item(), 0.01) == 0.4
    # Inverted gets 0.8 (think + answer but no order bonus)
    assert pytest.approx(scores[3].item(), 0.01) == 0.8
    # Empty thought hack: only gets answer weight (0.4), NOT think or order bonus
    assert pytest.approx(scores[4].item(), 0.01) == 0.4


def test_repetition_penalty_reward():
    from slm_rl.rewards.rule_based import RepetitionPenaltyReward
    rep_fn = RepetitionPenaltyReward(n=3, threshold=0.25, penalty_weight=0.5)

    prompts = ["Test prompt"] * 2
    completions = [
        "First step is calculate 5 plus 5 equals 10 then we get answer.",
        "the answer is 10 the answer is 10 the answer is 10 the answer is 10",
    ]
    scores = rep_fn(prompts, completions)
    assert scores[0].item() == 0.0  # clean text, no repetition penalty
    assert scores[1].item() < 0.0   # high repetition loop penalized



def test_math_correctness_reward():
    reward_fn = MathCorrectnessReward()

    prompts = ["Calculate 15 + 5"] * 3
    targets = [20, 20, 20]
    completions = [
        "<think>15+5=20</think><answer>20</answer>",
        "<think>15+5=25</think><answer>25</answer>",
        "The answer is 20",  # fallback extraction
    ]

    scores = reward_fn(prompts, completions, targets=targets)
    assert scores[0].item() == 1.0
    assert scores[1].item() == 0.0
    assert scores[2].item() == 1.0


def test_composite_reward():
    format_fn = ReasoningFormatReward()
    math_fn = MathCorrectnessReward()

    composite = CompositeReward([
        (format_fn, 0.5),
        (math_fn, 0.5),
    ])

    prompts = ["What is 3 * 3?"]
    targets = [9]
    completions = ["<think>3 times 3 equals 9 step by step</think><answer>9</answer>"]

    scores = composite(prompts, completions, targets=targets)
    # 0.5 * 1.0 (format) + 0.5 * 1.0 (math) = 1.0
    assert pytest.approx(scores[0].item(), 0.01) == 1.0

