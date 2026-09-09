"""Tests for Counterfactual Step-Level Advantage Optimization (CSAO) & Pivot Step Detection."""

import torch
import pytest

from slm_rl.config import CSAOConfig
from slm_rl.rewards.step_credit import StepCreditAssigner


def test_step_decomposition():
    assigner = StepCreditAssigner()
    completion = (
        "<think>\n"
        "Step 1: Compute 15 + 10 = 25.\n"
        "Step 2: Multiply 25 * 2 = 50.\n"
        "Step 3: Subtract 50 - 5 = 45.\n"
        "</think>\n"
        "<answer>45</answer>"
    )
    steps = assigner.decompose_steps(completion)
    assert len(steps) == 3
    assert all(s["is_valid"] for s in steps)
    assert all(s["has_equation"] for s in steps)


def test_pivot_error_detection():
    assigner = StepCreditAssigner()
    # Step 2 has an arithmetic fallacy: 20 * 2 = 50
    completion = (
        "<think>\n"
        "Step 1: We have 10 + 10 = 20.\n"
        "Step 2: Then 20 * 2 = 50.\n"
        "Step 3: Finally 50 - 5 = 45.\n"
        "</think>\n"
        "<answer>45</answer>"
    )
    steps = assigner.decompose_steps(completion)
    assert len(steps) == 3
    assert steps[0]["is_valid"] is True
    assert steps[1]["is_valid"] is False  # Fallacy
    assert steps[2]["is_valid"] is True   # Locally valid, but downstream of error

    pivot_idx = assigner.detect_pivot_error(steps)
    assert pivot_idx == 1


def test_counterfactual_advantages():
    config = CSAOConfig(step_reward=0.3, pivot_penalty=-1.0, downstream_penalty=-0.2, gamma=0.8)
    assigner = StepCreditAssigner(config)

    # In a 3-step reasoning chain where Step 2 failed:
    steps = [
        {"index": 0, "is_valid": True, "start_char": 0, "end_char": 10},
        {"index": 1, "is_valid": False, "start_char": 11, "end_char": 20},
        {"index": 2, "is_valid": True, "start_char": 21, "end_char": 30},
    ]

    advs = assigner.compute_step_advantages(steps, final_correct=False)
    assert len(advs) == 3
    # Step 0 was valid prior to pivot: gets positive reward!
    assert advs[0] > 0.0
    # Step 1 was the pivot error: gets full penalty
    assert advs[1] == -1.0
    # Step 2 was downstream: gets downstream penalty
    assert advs[2] == -0.2


def test_token_advantage_mapping():
    assigner = StepCreditAssigner()
    completion = (
        "<think>\n"
        "Step 1: 5 + 5 = 10.\n"
        "Step 2: 10 * 2 = 25.\n"
        "</think>\n"
        "<answer>25</answer>"
    )
    steps = assigner.decompose_steps(completion)
    advs = assigner.compute_step_advantages(steps, final_correct=False)
    seq_len = 32
    token_advs = assigner.map_to_token_advantages(completion, steps, advs, seq_len, final_advantage=-0.5)

    assert token_advs.shape == (seq_len,)
    assert not torch.isnan(token_advs).any()


def test_csao_train_step():
    from slm_rl.algorithms.step_csao import CSAOTrainer
    from slm_rl.config import ModelConfig, RLConfig
    from slm_rl.core.policy import SLMPolicy
    from slm_rl.rewards.rule_based import CompositeReward, MathCorrectnessReward, ReasoningFormatReward

    model_cfg = ModelConfig(is_mock=True)
    policy = SLMPolicy(model_cfg)
    reward_fn = CompositeReward([ReasoningFormatReward(), MathCorrectnessReward()])
    rl_cfg = RLConfig(learning_rate=1e-3, micro_batch_size=2)
    csao_cfg = CSAOConfig()

    trainer = CSAOTrainer(
        policy=policy,
        reward_fn=reward_fn,
        rl_config=rl_cfg,
        model_config=model_cfg,
        csao_config=csao_cfg,
    )

    prompts = ["Calculate 8 + 12.", "Calculate 20 - 7."]
    metrics = trainer.train_step(prompts, targets=[20.0, 13.0])

    assert "loss" in metrics
    assert "pivot_rate" in metrics
    assert "mean_step_advantage" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))

