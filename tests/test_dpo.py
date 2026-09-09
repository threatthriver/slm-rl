"""Tests for Direct Preference Optimization (DPO) Trainer."""

import torch
import torch.nn.functional as F
import pytest

from slm_rl.algorithms.dpo import DPOTrainer
from slm_rl.config import DPOConfig, ModelConfig, RLConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.rewards.rule_based import CompositeReward, MathCorrectnessReward, ReasoningFormatReward


def test_dpo_preference_pairing():
    policy = SLMPolicy(ModelConfig(is_mock=True))
    reward_fn = CompositeReward([MathCorrectnessReward(1.0)])
    rl_config = RLConfig()
    trainer = DPOTrainer(policy, reward_fn, rl_config, ModelConfig(is_mock=True))

    prompts = ["Calculate 5 + 5", "Calculate 5 + 5"]
    completions = ["<think>5+5=10</think><answer>10</answer>", "<think>5+5=9</think><answer>9</answer>"]
    rewards = torch.tensor([1.0, 0.0])

    p_prompts, winners, losers, win_r, lose_r = trainer.extract_preference_pairs(
        prompts, completions, rewards, group_size=2
    )

    assert len(winners) == 1
    assert "10" in winners[0]
    assert "9" in losers[0]
    assert win_r[0] > lose_r[0]


def test_dpo_train_step():
    model_cfg = ModelConfig(is_mock=True)
    policy = SLMPolicy(model_cfg)
    reward_fn = CompositeReward([ReasoningFormatReward(), MathCorrectnessReward()])
    rl_cfg = RLConfig(learning_rate=1e-3)
    dpo_cfg = DPOConfig(beta=0.1)

    trainer = DPOTrainer(
        policy=policy,
        reward_fn=reward_fn,
        rl_config=rl_cfg,
        model_config=model_cfg,
        dpo_config=dpo_cfg,
    )

    prompts = ["Calculate 7 + 8.", "Calculate 10 - 4."]
    metrics = trainer.train_step(prompts, targets=[15.0, 6.0])

    assert "loss" in metrics
    assert "dpo_margin" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))
    assert not torch.isinf(torch.tensor(metrics["loss"]))
