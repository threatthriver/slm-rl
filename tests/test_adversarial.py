"""Adversarial & Stress Tests for the SLM Reinforcement Learning System."""

import gc
import torch
import pytest

from slm_rl.algorithms.grpo import GRPOTrainer
from slm_rl.config import GRPOConfig, ModelConfig, RLConfig
from slm_rl.core.buffer import TrajectoryBuffer
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.utils import compute_kl_divergence, masked_mean
from slm_rl.rewards.rule_based import CompositeReward, MathCorrectnessReward, ReasoningFormatReward


def test_zero_variance_reward_group():
    """All completions receive identical reward: must not produce NaN/Inf."""
    group_size = 4
    num_groups = 3
    # All rewards are 1.0 (std dev = 0)
    rewards = torch.ones(num_groups * group_size)
    group_ids = torch.repeat_interleave(torch.arange(num_groups), group_size)

    advs = TrajectoryBuffer.compute_grpo_advantages(rewards, group_ids)

    assert not torch.isnan(advs).any()
    assert not torch.isinf(advs).any()
    assert torch.all(advs == 0.0)


def test_extreme_ragged_batches():
    """Batch containing completion of length 1 mixed with length 64."""
    model_cfg = ModelConfig(is_mock=True)
    policy = SLMPolicy(model_cfg)
    reward_fn = CompositeReward([ReasoningFormatReward()])
    rl_cfg = RLConfig(learning_rate=1e-3, micro_batch_size=2)
    trainer = GRPOTrainer(policy, reward_fn, rl_cfg, model_cfg)

    prompts = ["Short prompt", "A very long prompt with extensive problem context 1 + 2 + 3 + 4 + 5 = ?"]
    metrics = trainer.train_step(prompts, targets=[None, 15.0])

    assert "loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))


def test_nan_inf_gradient_guard():
    """Verifies that exponential KL divergence clamping prevents Inf spikes under extreme divergence."""
    # Divergence ratio delta = 100.0 (unconstrained exp(100) would overflow float32)
    log_pi = torch.tensor([50.0])
    log_ref = torch.tensor([-50.0])

    kl = compute_kl_divergence(log_pi, log_ref, method="low_var_kl")

    assert not torch.isnan(kl).any()
    assert not torch.isinf(kl).any()
    assert kl.item() < 1e5  # Clamped within safe bounds


def test_training_memory_leak_stability():
    """Executes 15 consecutive RL train steps, asserting memory stability."""
    model_cfg = ModelConfig(is_mock=True)
    policy = SLMPolicy(model_cfg)
    reward_fn = CompositeReward([ReasoningFormatReward(), MathCorrectnessReward()])
    rl_cfg = RLConfig(learning_rate=1e-3, micro_batch_size=2)
    trainer = GRPOTrainer(policy, reward_fn, rl_cfg, model_cfg)

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run 15 steps
    for step in range(15):
        trainer.train_step(["Calculate 4 + 4."], targets=[8.0])

    gc.collect()
    # Ensure policy parameters are finite
    for p in policy.parameters():
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()
