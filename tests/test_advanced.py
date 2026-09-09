"""Unit tests for advanced reward verifiers, GSM8k environment, and micro-batching."""

import pytest
import torch
from slm_rl.rewards.rule_based import ProcessStepReward, SelfReflectionReward
from slm_rl.envs.gsm8k_env import GSM8kTaskGenerator
from slm_rl.config import ModelConfig, RLConfig, GRPOConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.algorithms.grpo import GRPOTrainer
from slm_rl.rewards.rule_based import ReasoningFormatReward


def test_process_step_reward():
    prm = ProcessStepReward(step_reward=0.25, max_reward=0.5)

    prompts = ["Calculate 10 + 5 - 3"] * 3
    completions = [
        # 1. Correct intermediate step
        "<think>First 10 + 5 = 15. Then 15 - 3 = 12.</think><answer>12</answer>",
        # 2. False intermediate equation
        "<think>First 10 + 5 = 99. Then 99 - 3 = 12.</think><answer>12</answer>",
        # 3. No equations in think
        "<think>Just words without math equations.</think><answer>12</answer>",
    ]

    scores = prm(prompts, completions)
    # 2 valid steps * 0.25 = 0.5 (capped at max_reward)
    assert pytest.approx(scores[0].item(), 0.01) == 0.5
    # Only 1 valid step out of 2 (99 - 3 = 12 is false) -> 0.0
    assert pytest.approx(scores[1].item(), 0.01) == 0.0
    # No equations -> 0.0
    assert pytest.approx(scores[2].item(), 0.01) == 0.0


def test_self_reflection_reward():
    ref_reward = SelfReflectionReward(reward_per_marker=0.05, max_reward=0.2)

    prompts = ["Calculate 5 * 5"] * 2
    completions = [
        "<think>First, we multiply 5 by 5. Next, we verify the product. Therefore the result is 25.</think><answer>25</answer>",
        "<think>25</think><answer>25</answer>",
    ]

    scores = ref_reward(prompts, completions)
    # Contains "first", "next", "verify", "therefore" -> 4 markers * 0.05 = 0.20
    assert pytest.approx(scores[0].item(), 0.01) == 0.2
    assert pytest.approx(scores[1].item(), 0.01) == 0.0


def test_gsm8k_task_generator():
    env = GSM8kTaskGenerator(seed=42)
    batch = env.sample_batch(batch_size=3)

    assert len(batch["prompts"]) == 3
    assert len(batch["questions"]) == 3
    assert len(batch["targets"]) == 3
    assert len(batch["solutions"]) == 3
    assert all(isinstance(t, float) for t in batch["targets"])


def test_grpo_micro_batching():
    # Verify that micro_batch_size < total_samples accumulates gradients properly
    model_cfg = ModelConfig(is_mock=True, device="cpu", max_new_tokens=8)
    rl_cfg = RLConfig(
        learning_rate=1e-3,
        batch_size=2,
        micro_batch_size=1,  # Force micro-batching: 2 prompts * 2 group_size = 4 samples in 4 micro-batches
        ppo_epochs=1,
    )
    grpo_cfg = GRPOConfig(group_size=2)
    reward_fn = ReasoningFormatReward()

    policy = SLMPolicy(model_cfg)
    trainer = GRPOTrainer(
        policy=policy,
        reward_fn=reward_fn,
        rl_config=rl_cfg,
        model_config=model_cfg,
        grpo_config=grpo_cfg,
    )

    initial_param = next(policy.parameters()).clone()
    prompts = ["2 + 2", "3 + 3"]
    metrics = trainer.train_step(prompts)

    assert "loss" in metrics
    assert "mean_reward" in metrics
    # Ensure gradients were applied across micro-batches
    assert not torch.equal(initial_param, next(policy.parameters()))
