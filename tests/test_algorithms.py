"""Unit tests for GRPO and PPO training steps."""

import pytest
import torch
from slm_rl.config import ModelConfig, RLConfig, GRPOConfig, PPOConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.critic import SLMCritic
from slm_rl.rewards.rule_based import ReasoningFormatReward
from slm_rl.algorithms.grpo import GRPOTrainer
from slm_rl.algorithms.ppo import PPOTrainer


@pytest.fixture
def mock_setup():
    model_cfg = ModelConfig(is_mock=True, device="cpu", max_new_tokens=10)
    rl_cfg = RLConfig(learning_rate=1e-3, ppo_epochs=1, batch_size=2)
    reward_fn = ReasoningFormatReward()
    policy = SLMPolicy(model_cfg)
    return model_cfg, rl_cfg, reward_fn, policy


def test_grpo_step(mock_setup):
    model_cfg, rl_cfg, reward_fn, policy = mock_setup
    grpo_cfg = GRPOConfig(group_size=2)
    trainer = GRPOTrainer(
        policy=policy,
        reward_fn=reward_fn,
        rl_config=rl_cfg,
        model_config=model_cfg,
        grpo_config=grpo_cfg,
    )

    # Clone initial weights to check that update occurred
    initial_param = next(policy.parameters()).clone()

    prompts = ["Calculate 2 + 2", "Calculate 3 + 3"]
    metrics = trainer.train_step(prompts)

    assert "mean_reward" in metrics
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "kl_divergence" in metrics
    assert trainer.step_count == 1

    # Check that weights were modified by optimizer step
    updated_param = next(policy.parameters())
    assert not torch.equal(initial_param, updated_param)


def test_ppo_step(mock_setup):
    model_cfg, rl_cfg, reward_fn, policy = mock_setup
    critic = SLMCritic(model_cfg)
    ppo_cfg = PPOConfig()

    trainer = PPOTrainer(
        policy=policy,
        critic=critic,
        reward_fn=reward_fn,
        rl_config=rl_cfg,
        model_config=model_cfg,
        ppo_config=ppo_cfg,
    )

    initial_policy_param = next(policy.parameters()).clone()
    initial_critic_param = next(critic.parameters()).clone()

    prompts = ["Calculate 2 + 2", "Calculate 3 + 3"]
    metrics = trainer.train_step(prompts)

    assert "mean_reward" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert trainer.step_count == 1

    # Check that both policy and critic weights were updated
    assert not torch.equal(initial_policy_param, next(policy.parameters()))
    assert not torch.equal(initial_critic_param, next(critic.parameters()))
