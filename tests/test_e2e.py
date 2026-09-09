"""End-to-end integration tests for multi-step RL training and evaluation."""

import pytest
from slm_rl.config import ModelConfig, RLConfig, GRPOConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.rewards.rule_based import ReasoningFormatReward, MathCorrectnessReward, CompositeReward
from slm_rl.envs.reasoning_env import ReasoningTaskGenerator
from slm_rl.algorithms.grpo import GRPOTrainer
from slm_rl.evaluation.evaluator import Evaluator


def test_grpo_end_to_end_loop():
    # Setup fast mock model
    model_cfg = ModelConfig(is_mock=True, device="cpu", max_new_tokens=12)
    rl_cfg = RLConfig(
        learning_rate=5e-4,
        ppo_epochs=1,
        batch_size=2,
        total_steps=3,
    )
    grpo_cfg = GRPOConfig(group_size=2)

    reward_fn = CompositeReward([
        (ReasoningFormatReward(), 0.6),
        (MathCorrectnessReward(), 0.4),
    ])

    policy = SLMPolicy(model_cfg)
    trainer = GRPOTrainer(
        policy=policy,
        reward_fn=reward_fn,
        rl_config=rl_cfg,
        model_config=model_cfg,
        grpo_config=grpo_cfg,
    )

    env = ReasoningTaskGenerator(seed=123)

    # Run 3 training iterations
    for step in range(3):
        batch = env.sample_batch(batch_size=rl_cfg.batch_size)
        metrics = trainer.train_step(prompts=batch["prompts"], targets=batch["targets"])
        assert "mean_reward" in metrics
        assert "loss" in metrics

    assert len(trainer.history) == 3

    # Run evaluation
    evaluator = Evaluator(policy)
    test_batch = env.sample_batch(batch_size=2)
    eval_results = evaluator.evaluate(test_batch["prompts"], test_batch["targets"])

    assert "accuracy" in eval_results
    assert "format_adherence_rate" in eval_results
    assert eval_results["num_samples"] == 2
