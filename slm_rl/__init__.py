"""SLM-RL: Reinforcement Learning System for Small Language Models."""

from slm_rl.config import ModelConfig, RLConfig, GRPOConfig, PPOConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.critic import SLMCritic
from slm_rl.core.buffer import TrajectoryBuffer, RolloutBatch
from slm_rl.rewards.base import BaseRewardFunction
from slm_rl.rewards.rule_based import (
    ReasoningFormatReward,
    MathCorrectnessReward,
    CompositeReward,
    RepetitionPenaltyReward,
    ProcessStepReward,
    SelfReflectionReward,
)
from slm_rl.envs.reasoning_env import ReasoningTaskGenerator
from slm_rl.algorithms.grpo import GRPOTrainer
from slm_rl.algorithms.ppo import PPOTrainer
from slm_rl.evaluation.evaluator import Evaluator

__all__ = [
    "ModelConfig",
    "RLConfig",
    "GRPOConfig",
    "PPOConfig",
    "SLMPolicy",
    "SLMCritic",
    "TrajectoryBuffer",
    "RolloutBatch",
    "BaseRewardFunction",
    "ReasoningFormatReward",
    "MathCorrectnessReward",
    "CompositeReward",
    "RepetitionPenaltyReward",
    "ProcessStepReward",
    "SelfReflectionReward",
    "ReasoningTaskGenerator",
    "GRPOTrainer",
    "PPOTrainer",
    "Evaluator",
]

