"""Base trainer class for SLM Reinforcement Learning."""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Optional
import torch

from slm_rl.config import RLConfig, ModelConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.rewards.base import BaseRewardFunction


class BaseRLTrainer(ABC):
    """Abstract base class for SLM RL trainers."""

    def __init__(
        self,
        policy: SLMPolicy,
        reward_fn: BaseRewardFunction,
        rl_config: RLConfig,
        model_config: ModelConfig,
        ref_policy: Optional[SLMPolicy] = None,
    ):
        self.policy = policy
        self.reward_fn = reward_fn
        self.rl_config = rl_config
        self.model_config = model_config
        self.device = torch.device(model_config.device)

        # Create or assign reference policy for KL divergence
        if ref_policy is None:
            self.ref_policy = policy.create_reference_policy()
        else:
            self.ref_policy = ref_policy

        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def train_step(self, prompts: List[str], targets: Optional[List[Any]] = None) -> Dict[str, float]:
        """Perform a single rollout and policy optimization step."""
        pass

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save policy weights and training state."""
        save_dir = path or os.path.join(self.rl_config.output_dir, f"checkpoint_step_{self.step_count}")
        os.makedirs(save_dir, exist_ok=True)
        self.policy.model.save_pretrained(save_dir)
        self.policy.tokenizer.save_pretrained(save_dir)
        return save_dir

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Record step metrics in history and write to JSONL log."""
        import json
        metrics_with_step = {"step": self.step_count, **metrics}
        self.history.append(metrics_with_step)

        try:
            os.makedirs(self.rl_config.output_dir, exist_ok=True)
            log_path = os.path.join(self.rl_config.output_dir, "metrics.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_with_step) + "\n")
        except Exception:
            pass

