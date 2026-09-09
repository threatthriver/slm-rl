"""Configuration data structures for the SLM RL system."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


def get_default_device() -> str:
    """Auto-detect the best available hardware accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    """Configuration for the Small Language Model."""
    model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: str = field(default_factory=get_default_device)
    torch_dtype: str = "float32"  # "float32", "bfloat16", "float16"
    max_prompt_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    is_mock: bool = False  # For fast unit testing without internet or heavy downloads
    use_lora: bool = False  # Parameter-Efficient Fine-Tuning
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05



@dataclass
class GRPOConfig:
    """Configuration specific to Group Relative Policy Optimization."""
    group_size: int = 4  # Number of completions per prompt (G)
    clip_range: float = 0.2  # PPO/GRPO clipping parameter epsilon
    kl_coeff: float = 0.05  # KL penalty weight beta
    kl_penalty_type: Literal["kl", "abs", "mse", "low_var_kl"] = "low_var_kl"
    advantage_type: Literal["standard", "tournament", "loo"] = "standard"
    tournament_tau: float = 0.5  # Soft-margin temperature for Bradley-Terry tournament



@dataclass
class PPOConfig:
    """Configuration specific to Proximal Policy Optimization."""
    clip_range: float = 0.2  # Epsilon for policy clipping
    value_clip_range: float = 0.2  # Epsilon for value clipping
    gamma: float = 1.0  # Discount factor for token-level rewards
    lam: float = 0.95  # GAE lambda parameter
    vf_coeff: float = 0.5  # Critic value loss weight
    entropy_coeff: float = 0.01  # Policy entropy bonus weight
    kl_coeff: float = 0.05  # KL penalty weight


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    beta: float = 0.1  # DPO temperature parameter
    label_smoothing: float = 0.0
    reference_free: bool = False  # If True, train without reference model
    margin_scale: float = 1.0


@dataclass
class CSAOConfig:
    """Configuration for Counterfactual Step-Level Advantage Optimization."""
    gamma: float = 0.9  # Credit decay factor across preceding valid steps
    step_reward: float = 0.25  # Credit for mathematically verified deduction steps
    pivot_penalty: float = -1.0  # Penalty for the first fallacious step (the pivot error)
    downstream_penalty: float = -0.2  # Dampened penalty for downstream steps
    clip_range: float = 0.2
    kl_coeff: float = 0.05


@dataclass
class DeliberationConfig:
    """Configuration for Phase-Gated Deliberation & Cognitive Anchoring."""
    enabled: bool = False
    explore_entropy_coeff: float = 0.03  # High entropy bonus for hypothesis exploration
    deduce_entropy_coeff: float = 0.0    # Neutral entropy for intermediate steps
    converge_entropy_coeff: float = -0.02 # Negative entropy penalty (forces sharp answer collapse)
    anchor_bonus: float = 0.25  # Bonus for invoking verification anchors ('Wait, let me double check')


@dataclass
class RLConfig:
    """Global configuration for RL training."""
    algorithm: Literal["grpo", "ppo", "dpo", "csao"] = "grpo"
    learning_rate: float = 1e-5
    critic_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    total_steps: int = 50
    batch_size: int = 2  # Number of prompts per batch
    micro_batch_size: int = 2  # Max sequences per forward/backward pass (keeps VRAM constant)
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 2  # Optimization passes over the rollout buffer
    seed: int = 42
    output_dir: str = "runs/slm_rl"
    save_every: int = 25
    eval_every: int = 10


