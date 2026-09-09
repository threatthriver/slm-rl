"""Rollout and trajectory buffers for RL training."""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple
import torch

from slm_rl.core.utils import standardize, masked_mean


@dataclass
class RolloutBatch:
    """A batch of rollouts stored for policy updates."""
    seq_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    prompt_seq_len: int
    old_log_probs: torch.Tensor
    ref_log_probs: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    old_values: Optional[torch.Tensor] = None
    group_ids: Optional[torch.Tensor] = None
    completions: Optional[List[str]] = None


class TrajectoryBuffer:
    """
    Storage and advantage computation for policy rollouts.
    Supports both Group-Relative advantages (GRPO) and Generalized Advantage Estimation (PPO).
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        """Clear buffer data."""
        self.seq_ids: List[torch.Tensor] = []
        self.attention_mask: List[torch.Tensor] = []
        self.response_mask: List[torch.Tensor] = []
        self.old_log_probs: List[torch.Tensor] = []
        self.ref_log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.group_ids: List[torch.Tensor] = []
        self.completions: List[str] = []
        self.prompt_seq_len: int = 0

    @staticmethod
    def compute_grpo_advantages(
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute standardized group-relative advantages for GRPO.
        
        Args:
            rewards: (batch_size,) raw scalar rewards
            group_ids: (batch_size,) integer ID identifying which prompt group each sample belongs to
            eps: numerical stability epsilon
            
        Returns:
            advantages: (batch_size,) group-standardized advantages
        """
        advantages = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_ids)

        for gid in unique_groups:
            idx = (group_ids == gid)
            group_rewards = rewards[idx]
            if group_rewards.numel() > 1:
                std = group_rewards.std(unbiased=False)
                if std > 1e-6:
                    mean = group_rewards.mean()
                    raw_adv = (group_rewards - mean) / (std + eps)
                    advantages[idx] = torch.clamp(raw_adv, -4.0, 4.0)
                else:
                    # All completions in group received identical score (e.g. all failed or all succeeded)
                    advantages[idx] = 0.0
        return advantages

    @staticmethod
    def compute_tournament_advantages(
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
        tau: float = 0.5,
    ) -> torch.Tensor:
        """
        Bradley-Terry Tournament Relative Advantage:
        Evaluates each candidate via pairwise win/loss margins against all other
        candidates in its prompt group:
            A_i = 1/(G-1) * sum_{j != i} tanh((r_i - r_j) / tau)
            
        Guarantees:
        1. Bounded strictly in [-1.0, 1.0], preventing gradient explosions.
        2. Exact zero-sum: sum_i A_i = 0.
        3. Zero self-bias: candidate i does not contaminate its own baseline.
        """
        advantages = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_ids)

        for gid in unique_groups:
            idx = (group_ids == gid)
            group_r = rewards[idx]
            G = group_r.numel()
            if G > 1:
                # Pairwise difference matrix: diff[i, j] = r_i - r_j
                diff = group_r.unsqueeze(1) - group_r.unsqueeze(0)
                # Pairwise win margins: tanh(diff / tau)
                pairwise_margins = torch.tanh(diff / max(tau, 1e-4))
                # Average over all competing opponents (excluding self, diagonal is 0)
                adv = pairwise_margins.sum(dim=1) / (G - 1)
                advantages[idx] = adv
            else:
                advantages[idx] = 0.0

        return advantages

    @staticmethod
    def compute_loo_advantages(
        rewards: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Leave-One-Out (LOO) Baseline Advantage:
        Baseline for candidate i is computed strictly using the remaining G-1 candidates:
            A_i = r_i - (1 / (G - 1)) * sum_{j != i} r_j
        """
        advantages = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_ids)

        for gid in unique_groups:
            idx = (group_ids == gid)
            group_r = rewards[idx]
            G = group_r.numel()
            if G > 1:
                total_sum = group_r.sum()
                # loo_mean for each element i: (total_sum - r_i) / (G - 1)
                loo_mean = (total_sum - group_r) / (G - 1)
                advantages[idx] = group_r - loo_mean
            else:
                advantages[idx] = 0.0

        return advantages



    @staticmethod
    def compute_gae(
        token_rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token-level Generalized Advantage Estimation (GAE).
        
        Args:
            token_rewards: (batch_size, seq_len)
            values: (batch_size, seq_len)
            mask: (batch_size, seq_len) binary mask
            gamma: discount factor
            lam: GAE lambda parameter
            
        Returns:
            advantages: (batch_size, seq_len)
            returns: (batch_size, seq_len)
        """
        batch_size, seq_len = token_rewards.size()
        advantages = torch.zeros_like(token_rewards)
        last_gae_lam = 0.0

        # Backward scan across time steps
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_val = 0.0
            else:
                next_val = values[:, t + 1]

            delta = token_rewards[:, t] + gamma * next_val - values[:, t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            advantages[:, t] = last_gae_lam * mask[:, t]

        returns = advantages + values
        # Whiten advantages over active tokens
        advantages = standardize(advantages, mask=mask)
        return advantages, returns

    @staticmethod
    def get_batches(
        rollout: RolloutBatch,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[RolloutBatch]:
        """
        Yield mini-batches of RolloutBatch.
        """
        total_samples = rollout.seq_ids.size(0)
        indices = torch.randperm(total_samples) if shuffle else torch.arange(total_samples)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_idx = indices[start:end]

            yield RolloutBatch(
                seq_ids=rollout.seq_ids[batch_idx],
                attention_mask=rollout.attention_mask[batch_idx],
                response_mask=rollout.response_mask[batch_idx],
                prompt_seq_len=rollout.prompt_seq_len,
                old_log_probs=rollout.old_log_probs[batch_idx],
                ref_log_probs=rollout.ref_log_probs[batch_idx] if rollout.ref_log_probs is not None else None,
                rewards=rollout.rewards[batch_idx] if rollout.rewards is not None else None,
                advantages=rollout.advantages[batch_idx] if rollout.advantages is not None else None,
                returns=rollout.returns[batch_idx] if rollout.returns is not None else None,
                old_values=rollout.old_values[batch_idx] if rollout.old_values is not None else None,
                group_ids=rollout.group_ids[batch_idx] if rollout.group_ids is not None else None,
                completions=[rollout.completions[i] for i in batch_idx.tolist()] if rollout.completions else None,
            )
