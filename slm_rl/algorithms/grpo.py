"""Group Relative Policy Optimization (GRPO) Trainer."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

from slm_rl.algorithms.base import BaseRLTrainer
from slm_rl.config import GRPOConfig, ModelConfig, RLConfig
from slm_rl.core.buffer import TrajectoryBuffer
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.utils import compute_kl_divergence, masked_mean
from slm_rl.rewards.base import BaseRewardFunction


class GRPOTrainer(BaseRLTrainer):
    """
    GRPO Trainer (Group Relative Policy Optimization).
    Critic-free RL algorithm that normalizes rewards within groups of sampled completions.
    """
    def __init__(
        self,
        policy: SLMPolicy,
        reward_fn: BaseRewardFunction,
        rl_config: RLConfig,
        model_config: ModelConfig,
        grpo_config: Optional[GRPOConfig] = None,
        ref_policy: Optional[SLMPolicy] = None,
    ):
        super().__init__(policy, reward_fn, rl_config, model_config, ref_policy)
        self.grpo_config = grpo_config or GRPOConfig()
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=rl_config.learning_rate,
            weight_decay=rl_config.weight_decay,
        )

    def train_step(
        self,
        prompts: List[str],
        targets: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Perform one GRPO iteration:
        1. Sample G completions per prompt
        2. Score completions with reward function
        3. Compute group-standardized advantages
        4. Optimize policy with clipped surrogate loss + KL penalty
        """
        self.policy.eval()
        group_size = self.grpo_config.group_size
        num_prompts = len(prompts)
        total_samples = num_prompts * group_size

        # Expand prompts and targets for group sampling
        expanded_prompts = []
        expanded_targets = [] if targets is not None else None
        group_ids = []
        for i, p in enumerate(prompts):
            expanded_prompts.extend([p] * group_size)
            group_ids.extend([i] * group_size)
            if targets is not None:
                expanded_targets.extend([targets[i]] * group_size)

        group_ids_tensor = torch.tensor(group_ids, device=self.device, dtype=torch.long)

        # 1. Rollout: Generate completions from policy
        with torch.no_grad():
            rollout = self.policy.generate(
                prompts,
                num_return_sequences=group_size,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
            )

        seq_ids = rollout["seq_ids"]
        attention_mask = rollout["attention_mask"]
        prompt_seq_len = rollout["prompt_seq_len"]
        completions = rollout["completion_texts"]

        # 2. Compute log probabilities under old policy and reference policy
        with torch.no_grad():
            old_log_probs, response_mask = self.policy.compute_response_log_probs(
                seq_ids, attention_mask, prompt_seq_len
            )
            ref_log_probs, _ = self.ref_policy.compute_response_log_probs(
                seq_ids, attention_mask, prompt_seq_len, is_reference=True
            )

        # 3. Compute rewards
        rewards = self.reward_fn(
            expanded_prompts,
            completions,
            targets=expanded_targets,
        ).to(self.device)

        # 4. Compute group-relative advantages based on configured strategy
        if self.grpo_config.advantage_type == "tournament":
            advantages = TrajectoryBuffer.compute_tournament_advantages(
                rewards, group_ids_tensor, tau=self.grpo_config.tournament_tau
            )
        elif self.grpo_config.advantage_type == "loo":
            advantages = TrajectoryBuffer.compute_loo_advantages(rewards, group_ids_tensor)
        else:
            advantages = TrajectoryBuffer.compute_grpo_advantages(rewards, group_ids_tensor)

        # Expand advantage to token level (batch_size, 1)
        adv_expanded = advantages.unsqueeze(-1)


        # 5. Policy Optimization with Micro-Batch Gradient Accumulation
        self.policy.train()
        total_loss_val = 0.0
        total_policy_loss = 0.0
        total_kl_div = 0.0
        total_clip_frac = 0.0
        total_entropy = 0.0

        micro_batch_size = max(1, getattr(self.rl_config, "micro_batch_size", 2))
        num_chunks = (total_samples + micro_batch_size - 1) // micro_batch_size

        for epoch in range(self.rl_config.ppo_epochs):
            self.optimizer.zero_grad()
            indices = torch.randperm(total_samples)

            for chunk_i in range(num_chunks):
                start_idx = chunk_i * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, total_samples)
                mb_idx = indices[start_idx:end_idx]

                mb_seq_ids = seq_ids[mb_idx]
                mb_attention_mask = attention_mask[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_ref_log_probs = ref_log_probs[mb_idx]
                mb_adv = adv_expanded[mb_idx]
                mb_resp_mask = response_mask[mb_idx]

                mb_curr_log_probs, _ = self.policy.compute_response_log_probs(
                    mb_seq_ids, mb_attention_mask, prompt_seq_len
                )

                # Token probability ratio
                ratio = torch.exp(mb_curr_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.grpo_config.clip_range,
                    1.0 + self.grpo_config.clip_range,
                ) * mb_adv

                policy_loss = -torch.min(surr1, surr2)

                # KL divergence penalty
                kl = compute_kl_divergence(
                    mb_curr_log_probs,
                    mb_ref_log_probs,
                    method=self.grpo_config.kl_penalty_type,
                )

                # Total token loss
                token_loss = policy_loss + self.grpo_config.kl_coeff * kl

                # Aggregate masked loss across response tokens, scaled by number of chunks
                loss = masked_mean(token_loss, mb_resp_mask) / num_chunks

                # Backward pass
                loss.backward()

                # Record stats
                with torch.no_grad():
                    clipped = (ratio < 1.0 - self.grpo_config.clip_range) | (ratio > 1.0 + self.grpo_config.clip_range)
                    clip_frac = masked_mean(clipped.float(), mb_resp_mask).item()
                    mean_kl = masked_mean(kl, mb_resp_mask).item()
                    mean_pol_loss = masked_mean(policy_loss, mb_resp_mask).item()
                    approx_entropy = -masked_mean(mb_curr_log_probs, mb_resp_mask).item()

                    total_loss_val += (loss.item() * num_chunks) / num_chunks
                    total_policy_loss += mean_pol_loss / num_chunks
                    total_kl_div += mean_kl / num_chunks
                    total_clip_frac += clip_frac / num_chunks
                    total_entropy += approx_entropy / num_chunks

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.rl_config.max_grad_norm)
            self.optimizer.step()

        SLMPolicy.clear_memory()
        num_epochs = max(1, self.rl_config.ppo_epochs)


        # Compute reasoning length diagnostics (test-time compute scaling)
        import re
        think_lens = []
        for c in completions:
            m = re.search(r"<think>(.*?)</think>", c, re.DOTALL)
            think_lens.append(len(m.group(1).split()) if m else 0)
        avg_think_words = sum(think_lens) / len(think_lens) if think_lens else 0.0

        metrics = {
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "loss": total_loss_val / num_epochs,
            "policy_loss": total_policy_loss / num_epochs,
            "kl_divergence": total_kl_div / num_epochs,
            "clip_fraction": total_clip_frac / num_epochs,
            "policy_entropy": total_entropy / num_epochs,
            "avg_think_words": avg_think_words,
            "active_learning_fraction": (advantages.abs() > 1e-4).float().mean().item(),
        }

        self.step_count += 1
        self.log_metrics(metrics)
        return metrics

