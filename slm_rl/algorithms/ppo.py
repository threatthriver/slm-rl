"""Proximal Policy Optimization (PPO) Trainer for SLMs."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

from slm_rl.algorithms.base import BaseRLTrainer
from slm_rl.config import PPOConfig, ModelConfig, RLConfig
from slm_rl.core.buffer import TrajectoryBuffer
from slm_rl.core.critic import SLMCritic
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.utils import compute_kl_divergence, masked_mean
from slm_rl.rewards.base import BaseRewardFunction


class PPOTrainer(BaseRLTrainer):
    """
    PPO Trainer with Generalized Advantage Estimation (GAE),
    value function clipping, and KL penalty.
    """
    def __init__(
        self,
        policy: SLMPolicy,
        critic: SLMCritic,
        reward_fn: BaseRewardFunction,
        rl_config: RLConfig,
        model_config: ModelConfig,
        ppo_config: Optional[PPOConfig] = None,
        ref_policy: Optional[SLMPolicy] = None,
    ):
        super().__init__(policy, reward_fn, rl_config, model_config, ref_policy)
        self.critic = critic
        self.ppo_config = ppo_config or PPOConfig()

        self.policy_optimizer = AdamW(
            self.policy.parameters(),
            lr=rl_config.learning_rate,
            weight_decay=rl_config.weight_decay,
        )
        self.critic_optimizer = AdamW(
            self.critic.parameters(),
            lr=rl_config.critic_learning_rate,
            weight_decay=rl_config.weight_decay,
        )

    def train_step(
        self,
        prompts: List[str],
        targets: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Perform one PPO training iteration.
        """
        self.policy.eval()
        self.critic.eval()

        # 1. Rollout: Generate completions
        with torch.no_grad():
            rollout = self.policy.generate(
                prompts,
                num_return_sequences=1,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
            )

        seq_ids = rollout["seq_ids"]
        attention_mask = rollout["attention_mask"]
        prompt_seq_len = rollout["prompt_seq_len"]
        completions = rollout["completion_texts"]

        # 2. Compute old log probs, ref log probs, and old values
        with torch.no_grad():
            old_log_probs, response_mask = self.policy.compute_response_log_probs(
                seq_ids, attention_mask, prompt_seq_len
            )
            ref_log_probs, _ = self.ref_policy.compute_response_log_probs(
                seq_ids, attention_mask, prompt_seq_len, is_reference=True
            )

            old_values, _ = self.critic.compute_response_values(
                seq_ids, attention_mask, prompt_seq_len
            )

            # Ensure value sequence length matches response length
            resp_len = response_mask.size(1)
            old_values = old_values[:, :resp_len]

            # 3. Compute rewards
            seq_rewards = self.reward_fn(prompts, completions, targets=targets).to(self.device)

            # 4. Construct token-level rewards with KL penalty
            kl = compute_kl_divergence(old_log_probs, ref_log_probs, method="low_var_kl")
            token_rewards = -self.ppo_config.kl_coeff * kl

            # Add sequence reward at the final response token
            batch_size = seq_rewards.size(0)
            for i in range(batch_size):
                active_indices = torch.where(response_mask[i] == 1)[0]
                if len(active_indices) > 0:
                    last_idx = active_indices[-1]
                    token_rewards[i, last_idx] += seq_rewards[i]

            # 5. Compute GAE advantages and returns
            advantages, returns = TrajectoryBuffer.compute_gae(
                token_rewards,
                old_values,
                response_mask,
                gamma=self.ppo_config.gamma,
                lam=self.ppo_config.lam,
            )

        # 6. Policy and Critic Optimization
        self.policy.train()
        self.critic.train()

        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_kl = 0.0

        for epoch in range(self.rl_config.ppo_epochs):
            self.policy_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            curr_log_probs, _ = self.policy.compute_response_log_probs(
                seq_ids, attention_mask, prompt_seq_len
            )
            curr_values, _ = self.critic.compute_response_values(
                seq_ids, attention_mask, prompt_seq_len
            )
            curr_values = curr_values[:, :resp_len]

            # Policy Loss
            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.ppo_config.clip_range,
                1.0 + self.ppo_config.clip_range,
            ) * advantages
            policy_loss = -torch.min(surr1, surr2)
            mean_policy_loss = masked_mean(policy_loss, response_mask)

            # Value Loss with clipping
            val_clipped = old_values + torch.clamp(
                curr_values - old_values,
                -self.ppo_config.value_clip_range,
                self.ppo_config.value_clip_range,
            )
            vf1 = (curr_values - returns) ** 2
            vf2 = (val_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(vf1, vf2)
            mean_value_loss = masked_mean(value_loss, response_mask)

            # Total joint loss
            loss = mean_policy_loss + self.ppo_config.vf_coeff * mean_value_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.rl_config.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.rl_config.max_grad_norm)

            self.policy_optimizer.step()
            self.critic_optimizer.step()

            with torch.no_grad():
                total_pol_loss += mean_policy_loss.item()
                total_val_loss += mean_value_loss.item()
                total_kl += masked_mean(kl, response_mask).item()

        num_epochs = self.rl_config.ppo_epochs
        metrics = {
            "mean_reward": seq_rewards.mean().item(),
            "loss": (total_pol_loss + self.ppo_config.vf_coeff * total_val_loss) / num_epochs,
            "policy_loss": total_pol_loss / num_epochs,
            "value_loss": total_val_loss / num_epochs,
            "kl_divergence": total_kl / num_epochs,
        }

        self.step_count += 1
        self.log_metrics(metrics)
        return metrics
