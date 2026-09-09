"""Counterfactual Step-Level Advantage Optimization (CSAO) Trainer.

Optimizes policies by assigning counterfactual credit to individual deduction steps:
- Decomposes <think> reasoning chains into discrete deduction steps.
- Verifies intermediate equations to locate the pivot error.
- Awards positive reinforcement to valid steps preceding the error.
- Penalizes the pivot step and dampens downstream steps.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

from slm_rl.algorithms.base import BaseRLTrainer
from slm_rl.config import CSAOConfig, ModelConfig, RLConfig, DeliberationConfig
from slm_rl.core.deliberation import PhaseGatedDeliberationController
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.utils import compute_kl_divergence, masked_mean
from slm_rl.rewards.base import BaseRewardFunction
from slm_rl.rewards.step_credit import StepCreditAssigner


class CSAOTrainer(BaseRLTrainer):
    """
    CSAO Trainer: Step-Level Policy Gradient with Counterfactual Credit Assignment.
    """

    def __init__(
        self,
        policy: SLMPolicy,
        reward_fn: BaseRewardFunction,
        rl_config: RLConfig,
        model_config: ModelConfig,
        csao_config: Optional[CSAOConfig] = None,
        deliberation_config: Optional[DeliberationConfig] = None,
        ref_policy: Optional[SLMPolicy] = None,
    ):
        super().__init__(policy, reward_fn, rl_config, model_config, ref_policy)
        self.csao_config = csao_config or CSAOConfig()
        self.deliberation_config = deliberation_config or DeliberationConfig()
        self.step_assigner = StepCreditAssigner(self.csao_config)
        self.deliberation_controller = PhaseGatedDeliberationController(self.deliberation_config)

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
        Perform one CSAO step iteration:
        1. Sample completions
        2. Evaluate scalar outcome reward
        3. Decompose steps and assign counterfactual per-token advantages
        4. Optimize policy with step-weighted surrogate loss
        """
        self.policy.eval()
        num_samples = len(prompts)

        # 1. Rollout completions
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

        # 2. Compute old and reference log probabilities
        with torch.no_grad():
            old_log_probs, response_mask = self.policy.compute_response_log_probs(
                seq_ids, attention_mask, prompt_seq_len
            )
            ref_log_probs, _ = self.ref_policy.compute_response_log_probs(
                seq_ids, attention_mask, prompt_seq_len, is_reference=True
            )

        # 3. Base reward evaluation
        scalar_rewards = self.reward_fn(prompts, completions, targets=targets).to(self.device)

        # 4. Compute Step-Level Counterfactual Advantages
        resp_len = old_log_probs.size(1)
        token_advantages = torch.zeros((num_samples, resp_len), dtype=torch.float32, device=self.device)

        total_steps_count = 0
        total_pivots_count = 0
        valid_steps_count = 0

        for i in range(num_samples):
            comp_text = completions[i]
            r_val = float(scalar_rewards[i].item())
            is_correct = (r_val >= 0.7)  # Threshold for math correctness

            steps = self.step_assigner.decompose_steps(comp_text)
            total_steps_count += len(steps)

            if steps:
                valid_steps_count += sum(1 for s in steps if s["is_valid"])
                pivot = self.step_assigner.detect_pivot_error(steps)
                if pivot is not None:
                    total_pivots_count += 1

                step_advs = self.step_assigner.compute_step_advantages(steps, final_correct=is_correct)
                t_adv = self.step_assigner.map_to_token_advantages(
                    comp_text, steps, step_advs, resp_len, final_advantage=(r_val - 0.5)
                )
                token_advantages[i] = t_adv.to(self.device)
            else:
                token_advantages[i] = (r_val - 0.5)

        # 5. Policy Optimization with Micro-Batching
        self.policy.train()
        total_loss_val = 0.0
        total_policy_loss = 0.0
        total_kl_div = 0.0
        total_clip_frac = 0.0

        micro_batch_size = max(1, getattr(self.rl_config, "micro_batch_size", 2))
        num_chunks = (num_samples + micro_batch_size - 1) // micro_batch_size

        for epoch in range(self.rl_config.ppo_epochs):
            self.optimizer.zero_grad()
            indices = torch.randperm(num_samples)

            for chunk_i in range(num_chunks):
                start_idx = chunk_i * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, num_samples)
                mb_idx = indices[start_idx:end_idx]

                mb_seq_ids = seq_ids[mb_idx]
                mb_attention_mask = attention_mask[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_ref_log_probs = ref_log_probs[mb_idx]
                mb_token_adv = token_advantages[mb_idx]
                mb_resp_mask = response_mask[mb_idx]

                mb_curr_log_probs, _ = self.policy.compute_response_log_probs(
                    mb_seq_ids, mb_attention_mask, prompt_seq_len
                )

                # Ratio
                ratio = torch.exp(mb_curr_log_probs - mb_old_log_probs)

                # Clipped surrogate objective with per-token step advantage
                surr1 = ratio * mb_token_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.csao_config.clip_range,
                    1.0 + self.csao_config.clip_range,
                ) * mb_token_adv

                policy_loss = -torch.min(surr1, surr2)

                # KL divergence penalty
                kl = compute_kl_divergence(
                    mb_curr_log_probs,
                    mb_ref_log_probs,
                    method="low_var_kl",
                )

                token_loss = policy_loss + self.csao_config.kl_coeff * kl
                loss = masked_mean(token_loss, mb_resp_mask) / num_chunks

                # Add phase-gated deliberation loss if enabled
                if self.deliberation_config.enabled:
                    mb_completions = [completions[idx] for idx in mb_idx.tolist()]
                    phase_masks = self.deliberation_controller.compute_phase_masks(
                        mb_completions, resp_len, self.device
                    )
                    # Note: we pass mb_curr_log_probs for approx logits or evaluate through logits
                    ent_loss = -self.deliberation_config.explore_entropy_coeff * masked_mean(-mb_curr_log_probs, mb_resp_mask)
                    loss = loss + (ent_loss / num_chunks)

                loss.backward()

                with torch.no_grad():
                    clipped = (ratio < 1.0 - self.csao_config.clip_range) | (ratio > 1.0 + self.csao_config.clip_range)
                    clip_frac = masked_mean(clipped.float(), mb_resp_mask).item()
                    mean_kl = masked_mean(kl, mb_resp_mask).item()
                    mean_pol = masked_mean(policy_loss, mb_resp_mask).item()

                    total_loss_val += (loss.item() * num_chunks) / num_chunks
                    total_policy_loss += mean_pol / num_chunks
                    total_kl_div += mean_kl / num_chunks
                    total_clip_frac += clip_frac / num_chunks

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.rl_config.max_grad_norm)
            self.optimizer.step()

        SLMPolicy.clear_memory()

        metrics = {
            "loss": total_loss_val / self.rl_config.ppo_epochs,
            "policy_loss": total_policy_loss / self.rl_config.ppo_epochs,
            "kl_divergence": total_kl_div / self.rl_config.ppo_epochs,
            "clip_fraction": total_clip_frac / self.rl_config.ppo_epochs,
            "reward_mean": float(scalar_rewards.mean().item()),
            "reward_std": float(scalar_rewards.std().item()) if num_samples > 1 else 0.0,
            "pivot_rate": float(total_pivots_count / max(1, num_samples)),
            "valid_step_ratio": float(valid_steps_count / max(1, total_steps_count)),
            "mean_step_advantage": float(token_advantages.mean().item()),
        }

        return metrics
