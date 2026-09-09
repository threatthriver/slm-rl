"""Direct Preference Optimization (DPO) Trainer for SLMs.

Supports both online preference extraction from rollout groups and offline preference pairs:
L_DPO = - E_{(x, y_w, y_l)} [ log sigma( beta * ( log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)) ) ) ]
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from slm_rl.algorithms.base import BaseRLTrainer
from slm_rl.config import DPOConfig, ModelConfig, RLConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.utils import masked_mean
from slm_rl.rewards.base import BaseRewardFunction


class DPOTrainer(BaseRLTrainer):
    """
    Direct Preference Optimization Trainer.
    Extracts winning (y_w) and losing (y_l) completions dynamically from policy rollouts,
    optimizing implicit preference margins without an explicit reward model.
    """

    def __init__(
        self,
        policy: SLMPolicy,
        reward_fn: BaseRewardFunction,
        rl_config: RLConfig,
        model_config: ModelConfig,
        dpo_config: Optional[DPOConfig] = None,
        ref_policy: Optional[SLMPolicy] = None,
    ):
        super().__init__(policy, reward_fn, rl_config, model_config, ref_policy)
        self.dpo_config = dpo_config or DPOConfig()
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=rl_config.learning_rate,
            weight_decay=rl_config.weight_decay,
        )

    def extract_preference_pairs(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: torch.Tensor,
        group_size: int,
    ) -> Tuple[List[str], List[str], List[str], List[float], List[float]]:
        """
        Groups completions by prompt, picks the highest-reward completion as winner (y_w)
        and lowest-reward completion as loser (y_l).
        """
        pair_prompts = []
        winners = []
        losers = []
        winner_rewards = []
        loser_rewards = []

        num_prompts = len(prompts) // group_size

        for p_idx in range(num_prompts):
            start = p_idx * group_size
            end = start + group_size

            group_completions = completions[start:end]
            group_rewards = rewards[start:end]

            best_idx = torch.argmax(group_rewards).item()
            worst_idx = torch.argmin(group_rewards).item()

            # If all rewards are identical, pick distinct completions if available
            if best_idx == worst_idx and group_size > 1:
                worst_idx = (best_idx + 1) % group_size

            pair_prompts.append(prompts[start])
            winners.append(group_completions[best_idx])
            losers.append(group_completions[worst_idx])
            winner_rewards.append(float(group_rewards[best_idx].item()))
            loser_rewards.append(float(group_rewards[worst_idx].item()))

        return pair_prompts, winners, losers, winner_rewards, loser_rewards

    def compute_sequence_log_probs(
        self,
        policy: SLMPolicy,
        prompts: List[str],
        responses: List[str],
        is_reference: bool = False,
    ) -> torch.Tensor:
        """
        Computes sum of log probabilities for response tokens conditioned on prompt:
        log pi(y | x) = sum_{t} log pi(y_t | y_{<t}, x)
        Returns tensor of shape (batch_size,).
        """
        # Tokenize prompts and completions together
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        # Tokenize
        enc = policy.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.model_config.max_prompt_length + self.model_config.max_new_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Get prompt lengths to mask out prompt tokens
        prompt_enc = policy.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.model_config.max_prompt_length,
            return_tensors=None,
        )
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

        # Forward pass
        if is_reference and policy.is_lora:
            with policy.model.disable_adapter():
                outputs = policy.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        else:
            outputs = policy.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift logits and labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Create response-only mask (masking out prompt tokens)
        resp_mask = torch.zeros_like(shift_mask, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            # Prompt tokens are at the beginning (or adjust for left-pad)
            resp_mask[i, max(0, p_len - 1):] = True

        final_mask = shift_mask.bool() & resp_mask
        sum_log_probs = (per_token_log_probs * final_mask.float()).sum(dim=-1)

        return sum_log_probs

    def train_step(
        self,
        prompts: List[str],
        targets: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Perform one DPO iteration:
        1. Sample group of G completions per prompt
        2. Score completions to construct preference pairs (y_w, y_l)
        3. Compute log-ratio implicit margins under policy and ref_policy
        4. Compute DPO loss and optimize
        """
        self.policy.eval()
        group_size = 4
        num_prompts = len(prompts)

        # 1. Rollout completions
        with torch.no_grad():
            rollout = self.policy.generate(
                prompts,
                num_return_sequences=group_size,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
            )

        expanded_prompts = []
        expanded_targets = [] if targets is not None else None
        for i, p in enumerate(prompts):
            expanded_prompts.extend([p] * group_size)
            if targets is not None:
                expanded_targets.extend([targets[i]] * group_size)

        completions = rollout["completion_texts"]

        # 2. Score completions
        rewards = self.reward_fn(
            expanded_prompts,
            completions,
            targets=expanded_targets,
        ).to(self.device)

        # 3. Extract preference pairs
        pair_prompts, winners, losers, win_r, lose_r = self.extract_preference_pairs(
            expanded_prompts, completions, rewards, group_size
        )

        # 4. Compute reference log-probs (frozen)
        with torch.no_grad():
            ref_win_log_probs = self.compute_sequence_log_probs(
                self.ref_policy, pair_prompts, winners, is_reference=True
            )
            ref_lose_log_probs = self.compute_sequence_log_probs(
                self.ref_policy, pair_prompts, losers, is_reference=True
            )

        # 5. Compute policy log-probs and DPO loss
        self.policy.train()
        self.optimizer.zero_grad()

        pi_win_log_probs = self.compute_sequence_log_probs(
            self.policy, pair_prompts, winners, is_reference=False
        )
        pi_lose_log_probs = self.compute_sequence_log_probs(
            self.policy, pair_prompts, losers, is_reference=False
        )

        # Implicit rewards
        pi_log_ratio = pi_win_log_probs - pi_lose_log_probs
        ref_log_ratio = ref_win_log_probs - ref_lose_log_probs

        if self.dpo_config.reference_free:
            logits = self.dpo_config.beta * pi_log_ratio
        else:
            logits = self.dpo_config.beta * (pi_log_ratio - ref_log_ratio)

        # DPO loss with optional label smoothing
        # L = - log sigma(logits) = softplus(-logits)
        if self.dpo_config.label_smoothing > 0.0:
            eps = self.dpo_config.label_smoothing
            loss = -(1 - eps) * F.logsigmoid(logits) - eps * F.logsigmoid(-logits)
        else:
            loss = -F.logsigmoid(logits)

        mean_loss = loss.mean()
        mean_loss.backward()

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.rl_config.max_grad_norm)
        self.optimizer.step()
        SLMPolicy.clear_memory()

        # Diagnostics
        with torch.no_grad():
            implicit_margin = logits.mean().item()
            win_implicit_reward = (self.dpo_config.beta * (pi_win_log_probs - ref_win_log_probs)).mean().item()
            lose_implicit_reward = (self.dpo_config.beta * (pi_lose_log_probs - ref_lose_log_probs)).mean().item()
            accuracy = (logits > 0).float().mean().item()

        metrics = {
            "loss": float(mean_loss.item()),
            "dpo_margin": float(implicit_margin),
            "winner_reward_mean": sum(win_r) / max(1, len(win_r)),
            "loser_reward_mean": sum(lose_r) / max(1, len(lose_r)),
            "win_implicit_r": float(win_implicit_reward),
            "lose_implicit_r": float(lose_implicit_reward),
            "preference_accuracy": float(accuracy),
            "reward_mean": float(rewards.mean().item()),
        }

        return metrics
