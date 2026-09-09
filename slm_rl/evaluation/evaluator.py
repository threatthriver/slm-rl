"""Evaluation suite for assessing SLM policy performance."""

from typing import Any, Dict, List, Optional
import torch

from slm_rl.core.policy import SLMPolicy
from slm_rl.rewards.rule_based import ReasoningFormatReward, MathCorrectnessReward


class Evaluator:
    """Evaluates an SLM policy on reasoning tasks."""

    def __init__(self, policy: SLMPolicy):
        self.policy = policy
        self.format_reward = ReasoningFormatReward()
        self.math_reward = MathCorrectnessReward()

    @torch.no_grad()
    def evaluate(
        self,
        prompts: List[str],
        targets: List[Any],
        temperature: float = 0.0,  # Greedy for deterministic evaluation
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """
        Run greedy evaluation across test prompts.
        """
        self.policy.eval()

        rollout = self.policy.generate(
            prompts,
            num_return_sequences=1,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        completions = rollout["completion_texts"]
        format_scores = self.format_reward(prompts, completions, targets=targets)
        math_scores = self.math_reward(prompts, completions, targets=targets)

        total = len(prompts)
        correct_count = (math_scores == 1.0).sum().item()
        format_count = (format_scores >= 0.8).sum().item()

        accuracy = correct_count / total if total > 0 else 0.0
        format_rate = format_count / total if total > 0 else 0.0

        lengths = [len(c.split()) for c in completions]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0

        results = {
            "num_samples": total,
            "accuracy": accuracy,
            "format_adherence_rate": format_rate,
            "avg_word_length": avg_len,
            "avg_format_score": format_scores.mean().item(),
            "avg_math_score": math_scores.mean().item(),
            "samples": [
                {
                    "prompt": p,
                    "target": t,
                    "completion": c,
                    "format_score": fs.item(),
                    "math_score": ms.item(),
                }
                for p, t, c, fs, ms in zip(prompts, targets, completions, format_scores, math_scores)
            ],
        }
        return results
