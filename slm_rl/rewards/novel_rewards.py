"""Novel reward mechanisms tailored specifically for Small Language Model (SLM) reasoning:
1. SelfCorrectionBonusReward (rewards internal 'wait, let me double check' self-correction)
2. ComplexityCalibratedComputeReward (rewards test-time compute scaling proportional to problem difficulty)
3. TokenCreditAssignment (decomposes scalar completion rewards into structured per-token advantages)
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import torch

from slm_rl.rewards.base import BaseRewardFunction


class SelfCorrectionBonusReward(BaseRewardFunction):
    """
    Rewards 'Aha moments' / internal self-correction within <think>...</think>.
    Detects when a model evaluates a candidate intermediate thought, realizes a mistake
    using cognitive transition phrases ('wait', 'actually', 'let me recheck', 'hold on'),
    and provides a corrected calculation.
    
    This promotes test-time reflection and error recovery in Small Language Models.
    """
    def __init__(
        self,
        correction_bonus: float = 0.4,
        max_bonus: float = 0.5,
    ):
        self.correction_bonus = correction_bonus
        self.max_bonus = max_bonus
        # Markers indicating reconsideration or self-correction
        self.reconsider_markers = [
            r"\bwait\b",
            r"\bactually\b",
            r"\blet me (?:recheck|double[- ]check|recalculate|check again)\b",
            r"\bhold on\b",
            r"\bthat(?:'s| is) (?:incorrect|wrong|not right)\b",
            r"\bmistake\b",
        ]
        self.marker_regex = re.compile("|".join(self.reconsider_markers), re.IGNORECASE)
        self.eq_regex = re.compile(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)")

    def detect_self_correction(self, think_text: str) -> bool:
        """
        Detect whether the thinking trace contains a reconsideration marker
        accompanied by mathematical steps.
        """
        if not self.marker_regex.search(think_text):
            return False

        # Find position of first reconsideration marker
        marker_match = self.marker_regex.search(think_text)
        if not marker_match:
            return False

        pos = marker_match.start()
        before_text = think_text[:pos]
        after_text = think_text[pos:]

        # Must have text before (initial thought) and after (correction attempt)
        if len(before_text.strip()) < 10 or len(after_text.strip()) < 10:
            return False

        # If there's an equation after the marker, the model is re-computing
        has_post_eq = bool(self.eq_regex.search(after_text))
        return has_post_eq

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        rewards = []
        for text in completions:
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if not think_match:
                rewards.append(0.0)
                continue

            think_content = think_match.group(1)
            if self.detect_self_correction(think_content):
                rewards.append(self.correction_bonus)
            else:
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32)


class ComplexityCalibratedComputeReward(BaseRewardFunction):
    """
    Rewards test-time compute (number of reasoning words inside <think>)
    that is dynamically calibrated to the difficulty of the prompt.
    
    Penalizes:
    - Underthinking: solving complex 3-step problems in 3 words (shallow guessing).
    - Overthinking: spending 80 tokens rambling on a trivial 1-step arithmetic question (wasteful looping).
    """
    def __init__(
        self,
        base_target_words: int = 12,
        words_per_operation: int = 8,
        tolerance_factor: float = 0.6,
        reward_scale: float = 0.25,
    ):
        self.base_target_words = base_target_words
        self.words_per_operation = words_per_operation
        self.tolerance_factor = tolerance_factor
        self.reward_scale = reward_scale

    def estimate_problem_complexity(self, prompt: str) -> int:
        """Estimate number of operations / complexity in the prompt."""
        operators = len(re.findall(r"[\+\-\*\/]", prompt))
        # Word problem indicators
        word_markers = len(re.findall(r"\b(?:remaining|altogether|total|profit|spent|sold|received)\b", prompt, re.IGNORECASE))
        complexity = max(1, operators + word_markers)
        return complexity

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        rewards = []
        for prompt, text in zip(prompts, completions):
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if not think_match:
                rewards.append(0.0)
                continue

            think_words = len(think_match.group(1).strip().split())
            complexity = self.estimate_problem_complexity(prompt)

            # Target thinking budget for this specific problem
            target_words = self.base_target_words + (complexity * self.words_per_operation)
            lower_bound = target_words * (1.0 - self.tolerance_factor)
            upper_bound = target_words * (1.0 + self.tolerance_factor)

            if lower_bound <= think_words <= upper_bound:
                # In the optimal thinking zone: full calibrated reward
                r = self.reward_scale
            elif think_words < lower_bound:
                # Underthinking penalty (scaled by ratio)
                ratio = think_words / (lower_bound + 1e-6)
                r = self.reward_scale * max(0.0, ratio)
            else:
                # Overthinking / rambling penalty (gentle decay)
                excess = (think_words - upper_bound) / (upper_bound + 1e-6)
                r = self.reward_scale * max(0.0, 1.0 - 0.5 * excess)

            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32)


class TokenCreditAssignment:
    """
    Decomposes scalar sequence rewards into structured, segment-level token advantages.
    Solves the credit assignment dilemma in SLMs:
    Instead of assigning identical advantage to all response tokens,
    it identifies four semantic phases:
    1. Premise (<think> opening & problem parsing)
    2. Deductive Steps (intermediate equations)
    3. Self-Verification & Reflection
    4. Conclusion (<answer> final answer)
    
    Assigns higher reward weight to the actual deductive and conclusion phases,
    preventing valid reasoning from being penalized by formatting quirks or lucky final guesses.
    """
    @staticmethod
    def compute_segment_advantages(
        completion_text: str,
        seq_len: int,
        outcome_reward: float,
        process_reward: float,
        format_reward: float,
    ) -> torch.Tensor:
        """
        Produce a 1D tensor of shape (seq_len,) containing localized token advantages.
        """
        advantages = torch.full((seq_len,), outcome_reward, dtype=torch.float32)

        # Find where think ends and answer begins
        think_end_char = completion_text.find("</think>")
        answer_start_char = completion_text.find("<answer>")

        total_chars = max(1, len(completion_text))

        if think_end_char > 0:
            # Approximate token boundary using character proportion
            think_end_token = min(seq_len, max(1, int((think_end_char / total_chars) * seq_len)))
            # Reasoning tokens get heavily credited for intermediate process correctness
            advantages[:think_end_token] = (0.3 * outcome_reward) + (0.7 * process_reward)

        if answer_start_char > 0:
            answer_start_token = min(seq_len - 1, max(0, int((answer_start_char / total_chars) * seq_len)))
            # Final answer tokens are primarily tied to final outcome accuracy and format
            advantages[answer_start_token:] = (0.8 * outcome_reward) + (0.2 * format_reward)

        return advantages
