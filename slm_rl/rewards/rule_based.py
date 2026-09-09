"""Rule-based reward functions and verifiable reasoning evaluators."""

import re
from typing import Any, Dict, List, Optional, Tuple
import torch

from slm_rl.rewards.base import BaseRewardFunction


class ReasoningFormatReward(BaseRewardFunction):
    """
    Reward function that verifies reasoning tag structure:
    <think>
    step by step thinking (must be non-trivial, >= min_think_chars)
    </think>
    <answer>
    final answer
    </answer>
    
    Guards against reward hacking:
    - Rejects empty or whitespace-only <think></think> tags.
    - Penalizes unclosed tags.
    - Verifies sequential ordering (<think> before <answer>).
    """
    def __init__(
        self,
        think_weight: float = 0.4,
        answer_weight: float = 0.4,
        order_weight: float = 0.2,
        min_think_chars: int = 10,
    ):
        self.think_weight = think_weight
        self.answer_weight = answer_weight
        self.order_weight = order_weight
        self.min_think_chars = min_think_chars

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        rewards = []
        for text in completions:
            r = 0.0

            # 1. Verify <think>...</think> with non-trivial content
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            has_valid_think = False
            if think_match:
                content = think_match.group(1).strip()
                if len(content) >= self.min_think_chars:
                    has_valid_think = True
                    r += self.think_weight

            # 2. Verify <answer>...</answer> with non-empty content
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            has_valid_answer = False
            if answer_match:
                content = answer_match.group(1).strip()
                if len(content) > 0:
                    has_valid_answer = True
                    r += self.answer_weight

            # 3. Check sequential tag ordering: <think> must close before <answer> begins
            if has_valid_think and has_valid_answer:
                think_end = text.find("</think>")
                answer_start = text.find("<answer>")
                if 0 <= think_end < answer_start:
                    r += self.order_weight

            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32)


class RepetitionPenaltyReward(BaseRewardFunction):
    """
    Detects degenerative repeating n-gram loops (a common mode collapse in SLMs)
    and deducts reward proportionally.
    """
    def __init__(self, n: int = 3, threshold: float = 0.25, penalty_weight: float = 0.5):
        self.n = n
        self.threshold = threshold
        self.penalty_weight = penalty_weight

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        penalties = []
        for text in completions:
            words = text.strip().split()
            if len(words) < self.n + 2:
                penalties.append(0.0)
                continue

            ngrams = [tuple(words[i:i + self.n]) for i in range(len(words) - self.n + 1)]
            unique_ngrams = set(ngrams)
            rep_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))

            if rep_ratio > self.threshold:
                # Deduct penalty proportional to repetition severity
                penalty = -self.penalty_weight * ((rep_ratio - self.threshold) / (1.0 - self.threshold))
                penalties.append(penalty)
            else:
                penalties.append(0.0)

        return torch.tensor(penalties, dtype=torch.float32)



class MathCorrectnessReward(BaseRewardFunction):
    """
    Extracts the numerical answer and compares with target ground truth.
    Supports answers inside <answer>...</answer> or fallback to last number.
    """
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    @staticmethod
    def extract_answer(text: str) -> Optional[float]:
        """Extract number from <answer> tags or end of text."""
        # 1. Try extracting from <answer> tags first
        match = re.search(r"<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # 2. Try generic <answer>...</answer> content
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            numbers = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", content)
            if numbers:
                try:
                    return float(numbers[-1])
                except ValueError:
                    pass

        # 3. Fallback: last number in the entire response
        numbers = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        rewards = []
        if targets is None:
            return torch.zeros(len(completions), dtype=torch.float32)

        for text, target in zip(completions, targets):
            extracted = self.extract_answer(text)
            if extracted is None:
                rewards.append(0.0)
            else:
                try:
                    target_val = float(target)
                    is_correct = abs(extracted - target_val) <= self.tolerance
                    rewards.append(1.0 if is_correct else 0.0)
                except (ValueError, TypeError):
                    rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32)


class ProcessStepReward(BaseRewardFunction):
    """
    Process-Level Reward (PRM / Process Supervision):
    Inspects intermediate arithmetic statements within <think>...</think>
    (e.g., "12 + 8 = 20" or "45 - 20 = 25") and verifies their mathematical truth.
    Rewards the model for valid deductive steps, converting sparse outcome rewards
    into a dense learning signal for SLMs.
    """
    def __init__(self, step_reward: float = 0.2, max_reward: float = 0.6, tolerance: float = 1e-4):
        self.step_reward = step_reward
        self.max_reward = max_reward
        self.tolerance = tolerance
        # Regex to capture arithmetic equations like "A + B = C", "A * B = C", etc.
        self.eq_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
        )

    def evaluate_equation(self, a_str: str, op: str, b_str: str, c_str: str) -> bool:
        try:
            a, b, c = float(a_str), float(b_str), float(c_str)
            if op == "+":
                expected = a + b
            elif op == "-":
                expected = a - b
            elif op == "*":
                expected = a * b
            elif op == "/":
                if abs(b) < 1e-8:
                    return False
                expected = a / b
            else:
                return False
            return abs(expected - c) <= self.tolerance
        except (ValueError, ZeroDivisionError):
            return False

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
            equations = self.eq_pattern.findall(think_content)
            if not equations:
                rewards.append(0.0)
                continue

            valid_steps = 0
            for a, op, b, c in equations:
                if self.evaluate_equation(a, op, b, c):
                    valid_steps += 1

            r = min(valid_steps * self.step_reward, self.max_reward)
            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32)


class SelfReflectionReward(BaseRewardFunction):
    """
    Rewards deductive and self-checking markers inside <think>...</think>
    (e.g., 'check', 'verify', 'wait', 'therefore', 'first', 'next').
    Encourages structured step-by-step reasoning.
    """
    def __init__(self, reward_per_marker: float = 0.05, max_reward: float = 0.2):
        self.reward_per_marker = reward_per_marker
        self.max_reward = max_reward
        self.markers = ["first", "next", "then", "therefore", "check", "verify", "since", "so"]

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        rewards = []
        for text in completions:
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
            if not think_match:
                rewards.append(0.0)
                continue

            content = think_match.group(1).lower()
            count = sum(1 for m in self.markers if m in content)
            r = min(count * self.reward_per_marker, self.max_reward)
            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32)


class CompositeReward(BaseRewardFunction):
    """
    Weighted combination of multiple reward functions.
    """
    def __init__(self, reward_fns_with_weights: List[Any]):
        normalized = []
        for item in reward_fns_with_weights:
            if isinstance(item, tuple):
                normalized.append(item)
            else:
                normalized.append((item, 1.0))
        self.reward_fns = normalized

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: Optional[List[Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        total_rewards = torch.zeros(len(completions), dtype=torch.float32)
        for fn, weight in self.reward_fns:
            fn_reward = fn(prompts, completions, targets, **kwargs)
            total_rewards += weight * fn_reward
        return total_rewards


