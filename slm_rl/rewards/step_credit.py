"""Counterfactual Step-Level Credit Assignment & Pivot Step Detection for SLMs.

In multi-step reasoning, Small Language Models frequently generate 2 or 3 correct
intermediate steps before committing a single arithmetic or logic error.
Standard scalar RL (PPO/GRPO) penalizes the entire trajectory, unlearning
valid deductive patterns.

This module isolates the exact 'pivot step' where truth value flipped,
assigns positive counterfactual reinforcement to prior valid steps, and
penalizes only the locus of failure.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import torch

from slm_rl.config import CSAOConfig


class StepCreditAssigner:
    """
    Parses reasoning trajectories into deduction steps, checks intermediate arithmetic validity,
    identifies the pivot step where failure occurred, and computes counterfactual step advantages.
    """

    def __init__(self, config: Optional[CSAOConfig] = None):
        self.config = config or CSAOConfig()
        # Equation pattern: e.g. "12 + 8 = 20" or "4.5 * 2 = 9"
        self.eq_regex = re.compile(
            r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
        )

    def decompose_steps(self, completion_text: str) -> List[Dict[str, Any]]:
        """
        Decomposes <think>...</think> into sequential reasoning steps.
        Falls back to sentence segmentation if no explicit numbered steps exist.
        """
        think_match = re.search(r"<think>(.*?)</think>", completion_text, re.DOTALL)
        if not think_match:
            return []

        think_content = think_match.group(1).strip()
        if not think_content:
            return []

        think_start_offset = think_match.start(1)

        # Look for explicit step markers like "Step 1:", "1.", "First,", etc.
        raw_steps = re.split(r"(?:(?<=\n)|(?<=^))\s*(?:Step\s*\d+[:\.]|\d+[\.:]|\bFirst,|\bNext,|\bThen,|\bFinally,)\s*", think_content)
        raw_steps = [s.strip() for s in raw_steps if s.strip()]

        # Fallback to sentence boundaries if step splitting yielded only 1 chunk
        if len(raw_steps) <= 1:
            raw_steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", think_content) if s.strip()]

        if not raw_steps:
            raw_steps = [think_content]

        steps: List[Dict[str, Any]] = []
        current_pos = 0
        for idx, text in enumerate(raw_steps):
            pos = think_content.find(text, current_pos)
            if pos == -1:
                pos = current_pos
            start_char = think_start_offset + pos
            end_char = start_char + len(text)
            current_pos = pos + len(text)

            is_valid, has_equation = self.verify_step_math(text)
            steps.append({
                "index": idx,
                "text": text,
                "start_char": start_char,
                "end_char": end_char,
                "has_equation": has_equation,
                "is_valid": is_valid,
            })

        return steps

    def verify_step_math(self, step_text: str) -> Tuple[bool, bool]:
        """
        Checks all equations in a step for mathematical validity.
        Returns (is_valid, has_equation).
        """
        matches = self.eq_regex.findall(step_text)
        if not matches:
            # Semantic step with no equation is deemed structurally sound
            return True, False

        for left_str, op, right_str, ans_str in matches:
            try:
                a = float(left_str)
                b = float(right_str)
                expected = float(ans_str)

                if op == "+":
                    actual = a + b
                elif op == "-":
                    actual = a - b
                elif op == "*":
                    actual = a * b
                elif op == "/":
                    if abs(b) < 1e-8:
                        return False, True
                    actual = a / b
                else:
                    continue

                if not math.isclose(actual, expected, abs_tol=1e-3, rel_tol=1e-3):
                    return False, True
            except (ValueError, ZeroDivisionError):
                return False, True

        return True, True

    def detect_pivot_error(self, steps: List[Dict[str, Any]]) -> Optional[int]:
        """
        Locates the first step index where a mathematical or logical error occurred.
        """
        for step in steps:
            if not step["is_valid"]:
                return step["index"]
        return None

    def compute_step_advantages(
        self,
        steps: List[Dict[str, Any]],
        final_correct: bool,
    ) -> List[float]:
        """
        Computes counterfactual advantages for each step:
        - If final answer is correct: all steps receive positive credit.
        - If final answer is wrong and pivot error found at k*:
            * Steps 0 .. k*-1 receive positive credit discounted towards the error.
            * Step k* receives pivot_penalty (e.g. -1.0).
            * Steps k*+1 .. receive downstream_penalty.
        - If final answer is wrong but no specific equation error found: uniform penalty.
        """
        num_steps = len(steps)
        if num_steps == 0:
            return []

        if final_correct:
            # All steps contributed to a successful reasoning chain
            return [self.config.step_reward] * num_steps

        pivot_idx = self.detect_pivot_error(steps)

        if pivot_idx is None:
            # Wrong final answer, but intermediate equations were valid (e.g., misapplied formula)
            return [self.config.downstream_penalty] * num_steps

        advantages = []
        for i in range(num_steps):
            if i < pivot_idx:
                # Valid deduction prior to failure: receive discounted positive credit
                distance = pivot_idx - 1 - i
                adv = self.config.step_reward * (self.config.gamma ** distance)
                advantages.append(adv)
            elif i == pivot_idx:
                # The exact locus of failure
                advantages.append(self.config.pivot_penalty)
            else:
                # Consequence of prior error
                advantages.append(self.config.downstream_penalty)

        return advantages

    def map_to_token_advantages(
        self,
        completion_text: str,
        steps: List[Dict[str, Any]],
        step_advantages: List[float],
        seq_len: int,
        final_advantage: float,
    ) -> torch.Tensor:
        """
        Projects character-level step advantages onto a token sequence of shape (seq_len,).
        """
        token_advs = torch.full((seq_len,), final_advantage, dtype=torch.float32)
        total_chars = max(1, len(completion_text))

        for step, adv in zip(steps, step_advantages):
            start_token = int((step["start_char"] / total_chars) * seq_len)
            end_token = max(start_token + 1, int((step["end_char"] / total_chars) * seq_len))
            start_token = min(seq_len - 1, max(0, start_token))
            end_token = min(seq_len, max(start_token + 1, end_token))
            token_advs[start_token:end_token] = adv

        # Ensure answer segment reflects final outcome
        answer_pos = completion_text.find("<answer>")
        if answer_pos != -1:
            ans_token = int((answer_pos / total_chars) * seq_len)
            ans_token = min(seq_len - 1, max(0, ans_token))
            token_advs[ans_token:] = final_advantage

        return token_advs
