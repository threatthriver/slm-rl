"""Fuzzing & Property-Based Tests for Equation Parsing and XML Tag Extractors."""

import random
import string
import pytest

from slm_rl.rewards.rule_based import MathCorrectnessReward, ReasoningFormatReward
from slm_rl.rewards.step_credit import StepCreditAssigner


def test_fuzz_equation_parser():
    """Fuzzes equation parsing with random arithmetic and malformed strings."""
    assigner = StepCreditAssigner()
    random.seed(42)

    operators = ["+", "-", "*", "/"]
    edge_numbers = ["0", "0.0", "-1", "1e5", "999999", "0.0001", "nan", "inf", "abc"]

    for _ in range(50):
        a = random.choice(edge_numbers)
        op = random.choice(operators)
        b = random.choice(edge_numbers)
        c = random.choice(edge_numbers)

        # Random malformed expressions
        fuzz_str = f"Step X: {a} {op} {b} = {c}."
        # Must not raise unhandled exception
        is_valid, has_eq = assigner.verify_step_math(fuzz_str)
        assert isinstance(is_valid, bool)
        assert isinstance(has_eq, bool)


def test_fuzz_xml_tags():
    """Fuzzes XML extraction with corrupted, nested, and truncated strings."""
    fmt_reward = ReasoningFormatReward()
    math_reward = MathCorrectnessReward()

    for _ in range(50):
        # Generate random noise string with occasional tags
        chunks = [
            "".join(random.choices(string.ascii_letters + string.punctuation, k=15))
            for _ in range(4)
        ]
        text = "<think>" + chunks[0] + "</think>" + "<answer>" + chunks[1] + "</answer>"
        if random.random() < 0.5:
            text = text.replace("</think>", "")  # Malform tag
        if random.random() < 0.3:
            text = text.replace("<answer>", "<answer><answer>")  # Duplicate tag

        r_fmt = fmt_reward(["dummy prompt"], [text], targets=[10.0])
        r_math = math_reward(["dummy prompt"], [text], targets=[10.0])

        assert not float(r_fmt[0].isnan())
        assert not float(r_math[0].isnan())
