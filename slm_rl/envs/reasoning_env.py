"""Reasoning and arithmetic environment for SLM RL training."""

import random
from typing import Dict, List, Tuple


SYSTEM_PROMPT = (
    "You are an expert mathematical reasoning assistant. Solve the given problem step by step.\n"
    "Always structure your answer in this exact format:\n"
    "<think>\n"
    "[Explain your step by step reasoning here]\n"
    "</think>\n"
    "<answer>[final numeric answer]</answer>\n\n"
    "Example:\n"
    "Problem: Calculate 12 + 8 - 4.\n"
    "<think>\n"
    "First, compute 12 + 8 = 20. Next, subtract 4 from 20 to obtain 16.\n"
    "</think>\n"
    "<answer>16</answer>"
)



class ReasoningTaskGenerator:
    """
    Generates algorithmic arithmetic reasoning problems with exact verifiable answers.
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_problem(self) -> Tuple[str, float]:
        """
        Generate a single arithmetic or multi-step word problem and its ground-truth numerical value.
        """
        prob_type = self.rng.choice(["multistep", "mult_add", "div_sub", "word_problem"])

        if prob_type == "multistep":
            a = self.rng.randint(10, 80)
            b = self.rng.randint(10, 80)
            c = self.rng.randint(5, 50)
            target = a + b - c
            question = f"Calculate {a} + {b} - {c}."

        elif prob_type == "mult_add":
            a = self.rng.randint(3, 15)
            b = self.rng.randint(2, 9)
            c = self.rng.randint(10, 50)
            target = (a * b) + c
            question = f"What is ({a} * {b}) + {c}?"

        elif prob_type == "div_sub":
            b = self.rng.randint(2, 12)
            quotient = self.rng.randint(5, 20)
            a = b * quotient
            c = self.rng.randint(1, 10)
            target = quotient - c
            question = f"Evaluate ({a} / {b}) - {c}."

        else:  # word_problem
            item = self.rng.choice(["apples", "books", "loaves", "candies", "pens"])
            start = self.rng.randint(20, 60)
            add = self.rng.randint(10, 40)
            sub = self.rng.randint(5, 25)
            target = start + add - sub
            question = (
                f"A shop had {start} {item}. In the morning, they received {add} more {item}. "
                f"By evening, {sub} {item} were sold. How many {item} remain?"
            )

        return question, float(target)

    def sample_batch(self, batch_size: int) -> Dict[str, List]:
        """
        Sample a batch of formatted prompt strings and targets.
        """
        prompts = []
        questions = []
        targets = []

        for _ in range(batch_size):
            q, target = self.generate_problem()
            prompt = f"{SYSTEM_PROMPT}\n\nProblem: {q}\n"
            prompts.append(prompt)
            questions.append(q)
            targets.append(target)

        return {
            "prompts": prompts,
            "questions": questions,
            "targets": targets,
        }
