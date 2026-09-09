"""Curated GSM8K-style multi-step math word problems with exact verifiable ground truth."""

import random
from typing import Dict, List, Tuple
from slm_rl.envs.reasoning_env import SYSTEM_PROMPT


GSM8K_PROBLEMS: List[Tuple[str, str, float]] = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48 / 2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips.",
        72.0,
    ),
    (
        "Betty is saving money for a wallet which costs $100. Betty has only half of the money she needs ($50). "
        "Her parents gave her $15, and her grandparents gave her twice as much ($30). "
        "How much more money does Betty need to buy the wallet?",
        "Total Betty has = 50 + 15 + 30 = 95. Money needed = 100 - 95 = 5.",
        5.0,
    ),
    (
        "A bakery made 84 cookies in the morning. In the afternoon, they baked 36 more cookies. "
        "By closing time, 45 cookies were sold. How many cookies remain?",
        "Total baked = 84 + 36 = 120. Cookies remaining = 120 - 45 = 75.",
        75.0,
    ),
    (
        "A farmer has 14 hens. Each hen lays 6 eggs per week. "
        "If the farmer sells 50 eggs at the market, how many eggs are left for the week?",
        "Total eggs laid = 14 * 6 = 84. Eggs left = 84 - 50 = 34.",
        34.0,
    ),
    (
        "James buys 5 packs of baseball cards with 12 cards in each pack. "
        "He gives 18 cards to his brother and 12 cards to his friend. How many cards does James keep?",
        "Total cards = 5 * 12 = 60. Cards given away = 18 + 12 = 30. Remaining = 60 - 30 = 30.",
        30.0,
    ),
    (
        "A library has 150 books on history. Over the weekend, 32 books were checked out and 14 returned. "
        "How many history books are in the library now?",
        "History books left = 150 - 32 + 14 = 132.",
        132.0,
    ),
    (
        "Liam earned $45 on Monday and $55 on Tuesday mowing lawns. "
        "He spent $25 on gas and $15 on snacks. How much profit did he make?",
        "Total earnings = 45 + 55 = 100. Total expenses = 25 + 15 = 40. Profit = 100 - 40 = 60.",
        60.0,
    ),
    (
        "There are 4 boxes of pens. Each box contains 25 pens. "
        "If 38 pens are distributed to students, how many pens are left in total?",
        "Total pens = 4 * 25 = 100. Pens left = 100 - 38 = 62.",
        62.0,
    ),
]


class GSM8kTaskGenerator:
    """Task generator for GSM8K-style multi-step word problems."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.problems = list(GSM8K_PROBLEMS)

    def sample_batch(self, batch_size: int) -> Dict[str, List]:
        """Sample a batch of formatted word problems."""
        prompts = []
        questions = []
        targets = []
        solutions = []

        for _ in range(batch_size):
            q, sol, target = self.rng.choice(self.problems)
            prompt = f"{SYSTEM_PROMPT}\n\nProblem: {q}\n"
            prompts.append(prompt)
            questions.append(q)
            targets.append(target)
            solutions.append(sol)

        return {
            "prompts": prompts,
            "questions": questions,
            "targets": targets,
            "solutions": solutions,
        }
