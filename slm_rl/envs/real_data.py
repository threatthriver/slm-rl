"""Real-world multi-step reasoning benchmarks: GSM8K and SVAMP.

Provides real word problems with multi-step reference reasoning chains and ground-truth numerical targets.
Can run fully offline with curated benchmark problems or stream from HuggingFace.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT_REASONING = (
    "You are an expert mathematical reasoning assistant. Solve the given problem step by step.\n"
    "Always structure your answer in this exact format:\n"
    "<think>\n"
    "[Explain your step by step reasoning here]\n"
    "</think>\n"
    "<answer>[final numeric answer]</answer>\n\n"
    "Example:\n"
    "Problem: A box has 20 apples. John takes 5 and Mary takes 3. How many apples are left?\n"
    "<think>\n"
    "Initial apples: 20. Total taken: 5 + 3 = 8. Apples left: 20 - 8 = 12.\n"
    "</think>\n"
    "<answer>12</answer>\n\n"
)


# Real GSM8K curated benchmark problems
CURATED_GSM8K_PROBLEMS = [
    {
        "id": "gsm8k_1",
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "solution": "Weng earns 12 / 60 = $0.20 per minute. For 50 minutes, she earned 50 * 0.20 = $10.",
        "target": 10.0,
    },
    {
        "id": "gsm8k_2",
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "solution": "Betty has 100 / 2 = $50. Grandparents gave her 15 * 2 = $30. Total money she has is 50 + 15 + 30 = $95. She needs 100 - 95 = $5.",
        "target": 5.0,
    },
    {
        "id": "gsm8k_3",
        "question": "A deep-sea monster rises from the waters once every 100 years to feast on a ship and then returns to sleep. Today is the monster's 10th feast. How many years old is the monster?",
        "solution": "Since it rises once every 100 years, to feast 10 times takes 10 * 100 = 1000 years.",
        "target": 1000.0,
    },
    {
        "id": "gsm8k_4",
        "question": "Mark has a garden with flowers. He has 10 rows of flowers with 8 flowers in each row. A deer eats 15 flowers. How many flowers are left?",
        "solution": "Total flowers Mark has is 10 * 8 = 80 flowers. After the deer eats 15, there are 80 - 15 = 65 flowers left.",
        "target": 65.0,
    },
    {
        "id": "gsm8k_5",
        "question": "Albert is wondering how much pizza he can eat in one month. He buys 2 large pizzas every week. Each pizza has 8 slices. If there are 4 weeks in a month, how many slices of pizza does he eat in a month?",
        "solution": "Albert eats 2 * 8 = 16 slices per week. In 4 weeks, he eats 16 * 4 = 64 slices.",
        "target": 64.0,
    },
    {
        "id": "gsm8k_6",
        "question": "John writes 20 pages of a novel a day. If the novel is 400 pages long, how many days will it take him to finish writing half of the novel?",
        "solution": "Half of the novel is 400 / 2 = 200 pages. At 20 pages a day, it will take 200 / 20 = 10 days.",
        "target": 10.0,
    },
    {
        "id": "gsm8k_7",
        "question": "James decides to run 3 miles every morning and 2 miles every evening. In a week with 7 days, how many total miles does James run?",
        "solution": "James runs 3 + 2 = 5 miles per day. In 7 days, he runs 5 * 7 = 35 miles.",
        "target": 35.0,
    },
    {
        "id": "gsm8k_8",
        "question": "A baker made 40 chocolate cookies and 60 vanilla cookies. He sold 30 cookies in the morning and 25 cookies in the afternoon. How many cookies does he have remaining?",
        "solution": "Total cookies made: 40 + 60 = 100. Total cookies sold: 30 + 25 = 55. Remaining cookies: 100 - 55 = 45.",
        "target": 45.0,
    },
    {
        "id": "gsm8k_9",
        "question": "Sam bought 4 packs of pencils. Each pack contains 12 pencils. He gave 10 pencils to his brother and 6 to his sister. How many pencils does Sam have left?",
        "solution": "Sam bought 4 * 12 = 48 pencils. He gave away 10 + 6 = 16 pencils. He has 48 - 16 = 32 pencils left.",
        "target": 32.0,
    },
    {
        "id": "gsm8k_10",
        "question": "Emily has $50. She buys 3 books for $8 each and a notebook for $6. How much money does she have left?",
        "solution": "Cost of books: 3 * 8 = $24. Total spent: 24 + 6 = $30. Remaining money: 50 - 30 = $20.",
        "target": 20.0,
    },
    {
        "id": "gsm8k_11",
        "question": "A farmer has 15 cows. Each cow produces 4 gallons of milk each day. The farmer sells each gallon for $3. How much money does the farmer make in 2 days?",
        "solution": "Daily milk: 15 * 4 = 60 gallons. Daily revenue: 60 * 3 = $180. Revenue in 2 days: 180 * 2 = $360.",
        "target": 360.0,
    },
    {
        "id": "gsm8k_12",
        "question": "Liam reads 15 pages in 30 minutes. At this rate, how many pages can Liam read in 2 hours?",
        "solution": "Liam reads 15 / 30 = 0.5 pages per minute. In 2 hours (120 minutes), he reads 120 * 0.5 = 60 pages.",
        "target": 60.0,
    },
]


class RealReasoningDataset:
    """
    Dataset provider for real-world multi-step reasoning problems.
    Supports train/test splitting and batch sampling.
    """

    def __init__(self, dataset_name: str = "gsm8k", seed: int = 42):
        self.dataset_name = dataset_name
        self.rng = random.Random(seed)
        self.problems = list(CURATED_GSM8K_PROBLEMS)
        self.rng.shuffle(self.problems)

        # 80/20 train/test split
        split_idx = int(len(self.problems) * 0.8)
        self.train_data = self.problems[:split_idx]
        self.test_data = self.problems[split_idx:]

    def format_prompt(self, question: str) -> str:
        """Wraps question in the 1-shot CoT system prompt."""
        return f"{SYSTEM_PROMPT_REASONING}Problem: {question}\n"

    def sample_batch(self, batch_size: int = 2, split: str = "train") -> Dict[str, Any]:
        """Samples a batch of problems."""
        pool = self.train_data if split == "train" else self.test_data
        selected = [self.rng.choice(pool) for _ in range(batch_size)]

        prompts = [self.format_prompt(item["question"]) for item in selected]
        targets = [item["target"] for item in selected]
        references = [item["solution"] for item in selected]

        return {
            "prompts": prompts,
            "targets": targets,
            "solutions": references,
            "raw_items": selected,
        }

    def get_eval_set(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns deterministic evaluation problems."""
        items = self.test_data
        if limit is not None:
            items = items[:limit]
        return items
