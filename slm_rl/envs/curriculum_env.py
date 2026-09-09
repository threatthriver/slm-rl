"""Curriculum learning environment with dynamic difficulty adaptation for SLMs."""

import collections
import random
from typing import Dict, List, Optional, Tuple
from slm_rl.envs.reasoning_env import SYSTEM_PROMPT


class CurriculumReasoningEnv:
    """
    Dynamic Curriculum Environment for Small Language Models:
    Automatically regulates problem difficulty based on the agent's real-time accuracy.
    
    Tiers:
      1: Single-step addition / subtraction (A ± B)
      2: Two-step addition & subtraction (A + B - C)
      3: Multiplicative mixed arithmetic ((A * B) ± C)
      4: Division & multi-step arithmetic ((A / B) * C ± D)
      5: Multi-step word problems (GSM8k style)
      
    Promotion / Demotion:
      - Promotes when rolling accuracy over window W >= 0.75
      - Demotes when rolling accuracy over window W < 0.30
    """
    def __init__(
        self,
        seed: int = 42,
        initial_tier: int = 1,
        window_size: int = 10,
        promotion_threshold: float = 0.75,
        demotion_threshold: float = 0.30,
    ):
        self.rng = random.Random(seed)
        self.current_tier = initial_tier
        self.window_size = window_size
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.history = collections.deque(maxlen=window_size)

    def record_outcomes(self, correct_flags: List[bool]) -> None:
        """Record batch outcomes (list of booleans indicating correct final answers)."""
        for flag in correct_flags:
            self.history.append(1.0 if flag else 0.0)

        # Check for promotion or demotion
        if len(self.history) >= self.window_size:
            rolling_acc = sum(self.history) / len(self.history)
            if rolling_acc >= self.promotion_threshold and self.current_tier < 5:
                self.current_tier += 1
                self.history.clear()  # Reset history for the new tier
            elif rolling_acc < self.demotion_threshold and self.current_tier > 1:
                self.current_tier -= 1
                self.history.clear()

    def get_rolling_accuracy(self) -> float:
        """Return the current rolling accuracy estimate."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def generate_problem(self, tier: Optional[int] = None) -> Tuple[str, float]:
        """Generate a problem matching the specified or current difficulty tier."""
        active_tier = tier or self.current_tier

        if active_tier == 1:
            # Single-step addition / subtraction
            op = self.rng.choice(["+", "-"])
            a = self.rng.randint(5, 50)
            b = self.rng.randint(2, 30)
            target = a + b if op == "+" else a - b
            q = f"Calculate {a} {op} {b}."

        elif active_tier == 2:
            # Two-step addition / subtraction
            a = self.rng.randint(15, 80)
            b = self.rng.randint(10, 60)
            c = self.rng.randint(5, 40)
            target = a + b - c
            q = f"Calculate {a} + {b} - {c}."

        elif active_tier == 3:
            # Multiplicative mixed operations
            a = self.rng.randint(3, 12)
            b = self.rng.randint(3, 9)
            c = self.rng.randint(10, 50)
            op = self.rng.choice(["+", "-"])
            target = (a * b) + c if op == "+" else (a * b) - c
            q = f"What is ({a} * {b}) {op} {c}?"

        elif active_tier == 4:
            # Division & multi-step operations
            divisor = self.rng.randint(2, 10)
            quotient = self.rng.randint(4, 15)
            a = divisor * quotient
            c = self.rng.randint(2, 6)
            d = self.rng.randint(5, 25)
            target = (quotient * c) + d
            q = f"Evaluate (({a} / {divisor}) * {c}) + {d}."

        else:  # Tier 5: Word problems
            item = self.rng.choice(["apples", "books", "notebooks", "candies", "stickers"])
            packs = self.rng.randint(3, 8)
            per_pack = self.rng.randint(6, 12)
            gifted = self.rng.randint(4, 15)
            bought = self.rng.randint(5, 20)
            target = (packs * per_pack) - gifted + bought
            q = (
                f"Leo bought {packs} packs of {item} with {per_pack} {item} per pack. "
                f"He gave {gifted} {item} to his friend and then bought {bought} more individual {item}. "
                f"How many {item} does Leo have now?"
            )

        return q, float(target)

    def sample_batch(self, batch_size: int) -> Dict[str, List]:
        """Sample a batch of problems formatted with system prompt and 1-shot example."""
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
            "current_tier": self.current_tier,
            "rolling_accuracy": self.get_rolling_accuracy(),
        }
