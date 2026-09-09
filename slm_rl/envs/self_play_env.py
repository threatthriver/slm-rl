"""Prover-Skeptic Self-Play Duel Environment for SLMs.

Enables unsupervised reasoning refinement via an adversarial duel game:
1. Prover generates candidate reasoning chain.
2. Skeptic analyzes the chain and outputs either [Challenge] or [Consensus].
3. Game Referee evaluates the matrix to assign payoff rewards to both agents.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple
import torch

from slm_rl.rewards.step_credit import StepCreditAssigner


@dataclass
class DuelOutcome:
    """Outcome of a single Prover vs Skeptic duel round."""
    problem: str
    target: Any
    prover_text: str
    prover_correct: bool
    skeptic_text: str
    skeptic_challenged: bool
    skeptic_valid: bool
    prover_reward: float
    skeptic_reward: float
    outcome_tag: str  # 'MUTUAL_CONSENSUS', 'FALSE_CHALLENGE', 'SKEPTIC_CATCH', 'MISSED_FALLACY'


class SelfPlayDuelEnv:
    """
    Manages the Prover-Skeptic duel game loop and reward arbitration.
    """

    def __init__(
        self,
        mutual_consensus_reward: float = 1.0,
        skeptic_catch_reward: float = 1.5,
        false_challenge_penalty: float = -1.0,
        missed_fallacy_penalty: float = -1.0,
    ):
        self.r_consensus = mutual_consensus_reward
        self.r_catch = skeptic_catch_reward
        self.r_false_challenge = false_challenge_penalty
        self.r_missed_fallacy = missed_fallacy_penalty
        self.step_assigner = StepCreditAssigner()

    def format_skeptic_prompt(self, problem: str, prover_solution: str) -> str:
        """Constructs prompt for the Skeptic agent."""
        return (
            "You are a rigorous mathematical verifier. Analyze the following proposed solution step by step.\n"
            f"Problem: {problem}\n"
            f"Proposed Solution:\n{prover_solution}\n\n"
            "If you detect an arithmetic fallacy or flawed step, respond with:\n"
            "[Challenge] <Explain the exact calculation mistake>\n"
            "If all calculations and the final answer are completely sound, respond with:\n"
            "[Consensus] The deduction and arithmetic are correct."
        )

    def parse_skeptic_verdict(self, skeptic_response: str) -> Tuple[bool, str]:
        """
        Parses whether the Skeptic issued a challenge or declared consensus.
        Returns (is_challenge, reason).
        """
        if "[challenge]" in skeptic_response.lower():
            return True, skeptic_response
        elif "[consensus]" in skeptic_response.lower():
            return False, skeptic_response
        else:
            # If neither tag is present, check for critical keywords
            is_critique = bool(re.search(r"\b(incorrect|wrong|mistake|error|flaw)\b", skeptic_response, re.IGNORECASE))
            return is_critique, skeptic_response

    def arbitrate_duel(
        self,
        problem: str,
        target: Any,
        prover_solution: str,
        skeptic_response: str,
    ) -> DuelOutcome:
        """
        Arbitrates the payoff matrix between Prover and Skeptic.
        """
        # 1. Determine ground-truth correctness of Prover
        prover_correct = False
        ans_match = re.search(r"<answer>(.*?)</answer>", prover_solution, re.DOTALL)
        if ans_match:
            try:
                # Extract number
                nums = re.findall(r"[-+]?\d*\.?\d+", ans_match.group(1))
                if nums and target is not None:
                    pred = float(nums[-1])
                    prover_correct = abs(pred - float(target)) < 1e-3
            except (ValueError, IndexError):
                prover_correct = False

        # Also verify intermediate math steps
        steps = self.step_assigner.decompose_steps(prover_solution)
        steps_valid = all(s["is_valid"] for s in steps) if steps else False
        if not steps_valid:
            prover_correct = False

        # 2. Determine Skeptic verdict
        is_challenge, reason = self.parse_skeptic_verdict(skeptic_response)

        # 3. Payoff Matrix Arbitration
        if prover_correct:
            if not is_challenge:
                # Case 1: Mutual Consensus (both right)
                prover_reward = self.r_consensus
                skeptic_reward = self.r_consensus
                skeptic_valid = True
                tag = "MUTUAL_CONSENSUS"
            else:
                # Case 2: False Challenge (Skeptic hallucinated an error)
                prover_reward = self.r_consensus + 0.2  # Bonus for surviving skepticism
                skeptic_reward = self.r_false_challenge
                skeptic_valid = False
                tag = "FALSE_CHALLENGE"
        else:
            if is_challenge:
                # Case 3: Skeptic Catch (Skeptic successfully exposed fallacy)
                prover_reward = -1.0
                skeptic_reward = self.r_catch
                skeptic_valid = True
                tag = "SKEPTIC_CATCH"
            else:
                # Case 4: Missed Fallacy (Skeptic failed to detect mistake)
                prover_reward = -1.0
                skeptic_reward = self.r_missed_fallacy
                skeptic_valid = False
                tag = "MISSED_FALLACY"

        return DuelOutcome(
            problem=problem,
            target=target,
            prover_text=prover_solution,
            prover_correct=prover_correct,
            skeptic_text=skeptic_response,
            skeptic_challenged=is_challenge,
            skeptic_valid=skeptic_valid,
            prover_reward=prover_reward,
            skeptic_reward=skeptic_reward,
            outcome_tag=tag,
        )
