"""Phase-Gated Deliberation Controller & Cognitive Anchor Modulation for SLMs.

Addresses the SLM exploration-exploitation dilemma:
1. High entropy is desirable during early hypothesis formation (opening of <think>).
2. Near-zero entropy (sharp determinism) is required during arithmetic steps and final answer extraction.
3. Rewarding verification anchors ('Wait, let me check') when token entropy crosses an uncertainty threshold.
"""

import math
import re
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from slm_rl.config import DeliberationConfig


class PhaseGatedDeliberationController:
    """
    Modulates token-level exploration entropy based on the cognitive phase of reasoning:
    - Explore Phase: High entropy bonus (+ beta_explore)
    - Deduce Phase: Neutral entropy (beta_deduce)
    - Converge Phase: Negative entropy penalty / sharpness enforcement (- beta_converge)
    """

    def __init__(self, config: Optional[DeliberationConfig] = None):
        self.config = config or DeliberationConfig()
        self.anchor_patterns = [
            r"\bwait\b",
            r"\bhold on\b",
            r"\blet me recheck\b",
            r"\blet me check\b",
            r"\bactually\b",
        ]
        self.anchor_regex = re.compile("|".join(self.anchor_patterns), re.IGNORECASE)

    def compute_phase_masks(
        self,
        completions: List[str],
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Creates binary masks of shape (batch_size, seq_len) for:
        - 'explore': early reasoning exploration
        - 'deduce': intermediate calculation
        - 'converge': final answer and closing tags
        """
        batch_size = len(completions)
        explore_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        deduce_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        converge_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)

        for b, text in enumerate(completions):
            total_chars = max(1, len(text))
            think_start = text.find("<think>")
            think_end = text.find("</think>")
            ans_start = text.find("<answer>")

            t_start_tok = 0
            if think_start != -1:
                t_start_tok = int((think_start / total_chars) * seq_len)

            t_end_tok = seq_len
            if think_end != -1:
                t_end_tok = int((think_end / total_chars) * seq_len)

            a_start_tok = seq_len
            if ans_start != -1:
                a_start_tok = int((ans_start / total_chars) * seq_len)

            # Partition think segment into 35% explore and 65% deduce
            think_span = max(1, t_end_tok - t_start_tok)
            explore_split = t_start_tok + int(0.35 * think_span)

            explore_mask[b, t_start_tok:explore_split] = 1.0
            deduce_mask[b, explore_split:t_end_tok] = 1.0
            converge_mask[b, min(t_end_tok, a_start_tok):] = 1.0

        return {
            "explore": explore_mask,
            "deduce": deduce_mask,
            "converge": converge_mask,
        }

    def compute_phase_entropy_loss(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        phase_masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the phase-gated entropy regularization loss.
        logits: (batch_size, seq_len, vocab_size)
        attention_mask: (batch_size, seq_len)
        """
        # Numerically stable per-token entropy: H = - sum(p * log_p) = log_sum_exp(logits) - sum(p * logits)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        per_token_entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_len)

        # Apply attention mask
        valid_mask = attention_mask.float()

        def mean_entropy(mask: torch.Tensor) -> torch.Tensor:
            combined = mask * valid_mask
            denom = combined.sum().clamp(min=1.0)
            return (per_token_entropy * combined).sum() / denom

        h_explore = mean_entropy(phase_masks["explore"])
        h_deduce = mean_entropy(phase_masks["deduce"])
        h_converge = mean_entropy(phase_masks["converge"])

        # Loss: maximize explore entropy, keep deduce neutral, minimize converge entropy
        # Objective = + explore_coeff * H_exp + deduce_coeff * H_ded + converge_coeff * H_conv
        # Loss = - Objective
        entropy_loss = -(
            self.config.explore_entropy_coeff * h_explore
            + self.config.deduce_entropy_coeff * h_deduce
            + self.config.converge_entropy_coeff * h_converge
        )

        metrics = {
            "h_explore": float(h_explore.item()),
            "h_deduce": float(h_deduce.item()),
            "h_converge": float(h_converge.item()),
            "entropy_loss": float(entropy_loss.item()),
        }

        return entropy_loss, metrics

    def compute_cognitive_anchor_bonus(
        self,
        completion_text: str,
    ) -> float:
        """
        Detects whether an SLM deliberately paused to reconsider during reasoning.
        """
        if self.anchor_regex.search(completion_text):
            return self.config.anchor_bonus
        return 0.0
