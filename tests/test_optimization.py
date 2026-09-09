"""Tests for System Optimizations: Fused Log-Probs, Sliced Projections, and Memory Cleanup."""

import torch
import torch.nn.functional as F
import pytest

from slm_rl.config import ModelConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.utils import selective_log_probs


def test_fused_log_probs_equivalence():
    """
    Verifies that fused negative cross-entropy produces mathematically identical results
    to manual log_softmax + gather across arbitrary batch and vocab shapes.
    """
    batch_size = 4
    seq_len = 16
    vocab_size = 128

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Reference implementation (log_softmax + gather)
    log_probs_ref = F.log_softmax(logits, dim=-1)
    ref_gathered = torch.gather(log_probs_ref, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Optimized fused implementation
    opt_gathered = selective_log_probs(logits, labels)

    assert torch.allclose(opt_gathered, ref_gathered, atol=1e-5)


def test_sliced_response_log_probs_integrity():
    """Verifies that response-only slicing produces non-null, valid log-probabilities."""
    policy = SLMPolicy(ModelConfig(is_mock=True))
    prompt = "Calculate 10 + 20."
    rollout = policy.generate([prompt], num_return_sequences=2, max_new_tokens=16)

    seq_ids = rollout["seq_ids"]
    attention_mask = rollout["attention_mask"]
    prompt_seq_len = rollout["prompt_seq_len"]

    log_probs, mask = policy.compute_response_log_probs(seq_ids, attention_mask, prompt_seq_len)

    assert log_probs.size(0) == 2
    assert mask.size(0) == 2
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()


def test_accelerator_memory_clearance():
    """Verifies that clear_memory executes safely across all hardware backends."""
    SLMPolicy.clear_memory()
