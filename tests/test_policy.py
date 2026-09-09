"""Unit tests for SLMPolicy and critic wrappers."""

import pytest
import torch
from slm_rl.config import ModelConfig
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.critic import SLMCritic


@pytest.fixture
def mock_policy():
    cfg = ModelConfig(is_mock=True, device="cpu")
    return SLMPolicy(cfg)


def test_mock_policy_initialization(mock_policy):
    assert mock_policy is not None
    assert mock_policy.tokenizer is not None
    assert mock_policy.model is not None


def test_policy_generation(mock_policy):
    prompts = ["Problem: 2 + 2", "Problem: 5 - 3"]
    rollout = mock_policy.generate(
        prompts,
        max_new_tokens=10,
        num_return_sequences=2,
        temperature=0.8,
    )

    assert "seq_ids" in rollout
    assert "completion_texts" in rollout
    # 2 prompts * 2 return sequences = 4 completions
    assert len(rollout["completion_texts"]) == 4
    assert rollout["seq_ids"].size(0) == 4


def test_response_log_probs(mock_policy):
    prompts = ["Problem: 1 + 1"]
    rollout = mock_policy.generate(prompts, max_new_tokens=8, num_return_sequences=1)

    log_probs, mask = mock_policy.compute_response_log_probs(
        rollout["seq_ids"],
        rollout["attention_mask"],
        rollout["prompt_seq_len"],
    )

    assert log_probs.shape == mask.shape
    # Log probs should be non-positive
    assert (log_probs <= 0.0).all()
    # Mask should be binary
    assert ((mask == 0) | (mask == 1)).all()


def test_reference_policy_frozen(mock_policy):
    ref_policy = mock_policy.create_reference_policy()

    for param in ref_policy.parameters():
        assert not param.requires_grad
