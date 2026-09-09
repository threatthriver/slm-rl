"""Unit tests for trajectory buffer and advantage computation."""

import pytest
import torch
from slm_rl.core.buffer import TrajectoryBuffer
from slm_rl.core.utils import compute_kl_divergence


def test_grpo_advantage_computation():
    # 2 prompt groups, 4 completions each (total 8 samples)
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 6.0, 8.0])
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    advantages = TrajectoryBuffer.compute_grpo_advantages(rewards, group_ids)

    # Within group 0: mean is 0.5, std is 0.5
    # (0 - 0.5) / 0.5 = -1.0, (1 - 0.5) / 0.5 = 1.0
    group0_adv = advantages[:4]
    assert pytest.approx(group0_adv[0].item(), 0.01) == -1.0
    assert pytest.approx(group0_adv[1].item(), 0.01) == -1.0
    assert pytest.approx(group0_adv[2].item(), 0.01) == 1.0
    assert pytest.approx(group0_adv[3].item(), 0.01) == 1.0

    # Within group 1: mean is 5.0, sum of advantages must sum to ~0
    group1_adv = advantages[4:]
    assert pytest.approx(group1_adv.mean().item(), 0.01) == 0.0


def test_gae_computation():
    # Batch of 1 sequence, length 3
    token_rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.6, 0.8]])
    mask = torch.tensor([[1, 1, 1]])

    adv, returns = TrajectoryBuffer.compute_gae(
        token_rewards, values, mask, gamma=1.0, lam=1.0
    )

    assert adv.shape == (1, 3)
    assert returns.shape == (1, 3)
    # Returns = advantages + values (before whitening)
    assert not torch.isnan(adv).any()
    assert not torch.isnan(returns).any()


def test_kl_divergence():
    log_probs = torch.tensor([-1.0, -2.0, -0.5])
    ref_log_probs = torch.tensor([-1.0, -1.5, -0.5])

    # Schulman's low variance KL: ratio - 1 - log(ratio) >= 0
    kl = compute_kl_divergence(log_probs, ref_log_probs, method="low_var_kl")
    assert (kl >= -1e-6).all()
    # At index 0, log_probs == ref_log_probs, KL should be 0
    assert pytest.approx(kl[0].item(), 1e-5) == 0.0
