"""Mathematical Invariant Tests for the SLM Reinforcement Learning System.

Validates that fundamental mathematical and algorithmic properties hold strictly:
1. Zero-KL Invariant: When policy equals reference, KL divergence must be strictly zero.
2. Zero-Sum Invariant: Standard GRPO advantages within any group sum to zero.
3. Tournament Skew-Symmetry: Pairwise Bradley-Terry win/loss advantages are anti-symmetric.
4. Leave-One-Out (LOO) Unbiasedness: Evaluator baseline excludes self-sample.
5. DPO Gradient Monotonicity: Winning completion log-prob decreases loss; losing increases loss.
"""

import math
import torch
import pytest

from slm_rl.core.buffer import TrajectoryBuffer
from slm_rl.core.utils import compute_kl_divergence


def test_zero_kl_invariant():
    """If pi_theta == pi_ref, KL divergence must be strictly 0.0 everywhere."""
    log_probs = torch.randn(4, 32)
    ref_log_probs = log_probs.clone()

    kl_low_var = compute_kl_divergence(log_probs, ref_log_probs, method="low_var_kl")
    kl_exact = compute_kl_divergence(log_probs, ref_log_probs, method="kl")

    assert torch.allclose(kl_low_var, torch.zeros_like(kl_low_var), atol=1e-6)
    assert torch.allclose(kl_exact, torch.zeros_like(kl_exact), atol=1e-6)


def test_grpo_advantage_zero_sum_invariant():
    """Within every rollout group, standard GRPO advantages must sum to zero."""
    torch.manual_seed(42)
    group_size = 6
    num_groups = 5

    # Random rewards across 5 groups of size 6
    rewards = torch.rand(num_groups * group_size) * 10.0
    group_ids = torch.repeat_interleave(torch.arange(num_groups), group_size)

    advs = TrajectoryBuffer.compute_grpo_advantages(rewards, group_ids)

    for g in range(num_groups):
        g_mask = (group_ids == g)
        group_adv_sum = advs[g_mask].sum().item()
        assert abs(group_adv_sum) < 1e-4, f"Group {g} advantages did not sum to zero: {group_adv_sum}"


def test_tournament_advantage_skew_symmetry():
    """In a 2-sample group, Bradley-Terry tournament advantages must be anti-symmetric."""
    rewards = torch.tensor([0.8, 0.2])
    group_ids = torch.tensor([0, 0])

    advs = TrajectoryBuffer.compute_tournament_advantages(rewards, group_ids, tau=0.5)

    assert len(advs) == 2
    # anti-symmetry: A[0] + A[1] == 0
    assert abs(advs[0].item() + advs[1].item()) < 1e-4
    assert advs[0].item() > 0.0  # Winner gets positive advantage
    assert advs[1].item() < 0.0  # Loser gets negative advantage


def test_leave_one_out_unbiasedness():
    """
    LOO advantage A_i = r_i - (1 / (G - 1)) * sum_{j != i} r_j.
    Verify exact analytical equivalence.
    """
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.5])
    group_ids = torch.tensor([0, 0, 0, 0])

    advs = TrajectoryBuffer.compute_loo_advantages(rewards, group_ids)

    G = 4
    for i in range(G):
        other_rewards = [rewards[j].item() for j in range(G) if j != i]
        expected_baseline = sum(other_rewards) / (G - 1)
        expected_adv = rewards[i].item() - expected_baseline
        assert math.isclose(advs[i].item(), expected_adv, abs_tol=1e-5)


def test_dpo_gradient_monotonicity():
    """
    DPO Loss = - log sigma(beta * ((pi_w - ref_w) - (pi_l - ref_l))).
    Verifies that increasing pi_w decreases loss, and increasing pi_l increases loss.
    """
    beta = 0.1
    ref_w = torch.tensor([-2.0])
    ref_l = torch.tensor([-3.0])

    pi_w = torch.tensor([-1.5], requires_grad=True)
    pi_l = torch.tensor([-3.5], requires_grad=True)

    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -torch.nn.functional.logsigmoid(logits)
    loss.backward()

    # Gradient w.r.t winning log-prob must be negative (encouraging higher probability)
    assert pi_w.grad.item() < 0.0
    # Gradient w.r.t losing log-prob must be positive (penalizing higher probability)
    assert pi_l.grad.item() > 0.0
