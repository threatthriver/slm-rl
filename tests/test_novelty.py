"""Tests for novel reward mechanisms, tournament GRPO, and curriculum learning."""

import os
import pytest
import torch
from slm_rl.rewards.novel_rewards import (
    SelfCorrectionBonusReward,
    ComplexityCalibratedComputeReward,
    TokenCreditAssignment,
)
from slm_rl.envs.curriculum_env import CurriculumReasoningEnv
from slm_rl.core.buffer import TrajectoryBuffer
from slm_rl.telemetry.cognition_visualizer import generate_html_report


def test_self_correction_bonus_reward():
    reward_fn = SelfCorrectionBonusReward(correction_bonus=0.4)

    prompts = ["Calculate 12 * 4"] * 3
    completions = [
        # 1. Clear self-correction: initial thought, marker ("wait"), corrected equation
        "<think>First 12 * 4 is 46... wait, that is incorrect! 12 * 4 = 48.</think><answer>48</answer>",
        # 2. Straightforward reasoning without mistake or correction
        "<think>Compute 12 * 4 directly to obtain 48.</think><answer>48</answer>",
        # 3. Empty thinking
        "<think></think><answer>48</answer>",
    ]

    scores = reward_fn(prompts, completions)
    assert pytest.approx(scores[0].item(), 0.01) == 0.4
    assert scores[1].item() == 0.0
    assert scores[2].item() == 0.0


def test_complexity_calibrated_compute_reward():
    reward_fn = ComplexityCalibratedComputeReward(
        base_target_words=10,
        words_per_operation=5,
        tolerance_factor=0.5,
        reward_scale=0.25,
    )

    # Prompt with 1 operator: target words ~ 15 (range: 7.5 to 22.5 words)
    simple_prompt = "Calculate 15 + 10."

    # Optimal length completion (~12 words in think)
    optimal_comp = "<think>First we add 15 and 10 together to easily arrive at 25.</think><answer>25</answer>"
    # Underthinking completion (2 words in think)
    under_comp = "<think>15+10=25</think><answer>25</answer>"
    # Overthinking completion (50 words in think)
    over_words = " ".join(["thinking"] * 50)
    over_comp = f"<think>{over_words}</think><answer>25</answer>"

    scores = reward_fn([simple_prompt] * 3, [optimal_comp, under_comp, over_comp])

    # Optimal should score higher than underthinking and severe overthinking
    assert scores[0].item() > scores[1].item()
    assert scores[0].item() > scores[2].item()


def test_token_credit_assignment():
    text = "<think>First step is 12 + 8 = 20.</think><answer>20</answer>"
    seq_len = 20
    adv = TokenCreditAssignment.compute_segment_advantages(
        completion_text=text,
        seq_len=seq_len,
        outcome_reward=1.0,
        process_reward=0.5,
        format_reward=0.8,
    )

    assert adv.shape == (seq_len,)
    # Reasoning tokens and answer tokens should reflect their respective segment weights
    assert not torch.isnan(adv).any()
    assert adv[0].item() != adv[-1].item()


def test_curriculum_environment_dynamic_adaptation():
    env = CurriculumReasoningEnv(
        seed=101,
        initial_tier=1,
        window_size=4,
        promotion_threshold=0.75,
        demotion_threshold=0.30,
    )

    assert env.current_tier == 1

    # Record 4 consecutive successes -> should promote to Tier 2
    env.record_outcomes([True, True, True, True])
    assert env.current_tier == 2

    # Record 4 consecutive failures -> should demote back to Tier 1
    env.record_outcomes([False, False, False, False])
    assert env.current_tier == 1

    # Verify problem generation matches active tier
    q1, target1 = env.generate_problem(tier=1)
    assert "Calculate" in q1
    assert isinstance(target1, float)


def test_tournament_advantage_mathematical_invariants():
    # 4 candidates with arbitrary scores
    rewards = torch.tensor([0.2, 0.5, 0.8, 1.0])
    group_ids = torch.tensor([0, 0, 0, 0])

    adv = TrajectoryBuffer.compute_tournament_advantages(rewards, group_ids, tau=0.5)

    assert adv.shape == (4,)
    # Invariant 1: Strictly bounded in [-1.0, 1.0]
    assert (adv.abs() <= 1.0).all()

    # Invariant 2: Exact zero-sum across group
    assert pytest.approx(adv.sum().item(), abs=1e-5) == 0.0

    # Invariant 3: Strictly monotonic with reward rank
    assert adv[0] < adv[1] < adv[2] < adv[3]


def test_leave_one_out_advantage_invariants():
    rewards = torch.tensor([2.0, 4.0, 6.0])
    group_ids = torch.tensor([0, 0, 0])

    # For element 0: r_0 = 2.0, LOO mean = (4.0 + 6.0) / 2 = 5.0 -> A_0 = 2.0 - 5.0 = -3.0
    # For element 2: r_2 = 6.0, LOO mean = (2.0 + 4.0) / 2 = 3.0 -> A_2 = 6.0 - 3.0 = +3.0
    adv = TrajectoryBuffer.compute_loo_advantages(rewards, group_ids)

    assert pytest.approx(adv[0].item(), 0.01) == -3.0
    assert pytest.approx(adv[1].item(), 0.01) == 0.0
    assert pytest.approx(adv[2].item(), 0.01) == 3.0


def test_html_cognition_report_generator(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    report_path = tmp_path / "report.html"

    # Write sample metrics
    import json
    with open(log_path, "w") as f:
        f.write(json.dumps({"step": 1, "mean_reward": 0.5, "loss": 0.2, "kl_divergence": 0.01, "avg_think_words": 15.0}) + "\n")

    generate_html_report(str(log_path), str(report_path))
    assert os.path.exists(report_path)

    with open(report_path) as f:
        content = f.read()
        assert "SLM-RL Cognition & Reasoning Inspector" in content
        assert "#1" in content
