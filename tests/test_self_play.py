"""Tests for Prover-Skeptic Self-Play Duel Environment."""

import pytest
from slm_rl.envs.self_play_env import SelfPlayDuelEnv


def test_mutual_consensus():
    env = SelfPlayDuelEnv()
    problem = "Calculate 15 + 10."
    target = 25.0
    prover_sol = "<think>\nStep 1: 15 + 10 = 25.\n</think>\n<answer>25</answer>"
    skeptic_resp = "[Consensus] The deduction 15 + 10 = 25 is completely correct."

    outcome = env.arbitrate_duel(problem, target, prover_sol, skeptic_resp)

    assert outcome.outcome_tag == "MUTUAL_CONSENSUS"
    assert outcome.prover_correct is True
    assert outcome.skeptic_challenged is False
    assert outcome.prover_reward == 1.0
    assert outcome.skeptic_reward == 1.0


def test_skeptic_catches_fallacy():
    env = SelfPlayDuelEnv()
    problem = "Calculate 20 * 3."
    target = 60.0
    # Prover makes an arithmetic error: 20 * 3 = 50
    prover_sol = "<think>\nStep 1: 20 * 3 = 50.\n</think>\n<answer>50</answer>"
    skeptic_resp = "[Challenge] Step 1 claims 20 * 3 = 50, but 20 * 3 is 60!"

    outcome = env.arbitrate_duel(problem, target, prover_sol, skeptic_resp)

    assert outcome.outcome_tag == "SKEPTIC_CATCH"
    assert outcome.prover_correct is False
    assert outcome.skeptic_challenged is True
    assert outcome.prover_reward == -1.0
    assert outcome.skeptic_reward == 1.5


def test_false_challenge_penalized():
    env = SelfPlayDuelEnv()
    problem = "Calculate 12 / 3."
    target = 4.0
    prover_sol = "<think>\nStep 1: 12 / 3 = 4.\n</think>\n<answer>4</answer>"
    # Skeptic hallucinates an error
    skeptic_resp = "[Challenge] I disagree with this solution."

    outcome = env.arbitrate_duel(problem, target, prover_sol, skeptic_resp)

    assert outcome.outcome_tag == "FALSE_CHALLENGE"
    assert outcome.prover_correct is True
    assert outcome.skeptic_challenged is True
    assert outcome.prover_reward > 1.0  # Bonus for surviving false critique
    assert outcome.skeptic_reward == -1.0


def test_missed_fallacy_penalized():
    env = SelfPlayDuelEnv()
    problem = "Calculate 10 + 20."
    target = 30.0
    prover_sol = "<think>\nStep 1: 10 + 20 = 99.\n</think>\n<answer>99</answer>"
    # Skeptic gullibly approves wrong answer
    skeptic_resp = "[Consensus] Looks good."

    outcome = env.arbitrate_duel(problem, target, prover_sol, skeptic_resp)

    assert outcome.outcome_tag == "MISSED_FALLACY"
    assert outcome.prover_correct is False
    assert outcome.skeptic_challenged is False
    assert outcome.prover_reward == -1.0
    assert outcome.skeptic_reward == -1.0
