#!/usr/bin/env python3
"""
CLI Training Script for SLM Reinforcement Learning (GRPO & PPO).
"""

import argparse
import os
import sys
import time
from typing import Optional
from tqdm import tqdm

from slm_rl.config import (
    ModelConfig, RLConfig, GRPOConfig, PPOConfig, DPOConfig, CSAOConfig, DeliberationConfig, get_default_device
)
from slm_rl.core.policy import SLMPolicy
from slm_rl.core.critic import SLMCritic
from slm_rl.rewards.rule_based import ReasoningFormatReward, MathCorrectnessReward, CompositeReward
from slm_rl.envs.reasoning_env import ReasoningTaskGenerator
from slm_rl.algorithms.grpo import GRPOTrainer
from slm_rl.algorithms.ppo import PPOTrainer
from slm_rl.algorithms.dpo import DPOTrainer
from slm_rl.algorithms.step_csao import CSAOTrainer
from slm_rl.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Small Language Models with Reinforcement Learning")
    parser.add_argument("--algorithm", type=str, choices=["grpo", "ppo", "dpo", "csao"], default="grpo",
                        help="RL algorithm: 'grpo', 'ppo', 'dpo' (Direct Preference Opt), or 'csao' (Counterfactual Step-Level)")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="HuggingFace model name/path (e.g. HuggingFaceTB/SmolLM2-135M-Instruct, openai-community/gpt2)")
    parser.add_argument("--mock", action="store_true",
                        help="Use a lightweight synthetic model for testing without downloading weights")
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA adapters (freezes base model for zero-memory reference pass)")
    parser.add_argument("--steps", type=int, default=20, help="Total training steps")
    parser.add_argument("--deliberation", action="store_true",
                        help="Enable Phase-Gated Deliberation (modulates entropy across cognitive reasoning phases)")
    parser.add_argument("--self-play", action="store_true",
                        help="Enable Prover-Skeptic Duel Self-Play evaluation")

    parser.add_argument("--batch-size", type=int, default=2, help="Number of prompts per batch")
    parser.add_argument("--micro-batch", type=int, default=2, help="Max sequences per backward chunk (constant VRAM)")
    parser.add_argument("--group-size", type=int, default=4, help="Completions per prompt for GRPO")
    parser.add_argument("--advantage-type", type=str, choices=["standard", "tournament", "loo"], default="tournament",
                        help="Advantage estimation: 'tournament' (Bradley-Terry pairwise), 'loo' (Leave-one-out), 'standard'")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable dynamic 5-tier difficulty adaptation based on rolling pass rate")
    parser.add_argument("--self-correct", action="store_true",
                        help="Enable Self-Correction Bonus (+0.4 for internal error recovery)")
    parser.add_argument("--calibrated-compute", action="store_true",
                        help="Enable Complexity-Calibrated Thinking Compute (penalizes over/underthinking)")
    parser.add_argument("--env", type=str, choices=["arithmetic", "gsm8k", "real_gsm8k"], default="arithmetic",
                        help="Reasoning environment: 'arithmetic', 'gsm8k' (synthetic), or 'real_gsm8k' (real benchmark)")
    parser.add_argument("--prm", action="store_true",
                        help="Enable Process-Level Intermediate Equation Verification (PRM dense supervision)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--critic-lr", type=float, default=5e-5, help="Critic learning rate (PPO)")
    parser.add_argument("--kl-coeff", type=float, default=0.05, help="KL divergence penalty coefficient")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens per completion")
    parser.add_argument("--output-dir", type=str, default="runs/slm_rl", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=10, help="Steps between checkpoints")
    parser.add_argument("--eval-every", type=int, default=5, help="Steps between evaluations")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    return parser.parse_args()




def main():
    args = parse_args()
    device = args.device or get_default_device()

    print("=" * 70)
    print(f"SLM Reinforcement Learning System")
    print(f"Algorithm:   {args.algorithm.upper()}")
    print(f"Model:       {'[MOCK SYNTHETIC]' if args.mock else args.model}")
    print(f"Device:      {device}")
    print(f"Steps:       {args.steps} | Batch Size: {args.batch_size}")
    if args.algorithm == "grpo":
        print(f"Group Size:  {args.group_size} (samples per prompt)")
    print(f"KL Coeff:    {args.kl_coeff} | Learning Rate: {args.lr}")
    print("=" * 70)

    # 1. Setup Configurations
    model_config = ModelConfig(
        model_name_or_path=args.model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        is_mock=args.mock,
        use_lora=args.lora,
    )
    rl_config = RLConfig(
        algorithm=args.algorithm,
        learning_rate=args.lr,
        critic_learning_rate=args.critic_lr,
        total_steps=args.steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch,
        output_dir=args.output_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )

    # 2. Initialize Policy
    print("\n[1/4] Initializing SLM Policy...")
    policy = SLMPolicy(model_config)
    print(f"✓ Policy initialized on {device} (LoRA: {model_config.use_lora})")

    # 3. Setup Reward Engine
    # 3. Setup Reward Engine
    print("[2/4] Initializing Verifiable Reasoning Reward Engine...")
    from slm_rl.rewards.rule_based import (
        RepetitionPenaltyReward,
        ProcessStepReward,
        SelfReflectionReward,
    )
    from slm_rl.rewards.novel_rewards import (
        SelfCorrectionBonusReward,
        ComplexityCalibratedComputeReward,
    )
    format_reward = ReasoningFormatReward(think_weight=0.3, answer_weight=0.3, order_weight=0.2, min_think_chars=10)
    math_reward = MathCorrectnessReward()
    rep_penalty = RepetitionPenaltyReward(n=3, threshold=0.25, penalty_weight=0.3)
    reflection_reward = SelfReflectionReward(reward_per_marker=0.05, max_reward=0.2)

    reward_components = [
        (format_reward, 0.3),
        (math_reward, 0.4),
        (rep_penalty, 0.1),
        (reflection_reward, 0.1),
    ]

    if args.prm:
        prm_reward = ProcessStepReward(step_reward=0.2, max_reward=0.4)
        reward_components.append((prm_reward, 0.2))
        print("✓ Process-Level Supervision (PRM) Enabled")

    if args.self_correct:
        self_corr_reward = SelfCorrectionBonusReward(correction_bonus=0.3)
        reward_components.append((self_corr_reward, 0.2))
        print("✓ 'Aha Moment' Self-Correction Bonus Enabled (+0.3 for error recovery)")

    if args.calibrated_compute:
        calib_reward = ComplexityCalibratedComputeReward(base_target_words=12, words_per_operation=6)
        reward_components.append((calib_reward, 0.15))
        print("✓ Complexity-Calibrated Thinking Compute Enabled")

    reward_fn = CompositeReward(reward_components)
    print("✓ Composite reward engine initialized.")

    # 4. Initialize Trainer
    print(f"[3/4] Initializing {args.algorithm.upper()} Trainer...")
    if args.algorithm == "grpo":
        grpo_config = GRPOConfig(
            group_size=args.group_size,
            kl_coeff=args.kl_coeff,
            advantage_type=args.advantage_type,
        )
        trainer = GRPOTrainer(
            policy=policy,
            reward_fn=reward_fn,
            rl_config=rl_config,
            model_config=model_config,
            grpo_config=grpo_config,
        )
        print(f"✓ GRPO Strategy: {args.advantage_type.upper()} relative advantages")
    elif args.algorithm == "dpo":
        dpo_config = DPOConfig(beta=0.1)
        trainer = DPOTrainer(
            policy=policy,
            reward_fn=reward_fn,
            rl_config=rl_config,
            model_config=model_config,
            dpo_config=dpo_config,
        )
        print("✓ DPO Mode: Online Direct Preference Optimization from rollout groups")
    elif args.algorithm == "csao":
        csao_config = CSAOConfig(kl_coeff=args.kl_coeff)
        delib_config = DeliberationConfig(enabled=args.deliberation)
        trainer = CSAOTrainer(
            policy=policy,
            reward_fn=reward_fn,
            rl_config=rl_config,
            model_config=model_config,
            csao_config=csao_config,
            deliberation_config=delib_config,
        )
        print(f"✓ CSAO Mode: Counterfactual Step-Level Credit Assignment (Deliberation: {args.deliberation})")
    else:
        critic = SLMCritic(model_config)
        ppo_config = PPOConfig(kl_coeff=args.kl_coeff)
        trainer = PPOTrainer(
            policy=policy,
            critic=critic,
            reward_fn=reward_fn,
            rl_config=rl_config,
            model_config=model_config,
            ppo_config=ppo_config,
        )
    print(f"✓ Trainer ready (Micro-batch size: {rl_config.micro_batch_size}).")

    # 5. Environment & Evaluator
    print(f"[4/4] Setting up Reasoning Task Environment (Curriculum: {args.curriculum})...")
    if args.curriculum:
        from slm_rl.envs.curriculum_env import CurriculumReasoningEnv
        env = CurriculumReasoningEnv(seed=42)
        print("✓ Dynamic 5-Tier Curriculum Active: auto-adapting difficulty to policy performance")
    elif args.env == "real_gsm8k":
        from slm_rl.envs.real_data import RealReasoningDataset
        env = RealReasoningDataset(seed=42)
        print("✓ Real GSM8K Multi-Step Reasoning Benchmark Environment Active")
    elif args.env == "gsm8k":
        from slm_rl.envs.gsm8k_env import GSM8kTaskGenerator
        env = GSM8kTaskGenerator(seed=42)
    else:
        env = ReasoningTaskGenerator(seed=42)
    evaluator = Evaluator(policy)



    # Run initial baseline evaluation
    print("\n--- Running Baseline Evaluation (Pre-training) ---")
    val_batch = env.sample_batch(batch_size=4)
    pre_eval = evaluator.evaluate(val_batch["prompts"], val_batch["targets"])
    print(f"Pre-RL Accuracy:       {pre_eval['accuracy'] * 100:.1f}%")
    print(f"Pre-RL Format Rate:    {pre_eval['format_adherence_rate'] * 100:.1f}%")
    print(f"Pre-RL Avg Format:     {pre_eval['avg_format_score']:.3f}")
    print(f"Pre-RL Avg Math Score: {pre_eval['avg_math_score']:.3f}")

    # 6. Training Loop
    print(f"\n--- Starting RL Training ({args.steps} steps) ---")
    start_time = time.time()
    pbar = tqdm(range(1, args.steps + 1), desc="RL Training")

    for step in pbar:
        batch = env.sample_batch(batch_size=args.batch_size)
        metrics = trainer.train_step(prompts=batch["prompts"], targets=batch["targets"])

        r_val = metrics.get('reward_mean', metrics.get('mean_reward', 0.0))
        postfix = {
            "r_mean": f"{r_val:.3f}",
            "loss": f"{metrics.get('loss', metrics.get('policy_loss', 0.0)):.3f}",
            "kl": f"{metrics.get('kl_divergence', 0.0):.4f}",
        }
        if "dpo_margin" in metrics:
            postfix["margin"] = f"{metrics['dpo_margin']:.2f}"
        if "pivot_rate" in metrics:
            postfix["pivots"] = f"{metrics['pivot_rate']:.1f}"
        if "policy_entropy" in metrics:
            postfix["ent"] = f"{metrics['policy_entropy']:.2f}"
        if "avg_think_words" in metrics:
            postfix["think_w"] = f"{metrics['avg_think_words']:.1f}"
        pbar.set_postfix(postfix)

        if step % args.eval_every == 0 or step == args.steps:
            eval_batch = env.sample_batch(batch_size=4)
            eval_metrics = evaluator.evaluate(eval_batch["prompts"], eval_batch["targets"])
            tqdm.write(
                f"[Step {step:03d}] "
                f"Reward: {r_val:.3f} | "
                f"Eval Acc: {eval_metrics['accuracy'] * 100:.1f}% | "
                f"Eval Format: {eval_metrics['format_adherence_rate'] * 100:.1f}% | "
                f"KL: {metrics.get('kl_divergence', 0.0):.4f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_path = trainer.save_checkpoint()
            tqdm.write(f"✓ Saved checkpoint to {save_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / args.steps:.2f}s/step)")

    # 7. Post-training Evaluation
    print("\n--- Running Final Evaluation (Post-training) ---")
    final_eval = evaluator.evaluate(val_batch["prompts"], val_batch["targets"])
    print(f"Post-RL Accuracy:       {final_eval['accuracy'] * 100:.1f}%")
    print(f"Post-RL Format Rate:    {final_eval['format_adherence_rate'] * 100:.1f}%")
    print(f"Post-RL Avg Format:     {final_eval['avg_format_score']:.3f}")
    print(f"Post-RL Avg Math Score: {final_eval['avg_math_score']:.3f}")

    # Show a sample response
    if final_eval["samples"]:
        sample = final_eval["samples"][0]
        print("\nSample Output Post-RL:")
        print("-" * 50)
        print("Prompt:\n" + sample["prompt"].strip())
        print(f"Target: {sample['target']}")
        print("-" * 50)

    # Prover-Skeptic Self-Play Duel Round
    if args.self_play:
        print("\n--- Running Prover-Skeptic Self-Play Duel ---")
        from slm_rl.envs.self_play_env import SelfPlayDuelEnv
        import torch
        duel_env = SelfPlayDuelEnv()
        sample_batch = env.sample_batch(batch_size=2)
        with torch.no_grad():
            prover_rollout = policy.generate(sample_batch["prompts"], num_return_sequences=1)
            for i in range(len(sample_batch["prompts"])):
                prob = sample_batch["prompts"][i]
                tgt = sample_batch["targets"][i]
                p_sol = prover_rollout["completion_texts"][i]
                skeptic_prompt = duel_env.format_skeptic_prompt(prob, p_sol)
                skeptic_rollout = policy.generate([skeptic_prompt], num_return_sequences=1)
                s_resp = skeptic_rollout["completion_texts"][0]
                outcome = duel_env.arbitrate_duel(prob, tgt, p_sol, s_resp)
                print(f"[Duel #{i+1}] Outcome: {outcome.outcome_tag} | Prover Payoff: {outcome.prover_reward:+.1f} | Skeptic Payoff: {outcome.skeptic_reward:+.1f}")

    # 8. Generate and save training dynamics curves & Cognition Inspector HTML
    try:
        from plot_training import main as plot_main
        import sys
        saved_argv = sys.argv
        sys.argv = [
            sys.argv[0],
            "--log-file", os.path.join(args.output_dir, "metrics.jsonl"),
            "--output", os.path.join(args.output_dir, "training_curves.png"),
        ]
        plot_main()
        sys.argv = saved_argv
    except Exception as e:
        print(f"Note: Could not generate plot automatically: {e}")

    try:
        from slm_rl.telemetry.cognition_visualizer import generate_html_report
        html_out = os.path.join(args.output_dir, "cognition_report.html")
        generate_html_report(os.path.join(args.output_dir, "metrics.jsonl"), html_out)
    except Exception as e:
        print(f"Note: Could not generate HTML report automatically: {e}")



if __name__ == "__main__":

    main()
