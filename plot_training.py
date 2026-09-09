#!/usr/bin/env python3
"""
Plotting utility for visualizing SLM RL training dynamics from metrics.jsonl.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RL training metrics")
    parser.add_argument("--log-file", type=str, default="runs/slm_rl/metrics.jsonl",
                        help="Path to metrics.jsonl file")
    parser.add_argument("--output", type=str, default="runs/slm_rl/training_curves.png",
                        help="Path to save generated PNG plot")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found at {args.log_file}")
        return

    records = []
    with open(args.log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        print("No valid metric records found.")
        return

    steps = [r.get("step", i) for i, r in enumerate(records, 1)]
    rewards = [r.get("mean_reward", 0.0) for r in records]
    losses = [r.get("loss", r.get("policy_loss", 0.0)) for r in records]
    kls = [r.get("kl_divergence", 0.0) for r in records]
    entropies = [r.get("policy_entropy", 0.0) for r in records]
    think_words = [r.get("avg_think_words", 0.0) for r in records]

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("SLM Reinforcement Learning (GRPO) Training Dynamics", fontsize=16, fontweight="bold")

    # 1. Mean Reward
    axs[0, 0].plot(steps, rewards, color="#10b981", marker="o", linewidth=2)
    axs[0, 0].set_title("Mean Reward Progression")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 2. Total Loss
    axs[0, 1].plot(steps, losses, color="#ef4444", marker="s", linewidth=2)
    axs[0, 1].set_title("Policy Loss")
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. KL Divergence
    axs[0, 2].plot(steps, kls, color="#8b5cf6", marker="^", linewidth=2)
    axs[0, 2].set_title("KL Divergence vs Reference Policy")
    axs[0, 2].set_xlabel("Step")
    axs[0, 2].set_ylabel("D_KL")
    axs[0, 2].grid(True, linestyle="--", alpha=0.6)

    # 4. Policy Entropy
    axs[1, 0].plot(steps, entropies, color="#f59e0b", marker="d", linewidth=2)
    axs[1, 0].set_title("Policy Entropy (Exploration vs Collapse)")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("Entropy")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 5. Thinking Length (Test-time compute scaling)
    axs[1, 1].plot(steps, think_words, color="#06b6d4", marker="p", linewidth=2)
    axs[1, 1].set_title("Test-Time Compute (Words inside <think>)")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Words")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    # 6. Reward Distribution Summary
    max_rewards = [r.get("max_reward", rewards[i]) for i, r in enumerate(records)]
    min_rewards = [r.get("min_reward", rewards[i]) for i, r in enumerate(records)]
    axs[1, 2].plot(steps, max_rewards, label="Max Reward", color="#059669", linestyle="--")
    axs[1, 2].plot(steps, rewards, label="Mean Reward", color="#10b981", linewidth=2)
    axs[1, 2].plot(steps, min_rewards, label="Min Reward", color="#f87171", linestyle=":")
    axs[1, 2].set_title("Reward Envelope (Min / Mean / Max)")
    axs[1, 2].set_xlabel("Step")
    axs[1, 2].set_ylabel("Reward")
    axs[1, 2].legend(loc="upper left")
    axs[1, 2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"✓ Training curves successfully saved to {args.output}")


if __name__ == "__main__":
    main()
