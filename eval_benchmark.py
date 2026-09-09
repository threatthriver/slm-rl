#!/usr/bin/env python3
"""
Evaluation and Benchmarking Suite for Small Language Models on Real Reasoning Tasks.

Computes:
- Pass@1 Accuracy
- Pass@k (Best-of-N & Majority-Voting Self-Consistency)
- Format Adherence Rate
- Thinking Tokens (Test-time compute)
- Generates formatted side-by-side solution traces
"""

import argparse
import collections
import re
from typing import Any, Dict, List, Optional
import torch
from tqdm import tqdm

from slm_rl.config import ModelConfig, get_default_device
from slm_rl.core.policy import SLMPolicy
from slm_rl.envs.real_data import RealReasoningDataset
from slm_rl.rewards.rule_based import MathCorrectnessReward, ReasoningFormatReward


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SLM on Real Multi-step Reasoning Tasks")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="Model name or checkpoint path")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock model for quick testing")
    parser.add_argument("--k-samples", type=int, default=4, help="Samples per problem for Pass@k & Majority Voting")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num-problems", type=int, default=8, help="Number of real problems to evaluate")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    return parser.parse_args()


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extracts final numerical prediction from <answer> tag or text."""
    ans_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    candidate_text = ans_match.group(1) if ans_match else text

    # Find last number in text
    nums = re.findall(r"[-+]?\d*\.?\d+", candidate_text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def run_benchmark(
    policy: SLMPolicy,
    dataset: RealReasoningDataset,
    num_problems: int = 8,
    k_samples: int = 4,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Evaluates policy on real reasoning problems.
    """
    eval_set = dataset.get_eval_set(limit=num_problems)
    format_evaluator = ReasoningFormatReward()

    pass1_correct = 0
    majority_correct = 0
    any_correct = 0
    format_compliant = 0
    total_samples = 0
    total_think_words = 0

    results = []

    print(f"\nEvaluating {len(eval_set)} real problems (k={k_samples} samples per problem)...")

    for item in tqdm(eval_set, desc="Benchmarking"):
        question = item["question"]
        target = item["target"]
        solution = item["solution"]
        prompt = dataset.format_prompt(question)

        with torch.no_grad():
            rollout = policy.generate(
                [prompt],
                num_return_sequences=k_samples,
                temperature=temperature,
                max_new_tokens=128,
            )

        completions = rollout["completion_texts"]
        extracted_answers = [extract_numeric_answer(c) for c in completions]

        # Format adherence
        fmt_scores = format_evaluator([prompt] * k_samples, completions, targets=[target] * k_samples)
        valid_format_count = sum(1 for s in fmt_scores if s >= 0.7)
        format_compliant += valid_format_count
        total_samples += k_samples

        # Thinking words
        for c in completions:
            m = re.search(r"<think>(.*?)</think>", c, re.DOTALL)
            if m:
                total_think_words += len(m.group(1).split())

        # Pass@1: first sample
        p1_ans = extracted_answers[0]
        is_p1_correct = (p1_ans is not None and abs(p1_ans - target) < 1e-3)
        if is_p1_correct:
            pass1_correct += 1

        # Any-correct (Pass@k upper bound)
        is_any_correct = any(a is not None and abs(a - target) < 1e-3 for a in extracted_answers)
        if is_any_correct:
            any_correct += 1

        # Majority voting (Self-Consistency)
        valid_answers = [a for a in extracted_answers if a is not None]
        if valid_answers:
            most_common = collections.Counter(valid_answers).most_common(1)[0][0]
            if abs(most_common - target) < 1e-3:
                majority_correct += 1

        results.append({
            "id": item["id"],
            "question": question,
            "target": target,
            "solution": solution,
            "sample_completion": completions[0],
            "extracted_answers": extracted_answers,
            "p1_correct": is_p1_correct,
            "majority_correct": (valid_answers and abs(most_common - target) < 1e-3),
        })

    num_items = len(eval_set)
    metrics = {
        "pass_at_1": pass1_correct / max(1, num_items),
        "majority_voting_acc": majority_correct / max(1, num_items),
        "pass_at_k_coverage": any_correct / max(1, num_items),
        "format_adherence_rate": format_compliant / max(1, total_samples),
        "avg_think_words": total_think_words / max(1, total_samples),
        "num_evaluated": num_items,
        "results": results,
    }

    return metrics


def main():
    args = parse_args()
    device = args.device or get_default_device()

    print("=" * 70)
    print("SLM Real-World Reasoning Benchmark")
    print(f"Model:       {args.model if not args.mock else '[MOCK SYNTHETIC]'}")
    print(f"Device:      {device}")
    print(f"Problems:    {args.num_problems} | Samples per Problem (k): {args.k_samples}")
    print("=" * 70)

    model_config = ModelConfig(
        model_name_or_path=args.model,
        device=device,
        is_mock=args.mock,
    )
    policy = SLMPolicy(model_config)
    dataset = RealReasoningDataset(seed=42)

    metrics = run_benchmark(
        policy=policy,
        dataset=dataset,
        num_problems=args.num_problems,
        k_samples=args.k_samples,
        temperature=args.temperature,
    )

    print("\n" + "=" * 70)
    print("Benchmark Results Summary:")
    print("-" * 70)
    print(f"Pass@1 Accuracy:            {metrics['pass_at_1'] * 100:.1f}%")
    print(f"Majority-Voting Accuracy:   {metrics['majority_voting_acc'] * 100:.1f}%")
    print(f"Pass@{args.k_samples} Coverage:             {metrics['pass_at_k_coverage'] * 100:.1f}%")
    print(f"Format Compliance:          {metrics['format_adherence_rate'] * 100:.1f}%")
    print(f"Avg Deliberation Words:     {metrics['avg_think_words']:.1f} words")
    print("=" * 70)

    # Show a sample trace
    if metrics["results"]:
        sample = metrics["results"][0]
        print("\nRepresentative Reasoning Trace:")
        print(f"Problem:  {sample['question']}")
        print(f"Target:   {sample['target']}")
        print(f"Ground Truth Solution:\n  {sample['solution']}")
        print("\nModel Output:")
        print(sample["sample_completion"].strip())
        print("=" * 70)


if __name__ == "__main__":
    main()
