#!/usr/bin/env python3
"""
CLI Evaluation Script for evaluating trained SLM checkpoints.
"""

import argparse
from slm_rl.config import ModelConfig, get_default_device
from slm_rl.core.policy import SLMPolicy
from slm_rl.envs.reasoning_env import ReasoningTaskGenerator
from slm_rl.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SLM policy on reasoning tasks")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="Path or name of the model/checkpoint to evaluate")
    parser.add_argument("--mock", action="store_true", help="Use mock model for testing")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test problems to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default=None, help="Device to evaluate on")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or get_default_device()

    print("=" * 60)
    print("Evaluating Model Checkpoint")
    print(f"Model:       {args.model}")
    print(f"Device:      {device}")
    print(f"Samples:     {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print("=" * 60)

    model_config = ModelConfig(
        model_name_or_path=args.model,
        device=device,
        is_mock=args.mock,
        max_new_tokens=args.max_new_tokens,
    )
    policy = SLMPolicy(model_config)
    evaluator = Evaluator(policy)

    env = ReasoningTaskGenerator(seed=999)
    test_batch = env.sample_batch(batch_size=args.num_samples)

    print("\nRunning evaluation...")
    results = evaluator.evaluate(
        prompts=test_batch["prompts"],
        targets=test_batch["targets"],
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Total Samples:         {results['num_samples']}")
    print(f"Exact Math Accuracy:   {results['accuracy'] * 100:.1f}%")
    print(f"Format Adherence Rate: {results['format_adherence_rate'] * 100:.1f}%")
    print(f"Average Format Score:  {results['avg_format_score']:.3f} / 1.000")
    print(f"Average Math Score:    {results['avg_math_score']:.3f} / 1.000")
    print(f"Average Word Count:    {results['avg_word_length']:.1f} words")
    print("=" * 60)

    print("\nDetailed Sample Predictions:")
    for i, s in enumerate(results["samples"][:3], 1):
        print(f"\n--- Sample #{i} ---")
        print("Prompt:")
        print(s["prompt"].strip())
        print(f"Target: {s['target']}")
        print(f"Format Score: {s['format_score']:.2f} | Math Score: {s['math_score']:.2f}")
        print("Generated Output:")
        print(s["completion"].strip())
        print("-" * 40)


if __name__ == "__main__":
    main()
