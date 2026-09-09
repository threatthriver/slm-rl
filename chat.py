#!/usr/bin/env python3
"""
Interactive CLI playground to test and chat with trained SLM reasoning models.
"""

import argparse
import re
import sys
from slm_rl.config import ModelConfig, get_default_device
from slm_rl.core.policy import SLMPolicy
from slm_rl.envs.reasoning_env import SYSTEM_PROMPT


import os

# ANSI color codes
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def parse_args():
    default_model = "runs/slm_rl/checkpoint_step_4" if os.path.exists("runs/slm_rl/checkpoint_step_4") else "HuggingFaceTB/SmolLM2-135M-Instruct"
    parser = argparse.ArgumentParser(description="Interactive Chat & Reasoning with SLM")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Model name or checkpoint path")
    parser.add_argument("--mock", action="store_true", help="Use mock model")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max response tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or get_default_device()

    print(f"\n{BOLD}{CYAN}================================================================={RESET}")
    print(f"{BOLD}[SLM Interactive Deliberative Reasoning Shell]{RESET}")
    print(f"Model:       {args.model}")
    print(f"Device:      {device}")
    print(f"{DIM}Type your arithmetic or multi-step word problem below.{RESET}")
    print(f"{DIM}Type 'exit' or 'quit' to exit.{RESET}")
    print(f"{BOLD}{CYAN}================================================================={RESET}")

    model_config = ModelConfig(
        model_name_or_path=args.model,
        device=device,
        is_mock=args.mock,
        max_prompt_length=512,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n{YELLOW}Loading model weights into memory...{RESET}")
    policy = SLMPolicy(model_config)
    print(f"{GREEN}[Ready] Model loaded for real-time deliberative inference.{RESET}\n")

    while True:
        try:
            user_input = input(f"\n{BOLD}Problem > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        prompt = f"{SYSTEM_PROMPT}\n\nProblem: {user_input}\n"

        rollout = policy.generate(
            [prompt],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_return_sequences=1,
        )
        completion = rollout["completion_texts"][0].strip()

        print("\n" + "-" * 55)
        # Parse think vs answer blocks
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

        if think_match:
            print(f"{CYAN}Thinking Process (<think>):{RESET}")
            print(f"{DIM}{think_match.group(1).strip()}{RESET}\n")
        if answer_match:
            print(f"{GREEN}Final Answer (<answer>): {BOLD}{answer_match.group(1).strip()}{RESET}")
        else:
            print(f"{YELLOW}Output:{RESET}\n{completion}")
        print("-" * 55)


if __name__ == "__main__":
    main()
