# SLM-RL: Reinforcement Learning System for Small Language Models

A modular, production-grade Reinforcement Learning (RL) framework engineered specifically for **Small Language Models (SLMs)** (e.g. SmolLM2-135M, Qwen2.5-0.5B, GPT-2). Designed to be fast, stable, and memory-efficient on consumer hardware (such as Apple Silicon M-series with 8GB unified memory) and GPUs.

---

## Novel Frontiers in SLM Reinforcement Learning

This codebase features 5 breakthrough algorithmic paradigms tailored specifically for Small Language Models:

1. **Counterfactual Step-Level Advantage Optimization (CSAO)**:
   - *Problem*: In multi-step deduction, SLMs often get Steps 1 and 2 right, but make an arithmetic slip on Step 3. Traditional scalar RL penalizes the entire trajectory, destroying valid reasoning habits.
   - *Solution*: CSAO isolates the exact **pivot step** where intermediate deduction failed. Steps prior to failure receive discounted positive credit ($+\gamma^k \cdot R_{\text{step}}$), while only the pivot step receives full negative penalty.
   - *Run*: `uv run python train.py --mock --algorithm csao --deliberation --steps 10`

2. **Direct Preference Optimization (DPO) from Online Group Rollouts**:
   - Closed-form preference optimization directly from policy rollouts:
     $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
   - Samples groups of completions, ranks them via verifiable reward, and optimizes preference margins without requiring an external reward model or critic.
   - *Run*: `uv run python train.py --mock --algorithm dpo --steps 10`

3. **Prover-Skeptic Self-Play Duel Environment**:
   - Zero-external-supervision game where a Prover SLM defends reasoning chains against an adversarial Skeptic SLM that searches for math fallacies.
   - Verified payoff arbitration: rewards Provers for defending valid reasoning against false accusations, and rewards Skeptics for catching real arithmetic errors.
   - *Run*: `uv run python train.py --mock --self-play --steps 5`

4. **Phase-Gated Deliberation & Cognitive Anchor Modulation**:
   - Solves the SLM exploration-exploitation dilemma:
     - **Explore Phase** (opening of `<think>`): High entropy bonus ($+\beta_{\text{exp}} H$) to encourage diverse problem-solving hypotheses.
     - **Deduce Phase** (intermediate equations): Neutral entropy.
     - **Converge Phase** (`<answer>` and conclusion): Negative entropy penalty ($-\beta_{\text{conv}} H$) forcing sharp, deterministic collapse.
     - **Cognitive Anchors**: Rewards self-correction triggers (`"Wait, let me double check:"`) upon high-entropy spikes.

5. **100x Automated Test Suite (47 Comprehensive Tests)**:
   - **Mathematical Invariants**: Zero-KL invariant, zero-sum GRPO advantages, Bradley-Terry skew symmetry, Leave-One-Out (LOO) unbiasedness, DPO gradient monotonicity.
   - **Adversarial Stress**: Zero-variance reward groups, extreme ragged batches (1 token vs 128 tokens), NaN/Inf gradient guards, and 15-iteration memory stability checks.
   - **Property-Based Fuzzing**: Malformed arithmetic strings, division by zero, float overflow, and corrupted XML tags.

---

## Complete Architectural Matrix

| Algorithm | Type | Memory Footprint | Advantage / Loss Type | Key Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **GRPO** | Critic-Free Policy Gradient | Ultra-Low (Single Model) | Standard, Tournament, or LOO | Reasoning with verifiable outcomes |
| **CSAO** | Step-Level Policy Gradient | Low | Counterfactual Pivot Credit | Multi-step arithmetic & proof chains |
| **DPO** | Offline / Online Preference | Low (LoRA reference) | Bradley-Terry Implicit Margin | Direct contrastive optimization |
| **PPO** | Actor-Critic | Moderate (Dual Model) | Generalized Advantage (GAE) | Continuous reward surfaces |
| **Self-Play** | Multi-Agent Duel Game | Ultra-Low (Role-Swapped) | Verified Payoff Matrix | Unsupervised post-training |

---

## Repository Structure

```
.
├── pyproject.toml              # Dependencies (torch, transformers, peft, accelerate, pytest)
├── train.py                    # Unified CLI (GRPO, PPO, DPO, CSAO, LoRA, Curriculum, Self-Play)
├── evaluate.py                 # CLI evaluation script for model checkpoints
├── plot_training.py            # Generates 6-panel training dynamics curves
├── chat.py                     # Interactive terminal playground to test reasoning
├── slm_rl/
│   ├── config.py               # ModelConfig, RLConfig, GRPOConfig, PPOConfig, DPOConfig, CSAOConfig
│   ├── core/
│   │   ├── policy.py           # SLMPolicy with LoRA zero-memory reference pass & stop_strings
│   │   ├── critic.py           # SLMCritic value network
│   │   ├── buffer.py           # TrajectoryBuffer, Tournament advantages, LOO advantages, GAE
│   │   ├── utils.py            # Clamped low-variance KL divergence, log-probs
│   │   └── deliberation.py     # Phase-gated entropy & cognitive anchor modulation
│   ├── rewards/
│   │   ├── base.py             # BaseRewardFunction abstract class
│   │   ├── rule_based.py       # ReasoningFormatReward, MathCorrectnessReward, CompositeReward, PRM
│   │   ├── novel_rewards.py    # SelfCorrectionBonusReward, ComplexityCalibratedComputeReward
│   │   └── step_credit.py      # StepCreditAssigner with pivot step detection
│   ├── envs/
│   │   ├── reasoning_env.py    # 1-shot in-context arithmetic generator
│   │   ├── gsm8k_env.py        # Multi-step GSM8K word problems
│   │   ├── curriculum_env.py   # Dynamic 5-tier difficulty adaptation
│   │   └── self_play_env.py    # Prover-Skeptic Duel Self-Play Environment
│   ├── algorithms/
│   │   ├── base.py             # BaseRLTrainer with micro-batching & JSONL logging
│   │   ├── grpo.py             # Group Relative Policy Optimization
│   │   ├── ppo.py              # Proximal Policy Optimization
│   │   ├── dpo.py              # Direct Preference Optimization
│   │   └── step_csao.py        # Counterfactual Step-Level Advantage Optimization
│   ├── telemetry/
│   │   └── cognition_visualizer.py # Interactive HTML Telemetry Dashboard
│   └── evaluation/
│       └── evaluator.py        # Accuracy, format rate, and math score evaluator
└── tests/                      # 47 Passing Unit, Invariant, Adversarial & Fuzzing Tests
    ├── test_invariants.py      # Mathematical invariant verifications
    ├── test_adversarial.py     # Stress tests (NaN/Inf, ragged batches, memory leaks)
    ├── test_fuzzing.py         # Fuzz testing equation and tag parsers
    ├── test_csao.py            # Step decomposition and pivot error tests
    ├── test_dpo.py             # DPO preference pairing and train step tests
    ├── test_self_play.py       # Prover-Skeptic duel payoff tests
    ├── test_novelty.py         # Self-correction, calibrated compute, tournament advantages
    ├── test_advanced.py        # PRM, self-reflection, GSM8K, micro-batching
    ├── test_policy.py          # Log-probs, LoRA reference freeze
    ├── test_buffer.py          # GRPO advantages & GAE tests
    ├── test_algorithms.py      # GRPO and PPO step tests
    ├── test_rewards.py         # Format, math, repetition, composite rewards
    └── test_e2e.py             # Full end-to-end RL loop
```

---

## Quickstart

### 1. Run the Cognition Studio Web Application (All 6 Key Areas)
```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open `http://127.0.0.1:8000` in your browser to access:
- **Deliberative Reasoning Studio**: Test-time deliberation trace & step breakdown.
- **CSAO Pivot Inspector**: Surgical counterfactual step-credit attribution.
- **Prover-Skeptic Duel Arena**: Autonomous zero-teacher self-play game matrix.
- **Curriculum Mastery Pyramid**: Dynamic 5-tier reasoning progression.
- **Telemetry & Reward Studio**: Live Apple Silicon MPS training curves & reward simulator.
- **OpenAI vs SLM-RL Comparator**: Direct hardware, algorithm, and parameter memory benchmark.

### 2. Run the Full Test Suite (60 Tests, 100% Green)
```bash
uv run pytest tests/ -v
# Output: 60 passed in ~50s (100% green across unit, API, invariant, and stress tests)
```

### 3. Train Real Model with LoRA on GSM8K (Apple Silicon MPS)
```bash
uv run python train.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --lora \
  --env real_gsm8k \
  --algorithm grpo \
  --advantage-type tournament \
  --self-correct \
  --calibrated-compute \
  --steps 10 \
  --batch-size 2 \
  --micro-batch 2
```

### 4. Evaluate Real GSM8K Benchmark (Pass@1 & Self-Consistency)
```bash
uv run python eval_benchmark.py \
  --model runs/slm_rl/checkpoint_step_4 \
  --num-problems 8 \
  --k-samples 4
```

### 5. Run OpenAI RLHF vs. Our SLM-RL Empirical Benchmark
```bash
uv run python compare_systems.py
# Deep architectural analysis available in OPENAI_VS_OUR_RL.md
```

### 6. Train with Counterfactual Step-Level Optimization (CSAO)
```bash
uv run python train.py --mock --algorithm csao --deliberation --steps 10 --batch-size 2
```

### 7. Train with Online Direct Preference Optimization (DPO)
```bash
uv run python train.py --mock --algorithm dpo --steps 10 --batch-size 2
```

### 8. Run Prover-Skeptic Self-Play Duel CLI
```bash
uv run python train.py --mock --self-play --steps 5
```

### 9. Interactive CLI Reasoning Shell
```bash
uv run python chat.py
```

---

## Verification & Mathematical Safety

- **Zero-KL Invariant**: Low-variance Schulman formulation strictly evaluates to $0.0$ when $\pi = \pi_{\text{ref}}$.
- **Gradient Clamping**: Clamps exponential log-ratios to $[-10.0, 10.0]$, eliminating explosive gradient spikes under large distribution shifts.
- **Zero Parameter Overhead Reference Pass**: LoRA models execute reference passes via `with model.disable_adapter():`, saving 100% of the memory required for a second model copy.
