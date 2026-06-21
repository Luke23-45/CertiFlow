# CertiQ-Net

Learned certified dispatch for queueing systems, powered by [QGym](https://github.com/namkoong-lab/QGym).

**CertiQ-Net** introduces a **learned marginal-cost index policy** with a **closed-form certificate guarantee** for queueing network control. The policy is a Set Transformer that maps per-queue state (queue length, service rate) to dispatch probabilities, with an analytic delay geometry used for certification.

Training and evaluation use QGym's **identical RL/PPO pipeline** (`CustomPPOTrainer` + `CustomRolloutBuffer` + `parallel_eval`) — the CertiQ model drops in as a policy class, same as `Vanilla` or `WC` baselines, ensuring fair comparison.

## Architecture

```
┌─────────────────────────────────────────┐
│           QGym Environment              │  Discrete-event queueing simulation
│    DiffDiscreteEventSystem              │  (reentrant, criss-cross, parallel, ...)
│    action_space: Box(0, 1) (S, Q)      │
│    observation_space: Box(0, inf) (Q,)  │
└────────┬────────────────────────────────┘
         │ queues (B, Q)
         ▼
┌──────────────────────────────────────────────────────┐
│               CertiQSB3Policy                         │
│            (ActorCriticPolicy subclass)               │
│  ┌──────────────────────────────────────────────────┐│
│  │  MarginalIndexHead                               ││
│  │  ┌──────────────────────────────────────────────┐││
│  │  │ DispatchInteractionEncoder                    │││
│  │  │  (Set Transformer with Induced Attention)    │││
│  │  │  token_features: [Q, μ, log(1+Q), ...]      │││
│  │  └──────────────┬───────────────────────────────┘││
│  │                 ▼ z_local, z_global              ││
│  │  ┌──────────────────────────────────────────────┐││
│  │  │ index_head MLP → logits (B, Q)              │││
│  │  │ value_head MLP → value (B, 1)               │││
│  │  └──────────────────────────────────────────────┘││
│  │  output: π = softmax(-logits / τ)               ││
│  │  expanded to per-server priority (B, S, Q)       ││
│  └──────────────────────────────────────────────────┘│
│  SB3 components:                                     │
│  • pi_features_extractor = encoder (policy opt.)     │
│  • action_net = index_head                           │
│  • value_net = value_head                            │
│  • predict() → (action_np, action_probs_np)          │
└────────┬────────────────────────────────────────────┘
         │ action (B, S, Q) — one-hot per server
         ▼
┌─────────────────────────────────────────┐
│      CustomPPOTrainer (unchanged)       │  QGym's standard PPO trainer
│  • collect_rollouts via policy()        │  Same code path for all policies
│  • train() via evaluate_actions()       │
│  • optional Lagrangian penalty loss     │
└─────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone with submodules
git clone <your-repo-url>
cd certiq_net
git submodule update --init

# 2. Install
pip install -e .

# 3. Run tests
pytest tests/ -v
```

## Training a Model

Uses QGym's PPO pipeline. The first positional arg is the **policy-config** stem (from `extern/QGym/RL/policy_configs/`) and the second is the **env-config** stem (from `extern/QGym/configs/env/`).

```bash
python -m certiq_net.studies.qgym_eval.train.train \
    vanilla \
    reentrant_2

# With Lagrangian certificate constraint
python -m certiq_net.studies.qgym_eval.train.train \
    vanilla \
    criss_cross_bh \
    --use-lagrangian \
    --lr-nu 1e-3
```

### Training Arguments

| Argument | Description |
|----------|-------------|
| `policy_config` | Policy config stem (e.g. `vanilla`) |
| `env_config` | Env config stem (e.g. `reentrant_2`) |
| `--use-lagrangian` | Enable CertiQ Lagrangian constraint in PPO loss |
| `--lr-nu` | Learning rate for the Lagrangian dual variable (default: `1e-3`) |

Hyperparameters (learning rate, batch size, architecture, etc.) are set in the policy config YAML files under `extern/QGym/RL/policy_configs/`.

## Evaluate a Trained Model

Evaluates a checkpoint using the identical RL/PPO code path as training (`load_rl_p_env` + `model.predict` + `env.step`).

```bash
# Single environment
python -m certiq_net.studies.qgym_eval.evaluate \
    --env reentrant_2 \
    --checkpoint path/to/checkpoint.pt \
    --test-batch 100

# With hyperexponential arrivals
python -m certiq_net.studies.qgym_eval.evaluate \
    --env reentrant_2_hyper \
    --checkpoint path/to/checkpoint.pt
```

### Evaluation Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `reentrant_2` | QGym environment name |
| `--checkpoint` | required | Trained model `.pt` file (`CertiQSB3Policy` or `MarginalIndexHead` state dict) |
| `--device` | `cpu` | Torch device |
| `--test-batch` | `100` | Parallel environments for evaluation |
| `--test-steps` | env default | Simulation horizon |
| `--seed` | `42` | Random seed for reproducibility |
| `--model-config` | `configs/model/certiq_index.yaml` | Model architecture config |
| `--qgym-root` | `./extern/QGym` | Path to QGym directory |
| `--output-dir` | `results` | Results output directory |

Checkpoint loading handles multiple formats automatically: `CertiQSB3Policy` state dicts, `MarginalIndexHead` state dicts, and legacy `CertiQIndexModel` formats — all with automatic key remapping.

## Batch Benchmark

Run the full benchmark suite across all environments with automatic LaTeX/Markdown table generation:

```bash
# Single checkpoint evaluated on all envs (generalization test)
python -m certiq_net.studies.run_benchmark \
    --checkpoint path/to/model.pt \
    --output-dir benchmark_results

# Per-environment checkpoints
python -m certiq_net.studies.run_benchmark \
    --checkpoint-dir checkpoints/ \
    --output-dir benchmark_results

# Regenerate tables from existing results (no re-evaluation)
python -m certiq_net.studies.run_benchmark \
    --results-dir benchmark_results/results \
    --output-dir benchmark_results \
    --table-only
```

## Benchmark Environments

Supported QGym queueing networks:

| Network | Exponential | Hyperexponential | Published Baselines |
|---------|-------------|------------------|-------------------|
| Reentrant (2–10) | ✅ | ✅ | c-μ, MW, MP, FP, PPO, PPO-BC, PPO-WC |
| Re-entrant (2–10) | ✅ | ✅ | c-μ, MW, MP, FP, PPO, PPO-BC, PPO-WC |
| Criss Cross BH | ✅ | ❌ | c-μ, MW, MP, FP, PPO, PPO-BC, PPO-WC |
| N Model (5×5) | ✅ | ❌ | c-μ, MW, MP, FP, PPO, PPO-BC, PPO-WC |
| Input Switch | ✅ | ❌ | c-μ, MW, MP, FP, PPO, PPO-BC, PPO-WC |
| Hospital | ✅ | ❌ | c-μ, MW, MP, FP, PPO, PPO-BC, PPO-WC |

Baseline results published in [QGym: Scalable Simulation and Benchmarking of Queuing Network Controllers](https://arxiv.org/abs/2410.06170) (NeurIPS 2024).

## Model Configuration

Default architecture (`configs/model/certiq_index.yaml`):

```yaml
model:
  hidden_dim: 128            # Transformer embedding dimension
  tau: 1.0                   # Softmax temperature
  d_xi: 0                    # Exogenous context dimension
  encoder_layers: 2          # Set Transformer layers
  num_heads: 4               # Attention heads
  num_inducing_points: 4     # Induced attention points (0 = full self-attention)
  dropout: 0.0
  C: 20.0                    # Certificate slack budget constant
  cost_fn: qmd               # Cost function: "sed" or "qmd"
  constraint_mode: lagrangian  # "lagrangian", "projection", or "unconstrained"
```

### Constraint Modes

| Mode | Description | Training Loss |
|------|-------------|---------------|
| `lagrangian` | Soft constraint via PPO-Lagrangian dual variable | `L = L_PPO + ν · violation` |
| `projection` | Hard constraint via Differentiable KL Projection | Standard PPO (constraint enforced by projection layer) |
| `unconstrained` | Plain softmax, no certificate | Standard PPO |

### Cost Functions

The internal cost used by the certificate:

- **`sed`**: Shortest Expected Delay — `(Q_i + 1) / μ_i`
- **`qmd`**: Quadratic Drift — `(2·Q_i + 1) / μ_i`

## Project Structure

```
certiq_net/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── src/certiq_net/             # Python package
│   ├── dispatcher/             # Core model
│   │   ├── certiq/
│   │   │   ├── index_model.py  # CertiQIndexModel + MarginalIndexHead
│   │   │   ├── certificate.py  # Differentiable KL Projection
│   │   │   ├── interaction.py  # Set Transformer encoder
│   │   │   ├── geometry.py     # CertifiedGeometry parametric baseline
│   │   │   └── cost_learner.py # Learned cost MLP
│   │   ├── delay_geometry.py   # SED, QMD index functions
│   │   └── types.py            # DispatcherDiagnostics, DispatcherForward
│   ├── studies/
│   │   ├── run_benchmark.py    # Batch benchmark + LaTeX table generation
│   │   └── qgym_eval/
│   │       ├── evaluate.py     # Standalone evaluation (RL/PPO pipeline)
│   │       └── train/          # Training pipeline
│   │           ├── train.py              # Training entry point
│   │           ├── certiq_sb3_policy.py  # SB3 ActorCriticPolicy wrapper
│   │           ├── certiq_ppo_trainer.py # CertiqPPOTrainer (Lagrangian)
│   │           └── qgym_import.py        # QGym component imports
│   ├── experiments/
│   │   └── checkpoint_state.py # Checkpoint save/load utilities
│   └── utils/
│       └── platform.py         # Platform detection helpers
├── configs/model/
│   └── certiq_index.yaml       # Model architecture reference
├── tests/                      # pytest test suite
├── training_results/           # Checkpoints and logs (generated)
└── extern/QGym/                # QGym simulator (git submodule, unchanged)
```

## Running All Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@misc{certiq2025,
  title={CertiQ-Net: Learned Certified Dispatch for Queueing Systems},
  author={...},
  year={2025},
}
```
