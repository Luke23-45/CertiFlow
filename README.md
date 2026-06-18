# CertiQ-Net

Learned certified dispatch for queueing systems, powered by [QGym](https://github.com/namkoong-lab/QGym).

**CertiQ-Net** introduces a **learned marginal-cost index policy** with a **closed-form certificate guarantee** for queueing network control. The policy is a Set Transformer that maps per-queue state (queue length, service rate) to dispatch probabilities, with a certificate that the expected delay cost stays within a provable budget.

## Architecture

```
┌──────────────────────┐
│   QGym Environment   │  Discrete-event queueing simulation
│ DiffDiscreteEventSys │  (reentrant, criss-cross, parallel server, ...)
└──────┬───────────────┘
       │ queues (B, Q), time (B, 1)
       ▼
┌──────────────────────────────┐
│       CertiQPolicy           │  Wraps model for QGym interface
│  (policy.py)                 │  Converts per-queue pi → (B, S, Q) priority
└──────┬───────────────────────┘
       │ Q (B, Q), μ_eff (B, Q)
       ▼
┌──────────────────────────────────────────────────────┐
│                  CertiQIndexModel                     │
│  ┌──────────────────────────────────────────────────┐│
│  │ DispatchInteractionEncoder                       ││
│  │  (Set Transformer with Induced Attention)        ││
│  │  token_features: [Q, μ, log(1+Q), log(μ), ...]  ││
│  └──────────────┬───────────────────────────────────┘│
│                 ▼ z_local (B, N, d), z_global (B, d) │
│  ┌──────────────────────────────────────────────────┐│
│  │ MarginalIndexHead                                ││
│  │  concat(z_local, z_global) → MLP → per-queue     ││
│  │  logits (B, Q) → softmax → dispatch policy π    ││
│  └──────────────┬───────────────────────────────────┘│
│                 ▼                                     │
│  ┌──────────────────────────────────────────────────┐│
│  │ Certificate Mechanism                            ││
│  │  budget = min_i cost_i + C                      ││
│  │  • Lagrangian: constraint in PPO loss            ││
│  │  • Projection: Differentiable KL projection      ││
│  │    onto constraint set {π | E_π[cost] ≤ budget} ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
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

```bash
python -m certiq_net.studies.train \
    --env reentrant_2 \
    --epochs 200 \
    --lr 3e-4 \
    --eval-interval 20

# Optional: specify device, resume from checkpoint
python -m certiq_net.studies.train \
    --env criss_cross_bh \
    --epochs 300 \
    --device cuda \
    --resume training_results/checkpoint_epoch_0050.pt
```

### Training Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `reentrant_2` | QGym environment name |
| `--epochs` | `200` | Number of training epochs |
| `--train-steps` | env default | Steps per training episode |
| `--device` | `cpu` | Torch device |
| `--lr` | `3e-4` | Policy learning rate |
| `--lagrangian-lr` | `1e-2` | Lagrangian multiplier learning rate |
| `--seed` | `42` | Random seed |
| `--eval-interval` | `20` | Epochs between evaluations |
| `--eval-batch` | `50` | Evaluation parallel environments |
| `--eval-steps` | `5000` | Evaluation trajectory length |
| `--output-dir` | `training_results` | Checkpoint and log directory |
| `--resume` | None | Resume from checkpoint |

## Evaluate a Trained Model

```bash
python -m certiq_net.studies.qgym_eval.evaluate \
    --env reentrant_2 \
    --checkpoint training_results/final_model_state.pt \
    --test-batch 100

# With hyperexponential arrivals
python -m certiq_net.studies.qgym_eval.evaluate \
    --env reentrant_2_hyper \
    --checkpoint checkpoints/reentrant_2_hyper.pt
```

### Evaluation Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `reentrant_2` | QGym environment name |
| `--checkpoint` | required | Trained model `.pt` file |
| `--device` | `cpu` | Torch device |
| `--test-batch` | `100` | Parallel environments for evaluation |
| `--test-steps` | env default | Simulation horizon |
| `--seed` | `42` | Random seed for reproducibility |
| `--model-config` | `configs/model/certiq_index.yaml` | Model architecture config |
| `--qgym-root` | `./extern/QGym` | Path to QGym directory |
| `--output-dir` | `results` | Results output directory |

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
  _target_: certiq_net.dispatcher.certiq.index_model.CertiQIndexModel
  N: 8                       # Number of queues
  hidden_dim: 128            # Transformer embedding dimension
  tau: 1.0                   # Softmax temperature
  exploration_temperature: 1.5  # Training exploration temperature
  C: 20.0                    # Certificate slack budget constant
  cost_fn: qmd               # Cost function: "sed", "qmd", or "learned"
  encoder_layers: 2          # Set Transformer layers
  num_heads: 4               # Attention heads
  num_inducing_points: 4     # Induced attention points (0 = full self-attention)
  constraint_mode: lagrangian  # "lagrangian", "projection", or "unconstrained"
```

### Constraint Modes

| Mode | Description | Training Loss |
|------|-------------|---------------|
| `lagrangian` | Soft constraint via PPO-Lagrangian dual variable | `L = L_PPO + λ · constraint_violation` |
| `projection` | Hard constraint via Differentiable KL Projection | Standard PPO (constraint enforced by projection layer) |
| `unconstrained` | Plain softmax, no certificate | Standard PPO |

### Cost Functions

The internal cost used by the certificate:

- **`sed`**: Shortest Expected Delay — `(Q_i + 1) / μ_i`
- **`qmd`**: Quadratic Drift — `(2·Q_i + 1) / μ_i`
- **`learned`**: MLP that learns cost from `(Q_i, μ_i, ξ_i)`

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
│   │   ├── train.py            # PPO-Lagrangian training CLI
│   │   ├── ppo_trainer.py      # LagrangianPPOTrainer implementation
│   │   └── qgym_eval/
│   │       ├── evaluate.py     # Evaluation CLI
│   │       └── policy.py       # CertiQPolicy wrapper
│   ├── experiments/
│   │   └── checkpoint_state.py # Checkpoint save/load utilities
│   └── utils/
│       └── platform.py         # Platform detection helpers
├── configs/model/
│   └── certiq_index.yaml       # Model architecture reference
├── tests/                      # pytest test suite
├── training_results/           # Checkpoints and logs (generated)
└── extern/QGym/                # QGym simulator (git submodule)
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
