# CertiQ-Net

Learned certified dispatch for queueing systems, powered by [QGym](https://github.com/namkoong-lab/QGym).

This project provides the **CertiQIndexModel** — a transformer-based dispatch policy with a closed-form certificate guarantee — and tooling for evaluating it inside QGym's discrete-event simulator.

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

## Evaluate a Trained Model

```bash
python -m certiq_net.studies.qgym_eval.evaluate \
    --env reentrant_2 \
    --checkpoint /path/to/final_model_state.pt \
    --test-batch 100
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `reentrant_2` | QGym env name (matches `configs/env/<name>.yaml` in QGym) |
| `--checkpoint` | required | Trained model `.pt` file |
| `--device` | `cpu` | Torch device (`cpu`, `cuda`, `mps`) |
| `--test-batch` | `100` | Parallel QGym environments |
| `--test-steps` | env default | Override simulation horizon |
| `--qgym-root` | `./extern/QGym` | Path to QGym (auto-detected from submodule) |
| `--output-dir` | `results` | Where to save results JSON |

## Project Structure

```
certiq_net/
├── pyproject.toml          # Package metadata & dependencies
├── README.md
├── src/certiq_net/         # The certiq_net package
│   ├── dispatcher/         # Core model (CertiQIndexModel, certificate, encoder)
│   ├── experiments/        # Checkpoint save/load utilities
│   ├── studies/qgym_eval/  # QGym evaluation entrypoint & policy wrapper
│   └── utils/              # Platform helpers
├── configs/model/          # Model architecture reference YAML
├── tests/                  # Smoke tests (model instantiation, forward pass)
└── extern/QGym/            # QGym simulator (git submodule)
```

## Model

`CertiQIndexModel` computes a per-resource marginal cost index using a Set Transformer encoder, projects it through a softmax to produce a dispatch policy, and tracks certificate slack for constraint verification. The certificate slack is available in `DispatcherDiagnostics.certificate_slack` during evaluation.

## Dependencies

- **Runtime**: PyTorch, NumPy, SciPy, tqdm, PyYAML
- **QGym evaluation** (optional): `pip install -e ".[qgym]"` adds pytorch-lightning, torchmetrics
- **Development**: `pip install -e ".[dev]"` adds pytest, ruff
