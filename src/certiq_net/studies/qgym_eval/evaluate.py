"""Evaluate a trained CertiQIndexModel inside QGym's discrete-event simulator.

Usage
-----
    # With QGym submodule at extern/QGym:
    python -m certiq_net.studies.qgym_eval.evaluate ^
        --env reentrant_2 ^
        --checkpoint path/to/model.pt

    # Or specify qgym-root if QGym is elsewhere:
    python -m certiq_net.studies.qgym_eval.evaluate ^
        --env criss_cross_bh ^
        --checkpoint path/to/model.pt ^
        --qgym-root /path/to/QGym
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CertiQIndexModel inside QGym discrete-event simulator"
    )
    parser.add_argument("--env", type=str, default="reentrant_2",
                        help="QGym environment name (matches configs/env/<name>.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu, cuda, mps)")
    parser.add_argument("--test-batch", type=int, default=100,
                        help="Number of parallel QGym environments for evaluation")
    parser.add_argument("--test-steps", type=int, default=None,
                        help="Override env test_T (default: use env config value)")
    parser.add_argument("--qgym-root", type=str, default=None,
                        help="Path to QGym root (default: <project_root>/extern/QGym)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results JSON (default: results/)")
    return parser


def load_env_config(env_name: str, qgym_root: Path) -> dict:
    import yaml

    env_yaml = qgym_root / "configs" / "env" / f"{env_name}.yaml"
    if not env_yaml.exists():
        available = list((qgym_root / "configs" / "env").glob("*.yaml"))
        print(f"Env config not found: {env_yaml}", file=sys.stderr)
        print(f"Available: {[p.stem for p in available]}", file=sys.stderr)
        sys.exit(1)

    with open(env_yaml) as f:
        cfg = yaml.safe_load(f)

    env_type = cfg.get("env_type", cfg["name"])
    data_dir = qgym_root / "configs" / "env_data" / env_type

    if cfg.get("network") is None:
        cfg["network"] = np.load(data_dir / f"{env_type}_network.npy")
    if cfg.get("mu") is None:
        cfg["mu"] = np.load(data_dir / f"{env_type}_mu.npy")

    cfg["network"] = torch.tensor(cfg["network"], dtype=torch.float)
    cfg["mu"] = torch.tensor(cfg["mu"], dtype=torch.float)

    lam_params = cfg["lam_params"]
    if lam_params.get("val") is None:
        lam_params["val"] = np.load(data_dir / f"{env_type}_lam.npy")
    else:
        lam_params["val"] = np.array(lam_params["val"], dtype=float)

    if cfg.get("queue_event_options") == "custom":
        cfg["queue_event_options"] = torch.tensor(
            np.load(data_dir / f"{env_type}_delta.npy"), dtype=torch.float
        )
    elif isinstance(cfg.get("queue_event_options"), list):
        cfg["queue_event_options"] = torch.tensor(cfg["queue_event_options"], dtype=torch.float)

    if "server_pool_size" not in cfg or cfg["server_pool_size"] is None:
        cfg["server_pool_size"] = torch.ones(cfg["network"].shape[0])

    return cfg


def build_model_config(device: str, test_batch: int) -> dict:
    return {
        "name": "certiq",
        "env": {
            "device": device,
            "env_temp": 1.0,
            "test_seed": 42,
            "test_restart": True,
            "train_restart": False,
            "print_grads": False,
        },
        "opt": {"test_batch": test_batch, "train_batch": 1},
        "policy": {"test_policy": "linear_assigment", "train_policy": "linear_assigment"},
    }


def make_lam_fn(env_config: dict):
    lam_type = env_config["lam_type"]
    lam_params = env_config["lam_params"]
    lam_r_value = lam_params["val"]

    if lam_type == "constant":
        def lam_fn(t):
            return lam_r_value

    elif lam_type == "step":
        t_step = lam_params["t_step"]
        val1 = np.array(lam_params["val1"], dtype=float)
        val2 = np.array(lam_params["val2"], dtype=float)

        def lam_fn(t):
            is_surge = 1 * (t <= t_step)
            return is_surge * val1 + (1 - is_surge) * val2

    elif lam_type == "hyper":
        scale = lam_params.get("scale", 0.8)

        def lam_fn(t, rng=None, batch=None):
            if rng is None or batch is None:
                return lam_r_value
            lam_r_2d = lam_r_value.reshape((1, len(lam_r_value))).repeat(batch, axis=0)
            switch = rng.binomial(1, 0.5, (batch, 1))
            return switch * (lam_r_2d / (1 + scale)) + (1 - switch) * (lam_r_2d / (1 - scale))

    else:
        raise ValueError(f"Unknown lam_type: {lam_type}")

    return lam_fn


def make_draw_fns(env_config: dict, lam_fn):
    orig_q = env_config["network"].shape[1]
    service_type = env_config.get("service_type", "exponential")
    hyper_scale = env_config.get("lam_params", {}).get("scale", 0.8)

    def draw_service(self, time):
        def service_dists(state, batch, t):
            if service_type == "hyper":
                coins = state.binomial(1, 0.5, size=(batch, orig_q))
                a = state.exponential((1 + hyper_scale), (batch, orig_q))
                b = state.exponential((1 - hyper_scale), (batch, orig_q))
                return coins * a + (1 - coins) * b
            return state.exponential(1, (batch, orig_q))
        service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)
        return service

    def draw_inter_arrivals(self, time):
        def inter_arrival_dists(state, batch, t):
            exps = state.exponential(1, (batch, orig_q))
            lam_rate = lam_fn(t, rng=state, batch=batch)
            return exps / lam_rate
        interarrivals = torch.tensor(inter_arrival_dists(self.state, self.batch, time)).to(self.device)
        return interarrivals

    return draw_service, draw_inter_arrivals


def instantiate_model(env_config: dict, device: str) -> torch.nn.Module:
    from certiq_net.dispatcher.certiq.index_model import CertiQIndexModel

    N = env_config["network"].shape[1]
    model = CertiQIndexModel(
        N=N,
        hidden_dim=128,
        tau=1.0,
        exploration_temperature=1.0,
        C=20.0,
        beta=1.0,
        cost_fn="qmd",
        encoder_layers=2,
        num_heads=4,
        num_inducing_points=4,
        dropout=0.0,
        constraint_mode="lagrangian",
    )
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        cleaned = {
            k.removeprefix("model."): v
            for k, v in sd.items()
            if k.startswith("model.")
        }
        model.load_state_dict(cleaned, strict=False)
    elif isinstance(raw, dict):
        model.load_state_dict(raw, strict=False)
    else:
        model.load_state_dict(raw, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}", file=sys.stderr)


def run_evaluation(args: argparse.Namespace) -> dict:
    # Lazily resolve project root and QGym submodule
    certiq_net_root = Path(__file__).resolve().parent.parent.parent.parent

    if args.qgym_root:
        qgym_root = Path(args.qgym_root).resolve()
    else:
        qgym_root = certiq_net_root / "extern" / "QGym"

    if not qgym_root.exists():
        print(f"QGym root not found: {qgym_root}", file=sys.stderr)
        print("Pass --qgym-root to point to the QGym directory.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(qgym_root))
    from main.trainer import Trainer

    from certiq_net.studies.qgym_eval.policy import CertiQPolicy

    device = args.device
    env_name = args.env

    print(f"Loading env: {env_name}", file=sys.stderr)
    env_config = load_env_config(env_name, qgym_root)
    if args.test_steps is not None:
        old_T = env_config.get("test_T", "default")
        env_config["test_T"] = args.test_steps
        print(f"  test_T: {old_T} -> {args.test_steps}", file=sys.stderr)

    print(f"Instantiating CertiQ model for N={env_config['network'].shape[1]}", file=sys.stderr)
    model = instantiate_model(env_config, device)

    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    load_checkpoint(model, args.checkpoint)
    model.to(device)

    policy = CertiQPolicy(model, device=device)
    lam_fn = make_lam_fn(env_config)
    draw_service, draw_inter_arrivals = make_draw_fns(env_config, lam_fn)
    model_config = build_model_config(device, args.test_batch)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = f"{env_name}_certiq"

    print(f"Creating Trainer (test_batch={args.test_batch}, T={env_config.get('test_T', 10000)})", file=sys.stderr)
    trainer = Trainer(
        model_config, env_config, policy, optimizer=None,
        draw_service=draw_service,
        draw_inter_arrivals=draw_inter_arrivals,
        experiment_name=experiment_name,
    )

    print("Running test_epoch...", file=sys.stderr)
    trainer.test_epoch(0)

    result = trainer.test_loss[-1] if trainer.test_loss else {}
    print(f"\nResults: test_loss={result.get('test_loss'):.4f} +/- {result.get('test_loss_std'):.4f}", file=sys.stderr)
    print(f"Queue lengths: {result.get('mean_queue_length')}", file=sys.stderr)

    out_path = output_dir / f"{experiment_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}", file=sys.stderr)

    return result


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)
