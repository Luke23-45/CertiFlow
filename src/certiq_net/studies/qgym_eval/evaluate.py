"""Evaluate a trained CertiQSB3Policy using the QGym RL/PPO pipeline.

Uses the identical evaluation code path as QGym's ``parallel_eval``
callback (``load_rl_p_env`` + ``model.predict`` + ``env.step``) so
results are directly comparable with baseline policies trained via
the same pipeline.

Usage
-----
    python -m certiq_net.studies.qgym_eval.evaluate ^
        --env reentrant_2 ^
        --checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from certiq_net.studies.qgym_eval._qgym_paths import project_root, resolve_qgym_root
from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy
from certiq_net.studies.qgym_eval.train.qgym_import import load_rl_p_env


class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CertiQSB3Policy using the QGym RL pipeline"
    )
    parser.add_argument("--env", type=str, default="reentrant_2",
                        help="QGym environment name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--test-batch", type=int, default=100,
                        help="Number of parallel environments")
    parser.add_argument("--test-steps", type=int, default=None,
                        help="Override env test_T")
    parser.add_argument("--qgym-root", type=str, default=None,
                        help="Path to QGym root")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config YAML")
    return parser


def _load_env_config(env_name: str, qgym_root: Path) -> dict:
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


def _constant_lr(_):
    return 1e-4


def _construct_policy(
    env_config: dict,
    env_sample,
    device: str,
    model_config_path: Path | None = None,
) -> CertiQSB3Policy:
    import yaml

    if model_config_path is not None and model_config_path.exists():
        with open(model_config_path) as f:
            cfg = yaml.safe_load(f)
            if "model" in cfg:
                cfg = cfg["model"]
    else:
        cfg = {}

    network = env_config["network"].to(device)
    mu = env_config["mu"].to(device)

    if hasattr(env_sample, "queue_event_options") and env_sample.queue_event_options is not None:
        opts = env_sample.queue_event_options
        D = opts.to(device) if torch.is_tensor(opts) else torch.tensor(opts).to(device)
    else:
        D = torch.tensor(0.0).to(device)

    policy = CertiQSB3Policy(
        observation_space=env_sample.observation_space,
        action_space=env_sample.action_space,
        lr_schedule=_constant_lr,
        network=network,
        mu=mu,
        alpha=torch.tensor(0.0).to(device),
        D=D,
        tau=cfg.get("tau", 1.0),
        randomize=True,
        scale=cfg.get("scale", 20),
        rescale_v=False,
        time_f=False,
        net_arch=dict(pi=[], vf=[]),
    )
    policy.to(device)
    policy.eval()
    return policy


def _load_checkpoint_into_policy(policy: CertiQSB3Policy, checkpoint_path: str, device: str) -> None:
    raw = torch.load(str(checkpoint_path), map_location=device, weights_only=True)

    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(raw)}")

    if "state_dict" in raw:
        raw = raw["state_dict"]

    head_sd = {}
    for k, v in raw.items():
        key = k
        if key.startswith("model."):
            key = key[6:]
        if key.startswith("marginal_index_head."):
            key = key[21:]
        if key.startswith("index_head."):
            key = key[11:]
        if key.startswith("encoder.") or key.startswith("index_head.") or key.startswith("value_head."):
            head_sd[key] = v

    if head_sd:
        policy.marginal_index_head.load_state_dict(head_sd, strict=True)
        print(f"Loaded checkpoint into policy.marginal_index_head ({len(head_sd)} keys)", file=sys.stderr)
        return

    missing, unexpected = policy.load_state_dict(raw, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}", file=sys.stderr)
    if unexpected:
        redundant = {"action_net", "value_net", "pi_features_extractor",
                     "vf_features_extractor", "mlp_extractor"}
        real_unexpected = [k for k in unexpected if k.split(".")[0] not in redundant]
        if real_unexpected:
            print(f"Warning: unexpected keys: {real_unexpected}", file=sys.stderr)

    print(f"Loaded checkpoint: {checkpoint_path}", file=sys.stderr)


def evaluate_checkpoint(
    env_name: str,
    checkpoint_path: Path,
    *,
    qgym_root: Path,
    device: str = "cpu",
    test_batch: int = 100,
    test_steps: int | None = None,
    seed: int = 42,
    model_config_path: Path | None = None,
) -> dict:
    env_config = _load_env_config(env_name, qgym_root)
    if test_steps is not None:
        env_config["test_T"] = test_steps
    test_T = env_config.get("test_T", 10000)

    orig_q = env_config["network"].shape[1]

    def _make_env(s):
        return load_rl_p_env(
            env_config=env_config,
            temp=1.0,
            batch=1,
            seed=s,
            policy_name="vanilla",
            device=torch.device(device),
        )

    envs = [_make_env(seed + i) for i in range(test_batch)]
    env_sample = envs[0]

    policy = _construct_policy(env_config, env_sample, device, model_config_path)
    _load_checkpoint_into_policy(policy, str(checkpoint_path), device)

    obs_batch: list[Obs] = []
    total_cost_batch: list[torch.Tensor] = []
    total_time_batch: list[torch.Tensor] = []
    twql_batch: list[torch.Tensor] = []

    for i in range(test_batch):
        obs_np, _ = envs[i].reset()
        obs_batch.append(Obs(queues=torch.tensor(obs_np, dtype=torch.float).to(device),
                             time=torch.tensor(0.0).to(device)))
        total_cost_batch.append(torch.tensor(0.0).to(device))
        total_time_batch.append(torch.tensor(0.0).to(device))
        twql_batch.append(torch.zeros(orig_q).to(device))

    for _ in range(test_T):
        batch_queue = torch.cat([o.queues.view(1, -1) for o in obs_batch], dim=0)
        raw_actions, _ = policy.predict(batch_queue.cpu().numpy())
        action = torch.tensor(raw_actions).float().to(device)

        for i in range(test_batch):
            _, _, _, _, info = envs[i].step(action[i])
            obs_batch[i] = info["obs"]
            total_cost_batch[i] = total_cost_batch[i] + info["cost"]
            total_time_batch[i] = total_time_batch[i] + info["event_time"]
            twql_batch[i] = twql_batch[i] + info["queues"] * info["event_time"]

    test_cost_batch = torch.stack([
        total_cost_batch[i] / total_time_batch[i]
        for i in range(test_batch)
    ])
    test_loss = test_cost_batch.mean().item()
    test_loss_std = test_cost_batch.std().item()

    mean_queue_lengths = torch.stack([
        twql_batch[i] / total_time_batch[i]
        for i in range(test_batch)
    ]).mean(dim=0).cpu().tolist()
    mean_queue_length_avg = float(np.mean(mean_queue_lengths))

    result = {
        "env_name": env_name,
        "test_loss": test_loss,
        "test_loss_std": test_loss_std,
        "mean_queue_length": mean_queue_lengths,
        "mean_queue_length_avg": mean_queue_length_avg,
    }

    return result


def run_evaluation(args: argparse.Namespace) -> dict:
    if args.qgym_root:
        qgym_root = Path(args.qgym_root).resolve()
        if not (qgym_root / "RL" / "PPO" / "trainer.py").exists():
            print(f"QGym root not found or invalid: {qgym_root}", file=sys.stderr)
            sys.exit(1)
    else:
        qgym_root = resolve_qgym_root()

    if args.model_config:
        model_config_path = Path(args.model_config).resolve()
    else:
        model_config_path = project_root() / "configs" / "model" / "certiq_index.yaml"

    result = evaluate_checkpoint(
        env_name=args.env,
        checkpoint_path=Path(args.checkpoint),
        qgym_root=qgym_root,
        device=args.device,
        test_batch=args.test_batch,
        test_steps=args.test_steps,
        seed=args.seed,
        model_config_path=model_config_path,
    )

    print(f"\nResults for {args.env}:", file=sys.stderr)
    print(f"  test_loss:     {result['test_loss']:.4f} +/- {result['test_loss_std']:.4f}", file=sys.stderr)
    print(f"  mean queue len: {result['mean_queue_length_avg']:.4f}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.env}_certiq_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}", file=sys.stderr)

    return result


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)
