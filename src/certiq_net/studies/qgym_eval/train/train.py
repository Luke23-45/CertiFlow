"""Train CertiQIndexModel using QGym's PPO pipeline.

Usage
-----
    python -m certiq_net.studies.qgym_eval.train.train ^
        vanilla ^
        reentrant_2
    python -m certiq_net.studies.qgym_eval.train.train ^
        vanilla ^
        reentrant_2 ^
        --parallel --num-actors 8

The first positional arg is the policy-config stem (``vanilla.yaml`` in
``extern/QGym/RL/policy_configs/``). The second is the env-config stem
(``reentrant_2.yaml`` in ``extern/QGym/configs/env/``).

Flags
-----
``--device cpu``
    Force CPU training (recommended for MLP policy without CNN).
``--parallel``
    Use ``SubprocVecEnv`` for parallel environment stepping.
``--num-actors N``
    Override number of parallel envs (default: from config; keep <=~8 on
    Windows to avoid page-file limits from spawn subprocesses).
``--compile``
    Apply ``torch.compile`` to the policy forward pass (experimental).
``--num-threads N``
    Override ``torch.set_num_threads``. For the default ``DummyVecEnv`` path
    with tiny per-step ops, ``1`` often beats the default because it avoids
    thread oversubscription across the serial env loop.
``--use-lagrangian``
    Enable the CertiQ certificate constraint in the PPO loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

import certiq_net
from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied
from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv

from certiq_net.studies.qgym_eval.train.qgym_import import (
    CustomPPOTrainer,
    CustomRolloutBuffer,
    load_rl_p_env,
    parallel_eval,
)

_project_root = Path(certiq_net.__file__).resolve().parents[2]
_QGYM_ROOT = _project_root / "extern" / "QGym"
_RL_ROOT = _QGYM_ROOT / "RL"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    # Apply QGym submodule patches (tensor fixes, etc.) before any QGym imports
    ensure_patches_applied()

    parser = argparse.ArgumentParser(
        description="Train CertiQ using QGym's PPO pipeline"
    )
    parser.add_argument("policy_config", type=str, help="Policy config stem (e.g. vanilla)")
    parser.add_argument("env_config", type=str, help="Env config stem (e.g. reentrant_2)")
    parser.add_argument(
        "--use-lagrangian",
        action="store_true",
        help="Enable CertiQ Lagrangian constraint in the PPO loss",
    )
    parser.add_argument(
        "--lr-nu", type=float, default=1e-3, help="Learning rate for the Lagrangian dual variable"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (e.g. 'cuda', 'cpu')"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile to policy forward pass (experimental)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use SubprocVecEnv for parallel env stepping (requires --num-actors <= ~8 on Windows)",
    )
    parser.add_argument(
        "--num-actors", type=int, default=None,
        help="Override the number of actor envs (default: from policy config)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=None,
        help="Override torch.set_num_threads (default: torch default). For the "
             "DummyVecEnv path with tiny per-step ops, 1 often beats the default "
             "because it avoids thread oversubscription across the serial env loop.",
    )
    args = parser.parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"[train] torch.set_num_threads({args.num_threads})")

    # ── Load configs ────────────────────────────────────────────────────────
    policy_cfg = _load_yaml(_RL_ROOT / "policy_configs" / f"{args.policy_config}.yaml")
    if args.device is not None:
        policy_cfg["env"]["device"] = args.device
    env_cfg = _load_yaml(_QGYM_ROOT / "configs" / "env" / f"{args.env_config}.yaml")
    env_type = env_cfg.get("env_type", env_cfg["name"])

    # ── Load environment data ──────────────────────────────────────────────
    data_dir = _QGYM_ROOT / "configs" / "env_data" / env_type

    if env_cfg.get("network") is None:
        network = np.load(data_dir / f"{env_type}_network.npy")
    else:
        network = env_cfg["network"]

    if env_cfg.get("mu") is None:
        mu = np.load(data_dir / f"{env_type}_mu.npy")
    else:
        mu = env_cfg["mu"]

    network_t = torch.tensor(network, dtype=torch.float)
    mu_t = torch.tensor(mu, dtype=torch.float)
    orig_s, orig_q = network_t.shape

    num_pool = env_cfg.get("num_pool", 1)
    network_t = network_t.repeat_interleave(num_pool, dim=0)
    mu_t = mu_t.repeat_interleave(num_pool, dim=0)

    init_test_queues = torch.tensor([env_cfg["init_queues"]]).float()

    # ── Training hyperparams ───────────────────────────────────────────────
    device = policy_cfg["env"]["device"]
    env_device = "cpu" if device == "cuda" else device
    test_seed = policy_cfg["env"]["test_seed"]
    train_seed = policy_cfg["env"]["train_seed"]
    env_temp = policy_cfg["env"]["env_temp"]
    randomize = policy_cfg["env"]["randomize"]
    time_f = policy_cfg["env"].get("time_f", False)
    policy_name = policy_cfg["model"]["policy_name"]

    actors = args.num_actors if args.num_actors is not None else policy_cfg["training"]["actors"]
    num_epochs = policy_cfg["training"]["num_epochs"]
    episode_steps = policy_cfg["training"]["episode_steps"]
    total_steps = num_epochs * episode_steps * actors
    eval_freq = episode_steps
    test_T = env_cfg.get("test_T", 10000)

    # ── Create environments ────────────────────────────────────────────────
    def make_env(seed):
        return load_rl_p_env(
            env_config=env_cfg,
            temp=env_temp,
            batch=1,
            seed=seed,
            policy_name=policy_name,
            device=torch.device(env_device),
        )

    def make_test_env(seed):
        return load_rl_p_env(
            env_config=env_cfg,
            temp=env_temp,
            batch=1,
            seed=seed,
            policy_name=policy_name,
            device=torch.device(env_device),
        )

    dq_raw = load_rl_p_env(
        env_config=env_cfg,
        temp=env_temp,
        batch=1,
        seed=train_seed,
        policy_name=policy_name,
        device=torch.device(env_device),
    )

    env_fns = [lambda s=seed: make_env(s) for seed in range(train_seed, train_seed + actors)]
    if args.parallel:
        start_method = "spawn" if sys.platform == "win32" else "fork"
        dq = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        dq = DummyVecEnv(env_fns)

    dq_test_list = [make_test_env(seed) for seed in range(test_seed, test_seed + 100)]

    # ── Policy kwargs ──────────────────────────────────────────────────────
    scale = policy_cfg["model"]["scale"]
    L, J = orig_q, orig_s
    gmLJ = int(np.sqrt(L * J))
    pi_arch = [scale * L, scale * gmLJ, scale * J]
    vi_arch = [scale * L, scale * gmLJ, scale * J]

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        network=network_t,
        mu=mu_t,
        alpha=0,
        D=dq_raw.queue_event_options,
        tau=1.0,
        randomize=randomize,
        scale=scale,
        rescale_v=policy_cfg["training"]["rescale_v"],
        time_f=time_f,
        net_arch=dict(pi=pi_arch, vf=vi_arch),
    )

    rollout_buffer_kwargs = dict(
        q=orig_q,
        normalize_advantage=policy_cfg["training"]["normalize_advantage"],
        normalize_value=policy_cfg["training"]["normalize_value"],
        normalize_reward=policy_cfg["training"]["normalize_reward"],
        truncation=policy_cfg["training"]["truncation"],
        var_scaler=policy_cfg["training"]["var_scaler"],
        per_iter_normal_value=policy_cfg["training"]["per_iter_normal_value"],
    )

    trainer_kwargs = dict(
        learning_rate=policy_cfg["training"]["lr"],
        lr_policy=policy_cfg["training"]["lr_policy"],
        lr_value=policy_cfg["training"]["lr_value"],
        min_lr_policy=policy_cfg["training"]["min_lr_policy"],
        min_lr_value=policy_cfg["training"]["min_lr_value"],
        amp_value=policy_cfg["training"].get("amp_value", False),
        n_steps=episode_steps,
        batch_size=policy_cfg["training"]["batch_size"],
        num_epochs=num_epochs,
        n_epochs=policy_cfg["training"]["ppo_epochs"],
        gamma=policy_cfg["training"]["gamma"],
        gae_lambda=policy_cfg["training"]["gae_lambda"],
        clip_range=0.2,
        clip_range_vf=policy_cfg["training"]["clip_range_vf"],
        normalize_advantage=policy_cfg["training"]["normalize_advantage"],
        normalize_value=policy_cfg["training"]["normalize_value"],
        rescale_v=policy_cfg["training"]["rescale_v"],
        ent_coef=policy_cfg["training"]["ent_coef"],
        actors=actors,
        vf_coef=policy_cfg["training"]["vf_coef"],
        max_grad_norm=1.0,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=CustomRolloutBuffer,
        rollout_buffer_kwargs=rollout_buffer_kwargs,
        target_kl=policy_cfg["training"]["target_kl"],
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=None,
        device=device,
        _init_setup_model=True,
    )

    trainer_cls = CustomPPOTrainer
    if args.use_lagrangian:
        from certiq_net.studies.qgym_eval.train.certiq_ppo_trainer import CertiqPPOTrainer

        trainer_cls = CertiqPPOTrainer
        trainer_kwargs["lr_nu"] = args.lr_nu

    model = trainer_cls(CertiQSB3Policy, dq, **trainer_kwargs)

    # ── Optional: compile policy for faster forward pass ──────────────────
    if args.compile:
        print("[train] Applying torch.compile to policy (mode=reduce-overhead) …")
        model.policy._run_certiq = torch.compile(
            model.policy._run_certiq, mode="reduce-overhead"
        )

    # ── Evaluation callback ────────────────────────────────────────────────
    test_policy = policy_cfg["policy"]["test_policy"]
    test_batch = policy_cfg["training"]["test_batch"]
    per_iter_normal_obs = policy_cfg["training"]["per_iter_normal_obs"]
    bc = policy_cfg["training"].get("behavior_cloning", False)

    eval_callback = parallel_eval(
        model=model,
        eval_env=dq_test_list,
        eval_freq=eval_freq,
        eval_t=test_T,
        test_policy=test_policy,
        test_seed=test_seed,
        init_test_queues=init_test_queues,
        test_batch=test_batch,
        device=device,
        num_pool=num_pool,
        time_f=time_f,
        randomize=randomize,
        policy_name=policy_name,
        per_iter_normal_obs=per_iter_normal_obs,
        env_config_name=args.env_config,
        bc=bc,
        verbose=1,
    )

    eval_callback.pre_train_eval()

    # ── Train ──────────────────────────────────────────────────────────────
    model.learn(total_timesteps=total_steps, log_interval=1, callback=eval_callback)

    test_cost_list = eval_callback.test_costs
    final_cost_list = eval_callback.final_costs

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test_cost_list.json", "w") as f:
        json.dump(test_cost_list, f)
    with open(output_dir / "final_costs.json", "w") as f:
        json.dump(final_cost_list, f)

    print(f"Training complete. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
