"""Quick bottleneck profiler — times each component in isolation.

Usage
-----
    python -m certiq_net.studies.qgym_eval.train.profile_bottleneck ^
        vanilla ^
        reentrant_2
    python -m certiq_net.studies.qgym_eval.train.profile_bottleneck ^
        vanilla ^
        reentrant_2 ^
        --device cuda

Each component is timed independently with warmup runs so you can see
exactly where time goes.  Results are printed in a table and a summary
line shows the estimated bottleneck.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

import certiq_net
from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied
from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy

from certiq_net.studies.qgym_eval.train.qgym_import import (
    load_rl_p_env,
)

# ── Paths ────────────────────────────────────────────────────────────────────

_project_root = Path(certiq_net.__file__).resolve().parents[2]
_QGYM_ROOT = _project_root / "extern" / "QGym"
_RL_ROOT = _QGYM_ROOT / "RL"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Timing helper ────────────────────────────────────────────────────────────

class _Timer:
    """Simple wall-clock + sync timer with warmup."""

    def __init__(self, name: str, reps: int, warmup: int = 10, sync: bool = True):
        self.name = name
        self.reps = reps
        self.warmup = warmup
        self.sync = sync

    def __call__(self, fn, *args, **kwargs):
        # warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        if self.sync:
            torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(self.reps):
            fn(*args, **kwargs)
        if self.sync:
            torch.cuda.synchronize()
        ms_per = (_time.perf_counter() - t0) * 1000 / self.reps
        print(f"  {self.name:40s}  {ms_per:>8.2f} ms  ({self.reps}x)")
        return ms_per


# ── Component runners ────────────────────────────────────────────────────────

def _run_env_step(env, action, reps: int) -> None:
    """Single env.step() — the QGym kernel."""
    for _ in range(reps):
        env.step(action)


def _run_policy_forward(policy, obs_tensor) -> None:
    """Full policy forward: action, value, log_prob."""
    with torch.no_grad():
        policy(obs_tensor)


def _run_evaluate_actions(policy, obs_tensor, action_tensor) -> None:
    """Policy evaluate_actions (used in training)."""
    with torch.no_grad():
        policy.evaluate_actions(obs_tensor, action_tensor)


def _run_evaluate_values(policy, obs_tensor) -> None:
    """Value forward (used in training)."""
    with torch.no_grad():
        policy.evaluate_values(obs_tensor)


def _run_ceritq_forward(policy, obs_tensor) -> None:
    """Raw CertiQ _run_certiq only (marginal_index_head forward)."""
    with torch.no_grad():
        policy._run_certiq(obs_tensor)


def _run_training_step(policy, obs_tensor, action_tensor) -> None:
    """One full training step: forward + backward + optimizer."""
    # zero grad
    policy.optimizer.zero_grad()

    # Policy loss
    log_prob, _ = policy.evaluate_actions(obs_tensor, action_tensor)
    # Fake advantage = 1, fake ratio = 1  (standard PPO surrogate)
    policy_loss = -log_prob.mean()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    policy.optimizer.step()

    # Reset for next call (don't accumulate)
    policy.zero_grad(set_to_none=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_patches_applied()

    import argparse
    parser = argparse.ArgumentParser(description="Profile bottleneck")
    parser.add_argument("policy_config", type=str)
    parser.add_argument("env_config", type=str)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. 'cuda', 'cpu')")
    args = parser.parse_args()

    # ── Load configs ────────────────────────────────────────────────────────
    policy_cfg = _load_yaml(_RL_ROOT / "policy_configs" / f"{args.policy_config}.yaml")
    if args.device is not None:
        policy_cfg["env"]["device"] = args.device
    env_cfg = _load_yaml(_QGYM_ROOT / "configs" / "env" / f"{args.env_config}.yaml")
    env_type = env_cfg.get("env_type", env_cfg["name"])
    device_str = policy_cfg["env"]["device"]
    env_device = "cpu" if device_str == "cuda" else device_str
    device = torch.device(device_str)
    env_device_t = torch.device(env_device)

    train_seed = policy_cfg["env"]["train_seed"]
    env_temp = policy_cfg["env"]["env_temp"]
    policy_name = policy_cfg["model"]["policy_name"]
    actors = policy_cfg["training"]["actors"]
    batch_size = policy_cfg["training"]["batch_size"] or 5000
    episode_steps = policy_cfg["training"]["episode_steps"]

    data_dir = _QGYM_ROOT / "configs" / "env_data" / env_type
    network = np.load(data_dir / f"{env_type}_network.npy") if env_cfg.get("network") is None else env_cfg["network"]
    mu = np.load(data_dir / f"{env_type}_mu.npy") if env_cfg.get("mu") is None else env_cfg["mu"]
    network_t = torch.tensor(network, dtype=torch.float)
    mu_t = torch.tensor(mu, dtype=torch.float)
    orig_s, orig_q = network_t.shape

    num_pool = env_cfg.get("num_pool", 1)
    network_t = network_t.repeat_interleave(num_pool, dim=0)
    mu_t = mu_t.repeat_interleave(num_pool, dim=0)

    # ── Create a single env ─────────────────────────────────────────────────
    dq = load_rl_p_env(
        env_config=env_cfg,
        temp=env_temp,
        batch=1,
        seed=train_seed,
        policy_name=policy_name,
        device=env_device_t,
    )

    # ── Build policy ────────────────────────────────────────────────────────
    scale = policy_cfg["model"]["scale"]
    L, J = orig_q, orig_s
    gmLJ = int(np.sqrt(L * J))
    pi_arch = [scale * L, scale * gmLJ, scale * J]
    vi_arch = [scale * L, scale * gmLJ, scale * J]
    randomize = policy_cfg["env"]["randomize"]
    time_f = policy_cfg["env"].get("time_f", False)

    # Convert old gym → gymnasium spaces for SB3 compatibility
    import gymnasium.spaces as gspaces
    obs_space = gspaces.Box(low=0.0, high=np.inf, shape=(orig_q,), dtype=np.float32)
    act_shape = dq.action_space.shape  # (S, Q)
    act_space = gspaces.Box(low=0.0, high=1.0, shape=act_shape, dtype=np.float32)
    lr_schedule = lambda _: policy_cfg["training"]["lr"]

    from stable_baselines3.common.policies import ActorCriticPolicy
    policy = CertiQSB3Policy(
        obs_space, act_space, lr_schedule,
        activation_fn=nn.Tanh,
        network=network_t,
        mu=mu_t,
        alpha=0,
        D=dq.queue_event_options,
        tau=1.0,
        randomize=randomize,
        scale=scale,
        rescale_v=policy_cfg["training"]["rescale_v"],
        time_f=time_f,
        net_arch=dict(pi=pi_arch, vf=vi_arch),
    )
    policy = policy.to(device)
    policy.eval()

    # Make optimizers for training-step test (same as CustomPPOTrainer)
    if hasattr(policy, "log_std"):
        policy.optimizer = torch.optim.Adam([
            {'params': policy.log_std},
            {'params': policy.features_extractor.parameters()},
            {'params': policy.pi_features_extractor.parameters()},
            {'params': policy.mlp_extractor.policy_net.parameters()},
            {'params': policy.action_net.parameters()}
        ], lr=policy_cfg["training"]["lr"])
    else:
        policy.optimizer = torch.optim.Adam([
            {'params': policy.features_extractor.parameters()},
            {'params': policy.pi_features_extractor.parameters()},
            {'params': policy.mlp_extractor.policy_net.parameters()},
            {'params': policy.action_net.parameters()}
        ], lr=policy_cfg["training"]["lr"])
    policy.value_optimizer = torch.optim.Adam([
        {'params': policy.vf_features_extractor.parameters()},
        {'params': policy.mlp_extractor.value_net.parameters()},
        {'params': policy.value_net.parameters()}
    ], lr=policy_cfg["training"]["lr"])

    # ── Prepare test data ───────────────────────────────────────────────────
    obs, _ = dq.reset()  # shape: (q,) or (1, q)
    obs_1d = obs.squeeze()
    single_obs = torch.as_tensor(obs_1d, dtype=torch.float32, device=device)  # (q,)
    batch_obs = single_obs.unsqueeze(0).expand(batch_size, -1).contiguous()   # (B, q)
    actors_obs = single_obs.unsqueeze(0).expand(actors, -1).contiguous()      # (A, q)

    # Single action and batch action
    action = policy(single_obs.unsqueeze(0))[0][0]  # (S, Q)
    dq_action = action.cpu().numpy()
    batch_action = action.unsqueeze(0).expand(batch_size, -1, -1).contiguous()  # (B, S, Q)

    print(f"\nDevice: {device_str}   Env: {env_type}   Actors: {actors}   "
          f"Episode_steps: {episode_steps}   Batch_size: {batch_size}")
    print(f"Observation dim: {orig_q}  Servers: {orig_s}")
    print(f"Per-step workload: {actors} envs x (1 obs + 1 step) per timestep")
    print(f"{'-'*70}")

    timer = _Timer

    # ── 1. Environment stepping ─────────────────────────────────────────────
    print("\n--- QGym Environment ---")
    t_env_single = timer("env.step (1 env, 1 step)", reps=500, warmup=50, sync=False)(
        _run_env_step, dq, dq_action, 500
    )

    # ── 2. Policy forward (single) ──────────────────────────────────────────
    print("\n--- CertiQ Policy Forward ---")
    t_forward_single = timer("policy.forward (batch=1)", reps=500, warmup=50)(
        _run_policy_forward, policy, single_obs.unsqueeze(0)
    )
    t_forward_actors = timer(f"policy.forward (batch={actors})", reps=200, warmup=20)(
        _run_policy_forward, policy, actors_obs
    )
    t_forward_batch = timer(f"policy.forward (batch={batch_size})", reps=50, warmup=10)(
        _run_policy_forward, policy, batch_obs
    )

    # ── 3. Raw CertiQ inference only (no action sampling) ──────────────────
    t_certiq_actors = timer(f"_run_certiq (batch={actors})", reps=200, warmup=20)(
        _run_ceritq_forward, policy, actors_obs
    )
    t_certiq_batch = timer(f"_run_certiq (batch={batch_size})", reps=100, warmup=20)(
        _run_ceritq_forward, policy, batch_obs
    )

    # ── 4. Training components ──────────────────────────────────────────────
    print("\n--- Training Components (used inside model.train()) ---")
    t_eval_act = timer(f"evaluate_actions (batch={batch_size})", reps=50, warmup=10)(
        _run_evaluate_actions, policy, batch_obs, batch_action
    )
    t_eval_val = timer(f"evaluate_values (batch={batch_size})", reps=100, warmup=20)(
        _run_evaluate_values, policy, batch_obs
    )

    # ── 5. Full training step ───────────────────────────────────────────────
    policy.train()
    # For single training step, use a smaller batch
    train_batch = min(batch_size, 5000)
    train_obs = single_obs.unsqueeze(0).expand(train_batch, -1).contiguous()
    train_act = batch_action[:train_batch]
    t_train_step = timer(f"train step (fwd+bwd+opt, batch={train_batch})", reps=20, warmup=5)(
        _run_training_step, policy, train_obs, train_act
    )
    policy.eval()

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BOTTLENECK ANALYSIS (estimated for 1 epoch)")
    print(f"{'='*70}")

    # Per-iteration costs:
    # collect: actors × episode_steps × (forward + step) + 1× buffer
    # train: ppo_epochs × (batch_size/batch) × (eval_act + eval_val + backward*2)
    b = batch_size

    # Collect phase
    fwd_per_timestep = t_forward_actors  # ms for 1 policy forward over `actors` obs
    step_per_timestep = t_env_single * actors  # ms for stepping all `actors` envs
    collect_per_timestep = fwd_per_timestep + step_per_timestep
    collect_total = collect_per_timestep * episode_steps / 1000  # seconds

    # Train phase (estimate: ppo_epochs passes over batch_size)
    ppo_epochs = policy_cfg["training"]["ppo_epochs"]
    # One training pass over the full batch
    # evaluate_actions + evaluate_values + 2×backward
    t_batch_fwd = t_eval_act + t_eval_val
    # training step includes one forward+backward for policy; value is separate
    # Rough estimate: training step × 2 (policy + value)
    train_per_batch = t_train_step * 2  # rough: policy + value
    train_total = train_per_batch * (actors * episode_steps / b) * ppo_epochs / 1000

    iter_total = collect_total + train_total

    print(f"  {'Component':40s}  {'Time (s)':>10s}  {'%':>6s}")
    print(f"  {'-'*58}")
    print(f"  {'Collect: policy forward':40s}  {fwd_per_timestep*episode_steps/1000:>8.1f}s  "
          f"{fwd_per_timestep/collect_per_timestep*100:>5.0f}% of collect")
    print(f"  {'Collect: env stepping':40s}  {step_per_timestep*episode_steps/1000:>8.1f}s  "
          f"{step_per_timestep/collect_per_timestep*100:>5.0f}% of collect")
    print(f"  {'Collect total':40s}  {collect_total:>8.1f}s  "
          f"{collect_total/iter_total*100:>5.0f}% of iter")
    print(f"  {'Train total':40s}  {train_total:>8.1f}s  "
          f"{train_total/iter_total*100:>5.0f}% of iter")
    print(f"  {'Iteration total':40s}  {iter_total:>8.1f}s")

    if iter_total > 0:
        fwd_share = (fwd_per_timestep * episode_steps) / (
            collect_total * 1000 + train_total * 1000
        ) * 100
        step_share = (step_per_timestep * episode_steps) / (
            collect_total * 1000 + train_total * 1000
        ) * 100
        train_share = train_total / iter_total * 100

    print(f"\n{'-'*70}")
    if step_share > train_share and step_share > fwd_share:
        print(f">> BOTTLENECK: QGym environment stepping ({step_share:.0f}% of iteration)")
    elif train_share > step_share and train_share > fwd_share:
        print(f">> BOTTLENECK: CertiQ training ({train_share:.0f}% of iteration)")
    elif fwd_share > step_share and fwd_share > train_share:
        print(f">> BOTTLENECK: CertiQ policy forward ({fwd_share:.0f}% of iteration)")
    else:
        print(f">> Mixed: stepping {step_share:.0f}%, forward {fwd_share:.0f}%, train {train_share:.0f}%")
    print(f"{'-'*70}\n")


if __name__ == "__main__":
    main()
