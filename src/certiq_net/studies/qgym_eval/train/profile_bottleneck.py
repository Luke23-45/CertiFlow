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

# Apply QGym submodule patches BEFORE any import that triggers QGym module
# loading.  Python caches imported modules in sys.modules; if we patch the
# files on disk after the module is already loaded, the in-memory copy still
# has the old (unpatched) code.
ensure_patches_applied()

from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy

from certiq_net.studies.qgym_eval.train.qgym_import import (
    load_rl_p_env,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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
        print(f"  [{self.name}] warming up {self.warmup}x ...", end=" ", flush=True)
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        print(f"timing {self.reps}x ...", end=" ", flush=True)
        if self.sync:
            torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(self.reps):
            fn(*args, **kwargs)
        if self.sync:
            torch.cuda.synchronize()
        ms_per = (_time.perf_counter() - t0) * 1000 / self.reps
        print(f" done.  avg {ms_per:.2f} ms")
        return ms_per


# ── Component runners ────────────────────────────────────────────────────────

# NOTE on methodology:
# `DiffDiscreteEventSystem.step` cost grows with queue depth — it contains
# `for q in range(self.q)` loops (extern/QGym/main/env.py) and the `service_times`
# lists are Python lists that grow/shrink as jobs arrive and depart. A step from
# a congested state is therefore much more expensive than a step from a clean
# state. To get a steady-state per-step number we MUST reset the env before each
# timed step, and we must keep `env.reset()` OUTSIDE the timed window (it rebuilds
# the whole env state). The generic `_Timer` times the whole fn body, so it cannot
# exclude reset; we use the dedicated `_time_env` helper below instead.
#
# Additionally, the action matters: `allocator()` in env.py allocates servers via
# `int(torch.round(action))`. An action whose entries are < 0.5 rounds to 0, so no
# server is ever allocated and no job is ever served — queues only grow. A
# servicing action (entries >= 1.0 on at least one server per queue) lets the
# simulator drain, giving a realistic operating regime.


def _time_env(
    name: str,
    mode: str,
    env,
    action,
    *,
    reps: int,
    horizon: int = 1,
    warmup: int = 20,
) -> float:
    """Time env.step() with reset correctly excluded from the timed window.

    modes:
      - "steady":   reset before EACH step; time only the step. → ms/step from a
                    clean state. Apples-to-apples vs policy.forward / train step.
      - "horizon":  reset once, then step `horizon` times; repeat `reps` episodes.
                    → ms/step amortized over a full episode (realistic for training,
                      where one rollout = many consecutive steps, no reset).
      - "old_buggy": faithful reproduction of the PRE-FIX measurement. The old
                     code called _run_env_step(reps=N) from inside _Timer(reps=N),
                     so it timed N*N steps and divided by N — reporting "per-N-
                     steps" while labeling it "per-1-step" (a N× mislabel). With
                     N=200 that is the source of the old "~410 ms/step" number.
                     We print both the bogus old-label and the true per-step.

    Reset is always performed outside the perf_counter span.
    """
    print(f"  [{name}] mode={mode} warmup {warmup}x ...", end=" ", flush=True)

    if mode == "steady":
        for _ in range(warmup):
            env.reset()
            env.step(action)
        print(f"timing {reps}x (reset excluded) ...", end=" ", flush=True)
        t0 = _time.perf_counter()
        for _ in range(reps):
            env.reset()
            env.step(action)
        elapsed = _time.perf_counter() - t0
        ms_per = elapsed * 1000 / reps

    elif mode == "horizon":
        for _ in range(warmup):
            env.reset()
            for _ in range(horizon):
                env.step(action)
        print(f"timing {reps}x {horizon}-step episodes (reset excluded) ...",
              end=" ", flush=True)
        t0 = _time.perf_counter()
        for _ in range(reps):
            env.reset()
            for _ in range(horizon):
                env.step(action)
        elapsed = _time.perf_counter() - t0
        ms_per = elapsed * 1000 / (reps * horizon)

    elif mode == "old_buggy":
        # Faithful reproduction of the pre-fix measurement: the old code wrapped
        # _run_env_step(reps=N) inside _Timer(reps=N), so it timed N*N steps but
        # divided total by N → reported "per-N-steps" while labeling it per-step.
        # We expose the mislabel explicitly: return the bogus old-label value so
        # it can be compared to the true steady-state number.
        n_inner = horizon  # reuse horizon kwarg as the old reps-per-call (N)
        for _ in range(warmup):
            env.step(action)
        print(f"timing {reps}x outer x {n_inner}x inner "
              f"(={reps*n_inner} steps, ÷{reps} = old mislabel) ...",
              end=" ", flush=True)
        env.reset()
        t0 = _time.perf_counter()
        for _ in range(reps):
            for _ in range(n_inner):
                env.step(action)
        elapsed = _time.perf_counter() - t0
        ms_per = elapsed * 1000 / reps  # the BOGUS old per-(N-steps) label
        true_per_step = elapsed * 1000 / (reps * n_inner)
        print(f" done.  old label={ms_per:.3f} ms | true={true_per_step:.3f} ms/step")
        return ms_per

    else:
        raise ValueError(f"unknown mode {mode!r}")

    print(f" done.  avg {ms_per:.3f} ms/step")
    return ms_per


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


# ── VecEnv comparison ────────────────────────────────────────────────────────

def _compare_vec_envs(env_cfg, env_temp, policy_name, env_device_t, actors, action) -> None:
    """Time DummyVecEnv vs SubprocVecEnv stepping `actors` envs in tandem.

    Answers: does process-level parallelism beat the serial DummyVecEnv loop for
    THIS env? Builds envs via the same make_env pattern as train.py (batch=1,
    one seed per actor), runs a short rollout under each backend, and reports
    ms/timestep. Failures (e.g. spawn unsupported on the platform) are caught
    and reported, not fatal — this is a read-only diagnostic.
    """
    import copy
    import sys as _sys

    def _make_env(seed):
        # Deep-copy the cfg because load_rl_p_env MUTATES it in place (sets
        # network/mu to torch tensors), and each subprocess must get its own.
        cfg = copy.deepcopy(env_cfg)
        return load_rl_p_env(
            env_config=cfg, temp=env_temp, batch=1, seed=seed,
            policy_name=policy_name, device=env_device_t,
        )

    rollout_steps = 200
    # action is shape (s, q); DummyVecEnv/SubprocVecEnv expect a batch of
    # (actors, s, q) — broadcast.
    batch_action = np.broadcast_to(action, (actors,) + action.shape).copy()

    def _time_backend(name, build_fn, steps, warmup=20):
        print(f"\n  [{name}] building {actors} envs ...", end=" ", flush=True)
        try:
            env = build_fn()
        except Exception as e:  # noqa: BLE001 — diagnostic, must not abort profiler
            print(f"FAILED to build: {type(e).__name__}: {e}")
            return None
        try:
            print(f"stepping {steps}x ...", end=" ", flush=True)
            env.reset()
            for _ in range(warmup):
                env.step(batch_action)
            env.reset()
            t0 = _time.perf_counter()
            for _ in range(steps):
                env.step(batch_action)
            ms_per = (_time.perf_counter() - t0) * 1000 / steps
            print(f" done.  {ms_per:.2f} ms/timestep ({ms_per/actors:.3f} ms/env-step)")
            return ms_per
        except Exception as e:  # noqa: BLE001
            print(f"FAILED to step: {type(e).__name__}: {e}")
            return None
        finally:
            try:
                env.close()
            except Exception:
                pass

    print("\n--- DummyVecEnv vs SubprocVecEnv ---")
    print(f"  (rollout: {rollout_steps} timesteps x {actors} envs; "
          f"torch threads = {torch.get_num_threads()})")

    # Dummy: serial loop in this process.
    dummy_fns = [(lambda s=seed: _make_env(s)) for seed in range(3003, 3003 + actors)]
    t_dummy = _time_backend(
        f"DummyVecEnv (x{actors})",
        lambda: DummyVecEnv(dummy_fns),
        rollout_steps,
    )

    # Subproc: one process per env. spawn on win32 (matches train.py), fork elsewhere.
    # CUDA cannot be shared across forked processes, so skip when the env is on CUDA.
    if str(env_device_t) == "cuda":
        print(f"\n  [SubprocVecEnv] SKIPPED — CUDA cannot be forked.")
        t_subproc = None
    else:
        start_method = "spawn" if _sys.platform == "win32" else "fork"
        proc_fns = [(lambda s=seed: _make_env(s)) for seed in range(3003, 3003 + actors)]
        t_subproc = _time_backend(
            f"SubprocVecEnv (x{actors}, {start_method})",
            lambda: SubprocVecEnv(proc_fns, start_method=start_method),
            rollout_steps,
        )

    if t_dummy is not None and t_subproc is not None:
        if t_subproc < t_dummy:
            ratio = t_dummy / t_subproc
            print(f"  >> SubprocVecEnv WINS: {ratio:.2f}x faster "
                  f"({t_dummy:.2f} -> {t_subproc:.2f} ms/timestep). "
                  f"Parallelism is worth pursuing.")
        else:
            ratio = t_subproc / t_dummy
            print(f"  >> DummyVecEnv WINS: {ratio:.2f}x faster "
                  f"({t_dummy:.2f} vs {t_subproc:.2f} ms/timestep). "
                  f"IPC overhead exceeds the per-step cost — keep serial + optimize kernel.")
    elif t_dummy is not None:
        print("  >> SubprocVecEnv unavailable on this platform; only DummyVecEnv measured.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Profile bottleneck")
    parser.add_argument("policy_config", type=str)
    parser.add_argument("env_config", type=str)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. 'cuda', 'cpu')")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="Override torch.set_num_threads (default: torch default). "
                             "For the DummyVecEnv path with tiny per-step ops, 1 often "
                             "beats the default (avoids thread oversubscription).")
    args = parser.parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"[profile] torch.set_num_threads({args.num_threads})")
    print(f"[profile] torch.get_num_threads() = {torch.get_num_threads()}")

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

    # Servicing action: allocator() in env.py allocates servers via
    # int(torch.round(action)), so action entries must be >= 1.0 for at least one
    # server per queue to actually allocate a server and serve jobs. Allocate
    # server 0 fully to every queue → round(1.0)=1 server per non-empty queue,
    # so queues drain and the sim reaches a realistic operating regime.
    servicing_action = np.zeros((orig_s, orig_q), dtype=np.float32)
    servicing_action[0, :] = 1.0

    # Non-servicing action: every entry < 0.5 rounds to 0 in allocator(), so no
    # server is allocated and the per-queue allocation branches in step() are
    # skipped. NOTE: this is NOT what inflated the old 410 ms number (that was a
    # double-loop timing bug — see old_buggy mode below). Kept only because the
    # old code used it; it makes step() slightly cheaper, not pricier.
    congested_action = np.ones((orig_s, orig_q), dtype=np.float32) / orig_q

    # Policy action for training-component tests (needs valid graph)
    action = policy(single_obs.unsqueeze(0))[0][0].detach()  # (S, Q)
    batch_action = action.unsqueeze(0).expand(batch_size, -1, -1).contiguous()  # (B, S, Q)

    print(f"\nDevice: {device_str}   Env: {env_type}   Actors: {actors}   "
          f"Episode_steps: {episode_steps}   Batch_size: {batch_size}")
    print(f"Observation dim: {orig_q}  Servers: {orig_s}")
    print(f"Per-step workload: {actors} envs x (1 obs + 1 step) per timestep")
    print(f"{'-'*70}")

    timer = _Timer

    # ── 1. Environment stepping ─────────────────────────────────────────────
    # Steady-state: reset before each step, reset excluded from timing. This is
    # the trustworthy per-step number from a clean state.
    print("\n--- QGym Environment ---")
    t_env_single = _time_env(
        "env.step (steady-state)", mode="steady",
        env=dq, action=servicing_action, reps=200, warmup=20,
    )
    # Amortized over a full episode (no reset between steps) — realistic for
    # training, where one rollout = many consecutive steps.
    horizon = min(episode_steps, 1000)
    t_env_horizon = _time_env(
        "env.step (full episode)", mode="horizon",
        env=dq, action=servicing_action, reps=20, horizon=horizon, warmup=2,
    )
    # Diagnostic: faithfully reproduce the PRE-FIX (buggy) measurement. The old
    # code wrapped _run_env_step(reps=N) inside _Timer(reps=N), so it timed N*N
    # steps and divided by N — reporting "per-N-steps" while labeling it per-step.
    # With N=200 this is the source of the old "~410 ms/step" figure. We pass the
    # non-servicing action because that's what the old code used; the action itself
    # is NOT the cause (it just skips the allocation branches), the double-loop is.
    t_env_old_label = _time_env(
        "env.step (old buggy method)", mode="old_buggy",
        env=dq, action=congested_action, reps=1, horizon=200, warmup=0,
    )
    print(f"  >> the old '410 ms/step' figure was a {200}x mislabel "
          f"(timed {200*200} steps, divided by 200). "
          f"True per-step ≈ {t_env_old_label/200:.3f} ms — consistent with the "
          f"steady-state {t_env_single:.3f} ms measurement above.")

    # ── 1b. DummyVecEnv vs SubprocVecEnv comparison ─────────────────────────
    # Answer the parallelism question empirically: does stepping `actors` envs
    # across processes beat the serial DummyVecEnv loop for THIS env? The generic
    # wisdom (SubprocVecEnv for heavy envs, DummyVecEnv for light ones) is
    # inconclusive for a ~1ms env, so we measure. This is read-only — it does
    # not change training behavior.
    _compare_vec_envs(env_cfg, env_temp, policy_name, env_device_t, actors, servicing_action)

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
    # Use the amortized-over-episode per-step cost (t_env_horizon), since a
    # rollout is many consecutive steps with no reset. t_env_single (steady-state)
    # is a lower bound; the truth is between them.
    fwd_per_timestep = t_forward_actors  # ms for 1 policy forward over `actors` obs
    step_per_timestep = t_env_horizon * actors  # ms for stepping all `actors` envs
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
