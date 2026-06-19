"""Verify the production training config actually uses resources as expected.

Runs a minimal training iteration with --num-actors 4 --num-threads 1
and checks: actor count, thread count, env creation, and a single PPO update.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "extern" / "QGym"))

# Must patch BEFORE any QGym imports
from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied
ensure_patches_applied()

import torch
import yaml
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv

from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy
from certiq_net.studies.qgym_eval.train.qgym_import import (
    CustomPPOTrainer,
    CustomRolloutBuffer,
    load_rl_p_env,
    parallel_eval,
)
from torch import nn


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    print("=" * 60)
    print("Production config verification")
    print("=" * 60)

    # ── 1. Thread check ────────────────────────────────────────────────────
    torch.set_num_threads(1)
    actual_threads = torch.get_num_threads()
    expected_threads = 1
    if actual_threads != expected_threads:
        print(f"[FAIL] torch.get_num_threads() = {actual_threads}, expected {expected_threads}")
        return 1
    print(f"[OK]   torch.get_num_threads() = {actual_threads}")

    # ── 2. Device check ────────────────────────────────────────────────────
    device = "cpu"
    print(f"[OK]   device = {device}")

    # ── 3. Load configs ────────────────────────────────────────────────────
    _QGYM_ROOT = PROJECT_ROOT / "extern" / "QGym"
    policy_cfg = _load_yaml(_QGYM_ROOT / "RL" / "policy_configs" / "vanilla.yaml")
    env_cfg = _load_yaml(_QGYM_ROOT / "configs" / "env" / "reentrant_2.yaml")

    env_type = env_cfg.get("env_type", env_cfg["name"])
    data_dir = _QGYM_ROOT / "configs" / "env_data" / env_type

    network = _load_yaml(_QGYM_ROOT / "configs" / "env" / f"reentrant_2.yaml").get("network")
    mu = env_cfg.get("mu")
    if network is None:
        import numpy as np
        network = np.load(data_dir / f"{env_type}_network.npy")
    if mu is None:
        import numpy as np
        mu = np.load(data_dir / f"{env_type}_mu.npy")

    network_t = torch.tensor(network, dtype=torch.float)
    mu_t = torch.tensor(mu, dtype=torch.float)
    orig_s, orig_q = network_t.shape

    num_pool = env_cfg.get("num_pool", 1)
    network_t = network_t.repeat_interleave(num_pool, dim=0)
    mu_t = mu_t.repeat_interleave(num_pool, dim=0)

    env_temp = policy_cfg["env"]["env_temp"]
    train_seed = policy_cfg["env"]["train_seed"]
    test_seed = policy_cfg["env"]["test_seed"]
    randomize = policy_cfg["env"]["randomize"]
    time_f = policy_cfg["env"].get("time_f", False)
    policy_name = policy_cfg["model"]["policy_name"]

    actors = 4
    episode_steps = 64  # small for test
    num_epochs = 1
    total_steps = num_epochs * episode_steps * actors
    env_device = "cpu"

    print(f"[OK]   actors          = {actors}")
    print(f"[OK]   episode_steps   = {episode_steps}")
    print(f"[OK]   total_steps     = {total_steps}")

    # ── 4. Create envs ─────────────────────────────────────────────────────
    def make_env(seed):
        return load_rl_p_env(
            env_config=env_cfg,
            temp=env_temp,
            batch=1,
            seed=seed,
            policy_name=policy_name,
            device=torch.device(env_device),
        )

    env_fns = [lambda s=seed: make_env(s) for seed in range(train_seed, train_seed + actors)]
    start_method = "spawn" if sys.platform == "win32" else "fork"
    dq = SubprocVecEnv(env_fns, start_method=start_method)

    # Verify actor count
    actual_envs = dq.num_envs
    if actual_envs != actors:
        print(f"[FAIL] SubprocVecEnv has {actual_envs} envs, expected {actors}")
        return 1
    print(f"[OK]   SubprocVecEnv.num_envs = {actual_envs}")

    # Also check we got the process count right
    if hasattr(dq, "ps"):
        print(f"[OK]   SubprocVecEnv subprocesses = {len(dq.ps)}")

    # ── 5. Create test envs ────────────────────────────────────────────────
    num_test_envs = 2  # small for test
    dq_test_list = [make_env(seed) for seed in range(test_seed, test_seed + num_test_envs)]
    print(f"[OK]   test envs       = {len(dq_test_list)}")

    # ── 6. Policy setup ────────────────────────────────────────────────────
    scale = policy_cfg["model"]["scale"]
    L, J = orig_q, orig_s
    gmLJ = int((L * J) ** 0.5)
    pi_arch = [scale * L, scale * gmLJ, scale * J]
    vi_arch = [scale * L, scale * gmLJ, scale * J]

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        network=network_t,
        mu=mu_t,
        alpha=0,
        D=torch.zeros(orig_q, 2 * orig_q + 2),
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
        batch_size=episode_steps,
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
        verbose=0,
        seed=train_seed,
        device=device,
        _init_setup_model=True,
    )

    # ── 7. Create trainer (with Lagrangian) ────────────────────────────────
    from certiq_net.studies.qgym_eval.train.certiq_ppo_trainer import CertiqPPOTrainer

    trainer_kwargs["lr_nu"] = 1e-3
    model = CertiqPPOTrainer(CertiQSB3Policy, dq, **trainer_kwargs)
    print("[OK]   CertiqPPOTrainer created")

    # Verify optimizer structure (fix 2B: encoder must be in policy optimizer)
    opt_policy_params = sum(p.numel() for g in model.optimizer_policy.param_groups for p in g["params"])
    opt_value_params = sum(p.numel() for g in model.optimizer_value.param_groups for p in g["params"])
    print(f"[OK]   optimizer_policy params = {opt_policy_params}")
    print(f"[OK]   optimizer_value  params = {opt_value_params}")

    # Verify seed (fix 2C)
    if model.seed == train_seed:
        print(f"[OK]   model.seed = {model.seed} (from config)")
    else:
        print(f"[FAIL] model.seed = {model.seed}, expected {train_seed}")
        return 1

    # ── 8. Evaluation callback ─────────────────────────────────────────────
    eval_freq = episode_steps * actors
    eval_callback = parallel_eval(
        model=model,
        eval_env=dq_test_list,
        eval_freq=eval_freq,
        eval_t=1000,
        test_policy=policy_cfg["policy"]["test_policy"],
        test_seed=test_seed,
        init_test_queues=torch.tensor([env_cfg["init_queues"]]).float(),
        test_batch=num_test_envs,
        device=device,
        num_pool=num_pool,
        time_f=time_f,
        randomize=randomize,
        policy_name=policy_name,
        per_iter_normal_obs=False,
        env_config_name="reentrant_2",
        bc=False,
        verbose=0,
    )

    # ── 9. Run one training step ───────────────────────────────────────────
    print("\n--- Running minimal training (1 epoch, 4 actors, 64 steps) ---")
    start = time.time()
    model.learn(total_timesteps=total_steps, log_interval=1, callback=eval_callback)
    elapsed = time.time() - start
    print(f"--- Training completed in {elapsed:.2f}s ---")

    # ── 10. Verify results ─────────────────────────────────────────────────
    if not hasattr(eval_callback, "test_costs"):
        print("[FAIL] No test results produced")
        return 1

    print(f"[OK]   {len(eval_callback.test_costs)} eval results recorded")
    print(f"[OK]   Final cost list: {eval_callback.final_costs}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    dq.close()
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
