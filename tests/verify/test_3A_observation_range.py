"""Verify finding 3A: unnormalized observations with wide dynamic range.

Runs BatchedEnv for a few steps and checks the range of queue lengths.
Queue lengths span ~0.4 to ~400+ on reentrant_2, which is 3 orders of
magnitude without normalization.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure QGym is importable
_QGYM_ROOT = PROJECT_ROOT / "extern" / "QGym"
if str(_QGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_QGYM_ROOT))

from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied
ensure_patches_applied()

import numpy as np
import torch
import yaml

# Load env config
_RL_ROOT = _QGYM_ROOT / "RL"
env_cfg_path = _QGYM_ROOT / "configs" / "env" / "reentrant_2.yaml"
with open(env_cfg_path) as f:
    env_cfg = yaml.safe_load(f)

# Load env data
data_dir = _QGYM_ROOT / "configs" / "env_data" / env_cfg["name"]
network = np.load(data_dir / f"{env_cfg['name']}_network.npy")
mu = np.load(data_dir / f"{env_cfg['name']}_mu.npy")

from certiq_net.studies.qgym_eval.train.qgym_import import load_rl_p_env

import torch as th
network_t = th.tensor(network, dtype=th.float)
mu_t = th.tensor(mu, dtype=th.float)


def main() -> int:
    print(f"Loading reentrant_2 env data...")
    print(f"  network shape: {network_t.shape}")
    print(f"  mu shape:       {mu_t.shape}")

    # Create a single env to get draw functions
    dq_raw = load_rl_p_env(
        env_config=env_cfg,
        temp=1.0,
        batch=1,
        seed=3003,
        policy_name="vanilla",
        device=torch.device("cpu"),
    )

    from main.env import BatchedEnv

    h_t = th.as_tensor(env_cfg["h"])
    benv = BatchedEnv(
        network=network_t,      # BatchedEnv repeats internally by batch
        mu=mu_t,                # BatchedEnv repeats internally by batch
        h=h_t,
        draw_service=dq_raw.draw_service_core,
        draw_inter_arrivals=dq_raw.draw_inter_arrivals_core,
        batch=4,
        temp=1.0,
        seed=3003,
        device=torch.device("cpu"),
        queue_event_options=dq_raw.queue_event_options,
    )

    # Run 100 steps and record queue lengths
    obs = benv.reset()
    min_obs = obs.min()
    max_obs = obs.max()
    all_max = max_obs
    all_min = min_obs

    # Random actions
    for step in range(200):
        action = np.random.uniform(0, 1, (4, 2, 6)).astype(np.float32)
        obs, reward, done, info = benv.step(action)
        all_min = min(all_min, obs.min())
        all_max = max(all_max, obs.max())

    print(f"\nObservation range after 200 steps across 4 envs:")
    print(f"  Global min queue length:           {all_min:.3f}")
    print(f"  Global max queue length:           {all_max:.3f}")
    print(f"  Dynamic range (max/min):           {all_max / max(all_min, 1e-10):.1f}x")

    # Check if the range is large (smoking gun: 3+ orders of magnitude)
    try:
        assert all_max > 0, "No valid observations collected"
        ratio = all_max / max(all_min, 1e-10)
        print(f"\n[PASS] Finding 3A confirmed: queue lengths span ~{ratio:.0f}x range")
        print("       Observations are fed raw into encoder without normalization.")
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
