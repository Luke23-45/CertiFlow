"""Isolated test for BatchedEnv reward scaling.

Verifies that rewards from BatchedEnv.step() are properly scaled
(O(10) instead of O(10^4)) to prevent value gradient explosion
through the shared encoder during PPO training.
"""

import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml

# Apply QGym patches BEFORE any import from QGym
import certiq_net

from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied

ensure_patches_applied()

_qgym_root = Path(certiq_net.__file__).resolve().parents[2] / "extern" / "QGym"
if str(_qgym_root) not in sys.path:
    sys.path.insert(0, str(_qgym_root))

from main.env import BatchedEnv


def _create_batched_env(env_type="reentrant_2", batch=4):
    """Helper: create a BatchedEnv for the given env type."""
    data_dir = _qgym_root / "configs" / "env_data" / env_type

    with open(_qgym_root / "configs" / "env" / f"{env_type}.yaml") as f:
        env_cfg = yaml.safe_load(f)

    network = torch.tensor(np.load(data_dir / f"{env_type}_network.npy")).float()
    mu = torch.tensor(np.load(data_dir / f"{env_type}_mu.npy")).float()
    h = torch.tensor(env_cfg["h"])
    queue_event_options = torch.tensor(np.load(data_dir / f"{env_type}_delta.npy"))
    lam_r = np.load(data_dir / f"{env_type}_lam.npy")
    orig_q = network.shape[1]

    def draw_inter_arrivals(self, time):
        exps = np.random.exponential(1, (self.B, orig_q))
        return torch.tensor(exps / lam_r, dtype=torch.float32, device=self.device)

    def draw_service(self, time):
        service = np.random.exponential(1, (self.B, orig_q))
        return torch.tensor(service, dtype=torch.float32, device=self.device)

    return BatchedEnv(
        network=network, mu=mu, h=h,
        draw_service=draw_service,
        draw_inter_arrivals=draw_inter_arrivals,
        batch=batch, temp=1.0, seed=3003,
        device="cpu",
        queue_event_options=queue_event_options,
    )


def test_batched_env_reward_many_steps():
    """Run BatchedEnv for many steps and verify rewards stay O(10) not O(10^4).

    Over hundreds of steps queues build up, making cost = event_time * (queues @ h)
    grow to O(10^4).  If BatchedEnv does not divide by 1000, per-step rewards
    become O(10^4), causing value-loss explosion in PPO.
    """
    env = _create_batched_env(batch=2)
    env.reset()

    all_rewards = []
    for _ in range(500):
        actions = torch.rand(env.B, env.s, env.q) * env.network
        obs, reward, done, infos = env.step(actions.numpy())
        all_rewards.extend(reward)

    rewards = np.abs(np.array(all_rewards))
    mean_abs = rewards.mean()
    max_abs = rewards.max()

    print(f"rewards (500 steps): |mean|={mean_abs:.2f}, |max|={max_abs:.2f}")
    print(f"samples: {rewards[:12].tolist()}")

    # If BatchedEnv doesn't /1000, mean|reward| will be O(10^4).
    # With proper scaling, it should be O(10).
    assert mean_abs < 1e4, (
        f"Rewards too large (|mean|={mean_abs:.2f}). "
        f"BatchedEnv reward likely not divided by 1000."
    )
    print("PASS: rewards stay O(10) over 500 steps")


def test_batched_env_large_queue_reward():
    """Set queues to large values (simulating late-episode buildup) and step.

    After many steps, queues @ h can be O(10^4).  Cost = event_time * sum(queues)
    can be O(10^4) per step.  This test verifies the per-step reward magnitude
    even when queues are large, which is the regime that causes PPO collapse.
    """
    env = _create_batched_env(batch=2)
    env.reset()
    h_sum = env.h.sum().item()

    # Set queues to large values simulating many steps of buildup
    large_queues = torch.ones(env.B, env.q, device=env.device) * 5000
    env.queues = large_queues

    actions = torch.rand(env.B, env.s, env.q) * env.network
    obs, reward, done, infos = env.step(actions.numpy())

    cost_per_step = env.h.sum().item() * 5000  # ~30000 per step
    expected_unscaled = cost_per_step  # would be O(10^4)
    expected_scaled = cost_per_step / 1000.0  # would be O(10)

    print(f"h sum: {h_sum}, large_queue: 5000, expected cost/step: ~{cost_per_step}")
    print(f"reward: {reward}")
    print(f"expected (unscaled): ~{expected_unscaled}, expected (scaled): ~{expected_scaled}")

    # Verify reward is properly scaled (O(10) not O(10^4))
    mean_abs = np.mean(np.abs(reward))
    assert mean_abs < 1e3, (
        f"Reward too large (|mean|={mean_abs:.1f}). "
        f"Expected O(10) after /1000 scaling, got O(10^4). "
        f"Patch 0020 not applied."
    )
    assert mean_abs > 1.0, (
        f"Reward too small (|mean|={mean_abs:.1f}). "
        f"Cost should produce non-trivial rewards with large queues."
    )
    print("PASS: large-queue reward properly scaled")
