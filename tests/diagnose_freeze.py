"""Focused diagnostic: WHY does the policy freeze (ratio -> 1) at iter 4?

Hypothesis: the Lagrangian penalty (nu growing each iter) pushes the policy
toward a degenerate deterministic distribution. Once pi is near-deterministic,
the sampled one-hot action is (nearly) always the argmax, so when
evaluate_actions recomputes log_prob on the SAME action, the policy hasn't
moved, ratio -> 1, surrogate gradient -> 0, policy freezes.

This script reproduces that mechanism in isolation: train the policy with a
growing Lagrangian-like penalty and watch the action distribution + ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import certiq_net
from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied

ensure_patches_applied()

_qgym_root = Path(certiq_net.__file__).resolve().parents[2] / "extern" / "QGym"
if str(_qgym_root) not in sys.path:
    sys.path.insert(0, str(_qgym_root))

from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy


def make_policy(device):
    from gymnasium import spaces
    data_dir = _qgym_root / "configs" / "env_data" / "reentrant_2"
    network = torch.tensor(np.load(data_dir / "reentrant_2_network.npy")).float()
    mu = torch.tensor(np.load(data_dir / "reentrant_2_mu.npy")).float()
    delta = torch.tensor(np.load(data_dir / "reentrant_2_delta.npy")).float()
    obs_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
    act_space = spaces.Box(low=0, high=1, shape=(2, 6), dtype=np.float32)
    scale = 20
    L, J = 6, 2
    gmLJ = int(np.sqrt(L * J))
    policy = CertiQSB3Policy(
        observation_space=obs_space, action_space=act_space,
        lr_schedule=lambda _: 3e-4, network=network, mu=mu, alpha=0, D=delta,
        tau=1.0, randomize=True, scale=scale, rescale_v=True, time_f=False,
        net_arch=dict(pi=[scale*L, scale*gmLJ, scale*J], vf=[scale*L, scale*gmLJ, scale*J]),
    )
    return policy.to(device)


def main():
    torch.manual_seed(3003)
    np.random.seed(3003)
    device = torch.device("cpu")
    policy = make_policy(device)

    # Simulate the Lagrangian ratchet: each "iteration" nu grows, and the
    # Lagrangian penalty nu*violation pushes the policy to minimize violation
    # = put mass on the min-cost queue.
    opt = torch.optim.Adam(policy.parameters(), lr=9e-4)

    B = 64
    obs = torch.rand(B, 6, device=device) * 20.0

    print("iter | nu     | max_prob | entropy | ratio_after_1_step")
    print("-----|--------|----------|---------|--------------------")
    for it in range(8):
        # Collect old actions + old log probs (the rollout)
        with torch.no_grad():
            action_old, _, log_prob_old = policy(obs)
        with torch.no_grad():
            pi_old, _, _ = policy._run_certiq(obs)
        max_prob_old = pi_old.max(dim=-1).values.mean().item()
        ent_old = -(pi_old * pi_old.clamp_min(1e-9).log()).sum(dim=-1).mean().item()

        # nu ratchets up like in the real run: 0 -> 0.0045 -> 0.015 -> 0.029 -> ...
        nu = 0.0045 * (it + 1) ** 1.5

        # One PPO epoch with the Lagrangian penalty
        log_prob_new, _ = policy.evaluate_actions(obs, action_old)
        ratio = torch.exp(log_prob_new - log_prob_old)
        ratio_after = ratio.mean().item()

        # Surrogate
        advantages = torch.randn(B, device=device)
        pg1 = advantages * ratio
        pg2 = advantages * torch.clamp(ratio, 0.8, 1.2)
        pg_loss = -torch.min(pg1, pg2).mean()

        # Lagrangian
        cost, budget, pi = policy.compute_cost_and_budget(obs)
        a_final = (pi * cost).sum(dim=-1)
        violation = (a_final - budget).clamp(min=0.0)
        lag_loss = nu * violation.mean() / 1000.0

        loss = pg_loss + lag_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()

        # Check distribution after the step
        with torch.no_grad():
            pi_new, _, _ = policy._run_certiq(obs)
        max_prob_new = pi_new.max(dim=-1).values.mean().item()
        ent_new = -(pi_new * pi_new.clamp_min(1e-9).log()).sum(dim=-1).mean().item()

        print(f"  {it+1}  | {nu:.4f} | {max_prob_old:.3f}    | {ent_old:.3f}   | {ratio_after:.4f}  "
              f"-> max_prob={max_prob_new:.3f} ent={ent_new:.3f}")

    print()
    print("=== Now check: if policy is near-deterministic, does ratio -> 1? ===")
    with torch.no_grad():
        action_old, _, log_prob_old = policy(obs)
        # Force a near-deterministic policy by clamping
        pi, _, _ = policy._run_certiq(obs)
        print(f"current max_prob: {pi.max(dim=-1).values.mean():.4f}")
        log_prob_new, _ = policy.evaluate_actions(obs, action_old)
        ratio = torch.exp(log_prob_new - log_prob_old)
        print(f"ratio mean: {ratio.mean():.6f}  std: {ratio.std():.6f}")
        print(f"  -> if ratio≈1, the surrogate gradient is ~0 and policy is FROZEN")


if __name__ == "__main__":
    main()
