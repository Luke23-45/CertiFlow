"""Diagnostic: numerically confirm root-cause hypotheses for the exploding-queue
divergence in the CertiQ PPO pipeline.

This is NOT a unit test.  It is a focused, cheap (CPU, <2 min) experiment that
reproduces the failure mode observed in docs/imp/logs.md and prints the
quantities needed to decide on patches:

  * policy entropy / action-prob spread  -> is the policy collapsing?
  * ratio distribution after 1 PPO epoch -> is the gradient vanishing (ratio=1)?
  * returns std vs. value head magnitude  -> is the critic target too big?
  * reward magnitude as queues grow      -> is there a positive feedback loop?
  * cost / budget / violation for the Lagrangian -> is nu chasing a moving target?

Run::

    python -m tests.diagnose_divergence
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import certiq_net
from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied

# Patches must be applied before any QGym import (module caching).
ensure_patches_applied()

_qgym_root = Path(certiq_net.__file__).resolve().parents[2] / "extern" / "QGym"
if str(_qgym_root) not in sys.path:
    sys.path.insert(0, str(_qgym_root))

from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy
from certiq_net.studies.qgym_eval.train.qgym_import import (
    CustomRolloutBuffer,
    load_rl_p_env,
)

ENV_TYPE = "reentrant_2"


def _load_env_cfg():
    with open(_qgym_root / "configs" / "env" / f"{ENV_TYPE}.yaml") as f:
        return yaml.safe_load(f)


def make_policy(device) -> CertiQSB3Policy:
    from gymnasium import spaces

    env_cfg = _load_env_cfg()
    data_dir = _qgym_root / "configs" / "env_data" / ENV_TYPE
    network = torch.tensor(np.load(data_dir / f"{ENV_TYPE}_network.npy")).float()
    mu = torch.tensor(np.load(data_dir / f"{ENV_TYPE}_mu.npy")).float()
    delta = torch.tensor(np.load(data_dir / f"{ENV_TYPE}_delta.npy")).float()
    orig_s, orig_q = network.shape

    obs_space = spaces.Box(low=0, high=np.inf, shape=(orig_q,), dtype=np.float32)
    act_space = spaces.Box(low=0, high=1, shape=(orig_s, orig_q), dtype=np.float32)

    scale = 20
    L, J = orig_q, orig_s
    gmLJ = int(np.sqrt(L * J))
    pi_arch = [scale * L, scale * gmLJ, scale * J]
    vi_arch = [scale * L, scale * gmLJ, scale * J]

    policy = CertiQSB3Policy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        network=network,
        mu=mu,
        alpha=0,
        D=delta,
        tau=1.0,
        randomize=True,
        scale=scale,
        rescale_v=True,
        time_f=False,
        net_arch=dict(pi=pi_arch, vf=vi_arch),
    )
    policy = policy.to(device)
    return policy


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    torch.manual_seed(3003)
    np.random.seed(3003)
    device = torch.device("cpu")

    policy = make_policy(device)

    # ── H1: does the policy collapse after a few gradient steps? ────────────
    banner("H1: policy collapse / entropy after repeated gradient steps")
    # Simulate many PPO-style updates that all push the policy the same way
    # (mimics the Lagrangian penalty + advantage both favoring one queue).
    obs = torch.tensor([[10.0, 2.0, 5.0, 8.0, 3.0, 1.0]], device=device)

    def entropy_of_probs(o):
        with torch.no_grad():
            pi, _, _ = policy._run_certiq(o)
        ent = -(pi * pi.clamp_min(1e-9).log()).sum(dim=-1)
        max_prob = pi.max(dim=-1).values
        return ent.item(), max_prob.item(), pi.detach().cpu().numpy().ravel()

    ent0, mp0, pi0 = entropy_of_probs(obs)
    print(f"  init entropy={ent0:.4f}  max_prob={mp0:.4f}")
    print(f"  init pi={np.round(pi0, 4)}")

    # 50 gradient steps of a degenerate surrogate that pushes one queue up,
    # exactly like nu*violation repeatedly penalising the same argmax.
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    target_queue = 0  # push mass onto queue 0
    for i in range(50):
        opt.zero_grad()
        pi, _, _ = policy._run_certiq(obs)
        # surrogate: minimize mass on target (push it off) like advantage pulling
        loss = pi[:, target_queue].mean()
        loss.backward()
        opt.step()
    ent1, mp1, pi1 = entropy_of_probs(obs)
    print(f"  after 50 uni-directional updates: entropy={ent1:.4f}  max_prob={mp1:.4f}")
    print(f"  pi={np.round(pi1, 4)}")
    collapsed = mp1 > 0.95
    print(f"  -> POLICY COLLAPSED (max_prob>0.95): {collapsed}")

    # ── H2: ratio=1 -> zero gradient.  Demonstrate with a fresh policy. ────
    banner("H2: ratio distribution & policy_gradient_loss with mini-batch reuse")
    policy2 = make_policy(device)
    B = 64
    obs_b = torch.rand(B, 6, device=device) * 30.0
    # Collect old actions + old log probs
    with torch.no_grad():
        action, value, log_prob = policy2(obs_b)
    # Now do ONE epoch of PPO update on the SAME batch repeatedly
    opt2 = torch.optim.Adam(policy2.parameters(), lr=9e-4)
    clip_range = 0.2
    advantages = torch.randn(B, device=device)  # normalized advantages
    for epoch in range(6):  # 6 ppo epochs over the same buffer
        log_prob_new, _ = policy2.evaluate_actions(obs_b, action)
        ratio = torch.exp(log_prob_new - log_prob)
        pg1 = advantages * ratio
        pg2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(pg1, pg2).mean()
        clip_frac = ((ratio - 1).abs() > clip_range).float().mean().item()
        print(
            f"  epoch {epoch}: ratio mean={ratio.mean():.4f} std={ratio.std():.4f} "
            f"min={ratio.min():.4f} max={ratio.max():.4f} "
            f"clip_frac={clip_frac:.3f} pg_loss={policy_loss.item():.2e}"
        )
        opt2.zero_grad()
        policy_loss.backward()
        opt2.step()

    # ── H3: reward / cost growth as queues grow (positive feedback loop) ────
    banner("H3: per-step reward magnitude vs. queue length (feedback loop)")
    h = torch.ones(6)
    print("  queue_sum | event_time | cost | reward(-cost/1000)")
    for q_scale in [5, 20, 50, 100, 200, 500]:
        queues = torch.full((6,), float(q_scale))
        # event_time ~ exp(1)/mu_eff for the active queue (rough)
        event_time = 0.5  # representative
        cost = event_time * (queues @ h)
        reward = -cost / 1000.0
        print(
            f"  {q_scale:9.0f} | {event_time:10.2f} | {cost.item():10.1f} | "
            f"{reward.item():.3f}"
        )
    print("  => as policy worsens, queues grow, |reward| grows, returns std grows.")

    # ── H4: returns std vs value head output (critic target scale) ──────────
    banner("H4: returns variance vs. value head magnitude (critic mismatch)")
    env_cfg = _load_env_cfg()
    dq = load_rl_p_env(
        env_config=env_cfg, temp=1.0, batch=1, seed=3003,
        policy_name="vanilla", device=device,
    )
    # Roll out with the RANDOM (untrained) policy and look at the raw returns.
    rewards = []
    obs_np, _ = dq.reset()
    for _ in range(2000):
        with torch.no_grad():
            action, _, _ = policy2(torch.as_tensor(obs_np, dtype=torch.float32).view(1, -1))
        obs_np, reward, done, trunc, info = dq.step(action.numpy())
        rewards.append(float(reward))
    rewards = np.array(rewards)
    # GAE returns have std ~ sum of discounted rewards; rough proxy:
    gamma = 0.998
    rets = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rets[t] = running
    print(f"  reward   mean={rewards.mean():.3f}  std={rewards.std():.3f}  "
          f"min={rewards.min():.3f} max={rewards.max():.3f}")
    print(f"  returns  mean={rets.mean():.3f}  std={rets.std():.3f}  "
          f"min={rets.min():.3f} max={rets.max():.3f}")
    with torch.no_grad():
        _, val, _ = policy2._run_certiq(torch.as_tensor(obs_np, dtype=torch.float32).view(1, -1))
    print(f"  value head output magnitude: ~{val.abs().mean().item():.3f}")
    print(f"  -> returns std / |value| ratio: {rets.std() / max(abs(val.abs().mean().item()), 1e-3):.1f}x")
    print("     (a large ratio means the MSE critic target is huge -> value_loss explodes)")

    # ── H5: Lagrangian cost / budget / violation scale ──────────────────────
    banner("H5: Lagrangian cost/budget/violation magnitude at observed queues")
    for qv in [5.0, 20.0, 50.0, 100.0]:
        obs_q = torch.full((1, 6), qv, device=device)
        cost, budget, pi = policy2.compute_cost_and_budget(obs_q)
        a_final = (pi * cost).sum(dim=-1)
        excess = a_final - budget
        violation = excess.clamp(min=0.0)
        print(
            f"  q={qv:6.1f}: cost={cost.cpu().numpy().round(2)} "
            f"budget={budget.item():.2f} a_final={a_final.item():.2f} "
            f"violation={violation.item():.2f}"
        )
    print("  (budget grows with queue; nu/LAGR_SCALE penalty ~ violation*nu/1000)")

    print("\n[DONE] diagnostic complete.")


if __name__ == "__main__":
    main()
