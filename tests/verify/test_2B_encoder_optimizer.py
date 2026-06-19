"""Verify finding 2B: encoder is NOT in value optimizer.

Checks that optimizer_value does not contain marginal_index_head.encoder
parameters. Requires QGym patches to be applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure the QGym module is importable
_QGYM_ROOT = PROJECT_ROOT / "extern" / "QGym"
if str(_QGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_QGYM_ROOT))

# Apply patches first
from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied
ensure_patches_applied()

from certiq_net.studies.qgym_eval.train.certiq_sb3_policy import CertiQSB3Policy
import torch


def main() -> int:
    # Create a minimal policy with real gymnasium spaces
    import gymnasium.spaces
    import numpy as np
    obs_space = gymnasium.spaces.Box(low=0, high=1e6, shape=(6,), dtype=np.float32)
    act_space = gymnasium.spaces.Box(low=0, high=1, shape=(2, 6), dtype=np.float32)

    network = torch.tensor([[1, 1, 1, 1, 1, 1],
                            [0, 1, 0, 0, 0, 0]], dtype=torch.float)
    mu = torch.tensor([[0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
                       [0, 0.1, 0, 0, 0, 0]], dtype=torch.float)
    D = torch.zeros(6, 12)  # queue_event_options placeholder

    policy = CertiQSB3Policy(
        obs_space, act_space, lambda _: 0.0,
        network=network, mu=mu, alpha=0, D=D, tau=1.0, scale=20,
    )

    # Build the optimizers the same way QGym does (trainer.py lines 82-93)
    optimizer_policy = torch.optim.Adam([
        {'params': policy.features_extractor.parameters()},
        {'params': policy.pi_features_extractor.parameters()},
        {'params': policy.mlp_extractor.policy_net.parameters()},
        {'params': policy.action_net.parameters()},
    ], lr=0.0003)

    optimizer_value = torch.optim.Adam([
        {'params': policy.vf_features_extractor.parameters()},
        {'params': policy.mlp_extractor.value_net.parameters()},
        {'params': policy.value_net.parameters()},
    ], lr=0.0003)

    # Collect param id sets
    policy_params = set()
    for group in optimizer_policy.param_groups:
        for p in group['params']:
            policy_params.add(id(p))

    value_params = set()
    for group in optimizer_value.param_groups:
        for p in group['params']:
            value_params.add(id(p))

    all_params = set(id(p) for p in policy.parameters())
    missing = all_params - (policy_params | value_params)
    encoder_in_value = set()
    encoder_in_policy = set()

    encoder_params = set(id(p) for p in policy.marginal_index_head.encoder.parameters())

    for pid in encoder_params:
        if pid in value_params:
            encoder_in_value.add(pid)
        if pid in policy_params:
            encoder_in_policy.add(pid)

    print(f"Total policy parameters:        {len(all_params)}")
    print(f"In optimizer_policy:            {len(policy_params)}")
    print(f"In optimizer_value:             {len(value_params)}")
    print(f"Missing from both optimizers:   {len(missing)}")
    print(f"Encoder params in policy opt:   {len(encoder_in_policy)}")
    print(f"Encoder params in value opt:    {len(encoder_in_value)}")

    try:
        assert len(missing) == 0, f"Some params are not optimized! ({len(missing)} missing)"
        assert len(encoder_in_value) == 0, (
            f"Encoder IS in value optimizer ({len(encoder_in_value)} params)!"
        )
        assert len(encoder_in_policy) > 0, (
            f"Encoder NOT in policy optimizer!"
        )

        print(f"\n[PASS] Finding 2B confirmed: encoder is ONLY in optimizer_policy")
        print("       Value-loss gradients to encoder are computed but discarded")
        print("       by the next optimizer_policy.zero_grad() call.")
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
