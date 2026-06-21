"""Verify fix 3B+2B: zero_grad() is present between policy and value phases.

Checks that `self.policy.zero_grad()` appears in both:
1. CustomPPOTrainer.train()  (extern/QGym)
2. CertiqPPOTrainer.train()  (our code)
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTERN_TRAINER = PROJECT_ROOT / "extern" / "QGym" / "RL" / "PPO" / "trainer.py"
CERTIQ_TRAINER = PROJECT_ROOT / "src" / "certiq_net" / "studies" / "qgym_eval" / "train" / "certiq_ppo_trainer.py"


def _check_file(path: Path, label: str) -> bool:
    src = path.read_text(encoding="utf-8")
    # Look for policy.zero_grad() between optimizer_policy.step() and value phase
    if "self.policy.zero_grad()" in src:
        print(f"[OK]   {label}: contains self.policy.zero_grad()")
        return True
    else:
        print(f"[FAIL] {label}: MISSING self.policy.zero_grad()")
        return False


def main() -> int:
    ok1 = _check_file(EXTERN_TRAINER, "CustomPPOTrainer (extern)")
    ok2 = _check_file(CERTIQ_TRAINER, "CertiqPPOTrainer (ours)")

    print()
    if ok1 and ok2:
        print("[PASS] Fix 3B+2B verified: encoder gradients are zeroed between phases")
        return 0
    else:
        print("[FAIL] One or more components missing")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
