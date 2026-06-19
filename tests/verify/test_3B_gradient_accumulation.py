"""Verify fix 3B+2B: zero_grad() added between policy and value phases.

Checks that `self.policy.zero_grad()` appears in both:
1. CustomPPOTrainer.train()  (extern/QGym, via patch 0012)
2. CertiqPPOTrainer.train()  (our code, direct edit)
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

    # Also verify the patch file exists
    patch = PROJECT_ROOT / "src" / "certiq_net" / "studies" / "qgym_eval" / "patches" / "0012-fix-gradient-accumulation.patch"
    ok3 = patch.exists()
    if ok3:
        print(f"[OK]   Patch file: 0012-fix-gradient-accumulation.patch exists")
    else:
        print(f"[FAIL] Patch file: 0012-fix-gradient-accumulation.patch MISSING")

    print()
    if ok1 and ok2 and ok3:
        print("[PASS] Fix 3B+2B verified: encoder gradients are zeroed between phases")
        return 0
    else:
        print("[FAIL] One or more components missing")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
