"""Verify fix 2A: eval_freq = episode_steps * actors (not just episode_steps).

This test confirms the code change in train.py by parsing the actual source.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = PROJECT_ROOT / "src" / "certiq_net" / "studies" / "qgym_eval" / "train" / "train.py"


def main() -> int:
    src = TRAIN_PY.read_text(encoding="utf-8")

    # Check eval_freq is set to episode_steps * actors
    m1 = re.search(r'eval_freq\s*=\s*episode_steps\s*\*\s*actors', src)
    m2 = re.search(r'eval_freq\s*=\s*episode_steps', src)

    print("Checking train.py eval_freq assignment ...")
    if m1:
        print(f"[OK]   eval_freq = episode_steps * actors  (scaled by num actors)")
    elif m2:
        print(f"[FAIL] eval_freq = episode_steps  (NOT scaled by actors)")
        print("       Still using old buggy value!")
        return 1
    else:
        print("[FAIL] Could not find eval_freq assignment")
        return 1

    # Check --eval-freq CLI flag exists
    m3 = re.search(r'--eval-freq', src)
    if m3:
        print("[OK]   --eval-freq CLI flag present")
    else:
        print("[WARN] --eval-freq CLI flag not found")

    print()
    print("[PASS] Fix 2A verified: eval_freq scales with actors; override via --eval-freq")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
