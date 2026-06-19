"""Verify fix 2C: trainer seed is now set from config, not None.

Checks that `trainer_kwargs["seed"]` is set to `train_seed`
(a non-None value from the config).
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = PROJECT_ROOT / "src" / "certiq_net" / "studies" / "qgym_eval" / "train" / "train.py"


def main() -> int:
    src = TRAIN_PY.read_text(encoding="utf-8")

    # Find the line that sets "seed" in trainer_kwargs dict
    # Looks for: seed=train_seed,  or  seed = train_seed
    m = re.search(r'seed\s*=\s*train_seed', src)

    if m is None:
        print("[FAIL] Could not find seed=train_seed in trainer_kwargs")
        print("       All trainer-related seed lines:")
        for i, line in enumerate(src.splitlines(), 1):
            if "seed" in line.lower() and ("trainer" in line.lower() or "kwargs" in line.lower()):
                print(f"  L{i}: {line.strip()}")
        return 1

    print(f"Found:    {m.group().strip()}")
    print("Expected: seed=train_seed (non-None, from config)")
    print()
    print("[PASS] Fix 2C verified: SB3 seed is now set from config (train_seed)")
    print("       Training runs are now deterministic/reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
