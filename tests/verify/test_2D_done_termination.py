"""Verify 2D: the live env source exposes the expected done semantics."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERN_ENV = PROJECT_ROOT / "extern" / "QGym" / "main" / "env.py"


def main() -> int:
    src = EXTERN_ENV.read_text(encoding="utf-8")

    batched_match = re.search(r"done\s*=\s*np\.zeros\(self\.B,\s*dtype\s*=\s*bool\)", src)
    single_match = re.search(r"done\s*=\s*False", src)

    if batched_match is None:
        print("[FAIL] BatchedEnv.step() does not set done = np.zeros(self.B, dtype=bool)")
        return 1
    if single_match is None:
        print("[FAIL] DiffDiscreteEventSystem.step() does not set done = False")
        return 1

    print(f"BatchedEnv done line: {batched_match.group(0)}")
    print(f"Single-env done line: {single_match.group(0)}")
    print("\n[PASS] Live QGym env source exposes the expected done semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
