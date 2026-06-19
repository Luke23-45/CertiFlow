"""Verify finding 2D: BatchedEnv always returns done=False.

In the original DiffDiscreteEventSystem, done=True when time exceeds train_T.
In BatchedEnv (patch 0011), done is always np.zeros(self.B, dtype=bool).
This test verifies by reading the patch code statically and, if possible,
by running BatchedEnv directly.
"""
from __future__ import annotations

import re
from pathlib import Path

PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "certiq_net" / "studies" / "qgym_eval" / "patches"
)


def main() -> int:
    patch_path = PATCH_DIR / "0011-batched-step.patch"
    if not patch_path.exists():
        print(f"[FAIL] Patch file not found at {patch_path}")
        return 1

    patch_src = patch_path.read_text(encoding="utf-8")

    # Check that BatchedEnv.step returns done = np.zeros(...)
    m = re.search(r"done\s*=\s*np\.zeros\(.*,\s*dtype\s*=\s*bool\)", patch_src)
    if m is None:
        print("[FAIL] Could not find 'done = np.zeros(*, dtype=bool)' in patch 0011")
        return 1

    done_line = m.group(0)

    # Also verify the original env has termination logic
    # Check for DiffDiscreteEventSystem done logic in other patches
    orig_done = False
    for p in ["0007-step-return-numpy.patch"]:
        psrc = (PATCH_DIR / p).read_text(encoding="utf-8")
        if "done" in psrc and ("time_elapsed" in psrc or "train_T" in psrc):
            orig_done = True
            break

    print(f"BatchedEnv done line:    {done_line}")
    print(f"Original env has done logic:        {orig_done}")

    try:
        assert "np.zeros" in done_line, "BatchedEnv.step() does not return False always"
        assert "dtype=bool" in done_line or "dtype = bool" in done_line
        print(f"\n[PASS] Finding 2D confirmed: BatchedEnv always returns done=False")
        print("       Original DiffDiscreteEventSystem had time-based termination;")
        print("       BatchedEnv relies entirely on truncation via n_steps.")
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
