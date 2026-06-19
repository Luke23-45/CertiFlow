"""Verify finding 3C: test envs use env_device (CPU) even when training on CUDA.

Checks that make_test_env passes device=torch.device(env_device) where
env_device = "cpu" when str(device) == "cuda".
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = (
    PROJECT_ROOT
    / "src" / "certiq_net" / "studies" / "qgym_eval" / "train" / "train.py"
)


def main() -> int:
    src = TRAIN_PY.read_text(encoding="utf-8")

    # Find the env_device line
    m = re.search(r'env_device\s*=\s*"cpu"\s+if\s+str\(device\)\s*==\s*"cuda"\s+else\s+device', src)
    if m is None:
        print("[FAIL] Could not find env_device assignment line")
        print("       Searching for 'env_device' in train.py:")
        for i, line in enumerate(src.splitlines(), 1):
            if "env_device" in line:
                print(f"  L{i}: {line.strip()}")
        return 1

    env_device_line = m.group(0)

    # Find make_test_env and its device argument
    # Look for the function definition
    func_match = re.search(
        r"def make_test_env\(seed\):.*?device=torch\.device\(env_device\)",
        src, re.DOTALL
    )
    if func_match is None:
        # Fallback: find the device= line inside make_test_env
        print("[WARN] Could not find device=torch.device(env_device) in make_test_env")
        print("       Searching for 'env_device' references in train.py:")
        for i, line in enumerate(src.splitlines(), 1):
            if "env_device" in line:
                print(f"  L{i}: {line.strip()}")
        return 2

    make_test_env_section = func_match.group(0)

    print(f"env_device assignment:     {env_device_line}")
    print(f"make_test_env device:      device=torch.device(env_device)")

    # Also check make_env (same pattern)
    make_env_match = re.search(
        r"def make_env\(seed\):.*?device=torch\.device\(env_device\)",
        src, re.DOTALL
    )
    if make_env_match:
        print(f"make_env device:           device=torch.device(env_device) (same)")

    try:
        assert "env_device" in str(make_test_env_section if func_match else "")
        print(f"\n[PASS] Finding 3C confirmed: test envs use env_device=CPU when device=cuda")
        print("       CPU env + GPU policy = CPU->GPU transfer each step.")
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
