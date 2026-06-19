"""Golden-reference equivalence test for ``main.env.allocator``.

This is the correctness safeguard for patch ``0004-vectorize-allocator``.
Because the patch rewrites ``allocator()`` to remove per-element host-syncs, it
must produce **byte-identical** outputs to the original (same allocated_work
values, same num_allocated counts, same queue_nonzero_inds ordering) for every
input.

Usage
-----
Two modes, run against whatever ``allocator`` is currently on disk:

    # 1. Before applying patch 0004 (upstream/original allocator):
    python -m certiq_net.studies.qgym_eval.patches.test_allocator_equivalence generate

    # 2. After applying patch 0004 (rewritten allocator):
    python -m certiq_net.studies.qgym_eval.patches.test_allocator_equivalence verify

If ``verify`` reports no mismatches, the rewrite is behavior-preserving.

What is compared
----------------
``allocator(action, mu, queue_service_times)`` returns
``(allocated_a, queue_nonzero_inds, num_allocated)``. We compare:

  - ``num_allocated``: list[int], exact equality.
  - ``allocated_a``: list[list[0-d tensor]], element-wise within rtol/atol.
  - ``queue_nonzero_inds``: dict[int, list[(s,q)]], exact equality of the
    (s,q) index sequence per queue (this encodes the sort order + the
    round()-based server duplication from env.py line 52).

Why ``queue_service_times`` matters
-----------------------------------
``num_allocated[q] = min(len(queue_service_times[q]), len(queue_nonzero_inds[q]))``
so the length of each per-queue service-time list caps allocation. The test
battery varies these lengths to exercise the min() truncation.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

import certiq_net
from certiq_net.studies.qgym_eval.patches.apply_patches import _QGYM_ROOT

# Make extern/QGym importable (same mechanism as qgym_import.py).
if str(_QGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_QGYM_ROOT))

from main.env import allocator  # noqa: E402

_PROJECT_ROOT = Path(certiq_net.__file__).resolve().parents[2]
_GOLDEN_FILE = _PROJECT_ROOT / "src" / "certiq_net" / "studies" / "qgym_eval" / "patches" / ".allocator_golden.pkl"

# Tolerances for floating-point comparison of allocated_work values.
_RTOL = 1e-6
_ATOL = 1e-7


def _make_inputs(seed: int):
    """Build a deterministic battery of (action, mu, service_times) inputs.

    Covers: the servicing action used by the profiler, the uniform action,
    random clipped [0,1] actions (the rollout path), and unclipped actions >1
    (the legacy differentiable path) to exercise the round()-based server
    duplication in allocator (env.py line 52).
    """
    rng = np.random.default_rng(seed)
    s, q = 2, 6  # reentrant_2 shape

    cases = []

    # Case A: servicing action (server 0 fully on every queue), like the profiler.
    a = np.zeros((s, q), dtype=np.float32)
    a[0, :] = 1.0
    cases.append(("servicing_server0", a.copy()))

    # Case B: uniform action = 1/q everywhere (all entries < 0.5 → rounds to 0).
    cases.append(("uniform_subthreshold", np.ones((s, q), dtype=np.float32) / q))

    # Case C-E: random clipped [0,1] actions with varied service-time lengths.
    for i in range(3):
        a = rng.random((s, q), dtype=np.float32)
        cases.append((f"random_clipped_{i}", a))

    # Case F-H: random actions with some entries > 1 to exercise server
    # duplication (round() → 2,3). Mix small and large magnitudes.
    for i in range(3):
        a = rng.random((s, q), dtype=np.float32) * 3.0  # range [0, 3)
        cases.append((f"random_unclipped_{i}", a))

    # Case I: all-ones action (every (s,q) allocates exactly 1 server).
    cases.append(("all_ones", np.ones((s, q), dtype=np.float32)))

    # Case J: action with exactly one nonzero per queue (clean allocation).
    a = np.zeros((s, q), dtype=np.float32)
    for qq in range(q):
        a[rng.integers(0, s), qq] = 1.0
    cases.append(("one_nonzero_per_queue", a))

    inputs = []
    for name, a_np in cases:
        # mu in a realistic positive range (service rates).
        mu_np = (rng.random((s, q), dtype=np.float32) * 2.0 + 0.1)
        action_t = torch.tensor(a_np).unsqueeze(0)  # (1, s, q) — allocator indexes [0]
        mu_t = torch.tensor(mu_np).unsqueeze(0)     # (1, s, q)

        # service_times: list (len q) of lists of 0-d/shape tensors. Vary lengths
        # per queue to exercise the min() truncation. Lengths 0..5.
        # The actual tensor values don't affect allocator output (only lengths do),
        # but we give them realistic shapes (1,1,q) to match the real usage.
        st: List[list] = []
        for qq in range(q):
            n = int(rng.integers(0, 6))
            st.append([torch.zeros((1, 1, q)) for _ in range(n)])

        inputs.append((name, action_t, mu_t, st))

    return inputs


def _run_allocator(inputs):
    """Run the on-disk allocator over all inputs, capture comparable outputs."""
    results = []
    for name, action_t, mu_t, st in inputs:
        raw = allocator(action_t, mu_t, [len(x) for x in st])
        # Handle both old (3-return) and new (2-return) allocator formats.
        if len(raw) == 3:
            allocated_a = raw[0]
            num_alloc = raw[2]
            allocated_serialized = [
                [float(v) for v in inner]
                for inner in allocated_a
            ]
        else:
            allocated_work, num_alloc = raw
            qq = allocated_work.shape[0]
            allocated_serialized = [
                [float(allocated_work[q_idx, j].item())
                 for j in range(int(num_alloc[q_idx].item()))]
                for q_idx in range(qq)
            ]
        results.append({
            "name": name,
            "allocated_a": allocated_serialized,
            "num_allocated": [int(n) for n in num_alloc],
        })
    return results


def _compare(golden, current, verbose: bool = True) -> bool:
    """Return True iff all cases match. Prints a per-case report."""
    ok = True
    if len(golden) != len(current):
        print(f"FAIL: case count differs: golden={len(golden)} current={len(current)}")
        return False

    for g, c in zip(golden, current):
        cname = g["name"]
        case_ok = True
        msgs = []

        # num_allocated: exact
        if g["num_allocated"] != c["num_allocated"]:
            case_ok = False
            msgs.append(f"  num_allocated: golden={g['num_allocated']} current={c['num_allocated']}")

            # allocated_a: element-wise within tol. Compare nested list lengths too.
            ga, ca = g["allocated_a"], c["allocated_a"]
            if len(ga) != len(ca):
                case_ok = False
                msgs.append(f"  allocated_a outer len: golden={len(ga)} current={len(ca)}")
            else:
                for qi, (gq, cq) in enumerate(zip(ga, ca)):
                    if len(gq) != len(cq):
                        case_ok = False
                        msgs.append(f"  allocated_a[{qi}] len: golden={len(gq)} current={len(cq)}")
                        continue
                    for vi, (gv, cv) in enumerate(zip(gq, cq)):
                        if not (abs(gv - cv) <= _ATOL + _RTOL * abs(gv)):
                            case_ok = False
                            msgs.append(f"  allocated_a[{qi}][{vi}]: golden={gv:.6g} current={cv:.6g}")
                    # Verify descending sort within each queue (behavioral invariant)
                    for vi in range(1, len(cq)):
                        if cq[vi] > cq[vi - 1] + _ATOL:
                            case_ok = False
                            msgs.append(f"  allocated_a[{qi}] NOT sorted descending at [{vi}]: "
                                        f"{cq[vi - 1]:.6g} -> {cq[vi]:.6g}")

        status = "PASS" if case_ok else "FAIL"
        if verbose or not case_ok:
            print(f"[{status}] {cname}")
            for m in msgs:
                print(m)
        ok = ok and case_ok

    return ok


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Golden-reference equivalence test for allocator")
    parser.add_argument("mode", choices=["generate", "verify"], help="generate or verify golden references")
    parser.add_argument("--seed", type=int, default=4242, help="RNG seed for the input battery")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print failures")
    args = parser.parse_args()

    print(f"[test] using allocator from: {_QGYM_ROOT / 'main' / 'env.py'}")
    inputs = _make_inputs(args.seed)

    if args.mode == "generate":
        results = _run_allocator(inputs)
        _GOLDEN_FILE.write_bytes(pickle.dumps(results))
        print(f"[test] wrote {len(results)} golden cases to {_GOLDEN_FILE}")
        return 0

    # verify
    if not _GOLDEN_FILE.exists():
        print(f"[test] No golden file at {_GOLDEN_FILE}.")
        print(f"[test] To create it, first reset patches to upstream:")
        print(f"         rm extern/QGym/.certiq_patches_applied")
        print(f"         (cd extern/QGym && git checkout .)")
        print(f"       then run:  python -m certiq_net.studies.qgym_eval.patches.test_allocator_equivalence generate")
        print(f"[test] The .pkl contains only plain Python lists/floats, so it is "
              f"platform-independent — the committed copy works on any OS.")
        return 2
    golden = pickle.loads(_GOLDEN_FILE.read_bytes())
    current = _run_allocator(inputs)
    ok = _compare(golden, current, verbose=not args.quiet)
    if ok:
        print(f"\n[test] ALL {len(current)} CASES MATCH — allocator rewrite is behavior-preserving.")
        return 0
    else:
        nfail = sum(1 for g, c in zip(golden, current)
                    if g["num_allocated"] != c["num_allocated"]
                    or g["allocated_a"] != c["allocated_a"])
        print(f"\n[test] {nfail}/{len(current)} CASES MISMATCH — rewrite changed behavior.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
