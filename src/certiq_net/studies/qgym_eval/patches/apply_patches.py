"""Apply QGym patches to the extern/QGym submodule.

This module resets the QGym submodule to a clean state and then applies
all ``.patch`` files from this directory.  It is designed to be called
once at the start of a training run so that local optimizations (e.g.
removing ``torch.tensor`` copy overhead) are always present without
permanently modifying the submodule.

Usage
-----
As a module::

    python -m certiq_net.studies.qgym_eval.patches.apply_patches

Programmatically::

    from certiq_net.studies.qgym_eval.patches.apply_patches import ensure_patches_applied
    ensure_patches_applied()
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import certiq_net

_PROJECT_ROOT = Path(certiq_net.__file__).resolve().parents[2]
_QGYM_ROOT = _PROJECT_ROOT / "extern" / "QGym"
_PATCHES_DIR = Path(__file__).resolve().parent

# Sentinel: after patches are applied we drop a marker file inside the
# QGym tree so we don't re-apply on every import.
_SENTINEL = _QGYM_ROOT / ".certiq_patches_applied"


def _run_git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the result."""
    cmd = ["git"] + list(args)
    return subprocess.run(
        cmd,
        cwd=cwd or _QGYM_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _reset_qgym() -> None:
    """Hard-reset the QGym submodule to its committed state."""
    result = _run_git("checkout", ".")
    if result.returncode != 0:
        print(f"[patches] WARNING: git checkout failed: {result.stderr}", file=sys.stderr)


def _collect_patches() -> list[Path]:
    """Return sorted list of .patch files in the patches directory."""
    patches = sorted(_PATCHES_DIR.glob("*.patch"))
    return patches


def _apply_patch(patch_path: Path) -> bool:
    """Apply a single patch file.  Returns True on success."""
    # First do a dry run
    check = _run_git("apply", "--check", str(patch_path))
    if check.returncode != 0:
        print(
            f"[patches] Patch {patch_path.name} cannot be applied cleanly:\n"
            f"  {check.stderr.strip()}",
            file=sys.stderr,
        )
        return False

    # Apply for real
    result = _run_git("apply", str(patch_path))
    if result.returncode != 0:
        print(
            f"[patches] Failed to apply {patch_path.name}:\n"
            f"  {result.stderr.strip()}",
            file=sys.stderr,
        )
        return False

    print(f"[patches] Applied {patch_path.name}")
    return True


def apply_all_patches(force: bool = False) -> None:
    """Reset QGym and apply all patches.

    Parameters
    ----------
    force : bool
        If True, re-apply even if the sentinel file exists.
    """
    if not _QGYM_ROOT.exists():
        print("[patches] QGym submodule not found — skipping patches.", file=sys.stderr)
        return

    if _SENTINEL.exists() and not force:
        return  # already applied

    print("[patches] Resetting QGym submodule to clean state …")
    _reset_qgym()

    patches = _collect_patches()
    if not patches:
        print("[patches] No .patch files found.")
        return

    print(f"[patches] Applying {len(patches)} patch(es) to {_QGYM_ROOT} …")
    all_ok = True
    for p in patches:
        if not _apply_patch(p):
            all_ok = False

    if all_ok:
        _SENTINEL.write_text("applied\n")
        print("[patches] All patches applied successfully.")
    else:
        print("[patches] Some patches failed — see warnings above.", file=sys.stderr)


def ensure_patches_applied() -> None:
    """Idempotent entry point — apply patches only if not already applied."""
    apply_all_patches(force=False)


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply QGym patches")
    parser.add_argument(
        "--force", action="store_true", help="Re-apply even if already applied"
    )
    args = parser.parse_args()
    apply_all_patches(force=args.force)
