"""Helpers for locating a QGym checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import certiq_net


def project_root() -> Path:
    return Path(certiq_net.__file__).resolve().parents[2]


def certiq_package_root() -> Path:
    """Return the installed package directory ``src/certiq_net``."""
    return Path(certiq_net.__file__).resolve().parent


def certiq_model_config_path() -> Path:
    """Return the default CertiQ model config path."""
    return certiq_package_root() / "configs" / "model" / "certiq_index.yaml"


def _candidate_roots() -> list[Path]:
    root = project_root()
    candidates: list[Path] = []

    for name in ("CERTIQ_QGYM_ROOT", "QGYM_ROOT"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser().resolve())

    candidates.extend(
        [
            root / "extern" / "QGym",
            root.parent / "QGym",
            root / "QGym",
            Path.cwd() / "extern" / "QGym",
            Path.cwd() / "QGym",
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_qgym_root() -> Path:
    for candidate in _candidate_roots():
        if (candidate / "RL" / "PPO" / "trainer.py").exists():
            return candidate

    searched = "\n".join(f"  - {path}" for path in _candidate_roots())
    raise FileNotFoundError(
        "Could not locate a usable QGym checkout. Set QGYM_ROOT or "
        "CERTIQ_QGYM_ROOT to the forked QGym root.\n"
        f"Searched:\n{searched}"
    )


def ensure_qgym_on_path() -> Path:
    qgym_root = resolve_qgym_root()
    qgym_str = str(qgym_root)
    if qgym_str not in sys.path:
        sys.path.insert(0, qgym_str)
    return qgym_root
