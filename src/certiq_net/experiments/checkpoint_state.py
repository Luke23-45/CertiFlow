"""Persistent checkpoint state manifest for cross-process model loading.

After training completes, ``save_checkpoint_state`` writes a small JSON
manifest to the experiment run directory plus an experiment-level
``.last_run.json`` that points to the latest trained run.  Downstream
evaluation functions use ``require_checkpoint_state`` to discover the
checkpoint path, or exit with a clear error if no trained checkpoint exists.

Paths are stored **relative** to the run root so manifests are portable
across machines (cloud â†’ local, different mount points, etc.).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from certiq_net.utils.platform import windows_safe_path

_CHECKPOINT_STATE_FILE = ".checkpoint_state.json"
_LAST_RUN_FILE = ".last_run.json"


class CheckpointNotFoundError(Exception):
    """Raised when no valid checkpoint state is found for the experiment root."""


@dataclass(frozen=True)
class CheckpointState:
    experiment_name: str
    run_id: str
    checkpoint_path: str
    model_target: str
    seed: int
    max_epochs: int
    timestamp_utc: str
    status: str


def save_checkpoint_state(
    paths_root: Path,
    checkpoint_path: Path,
    *,
    experiment_name: str,
    run_id: str,
    model_target: str,
    seed: int,
    max_epochs: int,
) -> Path:
    relative = checkpoint_path.relative_to(paths_root)
    state = CheckpointState(
        experiment_name=experiment_name,
        run_id=run_id,
        checkpoint_path=str(relative.as_posix()),
        model_target=model_target,
        seed=seed,
        max_epochs=max_epochs,
        timestamp_utc=datetime.now(UTC).isoformat(),
        status="completed",
    )
    state_file = paths_root / _CHECKPOINT_STATE_FILE
    with open(windows_safe_path(state_file), "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)
    return state_file


def read_checkpoint_state(paths_root: Path) -> CheckpointState | None:
    state_file = paths_root / _CHECKPOINT_STATE_FILE
    if not state_file.exists():
        return None
    try:
        with open(windows_safe_path(state_file), encoding="utf-8") as f:
            data = json.load(f)
        return CheckpointState(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        print(
            f"[error] Corrupted checkpoint state file: {state_file}\n"
            f"        {exc}",
            file=sys.stderr,
        )
        return None


def _resolve_checkpoint(checkpoint_path: str, paths_root: Path) -> Path:
    """Resolve a checkpoint path with portable fallback logic.

    Resolution order:
      1. Use the stored path as-is (absolute or relative).
      2. Resolve relative to *paths_root*.
      3. Extract the filename and search in standard subdirectories
         (``artifacts/``, then the run root itself).

    This lets manifests trained on one machine work seamlessly on another.
    """
    stored = Path(checkpoint_path)

    # 1 â€” Try the stored path verbatim
    if stored.exists():
        return stored.resolve()

    # 2 â€” Try resolving relative to the run root
    if not stored.is_absolute():
        candidate = (paths_root / stored).resolve()
        if candidate.exists():
            return candidate

    # 3 â€” Fall back: extract filename, search standard locations
    filename = stored.name
    for candidate in [
        paths_root / "artifacts" / filename,
        paths_root / filename,
    ]:
        if candidate.exists():
            return candidate.resolve()

    raise CheckpointNotFoundError(
        f"Checkpoint file referenced in state manifest does not exist:\n"
        f"        {stored}\n"
    )


def require_checkpoint_state(paths_root: Path) -> Path:
    state = read_checkpoint_state(paths_root)
    if state is None:
        raise CheckpointNotFoundError(
            f"No trained checkpoint found.\n"
            f"        Expected state file: {paths_root / _CHECKPOINT_STATE_FILE}\n"
            f"        Run training first, or verify the experiment output root\n"
            f"        and run-id match the trained run."
        )

    return _resolve_checkpoint(state.checkpoint_path, paths_root)


def load_checkpoint_weights(model: torch.nn.Module, paths_root: Path) -> None:
    ckpt_path = require_checkpoint_state(paths_root)
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        cleaned = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(cleaned, strict=False)
    else:
        model.load_state_dict(raw, strict=False)


# Experiment-level last run discovery


def save_last_run(experiment_root: Path, *, run_id: str, experiment_name: str) -> Path:
    last_run_file = experiment_root / _LAST_RUN_FILE
    experiment_root.mkdir(parents=True, exist_ok=True)
    with open(windows_safe_path(last_run_file), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "experiment_name": experiment_name}, f, indent=2)
    return last_run_file


def read_last_run(experiment_root: Path) -> dict | None:
    last_run_file = experiment_root / _LAST_RUN_FILE
    if not last_run_file.exists():
        return None
    try:
        with open(windows_safe_path(last_run_file), encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, TypeError) as exc:
        print(
            f"[error] Corrupted last-run file: {last_run_file}\n"
            f"        {exc}",
            file=sys.stderr,
        )
        return None
