"""Tests for certiq_net utility modules."""

from __future__ import annotations

from pathlib import Path

import torch


def test_platform_detect():
    from certiq_net.utils import detect_platform

    info = detect_platform()
    assert info.os_name is not None
    assert info.python_version is not None
    assert info.torch_version is not None
    assert info.best_accelerator in ("cpu", "cuda", "mps")


def test_windows_safe_path():
    from certiq_net.utils import windows_safe_path

    p = windows_safe_path(Path("/some/path"))
    assert isinstance(p, str)
    assert len(p) > 0


def test_resolve_num_workers():
    from certiq_net.utils import resolve_num_workers

    n = resolve_num_workers(None)
    assert n >= 0
    assert isinstance(n, int)


def test_checkpoint_state_save_load(tmp_path):
    from certiq_net.experiments import require_checkpoint_state, save_checkpoint_state

    ckpt = tmp_path / "model.pt"
    ckpt.write_text("dummy")
    torch.save({"dummy": torch.zeros(1)}, ckpt)

    save_checkpoint_state(
        paths_root=tmp_path,
        checkpoint_path=ckpt,
        experiment_name="test_exp",
        run_id="run_001",
        model_target="certiq_net",
        seed=42,
        max_epochs=10,
    )

    loaded = require_checkpoint_state(tmp_path)
    assert loaded.exists()
    assert loaded.name == "model.pt"
