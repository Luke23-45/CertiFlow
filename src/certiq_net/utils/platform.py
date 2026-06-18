"""Intelligent hardware/platform detection and config resolution."""

from __future__ import annotations

import multiprocessing
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PlatformInfo:
    os_name: str
    os_version: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    rocm_available: bool
    mps_available: bool
    gpu_count: int
    gpu_names: list[str]
    cpu_count_physical: int
    cpu_count_logical: int

    @property
    def best_accelerator(self) -> str:
        if self.cuda_available:
            return "cuda"
        if self.mps_available:
            return "mps"
        return "cpu"

    @property
    def supported_precisions(self) -> list[str]:
        precisions = ["32-true"]
        if self.cuda_available:
            try:
                if torch.cuda.is_bf16_supported():
                    precisions.append("bf16-mixed")
            except (RuntimeError, AttributeError):
                pass
            precisions.append("16-mixed")
        elif self.mps_available:
            precisions.append("16-mixed")
        return precisions

    @property
    def recommended_precision(self) -> str:
        if "bf16-mixed" in self.supported_precisions:
            return "bf16-mixed"
        if "16-mixed" in self.supported_precisions:
            return "16-mixed"
        return "32-true"

    @property
    def safe_num_workers(self) -> int:
        return max(1, self.cpu_count_logical - 1) if self.cpu_count_logical > 1 else 0


def detect_platform() -> PlatformInfo:
    gpu_names: list[str] = []
    gpu_count = 0
    cuda_ver: str | None = None
    rocm = False
    mps = False

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        cuda_ver = torch.version.cuda
        rocm = torch.version.hip is not None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps = True

    cpu_count_physical = multiprocessing.cpu_count()

    return PlatformInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=sys.version,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_ver,
        rocm_available=rocm,
        mps_available=mps,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        cpu_count_physical=max(1, cpu_count_physical),
        cpu_count_logical=max(1, multiprocessing.cpu_count()),
    )


def resolve_trainer_config(
    trainer_cfg: dict[str, object],
    platform_info: PlatformInfo | None = None,
) -> dict[str, object]:
    """Resolve platform-dependent trainer settings with intelligent fallbacks."""
    info = platform_info or detect_platform()
    resolved = dict(trainer_cfg)

    accel = str(resolved.get("accelerator", "auto"))
    if accel == "auto":
        resolved["accelerator"] = info.best_accelerator
    elif accel == "cuda" and not info.cuda_available:
        resolved["accelerator"] = "cpu"

    precision = str(resolved.get("precision", "32-true"))
    if precision not in info.supported_precisions:
        resolved["precision"] = info.recommended_precision

    devices_raw: object = resolved.get("devices", 1)
    devices: int = 1
    if isinstance(devices_raw, str) and devices_raw in ("auto", "max"):
        devices = info.gpu_count if info.gpu_count > 0 else 1
    elif isinstance(devices_raw, (int, float)):
        devices = int(devices_raw)
    if devices > info.gpu_count and info.gpu_count > 0:
        devices = info.gpu_count
    resolved["devices"] = max(1, devices)

    return resolved


def resolve_num_workers(requested: int | None, platform_info: PlatformInfo | None = None) -> int:
    info = platform_info or detect_platform()
    if requested is None or requested < 0:
        if "google.colab" in sys.modules or "ipykernel" in sys.modules:
            return 0
        return info.safe_num_workers
    return min(requested, info.safe_num_workers)


def windows_safe_path(path: Path) -> str:
    """Return a path string safe for long Windows paths.

    On Windows, the ``\\\\?\\`` prefix bypasses the legacy MAX_PATH limit.
    Other platforms return the normal absolute path string.
    """
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text
