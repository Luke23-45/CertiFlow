"""Platform and hardware utilities."""

from certiq_net.utils.platform import (
    PlatformInfo,
    detect_platform,
    resolve_num_workers,
    resolve_trainer_config,
    windows_safe_path,
)

__all__ = [
    "PlatformInfo",
    "detect_platform",
    "resolve_trainer_config",
    "resolve_num_workers",
    "windows_safe_path",
]
