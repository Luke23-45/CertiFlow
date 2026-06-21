"""Legacy compatibility hook for QGym patch bootstrap."""

from __future__ import annotations

from certiq_net.studies.qgym_eval._qgym_paths import ensure_qgym_on_path


def ensure_patches_applied() -> None:
    """Ensure QGym is importable.

    The old patch files are no longer required because the fork now carries the
    implementation directly. This function remains so existing tests and entry
    points keep working.
    """
    ensure_qgym_on_path()
