"""Import QGym's RL components, managing sys.path automatically.

This module handles the ``sys.path`` insertion for QGym and re-exports
the specific components needed by the training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import certiq_net

_project_root = Path(certiq_net.__file__).resolve().parents[2]
_qgym_root = _project_root / "extern" / "QGym"
if _qgym_root.exists() and str(_qgym_root) not in sys.path:
    sys.path.insert(0, str(_qgym_root))

# ruff: noqa: E402 — sys.path manipulation before QGym imports
from RL.PPO.trainer import CustomPPOTrainer  # noqa: F401
from RL.utils.eval import parallel_eval  # noqa: F401
from RL.utils.rl_env import load_rl_p_env  # noqa: F401
from RL.utils.rollout_buffer import CustomRolloutBuffer  # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: F401
