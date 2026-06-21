"""Import QGym's RL components, managing sys.path automatically."""

from __future__ import annotations

from certiq_net.studies.qgym_eval._qgym_paths import ensure_qgym_on_path

_qgym_root = ensure_qgym_on_path()

# ruff: noqa: E402 - sys.path manipulation before QGym imports
from RL.PPO.trainer import CustomPPOTrainer  # noqa: F401,E402
from RL.utils.eval import parallel_eval  # noqa: F401,E402
from RL.utils.rl_env import load_rl_p_env  # noqa: F401,E402
from RL.utils.rollout_buffer import CustomRolloutBuffer  # noqa: F401,E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: F401,E402
