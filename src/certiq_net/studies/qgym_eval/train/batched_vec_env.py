"""BatchedVecEnv — SB3-compatible VecEnv wrapping a single BatchedEnv."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class BatchedVecEnv(VecEnv):
    """Wraps a BatchedEnv (B parallel envs in one object) as an SB3 VecEnv."""

    def __init__(self, env):
        self.env = env
        import gymnasium.spaces
        obs_space = gymnasium.spaces.Box(low=0, high=np.inf, shape=(env.q,), dtype=np.float32)
        act_space = gymnasium.spaces.Box(low=0, high=1, shape=(env.s, env.q), dtype=np.float32)
        super().__init__(env.B, obs_space, act_space)

    @property
    def device(self):
        return self.env.device

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, dones, infos = self.env.step(self._actions)
        self._actions = None
        return obs, rewards, dones, infos

    def reset(self):
        return self.env.reset()

    def close(self):
        pass

    def env_method(self, method_name, *args, **kwargs):
        result = getattr(self.env, method_name)(*args, **kwargs)
        return [result] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        val = getattr(self.env, attr_name)
        return [val] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
