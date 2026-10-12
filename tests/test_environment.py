import gymnasium as gym
import numpy as np

import panda_maniskill_vla.environment as environment_module
from panda_maniskill_vla.config import EnvironmentConfig
from panda_maniskill_vla.environment import EnvironmentRunner


class FakeEnvironment:
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.unwrapped = self
        self.reset_calls = 0

    def reset(self, seed=None):
        self.reset_calls += 1
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.ones(2, dtype=np.float32), 1.0, True, False, {}

    def render_rgb_array(self):
        return np.zeros((1, 8, 8, 3), dtype=np.uint8)


def test_terminal_action_does_not_reset_environment(monkeypatch) -> None:
    env = FakeEnvironment()
    monkeypatch.setattr(environment_module, "make_environment", lambda config: env)
    runner = EnvironmentRunner(EnvironmentConfig(render_mode="rgb_array"))
    runner.reset(seed=0)

    runner.step(np.zeros(7, dtype=np.float32))

    assert env.reset_calls == 1
    assert runner.episode == 1
    assert runner.episode_done is True
