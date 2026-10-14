import argparse
import numpy as np

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from dataclasses import dataclass

@dataclass(slots=True) # can't add attr with slots
class PandaEnvConfig:
    # Default sim in maniskill (Pickup cube)
    env_id: str = "PickCube-v1"
    robot_uid: str = "panda"
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str | None = "rgb_array"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "auto"

    # parameters for zeroMQ server
    action_endpoint: str = "tcp://127.0.0.1:5555",
    observation_endpoint: str = "tcp://127.0.0.1:5556",
    hz: float = 20.0,
    seed: int = 0,

def make_environment_continuous(config: PandaEnvConfig):
    env = gym.make(
        config.env_id,
        robot_uids=config.robot_uid,
        num_envs=1,
        obs_mode=config.obs_mode,
        control_mode=config.control_mode,
        render_mode=config.render_mode,
        reward_mode=config.reward_mode,
        sim_backend=config.sim_backend,
    )
    # This application runs a continuous simulator, not fixed-length episodes.
    return env.unwrapped

