from dataclasses import dataclass

import gymnasium as gym
import mani_skill.envs  # noqa: F401

@dataclass(slots=True)  # can't add attr with slots
class PandaEnvConfig:
    # Default sim in maniskill (Pickup cube)
    env_id: str = "PickCube-v1"
    # panda_wristcam adds a "hand_camera" next to the default "base_camera"
    robot_uid: str = "panda_wristcam"
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str | None = "rgb_array"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "auto"
    camera_width: int = 256
    camera_height: int = 256

    # parameters for zeroMQ server
    action_endpoint: str = "tcp://127.0.0.1:5555"
    observation_endpoint: str = "tcp://127.0.0.1:5556"
    hz: float = 20.0
    seed: int = 0
    # 0 means episodes only end on success
    max_episode_steps: int = 0
    # Only step when actions are queued and hold finished episodes until a reset request
    lockstep: bool = False


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
        sensor_configs=dict(width=config.camera_width, height=config.camera_height),
    )
    # Episode limits are handled by the server instead of the TimeLimit wrapper.
    return env.unwrapped
