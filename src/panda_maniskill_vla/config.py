from dataclasses import dataclass

@dataclass(slots=True) # can't add attr with slots
class PandaEnvConfig:
    env_id: str = "PickCube-v1"
    robot_uid: str = "panda"
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str | None = "rgb_array"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "auto"