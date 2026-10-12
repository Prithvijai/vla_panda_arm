import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

from panda_maniskill_vla.config import PandaEnvConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Display a ManiSkill environment")
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--robot", type=str, default="panda")
    parser.add_argument("--obs-mode", type=str, default="rgb")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument(
        "--render-mode", type=str, default="human", choices=("human", "rgb_array")
    )
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def make_environment(config: PandaEnvConfig):
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


def main():
    args = parse_args()
    config = PandaEnvConfig(
        env_id=args.env_id,
        robot_uid=args.robot,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )
    env = make_environment(config)
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    try:
        env.reset(seed=args.seed)
        while True:
            env.step(zero_action)
            env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
