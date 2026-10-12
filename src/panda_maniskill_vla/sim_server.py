import argparse
import time
from typing import Any

import gymnasium as gym
import numpy as np
import zmq

from panda_maniskill_vla.config import PandaEnvConfig
from panda_maniskill_vla.panda_env import make_environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Panda ManiSkill env at a fixed rate over ZeroMQ Server"
    )
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", default="human", choices=("human", "rgb_array"))
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decode_action(message: Any, action_space: gym.spaces.Space) -> np.ndarray:
    if isinstance(message, dict):
        if "action" not in message:
            raise ValueError("Action message must contain an 'action' field")
        message = message["action"]

    action = np.asarray(message, dtype=action_space.dtype)
    expected_shape = action_space.shape

    # ManiSkill may expose one environment as either (action_dim,) or (1, action_dim).
    if action.shape != expected_shape and expected_shape[:1] == (1,):
        if action.shape == expected_shape[1:]:
            action = action.reshape(expected_shape)

    if action.shape != expected_shape:
        raise ValueError(f"Expected action shape {expected_shape}, received {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("Action contains NaN or infinity")
    if not action_space.contains(action):
        raise ValueError("Action is outside the environment action space")
    return action


def receive_latest_action(socket: zmq.Socket, action_space: gym.spaces.Space):
    latest_action = None
    while True:
        try:
            message = socket.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return latest_action

        latest_action = decode_action(message, action_space)


def run_server(
    config: PandaEnvConfig,
    endpoint: str = "tcp://127.0.0.1:5555",
    hz: float = 20.0,
    seed: int = 0,
) -> None:
    
    env = make_environment(config)
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    context = zmq.Context()
    action_socket = context.socket(zmq.PULL)
    action_socket.setsockopt(zmq.LINGER, 0)
    action_socket.bind(endpoint)

    period = 1.0 / hz
    next_tick = time.monotonic()
    print(f"ManiSkill server listening for actions on {endpoint} at {hz:g} Hz")

    try:
        env.reset(seed=seed)
        while True:
            action = receive_latest_action(action_socket, env.action_space)
            if action is None:
                action = zero_action

            env.step(action)
            if config.render_mode == "human":
                env.render()

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        action_socket.close()
        context.term()
        env.close()


def main():
    args = parse_args()
    config = PandaEnvConfig(
        env_id=args.env_id,
        robot_uid=args.robot,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
    )
    run_server(config, endpoint=args.endpoint, hz=args.hz, seed=args.seed)


if __name__ == "__main__":
    main()
