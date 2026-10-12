import argparse
import math
import time

import gymnasium as gym
import numpy as np
import zmq
from google.protobuf.message import DecodeError

from panda_maniskill_vla.action_pb2 import (
    ActionChunkReply,
    ActionChunkRequest,
    ObservationMetadata,
)
from panda_maniskill_vla.config import PandaEnvConfig
from panda_maniskill_vla.panda_env import make_environment

ACTION_DIM = 7
ACTION_DTYPE = np.dtype(np.float32)
CAMERA_NAME = "base_camera"
OBSERVATION_TOPIC = b"observation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZMQ Server with panda robotic arm sim from maniskill"
    )
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--obs-mode", default="rgb", choices=("rgb",))
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", default="human", choices=("human", "rgb_array"))
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--observation-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decode_action_chunk(
    request: ActionChunkRequest, action_space: gym.spaces.Box
) -> np.ndarray:
    values = np.asarray(request.values, dtype=ACTION_DTYPE)
    if values.size % ACTION_DIM != 0:
        raise ValueError(f"Action chunk must contain a multiple of {ACTION_DIM} values")

    actions = values.reshape(-1, ACTION_DIM)
    if not np.all(np.isfinite(actions)):
        raise ValueError("Action chunk contains NaN or infinity")
    if np.any(actions < action_space.low) or np.any(actions > action_space.high):
        raise ValueError("Action chunk contains values outside the action space")
    return actions


def receive_action_chunk(
    socket: zmq.Socket, action_space: gym.spaces.Box
) -> tuple[int, np.ndarray] | None:
    request = ActionChunkRequest()
    try:
        request.ParseFromString(socket.recv())
        actions = decode_action_chunk(request, action_space)
    except (DecodeError, ValueError) as error:
        reply = ActionChunkReply(sequence_id=request.sequence_id, error=str(error))
        socket.send(reply.SerializeToString())
        return None

    reply = ActionChunkReply(
        sequence_id=request.sequence_id,
        accepted_actions=len(actions),
    )
    socket.send(reply.SerializeToString())
    return request.sequence_id, actions


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_environment_array(value, name: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 0 or array.shape[0] != 1:
        raise ValueError(f"Expected {name} to have one environment, received {array.shape}")
    return array[0]


def encode_observation(
    observation: dict,
    *,
    step_id: int,
    action_sequence_id: int,
    action_index: int,
    used_zero_action: bool,
    is_reset: bool = False,
) -> list[bytes]:
    rgb = _single_environment_array(
        observation["sensor_data"][CAMERA_NAME]["rgb"], "RGB image"
    )
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected an HWC uint8 RGB image, received {rgb.shape} {rgb.dtype}")
    rgb = np.ascontiguousarray(rgb)

    qpos = _single_environment_array(observation["agent"]["qpos"], "qpos")
    qvel = _single_environment_array(observation["agent"]["qvel"], "qvel")
    tcp_pose = _single_environment_array(observation["extra"]["tcp_pose"], "TCP pose")

    metadata = ObservationMetadata(
        step_id=step_id,
        timestamp_ns=time.time_ns(),
        action_sequence_id=action_sequence_id,
        action_index=action_index,
        used_zero_action=used_zero_action,
        is_reset=is_reset,
        width=rgb.shape[1],
        height=rgb.shape[0],
        channels=rgb.shape[2],
        qpos=np.asarray(qpos, dtype=ACTION_DTYPE).reshape(-1).tolist(),
        qvel=np.asarray(qvel, dtype=ACTION_DTYPE).reshape(-1).tolist(),
        tcp_pose=np.asarray(tcp_pose, dtype=ACTION_DTYPE).reshape(-1).tolist(),
    )
    return [OBSERVATION_TOPIC, metadata.SerializeToString(), rgb.tobytes()]


def run_server(
    config: PandaEnvConfig,
    action_endpoint: str = "tcp://127.0.0.1:5555",
    observation_endpoint: str = "tcp://127.0.0.1:5556",
    hz: float = 20.0,
    seed: int = 0,
) -> None:
    if hz <= 0:
        raise ValueError("hz must be greater than zero")
    if config.obs_mode != "rgb":
        raise ValueError("sim server requires obs_mode='rgb'")

    env = make_environment(config)
    if (
        not isinstance(env.action_space, gym.spaces.Box)
        or env.action_space.shape != (ACTION_DIM,)
        or env.action_space.dtype != ACTION_DTYPE
    ):
        env.close()
        raise ValueError(
            f"Expected a float32 Box action space with shape ({ACTION_DIM},)"
        )

    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    pending_actions = np.empty((0, ACTION_DIM), dtype=ACTION_DTYPE)
    pending_index = 0
    pending_sequence_id = 0
    step_id = 0

    context = zmq.Context()
    action_socket = context.socket(zmq.REP)
    action_socket.setsockopt(zmq.LINGER, 0)
    observation_socket = context.socket(zmq.PUB)
    observation_socket.setsockopt(zmq.LINGER, 0)
    observation_socket.setsockopt(zmq.SNDHWM, 1)

    period = 1.0 / hz
    next_tick = time.monotonic()

    try:
        action_socket.bind(action_endpoint)
        observation_socket.bind(observation_endpoint)
        print(f"Action server listening on {action_endpoint}")
        print(f"Observation publisher listening on {observation_endpoint} at {hz:g} Hz")

        observation, _ = env.reset(seed=seed)
        observation_socket.send_multipart(
            encode_observation(
                observation,
                step_id=step_id,
                action_sequence_id=0,
                action_index=0,
                used_zero_action=False,
                is_reset=True,
            )
        )
        while True:
            now = time.monotonic()
            if now < next_tick:
                timeout_ms = math.ceil((next_tick - now) * 1000)
                if action_socket.poll(timeout_ms, zmq.POLLIN):
                    received = receive_action_chunk(action_socket, env.action_space)
                    if received is not None:
                        pending_sequence_id, pending_actions = received
                        pending_index = 0
                continue

            if pending_index < len(pending_actions):
                action_sequence_id = pending_sequence_id
                action_index = pending_index
                action = pending_actions[pending_index]
                pending_index += 1
                used_zero_action = False
            else:
                action_sequence_id = 0
                action_index = 0
                action = zero_action
                used_zero_action = True

            observation, _, _, _, _ = env.step(action)
            step_id += 1
            observation_socket.send_multipart(
                encode_observation(
                    observation,
                    step_id=step_id,
                    action_sequence_id=action_sequence_id,
                    action_index=action_index,
                    used_zero_action=used_zero_action,
                )
            )
            if config.render_mode == "human":
                env.render()

            next_tick += period
            if next_tick < time.monotonic():
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        action_socket.close()
        observation_socket.close()
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
    run_server(
        config,
        action_endpoint=args.action_endpoint,
        observation_endpoint=args.observation_endpoint,
        hz=args.hz,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
