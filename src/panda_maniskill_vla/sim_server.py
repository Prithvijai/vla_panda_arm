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
from panda_maniskill_vla.panda_env import PandaEnvConfig, make_environment_continuous

ACTION_DIM = 7
ACTION_DTYPE = np.dtype(np.float32)
OBSERVATION_TOPIC = b"observation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZMQ Server with panda robotic arm sim from maniskill"
    )
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument(
        "--robot",
        default="panda_wristcam",
        help="panda_wristcam publishes base_camera + hand_camera; panda only base_camera",
    )
    parser.add_argument("--obs-mode", default="rgb", choices=("rgb",))
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", default="human", choices=("human", "rgb_array"))
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--observation-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--hz", type=float, default=20.0, choices=(20.0, 10.0, 5.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="End episodes after this many steps (0 = only end on success)",
    )
    parser.add_argument(
        "--lockstep",
        action="store_true",
        help=(
            "Only step while actions are queued (no zero-action padding) and hold finished "
            "episodes until the client sends a reset. Use this for evaluation."
        ),
    )
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


def read_request(
    socket: zmq.Socket, action_space: gym.spaces.Box
) -> tuple[ActionChunkRequest, np.ndarray] | None:
    """Parse one request. Invalid requests are answered with an error and return None."""
    request = ActionChunkRequest()
    try:
        request.ParseFromString(socket.recv())
        if request.reset:
            actions = np.empty((0, ACTION_DIM), dtype=ACTION_DTYPE)
        else:
            actions = decode_action_chunk(request, action_space)
    except (DecodeError, ValueError) as error:
        reply = ActionChunkReply(sequence_id=request.sequence_id, error=str(error))
        socket.send(reply.SerializeToString())
        return None
    return request, actions


def receive_action_chunk(
    socket: zmq.Socket, action_space: gym.spaces.Box
) -> tuple[int, np.ndarray] | None:
    """Receive and acknowledge one action chunk (no episode handling)."""
    parsed = read_request(socket, action_space)
    if parsed is None:
        return None
    request, actions = parsed
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


def _flat_floats(value, name: str) -> list[float]:
    array = _single_environment_array(value, name)
    return np.asarray(array, dtype=ACTION_DTYPE).reshape(-1).tolist()


def _scalar(value, default=0.0):
    if value is None:
        return default
    return _as_numpy(value).reshape(-1)[0].item()


def encode_observation(
    observation: dict,
    *,
    step_id: int,
    action_sequence_id: int,
    action_index: int,
    used_zero_action: bool,
    is_reset: bool = False,
    episode_id: int = 0,
    elapsed_steps: int = 0,
    seed: int = 0,
    success: bool = False,
    truncated: bool = False,
    reward: float = 0.0,
    is_grasped: bool = False,
) -> list[bytes]:
    camera_names = []
    images = []
    for name, sensor in observation["sensor_data"].items():
        if "rgb" not in sensor:
            continue
        rgb = _single_environment_array(sensor["rgb"], f"{name} RGB image")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(
                f"Expected an HWC uint8 RGB image for {name}, received {rgb.shape} {rgb.dtype}"
            )
        if images and rgb.shape != images[0].shape:
            raise ValueError("All cameras must share one resolution")
        camera_names.append(name)
        images.append(np.ascontiguousarray(rgb))
    if not images:
        raise ValueError("Observation contains no RGB camera")

    extra = observation["extra"]
    goal_pos = extra.get("goal_pos")
    metadata = ObservationMetadata(
        step_id=step_id,
        timestamp_ns=time.time_ns(),
        action_sequence_id=action_sequence_id,
        action_index=action_index,
        used_zero_action=used_zero_action,
        is_reset=is_reset,
        width=images[0].shape[1],
        height=images[0].shape[0],
        channels=images[0].shape[2],
        qpos=_flat_floats(observation["agent"]["qpos"], "qpos"),
        qvel=_flat_floats(observation["agent"]["qvel"], "qvel"),
        tcp_pose=_flat_floats(extra["tcp_pose"], "TCP pose"),
        goal_pos=[] if goal_pos is None else _flat_floats(goal_pos, "goal position"),
        success=success,
        episode_id=episode_id,
        elapsed_steps=elapsed_steps,
        truncated=truncated,
        camera_names=camera_names,
        seed=seed,
        reward=reward,
        is_grasped=is_grasped,
    )
    return [OBSERVATION_TOPIC, metadata.SerializeToString(), *(image.tobytes() for image in images)]


def is_valid_env(env):
    if not isinstance(env, gym.Env):
        return False
    if env.action_space.shape != (ACTION_DIM,):
        return False
    if env.action_space.dtype != ACTION_DTYPE:
        return False
    return True


class SimServer:
    """Owns the environment, the queued action chunk, and the current episode."""

    def __init__(self, config: PandaEnvConfig, env, observation_socket: zmq.Socket) -> None:
        self.config = config
        self.env = env
        self.observation_socket = observation_socket
        self.zero_action = np.zeros(ACTION_DIM, dtype=ACTION_DTYPE)
        self._clear_actions()
        self.step_id = 0
        self.episode_id = 0
        self.seed = config.seed
        self.elapsed_steps = 0
        self.done = False
        self.last_frames: list[bytes] | None = None

    def _clear_actions(self) -> None:
        self.pending_actions = np.empty((0, ACTION_DIM), dtype=ACTION_DTYPE)
        self.pending_index = 0
        self.pending_sequence_id = 0

    def _publish(self, observation, **fields) -> None:
        self.last_frames = encode_observation(
            observation,
            step_id=self.step_id,
            episode_id=self.episode_id,
            elapsed_steps=self.elapsed_steps,
            seed=self.seed,
            **fields,
        )
        self.observation_socket.send_multipart(self.last_frames)

    def reset_episode(self, seed: int) -> None:
        observation, _ = self.env.reset(seed=seed)
        self.episode_id += 1
        self.seed = seed
        self.elapsed_steps = 0
        self.done = False
        self._clear_actions()
        self._publish(
            observation,
            action_sequence_id=0,
            action_index=0,
            used_zero_action=False,
            is_reset=True,
        )

    def handle_request(self, socket: zmq.Socket) -> None:
        parsed = read_request(socket, self.env.action_space)
        if parsed is None:
            return
        request, actions = parsed
        reply = ActionChunkReply(sequence_id=request.sequence_id)
        if request.reset:
            self.reset_episode(int(request.seed))
        elif self.done:
            reply.error = "Episode finished; send a reset request"
        else:
            self.pending_sequence_id = request.sequence_id
            self.pending_actions = actions
            self.pending_index = 0
            reply.accepted_actions = len(actions)
        reply.episode_id = self.episode_id
        socket.send(reply.SerializeToString())

    def tick(self) -> None:
        has_actions = self.pending_index < len(self.pending_actions)
        if self.done or (self.config.lockstep and not has_actions):
            # Hold still and repeat the latest observation so late subscribers catch up.
            if self.last_frames is not None:
                self.observation_socket.send_multipart(self.last_frames)
            self._render()
            return

        if has_actions:
            action_sequence_id = self.pending_sequence_id
            action_index = self.pending_index
            action = self.pending_actions[self.pending_index]
            self.pending_index += 1
            used_zero_action = False
        else:
            action_sequence_id = 0
            action_index = 0
            action = self.zero_action
            used_zero_action = True

        observation, reward, terminated, _, info = self.env.step(action)
        self.step_id += 1
        self.elapsed_steps += 1
        success = bool(_scalar(info.get("success", terminated), False))
        time_up = 0 < self.config.max_episode_steps <= self.elapsed_steps
        self._publish(
            observation,
            action_sequence_id=action_sequence_id,
            action_index=action_index,
            used_zero_action=used_zero_action,
            success=success,
            truncated=time_up and not success,
            reward=float(_scalar(reward)),
            is_grasped=bool(_scalar(info.get("is_grasped"), False)),
        )
        self._render()

        if success or time_up:
            if self.config.lockstep:
                self.done = True
                self._clear_actions()
            else:
                self.reset_episode(self.config.seed + self.episode_id)

    def _render(self) -> None:
        if self.config.render_mode == "human":
            self.env.render()


def run_server(config: PandaEnvConfig) -> None:
    env = make_environment_continuous(config)

    if not is_valid_env(env):
        env.close()
        raise ValueError(
            f"Expected a float32 Box action space with shape ({ACTION_DIM},)"
        )

    context = zmq.Context()
    action_socket = context.socket(zmq.REP)
    action_socket.setsockopt(zmq.LINGER, 0)
    observation_socket = context.socket(zmq.PUB)
    observation_socket.setsockopt(zmq.LINGER, 0)
    observation_socket.setsockopt(zmq.SNDHWM, 4)

    server = SimServer(config, env, observation_socket)
    period = 1.0 / config.hz
    next_tick = time.monotonic()

    try:
        action_socket.bind(config.action_endpoint)
        observation_socket.bind(config.observation_endpoint)
        mode = "lockstep" if config.lockstep else "continuous"
        print(f"Action server listening on {config.action_endpoint} ({mode})")
        print(
            f"Observation publisher listening on {config.observation_endpoint} "
            f"at {config.hz:g} Hz"
        )

        server.reset_episode(config.seed)
        while True:
            now = time.monotonic()
            if now < next_tick:
                timeout_ms = math.ceil((next_tick - now) * 1000)
                if action_socket.poll(timeout_ms, zmq.POLLIN):
                    server.handle_request(action_socket)
                continue

            server.tick()
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
        action_endpoint=args.action_endpoint,
        observation_endpoint=args.observation_endpoint,
        hz=args.hz,
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        max_episode_steps=args.max_episode_steps,
        lockstep=args.lockstep,
    )

    run_server(config)


if __name__ == "__main__":
    main()
