import gymnasium as gym
import numpy as np

from panda_maniskill_vla.image_viewer import decode_observation_images
from panda_maniskill_vla.panda_env import PandaEnvConfig
from panda_maniskill_vla.sim_server import SimServer


def _observation(value: int) -> dict:
    rgb = np.full((1, 2, 2, 3), value, dtype=np.uint8)
    return {
        "sensor_data": {"base_camera": {"rgb": rgb}, "hand_camera": {"rgb": rgb + 1}},
        "agent": {"qpos": np.zeros((1, 9), np.float32), "qvel": np.zeros((1, 9), np.float32)},
        "extra": {
            "tcp_pose": np.zeros((1, 7), np.float32),
            "goal_pos": np.array([[0.1, 0.2, 0.3]], np.float32),
        },
    }


class FakeEnv:
    """Succeeds on the step where `succeed_at` steps have been taken in the episode."""

    def __init__(self, succeed_at: int | None = None) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.succeed_at = succeed_at
        self.steps = 0
        self.actions = []
        self.reset_seeds = []

    def reset(self, seed=None):
        self.steps = 0
        self.reset_seeds.append(seed)
        return _observation(0), {}

    def step(self, action):
        self.steps += 1
        self.actions.append(np.array(action))
        success = self.succeed_at is not None and self.steps >= self.succeed_at
        info = {"success": np.array([success]), "is_grasped": np.array([True])}
        return _observation(self.steps), np.array([0.5]), np.array([success]), False, info


class FakeSocket:
    def __init__(self) -> None:
        self.sent = []

    def send_multipart(self, frames) -> None:
        self.sent.append(frames)

    def last(self):
        return decode_observation_images(self.sent[-1])


def _server(env, **config) -> tuple[SimServer, FakeSocket]:
    socket = FakeSocket()
    server = SimServer(PandaEnvConfig(render_mode=None, **config), env, socket)
    server.reset_episode(7)
    return server, socket


def _queue(server: SimServer, sequence_id: int, count: int) -> None:
    server.pending_sequence_id = sequence_id
    server.pending_actions = np.full((count, 7), 0.5, dtype=np.float32)
    server.pending_index = 0


def test_reset_publishes_both_cameras_and_goal() -> None:
    _, socket = _server(FakeEnv())

    metadata, images = socket.last()

    assert metadata.is_reset and metadata.episode_id == 1 and metadata.seed == 7
    assert list(images) == ["base_camera", "hand_camera"]
    assert list(metadata.goal_pos) == [np.float32(0.1), np.float32(0.2), np.float32(0.3)]


def test_lockstep_does_not_step_without_actions() -> None:
    env = FakeEnv()
    server, socket = _server(env, lockstep=True)

    server.tick()
    server.tick()

    assert env.actions == []
    assert socket.last()[0].is_reset  # the reset observation is repeated


def test_continuous_mode_pads_with_zero_actions() -> None:
    env = FakeEnv()
    server, socket = _server(env)

    server.tick()

    np.testing.assert_array_equal(env.actions[0], np.zeros(7))
    assert socket.last()[0].used_zero_action


def test_lockstep_success_holds_until_reset() -> None:
    env = FakeEnv(succeed_at=2)
    server, socket = _server(env, lockstep=True)
    _queue(server, sequence_id=3, count=5)

    for _ in range(4):
        server.tick()

    assert len(env.actions) == 2  # remaining actions were dropped at success
    metadata, _ = socket.last()
    assert metadata.success and metadata.episode_id == 1 and metadata.elapsed_steps == 2
    assert metadata.reward == 0.5 and metadata.is_grasped


def test_time_limit_truncates_and_continuous_mode_auto_resets() -> None:
    env = FakeEnv()
    server, socket = _server(env, max_episode_steps=2, seed=100)

    server.tick()
    server.tick()

    truncated = decode_observation_images(socket.sent[-2])[0]
    assert truncated.truncated and not truncated.success
    metadata, _ = socket.last()
    assert metadata.is_reset and metadata.episode_id == 2
    assert env.reset_seeds == [7, 101]
