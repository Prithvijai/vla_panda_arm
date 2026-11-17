"""Run a policy (SmolVLA or a baseline) against the ZMQ sim server and measure it.

Start the server in lockstep mode first:

    pms-sim-server --lockstep --max-episode-steps 200

then, from the environment that has LeRobot installed:

    pms-vla --policy smolvla --episodes 20

This module only needs numpy, pyzmq and protobuf (plus torch/lerobot for SmolVLA),
so it can run in a different Python environment from ManiSkill.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zmq

from panda_maniskill_vla.action_pb2 import (
    ActionChunkReply,
    ActionChunkRequest,
    ObservationMetadata,
)
from panda_maniskill_vla.messages import OBSERVATION_TOPIC, decode_observation_images

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLA

ACTION_DIM = 7
# ManiSkill pd_ee_delta_pose maps a normalized action of 1.0 to 0.1 m / 0.1 rad per step.
MANISKILL_POS_LIMIT = 0.1
MANISKILL_ROT_LIMIT = 0.1
# LIBERO (robosuite OSC_POSE) maps a normalized action of 1.0 to 0.05 m / 0.5 rad per step.
LIBERO_POS_LIMIT = 0.05
LIBERO_ROT_LIMIT = 0.5
# Shifts ManiSkill's start TCP position (~[0, 0, 0.16] world) onto the mean end-effector
# position in smolvla_libero's normalization stats ([-0.047, 0.034, 0.765]).
DEFAULT_STATE_OFFSET = (-0.043, 0.038, 0.601)
# ManiSkill's TCP frame is yawed 90 degrees relative to LIBERO's (both point down).
DEFAULT_STATE_YAW_DEG = 90.0

Images = dict[str, np.ndarray]


# --------------------------------------------------------------------------- sim connection


@dataclass
class EpisodeTrace:
    max_reward: float = 0.0
    ever_grasped: bool = False
    frames: list[np.ndarray] = field(default_factory=list)


class SimConnection:
    """REQ socket for action chunks and resets, SUB socket for observations."""

    def __init__(
        self,
        action_endpoint: str,
        observation_endpoint: str,
        timeout_s: float = 30.0,
        record: bool = False,
    ) -> None:
        self.context = zmq.Context()
        self.requests = self.context.socket(zmq.REQ)
        self.requests.setsockopt(zmq.LINGER, 0)
        self.requests.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self.requests.connect(action_endpoint)
        self.observations = self.context.socket(zmq.SUB)
        self.observations.setsockopt(zmq.LINGER, 0)
        self.observations.setsockopt(zmq.SUBSCRIBE, OBSERVATION_TOPIC)
        self.observations.connect(observation_endpoint)
        self.timeout_s = timeout_s
        self.record = record
        self.sequence_id = 0
        self.episode_id = 0
        self.trace = EpisodeTrace()
        self.saw_zero_action = False

    def close(self) -> None:
        self.requests.close()
        self.observations.close()
        self.context.term()

    def _request(self, request: ActionChunkRequest) -> ActionChunkReply:
        self.requests.send(request.SerializeToString())
        try:
            data = self.requests.recv()
        except zmq.Again as error:
            raise RuntimeError("Sim server did not answer; is pms-sim-server running?") from error
        reply = ActionChunkReply()
        reply.ParseFromString(data)
        if reply.error:
            raise RuntimeError(f"Sim server rejected request: {reply.error}")
        return reply

    def _wait_for(self, done) -> tuple[ObservationMetadata, Images]:
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0 or not self.observations.poll(remaining_ms, zmq.POLLIN):
                raise RuntimeError("Timed out waiting for an observation from the sim server")
            metadata, images = decode_observation_images(self.observations.recv_multipart())
            if metadata.episode_id != self.episode_id:
                continue
            self.trace.max_reward = max(self.trace.max_reward, metadata.reward)
            self.trace.ever_grasped |= metadata.is_grasped
            self.saw_zero_action |= metadata.used_zero_action
            if done(metadata):
                return metadata, images
            if self.record and not metadata.is_reset:
                self.trace.frames.append(np.hstack(list(images.values())))

    def reset(self, seed: int) -> tuple[ObservationMetadata, Images]:
        self.sequence_id += 1
        request = ActionChunkRequest(sequence_id=self.sequence_id, reset=True, seed=seed)
        reply = self._request(request)
        self.episode_id = reply.episode_id
        self.trace = EpisodeTrace()
        metadata, images = self._wait_for(lambda m: True)
        self.trace.max_reward = 0.0
        if self.record:
            self.trace.frames.append(np.hstack(list(images.values())))
        return metadata, images

    def execute(self, chunk: np.ndarray) -> tuple[ObservationMetadata, Images]:
        """Send a chunk and wait until it has been executed or the episode has ended."""
        chunk = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
        self.sequence_id += 1
        sequence_id, last_index = self.sequence_id, len(chunk) - 1
        self._request(ActionChunkRequest(sequence_id=sequence_id, values=chunk.ravel().tolist()))

        def finished(m: ObservationMetadata) -> bool:
            executed = m.action_sequence_id == sequence_id and m.action_index == last_index
            return executed or m.success or m.truncated

        metadata, images = self._wait_for(finished)
        if self.record:
            self.trace.frames.append(np.hstack(list(images.values())))
        return metadata, images


# --------------------------------------------------------------------------------- policies


class ZeroPolicy:
    """Holds still with the gripper open: shows what 'doing nothing' scores."""

    def reset(self) -> None:
        pass

    def act(self, metadata: ObservationMetadata, images: Images) -> np.ndarray:
        chunk = np.zeros((50, ACTION_DIM), dtype=np.float32)
        chunk[:, 6] = 1.0
        return chunk


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, metadata: ObservationMetadata, images: Images) -> np.ndarray:
        chunk = self.rng.uniform(-1.0, 1.0, size=(50, ACTION_DIM)).astype(np.float32)
        chunk[:, 6] = self.rng.choice([-1.0, 1.0])
        return chunk


def quaternion_wxyz_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    angle = 2.0 * math.acos(min(1.0, q[0]))
    sin_half = math.sqrt(max(0.0, 1.0 - q[0] * q[0]))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[1:] / sin_half * angle).astype(np.float32)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def libero_state(
    metadata: ObservationMetadata,
    position_offset: np.ndarray,
    yaw_deg: float = DEFAULT_STATE_YAW_DEG,
) -> np.ndarray:
    """ManiSkill observation -> LIBERO 8-D state: eef xyz, eef axis-angle, 2 finger joints."""
    tcp_pose = np.asarray(metadata.tcp_pose, dtype=np.float32)
    qpos = np.asarray(metadata.qpos, dtype=np.float32)
    position = tcp_pose[:3] + position_offset
    half_yaw = math.radians(yaw_deg) / 2
    yaw = np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])
    rotation = quaternion_wxyz_to_axis_angle(quaternion_multiply(tcp_pose[3:7], yaw))
    # LIBERO's second finger joint is negative when open; ManiSkill's are both positive.
    fingers = np.array([qpos[7], -qpos[8]], dtype=np.float32)
    return np.concatenate([position, rotation, fingers])


def libero_to_maniskill_actions(actions: np.ndarray, rot_gain: float = 1.0) -> np.ndarray:
    """LIBERO OSC actions -> ManiSkill normalized pd_ee_delta_pose actions."""
    actions = np.asarray(actions, dtype=np.float32)
    out = np.empty_like(actions)
    out[:, :3] = actions[:, :3] * (LIBERO_POS_LIMIT / MANISKILL_POS_LIMIT)
    out[:, 3:6] = actions[:, 3:6] * (LIBERO_ROT_LIMIT / MANISKILL_ROT_LIMIT) * rot_gain
    # LIBERO: -1 open / +1 close. ManiSkill: +1 open / -1 close.
    out[:, 6] = -np.sign(actions[:, 6])
    return np.clip(out, -1.0, 1.0)


class SmolVLAPolicy:
    """Wraps a LeRobot SmolVLA checkpoint trained on LIBERO-style inputs and actions."""

    def __init__(
        self,
        checkpoint: str,
        task: str,
        device: str = "cuda",
        flip_images: bool = False,
        state_offset: tuple[float, float, float] = DEFAULT_STATE_OFFSET,
        state_yaw_deg: float = DEFAULT_STATE_YAW_DEG,
        rot_gain: float = 1.0,
        cameras: tuple[str, ...] = ("base_camera", "hand_camera"),
    ) -> None:
        

        self.torch = torch
        self.device = device
        self.policy = LeRobotSmolVLA.from_pretrained(checkpoint)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        features = self.policy.config.input_features
        self.image_keys = [key for key in features if key.startswith("observation.images.")]
        self.cameras = cameras
        self.task = task
        self.flip_images = flip_images
        self.state_offset = np.asarray(state_offset, dtype=np.float32)
        self.state_yaw_deg = state_yaw_deg
        self.rot_gain = rot_gain

    def reset(self) -> None:
        self.policy.reset()

    def _image_tensor(self, image: np.ndarray):
        if self.flip_images:
            image = image[::-1, ::-1]
        tensor = self.torch.from_numpy(np.array(image))  # copy: ZMQ buffers are read-only
        return tensor.permute(2, 0, 1).float().div(255.0)

    def act(self, metadata: ObservationMetadata, images: Images) -> np.ndarray:
        frame = {
            "observation.state": self.torch.from_numpy(
                libero_state(metadata, self.state_offset, self.state_yaw_deg)
            ),
            "task": self.task,
        }
        for index, key in enumerate(self.image_keys):
            camera = self.cameras[min(index, len(self.cameras) - 1)]
            frame[key] = self._image_tensor(images.get(camera, next(iter(images.values()))))

        batch = self.preprocessor(frame)
        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        chunk = self.postprocessor(chunk)
        chunk = chunk.reshape(-1, chunk.shape[-1])[:, :ACTION_DIM].float().cpu().numpy()
        return libero_to_maniskill_actions(chunk, self.rot_gain)


# ------------------------------------------------------------------------------- evaluation


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(episodes: list[dict], wall_time_s: float) -> dict:
    n = len(episodes)
    successes = sum(e["success"] for e in episodes)
    low, high = wilson_interval(successes, n)
    latencies = [t for e in episodes for t in e["inference_ms"]]
    steps = sum(e["steps"] for e in episodes)
    return {
        "episodes": n,
        "success_rate": successes / n if n else 0.0,
        "success_ci95": [low, high],
        "grasp_rate": sum(e["ever_grasped"] for e in episodes) / n if n else 0.0,
        "mean_max_reward": float(np.mean([e["max_reward"] for e in episodes])) if n else 0.0,
        "mean_steps": steps / n if n else 0.0,
        "inference_ms_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "inference_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "sim_steps_per_s": steps / wall_time_s if wall_time_s > 0 else 0.0,
    }


def format_summary(name: str, summary: dict) -> str:
    low, high = summary["success_ci95"]
    return (
        f"{name}: success {summary['success_rate']:.0%} (95% CI {low:.0%}-{high:.0%}, "
        f"n={summary['episodes']}) | grasp {summary['grasp_rate']:.0%} | "
        f"progress {summary['mean_max_reward']:.2f} | "
        f"inference p50 {summary['inference_ms_p50']:.0f} ms / p95 "
        f"{summary['inference_ms_p95']:.0f} ms | {summary['sim_steps_per_s']:.1f} steps/s"
    )


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not installed; skipping video")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def run_episode(sim: SimConnection, policy, seed: int, actions_per_chunk: int) -> dict:
    metadata, images = sim.reset(seed)
    policy.reset()
    inference_ms = []
    while not (metadata.success or metadata.truncated):
        start = time.perf_counter()
        chunk = policy.act(metadata, images)[:actions_per_chunk]
        inference_ms.append((time.perf_counter() - start) * 1000)
        metadata, images = sim.execute(chunk)
    return {
        "seed": seed,
        "success": bool(metadata.success),
        "steps": int(metadata.elapsed_steps),
        "max_reward": float(sim.trace.max_reward),
        "ever_grasped": bool(sim.trace.ever_grasped),
        "inference_ms": inference_ms,
    }


def make_policy(args):
    if args.policy == "zero":
        return ZeroPolicy()
    if args.policy == "random":
        return RandomPolicy(seed=args.first_seed)
    return SmolVLAPolicy(
        checkpoint=args.checkpoint,
        task=args.task,
        device=args.device,
        flip_images=args.flip_images,
        state_offset=tuple(args.state_offset),
        state_yaw_deg=args.state_yaw_deg,
        rot_gain=args.rot_gain,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", choices=("smolvla", "random", "zero"), default="smolvla")
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--task", default="pick up the red cube")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--first-seed", type=int, default=1000, help="Episodes use seeds first_seed, +1, ..."
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=10,
        help="How many of the predicted actions to execute before re-planning",
    )
    parser.add_argument("--flip-images", action="store_true", help="Rotate images by 180 degrees")
    parser.add_argument(
        "--state-offset",
        type=float,
        nargs=3,
        default=DEFAULT_STATE_OFFSET,
        help="Added to the world-frame TCP position before it reaches SmolVLA",
    )
    parser.add_argument(
        "--state-yaw-deg",
        type=float,
        default=DEFAULT_STATE_YAW_DEG,
        help="Yaw applied to the TCP orientation to match LIBERO's end-effector frame",
    )
    parser.add_argument("--rot-gain", type=float, default=1.0)
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--observation-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--name", default=None, help="Run name (default: policy + chunk size)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--videos", type=int, default=0, help="Save videos of the first N episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actions_per_chunk < 1:
        raise ValueError("actions-per-chunk must be at least 1")
    name = args.name or f"{args.policy}_k{args.actions_per_chunk}"

    print(f"Loading policy '{args.policy}'...")
    policy = make_policy(args)
    sim = SimConnection(args.action_endpoint, args.observation_endpoint, record=args.videos > 0)
    episodes = []
    start = time.monotonic()
    try:
        for index in range(args.episodes):
            seed = args.first_seed + index
            sim.record = index < args.videos
            episode = run_episode(sim, policy, seed, args.actions_per_chunk)
            episodes.append(episode)
            print(
                f"[{index + 1}/{args.episodes}] seed {seed}: "
                f"{'SUCCESS' if episode['success'] else 'fail'} in {episode['steps']} steps, "
                f"progress {episode['max_reward']:.2f}, grasped {episode['ever_grasped']}"
            )
            if sim.record:
                save_video(sim.trace.frames, args.results_dir / "videos" / f"{name}_{seed}.mp4", 20)
    except KeyboardInterrupt:
        print("Interrupted; summarizing finished episodes")
    finally:
        sim.close()

    wall_time = time.monotonic() - start
    if sim.saw_zero_action:
        print("Warning: the server padded with zero actions; run it with --lockstep for evaluation")
    summary = summarize(episodes, wall_time)
    print(format_summary(name, summary))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    output = args.results_dir / f"{name}.json"
    output.write_text(json.dumps({"name": name, "config": config, "summary": summary,
                                  "episodes": episodes}, indent=2))
    print(f"Saved {output}")


def report_main() -> None:
    """Print a markdown table from saved result files: pms-vla-report results/*.json"""
    parser = argparse.ArgumentParser(description=report_main.__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    files = parser.parse_args().files
    print("| Run | Success (95% CI) | Grasp | Progress | Inference p50 / p95 | Steps/s |")
    print("|---|---|---|---|---|---|")
    for path in files:
        result = json.loads(path.read_text())
        s = result["summary"]
        low, high = s["success_ci95"]
        print(
            f"| {result['name']} | {s['success_rate']:.0%} ({low:.0%}-{high:.0%}), "
            f"n={s['episodes']} | {s['grasp_rate']:.0%} | {s['mean_max_reward']:.2f} | "
            f"{s['inference_ms_p50']:.0f} / {s['inference_ms_p95']:.0f} ms | "
            f"{s['sim_steps_per_s']:.1f} |"
        )


if __name__ == "__main__":
    main()
