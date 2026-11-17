import numpy as np
import pytest

from panda_maniskill_vla.action_pb2 import ObservationMetadata
from panda_maniskill_vla.vla_client import (
    DEFAULT_STATE_OFFSET,
    libero_state,
    libero_to_maniskill_actions,
    summarize,
    wilson_interval,
)


def test_libero_actions_are_rescaled_and_gripper_flipped() -> None:
    libero = np.array([[1.0, -0.5, 0.2, 0.1, 0.0, -0.1, 1.0]], dtype=np.float32)

    actions = libero_to_maniskill_actions(libero)

    # 1.0 * 0.05 m / 0.1 m = 0.5; 0.1 * 0.5 rad / 0.1 rad = 0.5; close (+1) -> -1
    np.testing.assert_allclose(actions[0], [0.5, -0.25, 0.1, 0.5, 0.0, -0.5, -1.0], atol=1e-6)


def test_large_rotations_are_clipped() -> None:
    actions = libero_to_maniskill_actions(np.array([[0, 0, 0, 0.9, 0, 0, -1]], np.float32))
    assert actions[0, 3] == 1.0 and actions[0, 6] == 1.0


def test_start_pose_maps_to_libero_downward_orientation() -> None:
    metadata = ObservationMetadata(
        tcp_pose=[-0.004, -0.004, 0.164, 0.007, 0.711, 0.703, 0.003],
        qpos=[0.0] * 7 + [0.04, 0.04],
    )

    state = libero_state(metadata, np.asarray(DEFAULT_STATE_OFFSET))

    np.testing.assert_allclose(state[:3], [-0.047, 0.034, 0.765], atol=1e-3)
    assert abs(state[3]) == pytest.approx(np.pi, abs=0.05)  # pi about x: pointing down
    np.testing.assert_allclose(state[6:], [0.04, -0.04])


def test_wilson_interval() -> None:
    assert wilson_interval(0, 20)[0] == 0.0
    low, high = wilson_interval(10, 20)
    assert low == pytest.approx(0.299, abs=1e-3) and high == pytest.approx(0.701, abs=1e-3)


def test_summarize() -> None:
    episodes = [
        {"success": True, "steps": 50, "max_reward": 1.0, "ever_grasped": True,
         "inference_ms": [100.0]},
        {"success": False, "steps": 150, "max_reward": 0.2, "ever_grasped": False,
         "inference_ms": [300.0]},
    ]

    summary = summarize(episodes, wall_time_s=10.0)

    assert summary["success_rate"] == 0.5 and summary["grasp_rate"] == 0.5
    assert summary["mean_max_reward"] == pytest.approx(0.6)
    assert summary["inference_ms_p50"] == pytest.approx(200.0)
    assert summary["sim_steps_per_s"] == pytest.approx(20.0)
