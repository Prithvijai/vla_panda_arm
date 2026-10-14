import numpy as np
import pytest

from panda_maniskill_vla.teleop import build_action


def test_movement_key_mapping() -> None:
    np.testing.assert_array_equal(
        build_action({"w"}, gripper=1.0, speed=0.2),
        np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        build_action({"z"}, gripper=-1.0, speed=0.2),
        np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    )


def test_diagonal_movement_keeps_configured_speed() -> None:
    action = build_action({"w", "a", "q"}, gripper=1.0, speed=0.3)

    assert np.linalg.norm(action[:3]) == pytest.approx(0.3)
    assert np.all(action[:3] > 0)


def test_opposite_keys_cancel() -> None:
    action = build_action({"w", "s", "q", "z"}, gripper=-1.0, speed=0.2)

    np.testing.assert_array_equal(action[:6], np.zeros(6, dtype=np.float32))
    assert action[6] == -1.0
