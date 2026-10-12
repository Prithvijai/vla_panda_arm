import gymnasium as gym
import numpy as np
import pytest

from panda_maniskill_vla.spaces import coerce_action, neutral_action


def test_coerce_box() -> None:
    space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    action = coerce_action(space, [0.0] * 7)

    assert action.dtype == np.float32
    assert space.contains(action)


def test_coerce_rejects_wrong_shape() -> None:
    space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected action shape"):
        coerce_action(space, [0.0] * 6)


def test_dict_action_space() -> None:
    space = gym.spaces.Dict(
        {
            "arm": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "mode": gym.spaces.Discrete(3),
        }
    )

    action = coerce_action(space, {"arm": [0.5, -0.5], "mode": 2})

    assert space.contains(action)


def test_neutral_box_action_is_valid() -> None:
    space = gym.spaces.Box(-0.5, 1.0, shape=(7,), dtype=np.float32)

    action = neutral_action(space)

    np.testing.assert_array_equal(action, np.zeros(7, dtype=np.float32))
    assert space.contains(action)
