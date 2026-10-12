from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import zmq

from panda_maniskill_vla.zmq_server import ZmqEnvironmentServer


class Box:
    def __init__(self) -> None:
        self.shape = (7,)
        self.dtype = np.dtype(np.float32)
        self.low = np.full(self.shape, -1.0, dtype=self.dtype)
        self.high = np.full(self.shape, 1.0, dtype=self.dtype)

    def contains(self, value) -> bool:
        array = np.asarray(value)
        return (
            array.shape == self.shape
            and array.dtype == self.dtype
            and bool(np.all(array >= self.low) and np.all(array <= self.high))
        )


class FakeRunner:
    def __init__(self) -> None:
        self.config = SimpleNamespace(render_mode="rgb_array")
        self.single_action_space = Box()
        self.episode = 1
        self.episode_step = 4
        self.episode_done = False
        self.closed = False

    def prepare_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.single_action_space.contains(action):
            raise ValueError("invalid action")
        return action

    def close(self) -> None:
        self.closed = True


def test_action_is_consumed_once_then_replaced_with_neutral_action() -> None:
    context = zmq.Context()
    suffix = uuid4().hex
    runner = FakeRunner()
    server = ZmqEnvironmentServer(
        runner,
        control_endpoint=f"inproc://control-{suffix}",
        observation_endpoint=f"inproc://observations-{suffix}",
        context=context,
    )

    try:
        reply = server._dispatch(
            {
                "message_type": "action",
                "episode_id": 1,
                "action_id": 9,
                "action": np.full(7, 0.25, dtype=np.float32),
            }
        )
        action, action_id = server._consume_action()
        neutral, neutral_id = server._consume_action()
    finally:
        server.close()
        context.term()

    assert reply["message_type"] == "action_queued"
    assert action_id == 9
    np.testing.assert_array_equal(action, np.full(7, 0.25, dtype=np.float32))
    assert neutral_id is None
    np.testing.assert_array_equal(neutral, np.zeros(7, dtype=np.float32))
    assert runner.closed is True
