from uuid import uuid4

import gymnasium as gym
import numpy as np
import pytest
import zmq

from panda_maniskill_vla.action_pb2 import (
    ActionChunkReply,
    ActionChunkRequest,
    ObservationMetadata,
)
from panda_maniskill_vla.image_viewer import decode_observation
from panda_maniskill_vla.sim_server import (
    OBSERVATION_TOPIC,
    decode_action_chunk,
    encode_observation,
    receive_action_chunk,
)


@pytest.fixture
def action_space() -> gym.spaces.Box:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)


def test_decode_action_chunk(action_space: gym.spaces.Box) -> None:
    request = ActionChunkRequest(
        sequence_id=12,
        values=[0.25] * 7 + [-0.5] * 7,
    )

    actions = decode_action_chunk(request, action_space)

    assert actions.shape == (2, 7)
    assert actions.dtype == np.float32
    np.testing.assert_array_equal(actions[0], np.full(7, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(actions[1], np.full(7, -0.5, dtype=np.float32))


def test_decode_rejects_incomplete_action(action_space: gym.spaces.Box) -> None:
    request = ActionChunkRequest(values=[0.0] * 8)

    with pytest.raises(ValueError, match="multiple of 7"):
        decode_action_chunk(request, action_space)


def test_req_rep_acknowledges_chunk(action_space: gym.spaces.Box) -> None:
    context = zmq.Context()
    endpoint = f"inproc://actions-{uuid4().hex}"
    server = context.socket(zmq.REP)
    client = context.socket(zmq.REQ)
    server.setsockopt(zmq.LINGER, 0)
    client.setsockopt(zmq.LINGER, 0)
    server.bind(endpoint)
    client.connect(endpoint)

    try:
        request = ActionChunkRequest(sequence_id=42, values=[0.1] * 14)
        client.send(request.SerializeToString())

        received = receive_action_chunk(server, action_space)
        reply = ActionChunkReply()
        reply.ParseFromString(client.recv())
    finally:
        client.close()
        server.close()
        context.term()

    assert received is not None
    sequence_id, actions = received
    assert sequence_id == 42
    assert actions.shape == (2, 7)
    assert reply.sequence_id == 42
    assert reply.accepted_actions == 2
    assert reply.error == ""


def test_encode_rgb_and_robot_state_observation() -> None:
    rgb = np.arange(18, dtype=np.uint8).reshape(1, 2, 3, 3)
    observation = {
        "sensor_data": {"base_camera": {"rgb": rgb}},
        "agent": {
            "qpos": np.array([[0.1, 0.2]], dtype=np.float32),
            "qvel": np.array([[0.3, 0.4]], dtype=np.float32),
        },
        "extra": {
            "tcp_pose": np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        },
    }

    topic, metadata_frame, image_frame = encode_observation(
        observation,
        step_id=7,
        action_sequence_id=42,
        action_index=1,
        used_zero_action=False,
    )
    metadata = ObservationMetadata()
    metadata.ParseFromString(metadata_frame)
    restored_rgb = np.frombuffer(image_frame, dtype=np.uint8).reshape(
        metadata.height, metadata.width, metadata.channels
    )

    assert topic == OBSERVATION_TOPIC
    assert metadata.step_id == 7
    assert metadata.action_sequence_id == 42
    assert metadata.action_index == 1
    assert metadata.used_zero_action is False
    assert (metadata.height, metadata.width, metadata.channels) == (2, 3, 3)
    assert list(metadata.qpos) == pytest.approx([0.1, 0.2])
    assert list(metadata.qvel) == pytest.approx([0.3, 0.4])
    assert list(metadata.tcp_pose) == pytest.approx([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(restored_rgb, rgb[0])

    decoded_metadata, decoded_rgb = decode_observation(
        [topic, metadata_frame, image_frame]
    )
    assert decoded_metadata.step_id == 7
    np.testing.assert_array_equal(decoded_rgb, rgb[0])
