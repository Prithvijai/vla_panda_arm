import numpy as np
import pytest

from panda_maniskill_vla.protocol import ProtocolError, decode_message, encode_message


def test_multipart_round_trip() -> None:
    source = {
        "protocol_version": 1,
        "observation": {
            "rgb": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            "state": np.array([1.25, -2.5], dtype=np.float32),
        },
        "metadata": (np.int64(4), b"data"),
    }

    frames = encode_message(source)
    restored = decode_message(frames)

    assert len(frames) == 4
    np.testing.assert_array_equal(
        restored["observation"]["rgb"], source["observation"]["rgb"]
    )
    np.testing.assert_array_equal(
        restored["observation"]["state"], source["observation"]["state"]
    )
    assert restored["observation"]["state"].dtype == np.float32
    assert restored["metadata"] == (4, b"data")


def test_rejects_array_frame_with_wrong_size() -> None:
    frames = encode_message({"array": np.arange(4, dtype=np.int16)})
    frames[-1] = memoryview(b"too short")

    with pytest.raises(ProtocolError, match="expected"):
        decode_message(frames)
