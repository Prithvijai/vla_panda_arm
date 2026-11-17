"""Observation decoding shared by clients. Depends only on numpy and protobuf."""

import numpy as np

from panda_maniskill_vla.action_pb2 import ObservationMetadata

OBSERVATION_TOPIC = b"observation"


def decode_observation_images(
    frames: list[bytes],
) -> tuple[ObservationMetadata, dict[str, np.ndarray]]:
    """Decode [topic, metadata, image...] into metadata and {camera name: HWC RGB image}."""
    if len(frames) < 3:
        raise ValueError(f"Expected at least 3 observation frames, received {len(frames)}")

    topic, metadata_frame, *image_frames = frames
    if topic != OBSERVATION_TOPIC:
        raise ValueError(f"Unexpected observation topic: {topic!r}")

    metadata = ObservationMetadata()
    metadata.ParseFromString(metadata_frame)
    if metadata.channels != 3:
        raise ValueError(f"Expected an RGB image, received {metadata.channels} channels")

    names = list(metadata.camera_names) or [f"camera{i}" for i in range(len(image_frames))]
    if len(names) != len(image_frames):
        raise ValueError(f"Expected {len(names)} images, received {len(image_frames)}")

    expected_bytes = metadata.width * metadata.height * metadata.channels
    images = {}
    for name, image_frame in zip(names, image_frames, strict=True):
        if len(image_frame) != expected_bytes:
            raise ValueError(
                f"Expected {expected_bytes} image bytes, received {len(image_frame)}"
            )
        images[name] = np.frombuffer(image_frame, dtype=np.uint8).reshape(
            metadata.height, metadata.width, metadata.channels
        )
    return metadata, images


def decode_observation(frames: list[bytes]) -> tuple[ObservationMetadata, np.ndarray]:
    """Decode an observation and return only the first camera image."""
    metadata, images = decode_observation_images(frames)
    return metadata, next(iter(images.values()))
