import argparse

import cv2
import numpy as np
import zmq

from panda_maniskill_vla.action_pb2 import ObservationMetadata

OBSERVATION_TOPIC = b"observation"


def decode_observation(frames: list[bytes]) -> tuple[ObservationMetadata, np.ndarray]:
    if len(frames) != 3:
        raise ValueError(f"Expected 3 observation frames, received {len(frames)}")

    topic, metadata_frame, image_frame = frames
    if topic != OBSERVATION_TOPIC:
        raise ValueError(f"Unexpected observation topic: {topic!r}")

    metadata = ObservationMetadata()
    metadata.ParseFromString(metadata_frame)
    if metadata.channels != 3:
        raise ValueError(f"Expected an RGB image, received {metadata.channels} channels")

    expected_bytes = metadata.width * metadata.height * metadata.channels
    if len(image_frame) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} image bytes, received {len(image_frame)}"
        )

    image = np.frombuffer(image_frame, dtype=np.uint8).reshape(
        metadata.height, metadata.width, metadata.channels
    )
    return metadata, image


def parse_args():
    parser = argparse.ArgumentParser(description="Display Panda simulation observations")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("scale must be greater than zero")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.SUBSCRIBE, OBSERVATION_TOPIC)
    socket.connect(args.endpoint)

    window_name = "Panda ManiSkill Observation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"Waiting for observations on {args.endpoint}; press q or Esc to exit")

    try:
        while True:
            if socket.poll(100, zmq.POLLIN):
                frames = socket.recv_multipart()
                while socket.poll(0, zmq.POLLIN):
                    frames = socket.recv_multipart()

                metadata, rgb = decode_observation(frames)
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                display = cv2.resize(
                    display,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_NEAREST,
                )

                if metadata.is_reset:
                    action_label = "reset"
                elif metadata.used_zero_action:
                    action_label = "zero"
                else:
                    action_label = (
                        f"chunk {metadata.action_sequence_id}:{metadata.action_index}"
                    )
                cv2.putText(
                    display,
                    f"step {metadata.step_id} | {action_label}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
