import argparse

import cv2
import numpy as np
import zmq

from panda_maniskill_vla.messages import (  # noqa: F401  (re-exported)
    OBSERVATION_TOPIC,
    decode_observation,
    decode_observation_images,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Display Panda simulation observations")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--scale", type=int, default=2)
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

                metadata, images = decode_observation_images(frames)
                rgb = np.hstack(list(images.values()))
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                display = cv2.resize(
                    display,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_NEAREST,
                )

                if metadata.success:
                    action_label = "SUCCESS"
                elif metadata.is_reset:
                    action_label = "reset"
                elif metadata.used_zero_action:
                    action_label = "zero"
                else:
                    action_label = (
                        f"chunk {metadata.action_sequence_id}:{metadata.action_index}"
                    )
                cv2.putText(
                    display,
                    f"ep {metadata.episode_id} step {metadata.elapsed_steps} | {action_label}",
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
