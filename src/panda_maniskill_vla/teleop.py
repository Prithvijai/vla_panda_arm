import argparse
import threading
import time

import numpy as np
import zmq

from panda_maniskill_vla.action_pb2 import ActionChunkReply, ActionChunkRequest

ACTION_DIM = 7
MOVE_DIRECTIONS = {
    "w": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "s": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "a": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "d": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "q": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
}


def build_action(pressed_keys: set[str], gripper: float, speed: float) -> np.ndarray:
    direction = np.zeros(3, dtype=np.float32)
    for key in pressed_keys:
        if key in MOVE_DIRECTIONS:
            direction += MOVE_DIRECTIONS[key]

    norm = np.linalg.norm(direction)
    if norm > 0:
        direction *= speed / norm

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:3] = direction
    action[6] = gripper
    return action


def send_action(socket: zmq.Socket, sequence_id: int, action: np.ndarray) -> None:
    request = ActionChunkRequest(sequence_id=sequence_id, values=action.tolist())
    socket.send(request.SerializeToString())

    reply = ActionChunkReply()
    reply.ParseFromString(socket.recv())
    if reply.sequence_id != sequence_id:
        raise RuntimeError(
            f"Expected reply for sequence {sequence_id}, received {reply.sequence_id}"
        )
    if reply.error:
        raise RuntimeError(f"Server rejected action: {reply.error}")
    if reply.accepted_actions != 1:
        raise RuntimeError(f"Server accepted {reply.accepted_actions} actions instead of one")


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for the Panda arm")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="Normalized translation command magnitude in the range (0, 1]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hz <= 0:
        raise ValueError("hz must be greater than zero")
    if not 0 < args.speed <= 1:
        raise ValueError("speed must be in the range (0, 1]")

    from pynput import keyboard

    pressed_keys: set[str] = set()
    gripper = 1.0
    state_lock = threading.Lock()
    stop_event = threading.Event()

    def on_press(key):
        nonlocal gripper
        if key == keyboard.Key.esc:
            stop_event.set()
            return False

        character = getattr(key, "char", None)
        if character is None:
            return None
        character = character.lower()
        with state_lock:
            if character in MOVE_DIRECTIONS:
                pressed_keys.add(character)
            elif character == "e":
                gripper = 1.0
            elif character == "c":
                gripper = -1.0
        return None

    def on_release(key):
        character = getattr(key, "char", None)
        if character is not None:
            with state_lock:
                pressed_keys.discard(character.lower())

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect(args.endpoint)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("W/S: +/-X | A/D: +/-Y | Q/Z: up/down | E/C: open/close | Esc: quit")
    print(f"Sending actions to {args.endpoint} at {args.hz:g} Hz")

    period = 1.0 / args.hz
    next_tick = time.monotonic()
    sequence_id = 1

    try:
        while not stop_event.is_set():
            with state_lock:
                action = build_action(set(pressed_keys), gripper, args.speed)

            try:
                send_action(socket, sequence_id, action)
            except zmq.Again as error:
                raise RuntimeError(f"Action server did not respond at {args.endpoint}") from error
            sequence_id += 1

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        listener.stop()
        listener.join()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
