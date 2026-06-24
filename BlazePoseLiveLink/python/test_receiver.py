"""
Standalone UDP receiver to verify pose_sender.py is working without needing
the Unreal plugin. Prints frame rate and a few key joint positions.

Run:
    python test_receiver.py --port 14043
"""

import argparse
import socket
import struct
import time

import numpy as np


MAGIC = b"UELP"
HEADER_FMT = "<4sBIdH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

BONE_NAMES = [
    "pelvis", "nose",
    "l_eye_inner", "l_eye", "l_eye_outer", "r_eye_inner", "r_eye", "r_eye_outer",
    "l_ear", "r_ear", "mouth_l", "mouth_r",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_pinky", "r_pinky", "l_index", "r_index", "l_thumb", "r_thumb",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "l_heel", "r_heel", "l_foot_idx", "r_foot_idx",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=14043)
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.setblocking(True)
    print(f"Listening on {args.host}:{args.port}")

    last = time.perf_counter()
    n = 0

    while True:
        data, _ = sock.recvfrom(8192)
        if len(data) < HEADER_SIZE:
            continue

        magic, version, frame_id, ts, joint_count = struct.unpack(
            HEADER_FMT, data[:HEADER_SIZE]
        )
        if magic != MAGIC:
            continue

        body = np.frombuffer(
            data, dtype=np.float32, count=joint_count * 4, offset=HEADER_SIZE
        ).reshape(joint_count, 4)

        n += 1
        now = time.perf_counter()
        if now - last >= 1.0:
            fps = n / (now - last)
            pelvis = body[0, :3]
            l_wrist = body[BONE_NAMES.index("l_wrist"), :3]
            r_wrist = body[BONE_NAMES.index("r_wrist"), :3]
            print(
                f"{fps:5.1f} fps  frame={frame_id}  "
                f"pelvis=({pelvis[0]:+6.1f},{pelvis[1]:+6.1f},{pelvis[2]:+6.1f})  "
                f"l_wrist=({l_wrist[0]:+6.1f},{l_wrist[1]:+6.1f},{l_wrist[2]:+6.1f})  "
                f"r_wrist=({r_wrist[0]:+6.1f},{r_wrist[1]:+6.1f},{r_wrist[2]:+6.1f})"
            )
            n = 0
            last = now


if __name__ == "__main__":
    main()
