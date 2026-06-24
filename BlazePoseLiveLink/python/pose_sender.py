"""
Real-time pose estimation -> Unreal Engine Live Link bridge.

Uses MediaPipe BlazePose GHUM (model_complexity=0 by default = Lite, fastest).
Streams 34 joints (synthetic pelvis + 33 BlazePose landmarks) over UDP
in a small binary protocol consumed by the BlazePoseLiveLink UE plugin.

Run:
    python pose_sender.py --host 127.0.0.1 --port 14043
"""

import argparse
import math
from pathlib import Path
import socket
import struct
import time
import urllib.error
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# One-Euro filter (vectorized)
# ---------------------------------------------------------------------------
class OneEuroFilter:
    """Vectorized one-euro filter operating on flat float32 arrays.

    Tune mincutoff lower for more smoothing at rest, beta higher to react
    faster to fast motion. mincutoff=1.0, beta=0.05 is a sane starting point
    for ~30-60 fps body pose.
    """

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.05, dcutoff: float = 1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, freq: float):
        cutoff = np.asarray(cutoff, dtype=np.float32)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        freq = 1.0 / dt

        dx = (x - self.x_prev) * freq
        a_d = self._alpha(self.dcutoff, freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, freq)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------
# Header (19 bytes, little-endian, no padding):
#   magic[4]='UELP' | version:u8 | frame_id:u32 | timestamp:f64 | joint_count:u16
# Body:
#   joint_count * (x:f32, y:f32, z:f32, visibility:f32)
#
# Coordinates are already in UE space: cm, X-forward, Y-right, Z-up.
MAGIC = b"UELP"
VERSION = 1
JOINT_COUNT = 34  # pelvis + 33 BlazePose landmarks
HEADER_FMT = "<4sBIdH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 19


def pack_frame(frame_id: int, timestamp: float,
               joints_xyz: np.ndarray, vis: np.ndarray) -> bytes:
    """joints_xyz: (34,3) f32 in UE cm-space. vis: (34,) f32 in [0,1]."""
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, frame_id, timestamp, JOINT_COUNT)
    body = np.empty((JOINT_COUNT, 4), dtype=np.float32)
    body[:, :3] = joints_xyz
    body[:, 3] = vis
    return header + body.tobytes()


# ---------------------------------------------------------------------------
# MediaPipe -> Unreal coordinate conversion
# ---------------------------------------------------------------------------
# MediaPipe pose_world_landmarks: meters, origin = midpoint of hips.
#   +X to subject's right in image space, +Y down, +Z away from camera.
# Unreal: cm, +X forward, +Y right, +Z up. Character forward = toward camera.
#
#   UE.X (forward) = -MP.z * 100   (+Z away from camera => character forward = toward camera = -MP.z)
#   UE.Y (right)   =  MP.x * 100
#   UE.Z (up)      = -MP.y * 100
def mp_world_to_ue(world_xyz: np.ndarray) -> np.ndarray:
    out = np.empty_like(world_xyz)
    out[:, 0] = -world_xyz[:, 2] * 100.0
    out[:, 1] =  world_xyz[:, 0] * 100.0
    out[:, 2] = -world_xyz[:, 1] * 100.0
    return out


# Index 0 of our 34-joint output is synthetic pelvis (midpoint of hips).
# The remaining 33 are BlazePose's standard landmarks in original order.
# Bone name list must match the UE plugin (BlazePoseLiveLinkSource.cpp).
LM = {
    "nose": 0, "l_eye_inner": 1, "l_eye": 2, "l_eye_outer": 3,
    "r_eye_inner": 4, "r_eye": 5, "r_eye_outer": 6,
    "l_ear": 7, "r_ear": 8, "mouth_l": 9, "mouth_r": 10,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20, "l_thumb": 21, "r_thumb": 22,
    "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28, "l_heel": 29, "r_heel": 30,
    "l_foot_idx": 31, "r_foot_idx": 32,
}


# ---------------------------------------------------------------------------
# MediaPipe compatibility layer
# ---------------------------------------------------------------------------
TASK_MODEL_NAMES = {
    0: "pose_landmarker_lite",
    1: "pose_landmarker_full",
    2: "pose_landmarker_heavy",
}


def ensure_task_model(complexity: int) -> str:
    """Download the Tasks pose model used by newer MediaPipe packages."""
    model_name = TASK_MODEL_NAMES[complexity]
    model_dir = Path(__file__).resolve().parent / ".mediapipe_models"
    model_path = model_dir / f"{model_name}.task"
    if model_path.exists():
        return str(model_path)

    model_dir.mkdir(exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"{model_name}/float16/latest/{model_name}.task"
    )
    print(f"Downloading {model_name}.task...")
    try:
        urllib.request.urlretrieve(url, model_path)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"MediaPipe {getattr(mp, '__version__', 'unknown')} needs a Tasks model, "
            f"but downloading {url} failed. Download it manually and place it at "
            f"{model_path}."
        ) from exc
    return str(model_path)


class PoseDetector:
    """Adapter for old mp.solutions.pose and newer mp.tasks.vision PoseLandmarker."""

    def __init__(self, complexity: int, detection_confidence: float,
                 presence_confidence: float, tracking_confidence: float):
        self.last_timestamp_ms = -1

        if hasattr(mp, "solutions"):
            self.kind = "solutions"
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=complexity,
                smooth_landmarks=False,        # we filter ourselves
                enable_segmentation=False,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            return

        self.kind = "tasks"
        self.mp_pose = mp.tasks.vision
        model_path = ensure_task_model(complexity)
        options = self.mp_pose.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=self.mp_pose.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            output_segmentation_masks=False,
        )
        self.pose = self.mp_pose.PoseLandmarker.create_from_options(options)

    def process(self, rgb: np.ndarray, timestamp_s: float):
        if self.kind == "solutions":
            return self.pose.process(rgb)

        timestamp_ms = max(int(timestamp_s * 1000.0), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.pose.detect_for_video(image, timestamp_ms)

    def world_landmarks(self, result):
        if self.kind == "solutions":
            return result.pose_world_landmarks.landmark if result.pose_world_landmarks else None
        return result.pose_world_landmarks[0] if result.pose_world_landmarks else None

    def has_pose_landmarks(self, result):
        if self.kind == "solutions":
            return result.pose_landmarks is not None
        return bool(result.pose_landmarks)

    def draw_landmarks(self, frame: np.ndarray, result):
        if self.kind == "solutions":
            if result.pose_landmarks is not None:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
            return

        if result.pose_landmarks:
            self.mp_pose.drawing_utils.draw_landmarks(
                frame,
                result.pose_landmarks[0],
                self.mp_pose.PoseLandmarksConnections.POSE_LANDMARKS,
            )

    def close(self):
        self.pose.close()


def is_blank_frame(frame: np.ndarray) -> bool:
    return float(np.mean(frame)) < 1.0 and float(np.std(frame)) < 1.0


def enhance_for_pose(frame: np.ndarray, mode: str, clahe) -> np.ndarray:
    if mode == "off":
        return frame

    brightness = float(np.mean(frame))
    if mode == "auto" and brightness >= 45.0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if brightness < 30.0:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.25, beta=24)
    return enhanced


def open_camera(index: int, width: int, height: int, backend: str):
    dshow_api = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else None
    msmf_api = cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else None
    backends = {
        "default": ("default", None),
        "dshow": ("dshow", dshow_api),
        "msmf": ("msmf", msmf_api),
    }
    if backend == "auto":
        candidates = [backends["dshow"], backends["msmf"], backends["default"]]
    else:
        candidates = [backends[backend]]

    attempts = []
    for label, api in candidates:
        if api is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, api)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't queue frames

        frame = None
        ok = False
        if cap.isOpened():
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None and not is_blank_frame(frame):
                    break

        blank = ok and frame is not None and is_blank_frame(frame)
        attempts.append((label, cap.isOpened(), ok, blank))
        if cap.isOpened() and ok and not blank:
            return cap, label

        if backend != "auto" and cap.isOpened() and ok:
            print(f"Warning: camera backend '{label}' is returning blank frames.")
            return cap, label

        cap.release()

    details = ", ".join(
        f"{label}: opened={opened} read={ok} blank={blank}"
        for label, opened, ok, blank in attempts
    )
    raise RuntimeError(f"Could not open a usable webcam index {index}. Attempts: {details}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=14043)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2],
                   help="BlazePose complexity (0=Lite/fastest, 1=Full, 2=Heavy).")
    p.add_argument("--mincutoff", type=float, default=1.0,
                   help="One-Euro mincutoff (lower = more smoothing at rest).")
    p.add_argument("--beta", type=float, default=0.05,
                   help="One-Euro beta (higher = more responsive to fast motion).")
    p.add_argument("--mirror", action="store_true",
                   help="Mirror webcam (selfie style). Default off.")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip the OpenCV preview window for max throughput.")
    p.add_argument("--lock-root", action="store_true",
                   help="Zero out the pelvis world position (recommended for monocular).")
    p.add_argument("--backend", choices=["auto", "default", "dshow", "msmf"], default="auto",
                   help="OpenCV camera backend. auto tries DirectShow then falls back.")
    p.add_argument("--enhance", choices=["auto", "on", "off"], default="auto",
                   help="Enhance dark webcam frames before pose inference.")
    p.add_argument("--detect-confidence", type=float, default=0.35,
                   help="Minimum pose detection confidence.")
    p.add_argument("--presence-confidence", type=float, default=0.35,
                   help="Minimum pose presence confidence for MediaPipe Tasks.")
    p.add_argument("--tracking-confidence", type=float, default=0.35,
                   help="Minimum pose tracking confidence.")
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)  # low delay
    addr = (args.host, args.port)

    cap, backend_label = open_camera(args.cam, args.width, args.height, args.backend)

    pose = PoseDetector(
        args.complexity,
        args.detect_confidence,
        args.presence_confidence,
        args.tracking_confidence,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    filt = OneEuroFilter(mincutoff=args.mincutoff, beta=args.beta)

    frame_id = 0
    t0 = time.perf_counter()
    last_print = t0
    camera_fps_accum = 0
    pose_fps_accum = 0
    send_fps_accum = 0
    brightness_accum = 0.0
    contrast_accum = 0.0

    print(f"Streaming to {args.host}:{args.port}  complexity={args.complexity}  "
          f"mirror={args.mirror}  lock_root={args.lock_root}  "
          f"backend={backend_label}  enhance={args.enhance}")
    print("Press 'q' in the preview window (or Ctrl+C) to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            camera_fps_accum += 1
            if args.mirror:
                frame = cv2.flip(frame, 1)

            brightness = float(np.mean(frame))
            contrast = float(np.std(frame))
            brightness_accum += brightness
            contrast_accum += contrast

            pose_frame = enhance_for_pose(frame, args.enhance, clahe)
            rgb = np.ascontiguousarray(cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB))
            t = time.perf_counter() - t0
            res = pose.process(rgb, t)
            world_landmarks = pose.world_landmarks(res)
            has_pose = pose.has_pose_landmarks(res)
            if has_pose:
                pose_fps_accum += 1

            if world_landmarks is not None:
                lm = np.asarray(
                    [[p.x, p.y, p.z] for p in world_landmarks],
                    dtype=np.float32,
                )
                vis = np.asarray(
                    [p.visibility if p.visibility is not None else 1.0 for p in world_landmarks],
                    dtype=np.float32,
                )

                ue = mp_world_to_ue(lm)  # (33, 3) in cm

                # Synthetic pelvis at midpoint of hips.
                pelvis = 0.5 * (ue[LM["l_hip"]] + ue[LM["r_hip"]])
                pelvis_vis = 0.5 * (vis[LM["l_hip"]] + vis[LM["r_hip"]])
                if args.lock_root:
                    pelvis[:] = 0.0

                joints = np.vstack([pelvis[None, :], ue])      # (34, 3)
                vis_full = np.concatenate([[pelvis_vis], vis]) # (34,)

                # Smooth all 34 * 3 channels jointly with shared filter state.
                joints = filt(joints.reshape(-1), t).reshape(JOINT_COUNT, 3)

                payload = pack_frame(frame_id, t, joints, vis_full)
                sock.sendto(payload, addr)
                frame_id += 1
                send_fps_accum += 1

            now = time.perf_counter()
            if now - last_print >= 1.0:
                elapsed = now - last_print
                cam_fps = camera_fps_accum / elapsed
                pose_fps = pose_fps_accum / elapsed
                send_fps = send_fps_accum / elapsed
                avg_brightness = brightness_accum / max(camera_fps_accum, 1)
                avg_contrast = contrast_accum / max(camera_fps_accum, 1)
                print(
                    f"\rcam={cam_fps:5.1f} fps  pose={pose_fps:5.1f} fps  "
                    f"send={send_fps:5.1f} fps  sent={frame_id}  "
                    f"brightness={avg_brightness:5.1f} contrast={avg_contrast:5.1f}",
                    end="",
                    flush=True,
                )
                camera_fps_accum = 0
                pose_fps_accum = 0
                send_fps_accum = 0
                brightness_accum = 0.0
                contrast_accum = 0.0
                last_print = now

            if not args.no_preview:
                pose.draw_landmarks(frame, res)
                if not has_pose:
                    cv2.putText(
                        frame,
                        f"No pose detected  brightness={brightness:.0f} contrast={contrast:.0f}",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("BlazePose -> UE", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        sock.close()


if __name__ == "__main__":
    main()
