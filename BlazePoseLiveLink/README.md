# BlazePose → Unreal Live Link

Real-time webcam pose estimation streamed into Unreal Engine 5 via a custom
Live Link source. Optimized for speed: BlazePose GHUM Lite (model_complexity=0),
small binary UDP protocol, vectorized One-Euro filter, dedicated receive
thread on the UE side.

```
webcam ──▶ MediaPipe BlazePose GHUM ──▶ One-Euro filter
       └── (Python) ───────────────────────── UDP binary ───┐
                                                            ▼
                                            FBlazePoseLiveLinkSource
                                            (UE5 plugin, recv thread)
                                                            │
                                            ILiveLinkClient::Push*
                                                            ▼
                                  34 Live Link Transform subjects
                                  ("BlazePose_pelvis", "BlazePose_nose", ...)
                                                            │
                                            Live Link Transform Controller
                                                            ▼
                                            34 marker Actors in the level
```

## Layout

```
python/
  pose_sender.py      # webcam capture + inference + UDP send
  test_receiver.py    # standalone UDP receiver to verify the sender
  requirements.txt
Plugins/
  BlazePoseLiveLink/  # drop into your UE project's Plugins/ folder
```

## 1. Python side

```
cd python
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python pose_sender.py --backend default # default --host 127.0.0.1 --port 14043
```

Useful flags:
- `--complexity 0|1|2` — 0 = Lite (default, fastest), 1 = Full, 2 = Heavy.
- `--mincutoff 1.0 --beta 0.05` — One-Euro tuning. Lower mincutoff = more
  smoothing at rest. Higher beta = snappier on fast motion.
- `--mirror` — flip the webcam (selfie style).
- `--lock-root` — zero out pelvis world position. **Recommended** for
  monocular video; root depth from a single camera is unreliable.
- `--no-preview` — skip the OpenCV window for max throughput.

Verify the sender independently:
```
# terminal A
python test_receiver.py

# terminal B
python pose_sender.py
```

You should see ~30–60 fps and live joint coordinates in cm (UE space).

## 2. UE plugin

Copy `Plugins/BlazePoseLiveLink/` into your **C++** UE project's `Plugins/`
folder (create it if it doesn't exist), then:

1. Right-click your `.uproject` → **Generate Visual Studio project files**.
2. Open the project in your IDE and build the editor target.
3. Launch the editor. Enable the **Live Link** plugin if it's not already on.

To start a session:

1. **Window → Virtual Production → Live Link**.
2. **+ Source → BlazePose UDP**.
3. Run `pose_sender.py` from a terminal.
4. The "BlazePose" subject should appear, with frame counter ticking.

The default port (14043) is hard-coded in
`BlazePoseLiveLinkSourceFactory.cpp`. Change `DefaultPort` and rebuild,
or wire up a custom Slate panel via `EMenuType::SubPanel` if you want it
configurable in-editor.

## 3. Visualizing the markers

The plugin publishes **one Live Link Transform subject per joint**:
`BlazePose_pelvis`, `BlazePose_nose`, `BlazePose_l_wrist`, etc. — 34 total.
Each carries the joint's world-space position (cm, X-forward, Y-right,
Z-up) with identity rotation.

To put a marker on each joint in the level:

1. Place 34 small Actors in the level — e.g. `StaticMeshActor` with the
   engine `Sphere` mesh scaled to ~0.1, or any visualization actor you
   like.
2. On each Actor add a **Live Link Controller** component
   (Actor Details → Add Component → "Live Link Controller").
3. Set **Subject Representation** to the matching `BlazePose_<bone>`
   subject, and set **Controller Class** to `Live Link Transform
   Controller`. The actor will follow its joint live.

A quick way to bulk-create them: select all 34 subjects in the Live Link
panel and drag them into the viewport — UE5 will spawn one Actor per
subject already wired up.

If you instead want to drive a character rig with this data, see the
previous version of this plugin (animation-role branch), or solve
rotations downstream with an IK Rig fed by these marker positions as
effector targets.

The full joint list:

```
pelvis (root)                         <- midpoint of hips, world space
├── nose, l/r_eye*, l/r_ear, mouth_*
├── l/r_shoulder, l/r_elbow, l/r_wrist
├── l/r_pinky, l/r_index, l/r_thumb
├── l/r_hip, l/r_knee, l/r_ankle
└── l/r_heel, l/r_foot_idx
```

All rotations are **identity** — each subject just carries a world
position. This is intentional: monocular pose estimation gives you
reliable positions, not rotations.

## Performance notes

- BlazePose Lite typically runs 60+ fps on a modern CPU. If you're CPU-
  bound, drop `--width`/`--height` to 320×240; quality holds up well.
- The UE side spends almost no time per frame: parse 563 bytes, build 34
  `FTransform`s, push to Live Link. The receive thread is `TPri_AboveNormal`.
- If you see jitter, lower `--mincutoff` (e.g. 0.5) and keep `--beta` low.
  If wrists feel sluggish, raise `--beta` (e.g. 0.1–0.2).
- Monocular world-space depth is wobbly. Always start with `--lock-root`
  and add an estimated root motion path separately if you need it.

## Wire format

```
Header (19 bytes, little-endian, packed):
  char[4]   magic        = 'UELP'
  uint8     version      = 1
  uint32    frame_id
  float64   timestamp    (seconds since sender start)
  uint16    joint_count  = 34

Body (joint_count * 16 bytes):
  for each joint:
    float32 x, y, z       (UE space: cm, X-forward, Y-right, Z-up)
    float32 visibility    [0..1]
```

Total frame size: 19 + 34·16 = **563 bytes**, fits comfortably in one MTU.
