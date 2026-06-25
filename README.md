# Side-Projects

A collection of standalone experiments. Each is documented below using the **STAR** method — *Situation, Task, Action, Result* — so the goal, the work, and the outcome are clear at a glance.

---

## 1. Color Jumper — Unity 2D Vertical Climber

A fast, score-driven climber where you bounce ever higher on platforms that only catch you when your color matches theirs.

### Situation (Color Jumper)

I wanted a small, self-contained Unity game to practice clean gameplay architecture (2D physics, power-ups, scoring) using Unity's URP 2D renderer and the new Input System.

### Task (Color Jumper)

Build a complete, playable vertical-climber loop: responsive controls, a color-matching bounce mechanic, a variety of power-ups, an endless spawner, and persistent high scores — all wired together with a clear, decoupled component design.

### Action (Color Jumper)

Implemented the game under `SmallGameAssets/` as a set of focused `MonoBehaviour` components in the `SmallGame` namespace:

- **Core loop** — `GameManager` tracks score (height climbed), best score, score multipliers, and game-over detection; `PlayerController` drives Rigidbody2D movement, screen-wrap, and color-matched bounce collisions (off-color platforms are passed through, not fatal).
- **Mechanics** — `ColorSwitcher` randomizes the player's color, `Platform` / `RocketPlatform` provide normal and boosted bounces, and `PlatformSpawner` generates an endless ascent with a sliding-window color-variety system that keeps the palette broad (configurable `colorWindow` and `minDistinctColors`).
- **Power-ups** — shield (`ShieldPickup`), jetpack (`JetpackPickup`), spring (`SpringPickup`), and multiplier coin (`MultiplierCoin`), all sharing a `PowerupPickup` base. Power-ups now **stack**: multipliers compound, durations add, and multiple springs compound the next bounce.
- **Presentation** — `CameraFollow`, `UIController` (migrated to **TextMeshPro**), and an `EffectsManager` / `OneShotParticles` system for bounce, switch, power-up, and death FX.
- **Persistence & tooling** — `HighScore` (PlayerPrefs) and an editor `SceneBuilder` to assemble the scene.

### Result (Color Jumper)

A fully playable, endless color-matching climber with power-ups, particle feedback, and persistent high scores — built on a decoupled, easy-to-extend component architecture. Off-color platforms pass the player through (fall-through), keeping the action fluid while still punishing wrong-color landings via missed bounces.

**Tech:** Unity (URP 2D), C#, Input System, Rigidbody2D, TextMeshPro.

---

## 2. Terrain Glacier — CUDA Terrain & Glaciation Simulator

A GPU heightmap generator that simulates climate, glaciation, and erosion, exporting RAW heightmaps ready to import into Unity terrains.

### Situation (Terrain)

Procedural heightmaps often look artificial. I wanted terrain shaped by *physical processes* — ice flow, weathering, and water — rather than raw noise, and fast enough to iterate on at high resolution.

### Task (Terrain)

Write a CUDA simulator that evolves a heightfield through coupled geological processes and exports a 16-bit RAW heightmap consumable by Unity.

### Action (Terrain)

Built `Terrain/terrain_glacier.cu`, a per-time-step pipeline of shared-memory–tiled stencil kernels (16×16 blocks + halo):

1. **Climate** — altitude/latitude temperature with ice-age cycles; snow accumulation, melt, rainfall, evaporation.
2. **Glaciation** — SIA-style two-pass glacial flow with basal scouring.
3. **Thermal + Wind** — fused talus creep with frost weathering and prevailing-wind erosion/deposition.
4. **Hydraulic** — Planchon-Darboux depression filling, D8 flow routing, drainage accumulation, stream-power erosion, and floodplain aggradation.

### Result (Terrain)

Produces `height_initial.raw` and `height_final.raw` (16-bit little-endian, `2^n + 1` sizes for Unity import). Ships with a prebuilt `terrain_glacier.exe` and a generated `height_final.raw`.

```bash
# Build
nvcc -O3 -use_fast_math -arch=sm_120 -o terrain_glacier terrain_glacier.cu

# Run: terrain_glacier [size] [steps] [seed]
terrain_glacier 1025 200 42
```

**Tech:** CUDA / C++, stencil computation, geomorphology simulation.

---

---

## 3. BlazePose → Unreal Live Link

Real-time webcam pose estimation streamed into Unreal Engine 5 via a custom Live Link source.

### Situation (BlazePose)

Driving character markers or rigs from live webcam footage inside UE5 requires bridging MediaPipe's Python output into Unreal's Live Link system — something no off-the-shelf plugin does at low latency.

### Task (BlazePose)

Build an end-to-end pipeline: Python captures webcam frames, runs BlazePose GHUM inference, filters the joints, and pushes them to a custom UE5 Live Link plugin over UDP — all fast enough for interactive use.

### Action (BlazePose)

- **Python sender** (`BlazePoseLiveLink/python/pose_sender.py`) — BlazePose GHUM Lite (model_complexity=0) for speed, vectorized One-Euro filter for smoothing, and a compact 563-byte binary UDP frame (`'UELP'` header + 34 × `float32[4]` joints in UE centimeter space).
- **UE5 plugin** (`BlazePoseLiveLink/Plugins/BlazePoseLiveLink/`) — `FBlazePoseLiveLinkSource` runs a dedicated `TPri_AboveNormal` receive thread, parses each frame, and pushes 34 `FTransform` subjects (`BlazePose_pelvis`, `BlazePose_nose`, …) into `ILiveLinkClient`. Rotations are identity; only world-space positions are transmitted (monocular depth yields reliable position, not rotation).
- Verified with a standalone `test_receiver.py` before any UE involvement.

### Result (BlazePose)

Achieves 30–60 fps end-to-end on a modern CPU with Lite model. Each Live Link subject can drive a marker Actor directly via a **Live Link Transform Controller** component, or feed an IK Rig as effector targets. The `--lock-root` flag zeroes unreliable monocular pelvis depth for cleaner results.

**Tech:** Python, MediaPipe BlazePose GHUM, Unreal Engine 5, Live Link, C++ UE plugin, UDP.

---

## Repository Layout

| Path                   | Project                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `SmallGameAssets/`     | Color Jumper Unity game (prefabs, scenes, scripts, settings)     |
| `Terrain/`             | CUDA terrain & glaciation simulator and RAW outputs              |
| `BlazePoseLiveLink/`   | BlazePose → UE5 Live Link (Python sender + C++ UE plugin)        |
