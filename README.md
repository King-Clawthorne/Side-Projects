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

- **Core loop** — `GameManager` tracks score (height climbed), best score, score multipliers, and game-over detection; `PlayerController` drives Rigidbody2D movement, screen-wrap, and color-matched bounce-or-die collisions.
- **Mechanics** — `ColorSwitcher` randomizes the player's color, `Platform` / `RocketPlatform` provide normal and boosted bounces, and `PlatformSpawner` generates an endless ascent.
- **Power-ups** — shield (`ShieldPickup`), jetpack (`JetpackPickup`), spring (`SpringPickup`), and multiplier coin (`MultiplierCoin`), all sharing a `PowerupPickup` base.
- **Presentation** — `CameraFollow`, `UIController`, and an `EffectsManager` / `OneShotParticles` system for bounce, switch, power-up, and death FX.
- **Persistence & tooling** — `HighScore` (PlayerPrefs) and an editor `SceneBuilder` to assemble the scene.

### Result (Color Jumper)

A fully playable, endless color-matching climber with power-ups, particle feedback, and persistent high scores — built on a decoupled, easy-to-extend component architecture.

**Tech:** Unity (URP 2D), C#, Input System, Rigidbody2D.

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

## 3. AI — Decoder-Only Transformer for Modular Arithmetic

A compact transformer trained to compute `(a ± b) % m` for floating-point operands, generating the answer digit by digit.

### Situation

I wanted to explore whether a small decoder-only transformer could learn exact arithmetic from scratch — not approximate it, but produce correct digit sequences for modular addition and subtraction over floats.

### Task

Build and train a character-level causal language model on randomly generated `(a op b) % m = result` sequences, optimized for a Linux + CUDA environment.

### Action

Implemented `AI/main.py`, a heavily fused transformer training stack:

- **Model** — 2-layer decoder-only transformer (D=128, 4 heads) with RoPE, QK-norm, QK-gain, SwiGLU FFN, LayerScale, and value residuals across layers.
- **Fused kernels** — Liger RMSNorm, SiLU-Mul, and fused linear cross-entropy (never materializes the full logit matrix); FlashAttention via PyTorch SDPA.
- **Optimizer** — Muon (momentum-orthogonalized Newton-Schulz) for 2D hidden weights + fused AdamW for embeddings, head, and norms; cosine LR schedule with linear warmup.
- **Data** — 600k randomly sampled examples rendered as token sequences; operands are integers or one-decimal floats up to 99, modulus 2–99.

### Result

The model learns to autoregressively decode correct modular arithmetic answers. Training runs entirely on-device (dataset kept GPU-resident) with `torch.compile(mode="max-autotune")`.

```bash
pip install liger-kernel triton
python AI/main.py
```

**Tech:** PyTorch, CUDA, Liger kernels, Triton, Muon optimizer.

---

## Repository Layout

| Path               | Project                                                       |
| ------------------ | ------------------------------------------------------------- |
| `SmallGameAssets/` | Color Jumper Unity game (prefabs, scenes, scripts, settings)  |
| `Terrain/`         | CUDA terrain & glaciation simulator and RAW outputs           |
| `AI/`              | Decoder-only transformer for modular arithmetic (PyTorch)     |
