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

---

## 3. Adaptive Generalized-RMSE Regression

An MLP trained with a *learnable* loss exponent that adapts to the noise structure of the data, recovering the planted noise shape rather than assuming Gaussian errors.

### Situation (Adaptive Loss)

Standard MSE assumes Gaussian noise. Real data often has heavier tails. Simply making the norm exponent a free parameter fails because the l_α norm is monotonically non-increasing in α — the optimizer collapses α to infinity for free, regardless of fit quality.

### Task (Adaptive Loss)

Design an identifiable, learnable loss that adapts its exponent α and scale s to the data, and verify that it recovers a known planted noise shape that MSE cannot detect.

### Action (Adaptive Loss)

Implemented `AI/idea.py`:

- **`AdaptivePowerLoss`** — interprets |residual/s|^α as the NLL of a Generalized Gaussian distribution. The log-partition term `log Γ(1/α) − log α` acts as a barrier that penalizes degenerate α, making it identifiable. Both α (via sigmoid-bounded parameterization) and s (via softplus) are learned parameters.
- **`MLP`** — small 2-layer GELU network for the regression target.
- **Synthetic benchmark** — data generated with a planted GGD noise shape; the script trains both the adaptive model and a plain MSE baseline, then plots loss vs epoch, α vs epoch, and side-by-side fit curves.

### Result (Adaptive Loss)

With planted noise shape α = 1.0 (heavy-tailed), the adaptive model learns α ≈ 0.976 by epoch 500 and achieves ~30% lower MAE and RMSE on the clean signal than MSE.

| Model        | MAE    | RMSE   | Learned α |
| ------------ | ------ | ------ | --------- |
| Adaptive NLL | 0.0140 | 0.0178 | 0.976     |
| Baseline MSE | 0.0200 | 0.0254 | 2.00      |

```bash
python AI/idea.py
# outputs training_curves.png and results.png
```

**Tech:** PyTorch, CUDA, NumPy, Matplotlib, Generalized Gaussian distribution.

---

## Repository Layout

| Path               | Project                                                       |
| ------------------ | ------------------------------------------------------------- |
| `SmallGameAssets/` | Color Jumper Unity game (prefabs, scenes, scripts, settings)  |
| `Terrain/`         | CUDA terrain & glaciation simulator and RAW outputs           |
| `AI/`              | Adaptive generalized-RMSE regression with learnable α         |
