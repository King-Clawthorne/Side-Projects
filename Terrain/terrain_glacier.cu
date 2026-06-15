// terrain_glacier.cu
// CUDA terrain generator + coupled glaciation / erosion simulator.
//
// Pipeline per time step:
//   1. Climate      - temperature from altitude + latitude + ice-age cycle;
//                     snow accumulation, melt, rainfall, evaporation
//   2. Glaciation   - glacial flow (SIA-style, two-pass), basal scouring
//   3. Thermal+Wind - fused: talus creep with frost weathering, and
//                     prevailing-wind erosion/deposition
//   4. Hydraulic    - depression filling (Planchon-Darboux), D8 routing,
//                     drainage accumulation, stream-power erosion,
//                     floodplain aggradation
//
// All stencil kernels use shared-memory tiling (16x16 blocks + halo).
//
// Build:   nvcc -O3 -use_fast_math -arch=sm_120 -o terrain_glacier terrain_glacier.cu
// Run:     terrain_glacier [size] [steps] [seed]   (size: Unity wants 2^n+1)
// Output:  height_initial.raw, height_final.raw (16-bit little-endian RAW)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define BLOCK 16

// ---------------------------------------------------------------- parameters

struct SimParams {
    int   N;              // grid side length
    float cellSize;       // meters per cell

    // climate
    float seaLevelTemp;   // deg C at elevation 0
    float lapseRate;      // deg C per meter of altitude
    float iceAgeAmp;      // amplitude of long-period temperature swing
    float iceAgePeriod;   // steps per ice-age cycle

    // glaciation
    float snowRate;       // m of ice-equivalent snowfall per step where T < 0
    float meltRate;       // m of melt per degree C above 0 per step
    float iceFlowRate;    // glacial creep coefficient
    float scourRate;      // bedrock abrasion per unit ice flux
    float maxIceSlope;    // ice surface relaxation threshold

    // thermal erosion
    float talusAngle;     // tangent of repose angle
    float thermalRate;    // fraction of excess moved per step
    float frostBoost;     // weathering multiplier near freezing line

    // hydraulic erosion (stream power)
    float rainRate;       // rain added to the precipitation proxy per step
    float evapRate;       // evaporation of the precipitation proxy
    float streamK;        // stream-power erodibility coefficient
    float streamDeposit;  // fraction of eroded rock re-deposited downstream
    float floodplainRate; // alluvial smoothing rate on depositional floors
    float floodThresh;    // drainage-area (cells) for "river" classification
    float fillEps;        // depression-fill outlet gradient (m per cell)
    float windSpan;       // wind sampling distance (cells)

    // wind erosion
    float windX, windY;   // prevailing wind direction (normalized)
    float windStrength;   // suspension capacity
    float windErode;      // surface lift rate
    float windDeposit;    // settling rate
};

__constant__ SimParams P;

// ------------------------------------------------------------------- helpers

__device__ __host__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline int idxOf(int x, int y) {
    x = clampi(x, 0, P.N - 1);
    y = clampi(y, 0, P.N - 1);
    return y * P.N + x;
}

// Note on shared-memory tiling: a tiled variant of every stencil kernel was
// implemented and benchmarked on sm_89 (RTX 4050). It was consistently SLOWER
// than plain global loads (2.54 vs 2.01 ms/step at 1025^2) -- Ada's L1/L2
// serve these 8-neighbor stencils well, and the cooperative halo loads plus
// __syncthreads() cost more than they save. Kernels therefore read globally.

__device__ __constant__ const int   c_dx[8]   = { 1,-1, 0, 0, 1, 1,-1,-1 };
__device__ __constant__ const int   c_dy[8]   = { 0, 0, 1,-1, 1,-1, 1,-1 };
__device__ __constant__ const float c_dist[8] =
    { 1.f,1.f,1.f,1.f, 1.41421f,1.41421f,1.41421f,1.41421f };

// ------------------------------------------------------- fractal noise (fBm)

__device__ float hash2(int x, int y, unsigned seed) {
    unsigned h = (unsigned)x * 374761393u + (unsigned)y * 668265263u + seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0xFFFFFF) / (float)0xFFFFFF; // [0,1]
}

__device__ float smoothNoise(float x, float y, unsigned seed) {
    int xi = (int)floorf(x), yi = (int)floorf(y);
    float fx = x - xi, fy = y - yi;
    // quintic fade
    float u = fx * fx * fx * (fx * (fx * 6.f - 15.f) + 10.f);
    float v = fy * fy * fy * (fy * (fy * 6.f - 15.f) + 10.f);
    float a = hash2(xi, yi, seed),     b = hash2(xi + 1, yi, seed);
    float c = hash2(xi, yi + 1, seed), d = hash2(xi + 1, yi + 1, seed);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

__global__ void k_generateTerrain(float *height, unsigned seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;

    float fx = (float)x / P.N, fy = (float)y / P.N;

    // domain warp for more organic ridges
    float wx = fx + 0.15f * smoothNoise(fx * 4.f, fy * 4.f, seed + 7u);
    float wy = fy + 0.15f * smoothNoise(fx * 4.f, fy * 4.f, seed + 13u);

    float amp = 1.f, freq = 4.f, sum = 0.f, norm = 0.f;
    for (int o = 0; o < 8; ++o) {
        float n = smoothNoise(wx * freq, wy * freq, seed + (unsigned)o * 101u);
        // ridged component on low octaves -> mountain crests
        if (o < 3) n = 1.f - fabsf(2.f * n - 1.f);
        sum  += n * amp;
        norm += amp;
        amp  *= 0.5f;
        freq *= 2.03f;
    }
    float h = sum / norm;
    h = powf(h, 1.6f);                 // sharpen peaks, broaden valleys
    height[y * P.N + x] = h * 1800.f;  // meters, up to ~1.8 km relief
}

// GPU-side step counter: with CUDA graphs the kernel arguments are baked in
// at capture time, so the time-dependent kernels read the step from device
// memory and this kernel advances it at the end of every simulated step.
__global__ void k_stepInc(int *step) { ++(*step); }

// Rock erodibility: multi-octave noise, uncorrelated with the terrain noise.
// >1 = soft rock (erodes fast), <1 = resistant rock.
__global__ void k_hardness(float *hard, unsigned seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;

    float fx = (float)x / P.N, fy = (float)y / P.N;
    float amp = 1.f, freq = 3.f, sum = 0.f, norm = 0.f;
    for (int o = 0; o < 4; ++o) {
        sum  += smoothNoise(fx * freq, fy * freq, seed + 777u + o * 53u) * amp;
        norm += amp;
        amp  *= 0.5f;
        freq *= 2.1f;
    }
    hard[y * P.N + x] = 0.45f + 1.1f * (sum / norm);  // 0.45 .. 1.55
}

// ---------------------------------------------------------------- climate

// Temperature at a cell, given altitude, latitude (y), and ice-age phase.
__device__ float temperatureAt(float h, int y, int step) {
    float lat = (float)y / P.N;                       // 0 = south, 1 = north
    float latCool = lat * 8.f;                        // colder northward
    float cycle = P.iceAgeAmp *
                  cosf(2.f * 3.14159265f * step / P.iceAgePeriod);
    return P.seaLevelTemp - h * P.lapseRate - latCool + cycle;
}

// Fused point-wise climate kernel: glacier mass balance (snow accumulation,
// melt feeding the water proxy) plus the water cycle (rainfall, evaporation,
// open map-edge boundaries).
__global__ void k_climate(const float *height, float *ice, float *water,
                          const int *step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;

    float hIce = ice[i];
    float w    = water[i];

    float T = temperatureAt(height[i] + hIce, y, *step);
    if (T < 0.f) {
        hIce += P.snowRate * fminf(-T / 10.f, 1.5f);  // colder -> more snow
    } else {
        float melt = fminf(hIce, P.meltRate * T);
        hIce -= melt;
        w    += melt;                                 // meltwater drives rivers
    }

    if (hIce < 0.5f) w += P.rainRate;                 // precip on ice is snow
    w *= (1.f - P.evapRate);
    if (x == 0 || y == 0 || x == P.N - 1 || y == P.N - 1) w = 0.f;

    ice[i]   = hIce;
    water[i] = w;
}

// ------------------------------------------------------------- glaciation

// Glacial flow, two-pass. Pass 1 computes each cell's outflow once and stores
// a per-cell flux coefficient plus the total leaving, so the gather pass can
// reconstruct any neighbor's flux toward us with one multiply instead of
// recomputing the neighbor's full 8-direction budget (the old 9x redundancy).
// Flux toward d:  coef[i] * max(surfDrop/dist - maxIceSlope, 0)
__global__ void k_glacierOutflow(const float *height, const float *ice,
                                 float *coef, float *outTot) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;

    float hIce = ice[i];
    if (hIce <= 0.05f) { coef[i] = 0.f; outTot[i] = 0.f; return; }

    float surfC = height[i] + hIce;
    // creep flux ~ thickness^2 * slope (SIA flavor, n softened)
    float base = P.iceFlowRate * hIce * hIce / P.cellSize;
    float total = 0.f;
    for (int d = 0; d < 8; ++d) {
        int j = idxOf(x + c_dx[d], y + c_dy[d]);
        float drop = (surfC - (height[j] + ice[j])) / c_dist[d];
        if (drop > P.maxIceSlope) total += base * (drop - P.maxIceSlope);
    }
    // cannot move more than (almost) all the ice
    float s = (total > hIce * 0.25f) ? hIce * 0.25f / total : 1.f;
    coef[i]   = base * s;
    outTot[i] = total * s;
}

__global__ void k_glacierGather(float *height, const float *iceIn,
                                float *iceOut, const float *coef,
                                const float *outTot) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;

    float surfC = height[i] + iceIn[i];
    float inflow = 0.f;
    for (int d = 0; d < 8; ++d) {
        int j = idxOf(x + c_dx[d], y + c_dy[d]);
        float nc = coef[j];
        if (nc <= 0.f) continue;   // lazy: only icy neighbors cost reads
        float drop = (height[j] + iceIn[j] - surfC) / c_dist[d];
        if (drop > P.maxIceSlope) inflow += nc * (drop - P.maxIceSlope);
    }

    float hIce = iceIn[i];
    iceOut[i] = hIce - outTot[i] + inflow;

    // basal scouring: moving ice grinds the bedrock
    float flux = outTot[i] + inflow;
    if (flux > 0.f && hIce > 0.5f)
        height[i] -= fminf(P.scourRate * flux * sqrtf(hIce), 0.05f);
}

// ------------------------------------------- thermal + wind erosion (fused)

// One pass over the height stencil computes both processes from the same
// pre-step surface: talus creep (amplified by freeze-thaw near 0 C, shielded
// under ice) and prevailing-wind lift/deposition (suppressed on wet or
// ice-covered ground). Halo of 2 covers the wind's +-2-cell sampling.
__global__ void k_thermalWind(const float *heightIn, float *heightOut,
                              const float *ice, const float *water,
                              const int *step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;

    float h = heightIn[i];
    float hIce = ice[i];

    // --- thermal (heat / frost) creep
    float T = temperatureAt(h, y, *step);
    float frost = (fabsf(T) < 4.f) ? P.frostBoost : 1.f;
    float maxDiff = P.talusAngle * P.cellSize;

    float delta = 0.f;
    for (int d = 0; d < 8; ++d) {
        float diff = h - heightIn[idxOf(x + c_dx[d], y + c_dy[d])];
        float lim = maxDiff * c_dist[d];
        if (diff > lim)        delta -= (diff - lim);  // we shed downhill
        else if (diff < -lim)  delta += (-diff - lim); // neighbor sheds to us
    }
    float shield = hIce > 1.f ? 0.2f : 1.f;  // ice cover shields the bedrock
    float change = delta * P.thermalRate * frost * shield * 0.125f;

    // --- wind: lift from exposed windward faces, deposit in wind shadows
    if (hIce < 0.1f && water[i] < 1e-3f) {
        int ox = (int)roundf(P.windX * P.windSpan);
        int oy = (int)roundf(P.windY * P.windSpan);
        float hUp   = heightIn[idxOf(x - ox, y - oy)];
        float hDown = heightIn[idxOf(x + ox, y + oy)];
        float exposure = h - hUp;     // positive: windward face
        float shadow   = hDown - h;   // positive: sheltered behind a ridge
        if (exposure > 0.f)
            change -= fminf(P.windErode * exposure * P.windStrength, 0.01f);
        if (shadow > 0.f)
            change += fminf(P.windDeposit * shadow * P.windStrength, 0.01f);
    }

    heightOut[i] = h + change;
}

// ------------------------------------- depression filling (lake outlets)

// Planchon-Darboux depression filling via FAST SWEEPING rather than Jacobi:
// in-place Gauss-Seidel scans in alternating raster directions. A thread
// walks serially along its row (or column) reading values it just updated,
// so information crosses the whole grid in ONE launch instead of one cell
// per launch -- the same alternating-scan scheme the original serial
// Planchon-Darboux algorithm uses. Four directional sweeps per step keep the
// warm-started surface converged as the terrain evolves. Cross-row reads are
// chaotic (other threads update concurrently), which is safe for this
// monotone fixed-point iteration.
__global__ void k_fillInit(const float *height, float *fill) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;
    bool edge = (x == 0 || y == 0 || x == P.N - 1 || y == P.N - 1);
    fill[i] = edge ? height[i] : height[i] + 1e6f;
}

__device__ inline void fillUpdateCell(const float *height, float *fill,
                                      int x, int y) {
    int i = y * P.N + x;
    if (x == 0 || y == 0 || x == P.N - 1 || y == P.N - 1) {
        fill[i] = height[i];
        return;
    }
    float m = 1e30f;
    for (int d = 0; d < 8; ++d) {
        int j = (y + c_dy[d]) * P.N + (x + c_dx[d]);
        m = fminf(m, fill[j] + P.fillEps * c_dist[d]);
    }
    fill[i] = fmaxf(height[i], m);
}

__global__ void k_fillSweepH(const float *height, float *fill, int xdir) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= P.N) return;
    for (int k = 0; k < P.N; ++k)
        fillUpdateCell(height, fill, xdir > 0 ? k : P.N - 1 - k, y);
}

__global__ void k_fillSweepV(const float *height, float *fill, int ydir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= P.N) return;
    for (int k = 0; k < P.N; ++k)
        fillUpdateCell(height, fill, x, ydir > 0 ? k : P.N - 1 - k);
}

// In-place chaotic Gauss-Seidel relaxation: one update per cell, fully
// parallel, reading whatever mix of old and updated neighbor values is
// present -- strictly better convergence than double-buffered Jacobi for
// this monotone fixed point, with no second buffer.
__global__ void k_fillRelax(const float *height, float *fill) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    fillUpdateCell(height, fill, x, y);
}

// ---------------------------------- hydraulic erosion (D8 + stream power)

// D8 steepest-descent flow direction on the depression-filled surface:
// index of the lowest of the 8 neighbors (slope-weighted), or -1 for flats.
__global__ void k_flowDir(const float *fill, int *dir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;

    float h = fill[i];
    int best = -1;
    float bestDrop = 0.f;
    for (int d = 0; d < 8; ++d) {
        int nx = x + c_dx[d], ny = y + c_dy[d];
        if (nx < 0 || nx >= P.N || ny < 0 || ny >= P.N) continue;
        float drop = (h - fill[ny * P.N + nx]) / c_dist[d];
        // multiplicative jitter breaks ties on near-flat (filled) surfaces,
        // otherwise D8 paths run in parallel stripes across lake floors
        drop *= 0.85f + 0.3f * hash2(x * 8 + d, y, 9173u);
        if (drop > bestDrop) { bestDrop = drop; best = ny * P.N + nx; }
    }
    dir[i] = best;
}

// Drainage accumulation, same fast-sweeping scheme: in-place Gauss-Seidel
// directional scans of  acc[i] = local input + sum of upstream neighbors.
// Flow chains aligned with the scan direction resolve in a single pass, so
// four alternating sweeps replace ~24 Jacobi iterations. Warm-started across
// simulation steps; stale values from re-routed paths decay as upstream
// corrections re-propagate.
__device__ inline void accUpdateCell(const int *dir, float *acc,
                                     const float *water, int x, int y) {
    int i = y * P.N + x;
    float a = 1.f + water[i] * 20.f;   // wetter cells contribute more runoff
    for (int d = 0; d < 8; ++d) {
        int nx = x + c_dx[d], ny = y + c_dy[d];
        if (nx < 0 || nx >= P.N || ny < 0 || ny >= P.N) continue;
        int j = ny * P.N + nx;
        if (dir[j] == i) a += acc[j];
    }
    acc[i] = a;
}

__global__ void k_accSweepH(const int *dir, float *acc, const float *water,
                            int xdir) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= P.N) return;
    for (int k = 0; k < P.N; ++k)
        accUpdateCell(dir, acc, water, xdir > 0 ? k : P.N - 1 - k, y);
}

__global__ void k_accSweepV(const int *dir, float *acc, const float *water,
                            int ydir) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= P.N) return;
    for (int k = 0; k < P.N; ++k)
        accUpdateCell(dir, acc, water, x, ydir > 0 ? k : P.N - 1 - k);
}

// In-place chaotic Gauss-Seidel relaxation (see k_fillRelax).
__global__ void k_accRelax(const int *dir, float *acc, const float *water) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    accUpdateCell(dir, acc, water, x, y);
}

// Stream-power law: E = K * A^0.5 * S. Eroded rock partially re-deposits at
// the downstream cell (alluviation), which flattens valley floors over time.
// Routing follows the filled surface, but erosion uses the real bedrock drop,
// so submerged lake floors see no incision; erodibility is modulated by the
// rock-hardness field. Atomic scatter, no tiling (one irregular neighbor).
__global__ void k_streamPower(float *height, const int *dir, const float *acc,
                              const float *ice, const float *hard) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;

    int j = dir[i];
    if (j < 0) return;                       // pit: deposition only
    float drop = height[i] - height[j];
    if (drop <= 0.f) return;

    int jx = j % P.N, jy = j / P.N;
    float d = (jx != x && jy != y) ? 1.41421f : 1.f;
    float S = drop / (d * P.cellSize);

    float E = P.streamK * sqrtf(acc[i]) * S * hard[i];
    if (ice[i] > 1.f) E *= 0.2f;             // thick ice shields the bed
    E = fminf(E, 0.3f * drop);               // stability: never invert a slope

    atomicAdd(&height[i], -E);
    atomicAdd(&height[j],  E * P.streamDeposit);
}

// Alluvial smoothing: depositional areas — pits and gentle, well-drained
// valley floors — diffuse toward the neighborhood mean, so sediment builds
// flat floodplains instead of lumpy fill.
__global__ void k_floodplain(const float *heightIn, float *heightOut,
                             const int *dir, const float *acc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.N || y >= P.N) return;
    int i = y * P.N + x;

    float h = heightIn[i];
    float mean = 0.f, maxDrop = 0.f;
    for (int d = 0; d < 8; ++d) {
        float nh = heightIn[idxOf(x + c_dx[d], y + c_dy[d])];
        mean += nh;
        float drop = h - nh;
        if (drop > maxDrop) maxDrop = drop;
    }
    mean *= 0.125f;
    float slope = maxDrop / P.cellSize;

    bool depositional = (dir[i] < 0) ||                 // pit / lake floor
                        (acc[i] > P.floodThresh && slope < 0.05f); // gentle reach
    heightOut[i] = depositional ? h + P.floodplainRate * (mean - h) : h;
}

// -------------------------------------------------------------- image output

// headerless 16-bit RAW, little-endian uint16, row-major from the top-left --
// the layout Unity/Unreal heightmap importers expect ("R16" / "16-bit RAW")
static void writeRAW(const char *path, const std::vector<float> &h, int N) {
    float lo = 1e30f, hi = -1e30f;
    for (float v : h) { lo = fminf(lo, v); hi = fmaxf(hi, v); }
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    std::vector<unsigned short> row(N);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x)
            row[x] = (unsigned short)(65535.f * (h[y * N + x] - lo) /
                                      (hi - lo + 1e-9f));
        fwrite(row.data(), sizeof(unsigned short), N, f);
    }
    fclose(f);
    printf("wrote %s  (16-bit RAW %dx%d, range %.1f .. %.1f m)\n",
           path, N, N, lo, hi);
}

// ----------------------------------------------------------------------- main

int main(int argc, char **argv) {
    int N        = argc > 1 ? atoi(argv[1]) : 1025;  // Unity wants 2^n + 1
    int steps    = argc > 2 ? atoi(argv[2]) : 3000;
    unsigned seed = argc > 3 ? (unsigned)atoi(argv[3]) : 1337u;

    // Resolution independence. All parameters below are tuned at REF_N; the
    // simulation models a FIXED physical domain (DOMAIN_M across), so a higher
    // N is finer sampling of the same world, not a bigger world. The base
    // terrain noise is already in normalized [0,1] coords, so its feature
    // scale is resolution-independent; what must scale are the parameters
    // expressed in cell units. With r = (N-1)/(REF_N-1) cells per reference
    // cell:
    //   cellSize ~ 1/r            (smaller cells keep the domain fixed)
    //   per-cell slope thresholds ~ 1/r   (same physical gradient)
    //   drainage-area thresholds  ~ r^2   (same physical area in cells)
    //   sampling distances        ~ r     (same physical distance)
    //   stream-power K            ~ 1/r   (E = K*sqrt(A_cells)*S stays physical
    //                                      since sqrt(A_cells) grows like r)
    const int   REF_N    = 1025;
    const float DOMAIN_M = 1.f * (REF_N - 1);   // 1 km reference world
    float r = (float)(N - 1) / (REF_N - 1);

    SimParams hp = {};
    hp.N = N;             hp.cellSize = DOMAIN_M / (N - 1);
    hp.seaLevelTemp = 22.f; hp.lapseRate = 0.0098f;
    hp.iceAgeAmp = 5.f;     hp.iceAgePeriod = 400.f;
    hp.snowRate = 0.06f;    hp.meltRate = 0.04f;
    hp.iceFlowRate = 0.004f; hp.scourRate = 0.002f; hp.maxIceSlope = 0.5f / r;
    hp.talusAngle = 0.6f;   hp.thermalRate = 0.25f; hp.frostBoost = 2.5f;
    hp.rainRate = 0.008f;   hp.evapRate = 0.12f;
    hp.streamK = 0.025f / r; hp.streamDeposit = 0.25f;
    hp.floodplainRate = 0.35f;
    hp.floodThresh = 60.f * r * r;
    hp.fillEps = 0.01f / r;
    hp.windSpan = 2.f * r;
    hp.windX = 0.7071f;     hp.windY = 0.7071f;
    hp.windStrength = 0.6f; hp.windErode = 0.004f;  hp.windDeposit = 0.006f;

    CUDA_CHECK(cudaMemcpyToSymbol(P, &hp, sizeof(SimParams)));

    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_height, *d_heightB, *d_ice, *d_iceB, *d_water,
          *d_acc, *d_fill, *d_hard, *d_coef, *d_outTot;
    int *d_dir;
    CUDA_CHECK(cudaMalloc(&d_height,  bytes));
    CUDA_CHECK(cudaMalloc(&d_heightB, bytes));
    CUDA_CHECK(cudaMalloc(&d_ice,     bytes));
    CUDA_CHECK(cudaMalloc(&d_iceB,    bytes));
    CUDA_CHECK(cudaMalloc(&d_water,   bytes));
    CUDA_CHECK(cudaMalloc(&d_acc,     bytes));
    CUDA_CHECK(cudaMalloc(&d_fill,    bytes));
    CUDA_CHECK(cudaMalloc(&d_hard,    bytes));
    CUDA_CHECK(cudaMalloc(&d_coef,    bytes));
    CUDA_CHECK(cudaMalloc(&d_outTot,  bytes));
    CUDA_CHECK(cudaMalloc(&d_dir,     (size_t)N * N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ice, 0, bytes));
    CUDA_CHECK(cudaMemset(d_water, 0, bytes));
    CUDA_CHECK(cudaMemset(d_acc, 0, bytes));

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    int block1D = 128;                          // for the row/column sweeps
    int grid1D  = (N + block1D - 1) / block1D;

    int *d_step;
    CUDA_CHECK(cudaMalloc(&d_step, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_step, 0, sizeof(int)));

    k_generateTerrain<<<grid, block>>>(d_height, seed);
    k_hardness<<<grid, block>>>(d_hard, seed);
    k_fillInit<<<grid, block>>>(d_height, d_fill);
    CUDA_CHECK(cudaGetLastError());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // one simulation step; everything on `stream` so it is graph-capturable
    auto launchStep = [&](bool fullSweep) {
        // 1: climate (mass balance + water cycle, fused)
        k_climate<<<grid, block, 0, stream>>>(d_height, d_ice, d_water, d_step);

        // 2: glacial flow (two-pass) + scour
        k_glacierOutflow<<<grid, block, 0, stream>>>(d_height, d_ice,
                                                     d_coef, d_outTot);
        k_glacierGather<<<grid, block, 0, stream>>>(d_height, d_ice, d_iceB,
                                                    d_coef, d_outTot);
        std::swap(d_ice, d_iceB);

        // 3: thermal + wind erosion (fused stencil)
        k_thermalWind<<<grid, block, 0, stream>>>(d_height, d_heightB,
                                                  d_ice, d_water, d_step);
        std::swap(d_height, d_heightB);

        // 4: hydraulic erosion (depression fill + D8 routing + stream power)
        // hybrid relaxation: cheap massively-parallel in-place Gauss-Seidel
        // every step maintains the warm-started surfaces; the exact (but
        // latency-bound) directional fast sweeps run periodically to resolve
        // long-range structure in one pass
        if (fullSweep) {
            k_fillSweepH<<<grid1D, block1D, 0, stream>>>(d_height, d_fill, +1);
            k_fillSweepV<<<grid1D, block1D, 0, stream>>>(d_height, d_fill, +1);
            k_fillSweepH<<<grid1D, block1D, 0, stream>>>(d_height, d_fill, -1);
            k_fillSweepV<<<grid1D, block1D, 0, stream>>>(d_height, d_fill, -1);
        } else {
            for (int it = 0; it < 3; ++it)
                k_fillRelax<<<grid, block, 0, stream>>>(d_height, d_fill);
        }
        k_flowDir<<<grid, block, 0, stream>>>(d_fill, d_dir);
        if (fullSweep) {
            k_accSweepH<<<grid1D, block1D, 0, stream>>>(d_dir, d_acc, d_water, +1);
            k_accSweepV<<<grid1D, block1D, 0, stream>>>(d_dir, d_acc, d_water, +1);
            k_accSweepH<<<grid1D, block1D, 0, stream>>>(d_dir, d_acc, d_water, -1);
            k_accSweepV<<<grid1D, block1D, 0, stream>>>(d_dir, d_acc, d_water, -1);
        } else {
            for (int it = 0; it < 3; ++it)
                k_accRelax<<<grid, block, 0, stream>>>(d_dir, d_acc, d_water);
        }
        k_streamPower<<<grid, block, 0, stream>>>(d_height, d_dir, d_acc,
                                                  d_ice, d_hard);
        k_floodplain<<<grid, block, 0, stream>>>(d_height, d_heightB,
                                                 d_dir, d_acc);
        std::swap(d_height, d_heightB);

        k_stepInc<<<1, 1, 0, stream>>>(d_step);   // advance device step counter
    };

    // CUDA graph: capture a 16-step macro block once, then replay it.
    // 16 steps = one full-sweep schedule period, and an even number of swaps
    // of every double buffer, so the pointers baked into the graph line up
    // with reality again at every block boundary.
    const int MACRO = 16;
    steps = ((steps + MACRO - 1) / MACRO) * MACRO;   // round up to whole blocks

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < MACRO; ++k) launchStep(k == 0);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

    printf("simulating %d steps on %dx%d grid (%d-step graph blocks)...\n",
           steps, N, N, MACRO);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    for (int g = 0; g < steps / MACRO; ++g)
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

    cudaEventRecord(t1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    printf("simulation took %.1f ms (%.2f ms/step)\n", ms, ms / steps);

    std::vector<float> host(N * N);
    CUDA_CHECK(cudaMemcpy(host.data(), d_height, bytes, cudaMemcpyDeviceToHost));
    writeRAW("height_final.raw", host, N);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_step);
    cudaFree(d_height);  cudaFree(d_heightB);
    cudaFree(d_ice);     cudaFree(d_iceB);
    cudaFree(d_water);   cudaFree(d_acc);
    cudaFree(d_fill);    cudaFree(d_hard);
    cudaFree(d_coef);    cudaFree(d_outTot);
    cudaFree(d_dir);
    return 0;
}
