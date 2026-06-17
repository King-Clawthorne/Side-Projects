"""
wave_operator.py
================
Learned wave-optics operators: a neural-operator surrogate for diffraction and
interference.

The expensive forward model
---------------------------
A single free-space propagation is cheap (one FFT pair).  What is expensive in
differentiable wave optics is **multi-slice beam propagation** (split-step BPM)
through a thick / scattering medium: K sequential rounds of

        U <- propagate(U, dz)          (diffraction over a slice; angular spectrum)
        U <- U * exp(i * phi_k)        (phase accumulated through the slice)

K is large (here 64), the steps are sequential, and the reverse-mode graph is
K-deep -- which is exactly where the ~2x forward / ~4x backward cost over ray
optics comes from, and exactly what a surrogate exists to amortize.

The surrogate and its structural guarantee
------------------------------------------
For a *fixed medium*, coherent propagation is a **linear** operator on the
complex field U.  So we build the surrogate to be exactly complex-linear:

        U_out  =  ASM(U, z_total)            <- analytic free-space (physics prior)
               +  Correction(U)              <- learned, linear, band-limited

with NO bias and NO nonlinearity anywhere on the field path.  Consequences:

  * Superposition holds to machine precision:  S(a U1 + b U2) = a S(U1) + b S(U2).
    This is the structural property that lets the surrogate predict the
    INTERFERENCE of field combinations it never saw in training -- the cross
    term in |U1+U2|^2 is reconstructed correctly because the operator is linear
    in the field and intensity is the squared modulus.  A black-box nonlinear
    surrogate trained on intensities gives no such guarantee.
  * The operator lives in the angular-spectrum (Fourier) basis, which is the
    eigenbasis of free-space propagation -- so an FNO-style spectral operator is
    unusually well matched to the physics (the spectral renderer connection):
    free propagation is diagonal here, and only the medium-induced scattering
    has to be learned, as a low-angle (band-limited) correction.
  * Physics-initialised: the correction starts near zero, so the untrained
    surrogate already equals exact free-space propagation; training only learns
    the scattering correction (a Born-series-flavoured residual).

Verified / benchmarked below:
  (1) angular-spectrum propagator is unitary and round-trip exact   (~1e-15)
  (2) the surrogate is exactly linear in the field (superposition)  (~machine)
  (3) accuracy vs the K-slice solver on held-out fields (complex + intensity)
  (4) it predicts the interference of unseen superpositions
  (5) forward AND backward wall-clock speedup vs the differentiable solver

Scope / honesty: fixed medium (operator linear in field); the open extension is
a medium-CONDITIONED operator S(U, medium) for inverse design, which is
nonlinear in the medium and where amortisation pays off most.  Periodic FFT
propagation, scalar/paraxial-friendly sampling, moderate scattering.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
#  Optical setup                                                              #
# --------------------------------------------------------------------------- #
N = 64                       # grid
DX = 1e-6                    # pixel pitch  (1 um)
LAM = 0.5e-6                 # wavelength   (0.5 um)
Z_TOTAL = 150e-6             # propagation distance (150 um)
K_SLICES = 128               # split-step slices in the ground-truth solver (thick medium)


def transfer_function(dz: float, dtype=torch.complex64, device="cpu") -> torch.Tensor:
    """Band-limited angular-spectrum transfer function H(fx,fy;dz).
    Unit modulus on propagating modes, zero on evanescent ones (=> unitary on
    the band-limited propagating subspace)."""
    fx = torch.fft.fftfreq(N, d=DX, device=device, dtype=torch.float64)
    FX, FY = torch.meshgrid(fx, fx, indexing="ij")
    arg = 1.0 - (LAM * FX) ** 2 - (LAM * FY) ** 2
    prop = (arg >= 0).to(torch.float64)
    H = torch.exp(1j * 2 * torch.pi / LAM * dz * torch.sqrt(torch.clamp(arg, min=0.0)))
    return (H * prop).to(dtype)


def asm(U: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Angular-spectrum propagation by a precomputed transfer function H."""
    return torch.fft.ifft2(torch.fft.fft2(U, norm="ortho") * H, norm="ortho")


# --------------------------------------------------------------------------- #
#  Ground-truth expensive forward model: multi-slice BPM through a medium      #
# --------------------------------------------------------------------------- #
class MultiSliceBPM(nn.Module):
    """The expensive differentiable forward model.  Fixed random thick phase
    medium; K sequential propagate-then-modulate steps."""

    def __init__(self, dtype=torch.complex64, device="cpu", seed=0):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(seed)
        dz = Z_TOTAL / K_SLICES
        self.register_buffer("H", transfer_function(dz, dtype, device))
        # smooth, moderate random phase screens (the medium)
        raw = torch.randn(K_SLICES, N, N, generator=g, device=device)
        ker = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=device)
        ker = (ker / ker.sum()).view(1, 1, 3, 3)
        for _ in range(5):                       # blur for spatial smoothness
            raw = torch.nn.functional.conv2d(
                raw.unsqueeze(1), ker, padding=1).squeeze(1)
        # thick but WEAK + SMOOTH (turbulence/tissue/GRIN regime): the composite
        # operator stays low-complexity (learnable) while remaining expensive to
        # simulate.  Strong thick scattering -> near-random operator, not
        # compressible by any low-rank surrogate (noted in the writeup).
        screens = 0.05 * raw / raw.std()         # ~0.05 rad rms per slice
        self.register_buffer("phase", torch.exp(1j * screens.to(dtype)))

    def forward(self, U):                        # U: (B,N,N) complex
        for k in range(K_SLICES):
            U = asm(U, self.H)
            U = U * self.phase[k]
        return U


# --------------------------------------------------------------------------- #
#  Complex-linear building blocks (no bias, no activation => exactly linear)   #
# --------------------------------------------------------------------------- #
class CplxChannelMix(nn.Module):
    """Pointwise complex linear map over channels (1x1 conv, no bias)."""

    def __init__(self, c_in, c_out, scale=0.1):
        super().__init__()
        w = (torch.randn(c_out, c_in) + 1j * torch.randn(c_out, c_in)) * scale
        if c_in == c_out:                         # bias toward identity stream
            w = w + torch.eye(c_in, dtype=torch.cfloat)
        self.w = nn.Parameter(w)

    def forward(self, x):                         # x: (B,Cin,N,N) complex
        return torch.einsum("oi,bixy->boxy", self.w, x)


class CplxSpectralConv(nn.Module):
    """Linear complex spectral convolution on a centred low-frequency band of
    kc x kc modes -- the diffraction / mode-coupling operator.  Initialised to
    identity-per-mode so an untrained layer just band-limits and passes through."""

    def __init__(self, c_in, c_out, kc, scale=0.02):
        super().__init__()
        self.kc = kc
        w = 0.0j + torch.zeros(c_out, c_in, kc, kc, dtype=torch.cfloat)
        if c_in == c_out:
            w = w + torch.eye(c_in, dtype=torch.cfloat).view(c_in, c_in, 1, 1)
        w = w + scale * (torch.randn(c_out, c_in, kc, kc) +
                         1j * torch.randn(c_out, c_in, kc, kc))
        self.w = nn.Parameter(w)

    def forward(self, x):                         # x: (B,Cin,N,N) complex
        B, Cin, _, _ = x.shape
        Xs = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        c, k = N // 2, self.kc
        sl = slice(c - k // 2, c - k // 2 + k)
        block = Xs[..., sl, sl]
        out = torch.einsum("oixy,bixy->boxy", self.w, block)
        Ys = torch.zeros(B, self.w.shape[0], N, N, dtype=x.dtype, device=x.device)
        Ys[..., sl, sl] = out
        return torch.fft.ifft2(torch.fft.ifftshift(Ys, dim=(-2, -1)), norm="ortho")


class CplxSpatialMask(nn.Module):
    """Learnable per-pixel complex multiply -- an effective phase/amplitude
    screen.  This is the space-varying degree of freedom that lets the operator
    represent a (translation-breaking) medium; a stationary spectral conv alone
    cannot.  Linear in the field; initialised near unity."""

    def __init__(self, channels):
        super().__init__()
        m = torch.ones(channels, N, N, dtype=torch.cfloat)
        m = m + 0.01 * (torch.randn(channels, N, N) + 1j * torch.randn(channels, N, N))
        self.m = nn.Parameter(m)

    def forward(self, x):
        return x * self.m


class WaveOperatorSurrogate(nn.Module):
    """S(U) = ASM(U, z_total)  +  learned correction.

    The correction is a 'learned beam-propagation' stack: each layer applies a
    spectral conv (diffraction / mode coupling) then a learnable spatial mask
    (an effective phase screen).  Alternating Fourier-domain and space-domain
    LINEAR maps is exactly the split-step structure and can represent the
    space-varying medium with L << K effective slices.  Everything is complex-
    linear with no bias / no activation, so S is exactly linear in U and
    superposition (hence interference) is preserved by construction."""

    def __init__(self, channels=6, n_layers=3, kc=24, dtype=torch.complex64):
        super().__init__()
        self.register_buffer("H_total", transfer_function(Z_TOTAL, dtype))
        self.lift = CplxChannelMix(1, channels)
        self.spectral = nn.ModuleList(
            CplxSpectralConv(channels, channels, kc) for _ in range(n_layers))
        self.mask = nn.ModuleList(
            CplxSpatialMask(channels) for _ in range(n_layers))
        self.proj = CplxChannelMix(channels, 1, scale=1.0 / channels)

    def forward(self, U):                         # U: (B,N,N) complex
        base = asm(U, self.H_total)               # analytic free-space (physics prior)
        x = self.lift(U.unsqueeze(1))             # (B,C,N,N)
        for sp, mk in zip(self.spectral, self.mask):
            x = mk(sp(x))                         # propagate (spectral) then screen (mask)
        corr = self.proj(x).squeeze(1)            # learned scattering correction
        return base + corr


# --------------------------------------------------------------------------- #
#  Field generator: smooth band-limited complex fields (rich phase => speckle/ #
#  interference under propagation), kept inside the propagating band.          #
# --------------------------------------------------------------------------- #
def make_fields(batch, dtype=torch.complex64, device="cpu", gen=None):
    """Smooth low-frequency complex fields (Gaussian spectral envelope): rich
    phase -> speckle/interference under propagation, but concentrated in low
    angles so a band-limited correction can represent the scattering.  Kept
    inside the propagating band."""
    fx = torch.fft.fftfreq(N, d=DX, device=device, dtype=torch.float64)
    FX, FY = torch.meshgrid(fx, fx, indexing="ij")
    sigma = 8.0 / (N * DX)                                  # ~8-mode radius
    rdt = torch.float64 if dtype == torch.complex128 else torch.float32
    env = torch.exp(-(FX ** 2 + FY ** 2) / (2 * sigma ** 2)).to(rdt).to(device)
    sp = (torch.randn(batch, N, N, generator=gen, device=device) +
          1j * torch.randn(batch, N, N, generator=gen, device=device))
    U = torch.fft.ifft2(torch.fft.fft2(sp.to(dtype), norm="ortho") * env, norm="ortho")
    U = U / U.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()   # unit mean intensity
    return U


# --------------------------------------------------------------------------- #
#  Checks + benchmarks                                                        #
# --------------------------------------------------------------------------- #
def check_physics():
    print("=" * 72)
    print("1. ANGULAR-SPECTRUM PROPAGATOR: unitary + round-trip exact")
    print("=" * 72)
    H = transfer_function(Z_TOTAL, torch.complex128)
    Hm = transfer_function(-Z_TOTAL, torch.complex128)
    U = make_fields(2, torch.complex128)
    e_in = U.abs().pow(2).sum().item()
    e_out = asm(U, H).abs().pow(2).sum().item()
    rt = asm(asm(U, H), Hm)
    print(f"   energy conserved (unitary) : {e_in:.6f} -> {e_out:.6f} "
          f"(rel err {abs(e_in-e_out)/e_in:.1e})")
    print(f"   round-trip +z then -z err  : {(rt - U).abs().max().item():.2e}")
    print()


def check_superposition(model):
    print("=" * 72)
    print("2. SURROGATE IS EXACTLY LINEAR IN THE FIELD (superposition)")
    print("   the structural property that makes interference correct")
    print("=" * 72)
    with torch.no_grad():
        U = make_fields(2, device=next(model.parameters()).device)
        U1, U2 = U[0:1], U[1:2]
        a = torch.tensor(0.7 + 0.3j, dtype=U.dtype)
        b = torch.tensor(-0.4 + 0.9j, dtype=U.dtype)
        lhs = model(a * U1 + b * U2)
        rhs = a * model(U1) + b * model(U2)
        err = (lhs - rhs).abs().max().item() / rhs.abs().max().item()
    print(f"   ||S(aU1+bU2) - (a S(U1)+b S(U2))|| / ||.|| : {err:.2e}")
    print(f"   SUPERPOSITION HOLDS                        : {err < 1e-5}")
    print()


def train(model, solver, steps=400, device="cpu"):
    print("=" * 72)
    print(f"3. TRAIN surrogate to match the {K_SLICES}-slice solver")
    print("=" * 72)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, eps=1e-12)
    gen = torch.Generator(device=device).manual_seed(123)
    for step in range(steps):
        U = make_fields(8, gen=gen, device=device)
        with torch.no_grad():
            target = solver(U)
        pred = model(U)
        loss = (pred - target).abs().pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 80 == 0 or step == steps - 1:
            with torch.no_grad():
                Ut = make_fields(16, device=device)
                tt, pp = solver(Ut), model(Ut)
                base = asm(Ut, model.H_total)            # free-space only (no learning)
                fe = (pp - tt).norm() / tt.norm()
                fe0 = (base - tt).norm() / tt.norm()
                ie = ((pp.abs()**2 - tt.abs()**2).norm() /
                      (tt.abs()**2).norm())
            print(f"   step {step:3d}  loss {loss.item():.3e}   "
                  f"field-err {fe.item():.3f} (free-space baseline {fe0.item():.3f})"
                  f"  intensity-err {ie.item():.3f}")
    print()


def check_interference(model, solver):
    print("=" * 72)
    print("4. INTERFERENCE OF UNSEEN SUPERPOSITIONS")
    print("   true solver vs surrogate on |U1+U2|^2 (fields not seen as a pair)")
    print("=" * 72)
    with torch.no_grad():
        U = make_fields(2, device=next(model.parameters()).device)
        comb = U[0:1] + U[1:2]
        true_I = solver(comb).abs().pow(2)
        pred_I = model(comb).abs().pow(2)
        # incoherent (interference-blind) baseline: sum of individual intensities
        incoh_I = solver(U[0:1]).abs().pow(2) + solver(U[1:2]).abs().pow(2)
        rel = lambda x: ((x - true_I).norm() / true_I.norm()).item()
    print(f"   surrogate intensity error          : {rel(pred_I):.3f}")
    print(f"   incoherent-sum (no interference)   : {rel(incoh_I):.3f}  "
          f"(what you get if you ignore the cross term)")
    print()


def benchmark(model, solver, device="cpu"):
    print("=" * 72)
    print("5. SPEEDUP: differentiable solver vs surrogate (forward + backward)")
    print("=" * 72)

    def timeit(fn, n=20):
        fn()  # warmup
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
        return sorted(t)[len(t) // 2]

    U = make_fields(8, device=device)

    def fwd_solver():
        with torch.no_grad():
            solver(U)

    def fwd_model():
        with torch.no_grad():
            model(U)

    def bwd_solver():
        Ug = U.detach().clone().requires_grad_(True)
        solver(Ug).abs().pow(2).mean().backward()

    def bwd_model():
        Ug = U.detach().clone().requires_grad_(True)
        model(Ug).abs().pow(2).mean().backward()

    fs, fm = timeit(fwd_solver), timeit(fwd_model)
    bs, bm = timeit(bwd_solver, n=10), timeit(bwd_model, n=10)
    print(f"   forward : solver {fs*1e3:7.2f} ms   surrogate {fm*1e3:7.2f} ms   "
          f"speedup x{fs/fm:5.1f}")
    print(f"   backward: solver {bs*1e3:7.2f} ms   surrogate {bm*1e3:7.2f} ms   "
          f"speedup x{bs/bm:5.1f}")
    print(f"   (solver is {K_SLICES} sequential slices; ratios grow with K and N "
          f"and on GPU)")
    print()


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_physics()
    solver = MultiSliceBPM(device=device).to(device)
    model = WaveOperatorSurrogate(channels=6, n_layers=8, kc=24).to(device)
    check_superposition(model)                 # holds even before training
    train(model, solver, steps=800, device=device)
    check_interference(model, solver)
    benchmark(model, solver, device=device)
    print("=" * 72)
    print("SUMMARY: a complex-linear spectral operator surrogates multi-slice")
    print("wave propagation, conserves the superposition structure exactly (so")
    print("interference transfers to unseen field combinations), and accelerates")
    print("both the forward model and its gradient.")
    print("=" * 72)
