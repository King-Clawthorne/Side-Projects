"""
polariton.py
============
Group-equivariant polarization networks.

Core thesis
-----------
Polarization state space is Lorentzian, not Euclidean.

  * A Stokes vector  S = (S0, S1, S2, S3)  lives in Minkowski space with
    metric  eta = diag(+1, -1, -1, -1).  The physical degree of polarization
    DOP = sqrt(S1^2+S2^2+S3^2)/S0 <= 1  is exactly the statement that S lies
    in the forward light cone (timelike/null, S0 >= 0).

  * Every *non-depolarizing* optical element (a Jones matrix T acting on the
    coherency matrix as  J -> T J T^dagger) induces a Mueller matrix that is a
    proper orthochronous Lorentz transformation  M in SO+(1,3).

  * The double cover  SL(2,C) -> SO+(1,3)  is the spinor map.  In the language
    of geometric algebra (Algebra of Physical Space, Cl(3,0)), the Stokes
    4-vector is a paravector  s = S0 + S1 e1 + S2 e2 + S3 e3,  a Lorentz
    transform is a rotor  L  with  L Lbar = 1  acting by the sandwich
    s -> L s L^dagger, and the polar decomposition  L = boost * rotation  is
    physically  diattenuator * retarder.  Retarders are rotations of the
    Poincare sphere (rotor exp of a space-space bivector, L^dagger = L^-1);
    diattenuators are boosts (rotor exp of a time-space bivector, L^dagger = L).

So a network that "respects polarization the way SE(3)-nets respect rigid
motion" must be equivariant to  SO+(1,3) / SL(2,C),  not merely regress Stokes
components as opaque features.  This file builds such a network and *verifies*
the equivariance numerically.

Relation to prior art (for the paper framing): Lorentz-equivariant networks
exist in particle physics (Bogatskiy et al. 2020; LorentzNet, Gong et al.
2022) and the general machinery is Clifford-group-equivariant nets / GATr
(Ruhe et al. 2023; Brehmer et al. 2023).  The block below is a LorentzNet-style
equivariant block.  The contribution this scaffolds is the *transfer of that
machinery to polarization optics*, where the group is not an abstract symmetry
but the literal action of optical elements (Mueller calculus = SO+(1,3)).
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Minkowski geometry                                                          #
# --------------------------------------------------------------------------- #

ETA = torch.tensor([1.0, -1.0, -1.0, -1.0])  # metric signature (+,-,-,-)


def mink(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product <x,y> = x0 y0 - x1 y1 - x2 y2 - x3 y3.

    Operates on the last axis; broadcasts over all leading axes.
    Returns the contracted tensor with the last (size-4) axis removed.
    """
    eta = ETA.to(x.dtype).to(x.device)
    return (x * eta * y).sum(dim=-1)


# --------------------------------------------------------------------------- #
#  Polarization physics: Jones / coherency / Stokes / Mueller                  #
#                                                                              #
#  Stokes-ordered Pauli basis (tau_0, tau_1, tau_2, tau_3) = (I, sigma_z,      #
#  sigma_x, sigma_y) so that  J = 1/2 * sum_mu S_mu tau_mu  and                #
#  S_mu = tr(J tau_mu).  This is the standard optics convention:              #
#      S0 = Jxx + Jyy,  S1 = Jxx - Jyy,  S2 = Jxy + Jyx,  S3 = i(Jxy - Jyx).   #
# --------------------------------------------------------------------------- #

_I = np.eye(2, dtype=np.complex128)
_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
TAU = np.stack([_I, _SZ, _SX, _SY])  # shape (4, 2, 2), Stokes order


def coherency_from_stokes(S: np.ndarray) -> np.ndarray:
    """Stokes 4-vector -> 2x2 Hermitian coherency matrix J = 1/2 sum S_mu tau_mu."""
    return 0.5 * np.einsum("...m,mij->...ij", S.astype(np.complex128), TAU)


def stokes_from_coherency(J: np.ndarray) -> np.ndarray:
    """Coherency matrix -> Stokes 4-vector, S_mu = tr(J tau_mu)."""
    return np.real(np.einsum("mij,...ji->...m", TAU, J))


def jones_to_mueller(T: np.ndarray) -> np.ndarray:
    """Mueller matrix induced by a Jones matrix T via J -> T J T^dagger.

    M_{mu nu} = 1/2 tr( tau_mu  T tau_nu T^dagger ).
    If T in SL(2,C) (det T = 1) then M in SO+(1,3) exactly.
    """
    Td = T.conj().T
    M = np.empty((4, 4))
    for mu in range(4):
        for nu in range(4):
            M[mu, nu] = 0.5 * np.real(np.trace(TAU[mu] @ T @ TAU[nu] @ Td))
    return M


# ---- generators of SL(2,C): retarders (rotations) and diattenuators (boosts) ----

def retarder_jones(axis: np.ndarray, phase: float) -> np.ndarray:
    """Unitary Jones matrix exp(-i phase/2 * n.sigma): a rotation of the
    Poincare sphere by `phase` about Stokes-space axis `axis` (unit 3-vector).
    This is the rotor exp of a *space-space* bivector;  T^dagger = T^-1."""
    n = axis / np.linalg.norm(axis)
    nsig = n[0] * _SZ + n[1] * _SX + n[2] * _SY  # Stokes-ordered
    return np.cos(phase / 2) * _I - 1j * np.sin(phase / 2) * nsig


def diattenuator_jones(axis: np.ndarray, rapidity: float) -> np.ndarray:
    """Hermitian Jones matrix exp(rapidity/2 * m.sigma): a Lorentz *boost*
    along Stokes-space axis `axis`.  Physically a partial polarizer /
    diattenuator;  the rotor exp of a *time-space* bivector;  T^dagger = T."""
    m = axis / np.linalg.norm(axis)
    msig = m[0] * _SZ + m[1] * _SX + m[2] * _SY
    return np.cosh(rapidity / 2) * _I + np.sinh(rapidity / 2) * msig


def random_lorentz(max_rapidity: float = 0.5, rng: np.random.Generator | None = None
                   ) -> np.ndarray:
    """Random proper orthochronous Lorentz matrix in SO+(1,3), built the
    GA-native way: sample a random retarder (rotation) and a random
    diattenuator (boost), compose the SL(2,C) element, push through the
    spinor map.  Polar form  L = boost * rotation  is guaranteed in SL(2,C)."""
    rng = np.random.default_rng() if rng is None else rng
    n_axis = rng.normal(size=3)
    m_axis = rng.normal(size=3)
    phase = rng.uniform(0, 2 * np.pi)
    rapidity = rng.uniform(-max_rapidity, max_rapidity)
    T = diattenuator_jones(m_axis, rapidity) @ retarder_jones(n_axis, phase)
    T = T / np.sqrt(np.linalg.det(T))  # enforce det = 1 (numerical hygiene)
    return jones_to_mueller(T)


def is_lorentz(M: np.ndarray, tol: float = 1e-9) -> bool:
    """Check M^T eta M = eta (Lorentz), det M > 0 (proper), M00 > 0 (orthochronous)."""
    eta = np.diag([1.0, -1.0, -1.0, -1.0])
    ortho = np.allclose(M.T @ eta @ M, eta, atol=tol)
    return bool(ortho and np.linalg.det(M) > 0 and M[0, 0] > 0)


def dop(S: np.ndarray) -> np.ndarray:
    """Degree of polarization (rotation-invariant, NOT boost-invariant)."""
    return np.linalg.norm(S[..., 1:], axis=-1) / np.clip(S[..., 0], 1e-12, None)


# --------------------------------------------------------------------------- #
#  Depolarization and the Mueller coherency (Cloude) matrix                    #
#                                                                              #
#  CAVEAT this addresses: everything above assumes *non-depolarizing* optics,  #
#  where a Jones T in SL(2,C) maps to a Mueller M in SO+(1,3) and Stokes stay  #
#  ON the light cone.  Real elements depolarize: they map an ensemble of pure  #
#  states to a partially polarized mixture, pushing Stokes strictly INSIDE the #
#  cone (DOP < 1).  The Mueller matrix is then a convex sum of non-depolarizing#
#  ones and is NOT in SO+(1,3); physical Mueller matrices form a *semigroup*   #
#  (closed under composition, but a depolarizer has no inverse), so the network#
#  symmetry is the non-depolarizing subgroup SO+(1,3), not the full semigroup. #
#                                                                              #
#  The object that linearises depolarization is the 4x4 Hermitian coherency    #
#  (Cloude) matrix  H(M) = 1/4 sum_ij M_ij (tau_i (x) conj(tau_j)).  It is the #
#  Mueller operator viewed in the (1/2,1/2) (x) conjugate rep of SL(2,C): each #
#  open spinor index carries a fundamental/conjugate-fundamental factor.  H is  #
#  PSD for any physical M; rank(H)=1 exactly for non-depolarizing M, and its   #
#  eigenvalue spread is the depolarization (Cloude entropy).  Under an OUTPUT   #
#  non-depolarizing transform M -> G M  (G = jones_to_mueller(U)), H undergoes  #
#  the exact congruence  H -> (U (x) I) H (U (x) I)^dagger  -- equivariance to  #
#  the subgroup, on a higher-rank object than a single Stokes 4-vector.        #
# --------------------------------------------------------------------------- #

def depolarizer_mueller(diag: tuple[float, float, float]) -> np.ndarray:
    """Ideal diagonal depolarizer M = diag(1, a, b, c) with |a|,|b|,|c| <= 1.
    Each factor shrinks one Stokes component toward the cone interior (DOP<1)."""
    a, b, c = diag
    return np.diag([1.0, a, b, c]).astype(float)


def mueller_coherency(M: np.ndarray) -> np.ndarray:
    """Cloude coherency matrix H(M) = 1/4 sum_ij M_ij (tau_i (x) conj(tau_j)).
    Hermitian; PSD for physical M; rank 1 iff M is non-depolarizing."""
    H = np.zeros((4, 4), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            H += M[i, j] * np.kron(TAU[i], TAU[j].conj())
    return H / 4.0


def polarimetric_entropy(M: np.ndarray) -> float:
    """Cloude depolarization entropy in [0,1]: 0 for non-depolarizing (rank-1
    coherency), ->1 for a total depolarizer.  H = -sum p_k log_4 p_k over the
    normalised eigenvalues p_k of the coherency matrix."""
    w = np.clip(np.linalg.eigvalsh(mueller_coherency(M)).real, 0.0, None)
    p = w / (w.sum() + 1e-15)
    p = p[p > 1e-15]
    return float(-(p * (np.log(p) / np.log(4))).sum())


# --------------------------------------------------------------------------- #
#  so(1,3) generators in the geometric-algebra (bivector) picture             #
#  6 bivectors: 3 boosts B_i (time-space, symmetric) + 3 rotations J_i        #
#  (space-space, antisymmetric).  expm(sum c_k G_k) is a Lorentz matrix.      #
# --------------------------------------------------------------------------- #

def so13_generators() -> np.ndarray:
    """Return the 6 generators stacked as (6, 4, 4): [B1,B2,B3, J1,J2,J3]."""
    G = np.zeros((6, 4, 4))
    for i in range(3):  # boosts: mix time axis 0 with space axis i+1
        G[i, 0, i + 1] = 1.0
        G[i, i + 1, 0] = 1.0
    rot_pairs = [(2, 3), (3, 1), (1, 2)]  # J1,J2,J3
    for k, (a, b) in enumerate(rot_pairs):
        G[3 + k, a, b] = -1.0
        G[3 + k, b, a] = 1.0
    return G


# --------------------------------------------------------------------------- #
#  Equivariant network                                                         #
#                                                                              #
#  State carried by every block:                                              #
#    V : (B, N, C, 4)  C four-vector channels per node  (transform as g.V)     #
#    H : (B, N, S)     S scalar channels per node       (Lorentz-invariant)    #
#                                                                              #
#  Allowed equivariant operations (and why they are equivariant):             #
#    1. Minkowski grams <V_c, V_d>  -> scalars     (g^T eta g = eta)           #
#    2. V'_a = sum_c W_{ac}(invariants) V_c         (scalar-weighted mixing)   #
#    3. H'  = MLP(H, invariants)                    (functions of invariants)  #
#    4. pairwise messages weighted by invariants of (V_i, V_j, H_i, H_j),      #
#       acting on differences (V_j - V_i)           (sum over j is perm-inv.)  #
# --------------------------------------------------------------------------- #


def grams(V: torch.Tensor) -> torch.Tensor:
    """All pairwise Minkowski inner products of the C channels.
    V: (..., C, 4) -> (..., C*C) flattened invariant features."""
    eta = ETA.to(V.dtype).to(V.device)
    g = torch.einsum("...ci,i,...di->...cd", V, eta, V)  # (..., C, C)
    return g.flatten(start_dim=-2)


class LorentzEquivariantBlock(nn.Module):
    """A LorentzNet-style equivariant message-passing block over a set of nodes."""

    def __init__(self, c_vec: int, s_scal: int, hidden: int = 64):
        super().__init__()
        self.c_vec, self.s_scal = c_vec, s_scal
        n_gram = c_vec * c_vec
        # edge network: consumes invariants of (i,j) -> scalar message + vector weight
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * s_scal + 2 * n_gram + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.edge_scal = nn.Linear(hidden, s_scal)        # invariant message
        self.edge_vecw = nn.Linear(hidden, c_vec)         # weight on (V_j - V_i) per channel
        # node update on scalars
        self.node_mlp = nn.Sequential(
            nn.Linear(s_scal + s_scal + n_gram, hidden), nn.SiLU(),
            nn.Linear(hidden, s_scal),
        )
        # self vector mixing weights from invariants
        self.vec_mix = nn.Sequential(
            nn.Linear(s_scal + n_gram, hidden), nn.SiLU(),
            nn.Linear(hidden, c_vec * c_vec),
        )

    def forward(self, V: torch.Tensor, H: torch.Tensor):
        B, N, C, _ = V.shape
        eta = ETA.to(V.dtype).to(V.device)

        gV = grams(V)                                            # (B,N,C*C) invariant
        # ---- pairwise invariants ----
        # pairwise cross-grams <V_i,c, V_j,c> and Minkowski sq of differences
        cross = torch.einsum("bick,k,bjck->bijc", V, eta, V)     # (B,N,N,C)
        diff = V.unsqueeze(2) - V.unsqueeze(1)                   # (B,N,N,C,4) equivariant
        dist2 = torch.einsum("bijck,k,bijck->bijc", diff, eta, diff)  # (B,N,N,C) invariant

        Hi = H.unsqueeze(2).expand(B, N, N, self.s_scal)
        Hj = H.unsqueeze(1).expand(B, N, N, self.s_scal)
        gVi = gV.unsqueeze(2).expand(B, N, N, C * C)
        gVj = gV.unsqueeze(1).expand(B, N, N, C * C)
        # summarize per-channel pairwise invariants into 2 scalars to keep edge input small
        edge_in = torch.cat([Hi, Hj, gVi, gVj,
                             cross.mean(-1, keepdim=True),
                             dist2.mean(-1, keepdim=True)], dim=-1)
        e = self.edge_mlp(edge_in)                              # (B,N,N,hidden)

        # ---- vector messages: invariant weights times equivariant differences ----
        w = self.edge_vecw(e)                                  # (B,N,N,C)
        # gate by inverse "distance" to damp far/large-norm contributions (still invariant)
        gate = torch.exp(-dist2.abs() / (1.0 + dist2.abs()))   # (B,N,N,C), in (0,1]
        vmsg = (w * gate).unsqueeze(-1) * diff                 # (B,N,N,C,4) equivariant
        V_msg = vmsg.sum(dim=2)                                # (B,N,C,4) sum over j: perm-inv

        # ---- scalar messages ----
        smsg = self.edge_scal(e).sum(dim=2)                    # (B,N,s) invariant

        # ---- self vector mixing (equivariant linear recombination of channels) ----
        mix = self.vec_mix(torch.cat([H, gV], dim=-1))         # (B,N,C*C)
        mix = mix.view(B, N, C, C)
        V_self = torch.einsum("bnac,bnck->bnak", mix, V)       # (B,N,C,4) equivariant

        V_new = V + V_self + V_msg
        H_new = H + self.node_mlp(torch.cat([H, smsg, gV], dim=-1))
        return V_new, H_new


class PolarizationEquivariantNet(nn.Module):
    """Takes a set of Stokes 4-vectors per sample; outputs an SO+(1,3)-invariant
    scalar head and an SO+(1,3)-equivariant 4-vector head."""

    def __init__(self, c_vec: int = 8, s_scal: int = 16, n_blocks: int = 3,
                 hidden: int = 64, n_inv_out: int = 1):
        super().__init__()
        self.c_vec = c_vec
        # lift the single input Stokes vector per node into C vector channels by
        # scalar-weighting it (the only equivariant linear map from 1 -> C vectors
        # uses invariant scalars; here we use learned constants, which are invariants).
        self.lift = nn.Parameter(torch.randn(c_vec) * 0.5)
        # initial scalars are invariants of the input (its own Minkowski norm)
        self.h0 = nn.Sequential(nn.Linear(1, s_scal), nn.SiLU(),
                                nn.Linear(s_scal, s_scal))
        self.blocks = nn.ModuleList(
            [LorentzEquivariantBlock(c_vec, s_scal, hidden) for _ in range(n_blocks)])
        self.inv_head = nn.Sequential(nn.Linear(s_scal, hidden), nn.SiLU(),
                                      nn.Linear(hidden, n_inv_out))
        self.equ_head = nn.Linear(c_vec, 1, bias=False)  # invariant-weighted vector readout

    def forward(self, S: torch.Tensor):
        """S: (B, N, 4) batch of Stokes-vector sets.
        Returns (inv_out: (B, n_inv_out), equ_out: (B, 4))."""
        B, N, _ = S.shape
        V = S.unsqueeze(2) * self.lift.view(1, 1, self.c_vec, 1)  # (B,N,C,4) equivariant
        self_norm = mink(S, S).unsqueeze(-1)                      # (B,N,1) invariant
        H = self.h0(self_norm)                                    # (B,N,s)
        for blk in self.blocks:
            V, H = blk(V, H)
        # invariant output: pool scalars over the set (sum is permutation-invariant)
        inv_out = self.inv_head(H.sum(dim=1))                     # (B, n_inv_out)
        # equivariant output: invariant-weighted sum of vector channels, pooled over set
        equ_out = self.equ_head(V.sum(dim=1).transpose(1, 2)).squeeze(-1)  # (B,4)
        return inv_out, equ_out


# --------------------------------------------------------------------------- #
#  Verification + demo                                                         #
# --------------------------------------------------------------------------- #

def random_stokes_set(batch: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random physically valid Stokes sets (forward light cone, DOP<=1)."""
    intensity = rng.uniform(0.5, 2.0, size=(batch, n))
    p = rng.uniform(0.0, 1.0, size=(batch, n))            # degree of polarization
    direction = rng.normal(size=(batch, n, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    S = np.empty((batch, n, 4))
    S[..., 0] = intensity
    S[..., 1:] = (intensity * p)[..., None] * direction
    return S


def verify_physics(rng):
    print("=" * 70)
    print("PHYSICS CHECKS")
    print("=" * 70)
    # Mueller of an SL(2,C) element is a Lorentz transform
    M = random_lorentz(0.8, rng)
    print(f"random Mueller matrix is in SO+(1,3):           {is_lorentz(M)}")
    # round-trip Stokes <-> coherency
    S = random_stokes_set(1, 1, rng)[0, 0]
    S_rt = stokes_from_coherency(coherency_from_stokes(S))
    print(f"Stokes <-> coherency round-trip max err:        {np.abs(S - S_rt).max():.2e}")
    # a retarder preserves DOP (pure rotation); a diattenuator changes it (boost)
    Sset = random_stokes_set(1, 1, rng)[0]
    Mr = jones_to_mueller(retarder_jones(np.array([0.3, 1.0, -0.5]), 1.1))
    Mb = jones_to_mueller(diattenuator_jones(np.array([1.0, 0.0, 0.0]), 0.6))
    dop_in = dop(Sset)[0]
    dop_rot = dop((Mr @ Sset.T).T)[0]
    dop_boost = dop((Mb @ Sset.T).T)[0]
    print(f"DOP: in={dop_in:.4f}  after retarder={dop_rot:.4f} (preserved) "
          f" after diattenuator={dop_boost:.4f} (changed)")
    # Minkowski norm S.S is the true invariant under any non-depolarizing element
    inv_in = (S * np.array([1, -1, -1, -1])).dot(S)
    inv_out = (M @ S * np.array([1, -1, -1, -1])).dot(M @ S)
    print(f"Minkowski norm S.S invariant under boost+rot:   "
          f"{inv_in:.6f} -> {inv_out:.6f}  (err {abs(inv_in-inv_out):.2e})")
    print()


def verify_equivariance(rng, dtype=torch.float64):
    print("=" * 70)
    print("EQUIVARIANCE TEST  (the proof of concept)")
    print("=" * 70)
    torch.manual_seed(0)
    model = PolarizationEquivariantNet(c_vec=8, s_scal=16, n_blocks=3,
                                       n_inv_out=1).to(dtype)
    model.eval()

    B, N = 4, 6
    S = random_stokes_set(B, N, rng)
    St = torch.tensor(S, dtype=dtype)

    # a single global Lorentz transform applied to every Stokes vector
    g = random_lorentz(max_rapidity=0.4, rng=rng)
    gt = torch.tensor(g, dtype=dtype)
    Sg = torch.einsum("uv,bnv->bnu", gt, St)

    with torch.no_grad():
        inv1, equ1 = model(St)
        inv2, equ2 = model(Sg)

    # invariant head must be unchanged; equivariant head must transform by g
    inv_err = (inv1 - inv2).abs().max().item()
    equ_target = torch.einsum("uv,bv->bu", gt, equ1)
    equ_err = (equ_target - equ2).abs().max().item()

    print(f"applied global Lorentz g in SO+(1,3):           {is_lorentz(g)}")
    print(f"INVARIANT head  | f(g.S) - f(S) |_max:          {inv_err:.3e}")
    print(f"EQUIVARIANT head| g.f(S) - f(g.S) |_max:        {equ_err:.3e}")
    ok = inv_err < 1e-8 and equ_err < 1e-8
    print(f"EQUIVARIANT TO THE POLARIZATION GROUP:          {ok}")
    print()
    return ok


def tiny_training_demo(rng, dtype=torch.float32):
    """Sanity check that the equivariant model can fit equivariant/invariant
    targets.  Target_inv = sum of Minkowski norms (invariant);
    Target_equ = Minkowski-weighted vector sum (equivariant)."""
    print("=" * 70)
    print("TINY TRAINING DEMO  (fits invariant + equivariant targets)")
    print("=" * 70)
    torch.manual_seed(1)
    model = PolarizationEquivariantNet(c_vec=8, s_scal=16, n_blocks=2,
                                       n_inv_out=1).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    B, N = 64, 6
    for step in range(150):
        S = torch.tensor(random_stokes_set(B, N, rng), dtype=dtype)
        tgt_inv = mink(S, S).sum(dim=1, keepdim=True)          # invariant target
        tgt_equ = S.sum(dim=1)                                 # equivariant target (4-vec)
        inv, equ = model(S)
        loss = ((inv - tgt_inv) ** 2).mean() + ((equ - tgt_equ) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 30 == 0 or step == 149:
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    print()


# --------------------------------------------------------------------------- #
#  Polarization-camera forward model (division-of-focal-plane sensor)          #
#                                                                              #
#  A DoFP polarimeter (e.g. Sony IMX250MZR) tiles a micro-polarizer array in   #
#  2x2 super-pixels, each cell a linear analyzer at one of four angles.  Under  #
#  Malus' law the intensity transmitted by an analyzer at angle theta is       #
#      I(theta) = 1/2 ( S0 + S1 cos 2theta + S2 sin 2theta ).                  #
#  A linear DoFP sensor measures only S0,S1,S2 (no quarter-wave plate -> S3=0),#
#  so the recoverable state lives on the 2D equatorial sub-cone of the Poincare#
#  sphere.  We keep the full 4-vector machinery; S3 simply stays ~0 and the    #
#  network's Lorentz-equivariance restricts to the relevant subgroup.          #
#                                                                              #
#  Super-pixel layout (row-major within each 2x2 block):                       #
#      [ 90   45 ]                                                             #
#      [135    0 ]   (degrees)                                                 #
# --------------------------------------------------------------------------- #

# analyzer angles per cell of the 2x2 super-pixel, in radians, row-major
_DOFP_ANGLES_DEG = np.array([[90.0, 45.0], [135.0, 0.0]])
_DOFP_ANGLES = np.deg2rad(_DOFP_ANGLES_DEG)


def stokes_to_mosaic(S_img: np.ndarray, read_noise: float = 0.0,
                     photons: float = 0.0,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Forward model: clean per-pixel Stokes image -> raw DoFP mosaic.

    S_img : (H, W, 4) Stokes per pixel (S3 is ignored by a linear DoFP sensor).
    Returns a raw intensity mosaic of shape (2H, 2W): every Stokes pixel is
    expanded into its 2x2 block of analyzer intensities I(theta) = 1/2
    (S0 + S1 cos2theta + S2 sin2theta).

    Noise (realistic DoFP regime):
      photons > 0   : Poisson shot noise at `photons` full-well electrons per
                      unit intensity (signal-dependent; the dominant noise).
      read_noise > 0: additive Gaussian read/quantisation noise.
    """
    H, W = S_img.shape[:2]
    S0, S1, S2 = S_img[..., 0], S_img[..., 1], S_img[..., 2]
    mosaic = np.empty((2 * H, 2 * W), dtype=np.float64)
    for r in range(2):
        for c in range(2):
            th = _DOFP_ANGLES[r, c]
            mosaic[r::2, c::2] = 0.5 * (S0 + S1 * np.cos(2 * th) + S2 * np.sin(2 * th))
    if photons > 0.0 or read_noise > 0.0:
        rng = np.random.default_rng() if rng is None else rng
    if photons > 0.0:
        mosaic = rng.poisson(np.clip(mosaic, 0, None) * photons) / photons
    if read_noise > 0.0:
        mosaic = mosaic + rng.normal(scale=read_noise, size=mosaic.shape)
    return mosaic


def mosaic_to_stokes(mosaic: np.ndarray) -> np.ndarray:
    """Super-pixel demosaicing: raw DoFP mosaic -> per-pixel Stokes (S3=0).

    Inverts Malus' law on each 2x2 block (no interpolation -> output is the
    half-resolution Stokes image, the standard 'super-pixel' demosaic):
        S0 = I0 + I90 = I45 + I135 ,  S1 = I0 - I90 ,  S2 = I45 - I135.
    Returns (H, W, 4) with H,W = mosaic dims // 2.
    """
    I90, I45 = mosaic[0::2, 0::2], mosaic[0::2, 1::2]
    I135, I0 = mosaic[1::2, 0::2], mosaic[1::2, 1::2]
    S = np.zeros((*I0.shape, 4), dtype=np.float64)
    S[..., 0] = 0.5 * ((I0 + I90) + (I45 + I135))   # average the two S0 estimates
    S[..., 1] = I0 - I90
    S[..., 2] = I45 - I135
    return S


def mosaic_to_stokes_interp(mosaic: np.ndarray) -> np.ndarray:
    """Full-resolution demosaic: bilinearly interpolate each analyzer plane back
    to the full sensor grid, then invert Malus' law per pixel.  Unlike the
    super-pixel demosaic this keeps full (2H x 2W) resolution and makes the
    inverse problem genuinely ill-posed (the four angles are never co-located),
    so spatial structure -- and a good prior -- actually matter."""
    Hf, Wf = mosaic.shape
    planes = {}
    for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        sub = torch.tensor(mosaic[r::2, c::2], dtype=torch.float32)[None, None]
        up = F.interpolate(sub, size=(Hf, Wf), mode="bilinear", align_corners=False)
        planes[(r, c)] = up[0, 0].numpy()
    I90, I45, I135, I0 = planes[(0, 0)], planes[(0, 1)], planes[(1, 0)], planes[(1, 1)]
    S = np.zeros((Hf, Wf, 4), dtype=np.float64)
    S[..., 0] = 0.5 * ((I0 + I90) + (I45 + I135))
    S[..., 1] = I0 - I90
    S[..., 2] = I45 - I135
    return S


def random_stokes_image(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth, physically valid Stokes image (S0>0, DOP<=1, S3=0): a few random
    low-frequency Fourier modes drive intensity, angle-of-polarization, and DOP."""
    yy, xx = np.meshgrid(np.linspace(0, 2 * np.pi, H), np.linspace(0, 2 * np.pi, W),
                         indexing="ij")

    def field(lo, hi):
        out = np.zeros((H, W))
        for _ in range(3):
            kx, ky = rng.integers(1, 4, size=2)
            ph = rng.uniform(0, 2 * np.pi)
            out += np.sin(kx * xx + ky * yy + ph)
        out = (out - out.min()) / (np.ptp(out) + 1e-9)   # -> [0,1]
        return lo + (hi - lo) * out

    S0 = field(0.5, 2.0)
    dop = field(0.0, 0.95)            # degree of (linear) polarization
    aop = field(0.0, np.pi)           # angle of polarization
    S = np.zeros((H, W, 4))
    S[..., 0] = S0
    S[..., 1] = S0 * dop * np.cos(2 * aop)
    S[..., 2] = S0 * dop * np.sin(2 * aop)
    return S


# --------------------------------------------------------------------------- #
#  Equivariant imaging network                                                 #
#                                                                              #
#  Turns the set-based block into a convolution-like operator: each output     #
#  pixel's node set is its K x K spatial neighborhood (extracted with unfold). #
#  Relative pixel offsets are fed in as *invariant scalar* features -- they are #
#  image coordinates, untouched by Lorentz transforms of the Stokes vectors -- #
#  so the operator is spatially anisotropic (a real filter) yet still strictly #
#  SO+(1,3)-equivariant in the polarization channels.                          #
# --------------------------------------------------------------------------- #


class PolarizationImagingNet(nn.Module):
    """Per-pixel SO+(1,3) invariant + equivariant outputs over K x K windows.

    Input  : Stokes image (B, H, W, 4).
    Output : inv_img (B, H, W, n_inv_out)  -- Lorentz-invariant scalar map
             equ_img (B, H, W, 4)          -- Lorentz-equivariant Stokes map
    """

    def __init__(self, k: int = 3, c_vec: int = 8, s_scal: int = 16,
                 n_blocks: int = 2, hidden: int = 64, n_inv_out: int = 1):
        super().__init__()
        assert k % 2 == 1, "kernel size must be odd"
        self.k, self.N, self.center = k, k * k, (k * k) // 2
        self.c_vec, self.s_scal = c_vec, s_scal
        self.lift = nn.Parameter(torch.randn(c_vec) * 0.5)
        # invariant positional scalars: one learned embedding per window position
        self.pos = nn.Parameter(torch.randn(self.N, s_scal) * 0.1)
        self.h0 = nn.Sequential(nn.Linear(1, s_scal), nn.SiLU(),
                                nn.Linear(s_scal, s_scal))
        self.blocks = nn.ModuleList(
            [LorentzEquivariantBlock(c_vec, s_scal, hidden) for _ in range(n_blocks)])
        self.inv_head = nn.Sequential(nn.Linear(s_scal, hidden), nn.SiLU(),
                                      nn.Linear(hidden, n_inv_out))
        self.equ_head = nn.Linear(c_vec, 1, bias=False)
        # ---- higher-grade equivariant heads (CAVEAT this addresses) ----
        # The vector-channel structure V:(...,C,4) already supports rank-2 outputs.
        # A single invariant-weighted contraction (equ_head) gives a 4-vector; for
        # tasks like surface-normal-from-polarization the natural target is a
        # higher-grade object: a symmetric Stokes-tensor or an antisymmetric
        # bivector, both transforming as  X -> g X g^T  (the (1,0)+(0,1) and
        # (1,1) pieces of  4 (x) 4).  Weights are learned *constants* = invariants.
        self.tensor_w = nn.Parameter(torch.randn(c_vec) * 0.3)        # -> symmetric tensor
        self.bivec_w = nn.Parameter(torch.randn(c_vec, c_vec) * 0.1)  # -> bivector (antisym)

    def _windows(self, S: torch.Tensor) -> torch.Tensor:
        """(B,H,W,4) -> (B, H*W, N, 4) reflect-padded K x K neighborhoods."""
        B, H, W, _ = S.shape
        p = self.k // 2
        x = S.permute(0, 3, 1, 2)                                   # (B,4,H,W)
        x = F.pad(x, (p, p, p, p), mode="reflect")
        patches = F.unfold(x, kernel_size=self.k)                  # (B, 4*N, H*W)
        patches = patches.view(B, 4, self.N, H * W)
        return patches.permute(0, 3, 2, 1)                         # (B, H*W, N, 4)

    def _core(self, P: torch.Tensor) -> dict:
        """Run lift + blocks + heads on a flat batch of windows P:(M,N,4).
        Returns center-pixel outputs keyed by head, each leading dim M."""
        V = P.unsqueeze(2) * self.lift.view(1, 1, self.c_vec, 1)   # (M,N,C,4) equivariant
        self_norm = mink(P, P).unsqueeze(-1)                       # (M,N,1) invariant
        H_feat = self.h0(self_norm) + self.pos.unsqueeze(0)        # + invariant positions
        for blk in self.blocks:
            V, H_feat = blk(V, H_feat)
        Vc = V[:, self.center]                                     # (M,C,4) equivariant
        Hc = H_feat[:, self.center]                                # (M,s) invariant
        A = self.bivec_w - self.bivec_w.t()                        # antisymmetric weights
        return {
            "inv": self.inv_head(Hc),                                       # (M, n_inv_out)
            "equ": self.equ_head(Vc.transpose(1, 2)).squeeze(-1),           # (M, 4)
            "tensor": torch.einsum("c,mcu,mcv->muv", self.tensor_w, Vc, Vc),    # symmetric
            "bivector": torch.einsum("cd,mcu,mdv->muv", A, Vc, Vc),            # antisymmetric
        }

    def forward(self, S: torch.Tensor, chunk: int = 4096):
        """chunk bounds peak memory: the block materialises O(chunk*N*N) pairwise
        tensors, so windows are streamed in chunks rather than all B*H*W at once."""
        B, H, W, _ = S.shape
        P = self._windows(S).reshape(B * H * W, self.N, 4)         # (M,N,4)
        M = P.shape[0]
        outs = [self._core(P[i:i + chunk]) for i in range(0, M, chunk)]
        cat = {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0]}
        return {
            "inv": cat["inv"].view(B, H, W, -1),
            "equ": cat["equ"].view(B, H, W, 4),
            "tensor": cat["tensor"].view(B, H, W, 4, 4),
            "bivector": cat["bivector"].view(B, H, W, 4, 4),
        }


# --------------------------------------------------------------------------- #
#  Imaging verification + demo                                                 #
# --------------------------------------------------------------------------- #

def train_step_chunked(net: "PolarizationImagingNet", S: torch.Tensor,
                       target: torch.Tensor, opt: torch.optim.Optimizer,
                       chunk: int = 2048) -> float:
    """One optimisation step with per-chunk gradient accumulation.

    Forward+backward each window-chunk separately so its autograd graph is freed
    before the next chunk, making *training* memory scale with `chunk` instead of
    batch*H*W*N.  This decouples GPU memory from image resolution; gradients from
    all chunks accumulate into .grad and a single opt.step() applies them.

    Equivalent (to fp round-off) to the full-batch MSE on net(S)['equ'] vs target.
    """
    B, H, W, _ = S.shape
    P = net._windows(S).reshape(B * H * W, net.N, 4).detach()      # (M,N,4), no input grad
    tgt = target.reshape(B * H * W, 4)
    M = P.shape[0]
    denom = float(M * 4)                                           # mean over all elements
    opt.zero_grad()
    total = 0.0
    for i in range(0, M, chunk):
        out = net._core(P[i:i + chunk])                           # graph for this chunk only
        sq = ((out["equ"] - tgt[i:i + chunk]) ** 2).sum()
        (sq / denom).backward()                                   # accumulate; frees graph
        total += sq.item()
    opt.step()
    return total / denom


# --------------------------------------------------------------------------- #
#  Non-equivariant baseline: a plain Stokes CNN (the control)                   #
# --------------------------------------------------------------------------- #

class StokesCNN(nn.Module):
    """Ordinary 4-channel conv denoiser over the Stokes image.  Spatially
    translation-equivariant but NOT Lorentz-equivariant -- the control that
    shows what the polarization symmetry buys."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch, 4, 3, padding=1),
        )

    def forward(self, S: torch.Tensor):            # (B,H,W,4) -> (B,H,W,4)
        return self.net(S.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


# --------------------------------------------------------------------------- #
#  Verification + demos                                                        #
# --------------------------------------------------------------------------- #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_g(img: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Apply a Lorentz matrix to every pixel's Stokes 4-vector of (B,H,W,4)."""
    return torch.einsum("uv,bhwv->bhwu", g, img)


def verify_forward_model(rng):
    print("=" * 70)
    print("CAMERA FORWARD MODEL  (DoFP mosaic <-> Stokes)")
    print("=" * 70)
    S = random_stokes_image(16, 16, rng)
    mosaic = stokes_to_mosaic(S, read_noise=0.0)
    S_sp = mosaic_to_stokes(mosaic)
    S_in = mosaic_to_stokes_interp(mosaic)
    err_sp = np.abs(S[..., :3] - S_sp[..., :3]).max()
    print(f"mosaic shape from {S.shape[:2]} Stokes image:           {mosaic.shape}")
    print(f"superpixel demosaic round-trip max err:         {err_sp:.2e}")
    print(f"interp demosaic output shape (full-res):        {S_in.shape}")
    print(f"all DOP <= 1 (physically valid image):          "
          f"{bool((dop(S) <= 1.0 + 1e-9).all())}")
    # Poisson shot noise actually corrupts the capture:
    noisy = mosaic_to_stokes(stokes_to_mosaic(S, photons=200.0, read_noise=0.01, rng=rng))
    print(f"shot+read noise demosaic MSE vs clean:          "
          f"{np.mean((S[...,:3]-noisy[...,:3])**2):.4e}")
    print()


def verify_imaging_equivariance(rng, dtype=torch.float64):
    print("=" * 70)
    print("IMAGING EQUIVARIANCE  (per-pixel, all heads, global Lorentz)")
    print("=" * 70)
    torch.manual_seed(0)
    net = PolarizationImagingNet(k=3, c_vec=6, s_scal=12, n_blocks=2).to(dtype)
    net.eval()

    B, H, W = 2, 12, 12
    S = np.stack([random_stokes_image(H, W, rng) for _ in range(B)])
    St = torch.tensor(S, dtype=dtype)
    g = random_lorentz(max_rapidity=0.3, rng=rng)
    gt = torch.tensor(g, dtype=dtype)
    Sg = _apply_g(St, gt)

    with torch.no_grad():
        o1, o2 = net(St), net(Sg)

    errs = {}
    errs["invariant"] = (o1["inv"] - o2["inv"]).abs().max().item()
    errs["vector (g X)"] = (_apply_g(o1["equ"], gt) - o2["equ"]).abs().max().item()
    # rank-2 heads transform as  X -> g X g^T
    gXgt = lambda X: torch.einsum("ua,nhwab,vb->nhwuv", gt, X, gt)
    errs["tensor (g X g^T)"] = (gXgt(o1["tensor"]) - o2["tensor"]).abs().max().item()
    errs["bivector (g X g^T)"] = (gXgt(o1["bivector"]) - o2["bivector"]).abs().max().item()

    print(f"applied global Lorentz g in SO+(1,3):           {is_lorentz(g)}")
    for name, e in errs.items():
        print(f"  {name:24s} error_max:            {e:.3e}")
    ok = max(errs.values()) < 1e-8
    # confirm the grade structure is genuine, not trivially zero
    sym = (o1["tensor"] - o1["tensor"].transpose(-1, -2)).abs().max().item()
    asy = (o1["bivector"] + o1["bivector"].transpose(-1, -2)).abs().max().item()
    print(f"  tensor symmetric? max|X-X^T|={sym:.1e}   bivector antisym? max|X+X^T|={asy:.1e}")
    print(f"ALL HEADS EQUIVARIANT TO POLARIZATION GROUP:    {ok}")
    print()
    return ok


def verify_depolarization(rng):
    """The depolarization caveat, made concrete: coherency rank/entropy separates
    non-depolarizing from depolarizing Mueller matrices, and the coherency matrix
    is equivariant (exact congruence) under the non-depolarizing subgroup while
    depolarizers -- having no inverse -- only form a semigroup."""
    print("=" * 70)
    print("DEPOLARIZATION & COHERENCY  (semigroup vs subgroup)")
    print("=" * 70)
    M_pure = random_lorentz(0.6, rng)
    w_pure = np.sort(np.linalg.eigvalsh(mueller_coherency(M_pure)).real)[::-1]
    print(f"non-depolarizing M: coherency eigs              "
          f"{np.round(w_pure, 4)}  (rank 1)")
    print(f"  polarimetric entropy (==0):                   {polarimetric_entropy(M_pure):.3e}")

    D = depolarizer_mueller((0.6, 0.4, 0.7))
    M_dep = D @ M_pure                          # physical, depolarizing, NOT in SO+(1,3)
    w_dep = np.sort(np.linalg.eigvalsh(mueller_coherency(M_dep)).real)[::-1]
    print(f"depolarizing M=D@M: coherency eigs              {np.round(w_dep, 4)}  (rank>1)")
    print(f"  polarimetric entropy (>0):                    {polarimetric_entropy(M_dep):.3f}")
    print(f"  M_dep in SO+(1,3)? (no -> semigroup):         {is_lorentz(M_dep)}")
    # a depolarizer pushes a fully polarized state strictly inside the cone:
    S = np.array([1.0, 1.0, 0.0, 0.0])          # DOP=1, on the cone
    print(f"  DOP: on-cone {dop(S):.3f}  ->  after depolarizer {dop(D @ S):.3f}  (inside)")

    # equivariance of the coherency under the non-depolarizing OUTPUT subgroup:
    U = retarder_jones(rng.normal(size=3), 0.7) @ diattenuator_jones(rng.normal(size=3), 0.3)
    U = U / np.sqrt(np.linalg.det(U))
    G = jones_to_mueller(U)
    W = np.kron(U, np.eye(2))                    # (U (x) I) in the (1/2,1/2)(x)conj rep
    H0 = mueller_coherency(M_dep)
    cong = W @ H0 @ W.conj().T
    err = np.abs(mueller_coherency(G @ M_dep) - cong).max()
    print(f"  coherency congruence H->(U x I)H(U x I)^H err: {err:.2e}  (subgroup-equivariant)")
    print()
    return err < 1e-9


def denoising_demo(rng, dtype=torch.float32):
    """Actual imaging task with realistic noise: recover clean Stokes from a
    shot+read-noise DoFP capture, full-resolution (interp) demosaic."""
    print("=" * 70)
    print(f"DENOISING DEMO  (shot+read noise, full-res interp demosaic)  [{DEVICE}]")
    print("=" * 70)
    torch.manual_seed(1)
    # Per-chunk gradient accumulation (train_step_chunked) bounds *training*
    # memory by `chunk`, not batch*H*W*N -- so we can run full 32x32 captures with
    # a k=5 window on the GPU without spilling into shared RAM.
    net = PolarizationImagingNet(k=5, c_vec=8, s_scal=16, n_blocks=3).to(dtype).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    H, W, photons, read = 16, 16, 60.0, 0.02      # -> 32x32 full-res Stokes
    bs, chunk = 8, 2048

    def batch(n):
        clean, noisy = [], []
        for _ in range(n):
            S = random_stokes_image(H, W, rng)
            mos = stokes_to_mosaic(S, photons=photons, read_noise=read, rng=rng)
            noisy.append(mosaic_to_stokes_interp(mos))            # full-res (2H,2W)
            # upsample clean target to the same full-res grid for a fair compare
            up = F.interpolate(torch.tensor(S).permute(2, 0, 1)[None], scale_factor=2,
                               mode="bilinear", align_corners=False)
            clean.append(up[0].permute(1, 2, 0).numpy())
        return (torch.tensor(np.stack(noisy), dtype=dtype).to(DEVICE),
                torch.tensor(np.stack(clean), dtype=dtype).to(DEVICE))

    def val_mse(x, y):
        with torch.no_grad():
            return ((net(x)["equ"] - y) ** 2).mean().item()

    xv, yv = batch(bs)
    base = ((xv - yv) ** 2).mean().item()
    for step in range(250):
        x, y = batch(bs)
        loss = train_step_chunked(net, x, y, opt, chunk=chunk)
        if step % 50 == 0 or step == 249:
            print(f"  step {step:3d}  train {loss:.5f}   val {val_mse(xv, yv):.5f}")
    final = val_mse(xv, yv)
    print(f"raw-demosaic MSE (no network):                  {base:.5f}")
    print(f"equivariant-net MSE:                            {final:.5f}")
    print(f"denoising improvement:                          {base / final:.2f}x")
    print()


def generalization_experiment(rng, dtype=torch.float32):
    """The payoff: train BOTH nets at one polarization 'pose' (canonical Stokes),
    then test on Lorentz-transformed inputs.  Errors are measured back in the
    canonical frame  ( g^{-1} f(g.S) vs clean ),  which an exactly equivariant
    net leaves unchanged (g^{-1} f(g.S) = f(S)) while a plain CNN does not."""
    print("=" * 70)
    print("GENERALIZATION  (train one pose -> test Lorentz-transformed pose)")
    print("=" * 70)
    H, W, noise = 14, 14, 0.04

    def canonical_batch(n):
        clean = np.stack([random_stokes_image(H, W, rng) for _ in range(n)])
        noisy = clean + rng.normal(scale=noise, size=clean.shape)
        return (torch.tensor(noisy, dtype=dtype).to(DEVICE),
                torch.tensor(clean, dtype=dtype).to(DEVICE))

    def train(net, steps, lr):
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        out = (lambda o: o["equ"]) if isinstance(net, PolarizationImagingNet) else (lambda o: o)
        for _ in range(steps):
            x, y = canonical_batch(16)
            loss = ((out(net(x)) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return out

    torch.manual_seed(2)
    equi = PolarizationImagingNet(k=3, c_vec=8, s_scal=16, n_blocks=2).to(dtype).to(DEVICE)
    cnn = StokesCNN(ch=32).to(dtype).to(DEVICE)
    out_e = train(equi, 250, 3e-3)
    out_c = train(cnn, 600, 3e-3)

    xv, yv = canonical_batch(16)

    def canon_frame_mse(net, out, g):
        """error measured in the canonical frame after pushing input through g."""
        gt = torch.tensor(g, dtype=dtype).to(DEVICE)
        ginv = torch.tensor(np.linalg.inv(g), dtype=dtype).to(DEVICE)
        with torch.no_grad():
            pred = out(net(_apply_g(xv, gt)))
            back = _apply_g(pred, ginv)
        return ((back - yv) ** 2).mean().item()

    id_e = ((out_e(equi(xv)) - yv) ** 2).mean().item()
    id_c = ((out_c(cnn(xv)) - yv) ** 2).mean().item()
    gs = [random_lorentz(0.4, rng) for _ in range(8)]
    ood_e = np.mean([canon_frame_mse(equi, out_e, g) for g in gs])
    ood_c = np.mean([canon_frame_mse(cnn, out_c, g) for g in gs])

    print(f"{'':22s}{'in-distribution':>18s}{'transformed pose':>18s}{'degrade':>10s}")
    print(f"{'equivariant net':22s}{id_e:>18.5f}{ood_e:>18.5f}{ood_e/id_e:>9.2f}x")
    print(f"{'plain CNN':22s}{id_c:>18.5f}{ood_c:>18.5f}{ood_c/id_c:>9.2f}x")
    print("equivariant net generalizes across poses with ~no degradation;")
    print("the CNN, trained at one pose, breaks on transformed polarization.")
    print()
    return ood_e / id_e < 1.05


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    verify_physics(rng)
    ok = verify_equivariance(rng)
    tiny_training_demo(rng)
    verify_forward_model(rng)
    ok_img = verify_imaging_equivariance(rng)
    ok_dep = verify_depolarization(rng)
    denoising_demo(rng)
    ok_gen = generalization_experiment(rng)
    print("=" * 70)
    print(f"RESULT: set-net equivariance     {'VERIFIED' if ok else 'FAILED'} (float64)")
    print(f"        imaging-net (all heads)  {'VERIFIED' if ok_img else 'FAILED'} (float64)")
    print(f"        coherency subgroup-equiv {'VERIFIED' if ok_dep else 'FAILED'} (float64)")
    print(f"        cross-pose generalization{'  PASS' if ok_gen else '  FAIL'}")
    print("=" * 70)
