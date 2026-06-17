"""
Adaptive generalized-RMSE regression: a loss with a *learnable* exponent alpha.

Your requested loss is the l_alpha norm of the residual vector:

        L(y) = ( sum_i |y_i - e_i|^alpha )^(1/alpha)

This generalizes RMSE: alpha=2 is the Euclidean (RMSE) case, alpha=1 is sum of
absolute errors, alpha->inf is the max error.

------------------------------------------------------------------------------
The trap (why you can't just make alpha a free parameter of *that* formula)
------------------------------------------------------------------------------
For any fixed residual vector r, the l_alpha norm ||r||_alpha is *monotonically
non-increasing in alpha*:

        ||r||_inf  <=  ...  <=  ||r||_2  <=  ||r||_1

So if you minimize L jointly over (model, alpha), the optimizer discovers it can
shrink the loss for free by pushing alpha -> +inf, regardless of fit quality.
alpha runs off to a corner and tells you nothing about the problem. The exponent
is not *identifiable* from the bare norm.

------------------------------------------------------------------------------
The fix: treat the loss as a negative log-likelihood
------------------------------------------------------------------------------
Keep the |residual|^alpha core, but read it as the NLL of a Generalized Gaussian
(a.k.a. exponential-power) distribution with shape alpha and scale s:

        p(r) = alpha / (2 s Gamma(1/alpha)) * exp( -(|r|/s)^alpha )

    -log p(r) = (|r|/s)^alpha  +  log s  +  log Gamma(1/alpha)  -  log alpha  + const

The extra log-partition term  ( log Gamma(1/alpha) - log alpha )  is a barrier
that *penalizes* degenerate alpha, so there is now a genuine optimum. alpha
becomes identifiable and adapts to the noise structure of the data:

        heavy-tailed / outliers  ->  alpha < 2   (robust, MAE-like)
        Gaussian noise           ->  alpha ~ 2   (recovers RMSE)
        bounded / near-uniform   ->  alpha > 2

Both the shape `alpha` and the scale `s` are learned alongside the network.

This script plants a known noise shape, fits an MLP with the adaptive loss, and
shows that the learned alpha recovers the planted exponent -- something a plain
MSE model cannot do.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- 
# The adaptive loss
# -----------------------------------------------------------------------------
class AdaptivePowerLoss(nn.Module):
    """Generalized-Gaussian NLL with a learnable shape (alpha) and scale (s).

    The penalty on each residual is |r/s|^alpha, generalizing RMSE. alpha is
    constrained to (alpha_min, alpha_max) via a scaled sigmoid for numerical
    stability; s is kept positive via softplus.
    """

    def __init__(self, init_alpha: float = 2.0, init_scale: float = 1.0,
                 alpha_min: float = 0.3, alpha_max: float = 8.0):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # invert the sigmoid parameterization so the module starts at init_alpha
        a0 = (init_alpha - alpha_min) / (alpha_max - alpha_min)
        a0 = min(max(a0, 1e-4), 1 - 1e-4)
        self._raw_alpha = nn.Parameter(torch.tensor(math.log(a0 / (1 - a0))))

        # invert softplus so the module starts at init_scale
        self._raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(init_scale))))

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self._raw_alpha)

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self._raw_scale) + 1e-6

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        s = self.scale
        # clamp the base away from 0 so the gradient of x^a stays finite for a < 1
        r = (target - pred).abs().clamp_min(1e-8)

        data_term = (r / s).pow(a)                       # |residual|^alpha core
        log_partition = torch.log(s) + torch.lgamma(1.0 / a) - torch.log(a)
        nll = data_term + log_partition                  # per-element NLL
        return nll.mean()


# -----------------------------------------------------------------------------
# A small regression network
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int = 1, hidden: int = 2, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Synthetic data with a *known* noise shape, so we can check recovery of alpha
# -----------------------------------------------------------------------------
def sample_generalized_gaussian(n, shape, scale, rng):
    """Draw n samples from GGD(0, scale, shape).

    If G ~ Gamma(1/shape, 1), then  scale * G**(1/shape) * Rademacher  is GGD.
    """
    g = rng.gamma(shape=1.0 / shape, scale=1.0, size=n)
    mag = scale * g ** (1.0 / shape)
    sign = rng.choice([-1.0, 1.0], size=n)
    return (mag * sign).astype(np.float32)


def make_dataset(n=4000, noise_shape=1.2, noise_scale=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float32)
    clean = np.sin(1.5 * x) + 0.3 * x                       # the function to learn
    noise = sample_generalized_gaussian(n, noise_shape, noise_scale, rng).reshape(-1, 1)
    y = (clean + noise).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(clean)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(model, loss_fn, x, y, epochs=2000, log_every=250):
    has_params = len(list(loss_fn.parameters())) > 0
    # mild weight decay on the network stops it from fitting (and thinning) the
    # noise, which would otherwise bias the shape estimate upward. The loss's own
    # alpha/scale get a faster, decay-free learning rate so they lock on quickly.
    groups = [{"params": model.parameters(), "lr": 3e-3, "weight_decay": 2e-3}]
    if has_params:
        groups.append({"params": loss_fn.parameters(), "lr": 3e-2, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups)
    all_params = list(model.parameters()) + list(loss_fn.parameters())

    history = {"epoch": [], "loss": [], "alpha": []}

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        opt.step()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        if hasattr(loss_fn, "alpha"):
            history["alpha"].append(float(loss_fn.alpha.detach()))

        if epoch % log_every == 0 or epoch == 1:
            if hasattr(loss_fn, "alpha"):
                a = float(loss_fn.alpha.detach())
                s = float(loss_fn.scale.detach())
                print(f"  epoch {epoch:5d} | loss {loss.item():8.4f} | "
                      f"alpha {a:6.3f} | scale {s:6.3f}")
            else:
                print(f"  epoch {epoch:5d} | loss {loss.item():8.4f}")

    return history


def evaluate(model, x, clean):
    with torch.no_grad():
        pred = model(x)
        # report fit to the *clean* signal (noise-free), so robustness shows up
        mae = (pred - clean).abs().mean().item()
        rmse = ((pred - clean) ** 2).mean().sqrt().item()
    return mae, rmse


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    PLANTED_SHAPE = 1.0   # < 2  => heavier-than-Gaussian tails; robust regime
    x, y, clean = make_dataset(noise_shape=PLANTED_SHAPE, noise_scale=0.3, seed=0)
    x, y, clean = x.to(device), y.to(device), clean.to(device)

    print(f"Planted noise shape (target for alpha): {PLANTED_SHAPE}\n")

    print("[1] Adaptive p-norm loss (alpha is learned):")
    model_a = MLP().to(device)
    loss_a = AdaptivePowerLoss(init_alpha=2.0).to(device)
    hist_a = train(model_a, loss_a, x, y)
    mae_a, rmse_a = evaluate(model_a, x, clean)
    print(f"  -> learned alpha = {float(loss_a.alpha.detach()):.3f}  (planted {PLANTED_SHAPE})")
    print(f"  -> fit to clean signal: MAE {mae_a:.4f} | RMSE {rmse_a:.4f}\n")

    print("[2] Baseline: plain MSE (alpha fixed at 2):")
    model_b = MLP().to(device)
    mse = nn.MSELoss()
    hist_b = train(model_b, mse, x, y)
    mae_b, rmse_b = evaluate(model_b, x, clean)
    print(f"  -> fit to clean signal: MAE {mae_b:.4f} | RMSE {rmse_b:.4f}\n")

    print("Summary")
    print(f"  adaptive  MAE {mae_a:.4f}  RMSE {rmse_a:.4f}  (alpha={float(loss_a.alpha.detach()):.2f})")
    print(f"  fixed MSE MAE {mae_b:.4f}  RMSE {rmse_b:.4f}  (alpha=2.00)")

    # --- training curves ---
    fig_t, axes_t = plt.subplots(1, 2, figsize=(12, 4))
    fig_t.suptitle("Training curves", fontsize=13)

    axes_t[0].plot(hist_a["epoch"], hist_a["loss"], label="adaptive", color="crimson")
    axes_t[0].plot(hist_b["epoch"], hist_b["loss"], label="MSE", color="steelblue")
    axes_t[0].set_xlabel("epoch")
    axes_t[0].set_ylabel("loss")
    axes_t[0].set_yscale("linear")
    axes_t[0].set_xscale("linear")
    axes_t[0].set_title("Loss vs epoch")
    axes_t[0].legend()

    axes_t[1].plot(hist_a["epoch"], hist_a["alpha"], color="crimson")
    axes_t[1].axhline(PLANTED_SHAPE, linestyle="--", color="black", linewidth=1, label=f"planted α={PLANTED_SHAPE}")
    axes_t[1].set_xlabel("epoch")
    axes_t[1].set_ylabel("alpha")
    axes_t[1].set_xscale("linear")
    axes_t[1].set_title("Alpha vs epoch  (adaptive model)")
    axes_t[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)

    # --- fit plot ---
    # move everything to CPU for plotting
    x_cpu = x.cpu().numpy().ravel()
    clean_cpu = clean.cpu().numpy().ravel()
    sort_idx = np.argsort(x_cpu)
    xs = x_cpu[sort_idx]
    ys_clean = clean_cpu[sort_idx]

    with torch.no_grad():
        pred_a = model_a(x).cpu().numpy().ravel()[sort_idx]
        pred_b = model_b(x).cpu().numpy().ravel()[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Adaptive vs MSE regression  (planted noise shape = {PLANTED_SHAPE})", fontsize=13)

    for ax, pred, label, alpha_str in zip(
        axes,
        [pred_a, pred_b],
        ["Adaptive (learned alpha)", "Baseline MSE (alpha=2)"],
        [f"α={float(loss_a.alpha.detach()):.2f} (learned)", "α=2.00 (fixed)"],
    ):
        ax.scatter(x_cpu, y.cpu().numpy().ravel(), s=6, alpha=0.25, color="steelblue", label="noisy data")
        ax.plot(xs, ys_clean, "k--", linewidth=1.5, label="clean signal")
        ax.plot(xs, pred, "r-", linewidth=2, label=f"model fit  ({alpha_str})")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("y")
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()
    print("\nPlots saved to training_curves.png and results.png")


if __name__ == "__main__":
    main()
