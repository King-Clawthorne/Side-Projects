"""
Adaptive cross-entropy: a loss with a *learnable* log base (temperature).

Standard cross-entropy uses log base e:

        L(y, f) = -log(softmax(f)_y)  =  -f_y + log( sum_j exp(f_j) )

This file generalises the base of the logarithm.  Using log base b is
equivalent to temperature-scaling the logits by T = 1 / log(b):

        L_T(y, f) = -f_y/T + log( sum_j exp(f_j / T) )

T < 1  sharpens the distribution (b > e), T > 1  flattens it (b < e),
T = 1  recovers standard cross-entropy (b = e).

------------------------------------------------------------------------------
The trap (why you can't just make b a free parameter of the bare log-loss)
------------------------------------------------------------------------------
log_b(p) = log(p) / log(b), so the bare loss is just a 1/log(b) scaling of
the standard loss.  For any fixed model output, the optimizer can shrink the
loss to zero by pushing b -> infinity (log b -> inf), with no improvement in
fit.  The base is not identifiable from the bare scaled loss.

------------------------------------------------------------------------------
The fix: use the full temperature-scaled NLL
------------------------------------------------------------------------------
The temperature-scaled loss is already the complete NLL of a softmax model
at temperature T -- it is not just a scaled version of the standard loss,
because T appears inside the log-sum-exp term as well:

        L_T(y, f) = -f_y/T + log( sum_j exp(f_j/T) )

As T -> 0 the model can collapse to a one-hot, making the data term vanish;
mild weight decay on the network prevents arbitrarily large logits, breaking
that collusion.  The remaining optimum is at the T that matches the
uncertainty actually present in the labels.

When data labels are sampled from softmax(f*(x) / T_true), fitting with the
adaptive loss recovers T_true -- something standard CE (T fixed at 1) cannot
do.  T_true > 1 means soft / uncertain labels; T_true < 1 means sharp /
near-deterministic labels.
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
class AdaptiveTemperatureLoss(nn.Module):
    """Cross-entropy with a learnable temperature T (= 1 / log(base)).

    Equivalent to using log base b = e^(1/T) in the cross-entropy.
    T is constrained to (T_min, T_max) via a scaled sigmoid.
    """

    def __init__(self, init_T: float = 1.0, T_min: float = 0.1, T_max: float = 6.0):
        super().__init__()
        self.T_min = T_min
        self.T_max = T_max

        t0 = (init_T - T_min) / (T_max - T_min)
        t0 = min(max(t0, 1e-4), 1 - 1e-4)
        self._raw_T = nn.Parameter(torch.tensor(math.log(t0 / (1 - t0))))

    @property
    def temperature(self) -> torch.Tensor:
        return self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self._raw_T)

    @property
    def base(self) -> torch.Tensor:
        return torch.exp(1.0 / self.temperature)   # b = e^(1/T)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        return F.cross_entropy(logits / T, targets)


# -----------------------------------------------------------------------------
# A small classification network
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Synthetic data with a *known* label temperature, so we can check recovery
# -----------------------------------------------------------------------------
def make_dataset(n: int = 4000, in_dim: int = 4, n_classes: int = 4,
                 planted_temp: float = 2.5, seed: int = 0):
    """Generate classification data whose labels are drawn from
    softmax(W_true @ x / planted_temp).  The true decision boundary is linear;
    label uncertainty is entirely controlled by planted_temp."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    W_true = torch.randn(n_classes, in_dim) * 0.8   # fixed true weight matrix

    x = rng.standard_normal((n, in_dim)).astype(np.float32)
    x_t = torch.from_numpy(x)

    with torch.no_grad():
        logits_true = x_t @ W_true.T                               # (n, n_classes)
        probs = torch.softmax(logits_true / planted_temp, dim=1)   # soft labels

    y = torch.multinomial(probs, num_samples=1).squeeze(1)         # hard labels
    return x_t, y


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(model, loss_fn, x, y, epochs: int = 3000, log_every: int = 500):
    has_params = len(list(loss_fn.parameters())) > 0
    groups = [{"params": model.parameters(), "lr": 3e-3, "weight_decay": 2e-3}]
    if has_params:
        groups.append({"params": loss_fn.parameters(), "lr": 3e-2, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups)
    all_params = list(model.parameters()) + list(loss_fn.parameters())

    history = {"epoch": [], "loss": [], "temperature": []}

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        opt.step()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        if has_params:
            history["temperature"].append(float(loss_fn.temperature.detach()))

        if epoch % log_every == 0 or epoch == 1:
            if has_params:
                T = float(loss_fn.temperature.detach())
                b = float(loss_fn.base.detach())
                print(f"  epoch {epoch:5d} | loss {loss.item():8.4f} | "
                      f"T {T:6.3f} | base {b:6.3f}")
            else:
                print(f"  epoch {epoch:5d} | loss {loss.item():8.4f}")

    return history


def evaluate(model, loss_fn, x, y):
    with torch.no_grad():
        logits = model(x)
        T = loss_fn.temperature if hasattr(loss_fn, "temperature") else torch.tensor(1.0)
        nll = F.cross_entropy(logits / T, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return nll, acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    IN_DIM, N_CLASSES = 4, 4
    PLANTED_TEMP = 2.5   # > 1: soft/uncertain labels; < 1: sharp/deterministic
    x, y = make_dataset(planted_temp=PLANTED_TEMP, in_dim=IN_DIM, n_classes=N_CLASSES)
    x, y = x.to(device), y.to(device)

    print(f"Planted temperature (target for T): {PLANTED_TEMP}\n")

    print("[1] Adaptive cross-entropy (temperature is learned):")
    model_a = MLP(in_dim=IN_DIM, n_classes=N_CLASSES).to(device)
    loss_a = AdaptiveTemperatureLoss(init_T=1.0).to(device)
    hist_a = train(model_a, loss_a, x, y)
    nll_a, acc_a = evaluate(model_a, loss_a, x, y)
    learned_T = float(loss_a.temperature.detach())
    learned_b = float(loss_a.base.detach())
    print(f"  -> learned T = {learned_T:.3f}  (planted {PLANTED_TEMP})")
    print(f"  -> learned base b = e^(1/T) = {learned_b:.3f}  (standard CE uses b=e={math.e:.3f})")
    print(f"  -> NLL {nll_a:.4f} | accuracy {acc_a:.4f}\n")

    print("[2] Baseline: standard cross-entropy (T fixed at 1, b fixed at e):")
    model_b = MLP(in_dim=IN_DIM, n_classes=N_CLASSES).to(device)
    ce = nn.CrossEntropyLoss()
    hist_b = train(model_b, ce, x, y)
    nll_b, acc_b = evaluate(model_b, ce, x, y)
    print(f"  -> NLL {nll_b:.4f} | accuracy {acc_b:.4f}\n")

    print("Summary")
    print(f"  adaptive CE  NLL {nll_a:.4f}  acc {acc_a:.4f}  (T={learned_T:.2f}, b={learned_b:.2f})")
    print(f"  standard CE  NLL {nll_b:.4f}  acc {acc_b:.4f}  (T=1.00, b={math.e:.2f})")

    # --- training curves ---
    fig_t, axes_t = plt.subplots(1, 2, figsize=(12, 4))
    fig_t.suptitle("Training curves", fontsize=13)

    axes_t[0].plot(hist_a["epoch"], hist_a["loss"], label="adaptive CE", color="crimson")
    axes_t[0].plot(hist_b["epoch"], hist_b["loss"], label="standard CE", color="steelblue")
    axes_t[0].set_xscale("linear")
    axes_t[0].set_yscale("linear")
    axes_t[0].set_xlabel("epoch")
    axes_t[0].set_ylabel("loss")
    axes_t[0].set_title("Loss vs epoch")
    axes_t[0].legend()

    axes_t[1].plot(hist_a["epoch"], hist_a["temperature"], color="crimson")
    axes_t[1].axhline(PLANTED_TEMP, linestyle="--", color="black", linewidth=1,
                      label=f"planted T={PLANTED_TEMP}")
    axes_t[1].set_xscale("linear")
    axes_t[1].set_xlabel("epoch")
    axes_t[1].set_ylabel("temperature  T = 1 / log(base)")
    axes_t[1].set_title("Temperature vs epoch  (adaptive model)")
    axes_t[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves2.png", dpi=150)

    # --- accuracy bar chart ---
    fig_r, ax_r = plt.subplots(figsize=(6, 4))
    labels = [f"Adaptive CE\n(T={learned_T:.2f}, b={learned_b:.2f})",
              f"Standard CE\n(T=1.00, b={math.e:.2f})"]
    accs = [acc_a, acc_b]
    bars = ax_r.bar(labels, accs, color=["crimson", "steelblue"], width=0.4)
    ax_r.bar_label(bars, fmt="%.4f", padding=3)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("accuracy")
    ax_r.set_title(f"Classification accuracy  (planted T={PLANTED_TEMP})")
    plt.tight_layout()
    plt.savefig("results2.png", dpi=150)

    plt.show()
    print("\nPlots saved to training_curves2.png and results2.png")


if __name__ == "__main__":
    main()
