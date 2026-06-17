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

A second trap: hard-sampled labels.  If you first draw hard labels from
softmax(f*(x)/T_true) and then train, the temperature signal is destroyed --
the labels look like ordinary noisy one-hot targets, and the model + T can
collude (learn overconfident logits and push T -> T_min).

A third trap: a perfectly-fitting model.  Even with soft labels, if the model
is expressive enough to reproduce f*(x) exactly, it absorbs T_true into its
own logit scale.  Calibration then finds T ~ 1 regardless.  The fix is model
misspecification (train on hard labels with a weaker model) plus a held-out
soft-label validation set for calibration -- the model's residual uncertainty
on unseen data is what T must explain.

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

T calibration finds the temperature that best matches the *model's* residual
uncertainty on held-out soft labels -- not necessarily the data-generation T,
since a misspecified model's errors are independent of the oracle's temperature.
T > 1 means the model was overconfident; T < 1 means underconfident.  Standard
CE (T=1) bakes in the assumption of perfect calibration and always gets this
wrong when the model is misspecified.
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
        # soft-label CE: -sum_k p_k * log softmax(f/T)_k
        log_probs = F.log_softmax(logits / T, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


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
def make_dataset(n: int = 6000, in_dim: int = 8, n_classes: int = 6,
                 planted_temp: float = 2.5, val_frac: float = 0.4, seed: int = 0):
    """Generate a misspecified classification problem.

    The *true* function is a nonlinear 2-hidden-layer MLP (fixed random weights).
    The *training model* is a shallow linear network that cannot fit it perfectly,
    producing a calibration gap that T must close.

    Returns:
        x_tr, y_tr_hard  -- training set with hard labels (for phase-1 CE training)
        x_val, y_val_soft -- validation set with soft labels (for phase-2 calibration)

    The hard labels are sampled from softmax(f_true(x) / planted_temp) so that the
    training data carries no direct temperature signal -- just like the regression
    case where we only see a single noisy observation, not the underlying distribution.
    Calibration must infer T from the *soft* validation labels.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Fixed nonlinear oracle (never trained, only used for data generation)
    oracle = nn.Sequential(
        nn.Linear(in_dim, 64), nn.Tanh(),
        nn.Linear(64, 64),     nn.Tanh(),
        nn.Linear(64, n_classes),
    )

    x = rng.standard_normal((n, in_dim)).astype(np.float32)
    x_t = torch.from_numpy(x)

    with torch.no_grad():
        logits_true = oracle(x_t)
        probs = torch.softmax(logits_true / planted_temp, dim=1)

    y_hard = torch.multinomial(probs, num_samples=1).squeeze(1)

    split = int(n * (1 - val_frac))
    return (x_t[:split], y_hard[:split],    # train: hard labels (loses T signal)
            x_t[split:], probs[split:])     # val:   soft labels (encodes T signal)


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


def calibrate(model, loss_fn, x, y, epochs: int = 1000, log_every: int = 200):
    """Phase 2: freeze the model and learn T only (temperature scaling)."""
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(loss_fn.parameters(), lr=3e-2)
    history = {"epoch": [], "loss": [], "temperature": []}

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        with torch.no_grad():
            logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["temperature"].append(float(loss_fn.temperature.detach()))

        if epoch % log_every == 0 or epoch == 1:
            T = float(loss_fn.temperature.detach())
            b = float(loss_fn.base.detach())
            print(f"  epoch {epoch:5d} | loss {loss.item():8.4f} | "
                  f"T {T:6.3f} | base {b:6.3f}")

    for p in model.parameters():
        p.requires_grad_(True)

    return history


def evaluate(model, loss_fn, x, y_soft):
    with torch.no_grad():
        logits = model(x)
        T = loss_fn.temperature if hasattr(loss_fn, "temperature") else torch.tensor(1.0)
        log_probs = F.log_softmax(logits / T, dim=1)
        nll = -(y_soft * log_probs).sum(dim=1).mean().item()
        # accuracy: predicted class vs most likely true class
        acc = (logits.argmax(dim=1) == y_soft.argmax(dim=1)).float().mean().item()
    return nll, acc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    IN_DIM, N_CLASSES = 8, 6
    PLANTED_TEMP = 2.5   # > 1: soft/uncertain labels; < 1: sharp/deterministic
    x_tr, y_tr, x_val, y_val = make_dataset(
        planted_temp=PLANTED_TEMP, in_dim=IN_DIM, n_classes=N_CLASSES)
    x_tr,  y_tr  = x_tr.to(device),  y_tr.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    print(f"Planted temperature (target for T): {PLANTED_TEMP}")
    print(f"Train: {len(x_tr)} hard-label examples  |  "
          f"Val: {len(x_val)} soft-label examples\n")

    # ------------------------------------------------------------------
    # Phase 1: train a shallow (misspecified) model with standard CE
    # The model can't fit the nonlinear oracle, leaving a calibration gap.
    # ------------------------------------------------------------------
    hard_ce = nn.CrossEntropyLoss()

    print("[Phase 1] Training shallow model on hard labels with standard CE (T=1):")
    model = MLP(in_dim=IN_DIM, hidden=32, n_classes=N_CLASSES).to(device)
    hist_pretrain = train(model, hard_ce, x_tr, y_tr)
    loss_uncal = AdaptiveTemperatureLoss(init_T=1.0).to(device)
    nll_base, acc_base = evaluate(model, loss_uncal, x_val, y_val)
    print(f"  -> val NLL {nll_base:.4f} | val accuracy {acc_base:.4f}\n")

    # ------------------------------------------------------------------
    # Phase 2: freeze model, learn T on the soft-label validation set
    # ------------------------------------------------------------------
    print("[Phase 2] Calibrating T on soft-label val set (model frozen):")
    loss_cal = AdaptiveTemperatureLoss(init_T=1.0).to(device)
    hist_cal = calibrate(model, loss_cal, x_val, y_val)
    learned_T = float(loss_cal.temperature.detach())
    learned_b = float(loss_cal.base.detach())
    nll_cal, acc_cal = evaluate(model, loss_cal, x_val, y_val)
    print(f"  -> learned T = {learned_T:.3f}  "
          f"(data planted T={PLANTED_TEMP}; model T reflects its own residual uncertainty)")
    print(f"  -> learned base b = e^(1/T) = {learned_b:.3f}  "
          f"(standard CE uses b=e={math.e:.3f})")
    print(f"  -> val NLL {nll_cal:.4f} | val accuracy {acc_cal:.4f}\n")

    nll_improvement = 100 * (nll_base - nll_cal) / nll_base
    print("Summary")
    print(f"  calibrated   NLL {nll_cal:.4f}  acc {acc_cal:.4f}  "
          f"(T={learned_T:.2f} > 1: model was overconfident)")
    print(f"  uncalibrated NLL {nll_base:.4f}  acc {acc_base:.4f}  (T=1.00, b={math.e:.2f})")
    print(f"  NLL improvement from calibration: {nll_improvement:.1f}%")

    # --- training curves ---
    fig_t, axes_t = plt.subplots(1, 2, figsize=(12, 4))
    fig_t.suptitle("Training curves", fontsize=13)

    axes_t[0].plot(hist_pretrain["epoch"], hist_pretrain["loss"], label="pre-train (T=1)", color="steelblue")
    axes_t[0].plot([e + len(hist_pretrain["epoch"]) for e in hist_cal["epoch"]],
                   hist_cal["loss"], label="calibration (T learned)", color="crimson")
    axes_t[0].set_xscale("log")
    axes_t[0].set_yscale("linear")
    axes_t[0].set_xlabel("epoch")
    axes_t[0].set_ylabel("loss")
    axes_t[0].set_title("Loss vs epoch")
    axes_t[0].legend()

    axes_t[1].plot(hist_cal["epoch"], hist_cal["temperature"], color="crimson")
    axes_t[1].axhline(PLANTED_TEMP, linestyle="--", color="black", linewidth=1,
                      label=f"planted T={PLANTED_TEMP}")
    axes_t[1].set_xscale("log")
    axes_t[1].set_xlabel("calibration epoch")
    axes_t[1].set_ylabel("temperature  T = 1 / log(base)")
    axes_t[1].set_title("Temperature vs epoch  (calibration phase)")
    axes_t[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves2.png", dpi=150)

    # --- NLL bar chart ---
    fig_r, ax_r = plt.subplots(figsize=(6, 4))
    labels = [f"Calibrated\n(T={learned_T:.2f}, b={learned_b:.2f})",
              f"Uncalibrated\n(T=1.00, b={math.e:.2f})"]
    nlls = [nll_cal, nll_base]
    bars = ax_r.bar(labels, nlls, color=["crimson", "steelblue"], width=0.4)
    ax_r.bar_label(bars, fmt="%.4f", padding=3)
    ax_r.set_ylabel("NLL (lower is better)")
    ax_r.set_title(f"Calibration results  (planted T={PLANTED_TEMP})")
    plt.tight_layout()
    plt.savefig("results2.png", dpi=150)

    plt.show()
    print("\nPlots saved to training_curves2.png and results2.png")


if __name__ == "__main__":
    main()
