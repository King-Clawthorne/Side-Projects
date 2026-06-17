"""
Linux / CUDA decoder-only transformer that learns (a +/- b) % m for
floating-point a, b, generating the answer digit by digit.

Heavily fused for a Linux + NVIDIA-GPU setup:
  * torch.compile()                  -- fuses the model graph (needs Triton).
  * Liger RoPE / SiLU-Mul / RMSNorm  -- fused Triton kernels.
  * Fused scaled-dot-product attention (FlashAttention via SDPA, is_causal).
  * SwiGLU feed-forward.
  * Fused AdamW.
  * LigerFusedLinearCrossEntropyLoss -- fuses the LM-head projection with the
                                        cross-entropy, never materializing the
                                        (N, vocab) logits during training.

Install the extras (Linux only):
    pip install liger-kernel triton

Vocabulary:
    0..9 digits, 10 ".", 11 "+", 12 "-", 13 "%", 14 "=", 15 <eos>, 16 <pad>.
    Sequence: [-] <digits a> Op <digits b> "%" <digits m> "=" <digits result> <eos>
    A leading "-" marks the first operand as negative (same token as subtract).
    Example:  9 . 1 + 6 % 2 3 = 1 5 . 1 <eos>   since (9.1 + 6) % 23 = 15.1
    Example:  - 9 . 1 + 6 % 2 3 = 1 9 . 9 <eos> since (-9.1 + 6) % 23 = 19.9
    Loss is applied only to the answer tokens (everything after "=").

Note: the input is right-padded and attention is purely causal, so the trailing
pad never needs an explicit pad mask -- causality already prevents any
supervised position from attending to it.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Liger fused Triton kernels (Linux + CUDA only) ------------------------- #
try:
    from liger_kernel.transformers import (
        LigerRMSNorm,
        LigerFusedLinearCrossEntropyLoss,
    )
    from liger_kernel.ops.rope import LigerRopeFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This variant requires the Liger kernels. Install them with:\n"
        "    pip install liger-kernel triton\n"
        "(Liger kernels are Triton-based and need a CUDA GPU on Linux.)"
    ) from e

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
# Digit tokens 0..9, then symbols.
DOT_TOKEN    = 10          # "."
PLUS_TOKEN   = 11
MINUS_TOKEN  = 12
MOD_TOKEN    = 13          # "%"
EQ_TOKEN     = 14          # "=" (prompt/answer separator + decode start)
EOS_TOKEN    = 15          # end of answer
PAD_TOKEN    = 16
VOCAB_SIZE   = 17

IGNORE_INDEX = -100        # cross-entropy ignore label (non-answer positions)

NUM_MAX      = 99          # operand integer part 0..99 (operands are x.y floats)
MOD_MIN      = 2           # smallest modulus
MOD_MAX      = 99          # largest modulus (integer)

# Longest sequence: "-99.9 + 99.9 % 99 = 99.9 <eos>"
#   sign 1 + operand 4 + op 1 + operand 4 + "%" 1 + mod 2 + "=" 1 + answer 4
#   + eos 1 = 19
MAX_LEN      = 19

D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 2
D_FF         = 512
DROPOUT      = 0.0

LAYERSCALE_INIT = 1e-1      # per-channel residual-branch scale init (CaiT LayerScale)
ROPE_BASE     = 10000.0     # rotary position embedding base frequency
RMS_EPS       = 1e-6

N_SAMPLES    = 600_000      # random examples (floats make full enumeration huge)
BATCH_SIZE   = 512
EPOCHS       = 100
LR           = 1e-3          # AdamW lr (embeddings, head, biases, norms)
MUON_LR      = 2e-2          # Muon lr (2D hidden weight matrices)
WEIGHT_DECAY = 1e-2
WARMUP_FRAC  = 0.05          # fraction of training steps spent in linear warmup
MIN_LR_FRAC  = 0.0           # final lr as a fraction of peak (cosine floor)
TRAIN_FRAC   = 0.8
SEED         = 0

COMPILE      = True          # wrap the model in torch.compile()
COMPILE_MODE = "max-autotune"  # or "default" / "reduce-overhead"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mixed precision: bf16 autocast (no GradScaler required) when supported,
# otherwise run in full precision.
USE_AMP   = (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) or DEVICE == "cpu"
AMP_DTYPE = torch.bfloat16

# Let TF32 matmuls run on Ampere+ for extra throughput.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------- #
# Data: random (a, op, b, %, mod) examples rendered as causal-LM sequences.
# Numbers are spelled character by character; arithmetic is in integer tenths.
# ----------------------------------------------------------------------------- #
def _digit_tokens(v):
    """Decimal digits of a non-negative integer v as a list of digit tokens."""
    return [int(c) for c in str(v)]


def _number_tokens(tenths, is_float):
    """Render a value given in tenths as number tokens."""
    if is_float:
        return _digit_tokens(tenths // 10) + [DOT_TOKEN, tenths % 10]
    return _digit_tokens(tenths // 10)


def _result_tokens(tenths):
    """Render a result in tenths, using a decimal point only when fractional."""
    if tenths % 10 == 0:
        return _digit_tokens(tenths // 10)
    return _digit_tokens(tenths // 10) + [DOT_TOKEN, tenths % 10]


def make_dataset(n_samples=N_SAMPLES, seed=SEED):
    rng = random.Random(seed)
    inputs, labels = [], []

    for _ in range(n_samples):
        # The first operand may be negative, written with a leading "-" token.
        def sample_operand(allow_negative=False):
            if rng.random() < 0.5:                     # integer operand
                v = rng.randint(0, NUM_MAX)
                t = v * 10
                tok = _number_tokens(t, is_float=False)
            else:                                       # float operand (tenths)
                t = rng.randint(0, NUM_MAX * 10 + 9)
                tok = _number_tokens(t, is_float=True)
            if allow_negative and t > 0 and rng.random() < 0.5:
                t, tok = -t, [MINUS_TOKEN] + tok
            return t, tok

        a_t, a_tok = sample_operand(allow_negative=True)
        b_t, b_tok = sample_operand()
        m = rng.randint(MOD_MIN, MOD_MAX)
        is_sub = rng.random() < 0.5
        op_tok = MINUS_TOKEN if is_sub else PLUS_TOKEN

        raw_t = a_t - b_t if is_sub else a_t + b_t
        res_t = raw_t % (m * 10)                        # exact, in tenths

        prompt = a_tok + [op_tok] + b_tok + [MOD_TOKEN] + _digit_tokens(m)
        full = prompt + [EQ_TOKEN] + _result_tokens(res_t) + [EOS_TOKEN]

        inp = full[:-1]
        lab = full[1:]
        for i in range(len(prompt)):                    # ignore prompt + "="
            lab[i] = IGNORE_INDEX
        inputs.append(inp)
        labels.append(lab)

    width = MAX_LEN - 1
    x = torch.full((n_samples, width), PAD_TOKEN, dtype=torch.long)
    y = torch.full((n_samples, width), IGNORE_INDEX, dtype=torch.long)
    for i, (inp, lab) in enumerate(zip(inputs, labels)):
        x[i, :len(inp)] = torch.tensor(inp)
        y[i, :len(lab)] = torch.tensor(lab)
    return x, y


# ----------------------------------------------------------------------------- #
# Muon optimizer (MomentUm Orthogonalized by Newton-schulz)
# Reference implementation: https://github.com/KellerJordan/Muon
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G via a quintic Newton-Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden weights. Use AdamW for everything else."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # scale so the update RMS is roughly consistent across shapes
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(g, alpha=-group["lr"] * scale)


# ----------------------------------------------------------------------------- #
# Rotary position embeddings (RoPE, Su et al. 2021), Liger NeoX convention.
# ----------------------------------------------------------------------------- #
def build_rope_cache(seq_len, head_dim, device, base=ROPE_BASE):
    """Precompute (cos, sin) of shape (1, seq_len, head_dim) for LigerRopeFunction."""
    inv_freq = 1.0 / (base ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                # (seq_len, head_dim // 2)
    emb = torch.cat([freqs, freqs], dim=-1)           # (seq_len, head_dim)
    return emb.cos()[None], emb.sin()[None]           # (1, seq_len, head_dim)


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
class LayerScale(nn.Module):
    """Per-channel learnable scaling of a residual branch (CaiT, Touvron 2021).

    Init near zero so each block starts close to the identity, which stabilizes
    the early training of deep pre-norm transformers.
    """

    def __init__(self, d_model, init=LAYERSCALE_INIT):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((d_model,), float(init)))

    def forward(self, x):
        return x * self.gamma


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020): down(SiLU(gate(x)) * up(x)).

    The SiLU-and-multiply is the fused Liger Triton kernel.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(LigerSiLUMulFunction.apply(self.gate(x), self.up(x)))


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with QK-norm, QK-gain, RoPE and value
    residuals, using fused scaled-dot-product attention.

    QK-norm        : RMS-normalize query/key vectors along the head dim.
    QK-gain        : a learnable per-head scalar replacing the fixed softmax
                     scale; folded into the query so SDPA runs with scale=1.
    RoPE           : Liger fused rotary embedding on queries and keys.
    value residual : mix this layer's values with the first layer's values via a
                     learnable per-head gate (Zhou et al. 2024).
    """

    def __init__(self, d_model, n_heads, dropout=0.0, value_residual=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.value_residual = value_residual

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        # QK-gain: one learnable scalar per head, init at the usual 1/sqrt(d) scale.
        init_gain = self.head_dim ** -0.5
        self.qk_gain = nn.Parameter(torch.full((n_heads, 1, 1), float(init_gain)))

        # Value-residual gate: per-head mix weight, init 0.5 (equal blend).
        if value_residual:
            self.v_lambda = nn.Parameter(torch.full((n_heads, 1, 1), 0.5))

    def forward(self, x, v_first=None, rope=None):
        B, T, _ = x.shape
        # (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Value residual: blend with the first layer's values.
        if self.value_residual and v_first is not None:
            lam = self.v_lambda
            v = lam * v + (1.0 - lam) * v_first

        # QK-norm: RMS-normalize along head_dim, then fold in the per-head gain.
        q = F.normalize(q, p=2, dim=-1) * (self.head_dim ** 0.5)
        k = F.normalize(k, p=2, dim=-1) * (self.head_dim ** 0.5)
        q = q * self.qk_gain[None]                        # (1, H, 1, 1) broadcast

        # RoPE (Liger fused) on queries and keys.
        if rope is not None:
            cos, sin = rope
            q, k = LigerRopeFunction.apply(q, k, cos.to(q.dtype), sin.to(q.dtype))

        # Fused causal SDPA (FlashAttention). Right-padding + causality means no
        # explicit pad mask is needed. scale=1 since the gain is already folded in.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=1.0,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out), v


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: RMSNorm + attention + SwiGLU, LayerScale on
    both residual branches."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, value_residual=False):
        super().__init__()
        self.norm1 = LigerRMSNorm(d_model, eps=RMS_EPS)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, value_residual)
        self.ls1 = LayerScale(d_model)
        self.norm2 = LigerRMSNorm(d_model, eps=RMS_EPS)
        self.ff = SwiGLU(d_model, d_ff)
        self.ls2 = LayerScale(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, v_first=None, rope=None):
        attn_out, v = self.attn(self.norm1(x), v_first, rope)
        x = x + self.dropout(self.ls1(attn_out))
        x = x + self.dropout(self.ls2(self.ff(self.norm2(x))))
        return x, v


class TransformerLM(nn.Module):
    """Returns final hidden states (B, T, D_MODEL); the LM-head projection is
    fused into the loss during training (LigerFusedLinearCrossEntropyLoss) and
    applied via `self.head` for evaluation / generation."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        # Positions are encoded with RoPE inside attention (no learned pos_emb).

        # First layer sources the value residual; later layers consume it.
        self.layers = nn.ModuleList(
            DecoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, value_residual=(i > 0))
            for i in range(N_LAYERS)
        )
        self.norm = LigerRMSNorm(D_MODEL, eps=RMS_EPS)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, x):
        # x: (B, T) token ids -> hidden states (B, T, D_MODEL)
        B, T = x.shape
        rope = build_rope_cache(T, D_MODEL // N_HEADS, x.device)
        h = self.tok_emb(x)
        v_first = None
        for layer in self.layers:
            h, v = layer(h, v_first, rope)
            if v_first is None:
                v_first = v                           # capture first-layer values
        return self.norm(h)


# ----------------------------------------------------------------------------- #
# Train / eval
# ----------------------------------------------------------------------------- #
# Fused linear + cross-entropy: never materializes the (N, vocab) logits.
fused_lce = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(fwd, head, x, y):
    """Teacher-forced loss and answer-token accuracy over device-resident data."""
    loss_sum = 0.0
    correct = total = 0
    for i in range(0, x.size(0), BATCH_SIZE):
        xb, yb = x[i:i + BATCH_SIZE], y[i:i + BATCH_SIZE]
        with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            hidden = fwd(xb)
            logits = head(hidden)
        logits = logits.float()
        loss_sum += F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum").item()
        mask = yb != IGNORE_INDEX
        pred = logits.argmax(-1)
        correct += ((pred == yb) & mask).sum().item()
        total += mask.sum().item()
    return loss_sum / total, correct / total


def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps,
                            min_lr_frac=0.0):
    """LambdaLR with linear warmup followed by cosine decay to min_lr_frac."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    torch.manual_seed(SEED)

    x, y = make_dataset()
    n = x.size(0)
    perm = torch.randperm(n)
    x, y = x[perm], y[perm]
    n_train = int(TRAIN_FRAC * n)

    # The dataset is small; keep it resident on the device so each step is a
    # plain slice instead of a DataLoader collation + host->device copy.
    x, y = x.to(DEVICE), y.to(DEVICE)
    train_x, train_y = x[:n_train], y[:n_train]
    test_x,  test_y  = x[n_train:], y[n_train:]

    model = TransformerLM().to(DEVICE)
    fwd = torch.compile(model, mode=COMPILE_MODE) if COMPILE else model

    # Muon handles the 2D hidden weight matrices inside the blocks; embeddings,
    # the output head, and norms stay on AdamW (fused).
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    muon = Muon(muon_params, lr=MUON_LR, momentum=0.95, weight_decay=WEIGHT_DECAY)
    adamw = torch.optim.AdamW(adamw_params, lr=LR, weight_decay=WEIGHT_DECAY,
                              fused=(DEVICE == "cuda"))
    optimizers = [muon, adamw]

    # Drop the last partial batch so every training step has the same shape,
    # which keeps torch.compile from recompiling for a different batch size.
    n_steps_train = n_train // BATCH_SIZE
    steps_per_epoch = n_steps_train

    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_FRAC * total_steps)
    schedulers = [
        warmup_cosine_scheduler(opt, warmup_steps, total_steps, MIN_LR_FRAC)
        for opt in optimizers
    ]

    print(f"device={DEVICE}  vocab={VOCAB_SIZE}  compile={COMPILE}  "
          f"train={n_train}  test={n - n_train}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n_train, device=DEVICE)   # reshuffle each epoch
        for s in range(n_steps_train):
            idx = order[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            xb, yb = train_x[idx], train_y[idx]
            for opt in optimizers:
                opt.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE,
                                enabled=USE_AMP):
                hidden = fwd(xb)
            # Fused linear-CE in fp32: head weight stays a leaf (grad flows),
            # and the (N, vocab) logits are never materialized.
            loss = fused_lce(model.head.weight,
                             hidden.float().reshape(-1, D_MODEL),
                             yb.reshape(-1))
            loss.backward()
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

        tr_loss, tr_acc = evaluate(fwd, model.head, train_x, train_y)
        te_loss, te_acc = evaluate(fwd, model.head, test_x, test_y)
        print(f"epoch {epoch:3d} | "
                f"train loss {tr_loss:.4f} tok-acc {tr_acc:.3f} | "
                f"test loss {te_loss:.4f} tok-acc {te_acc:.3f}")

    # quick sanity check
    demo(model)


# ----------------------------------------------------------------------------- #
# Generation / demo
# ----------------------------------------------------------------------------- #
def _operand_tokens(val):
    """Tokens + tenths for a demo operand given as an int or one-decimal float.

    Negative values get a leading "-" token.
    """
    neg = val < 0
    mag = abs(val)
    if isinstance(val, int):
        tenths, tok = mag * 10, _number_tokens(mag * 10, is_float=False)
    else:
        tenths = round(mag * 10)
        tok = _number_tokens(tenths, is_float=True)
    if neg:
        tenths, tok = -tenths, [MINUS_TOKEN] + tok
    return tok, tenths


def _decode(tokens):
    """Turn answer tokens back into a human-readable string."""
    out = []
    for t in tokens:
        if t == DOT_TOKEN:
            out.append(".")
        elif 0 <= t <= 9:
            out.append(str(t))
        else:
            break
    return "".join(out)


@torch.no_grad()
def generate(model, prompt_ids):
    """Greedy autoregressive decode starting after the "=" token.

    Runs at a fixed (1, MAX_LEN-1) shape -- slots beyond the current length are
    padding (causally masked out) -- so the compiled / CUDA graph is reused
    instead of re-recorded for every new length.
    """
    model.eval()
    width = MAX_LEN - 1
    ids = list(prompt_ids)
    x = torch.full((1, width), PAD_TOKEN, dtype=torch.long, device=DEVICE)
    x[0, :len(ids)] = torch.tensor(ids, device=DEVICE)
    while len(ids) < width:
        pos = len(ids) - 1                       # last real token position
        with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model.head(model(x))
        nxt = int(logits[0, pos].argmax())
        if nxt == EOS_TOKEN:
            break
        x[0, len(ids)] = nxt
        ids.append(nxt)
    return ids[len(prompt_ids):]


@torch.no_grad()
def demo(model):
    print("\n--- demo ---")
    cases = [
        ("+", 9.1, 6, 23), ("+", 13, 26, 5), ("+", 0.5, 0.5, 7),
        ("+", -9.1, 6, 23), ("-", -3.1, 5, 7),
        ("-", 13.2, 26, 5), ("-", 3, 7.5, 12), ("-", -88.8, 99, 2),
    ]
    op_token = {"+": PLUS_TOKEN, "-": MINUS_TOKEN}
    for op, a, b, mod in cases:
        a_tok, a_t = _operand_tokens(a)
        b_tok, b_t = _operand_tokens(b)
        prompt = a_tok + [op_token[op]] + b_tok + [MOD_TOKEN] \
            + _digit_tokens(mod) + [EQ_TOKEN]
        pred = _decode(generate(model, prompt))

        raw_t = a_t - b_t if op == "-" else a_t + b_t
        res_t = raw_t % (mod * 10)
        true = _decode(_result_tokens(res_t))
        print(f"{a:>6} {op} {b:>6} % {mod:>2} = {pred:>6}   (true {true})")


if __name__ == "__main__":
    main()
