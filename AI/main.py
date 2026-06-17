"""
Linux / CUDA decoder-only transformer that learns (a +/- b) % m for
non-negative integers a, b, generating the answer digit by digit.

Heavily fused for a Linux + NVIDIA-GPU setup:
  * torch.compile()                  -- fuses the model graph (needs Triton).
  * Liger SiLU-Mul / RMSNorm         -- fused Triton kernels.
  * Fused scaled-dot-product attention (FlashAttention via SDPA, is_causal).
  * SwiGLU feed-forward.
  * Fused AdamW.
  * Learned absolute positional embeddings.

Install the extras (Linux only):
    pip install liger-kernel triton

Vocabulary:
    0..9 digits, 10 "+", 11 "-", 12 "%", 13 "=", 14 <eos>, 15 <pad>.
    Sequence: <digits a> Op <digits b> "%" <digits m> "=" <digits result> <eos>
    Both operands are non-negative integers; "-" is only the subtraction operator.
    Example:  9 + 6 % 2 3 = 1 5 <eos>   since (9 + 6) % 23 = 15
    Example:  9 - 6 % 2 3 = 3 <eos>     since (9 - 6) % 23 = 3
    Loss is applied only to the answer tokens (everything after "=").

Note: the input is right-padded and attention is purely causal, so the trailing
pad never needs an explicit pad mask -- causality already prevents any
supervised position from attending to it.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import HigherOrderAttention

# --- Liger fused Triton kernels (Linux + CUDA only) ------------------------- #
try:
    from liger_kernel.transformers import LigerRMSNorm
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
PLUS_TOKEN   = 10
MINUS_TOKEN  = 11
MOD_TOKEN    = 12          # "%"
EQ_TOKEN     = 13          # "=" (prompt/answer separator + decode start)
EOS_TOKEN    = 14          # end of answer
PAD_TOKEN    = 15
VOCAB_SIZE   = 16

IGNORE_INDEX = -100        # cross-entropy ignore label (non-answer positions)

NUM_MIN      = 1           # smallest operand (single digit)
NUM_MAX      = 9           # largest operand (single digit)
MOD_MIN      = 2           # smallest modulus
MOD_MAX      = 9           # largest modulus (single digit)

# Longest sequence: "9 + 9 % 2 = 0 <eos>"
#   operand 1 + op 1 + operand 1 + "%" 1 + mod 1 + "=" 1 + answer 1 + eos 1 = 8
# Single-digit operands/modulus keep results single digit (res < mod <= 9).
MAX_LEN      = 8

D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 2
D_FF         = 512
DROPOUT      = 0.1

RMS_EPS       = 1e-6

BATCH_SIZE   = 256
EPOCHS       = 10_000
LR           = 1e-3          # AdamW lr (all parameters)
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
# Data: every (a, op, b, %, mod) combination rendered as a causal-LM sequence.
# Operands and modulus are single digits, so the full space is tiny and we
# enumerate it exhaustively instead of sampling.
# ----------------------------------------------------------------------------- #
def _digit_tokens(v):
    """Decimal digits of a non-negative integer v as a list of digit tokens."""
    return [int(c) for c in str(v)]


def make_dataset():
    inputs, labels = [], []

    for a in range(NUM_MIN, NUM_MAX + 1):
        for b in range(NUM_MIN, NUM_MAX + 1):
            for m in range(MOD_MIN, MOD_MAX + 1):
                for is_sub in (False, True):
                    op_tok = MINUS_TOKEN if is_sub else PLUS_TOKEN
                    res = (a - b if is_sub else a + b) % m

                    prompt = (_digit_tokens(a) + [op_tok] + _digit_tokens(b)
                              + [MOD_TOKEN] + _digit_tokens(m))
                    full = prompt + [EQ_TOKEN] + _digit_tokens(res) + [EOS_TOKEN]

                    inp = full[:-1]
                    lab = full[1:]
                    for i in range(len(prompt)):        # ignore prompt + "="
                        lab[i] = IGNORE_INDEX
                    inputs.append(inp)
                    labels.append(lab)

    n_samples = len(inputs)
    width = MAX_LEN - 1
    x = torch.full((n_samples, width), PAD_TOKEN, dtype=torch.long)
    y = torch.full((n_samples, width), IGNORE_INDEX, dtype=torch.long)
    for i, (inp, lab) in enumerate(zip(inputs, labels)):
        x[i, :len(inp)] = torch.tensor(inp)
        y[i, :len(lab)] = torch.tensor(lab)
    return x, y


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #
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


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: RMSNorm + attention + SwiGLU."""

    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.0):
        super().__init__()
        self.norm1 = LigerRMSNorm(d_model, eps=RMS_EPS)
        self.attn = HigherOrderAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = LigerRMSNorm(d_model, eps=RMS_EPS)
        self.ff = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    """Returns final hidden states (B, T, D_MODEL); the LM-head projection is
    applied via `self.head` for loss, evaluation and generation."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        # Learned absolute positional embeddings (sequences are at most MAX_LEN).
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)

        # Every sequence runs at a fixed width (right-padded to MAX_LEN-1), which
        # is what HigherOrderAttention enumerates its 2^T token subsets over.
        self.layers = nn.ModuleList(
            DecoderLayer(D_MODEL, N_HEADS, D_FF, MAX_LEN - 1, DROPOUT)
            for _ in range(N_LAYERS)
        )
        self.norm = LigerRMSNorm(D_MODEL, eps=RMS_EPS)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, x):
        # x: (B, T) token ids -> hidden states (B, T, D_MODEL)
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


# ----------------------------------------------------------------------------- #
# Train / eval
# ----------------------------------------------------------------------------- #
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

    # Single fused AdamW over all parameters.
    adamw = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                              fused=(DEVICE == "cuda"))
    optimizers = [adamw]

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
                logits = model.head(hidden)
            # Cross-entropy in fp32 over answer tokens only.
            loss = F.cross_entropy(
                logits.float().reshape(-1, VOCAB_SIZE),
                yb.reshape(-1),
                ignore_index=IGNORE_INDEX)
            loss.backward()
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

        if epoch % 1000 == 0 or epoch == EPOCHS or epoch == 1:
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
    """Digit tokens for a non-negative integer demo operand."""
    return _digit_tokens(val), val


def _decode(tokens):
    """Turn answer tokens back into a human-readable string."""
    out = []
    for t in tokens:
        if 0 <= t <= 9:
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
        ("+", 9, 6, 2), ("+", 1, 6, 5), ("+", 5, 5, 7),
        ("+", 9, 9, 7), ("-", 3, 5, 7),
        ("-", 1, 6, 5), ("-", 3, 7, 2), ("-", 8, 9, 2),
    ]
    op_token = {"+": PLUS_TOKEN, "-": MINUS_TOKEN}
    for op, a, b, mod in cases:
        a_tok, a_val = _operand_tokens(a)
        b_tok, b_val = _operand_tokens(b)
        prompt = a_tok + [op_token[op]] + b_tok + [MOD_TOKEN] \
            + _digit_tokens(mod) + [EQ_TOKEN]
        pred = _decode(generate(model, prompt))

        true = str((a_val - b_val if op == "-" else a_val + b_val) % mod)
        print(f"{a:>1} {op} {b:>1} % {mod:>1} = {pred:>1} (true {true})")


if __name__ == "__main__":
    main()
