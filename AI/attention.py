"""
From-scratch causal multi-head self-attention.

Every step is written out explicitly -- projections, head reshaping, QK-norm,
the QK^T score matrix, scaling, causal masking, the softmax and the value
mixing -- so each detail can be customized without going through the fused
`F.scaled_dot_product_attention` kernel.

Kept dependency-free (only torch) so this module can be edited and unit-tested
in isolation from the training script.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# alpha-entmax (Peters et al. 2019): a sparse drop-in for softmax. entmax_bisect
# supports a learnable scalar/tensor alpha with an exact custom backward, so
# gradients flow to both the scores and the per-layer alpha parameter.
try:
    from entmax import entmax_bisect
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Sparse attention needs the entmax package. Install it with:\n"
        "    pip install entmax"
    ) from e


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with optional QK-norm.

    Args:
        d_model:   model / residual width.
        n_heads:   number of attention heads (d_model must divide evenly).
        dropout:   dropout probability applied to the attention weights
                   (post-softmax) while training.
        qk_norm:   if True, RMS/L2-normalize q and k along the head dim before
                   scoring (stabilizes logits, decouples their scale from the
                   learned projection magnitude).
        scale:     score scaling factor. None -> 1/sqrt(head_dim) (the standard
                   choice). Override to experiment with sharper/softer softmax.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, qk_norm=True, scale=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qk_norm = qk_norm
        # 1/sqrt(head_dim) is the canonical scale; expose it so it's tweakable.
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)

        # Separate projections (no fused QKV) so each is easy to inspect / edit.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, t, B, T):
        # (B, T, d_model) -> (B, n_heads, T, head_dim)
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attn_mask=None, return_weights=False):
        """
        x:          (B, T, d_model) input hidden states.
        attn_mask:  optional additive mask broadcastable to (B, n_heads, T, T);
                    use -inf (or a large negative) to forbid an attention edge.
                    Causal masking is always applied on top of this.
        return_weights: also return the (B, n_heads, T, T) softmax weights.
        """
        B, T, _ = x.shape

        q = self._split_heads(self.q_proj(x), B, T)
        k = self._split_heads(self.k_proj(x), B, T)
        v = self._split_heads(self.v_proj(x), B, T)

        # QK-norm: unit-L2 per head vector, then rescale by sqrt(head_dim) so the
        # subsequent 1/sqrt(head_dim) scale leaves logits at O(1).
        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1) * math.sqrt(self.head_dim)
            k = F.normalize(k, p=2, dim=-1) * math.sqrt(self.head_dim)

        # Raw scores: (B, n_heads, T, T) = q @ k^T, scaled.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask: position i may only attend to j <= i. Upper triangle
        # (j > i) is set to -inf so it vanishes after softmax.
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax over the key axis, in fp32 for numerical stability, then back.
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        weights = self.attn_dropout(weights)

        # Mix values: (B, n_heads, T, head_dim).
        out = torch.matmul(weights, v)

        # Merge heads back to (B, T, d_model) and project out.
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.out_proj(out)

        if return_weights:
            return out, weights
        return out


class HigherOrderAttention(nn.Module):
    """Causal *subset* attention: score every subset of the visible tokens.

    Standard attention scores (query, single-key) pairs. This generalizes the
    "key" from a single token to an arbitrary *subset* of the causally-visible
    tokens. For a query at position i (with i+1 visible tokens 0..i) we form one
    score for every subset of {0..i}:

        - the empty set            -> the "constant" term (a learned null key),
        - all singletons           -> ordinary 1st-order attention,
        - all pairs, triples, ...  -> higher-order group interactions,
        - the full prefix {0..i}   -> one score over the entire visible context.

    There are 2^(i+1) such subsets (1 + C(i+1,1) + C(i+1,2) + ... = 2^(i+1)),
    e.g. for 5 visible tokens: 1 + 5 + 10 + 10 + 5 + 1 = 32.

    A subset's key/value is the sum of its members' key/value vectors (the empty
    set uses a learned null key/value). The query scores against every subset
    key, softmax runs over all subsets, and the output is the weighted sum of
    subset values.

    !!! Cost is exponential in sequence length: 2^T subsets per query, so this is
    only practical for short sequences (here T = MAX_LEN-1 = 7 -> 128 subsets).
    The subset enumeration is cached per sequence length T.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0, qk_norm=True,
                 scale=None, aggregate="sum", alpha_init=1.5):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert aggregate in ("sum", "mean")
        assert alpha_init > 1.0, "entmax alpha must be > 1 (alpha=1 is dense softmax)"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.qk_norm = qk_norm
        self.aggregate = aggregate
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)

        # Per-layer learnable entmax sparsity. We store a free parameter and map
        # it through softplus so alpha = 1 + softplus(raw) stays > 1 for any value
        # the optimizer picks (alpha -> 1 is dense softmax, larger is sparser).
        inv_softplus = math.log(math.expm1(alpha_init - 1.0))   # softplus^-1(a-1)
        self.alpha_raw = nn.Parameter(torch.tensor(inv_softplus))

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learned key/value for the empty subset (the "constant" term). Per head.
        self.null_key = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.null_value = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        self.attn_dropout = nn.Dropout(dropout)

        # Precompute the subset enumeration for the (fixed) max sequence length as
        # non-trainable buffers. Building these once -- rather than caching tensors
        # created inside the compiled region -- keeps their storage stable, which
        # is required for torch.compile()'s CUDA-graph capture.
        member, valid = self._build_subsets(max_seq_len)
        # member: (S, T) float {0,1}; row s is the membership bitmask of subset s.
        # valid:  (T, S) bool; valid[i, s] iff every member of s is at pos <= i.
        self.register_buffer("subset_member", member, persistent=False)
        self.register_buffer("subset_valid", valid, persistent=False)

    @staticmethod
    def _build_subsets(T):
        S = 1 << T                                  # 2^T subsets
        subsets = torch.arange(S)
        bit = torch.arange(T)
        member = ((subsets[:, None] >> bit[None, :]) & 1).to(torch.float32)  # (S, T)

        # Highest token index present in each subset (-1 for the empty set), so
        # we can causally forbid subsets that reach past the current query.
        idx = bit[None, :].expand(S, T)
        max_idx = torch.where(member > 0, idx, torch.full_like(idx, -1)).max(dim=1).values
        query_pos = torch.arange(T)
        valid = max_idx[None, :] <= query_pos[:, None]            # (T, S)
        return member, valid

    @property
    def alpha(self):
        # entmax sparsity exponent for this layer, constrained to (1, inf).
        return 1.0 + F.softplus(self.alpha_raw)

    def _split_heads(self, t, B, T):
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, return_weights=False):
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim

        q = self._split_heads(self.q_proj(x), B, T)              # (B, H, T, hd)
        k = self._split_heads(self.k_proj(x), B, T)
        v = self._split_heads(self.v_proj(x), B, T)

        assert T == self.max_seq_len, (
            f"HigherOrderAttention was built for seq_len={self.max_seq_len}, "
            f"got {T}. Pad inputs to a fixed width.")
        member = self.subset_member.to(k.dtype)                 # (S,T), match dtype
        valid = self.subset_valid                               # (T,S) bool
        S = member.shape[0]

        # Aggregate member key/value vectors per subset:
        #   subset_key[..., s, :] = sum_{j in s} k[..., j, :]
        # (B,H,S,T) x (B,H,T,hd) via member (S,T).
        subset_key = torch.einsum("st,bhtd->bhsd", member, k)
        subset_val = torch.einsum("st,bhtd->bhsd", member, v)

        if self.aggregate == "mean":
            size = member.sum(dim=1).clamp_min(1.0)             # (S,) members count
            subset_key = subset_key / size[None, None, :, None]
            subset_val = subset_val / size[None, None, :, None]

        # Empty subset (index 0) carries the learned null key/value -- the
        # "constant" term that attends to no token at all.
        subset_key = subset_key.clone()
        subset_val = subset_val.clone()
        subset_key[:, :, 0, :] = self.null_key[None]            # broadcast over B
        subset_val[:, :, 0, :] = self.null_value[None]

        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1) * math.sqrt(hd)
            subset_key = F.normalize(subset_key, p=2, dim=-1) * math.sqrt(hd)

        # Score every query against every subset key: (B, H, T, S).
        scores = torch.einsum("bhtd,bhsd->bhts", q, subset_key) * self.scale

        # Causal mask over subsets: query i may only use subsets within {0..i}.
        scores = scores.masked_fill(~valid[None, None], float("-inf"))

        # alpha-entmax over the subset axis: a *sparse* distribution that zeros
        # out low-scoring subsets entirely (vs. softmax, which keeps them all).
        weights = entmax_bisect(scores.float(), self.alpha, dim=-1).to(v.dtype)
        weights = self.attn_dropout(weights)

        # Mix subset values: (B, H, T, S) x (B, H, S, hd) -> (B, H, T, hd).
        out = torch.einsum("bhts,bhsd->bhtd", weights, subset_val)

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.out_proj(out)

        if return_weights:
            return out, weights
        return out


class DepthAttention(nn.Module):
    """Higher-order subset attention over the *depth* axis.

    Standard pre-norm transformers grow a residual stream h_l = h_{l-1} + Δ_l,
    i.e. they accumulate every layer output with a fixed unit weight. With depth
    this lets the hidden-state norm grow unchecked and dilutes any single
    layer's contribution.

    This replaces that fixed sum with the same higher-order, sparse mechanism as
    `HigherOrderAttention`, but applied along the depth axis instead of the token
    axis. Given the stack of the L outputs produced so far,

        O = stack([o_0, o_1, ..., o_{L-1}])      # (B, T, L, D)

    the most recent output o_{L-1} forms a query that scores not single outputs
    but every *subset* of {o_0..o_{L-1}} (2^L of them, the empty set being a
    learned "constant"). alpha-entmax over the subsets gives a sparse, learned,
    input-dependent aggregation, and the layer input is the weighted sum of the
    selected subset values. Because entmax weights are a convex combination, the
    aggregate stays norm-bounded with depth instead of compounding.

    The attention runs along L (depth), independently per (batch, token); the
    token/sequence axis is untouched (this is orthogonal to self-attention). L is
    fixed per instance, so the subset enumeration is precomputed as a buffer.
    """

    def __init__(self, d_model, num_outputs, n_heads=1, qk_norm=True, scale=None,
                 aggregate="sum", alpha_init=1.5):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert aggregate in ("sum", "mean")
        assert alpha_init > 1.0, "entmax alpha must be > 1 (alpha=1 is dense softmax)"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_outputs = num_outputs           # L: depth this instance reduces
        self.qk_norm = qk_norm
        self.aggregate = aggregate
        self.scale = scale if scale is not None else 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learned key/value for the empty subset (the "constant" term). Per head.
        self.null_key = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.null_value = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        # Per-instance learnable entmax sparsity (alpha = 1 + softplus(raw) > 1).
        inv_softplus = math.log(math.expm1(alpha_init - 1.0))
        self.alpha_raw = nn.Parameter(torch.tensor(inv_softplus))

        # Membership of every subset of the L outputs: (S, L), S = 2^L. No causal
        # mask -- a layer may aggregate any subset of the outputs before it.
        S = 1 << num_outputs
        idx = torch.arange(S)
        bit = torch.arange(num_outputs)
        member = ((idx[:, None] >> bit[None, :]) & 1).to(torch.float32)   # (S, L)
        self.register_buffer("subset_member", member, persistent=False)

    @property
    def alpha(self):
        return 1.0 + F.softplus(self.alpha_raw)

    def forward(self, outputs, return_weights=False):
        """
        outputs: (B, T, L, D) stack of the L layer outputs produced so far.
        returns: (B, T, D) aggregated input for the next layer.
        """
        B, T, L, _ = outputs.shape
        assert L == self.num_outputs, (
            f"DepthAttention built for L={self.num_outputs}, got {L}.")
        H, hd = self.n_heads, self.head_dim

        # Query from the frontier output; keys/values from all L outputs.
        # Shapes: q (B,T,H,hd); k,v (B,T,H,L,hd).
        q = self.q_proj(outputs[:, :, -1, :]).view(B, T, H, hd)
        k = self.k_proj(outputs).view(B, T, L, H, hd).permute(0, 1, 3, 2, 4)
        v = self.v_proj(outputs).view(B, T, L, H, hd).permute(0, 1, 3, 2, 4)

        member = self.subset_member.to(k.dtype)         # (S, L)
        S = member.shape[0]

        # Aggregate member key/value vectors per subset: (B,T,H,S,hd).
        subset_key = torch.einsum("sl,bthld->bthsd", member, k)
        subset_val = torch.einsum("sl,bthld->bthsd", member, v)

        if self.aggregate == "mean":
            size = member.sum(dim=1).clamp_min(1.0)     # (S,)
            subset_key = subset_key / size[None, None, None, :, None]
            subset_val = subset_val / size[None, None, None, :, None]

        # Empty subset (index 0) carries the learned null key/value -- the
        # "constant" depth term that aggregates none of the outputs.
        subset_key = subset_key.clone()
        subset_val = subset_val.clone()
        subset_key[:, :, :, 0, :] = self.null_key[None, None]
        subset_val[:, :, :, 0, :] = self.null_value[None, None]

        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1) * math.sqrt(hd)
            subset_key = F.normalize(subset_key, p=2, dim=-1) * math.sqrt(hd)

        # Score the frontier query against every subset: (B, T, H, S).
        scores = torch.einsum("bthd,bthsd->bths", q, subset_key) * self.scale

        # Sparse alpha-entmax over subsets (zeros out low-scoring subsets).
        weights = entmax_bisect(scores.float(), self.alpha, dim=-1).to(v.dtype)

        # Mix subset values and merge heads: (B, T, H, hd) -> (B, T, D).
        out = torch.einsum("bths,bthsd->bthd", weights, subset_val)
        out = out.reshape(B, T, self.d_model)
        out = self.out_proj(out)

        if return_weights:
            return out, weights
        return out
