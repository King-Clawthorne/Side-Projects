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

    def __init__(self, d_model, n_heads, dropout=0.0, qk_norm=True, scale=None,
                 aggregate="sum"):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert aggregate in ("sum", "mean")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
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

        self.attn_dropout = nn.Dropout(dropout)

        # Per-length cache of (membership, valid_mask) tensors, keyed by T.
        self._subset_cache = {}

    def _subsets(self, T, device):
        """Build, for sequence length T, the subset-enumeration tensors.

        Returns:
            member:  (S, T) float in {0,1}; row s lists which tokens are in
                     subset s. S = 2^T. Subset index s is the bitmask over tokens.
            valid:   (T, S) bool; valid[i, s] is True iff every member of subset s
                     is at position <= i (causal: query i can use subset s).
        """
        key = (T, device)
        cached = self._subset_cache.get(key)
        if cached is not None:
            return cached

        S = 1 << T                                  # 2^T subsets
        subsets = torch.arange(S, device=device)
        bit = torch.arange(T, device=device)
        member = ((subsets[:, None] >> bit[None, :]) & 1).to(torch.float32)  # (S, T)

        # Highest token index present in each subset (-1 for the empty set), so
        # we can causally forbid subsets that reach past the current query.
        idx = bit[None, :].expand(S, T)
        max_idx = torch.where(member > 0, idx, torch.full_like(idx, -1)).max(dim=1).values
        query_pos = torch.arange(T, device=device)
        valid = max_idx[None, :] <= query_pos[:, None]            # (T, S)

        self._subset_cache[key] = (member, valid)
        return member, valid

    def _split_heads(self, t, B, T):
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, return_weights=False):
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim

        q = self._split_heads(self.q_proj(x), B, T)              # (B, H, T, hd)
        k = self._split_heads(self.k_proj(x), B, T)
        v = self._split_heads(self.v_proj(x), B, T)

        member, valid = self._subsets(T, x.device)              # (S,T), (T,S)
        member = member.to(k.dtype)                             # match autocast dtype
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

        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        weights = self.attn_dropout(weights)

        # Mix subset values: (B, H, T, S) x (B, H, S, hd) -> (B, H, T, hd).
        out = torch.einsum("bhts,bhsd->bhtd", weights, subset_val)

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.out_proj(out)

        if return_weights:
            return out, weights
        return out
