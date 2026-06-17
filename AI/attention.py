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
