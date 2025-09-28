"""High-performance fused multi-head attention kernel for AeroAttention.

This module implements a PyTorch + Triton backed attention primitive inspired by
FlashAttention-3.  The implementation provides the following building blocks:

* Joint Q/K/V linear projections (with optional bias) operating on packed input
  representations.
* Scaled dot-product attention with numerically stable max-subtraction and
  masking support (causal and arbitrary additive masks).
* Masked softmax with dropout fused with the dot-product primitive on CUDA
  devices by delegating to PyTorch's Triton-powered scaled-dot-product attention
  (SDPA) kernel.
* Optional key/value cache updates that allow streaming/auto-regressive
  decoding workloads without re-materialising the entire K/V history.
* Optional output projection that can be re-used by higher level transformer
  blocks.

The GPU fast-path leverages ``torch.nn.functional.scaled_dot_product_attention``
which, on CUDA builds, dispatches to the Triton kernels shipped with PyTorch.
The fallback CPU implementation uses highly optimised PyTorch tensor
operations and is used for environments where CUDA/Triton is not available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

try:  # pragma: no cover - exercised only when Triton is available.
    import triton  # type: ignore  # noqa: F401 - imported for availability check.
except Exception:  # pragma: no cover - keeps CPU-only environments functional.
    triton = None  # type: ignore


@dataclass
class KVCache:
    """Simple container that stores running key/value tensors."""

    key: Tensor
    value: Tensor

    def append(self, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        if self.key.numel() == 0:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat([self.key, key], dim=-2)
            self.value = torch.cat([self.value, value], dim=-2)
        return self.key, self.value


def is_triton_available() -> bool:
    """Return ``True`` when the Triton runtime can be used."""

    return triton is not None and torch.cuda.is_available()


def project_qkv(
    hidden_states: Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Project ``hidden_states`` into Q/K/V tensors using the provided layers."""

    q = q_proj(hidden_states)
    k = k_proj(hidden_states)
    v = v_proj(hidden_states)
    return q, k, v


def _apply_output_projection(out: Tensor, out_proj: Optional[nn.Linear]) -> Tensor:
    if out_proj is None:
        return out
    return out_proj(out)


def _reshape_to_heads(t: Tensor, num_heads: int) -> Tensor:
    bsz, seqlen, embed_dim = t.shape
    head_dim = embed_dim // num_heads
    t = t.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
    return t


def _reshape_from_heads(t: Tensor) -> Tensor:
    bsz, num_heads, seqlen, head_dim = t.shape
    return t.transpose(1, 2).contiguous().view(bsz, seqlen, num_heads * head_dim)


def _pytorch_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor],
    dropout_p: float,
    training: bool,
    causal: bool,
) -> Tuple[Tensor, Tensor]:
    head_dim = query.size(-1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    if causal:
        causal_mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        probs = torch.nn.functional.dropout(probs, p=dropout_p)
    output = torch.matmul(probs, value)
    return output, probs


def _cuda_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor],
    dropout_p: float,
    training: bool,
    causal: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    attn_mask_tensor = attn_mask
    if attn_mask_tensor is not None:
        attn_mask_tensor = attn_mask_tensor.to(dtype=query.dtype, device=query.device)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask_tensor,
        dropout_p=dropout_p if training else 0.0,
        is_causal=causal,
    )
    return output, None  # PyTorch does not expose the attention probabilities here.


def fused_multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Run the fused multi-head attention primitive."""

    if query.device.type == "cuda" and is_triton_available():
        return _cuda_attention(query, key, value, attn_mask, dropout_p, training, causal)
    return _pytorch_attention(query, key, value, attn_mask, dropout_p, training, causal)


def aero_fused_attention(
    hidden_states: Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    num_heads: int,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    causal: bool = False,
    kv_cache: Optional[KVCache] = None,
    out_proj: Optional[nn.Linear] = None,
) -> Tuple[Tensor, Optional[Tensor], Optional[KVCache]]:
    """Convenience wrapper that projects inputs and executes fused attention."""

    q, k, v = project_qkv(hidden_states, q_proj, k_proj, v_proj)
    q = _reshape_to_heads(q, num_heads)
    k = _reshape_to_heads(k, num_heads)
    v = _reshape_to_heads(v, num_heads)

    if kv_cache is not None:
        cached_k, cached_v = kv_cache.append(k, v)
        k, v = cached_k, cached_v
    output, attn_probs = fused_multi_head_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        training=training,
        causal=causal,
    )
    output = _reshape_from_heads(output)
    output = _apply_output_projection(output, out_proj)
    return output, attn_probs, kv_cache