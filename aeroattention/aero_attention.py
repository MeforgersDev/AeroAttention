from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .fused_attention import (
    KVCache,
    aero_fused_attention,
    fused_multi_head_attention,
    project_qkv,
)


@dataclass
class AttentionOutput:
    """Container for attention outputs and optional metadata."""

    output: Tensor
    attn_probs: Optional[Tensor]
    kv_cache: Optional[KVCache]


class AeroAttention(nn.Module):
    """Fused multi-head attention module used throughout AeroAttention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        use_out_proj: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs) if use_out_proj else None

    def build_kv_cache(self, batch_size: int) -> KVCache:
        empty = torch.empty(
            (batch_size, self.num_heads, 0, self.head_dim),
            dtype=self.q_proj.weight.dtype,
            device=self.q_proj.weight.device,
        )
        return KVCache(key=empty, value=empty.clone())

    def project_qkv(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return project_qkv(hidden_states, self.q_proj, self.k_proj, self.v_proj)

    def _reshape_to_heads(self, tensor: Tensor) -> Tensor:
        batch, seq_len, embed_dim = tensor.shape
        head_dim = embed_dim // self.num_heads
        return tensor.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)

    def _reshape_from_heads(self, tensor: Tensor) -> Tensor:
        batch, num_heads, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[KVCache] = None,
        return_attn_probs: bool = False,
        dropout_p: Optional[float] = None,
    ) -> AttentionOutput:
        """Compute multi-head attention using the fused kernel."""

        dropout = self.dropout_p if dropout_p is None else dropout_p

        if query.dim() == 3 and query.size(-1) == self.embed_dim:
            query = self._reshape_to_heads(query)
        if key.dim() == 3 and key.size(-1) == self.embed_dim:
            key = self._reshape_to_heads(key)
        if value.dim() == 3 and value.size(-1) == self.embed_dim:
            value = self._reshape_to_heads(value)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache.append(key, value)
            key, value = cached_k, cached_v

        output, attn_probs = fused_multi_head_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout,
            training=self.training,
            causal=causal,
        )
        output = self._reshape_from_heads(output)
        if self.out_proj is not None:
            output = self.out_proj(output)
        return AttentionOutput(output=output, attn_probs=attn_probs if return_attn_probs else None, kv_cache=kv_cache)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        causal: bool = False,
        kv_cache: Optional[KVCache] = None,
        return_attn_probs: bool = False,
    ) -> AttentionOutput:
        """Project inputs and compute fused attention in a single call."""

        output, attn_probs, cache = aero_fused_attention(
            hidden_states,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            num_heads=self.num_heads,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            training=self.training,
            causal=causal,
            kv_cache=kv_cache,
            out_proj=self.out_proj,
        )
        if not return_attn_probs:
            attn_probs = None
        return AttentionOutput(output=output, attn_probs=attn_probs, kv_cache=cache)
