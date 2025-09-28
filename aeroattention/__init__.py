"""AeroAttention package public API."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on optional PyTorch dependency.
    from .aero_attention import AeroAttention, AttentionOutput
    from .fused_attention import KVCache, aero_fused_attention, fused_multi_head_attention, is_triton_available
    _ATTENTION_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - triggered when torch is missing in tests.
    AeroAttention = AttentionOutput = KVCache = aero_fused_attention = fused_multi_head_attention = is_triton_available = None  # type: ignore
    _ATTENTION_IMPORT_ERROR = exc

__all__ = [
    "AeroAttention",
    "AttentionOutput",
    "KVCache",
    "aero_fused_attention",
    "fused_multi_head_attention",
    "is_triton_available",
]


def __getattr__(name: str) -> Any:
    if name in __all__ and _ATTENTION_IMPORT_ERROR is not None:
        raise ImportError(
            "AeroAttention requires PyTorch. Please install torch>=2.1 to access attention functionality."
        ) from _ATTENTION_IMPORT_ERROR
    raise AttributeError(name)