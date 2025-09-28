"""Benchmark AeroAttention fused attention against FlashAttention-3."""

import argparse
import time
from contextlib import nullcontext
import torch

from aeroattention import AeroAttention

try:
    from flash_attn import flash_attn_func  # type: ignore

    HAS_FLASH_ATTENTION3 = True
except Exception:  # pragma: no cover - optional dependency
    HAS_FLASH_ATTENTION3 = False
    flash_attn_func = None


def _time_call(fn, *args, repeat: int = 10, warmup: int = 3, **kwargs) -> float:
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    return (end - start) / repeat


def run_benchmark(
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    flash_only: bool = False,
    repeat: int = 10,
    warmup: int = 3,
) -> None:
    context = torch.cuda.amp.autocast if device == "cuda" else nullcontext
    with context(dtype=dtype if device == "cuda" else None):
        attention = AeroAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_p=0.0, device=torch.device(device))
        attention = attention.to(device=device, dtype=dtype).eval()
        hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        def aero_run() -> torch.Tensor:
            return attention(hidden_states, causal=causal).output

        aero_latency = _time_call(aero_run, repeat=repeat, warmup=warmup)
        print(f"AeroAttention latency: {aero_latency * 1e3:.3f} ms")

        if flash_only:
            return

        if not HAS_FLASH_ATTENTION3:
            print("FlashAttention-3 not installed; skipping comparison.")
            return

        def flash_run() -> torch.Tensor:
            qkv = attention.project_qkv(hidden_states)
            q, k, v = [tensor.view(batch_size, seq_len, num_heads, -1).transpose(1, 2) for tensor in qkv]
            return flash_attn_func(q, k, v, causal=causal)

        flash_latency = _time_call(flash_run, repeat=repeat, warmup=warmup)
        print(f"FlashAttention-3 latency: {flash_latency * 1e3:.3f} ms")
        print(f"Speedup (AeroAttention / FlashAttention-3): {aero_latency / flash_latency:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--flash-only", action="store_true", help="Only benchmark FlashAttention-3")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    run_benchmark(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        device=args.device,
        dtype=dtype_map[args.dtype],
        causal=args.causal,
        flash_only=args.flash_only,
        repeat=args.repeat,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()