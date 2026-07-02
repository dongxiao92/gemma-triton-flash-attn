"""How to use the FlashAttention-compatible API.

This example shows the two ways to run code written for flash-attention
2.x on this package's Triton kernels (Hopper/H100), covering the common
usage patterns:

  1. change one import line                    (Example 1-4)
  2. change nothing at all — install() shim    (Example 5)

Shapes below are Gemma-4-E2B-it's attention shapes (GQA 8:1, D=512
global layers / D=256 sliding-window layers), but any power-of-2 head
dim in [16, 512] and any dividing GQA ratio works (see fa_compat.py's
docstring for the faithfully-unsupported FA features — those raise
NotImplementedError rather than silently degrading).

Run:  python examples/fa_compat_example.py
"""
import torch

# ---------------------------------------------------------------------
# Instead of:   from flash_attn import flash_attn_func, ...
# write:
from gemma_triton_flash_attn.fa_compat import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16


# ---------------------------------------------------------------------
# Example 1 — dense causal attention (FA layout: batch, seq, heads, dim)
# ---------------------------------------------------------------------
B, N, H_Q, H_KV, D = 2, 4096, 8, 1, 512   # Gemma-4 global-attention layer
q = torch.randn(B, N, H_Q, D, device=dev, dtype=dt)
k = torch.randn(B, N, H_KV, D, device=dev, dtype=dt)   # GQA: fewer KV heads
v = torch.randn(B, N, H_KV, D, device=dev, dtype=dt)

out = flash_attn_func(q, k, v, causal=True)
print(f"1. dense causal:            out {tuple(out.shape)} {out.dtype}")


# ---------------------------------------------------------------------
# Example 2 — sliding-window attention (FA convention: window_size)
# ---------------------------------------------------------------------
D_s = 256                                  # Gemma-4 sliding-attention layer
q = torch.randn(B, N, H_Q, D_s, device=dev, dtype=dt)
k = torch.randn(B, N, H_KV, D_s, device=dev, dtype=dt)
v = torch.randn(B, N, H_KV, D_s, device=dev, dtype=dt)

# window_size=(511, 0) == attend to the last 512 positions (incl. self)
out = flash_attn_func(q, k, v, causal=True, window_size=(511, 0))
print(f"2. sliding window (512):    out {tuple(out.shape)}")


# ---------------------------------------------------------------------
# Example 3 — training: autograd works through the same call
# ---------------------------------------------------------------------
q = torch.randn(B, 2048, H_Q, D, device=dev, dtype=dt, requires_grad=True)
k = torch.randn(B, 2048, H_KV, D, device=dev, dtype=dt, requires_grad=True)
v = torch.randn(B, 2048, H_KV, D, device=dev, dtype=dt, requires_grad=True)

out = flash_attn_func(q, k, v, causal=True)
out.sum().backward()
print(f"3. training bwd:            dq {tuple(q.grad.shape)}, "
      f"dk {tuple(k.grad.shape)}, dv {tuple(v.grad.shape)}")
# deterministic=True forces the atomic-free backward (bit-identical
# grads across runs), same flag as FA.


# ---------------------------------------------------------------------
# Example 4 — varlen / sample packing (padding-free SFT)
# ---------------------------------------------------------------------
# Three samples of different lengths packed into one stream — the same
# cu_seqlens contract flash-attention uses (and what HF's
# DataCollatorWithFlattening produces).
seqlens = [1000, 37, 2048]
total = sum(seqlens)
cu = torch.tensor([0, 1000, 1037, 3085], device=dev, dtype=torch.int32)

q = torch.randn(total, H_Q, D, device=dev, dtype=dt, requires_grad=True)
k = torch.randn(total, H_KV, D, device=dev, dtype=dt, requires_grad=True)
v = torch.randn(total, H_KV, D, device=dev, dtype=dt, requires_grad=True)

out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
    causal=True,
)
out.sum().backward()   # tokens never attend across sample boundaries
print(f"4. varlen packing:          out {tuple(out.shape)} "
      f"(samples {seqlens}, grads OK)")


# ---------------------------------------------------------------------
# Example 5 — decode with a KV cache (batched generation)
# ---------------------------------------------------------------------
B_dec, cache_cap, used = 32, 8192, 6000
q1 = torch.randn(B_dec, 1, H_Q, D, device=dev, dtype=dt)      # 1 new token
k_cache = torch.randn(B_dec, cache_cap, H_KV, D, device=dev, dtype=dt)
v_cache = torch.randn(B_dec, cache_cap, H_KV, D, device=dev, dtype=dt)
k_new = torch.randn(B_dec, 1, H_KV, D, device=dev, dtype=dt)
v_new = torch.randn(B_dec, 1, H_KV, D, device=dev, dtype=dt)

# appends k_new/v_new into the cache at position `used` (in place, like
# FA), then attends over the first used+1 entries. Long caches with few
# sessions automatically take the split-KV (flash-decoding) kernels.
out = flash_attn_with_kvcache(
    q1, k_cache, v_cache, k=k_new, v=v_new,
    cache_seqlens=torch.full((B_dec,), used, device=dev, dtype=torch.int32),
    causal=True,
)
print(f"5. kv-cache decode:         out {tuple(out.shape)} "
      f"(B={B_dec}, cache {used}+1 of {cache_cap})")


# ---------------------------------------------------------------------
# Example 6 — zero code change: shim `flash_attn` itself
# ---------------------------------------------------------------------
# If you cannot edit the downstream code (a library that does
# `from flash_attn import flash_attn_func` internally), install the shim
# BEFORE that code is imported:
from gemma_triton_flash_attn.fa_compat import install
install()

# ...anything importing flash_attn from here on gets these kernels:
from flash_attn import flash_attn_func as fa_func          # noqa: E402
import flash_attn                                          # noqa: E402

q = torch.randn(1, 1024, H_Q, D, device=dev, dtype=dt)
k = torch.randn(1, 1024, H_KV, D, device=dev, dtype=dt)
v = torch.randn(1, 1024, H_KV, D, device=dev, dtype=dt)
out = fa_func(q, k, v, causal=True)
print(f"6. install() shim:          flash_attn.__version__ = "
      f"{flash_attn.__version__}, out {tuple(out.shape)}")

print("\nAll examples ran on the Triton kernels — no flash-attn build needed.")
