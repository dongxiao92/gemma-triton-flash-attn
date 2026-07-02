"""Cross-length (KV cache) attention correctness — Hopper port.

Exercises q_len != kv_len calls that `model.generate()` makes:
  - decode step:            q_len=1,  kv_len=t
  - prefill continuation:   q_len=c,  kv_len=past+c
  - sliding layers with cache trimmed to the window (kv_len <= window)

Reference: explicit-mask SDPA with q rows as the suffix of the KV stream.

Usage: python tests/test_kv_cache_decode.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from gemma_triton_flash_attn import attention_flash_gqa, flash_attn_gqa_train


def ref_suffix_attention(q, k, v, *, causal, slide):
    """q: (B,H_Q,Nq,D), k/v: (B,H_KV,Nkv,D); q row i has abs pos i + (Nkv-Nq)."""
    B, H_Q, Nq, D = q.shape
    _, H_KV, Nkv, _ = k.shape
    if H_Q != H_KV:
        r = H_Q // H_KV
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    off = Nkv - Nq
    qi = torch.arange(Nq, device=q.device)[:, None] + off  # abs q positions
    kj = torch.arange(Nkv, device=q.device)[None, :]
    mask = torch.ones(Nq, Nkv, dtype=torch.bool, device=q.device)
    if causal:
        mask &= kj <= qi
    if slide > 0:
        mask &= (qi - kj) < slide
    fmask = torch.where(mask, 0.0, float("-inf")).to(q.dtype)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=fmask[None, None])


def cos_sim(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    cases = [
        # (H_Q, H_KV, q_len, kv_len, D, causal, slide)          — what it models
        (8, 1, 1, 33, 512, True, 0),      # global layer, early decode step
        (8, 1, 1, 512, 512, True, 0),     # global layer, longer cache
        (8, 1, 1, 2048, 512, True, 0),    # global layer, long cache
        (8, 1, 1, 512, 256, True, 512),   # sliding layer, cache == window
        (8, 1, 1, 300, 256, True, 512),   # sliding layer, cache < window
        (8, 1, 1, 2048, 256, True, 512),  # sliding layer, untrimmed cache
        (8, 1, 17, 529, 512, True, 0),    # chunked prefill continuation
        (8, 1, 64, 1024, 256, True, 512), # chunked prefill continuation, SWA
        (8, 1, 7, 7, 512, True, 0),       # tiny equal-length (N < 16)
        (8, 1, 1, 1, 512, True, 0),       # first decode step, empty past
    ]
    n_fail = 0
    for (H_Q, H_KV, Nq, Nkv, D, causal, slide) in cases:
        q = torch.randn(1, H_Q, Nq, D, dtype=dtype, device="cuda")
        k = torch.randn(1, H_KV, Nkv, D, dtype=dtype, device="cuda")
        v = torch.randn(1, H_KV, Nkv, D, dtype=dtype, device="cuda")
        tag = f"q={Nq:>4} kv={Nkv:>4} D={D} slide={slide}"
        try:
            out = attention_flash_gqa(q, k, v, causal=causal, slide_size=slide)
            with torch.no_grad():
                out_train = flash_attn_gqa_train(q, k, v, causal=causal,
                                                 slide_size=slide)
            ref = ref_suffix_attention(q, k, v, causal=causal, slide=slide)
        except Exception as e:
            print(f"{tag:<36} ERROR {type(e).__name__}: {str(e)[:100]}")
            n_fail += 1
            continue
        cs, mx = cos_sim(out, ref), (out - ref).abs().max().item()
        cs2 = cos_sim(out_train, ref)
        ok = cs > 0.999 and cs2 > 0.999
        print(f"{tag:<36} {'PASS' if ok else 'FAIL'} "
              f"cos={cs:.6f} max={mx:.1e} (train-entry cos={cs2:.6f})")
        n_fail += 0 if ok else 1

    print("ALL PASS" if n_fail == 0 else f"{n_fail} FAILURES")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
