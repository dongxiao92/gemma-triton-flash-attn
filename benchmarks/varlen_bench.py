"""Varlen (multi-sample packing) perf: packed stream vs equivalent dense calls.

Compares:
  a) varlen packed call over N samples of length L (total T = N*L)
  b) dense per-sample loop (N kernel calls at length L)
  c) dense single call at length T (upper-bound work: more causal FLOPs)

Usage: python benchmarks/varlen_bench.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.attention import flash_attn_gqa_train, flash_attn_gqa_varlen_train


def time_fn(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


def bench(n_seqs, L, H_Q, H_KV, D, slide, mode):
    torch.manual_seed(0)
    T = n_seqs * L
    cu = torch.arange(0, T + 1, L, dtype=torch.int32, device="cuda")
    dt = torch.float16
    rg = mode == "fwdbwd"
    q = torch.randn(1, H_Q, T, D, dtype=dt, device="cuda", requires_grad=rg)
    k = torch.randn(1, H_KV, T, D, dtype=dt, device="cuda", requires_grad=rg)
    v = torch.randn(1, H_KV, T, D, dtype=dt, device="cuda", requires_grad=rg)
    do = torch.randn(1, H_Q, T, D, dtype=dt, device="cuda")

    def varlen():
        out = flash_attn_gqa_varlen_train(q, k, v, cu, L, causal=True,
                                          slide_size=slide)
        if rg:
            out.backward(do)
            q.grad = k.grad = v.grad = None

    def per_sample():
        for i in range(n_seqs):
            s0, e0 = i * L, (i + 1) * L
            qs = q[:, :, s0:e0]
            out = flash_attn_gqa_train(qs.detach().requires_grad_(rg),
                                       k[:, :, s0:e0].detach().requires_grad_(rg),
                                       v[:, :, s0:e0].detach().requires_grad_(rg),
                                       causal=True, slide_size=slide)
            if rg:
                out.backward(do[:, :, s0:e0])

    t_v = time_fn(varlen)
    t_p = time_fn(per_sample)
    print(f"  {n_seqs}x{L:>5} D={D} slide={slide:>4} {mode:<7}: "
          f"varlen {t_v:8.3f} ms | per-sample loop {t_p:8.3f} ms "
          f"({t_p / t_v:4.2f}x)")


if __name__ == "__main__":
    print("packing throughput (fp16, H_Q=8, H_KV=1, causal)")
    for mode in ("fwd", "fwdbwd"):
        for (n, L, D, slide) in [(4, 2048, 512, 0), (2, 4096, 512, 0),
                                 (8, 2048, 512, 0),
                                 (4, 2048, 256, 512), (8, 4096, 256, 512)]:
            bench(n, L, 8, 1, D, slide, mode)
