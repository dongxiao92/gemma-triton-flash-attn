"""Five industry-typical workloads for the Gemma-4 attention kernels (H100).

W1  Padding-free SFT        packed [2919,1370,3105,798] (~8K), fwd+bwd, bf16
W2  Long-context RAG prefill B=1, N=32768, fwd, bf16
W3  Batched chat prefill     B=8, N=2048, fwd, bf16
W4  Batched decode           B=32, q=1, KV cache 8192 (global) / 512 (SWA), bf16
W5  MoE-26B-A4B training     N=8192 fwd+bwd, 16:2 D=512 + 16:8 D=256 slide=1024

Each workload runs both Gemma-4 layer types weighted by the model's layer
mix (E2B: 7 global + 28 sliding; MoE: 6 + 24). Reports per-op time and
per-model-forward attention time. NCU SOL is measured separately via
benchmarks/hopper_ncu_sol.py targets.

Usage: python benchmarks/workloads_sol.py [W1|W2|W3|W4|W5|all]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.attention import (
    attention_flash_gqa,
    flash_attn_gqa_train,
    flash_attn_gqa_varlen_train,
)

DT = torch.bfloat16


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


def qkv(B, H_Q, H_KV, N, D, rg=False, N_KV=None):
    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=DT, device="cuda", requires_grad=rg)
    k = torch.randn(B, H_KV, N_KV or N, D, dtype=DT, device="cuda", requires_grad=rg)
    v = torch.randn(B, H_KV, N_KV or N, D, dtype=DT, device="cuda", requires_grad=rg)
    return q, k, v


def report(name, layer_tag, ms, n_layers):
    print(f"  {name:<44} {ms:9.3f} ms/op  x{n_layers} layers = {ms*n_layers:9.2f} ms")
    return ms * n_layers


def w1():
    print("W1: padding-free SFT, packed ~8K, fwd+bwd, E2B")
    seqlens = [2919, 1370, 3105, 798]
    T = sum(seqlens)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)),
                      dtype=torch.int32, device="cuda")
    mx = max(seqlens)
    total = 0.0
    for (D, slide, n_layers, tag) in [(512, 0, 7, "global D=512"),
                                      (256, 512, 28, "sliding D=256")]:
        q, k, v = qkv(1, 8, 1, T, D, rg=True)
        do = torch.randn_like(q)

        def fb():
            out = flash_attn_gqa_varlen_train(q, k, v, cu, mx, causal=True,
                                              slide_size=slide)
            out.backward(do)
            q.grad = k.grad = v.grad = None
        total += report(f"varlen fwd+bwd {tag}", tag, time_fn(fb), n_layers)
    print(f"  => attention per model fwd+bwd step: {total:.2f} ms\n")


def w2():
    print("W2: long-context RAG prefill, B=1 N=32768, fwd, E2B")
    total = 0.0
    for (D, slide, n_layers, tag) in [(512, 0, 7, "global D=512"),
                                      (256, 512, 28, "sliding D=256")]:
        q, k, v = qkv(1, 8, 1, 32768, D)
        fn = lambda: attention_flash_gqa(q, k, v, causal=True, slide_size=slide)
        total += report(f"prefill fwd {tag}", tag, time_fn(fn), n_layers)
    print(f"  => attention per model forward: {total:.2f} ms\n")


def w3():
    print("W3: batched chat prefill, B=8 N=2048, fwd, E2B")
    total = 0.0
    for (D, slide, n_layers, tag) in [(512, 0, 7, "global D=512"),
                                      (256, 512, 28, "sliding D=256")]:
        q, k, v = qkv(8, 8, 1, 2048, D)
        fn = lambda: attention_flash_gqa(q, k, v, causal=True, slide_size=slide)
        total += report(f"prefill fwd {tag}", tag, time_fn(fn), n_layers)
    print(f"  => attention per model forward: {total:.2f} ms\n")


def w4():
    print("W4: batched decode, B=32, q=1, cache 8192 (global) / 512 (SWA), E2B")
    total = 0.0
    for (D, slide, kv_len, n_layers, tag) in [
            (512, 0, 8192, 7, "global D=512 kv=8K"),
            (256, 512, 512, 28, "sliding D=256 kv=512")]:
        q, k, v = qkv(32, 8, 1, 1, D, N_KV=kv_len)
        fn = lambda: attention_flash_gqa(q, k, v, causal=True, slide_size=slide)
        total += report(f"decode fwd {tag}", tag, time_fn(fn), n_layers)
    # roofline: global layers read B*H_KV*kv*D*2(dtype)*2(K+V) bytes minimum
    bytes_g = 32 * 1 * 8192 * 512 * 2 * 2
    print(f"  (global-layer KV bytes/op: {bytes_g/1e6:.0f} MB "
          f"-> DRAM roofline {bytes_g/3.35e12*1e6:.1f} us/op)")
    print(f"  => attention per decoded token: {total:.2f} ms\n")


def w5():
    print("W5: MoE-26B-A4B training, N=8192, fwd+bwd")
    total = 0.0
    for (H_Q, H_KV, D, slide, n_layers, tag) in [
            (16, 2, 512, 0, 6, "full 16:2 D=512"),
            (16, 8, 256, 1024, 24, "sliding 16:8 D=256")]:
        q, k, v = qkv(1, H_Q, H_KV, 8192, D, rg=True)
        do = torch.randn_like(q)

        def fb():
            out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=slide)
            out.backward(do)
            q.grad = k.grad = v.grad = None
        total += report(f"train fwd+bwd {tag}", tag, time_fn(fb), n_layers)
    print(f"  => attention per model fwd+bwd step: {total:.2f} ms\n")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    torch.manual_seed(0)
    fns = {"W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5}
    for name, fn in fns.items():
        if which in ("all", name):
            fn()
