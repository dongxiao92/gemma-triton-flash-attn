"""Hopper (H100 / sm_90) functional smoke test.

Covers the exact Gemma-4-E2B-it attention shapes:
  - global layers  (7):  H_Q=8, H_KV=1, D=512, full causal
  - sliding layers (28): H_Q=8, H_KV=1, D=256, causal + window=512

For each config, runs
  1. inference forward  (attention_flash_gqa)   vs SDPA reference
  2. training fwd+bwd   (flash_attn_gqa_train)  vs SDPA reference autograd

and reports fp32 cosine similarity + max abs diff. Triton resource errors
(shared-memory OOM etc.) are caught and reported per-config instead of
aborting the sweep.

Usage:
    python tests/test_hopper_functional.py            # full sweep
    python tests/test_hopper_functional.py --quick    # short-N subset
"""
from __future__ import annotations

import argparse
import sys
import traceback

import torch

from gemma_triton_flash_attn import (
    attention_flash_gqa,
    flash_attn_gqa_train,
    attention_gqa_ref,
    attention_swa_ref,
)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # fp32 accumulate — fp16 cos sim under-reports (see skills/2026-04-17)
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def ref_attention(q, k, v, causal, slide):
    if slide > 0 and slide < q.shape[2]:
        return attention_swa_ref(q, k, v, slide)
    return attention_gqa_ref(q, k, v, causal=causal)


def run_config(B, H_Q, H_KV, N, D, causal, slide, dtype, mode):
    torch.manual_seed(0)
    q = torch.randn(B, H_Q, N, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H_KV, N, D, dtype=dtype, device="cuda")

    if mode == "fwd":
        out = attention_flash_gqa(q, k, v, causal=causal, slide_size=slide)
        torch.cuda.synchronize()
        ref = ref_attention(q, k, v, causal, slide)
        return {"out": (cos_sim(out, ref), (out - ref).abs().max().item())}

    # fwd+bwd
    qt, kt, vt = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = flash_attn_gqa_train(qt, kt, vt, causal=causal, slide_size=slide)
    do = torch.randn_like(out)
    out.backward(do)
    torch.cuda.synchronize()

    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    ref = ref_attention(qr, kr, vr, causal, slide)
    ref.backward(do)

    res = {"out": (cos_sim(out, ref), (out - ref).abs().max().item())}
    for name, a, b in (("dq", qt.grad, qr.grad), ("dk", kt.grad, kr.grad),
                       ("dv", vt.grad, vr.grad)):
        res[name] = (cos_sim(a, b), (a - b).abs().max().item())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    Ns = [512, 2048] if args.quick else [128, 512, 1024, 2048, 4096]
    configs = []
    for N in Ns:
        # Gemma-4-E2B-it global attention layers
        configs.append((1, 8, 1, N, 512, True, 0))
        # Gemma-4-E2B-it sliding attention layers
        configs.append((1, 8, 1, N, 256, True, 512))
    if not args.quick:
        configs.append((2, 8, 1, 1024, 512, True, 0))    # B=2 global
        configs.append((2, 8, 1, 1024, 256, True, 512))  # B=2 sliding
        configs.append((1, 8, 1, 1024, 512, False, 0))   # non-causal D=512

    print(f"GPU: {torch.cuda.get_device_name(0)}  dtype={args.dtype}")
    print(f"{'config':<46} {'mode':<7} {'result'}")
    print("-" * 110)
    n_fail = 0
    for (B, H_Q, H_KV, N, D, causal, slide) in configs:
        tag = f"B={B} H_Q={H_Q} H_KV={H_KV} N={N:>5} D={D} c={int(causal)} s={slide}"
        for mode in ("fwd", "fwd+bwd"):
            try:
                res = run_config(B, H_Q, H_KV, N, D, causal, slide, dtype, mode)
            except Exception as e:
                msg = str(e).replace("\n", " ")[:120]
                print(f"{tag:<46} {mode:<7} ERROR {type(e).__name__}: {msg}")
                n_fail += 1
                continue
            ok = all(cs > 0.999 for cs, _ in res.values())
            detail = " ".join(f"{k}:{cs:.6f}/{mx:.1e}" for k, (cs, mx) in res.items())
            print(f"{tag:<46} {mode:<7} {'PASS' if ok else 'FAIL'} {detail}")
            if not ok:
                n_fail += 1

    print("-" * 110)
    print("ALL PASS" if n_fail == 0 else f"{n_fail} FAILURES")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
