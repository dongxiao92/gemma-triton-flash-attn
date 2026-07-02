"""Varlen / multi-sample packing correctness.

Packs samples of mixed lengths into one (1, H, T, D) stream and checks
fwd output + dq/dk/dv against running the (already-validated) dense
kernel per sample. Covers the Gemma-4-E2B-it shapes: D=512 full causal
and D=256 sliding (window=512), GQA 8:1, plus a 2:1 spot check.

Also exercises the HF adapter path with FlashAttentionKwargs-style
cu_seq_lens kwargs (transformers DataCollatorWithFlattening contract).

Usage: python tests/test_varlen_packing.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from gemma_triton_flash_attn import (
    flash_attn_gqa_train,
    flash_attn_gqa_varlen_train,
    triton_gqa_attention,
)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


def run_case(seqlens, H_Q, H_KV, D, slide, dtype, tag):
    torch.manual_seed(0)
    T = sum(seqlens)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)),
                      dtype=torch.int32, device="cuda")
    q = torch.randn(1, H_Q, T, D, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(1, H_KV, T, D, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(1, H_KV, T, D, dtype=dtype, device="cuda", requires_grad=True)
    do = torch.randn(1, H_Q, T, D, dtype=dtype, device="cuda")

    # packed varlen
    out = flash_attn_gqa_varlen_train(q, k, v, cu, max(seqlens),
                                      causal=True, slide_size=slide)
    out.backward(do)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # reference: dense kernel per sample
    out_ref = torch.empty_like(out)
    dq_ref = torch.empty_like(dq)
    dk_ref = torch.empty_like(dk)
    dv_ref = torch.empty_like(dv)
    for i in range(len(seqlens)):
        s, e = int(cu[i]), int(cu[i + 1])
        qs = q.detach()[:, :, s:e].clone().requires_grad_(True)
        ks = k.detach()[:, :, s:e].clone().requires_grad_(True)
        vs = v.detach()[:, :, s:e].clone().requires_grad_(True)
        o = flash_attn_gqa_train(qs, ks, vs, causal=True, slide_size=slide)
        o.backward(do[:, :, s:e])
        out_ref[:, :, s:e] = o
        dq_ref[:, :, s:e] = qs.grad
        dk_ref[:, :, s:e] = ks.grad
        dv_ref[:, :, s:e] = vs.grad

    res = {"out": (cos(out, out_ref), (out - out_ref).abs().max().item()),
           "dq": (cos(dq, dq_ref), (dq - dq_ref).abs().max().item()),
           "dk": (cos(dk, dk_ref), (dk - dk_ref).abs().max().item()),
           "dv": (cos(dv, dv_ref), (dv - dv_ref).abs().max().item())}
    ok = all(c > 0.999 for c, _ in res.values())
    detail = " ".join(f"{n}:{c:.6f}/{m:.1e}" for n, (c, m) in res.items())
    print(f"  {tag:<52} {'PASS' if ok else 'FAIL'} {detail}")
    return ok


def test_adapter_packed():
    """Adapter with cu_seq_lens kwargs vs per-sample adapter calls."""
    torch.manual_seed(1)
    seqlens = [700, 1348, 45, 2003]
    T = sum(seqlens)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)),
                      dtype=torch.int64, device="cuda")  # HF passes int64
    D, H_Q, H_KV, slide = 256, 8, 1, 512
    q = torch.randn(1, H_Q, T, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, H_KV, T, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, H_KV, T, D, dtype=torch.bfloat16, device="cuda")
    module = SimpleNamespace(is_causal=True, head_dim=D)

    out, _ = triton_gqa_attention(
        module, q, k, v, None, scaling=D ** -0.5, sliding_window=slide,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu,
        max_length_q=max(seqlens), max_length_k=max(seqlens),
    )
    ok = True
    for i in range(len(seqlens)):
        s, e = int(cu[i]), int(cu[i + 1])
        o_i, _ = triton_gqa_attention(
            module, q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], None,
            scaling=D ** -0.5, sliding_window=slide,
        )
        c = cos(out[:, s:e], o_i)
        ok &= c > 0.999
        print(f"  adapter sample {i} (len {seqlens[i]:>5}): cos={c:.6f} "
              f"{'OK' if c > 0.999 else 'FAIL'}")
    return ok


def main():
    n_fail = 0
    print("=== kernel-level: packed vs per-sample dense ===")
    cases = [
        # (seqlens, H_Q, H_KV, D, slide, dtype)
        ([2048], 8, 1, 512, 0, torch.float16),            # single seq == dense
        ([1024, 1024], 8, 1, 512, 0, torch.float16),
        ([1000, 37, 2048, 511], 8, 1, 512, 0, torch.float16),
        ([1000, 37, 2048, 511], 8, 1, 256, 512, torch.float16),
        ([3000, 900, 4196], 8, 1, 512, 0, torch.bfloat16),
        ([3000, 900, 4196], 8, 1, 256, 512, torch.bfloat16),
        ([256, 256, 256, 256, 256, 256, 256, 256], 8, 1, 256, 512, torch.float16),
        ([777, 1271], 16, 8, 256, 1024, torch.float16),   # MoE sliding shape
        ([17, 3], 8, 1, 512, 0, torch.float16),           # tiny samples
    ]
    for (sls, hq, hkv, d, sl, dt) in cases:
        tag = f"seqlens={sls} D={d} slide={sl} {str(dt)[6:]}"
        if len(tag) > 52:
            tag = tag[:49] + "..."
        try:
            ok = run_case(sls, hq, hkv, d, sl, dt, tag)
        except Exception as e:
            print(f"  {tag:<52} ERROR {type(e).__name__}: {str(e)[:90]}")
            ok = False
        n_fail += 0 if ok else 1

    print("=== adapter-level: FlashAttentionKwargs contract ===")
    try:
        n_fail += 0 if test_adapter_packed() else 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        n_fail += 1

    print("ALL PASS" if n_fail == 0 else f"{n_fail} FAILURES")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
