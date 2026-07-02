"""FA-compatibility layer: numerics + no-regression checks.

References are computed with FA *semantics* (layouts, window_size,
softmax_scale, bottom-right causal alignment) via eager SDPA, so a pass
here means user code written against flash-attention 2.x gets identical
behavior from the Triton kernels.

Usage: python tests/test_fa_compat.py [--perf]
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from gemma_triton_flash_attn.fa_compat import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    install,
)


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def ref_fa(q, k, v, causal, window_left, softmax_scale=None):
    """Eager reference in FA layout/semantics. q: (B,Nq,H,D), k/v: (B,Nk,Hk,D)."""
    B, Nq, H, D = q.shape
    _, Nk, Hk, _ = k.shape
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float().repeat_interleave(H // Hk, dim=1)
    vt = v.transpose(1, 2).float().repeat_interleave(H // Hk, dim=1)
    scores = qt @ kt.transpose(-1, -2) * scale
    qi = torch.arange(Nq, device=q.device)[:, None] + (Nk - Nq)  # bottom-right
    kj = torch.arange(Nk, device=q.device)[None, :]
    mask = torch.ones(Nq, Nk, dtype=torch.bool, device=q.device)
    if causal:
        mask &= kj <= qi
    if window_left >= 0:
        mask &= (qi - kj) <= window_left
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ vt
    return out.transpose(1, 2).to(q.dtype)


def check(tag, out, ref, n_fail, thresh=0.999):
    c, m = cos(out, ref), (out - ref).abs().max().item()
    ok = c > thresh
    print(f"  {tag:<58} {'PASS' if ok else 'FAIL'} cos={c:.6f} max={m:.1e}")
    return n_fail + (0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    nf = 0

    print("=== flash_attn_func (dense, FA layout) fwd+bwd ===")
    for (B, N, H, Hk, D, causal, wl, scale, dt) in [
        (1, 2048, 8, 1, 512, True, -1, None, torch.bfloat16),   # E2B global
        (1, 2048, 8, 1, 256, True, 511, None, torch.bfloat16),  # E2B sliding
        (2, 1024, 8, 1, 512, True, -1, 0.5, torch.float16),     # custom scale
        (2, 777, 16, 8, 256, True, 1023, None, torch.float16),  # MoE sliding, odd N
        (1, 512, 8, 8, 128, False, -1, None, torch.float16),    # MHA non-causal
        (1, 1024, 12, 4, 128, True, -1, None, torch.bfloat16),  # Llama-ish 12:4
    ]:
        q = torch.randn(B, N, H, D, dtype=dt, device="cuda", requires_grad=True)
        k = torch.randn(B, N, Hk, D, dtype=dt, device="cuda", requires_grad=True)
        v = torch.randn(B, N, Hk, D, dtype=dt, device="cuda", requires_grad=True)
        ws = (wl, -1) if wl >= 0 else (-1, -1)
        out = flash_attn_func(q, k, v, causal=causal, window_size=ws,
                              softmax_scale=scale)
        do = torch.randn_like(out)
        out.backward(do)
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        ref = ref_fa(qr, kr, vr, causal, wl, scale)
        ref.backward(do)
        tag = f"B={B} N={N} {H}:{Hk} D={D} c={int(causal)} wl={wl} s={scale}"
        nf = check(tag + " out", out, ref, nf)
        nf = check(tag + " dq", q.grad, qr.grad, nf)
        nf = check(tag + " dk", k.grad, kr.grad, nf)
        nf = check(tag + " dv", v.grad, vr.grad, nf)

    print("=== packed variants ===")
    B, N, H, D = 2, 1024, 8, 256
    qkv = torch.randn(B, N, 3, H, D, dtype=torch.bfloat16, device="cuda")
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    ref = ref_fa(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], True, -1)
    nf = check("qkvpacked", out, ref, nf)
    q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, N, 2, 1, D, dtype=torch.bfloat16, device="cuda")
    out = flash_attn_kvpacked_func(q, kv, causal=True)
    ref = ref_fa(q, kv[:, :, 0], kv[:, :, 1], True, -1)
    nf = check("kvpacked (GQA 8:1)", out, ref, nf)

    print("=== flash_attn_varlen_func (packing) fwd+bwd ===")
    for (seqlens, D, wl) in [([1000, 37, 2048, 511], 512, -1),
                             ([1000, 37, 2048, 511], 256, 511),
                             ([3000, 900], 256, 1023)]:
        T = sum(seqlens)
        cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)),
                          dtype=torch.int32, device="cuda")
        q = torch.randn(T, 8, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k = torch.randn(T, 1, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v = torch.randn(T, 1, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        ws = (wl, -1) if wl >= 0 else (-1, -1)
        out = flash_attn_varlen_func(q, k, v, cu, cu, max(seqlens), max(seqlens),
                                     causal=True, window_size=ws)
        do = torch.randn_like(out)
        out.backward(do)
        # per-sample reference
        ref = torch.empty_like(out)
        dq_r = torch.empty_like(q)
        dk_r = torch.empty_like(k)
        dv_r = torch.empty_like(v)
        for i in range(len(seqlens)):
            s0, e0 = int(cu[i]), int(cu[i + 1])
            qs = q.detach()[s0:e0].unsqueeze(0).clone().requires_grad_(True)
            ks = k.detach()[s0:e0].unsqueeze(0).clone().requires_grad_(True)
            vs = v.detach()[s0:e0].unsqueeze(0).clone().requires_grad_(True)
            o = ref_fa(qs, ks, vs, True, wl)
            o.backward(do[s0:e0].unsqueeze(0))
            ref[s0:e0] = o[0]
            dq_r[s0:e0] = qs.grad[0]
            dk_r[s0:e0] = ks.grad[0]
            dv_r[s0:e0] = vs.grad[0]
        tag = f"varlen {seqlens} D={D} wl={wl}"
        nf = check(tag + " out", out, ref, nf)
        nf = check(tag + " dq", q.grad, dq_r, nf)
        nf = check(tag + " dk", k.grad, dk_r, nf)
        nf = check(tag + " dv", v.grad, dv_r, nf)

    print("=== flash_attn_with_kvcache (decode) ===")
    for (B, Nq, cache_len, used, D, wl, append) in [
            (32, 1, 8192, 8192, 512, -1, False),
            (32, 1, 8192, 4000, 512, -1, True),
            (8, 1, 1024, 1000, 256, 511, True),
            (4, 4, 4096, 4092, 512, -1, True)]:
        q = torch.randn(B, Nq, 8, D, dtype=torch.bfloat16, device="cuda")
        kc = torch.randn(B, cache_len, 1, D, dtype=torch.bfloat16, device="cuda")
        vc = torch.randn(B, cache_len, 1, D, dtype=torch.bfloat16, device="cuda")
        ws = (wl, -1) if wl >= 0 else (-1, -1)
        if append:
            kn = torch.randn(B, Nq, 1, D, dtype=torch.bfloat16, device="cuda")
            vn = torch.randn(B, Nq, 1, D, dtype=torch.bfloat16, device="cuda")
            seqlens = torch.full((B,), used - Nq, dtype=torch.int32, device="cuda")
            out = flash_attn_with_kvcache(q, kc, vc, k=kn, v=vn,
                                          cache_seqlens=seqlens, causal=True,
                                          window_size=ws)
        else:
            out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=used,
                                          causal=True, window_size=ws)
        ref = ref_fa(q, kc[:, :used], vc[:, :used], True, wl)
        nf = check(f"kvcache B={B} Nq={Nq} used={used} D={D} wl={wl} "
                   f"append={append}", out, ref, nf)

    print("=== install() shim ===")
    mod = install()
    import flash_attn as fa  # noqa: F401 — resolves to the shim
    assert fa.flash_attn_func is flash_attn_func
    assert fa.__version__.endswith("gemma_triton_compat")
    print("  import flash_attn -> shim OK")

    print("=== deterministic=True bit-identical bwd ===")
    q = torch.randn(1, 1024, 8, 512, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    k = torch.randn(1, 1024, 1, 512, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    v = torch.randn(1, 1024, 1, 512, dtype=torch.bfloat16, device="cuda",
                    requires_grad=True)
    do = torch.randn(1, 1024, 8, 512, dtype=torch.bfloat16, device="cuda")
    grads = []
    for _ in range(2):
        out = flash_attn_func(q, k, v, causal=True, deterministic=True)
        out.backward(do)
        grads.append((q.grad.clone(), k.grad.clone(), v.grad.clone()))
        q.grad = k.grad = v.grad = None
    same = all(torch.equal(a, b) for a, b in zip(*grads))
    print(f"  two runs bit-identical: {'PASS' if same else 'FAIL'}")
    nf += 0 if same else 1

    if args.perf:
        print("=== perf: compat overhead vs native API ===")
        from gemma_triton_flash_attn import flash_attn_gqa_train

        def timeit(fn, warmup=5, rep=20):
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

        for (N, D, wl) in [(8192, 512, -1), (8192, 256, 511)]:
            # native layout tensors
            qn = torch.randn(1, 8, N, D, dtype=torch.bfloat16, device="cuda",
                             requires_grad=True)
            kn = torch.randn(1, 1, N, D, dtype=torch.bfloat16, device="cuda",
                             requires_grad=True)
            vn = torch.randn(1, 1, N, D, dtype=torch.bfloat16, device="cuda",
                             requires_grad=True)
            don = torch.randn_like(qn)
            # FA layout tensors
            qf = torch.randn(1, N, 8, D, dtype=torch.bfloat16, device="cuda",
                             requires_grad=True)
            kf = torch.randn(1, N, 1, D, dtype=torch.bfloat16, device="cuda",
                             requires_grad=True)
            vf = torch.randn(1, N, 1, D, dtype=torch.bfloat16, device="cuda",
                             requires_grad=True)
            dof = torch.randn_like(qf)
            slide = wl + 1 if wl >= 0 else 0
            ws = (wl, -1) if wl >= 0 else (-1, -1)

            def native():
                out = flash_attn_gqa_train(qn, kn, vn, causal=True,
                                           slide_size=slide)
                out.backward(don)
                qn.grad = kn.grad = vn.grad = None

            def compat():
                out = flash_attn_func(qf, kf, vf, causal=True, window_size=ws)
                out.backward(dof)
                qf.grad = kf.grad = vf.grad = None

            tn, tc = timeit(native), timeit(compat)
            print(f"  N={N} D={D} wl={wl}: native {tn:8.3f} ms | "
                  f"fa_compat {tc:8.3f} ms | overhead {100*(tc/tn-1):+5.1f}%")

    print("ALL PASS" if nf == 0 else f"{nf} FAILURES")
    sys.exit(0 if nf == 0 else 1)


if __name__ == "__main__":
    main()
