"""Sweep dQ and packed-dKV configs on E2B shapes @ N=8192 (H100)."""
import math, sys, torch, triton
sys.path.insert(0, '/home/scratch.xiaod_gpu_5/gemma-triton-flash-attn')
from flash_attn.attention import (
    _flash_attn_gqa_bwd_dq_kernel, _flash_attn_gqa_bwd_dkv_packed_kernel,
    FlashAttnGQAFunction,
)

def time_fn(fn, warmup=5, rep=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep

def setup(D, slide, N=8192):
    torch.manual_seed(0)
    q = torch.randn(1, 8, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(1, 1, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(1, 1, N, D, dtype=torch.float16, device="cuda", requires_grad=True)
    out = FlashAttnGQAFunction.apply(q, k, v, True, slide, None, None, None)
    do = torch.randn_like(out)
    # recover lse/delta via a manual fwd (apply saved them on ctx; easier to rerun)
    from flash_attn.attention import attention_flash_gqa
    lse = torch.empty(1, 8, N, dtype=torch.float32, device="cuda")
    # run fwd kernel with STORE_LSE via autograd ctx trick: redo through Function
    class Grab:
        def save_for_backward(self, *ts):
            self.saved = ts
    ctx = Grab()
    o_saved = FlashAttnGQAFunction.forward(ctx, q, k, v, True, slide, None, None, None)
    lse = ctx.saved[4]
    delta = (do.float() * o_saved.float()).sum(-1)
    return q, k, v, do, o_saved, lse, delta

def run_dq(q, k, v, do, o, lse, delta, slide, BQ, BKV, w, s):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    dq = torch.empty_like(q)
    grid = (triton.cdiv(N, BQ), B * H_Q)
    _flash_attn_gqa_bwd_dq_kernel[grid](
        q, k, v, do, o, dq, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0/math.sqrt(D), BLOCK_Q=BQ, BLOCK_KV=BKV,
        IS_CAUSAL=True, SLIDE_SIZE=slide, STORE_DELTA=False,
        GroupIds_ptr=None, GroupLo_ptr=None, GroupHi_ptr=None,
        stride_gb=0, stride_gn=0, HAS_GROUP_IDS=False,
        num_warps=w, num_stages=s)
    return dq

def run_dkv(q, k, v, do, lse, delta, slide, BKV, BQ, w, s, use_tma=True):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    grid = (triton.cdiv(N, BKV), B * H_KV, 1)
    if use_tma:
        from triton.tools.tensor_descriptor import TensorDescriptor
        qd = TensorDescriptor.from_tensor(q.reshape(B*H_Q*N, D), block_shape=[BQ, D])
        dod = TensorDescriptor.from_tensor(do.reshape(B*H_Q*N, D), block_shape=[BQ, D])
    else:
        qd = dod = None
    _flash_attn_gqa_bwd_dkv_packed_kernel[grid](
        q, k, v, do, dk, dv, lse, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        N_Q_HEADS=H_Q, N_KV_HEADS=H_KV, SEQ_LEN=N, HEAD_DIM=D,
        scale=1.0/math.sqrt(D), BLOCK_Q=BQ, BLOCK_KV=BKV,
        GQA_RATIO=H_Q//H_KV, IS_CAUSAL=True, SLIDE_SIZE=slide, Q_SPLITS=1,
        GroupIds_ptr=None, GroupLo_ptr=None, GroupHi_ptr=None,
        stride_gb=0, stride_gn=0, HAS_GROUP_IDS=False,
        Q_desc=qd, DO_desc=dod, USE_TMA=use_tma,
        num_warps=w, num_stages=s)
    return dk, dv

for (D, slide, tag) in [(512, 0, "D=512 global"), (256, 512, "D=256 SWA")]:
    q, k, v, do, o, lse, delta = setup(D, slide)
    print(f"=== dKV(packed+TMA) {tag} N=8192 ===")
    cur = (16, 64, 4, 2) if D >= 512 else (64, 128, 8, 1)
    refk, refv = run_dkv(q, k, v, do, lse, delta, slide, *cur, use_tma=False)
    for cfg in [cur, (16,64,4,3), (16,128,4,2), (16,128,4,1), (32,64,4,2),
                (32,64,8,2), (32,128,4,1), (16,256,4,1), (32,32,4,2),
                (64,64,8,1), (16,64,8,2), (16,128,8,2)]:
        try:
            t = time_fn(lambda: run_dkv(q, k, v, do, lse, delta, slide, *cfg))
            dk2, dv2 = run_dkv(q, k, v, do, lse, delta, slide, *cfg)
            ok = torch.allclose(dk2.float(), refk.float(), atol=6e-2, rtol=1e-2) and \
                 torch.allclose(dv2.float(), refv.float(), atol=6e-2, rtol=1e-2)
            print(f"  BKV={cfg[0]:>3} BQ={cfg[1]:>3} w={cfg[2]} s={cfg[3]}: {t:8.3f}ms {'OK' if ok else 'BAD'} {'<- current' if cfg==cur else ''}")
        except Exception as e:
            print(f"  BKV={cfg[0]:>3} BQ={cfg[1]:>3} w={cfg[2]} s={cfg[3]}: FAIL {str(e)[:45]}")
