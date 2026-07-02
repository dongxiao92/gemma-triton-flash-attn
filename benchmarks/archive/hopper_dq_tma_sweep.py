"""dQ config sweep WITH TMA descriptors (was only swept pointer-mode)."""
import math, sys, torch, triton
sys.path.insert(0, '/home/scratch.xiaod_gpu_5/gemma-triton-flash-attn')
from triton.tools.tensor_descriptor import TensorDescriptor
from flash_attn.attention import (
    _flash_attn_gqa_bwd_dq_kernel, FlashAttnGQAFunction,
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
    q = torch.randn(1, 8, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 1, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1, N, D, dtype=torch.float16, device="cuda")
    do = torch.randn(1, 8, N, D, dtype=torch.float16, device="cuda")
    class Grab:
        def save_for_backward(self, *ts): self.saved = ts
    ctx = Grab()
    o = FlashAttnGQAFunction.forward(ctx, q, k, v, True, slide, None, None, None)
    lse = ctx.saved[4]
    delta = (do.float() * o.float()).sum(-1)
    return q, k, v, do, o, lse, delta

def run_dq(q, k, v, do, o, lse, delta, slide, BQ, BKV, w, s, use_tma=True):
    B, H_Q, N, D = q.shape
    _, H_KV, _, _ = k.shape
    dq = torch.empty_like(q)
    grid = (triton.cdiv(N, BQ), B * H_Q)
    if use_tma:
        kd = TensorDescriptor.from_tensor(k.reshape(B*H_KV*N, D), block_shape=[BKV, D])
        vd = TensorDescriptor.from_tensor(v.reshape(B*H_KV*N, D), block_shape=[BKV, D])
    else:
        kd = vd = None
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
        K_desc=kd, V_desc=vd, USE_TMA=use_tma,
        num_warps=w, num_stages=s)
    return dq

for (D, slide, tag) in [(512, 0, "D=512 global"), (256, 512, "D=256 SWA")]:
    q, k, v, do, o, lse, delta = setup(D, slide)
    print(f"=== dQ+TMA {tag} N=8192 ===")
    cur = (32, 64, 8, 2) if D >= 512 else (64, 64, 4, 2)
    ref = run_dq(q, k, v, do, o, lse, delta, slide, *cur, use_tma=False)
    for cfg in [cur, (32,64,8,3), (32,128,8,1), (32,128,8,2), (64,64,8,2),
                (64,64,8,1), (32,32,8,2), (64,128,8,1), (32,64,4,2),
                (64,64,4,2), (64,32,8,2), (16,128,8,2), (32,256,8,1)]:
        try:
            t = time_fn(lambda: run_dq(q, k, v, do, o, lse, delta, slide, *cfg))
            d = run_dq(q, k, v, do, o, lse, delta, slide, *cfg)
            ok = torch.allclose(d.float(), ref.float(), atol=3e-2, rtol=1e-2)
            print(f"  BQ={cfg[0]:>3} BKV={cfg[1]:>3} w={cfg[2]} s={cfg[3]}: {t:8.3f}ms {'OK' if ok else 'BAD'} {'<- cur' if cfg==cur else ''}")
        except Exception as e:
            print(f"  BQ={cfg[0]:>3} BKV={cfg[1]:>3} w={cfg[2]} s={cfg[3]}: FAIL {str(e)[:45]}")
