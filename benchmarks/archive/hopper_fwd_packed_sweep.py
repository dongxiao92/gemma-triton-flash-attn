import math, sys, torch, triton
sys.path.insert(0, '/home/scratch.xiaod_gpu_5/gemma-triton-flash-attn')
sys.path.insert(0, '/home/scratch.xiaod_gpu_5/gemma-triton-flash-attn/hopper_work')
from bench_packed_fwd import run_packed, time_fn, cs
from flash_attn.attention import attention_flash_gqa, _flash_attn_gqa_fwd_packed_kernel

def get_ck():
    dc = _flash_attn_gqa_fwd_packed_kernel.device_caches
    bc, *_ = dc[0]
    return list(bc.values())[-1]

N = 8192
torch.manual_seed(0)
for (D, slide, tag, configs) in [
    (512, 0, "global D=512", [
        (64, 32, 8, 2),
        (64, 32, 16, 2), (64, 64, 16, 1), (128, 16, 16, 2), (128, 32, 16, 1),
        (64, 16, 16, 3), (128, 64, 16, 1), (64, 64, 16, 2), (256, 16, 16, 1),
    ]),
    (256, 512, "sliding D=256 slide=512", [
        (128, 64, 8, 2),
        (128, 64, 16, 2), (128, 128, 16, 1), (256, 32, 16, 2), (256, 64, 16, 1),
        (128, 32, 16, 2), (256, 128, 16, 1), (128, 128, 16, 2),
    ]),
]:
    q = torch.randn(1, 8, N, D, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 1, N, D, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1, N, D, dtype=torch.float16, device="cuda")
    ref = attention_flash_gqa(q, k, v, causal=True, slide_size=slide)
    if slide > 0:
        rows = torch.arange(N); keys = torch.minimum(rows + 1, torch.tensor(slide)).sum().item()
        flops = 4.0 * 8 * keys * D
    else:
        flops = 2.0 * 8 * N * N * D
    print(f"\n=== packed {tag} N={N} ===")
    print(f"{'BQ':>3} {'BKV':>3} {'w':>2} {'s':>2} | {'ms':>7} | {'TF':>4} | {'MFU%':>5} | {'shmem':>8} | {'regs':>4} | {'spl':>4} | ok")
    best = None
    for BQ, BKV, w, s in configs:
        try:
            fn = lambda: run_packed(q, k, v, causal=True, slide=slide, BQ=BQ, BKV=BKV, w=w, s=s)
            out = fn(); ck = get_ck()
            ms = time_fn(fn)
            c = cs(out, ref)
            tf = flops / (ms/1e3) / 1e12
            print(f"{BQ:>3} {BKV:>3} {w:>2} {s:>2} | {ms:>7.3f} | {tf:>4.0f} | {100*tf/989:>5.1f} | "
                  f"{ck.metadata.shared/1024:>6.1f}KB | {ck.n_regs:>4} | {ck.n_spills:>4} | {'Y' if c>0.9999 else 'FAIL'}")
            if c > 0.9999 and (best is None or ms < best[0]):
                best = (ms, BQ, BKV, w, s)
        except Exception as ex:
            print(f"{BQ:>3} {BKV:>3} {w:>2} {s:>2} | FAIL: {str(ex)[:50]}")
    if best:
        print(f"BEST: {best}")
