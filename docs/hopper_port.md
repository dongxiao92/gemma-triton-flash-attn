# Hopper (H100 / sm_90) port — report

Task: `hopper.md` — port `flash_attn/attention.py` to H100, pass the
gemma-4-E2B-it functionality test against transformers, then optimize
toward >70% SOL. Work tracked on branch `hopper-port`.

Environment: computelab-sc-01, 1× H100 80GB HBM3 (sm_90, `h100-80gb-hbm3@ts6`
partition), torch 2.9.1+cu128, triton 3.5.1, transformers 5.5.4
(`.venv` in repo root; `TRITON_CACHE_DIR` must point at scratch — compute
nodes cannot write `$HOME`).

## 1. What actually broke on Hopper

The suspected failure ("global attention not supported, related to
headdim=512") **did not reproduce** at the kernel level: every
equal-length shape — including D=512 global attention up to N=32K,
fwd and fwd+bwd — passed on H100 unchanged, on both triton 3.5.1 and
3.7.1, and the full-model prefill logits test passed at N=512..4096
(cos ≥ 0.99997, top-1 100%).

The real failure was **`model.generate()`**: the first decode step calls
attention with `q_len=1, kv_len=t`. The kernel had a single `SEQ_LEN`
for Q and KV (no cross-length concept), and the block-size clamp
`min(BLOCK, next_power_of_2(N))` produced BLOCK_Q=1 < 16, which
`tl.dot` rejects → Triton `CompilationError` on the fwd kernel. For an
instruction-tuned model, generation is *the* functionality test, and the
crash surfaces in the D=512 stack trace — hence the "headdim=512" guess.

### Fixes (commit `ab9e6be`)

- `KV_OFFSET` runtime arg (default 0) in the fwd kernel: q rows are the
  suffix of the KV stream; causal diagonal, SWA window, and KV bounds
  shift by `kv_len - q_len`. Default keeps all existing callers valid.
- Block sizes clamped to ≥16 (`tl.dot` minimum); masked padding rows
  handle tiny/odd lengths.
- `flash_attn_gqa_train` routes cross-length calls (inference-only) to
  the fwd kernel; the autograd path asserts equal lengths.
- New tests: `tests/test_kv_cache_decode.py` (suffix-attention
  reference: decode steps, chunked prefill continuation, trimmed sliding
  cache, tiny-N), `tests/test_hopper_functional.py` (E2B shape sweep).

### Functionality validation (gemma-4-E2B-it, H100, bf16)

| check | result |
|---|---|
| adapter unit tests (`test_adapter.py`) | 28/28 PASS |
| prefill logits vs SDPA, N=512 / 4096 | cos 0.99997 / 0.99998, top-1 100%, top-5 5/5 |
| kernel sweep N=128..32K, both head dims, fwd+bwd | ALL PASS (fp32 cos > 0.99998) |
| KV-cache decode suite | 10/10 PASS |
| greedy generation, short prompt | 6/6 tokens == SDPA |
| greedy generation, 543-token prompt (crosses SWA window) | first 13 tokens ==, then benign near-tie divergence; both coherent |
| teacher-forced stepwise decode parity (648-token prompt, 24 steps) | 0 top-1 flips, per-step logits cos ≥ 0.99998 |

Greedy trajectories can diverge on near-ties (top-2 margin ≲ bf16
attention noise); the teacher-forced stepwise check is the meaningful
parity metric.

## 2. Optimization (commits `300060f`, `c7a3f5c`)

NCU baseline on the E2B global-attention fwd kernel (D=512, N=4K):
Compute 33.6%, Memory 45.4%, DRAM 1.7%, L1/TEX 59.8% — latency-bound at
1 CTA/SM (192KB smem + 255 regs both cap occupancy at 8 warps/SM), with
K/V re-streamed through L1 by all 8 Q-head programs (H_KV=1 keeps K/V in
L2, so HBM is idle — the "memory-bound at 84-90% HBM" note in the old
docs applies to H_KV≥4 shapes at long N, not E2B).

What shipped:

1. **Pack-GQA fwd kernel** (`_flash_attn_gqa_fwd_packed_kernel`): tile
   rows map to `(q_pos, q_head)` pairs (`QPOS = BLOCK_Q / GQA_RATIO`),
   one `tl.dot` serves the whole GQA group → K/V L1 traffic ÷ 8 on E2B.
   Exactly one accumulator (the 2026-04 grouped-kernel failure was one
   accumulator *per head*). Same grid size and per-program tensor work.
2. **TMA tensor descriptors** for K/V (fwd), K/V (dQ), Q/dO (dKV):
   hardware bulk copies replace per-element `cp.async` address
   arithmetic. Pointer-load fallback kept via trailing default args.
3. **3-phase KV loop** (masked left edge / unmasked middle / masked
   diagonal): SWA no longer pays window-mask + NaN-clamp on ~80% of
   tiles; D=512 causal gets the unmasked split too (the classic kernel
   disabled it at D=512). Note: Triton `//` truncates toward zero, so
   phase bounds must keep division operands non-negative.
4. **dKV retune under TMA**: (BKV=32, BQ=64, w=8, s=2) for D=512 — TMA
   frees the address registers that previously made BKV=32 spill
   catastrophically (20.5 ms → 6.5 ms; −28% vs the BKV=16 default).
5. **Size gate**: TMA launch costs ~20 µs host-side, which regresses
   small SWA kernels → packed/TMA paths require `D ≥ 512 or N_KV ≥ 8192`.
   Short-N behavior is byte-identical to the classic kernels.

Failed experiments (documented so they are not retried):
`num_warps=16` (Triton caps 128 regs/thread → mass spills, several PTX
codegen errors), `tl.range(warp_specialize=True)` (no-op on sm_90 in
triton 3.5.1 and 3.7.1), `num_stages≥3` at D=512 (smem > 227KB),
BKV=64 fwd tiles at s=2 (smem), classic-kernel block re-sweep (the
2026-04 defaults are already optimal on H100 — 15-config sweep), dQ
config re-sweep (current default optimal).

### Kernel wall-clock (E2B shapes, B=1 H_Q=8 H_KV=1, fp16)

| kernel | N | before | after | Δ |
|---|---|---|---|---|
| fwd D=512 causal | 8K | 3.10 ms | 2.61 ms | −16% |
| fwd D=512 causal | 16K | 43.3 ms | 38.0 ms | −12% |
| fwd D=256 SWA512 | 16K | 0.326 ms | 0.307 ms | −6% |
| fwd+bwd D=512 | 8K | 16.0 ms | 13.4 ms | −16% |
| fwd+bwd D=256 SWA | 16K | 1.39 ms | 1.07 ms | −23% |

Tier-1 vs SDPA (fp16): F config (D=512 GQA 8:1) fwd 1.38× @1K → 2.24×
@16K (was 1.98× peak); fwd+bwd 3.27× @1K / 3.38× @4K / 3.00× @16K
(was 2.90 / 2.82 / 2.53). Config B fwd+bwd @16K 10.6× → 12.1×.

### NCU Speed-of-Light, final state (E2B shapes)

| kernel | shape | Compute SOL | Memory SOL | L1/TEX | DRAM |
|---|---|---|---|---|---|
| **dQ D=512** | N=8K | 29.1% | **75.6%** | **80.6%** | 1.8% |
| dKV D=512 | N=8K | 36.7% | 59.9% | 61.0% | 4.9% |
| fwd D=512 | N=8K | 38.6% | 60.0% | 68.4% | 1.3% |
| fwd D=512 | N=32K | 41.0% | 64.2% | 65.7% | 1.0% |
| dKV D=256 SWA | N=16K | 40.7% | 59.3% | 62.0% | 8.9% |
| fwd D=256 SWA | N=16K | 39.7% | 37.0% | 39.8% | 16.2% |
| dQ D=256 SWA | N=16K | 34.2% | 36.7% | 37.3% | 26.8% |

The dQ D=512 kernel exceeds the 70% SOL target (bound: L1TEX/shared —
wgmma operand reads — at 80.6%). fwd/dKV D=512 sit at 60-68%: the
binding resource is the same smem/L1TEX path, and the residual gap is an
occupancy ceiling Triton 3.5 cannot cross on sm_90 — the fp32
accumulator (BLOCK_Q×512) plus 192KB smem pin the kernels at 1 CTA/SM
(2 active warps/scheduler, issue rate 0.45 instr/cycle measured), and
the tools that close this gap in FA3/cuDNN (warp-specialized
producer/consumer pipelines, pingpong scheduling between warpgroups,
register-resident wgmma A-operands) are not expressible in Triton 3.5
(`warp_specialize=True` is accepted but has no effect on sm_90; verified
on 3.7.1 as well). The SWA kernels are latency-bound by short KV loops
(≤10 tiles/program at slide=512); their absolute times are sub-ms and
the E2E impact is small.

Reproduce: `benchmarks/hopper_ncu_sol.py` (NCU at
`/home/scratch.xiaod_sw/dynamic-kernel-generator/cuda-12.9.83-20250520/ncu`).

## 3. What did NOT need porting

- All equal-length kernel functionality (incl. D=512) worked on sm_90
  as-is with triton 3.5.1 — the old tuning (block sizes, warps, stages,
  exp2 softmax, split-causal, pack-GQA dKV, Q_SPLITS heuristics) holds
  on H100; the backward config sweep confirmed every default.
- The HF adapter, multimodal image-mask path, FSDP2 patches: untouched.
