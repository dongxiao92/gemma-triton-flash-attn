# Five industry workloads — SOL analysis and tuning (H100)

Third tuning round (branch `hopper-port`). Five workloads that real users
of these Gemma-4 kernels run, analyzed with NCU Speed-of-Light and
optimized without any numerics change (all correctness suites pass
bit-identically to the reference paths). Reproduce timings with
`benchmarks/workloads_sol.py`, SOL with `benchmarks/hopper_ncu_sol.py`.

## The workloads

| # | workload | shapes (E2B unless noted) | mode |
|---|---|---|---|
| W1 | Padding-free SFT (TRL/axolotl style) | packed [2919,1370,3105,798] ≈ 8K, both layer types | fwd+bwd, bf16 |
| W2 | Long-context RAG prefill | B=1, N=32768 | fwd, bf16 |
| W3 | Batched chat serving prefill | B=8, N=2048 | fwd, bf16 |
| W4 | Batched decode (32 sessions) | q=1, KV cache 8192 (global) / 512 (SWA) | fwd, bf16 |
| W5 | MoE-26B-A4B fine-tuning | N=8192, 16:2 D=512 + 16:8 D=256 slide=1024 | fwd+bwd, bf16 |

## What this round changed: split-KV decode (flash-decoding)

W4 baseline was the outlier: a global-attention decode step ran 0.681 ms
against a 0.160 ms DRAM roofline (537 MB of KV per op) — **23% of DRAM
SOL** — because the single-pass kernel launches only `B x H_KV = 32`
programs for 132 SMs. New `_flash_attn_gqa_decode_splitkv_kernel` +
`_flash_attn_gqa_decode_combine_kernel`: the KV range is partitioned
across `~264/(B*H_KV)` splits, each program computes a partial online
softmax (m, l, unnormalized acc), and the combine kernel merges partials
with the exact same online-softmax algebra (no approximation; decode
suite extended with B=32/B=8x32K cases, all PASS vs reference).

Results (B=32, kv=8192, D=512, bf16):

| | before | after |
|---|---|---|
| decode op (global layer) | 0.681 ms | 0.196 ms (**3.5x**) |
| split-KV kernel DRAM SOL | — | **86.5%** (186 µs NCU vs 160 µs roofline) |
| W4 attention / decoded token (35 layers) | 6.22 ms | 2.89 ms (2.15x) |

Routing: `_decode_splitkv_splits` gates on causal suffix queries with
`q_len * GQA_RATIO <= 32`, visible KV >= 2048, and a starved grid
(`B*H_KV < 264`); everything else keeps the single-pass paths. The
sliding-layer decode (kv <= window = 512) stays on the classic kernel:
it is launch-overhead-bound (49 µs vs 5 µs roofline), and split-KV was
measured *slower* there (75 µs — two-kernel launch cost dominates).
Fixing that tail requires CUDA graphs / cross-layer fusion, out of
kernel scope (see claude.md 2026-04-17 Python-overhead entries).

Also re-swept this round: dQ configs under TMA (13 configs; the
production default remains optimal — dQ is already at its L1TEX bound).

## Final per-workload SOL (NCU, dominant kernels)

Headline SOL = max(Compute, Memory); L1/TEX shown because it is the
bounding pipe of the wgmma kernels.

| workload | dominant kernel(s) | Compute | Memory | L1/TEX | DRAM |
|---|---|---|---|---|---|
| **W4** decode (global) | **split-KV decode** | 14.0% | **86.5%** | 30.8% | **86.5%** |
| W5 MoE train | dQ D=512 (16:2) | 30.0% | 78.0% | **80.7%** | 1.9% |
| | dKV D=512 | 35.4% | 58.0% | 59.7% | 7.0% |
| W1 packed SFT | dQ D=512 (varlen) | 28.2% | 74.6% | 79.0% | 4.6% |
| | dKV D=512 (varlen) | 28.3% | 45.8% | 60.1% | 2.7% |
| W2 32K prefill | fwd D=512 | 41.0% | 64.2% | 65.7% | 1.0% |
| W3 batched prefill | fwd D=512 (B=8) | 40.9% | 62.2% | 66.0% | 5.2% |
| | fwd D=256 SWA (B=8) | 40.5% | 35.3% | 38.4% | 15.1% |

Attention wall-clock per workload unit (35-layer E2B / 30-layer MoE):
W1 50.7 ms/step, W2 294 ms/fwd, W3 15.7 ms/fwd, W4 2.89 ms/token
(was 6.22), W5 201 ms/step.

## Where >80% SOL stands

- **W4: achieved** — 86.5% DRAM SOL on the kernel that carries the
  workload; the wall-clock is within 1.25x of the pure-DRAM roofline
  including the combine pass.
- **W5 / W1 backward**: the dQ kernel runs at 79-81% of its bounding
  pipe (L1TEX — wgmma operand bandwidth). dKV and the fwd kernel sit at
  60-68%: the residual gap is the 1-CTA/SM occupancy ceiling quantified
  in `docs/hopper_port.md` §2 (fp32 accumulator + smem pin 8 warps/SM;
  warp-specialization/pingpong not expressible in Triton 3.5 on sm_90 —
  the config/TMA/stage space around it has been exhaustively swept, incl.
  the dQ TMA re-sweep this round).
- **W2 / W3 prefill**: same fwd-kernel ceiling (62-66%). Against SDPA
  these workloads still run 2.2-3.4x faster; the analysis (and the
  failed-experiment matrix) documents exactly which hardware features a
  future CUTLASS/CuTe port would need to close the rest.

No optimization in this round touches numerics: split-KV reproduces the
single-pass results to reference tolerance (cos >= 0.999996 on all 17
decode cases), and every prior suite (functional sweep, varlen packing,
packed dKV, tier-1, Gemma-4-E2B-it e2e logits, teacher-forced generation
parity) passes unchanged.
