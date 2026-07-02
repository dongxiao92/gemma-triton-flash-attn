"""NCU Speed-of-Light driver for the Hopper port (see docs/hopper_port.md).

Runs one attention op a few times so ncu can profile it. Usage:

  NCU=/home/scratch.xiaod_sw/dynamic-kernel-generator/cuda-12.9.83-20250520/ncu
  $NCU --section SpeedOfLight --kernel-name regex:packed_kernel \
       --launch-count 1 --launch-skip 2 python benchmarks/hopper_ncu_sol.py fwd512 8192
  $NCU --section SpeedOfLight --kernel-name regex:bwd --launch-count 2 \
       python benchmarks/hopper_ncu_sol.py bwd512 8192

Targets: fwd512 | fwd256swa | bwd512 | bwd256swa   (E2B shapes, H_Q=8, H_KV=1)
Varlen targets (packed N_SEQS x LEN): vfwd512 | vfwd256swa | vbwd512 | vbwd256swa
  e.g. python benchmarks/hopper_ncu_sol.py vbwd512 4096 4   (4 samples of 4096)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flash_attn.attention import attention_flash_gqa, flash_attn_gqa_train

which = sys.argv[1] if len(sys.argv) > 1 else "fwd512"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
D = 512 if "512" in which else 256
slide = 512 if "swa" in which else 0
torch.manual_seed(0)
dt = torch.float16

if which.startswith("v"):
    from flash_attn.attention import flash_attn_gqa_varlen_train
    n_seqs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    T = n_seqs * N
    cu = torch.arange(0, T + 1, N, dtype=torch.int32, device="cuda")
    rg = "bwd" in which
    q = torch.randn(1, 8, T, D, dtype=dt, device="cuda", requires_grad=rg)
    k = torch.randn(1, 1, T, D, dtype=dt, device="cuda", requires_grad=rg)
    v = torch.randn(1, 1, T, D, dtype=dt, device="cuda", requires_grad=rg)
    for _ in range(3):
        out = flash_attn_gqa_varlen_train(q, k, v, cu, N, causal=True,
                                          slide_size=slide)
        if rg:
            out.backward(torch.randn_like(out))
            q.grad = k.grad = v.grad = None
elif which.startswith("fwd"):
    q = torch.randn(1, 8, N, D, dtype=dt, device="cuda")
    k = torch.randn(1, 1, N, D, dtype=dt, device="cuda")
    v = torch.randn(1, 1, N, D, dtype=dt, device="cuda")
    for _ in range(3):
        attention_flash_gqa(q, k, v, causal=True, slide_size=slide)
else:
    q = torch.randn(1, 8, N, D, dtype=dt, device="cuda", requires_grad=True)
    k = torch.randn(1, 1, N, D, dtype=dt, device="cuda", requires_grad=True)
    v = torch.randn(1, 1, N, D, dtype=dt, device="cuda", requires_grad=True)
    out = flash_attn_gqa_train(q, k, v, causal=True, slide_size=slide)
    out.backward(torch.randn_like(out))
torch.cuda.synchronize()
print("done")
