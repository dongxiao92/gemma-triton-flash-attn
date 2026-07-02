import sys, torch
sys.path.insert(0, '/home/scratch.xiaod_gpu_5/gemma-triton-flash-attn')
from flash_attn.attention import attention_flash_gqa, flash_attn_gqa_train, flash_attn_gqa_varlen_train
w = sys.argv[1]
torch.manual_seed(0)
DT = torch.bfloat16
if w == "w3g":    # batched prefill global
    q = torch.randn(8, 8, 2048, 512, dtype=DT, device="cuda")
    k = torch.randn(8, 1, 2048, 512, dtype=DT, device="cuda")
    v = torch.randn(8, 1, 2048, 512, dtype=DT, device="cuda")
    for _ in range(3): attention_flash_gqa(q, k, v, causal=True)
elif w == "w3s":  # batched prefill sliding
    q = torch.randn(8, 8, 2048, 256, dtype=DT, device="cuda")
    k = torch.randn(8, 1, 2048, 256, dtype=DT, device="cuda")
    v = torch.randn(8, 1, 2048, 256, dtype=DT, device="cuda")
    for _ in range(3): attention_flash_gqa(q, k, v, causal=True, slide_size=512)
elif w == "w5g":  # MoE full training
    q = torch.randn(1, 16, 8192, 512, dtype=DT, device="cuda", requires_grad=True)
    k = torch.randn(1, 2, 8192, 512, dtype=DT, device="cuda", requires_grad=True)
    v = torch.randn(1, 2, 8192, 512, dtype=DT, device="cuda", requires_grad=True)
    for _ in range(3):
        out = flash_attn_gqa_train(q, k, v, causal=True)
        out.backward(torch.randn_like(out)); q.grad = k.grad = v.grad = None
elif w == "w1g":  # packed SFT global fwd+bwd, actual W1 mix
    seq = [2919, 1370, 3105, 798]; T = sum(seq)
    cu = torch.tensor([0] + list(torch.tensor(seq).cumsum(0)), dtype=torch.int32, device="cuda")
    q = torch.randn(1, 8, T, 512, dtype=DT, device="cuda", requires_grad=True)
    k = torch.randn(1, 1, T, 512, dtype=DT, device="cuda", requires_grad=True)
    v = torch.randn(1, 1, T, 512, dtype=DT, device="cuda", requires_grad=True)
    for _ in range(3):
        out = flash_attn_gqa_varlen_train(q, k, v, cu, max(seq), causal=True)
        out.backward(torch.randn_like(out)); q.grad = k.grad = v.grad = None
torch.cuda.synchronize(); print("done")
