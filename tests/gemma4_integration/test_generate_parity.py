"""Generation parity vs SDPA on Gemma-4-E2B-it (KV-cache decode path).

Exercises the cross-length (q_len=1 vs kv_len=t) attention added in the
Hopper port. Two checks:

1. Greedy generation, short prompt: tokens must match SDPA exactly.
2. Teacher-forced stepwise decode on a long prompt (crosses the sliding
   window): per-step next-token logits must stay cos > 0.999 with zero
   top-1 flips. This is the meaningful parity metric — free-running
   greedy chains may diverge benignly on near-ties (top-2 margin within
   bf16 attention noise), which is NOT a kernel bug.

Run:
    python tests/gemma4_integration/test_generate_parity.py \
        --model /path/to/gemma-4-E2B-it
"""
import argparse
import sys

import torch
import transformers  # noqa: F401 — import before patching

from gemma_triton_flash_attn import (
    patch_transformers_5_5_4_flash_attn_key,
    register_triton_attention,
)
patch_transformers_5_5_4_flash_attn_key()

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402


def set_impl(model, impl):
    model.config._attn_implementation = impl
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = impl


def chat_ids(tok, content):
    enc = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(enc):
        enc = enc["input_ids"]
    return enc.cuda()


def test_short_greedy(model, tok):
    print("=== short-prompt greedy generation ===")
    ids = chat_ids(tok, "Name the three primary colors, one per line.")
    outs = {}
    for impl in ("sdpa", "triton_gqa"):
        set_impl(model, impl)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=48, do_sample=False)
        outs[impl] = out[0, ids.shape[1]:].tolist()
        print(f"  [{impl}] {tok.decode(outs[impl], skip_special_tokens=True)!r}")
    n = min(len(outs["sdpa"]), len(outs["triton_gqa"]))
    match = sum(a == b for a, b in zip(outs["sdpa"][:n], outs["triton_gqa"][:n]))
    print(f"  token match: {match}/{n}")
    return match == n


def test_stepwise_parity(model, tok, n_steps=24):
    print("=== teacher-forced stepwise decode parity (long prompt) ===")
    story = ("The city of Veridian was built on seven hills, each crowned "
             "with a tower of a different color. ") * 30
    ids = chat_ids(tok, story + "\n\nWhat color is the second tower?")
    print(f"  prompt len: {ids.shape[1]}")

    set_impl(model, "sdpa")
    with torch.no_grad():
        traj = model.generate(ids, max_new_tokens=n_steps, do_sample=False)[0]

    logits = {}
    for impl in ("sdpa", "triton_gqa"):
        set_impl(model, impl)
        cache = DynamicCache(config=model.config.get_text_config())
        steps = []
        with torch.no_grad():
            out = model(traj[:ids.shape[1]].unsqueeze(0),
                        past_key_values=cache, use_cache=True)
            steps.append(out.logits[0, -1].float())
            for t in range(ids.shape[1], traj.shape[0] - 1):
                out = model(traj[t:t + 1].unsqueeze(0),
                            past_key_values=cache, use_cache=True)
                steps.append(out.logits[0, -1].float())
        logits[impl] = steps

    n_flip, min_cos = 0, 1.0
    for a, b in zip(logits["sdpa"], logits["triton_gqa"]):
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        min_cos = min(min_cos, cos)
        n_flip += int(a.argmax().item() != b.argmax().item())
    print(f"  steps: {len(logits['sdpa'])}, top-1 flips: {n_flip}, "
          f"min logits cos: {min_cos:.6f}")
    return n_flip == 0 and min_cos > 0.999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()
    register_triton_attention()

    ok = test_short_greedy(model, tok)
    ok &= test_stepwise_parity(model, tok)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
