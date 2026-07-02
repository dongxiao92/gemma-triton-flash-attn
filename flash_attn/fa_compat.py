"""FlashAttention-compatible API — drop-in for code written against
`flash_attn` (Dao-AILab flash-attention 2.x).

Layouts and semantics follow FA exactly:

    flash_attn_func(q, k, v, ...)            q: (B, seqlen, nheads, d)
    flash_attn_varlen_func(q, k, v, ...)     q: (total, nheads, d)
    flash_attn_qkvpacked_func(qkv, ...)      qkv: (B, seqlen, 3, nheads, d)
    flash_attn_kvpacked_func(q, kv, ...)     kv:  (B, seqlen, 2, nheads_k, d)
    flash_attn_varlen_qkvpacked_func / flash_attn_varlen_kvpacked_func
    flash_attn_with_kvcache(q, k_cache, v_cache, ...)

Internally tensors are viewed/normalized into this package's
(B, H, N, D) layout and dispatched to the same Triton kernels the native
API uses (pack-GQA + TMA fwd, TMA bwd, split-KV decode, varlen packing) —
the compat layer adds only cheap layout transposes/copies (K/V are the
small tensors under GQA; the output copy replaces the transpose FA users
already get).

Two ways to switch:

    # 1. change the import
    from gemma_triton_flash_attn.fa_compat import flash_attn_func

    # 2. zero code change: shim the `flash_attn` module before user code
    from gemma_triton_flash_attn.fa_compat import install
    install()          # sys.modules["flash_attn"] -> this module
    from flash_attn import flash_attn_varlen_func   # -> Triton kernels

Faithfully unsupported (loud NotImplementedError, never silent wrong
results): dropout_p > 0, softcap, ALiBi, paged KV (block_table),
bidirectional sliding windows, return_attn_probs, non-power-of-2 head
dims, cross-length varlen (cu_seqlens_q != cu_seqlens_k).
"""
from __future__ import annotations

import math
import sys

import torch

from .attention import (
    attention_flash_gqa,
    flash_attn_gqa_train,
    flash_attn_gqa_varlen_train,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_with_kvcache",
    "install",
]


def _check_unsupported(dropout_p, softcap, alibi_slopes, return_attn_probs,
                       block_table=None):
    if dropout_p != 0.0:
        raise NotImplementedError("fa_compat: dropout_p > 0 is not supported")
    if softcap not in (0.0, None):
        raise NotImplementedError("fa_compat: softcap is not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("fa_compat: alibi_slopes is not supported")
    if return_attn_probs:
        raise NotImplementedError("fa_compat: return_attn_probs is not supported")
    if block_table is not None:
        raise NotImplementedError("fa_compat: paged KV (block_table) is not supported")


def _window_to_slide(window_size, causal):
    """FA window_size=(left, right) -> our slide_size (0 = disabled)."""
    left, right = window_size
    if left == -1 and right in (-1, 0):
        return 0, causal
    if not causal:
        raise NotImplementedError(
            "fa_compat: sliding window requires causal=True "
            f"(got window_size={tuple(window_size)}, causal={causal})")
    if right not in (0, -1):
        raise NotImplementedError(
            "fa_compat: right-lookahead windows are not supported")
    # FA causal window (left, 0): key j visible iff i - left <= j <= i
    # ours: i - j < slide  <=>  j >= i - slide + 1  =>  slide = left + 1
    return left + 1, True


def _scale_q(q, softmax_scale):
    """Our kernels bake in 1/sqrt(D); fold any custom scale into q."""
    D = q.shape[-1]
    default = D ** -0.5
    scale = default if softmax_scale is None else softmax_scale
    if scale != default:
        q = q * (scale / default)
    return q


def _pow2_headdim(D):
    if D & (D - 1) or not (16 <= D <= 512):
        raise NotImplementedError(
            f"fa_compat: head dim {D} unsupported (needs power of 2 in [16, 512])")


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False):
    """FA-layout dense attention. q: (B, N, H, D); k/v: (B, N_k, H_kv, D).

    Returns out: (B, N, H, D). Supports autograd (training) and
    cross-length inference (N < N_k, bottom-right-aligned causal — same
    as FA)."""
    _check_unsupported(dropout_p, softcap, alibi_slopes, return_attn_probs)
    _pow2_headdim(q.shape[-1])
    slide, causal = _window_to_slide(window_size, causal)
    q = _scale_q(q, softmax_scale)

    # FA (B, N, H, D) -> ours (B, H, N, D). q/out strides are handled by
    # the kernels; K/V are copied to contiguous so the TMA fast paths
    # engage (K/V are GQA-small; the copies are grad-transparent).
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    B, Nq, H, D = q.shape
    training = torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad)
    if training and D >= 512:
        # contiguous Q lets the TMA backward path engage (grad-transparent).
        # For D < 512 the copy chain (q, do, dq) costs more than the TMA
        # backward saves — measured on H100; the pointer path runs instead.
        qt = qt.contiguous()

    if qt.shape[2] == kt.shape[2]:
        # write the kernel output directly into FA-layout memory (the
        # kernels store through arbitrary strides) — no output copy
        out_fa = torch.empty(B, Nq, H, D, dtype=q.dtype, device=q.device)
        out = flash_attn_gqa_train(qt, kt, vt, causal=causal, slide_size=slide,
                                   deterministic=deterministic,
                                   out=out_fa.transpose(1, 2))
        return out.transpose(1, 2)
    # cross-length (KV cache) — inference only, matches FA's
    # bottom-right-aligned causal semantics
    out = flash_attn_gqa_train(qt, kt, vt, causal=causal, slide_size=slide)
    return out.transpose(1, 2).contiguous()


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None,
                              causal=False, window_size=(-1, -1), softcap=0.0,
                              alibi_slopes=None, deterministic=False,
                              return_attn_probs=False):
    """qkv: (B, N, 3, H, D)."""
    return flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                           dropout_p, softmax_scale, causal, window_size,
                           softcap, alibi_slopes, deterministic,
                           return_attn_probs)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None,
                             causal=False, window_size=(-1, -1), softcap=0.0,
                             alibi_slopes=None, deterministic=False,
                             return_attn_probs=False):
    """q: (B, N, H, D); kv: (B, N_k, 2, H_kv, D)."""
    return flash_attn_func(q, kv[:, :, 0], kv[:, :, 1],
                           dropout_p, softmax_scale, causal, window_size,
                           softcap, alibi_slopes, deterministic,
                           return_attn_probs)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                           causal=False, window_size=(-1, -1), softcap=0.0,
                           alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, block_table=None):
    """FA varlen. q: (total_q, H, D); k/v: (total_k, H_kv, D).

    Self-attention packing only (cu_seqlens_q == cu_seqlens_k), which is
    what padding-free training / sample packing uses."""
    _check_unsupported(dropout_p, softcap, alibi_slopes, return_attn_probs,
                       block_table)
    _pow2_headdim(q.shape[-1])
    if q.shape[0] != k.shape[0] or (
            cu_seqlens_q is not cu_seqlens_k
            and not torch.equal(cu_seqlens_q, cu_seqlens_k)):
        raise NotImplementedError(
            "fa_compat: varlen requires cu_seqlens_q == cu_seqlens_k "
            "(self-attention packing; cross-length varlen unsupported)")
    slide, causal = _window_to_slide(window_size, causal)
    q = _scale_q(q, softmax_scale)

    # (total, H, D) -> (1, H, total, D)
    T, H, D = q.shape
    qt = q.transpose(0, 1).unsqueeze(0)
    kt = k.transpose(0, 1).unsqueeze(0)
    vt = v.transpose(0, 1).unsqueeze(0)
    training = torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad)
    if training and D >= 512:
        qt = qt.contiguous()  # TMA backward path (grad-transparent)
    out_fa = torch.empty(T, H, D, dtype=q.dtype, device=q.device)
    out = flash_attn_gqa_varlen_train(
        qt, kt, vt, cu_seqlens_q, max_seqlen=int(max_seqlen_q),
        causal=causal, slide_size=slide, deterministic=deterministic,
        out=out_fa.transpose(0, 1).unsqueeze(0),
    )
    return out.squeeze(0).transpose(0, 1)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                     dropout_p=0.0, softmax_scale=None,
                                     causal=False, window_size=(-1, -1),
                                     softcap=0.0, alibi_slopes=None,
                                     deterministic=False,
                                     return_attn_probs=False):
    """qkv: (total, 3, H, D)."""
    return flash_attn_varlen_func(qkv[:, 0], qkv[:, 1], qkv[:, 2],
                                  cu_seqlens, cu_seqlens, max_seqlen,
                                  max_seqlen, dropout_p, softmax_scale,
                                  causal, window_size, softcap, alibi_slopes,
                                  deterministic, return_attn_probs)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                    max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                                    softmax_scale=None, causal=False,
                                    window_size=(-1, -1), softcap=0.0,
                                    alibi_slopes=None, deterministic=False,
                                    return_attn_probs=False):
    """q: (total_q, H, D); kv: (total_k, 2, H_kv, D)."""
    return flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], cu_seqlens_q,
                                  cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                  dropout_p, softmax_scale, causal,
                                  window_size, softcap, alibi_slopes,
                                  deterministic, return_attn_probs)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None,
                            rotary_cos=None, rotary_sin=None,
                            cache_seqlens=None, cache_batch_idx=None,
                            cache_leftpad=None, block_table=None,
                            softmax_scale=None, causal=False,
                            window_size=(-1, -1), softcap=0.0,
                            rotary_interleaved=True, alibi_slopes=None,
                            num_splits=0, return_softmax_lse=False):
    """FA decode entry. q: (B, N_q, H, D); caches: (B, N_cache, H_kv, D).

    Supported subset: uniform cache lengths (`cache_seqlens` None, int, or
    a tensor with all-equal entries), optional in-place append of new
    k/v, no rotary / paging / left-padding. `num_splits` is ignored —
    the split-KV heuristic picks automatically. Cache tensors are
    updated in place exactly like FA when `k`/`v` are given."""
    _check_unsupported(0.0, softcap, alibi_slopes, False, block_table)
    _pow2_headdim(q.shape[-1])
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError("fa_compat: in-kernel rotary is not supported")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise NotImplementedError(
            "fa_compat: cache_batch_idx / cache_leftpad are not supported")
    if return_softmax_lse:
        raise NotImplementedError("fa_compat: return_softmax_lse is not supported")

    B, N_q = q.shape[0], q.shape[1]
    if cache_seqlens is None:
        seqlen = k_cache.shape[1]
        if k is not None:
            raise ValueError("k/v append requires cache_seqlens")
    elif isinstance(cache_seqlens, int):
        seqlen = cache_seqlens
    else:
        u = torch.unique(cache_seqlens)
        if u.numel() != 1:
            raise NotImplementedError(
                "fa_compat: ragged cache_seqlens per batch entry is not "
                "supported (uniform lengths only)")
        seqlen = int(u.item())

    # in-place cache append (FA semantics)
    if k is not None:
        n_new = k.shape[1]
        k_cache[:, seqlen:seqlen + n_new] = k
        v_cache[:, seqlen:seqlen + n_new] = v
        seqlen += n_new

    slide, _ = _window_to_slide(window_size, True)
    q = _scale_q(q, softmax_scale)

    qt = q.transpose(1, 2)
    kt = k_cache[:, :seqlen].transpose(1, 2)
    vt = v_cache[:, :seqlen].transpose(1, 2)
    # causal=True gives FA's bottom-right alignment (q rows are the
    # suffix); causal=False attends the whole cache (identical for N_q=1).
    with torch.no_grad():
        out = attention_flash_gqa(qt, kt, vt, causal=causal or N_q == 1,
                                  slide_size=slide)
    return out.transpose(1, 2).contiguous()


def install():
    """Register this module as `flash_attn`, so unmodified user code
    (`import flash_attn` / `from flash_attn import ...`) uses these
    kernels. Call before user code imports flash_attn. No-op if a real
    flash_attn module is already imported (raises to avoid ambiguity)."""
    existing = sys.modules.get("flash_attn")
    if existing is not None and existing is not sys.modules[__name__]:
        raise RuntimeError(
            "fa_compat.install(): a `flash_attn` module is already imported; "
            "call install() before importing flash_attn-dependent code")
    sys.modules["flash_attn"] = sys.modules[__name__]
    return sys.modules[__name__]


# FA exposes __version__; some frameworks gate features on it.
__version__ = "2.9.0+gemma_triton_compat"
