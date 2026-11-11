"""KV4 on the cached keys and values: every method shares this one per-token, group-of-32 symmetric INT4 quantizer.

Fast-WAM: text cross-attention K/V of both experts, the observed-frame rows of video self-attention, and the
          video K/V cache that the prefill writes for the action loop.
Cosmos-Policy: K and V of every attention at every denoising step, since no K/V cache exists across steps.
"""
from __future__ import annotations

from collections import Counter

import scalarq as sq


def _q(counts, site, x, group):
    counts[site] += 1
    return sq.q_sym(x, group).to(x.dtype)


def install_fastwam(model, group, frame_tokens, video_tokens):
    from fastwam.models.wan22.wan_video_dit import flash_attention, rope_apply
    counts = Counter()
    for expert in (model.video_expert, model.action_expert):
        for block in expert.blocks:
            attn = block.cross_attn
            attn.norm_k.register_forward_hook(lambda _m, _i, out: _q(counts, "text", out, group))
            attn.v.register_forward_hook(lambda _m, _i, out: _q(counts, "text", out, group))

    def observed(attn):
        def forward(x, freqs, self_attn_mask=None):
            q = attn.norm_q(attn.q(x))
            k = attn.norm_k(attn.k(x))
            v = attn.v(x)
            q = rope_apply(q, freqs, attn.num_heads)
            k = rope_apply(k, freqs, attn.num_heads)
            if x.shape[1] != video_tokens:
                raise RuntimeError(f"Video self-attention expected {video_tokens} tokens, got {x.shape[1]}")
            k = sq_rows(k)
            v = sq_rows(v)
            return attn.o(flash_attention(q=q, k=k, v=v, num_heads=attn.num_heads, ctx_mask=self_attn_mask))

        def sq_rows(t):
            import torch
            return torch.cat((_q(counts, "observation", t[:, :frame_tokens], group), t[:, frame_tokens:]), 1)
        return forward

    for block in model.video_expert.blocks:
        block.self_attn.forward = observed(block.self_attn)

    original = model.mot.prefill_video_cache_tensor

    def prefill(*args, **kwargs):
        keys, values = original(*args, **kwargs)
        return ([_q(counts, "future", k, group) for k in keys], [_q(counts, "future", v, group) for v in values])
    model.mot.prefill_video_cache_tensor = prefill
    return counts


def install_cosmos(model, group, expected):
    counts = Counter()
    wrapped = 0
    for _, attn in model.named_modules():
        if hasattr(attn, "compute_qkv") and hasattr(attn, "k_proj"):
            original = attn.compute_qkv

            def compute(*args, _original=original, **kwargs):
                q, k, v, *rest = _original(*args, **kwargs)
                return (q, _q(counts, "kv", k, group), _q(counts, "kv", v, group),
                        *(_q(counts, "kv", t, group) for t in rest))
            attn.compute_qkv = compute
            wrapped += 1
    if wrapped != expected:
        raise RuntimeError(f"Expected {expected} attentions, wrapped {wrapped}")
    return counts
