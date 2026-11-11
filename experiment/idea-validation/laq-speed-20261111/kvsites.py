"""QuaRot / SpinQuant K and V quantization at the cached-KV positions of the two WAMs.

K: Hadamard over the head dimension (QuaRot R3), then a per-head asymmetric quantizer. Q needs no rotation of its own,
since attention depends only on q k^T and an orthogonal map on K can be inverted right after quantizing.
V: rotated over the head dimension (QuaRot Hadamard, SpinQuant learned R2), asymmetric per-token per-head-group
quantization, rotated back. The attention output then matches the official path and the o_proj input rotation absorbs
the same head rotation (see methods.py).

SVDQuant and ViDiT-Q define no KV quantization, so they use none.
"""
from __future__ import annotations

from collections import Counter

import torch

import qops as q


class KV:
    def __init__(self, arm, arch, rotations=None, ste=False):
        self.arm, self.h, self.d = arm, arch["heads"], arch["head_dim"]
        self.r = rotations
        self.clip = 0.95 if arm == "quarot" else 1.0
        self.ste = ste
        self.counts = Counter()

    def _flat(self, x):
        return x.shape[-1] != self.d

    def _heads(self, x):
        return x.reshape(*x.shape[:-1], self.h, self.d) if self._flat(x) else x

    def _final(self, x, value):
        return value.reshape(x.shape).to(x.dtype)

    def k(self, x, site):
        self.counts[site] += 1
        v = self._heads(x.float())
        r = q.fht(v)
        out = q.a_token(r, sym=False, clip=self.clip)
        return self._final(x, q.fht(q.ste(out, r) if self.ste else out))

    def v(self, x, site, key):
        self.counts[site] += 1
        v = self._heads(x.float())
        rotated = q.fht(v) if self.arm == "quarot" else v @ self.r[key]
        out = q.a_group(rotated, self.d, clip=self.clip)
        if self.ste:
            out = q.ste(out, rotated)
        return self._final(x, q.fht(out) if self.arm == "quarot" else out @ self.r[key].t())


def install_fastwam(model, kv, frame_tokens, video_tokens):
    """Text cross-attention K/V of both experts, the observed-frame rows of video self-attention, and the prefill cache."""
    from fastwam.models.wan22.wan_video_dit import flash_attention, rope_apply
    for expert, name in ((model.video_expert, "video"), (model.action_expert, "action")):
        for i, block in enumerate(expert.blocks):
            attn = block.cross_attn
            attn.norm_k.register_forward_hook(lambda _m, _i, out: kv.k(out, "text"))
            attn.v.register_forward_hook(lambda _m, _i, out, _key=f"{name}.{i}.cross": kv.v(out, "text", _key))

    def observed(attn, key):
        def forward(x, freqs, self_attn_mask=None):
            q_, k = attn.norm_q(attn.q(x)), attn.norm_k(attn.k(x))
            v = attn.v(x)
            q_, k = rope_apply(q_, freqs, attn.num_heads), rope_apply(k, freqs, attn.num_heads)
            if x.shape[1] != video_tokens:
                raise RuntimeError(f"Video self-attention expected {video_tokens} tokens, got {x.shape[1]}")
            k = torch.cat((kv.k(k[:, :frame_tokens], "observation"), k[:, frame_tokens:]), 1)
            v = torch.cat((kv.v(v[:, :frame_tokens], "observation", key), v[:, frame_tokens:]), 1)
            return attn.o(flash_attention(q=q_, k=k, v=v, num_heads=attn.num_heads, ctx_mask=self_attn_mask))
        return forward

    for i, block in enumerate(model.video_expert.blocks):
        block.self_attn.forward = observed(block.self_attn, f"video.{i}.self")
    original = model.mot.prefill_video_cache_tensor

    def prefill(*args, **kwargs):
        keys, values = original(*args, **kwargs)
        return ([kv.k(k, "future") for k in keys],
                [kv.v(v, "future", f"video.{i}.self") for i, v in enumerate(values)])
    model.mot.prefill_video_cache_tensor = prefill


def install_cosmos(model, kv, expected):
    """K and V of every attention at every denoising step."""
    import re
    pat = re.compile(r"blocks\.(\d+)\._checkpoint_wrapped_module\.(self|cross)_attn$")
    wrapped = 0
    for name, attn in model.named_modules():
        m = pat.search(name)
        if m and hasattr(attn, "compute_qkv"):
            if (attn.n_heads, attn.head_dim) != (kv.h, kv.d):
                raise RuntimeError(f"{name} has {attn.n_heads}x{attn.head_dim} heads, expected {kv.h}x{kv.d}")
            original, key = attn.compute_qkv, f"net.{m.group(1)}.{m.group(2)}"

            def compute(*args, _original=original, _key=key, **kwargs):
                q_, k, v, *rest = _original(*args, **kwargs)
                return (q_, kv.k(k, "kv"), kv.v(v, "kv", _key), *rest)
            attn.compute_qkv = compute
            wrapped += 1
    if wrapped != expected:
        raise RuntimeError(f"Expected {expected} attentions, wrapped {wrapped}")
