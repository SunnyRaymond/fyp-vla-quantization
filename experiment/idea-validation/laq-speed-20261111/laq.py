"""LAQ on Fast-WAM in eager fake quantization (FP32 arithmetic on rotated operands, decoded weights).

Base quantizers (the published QuaRot recipe, whose fitted weights are reused unchanged):
  weights      GPTQ, one MSE-clipped scale per row, in randomized-Hadamard coordinates (4.01 bpw)
  activations  per-token symmetric INT4, clip 0.9
  keys/values  per-head asymmetric INT4, clip 0.95, Hadamard over the head dimension
With nothing selected the model is the QuaRot arm bit for bit.

Refinement: a selected token gets two 4-bit passes, x^ = Q(x) + Q(x - Q(x)), each with its own per-token scale and
no clipping, in the activations (the Linear that reads the token) and in the cached K/V.

Units and their classes (rows of one operand):
  text.video / text.action   context rows at the cross-attention K/V Linears and the cached text K/V (frozen)
  obsA                       the observed-frame rows of every video Linear (frozen: bit-identical at every step)
  obsKV                      cached K/V of the observed-frame tokens (frozen)
  fut (step s)               future rows of every video Linear at denoising step s; s = steps is the prefill (fresh)
  futKV                      cached K/V of the future tokens, read by the action loop (frozen)
  action                     the action rows of every action Linear (fresh)

The gaze of a future token is the attention mass the action tokens put on it over the action loop; the gaze of an
observed token adds the mass read directly and the mass read through the future tokens (one hop, prefill attention).
The gaze of one query selects the tokens of the next; the first query of an episode uses the offline gaze.

CUDA graphs (graphs.py): the three phase calls go through State.call. With State.static set (freeze), the refined rows
of every operand live in fixed index buffers that refresh() rewrites in place whenever the selection changes, so a
captured graph reads the current selection; State.graphing turns off the Python-side reuse (text-site cache, memo),
whose decisions a capture would freeze.
"""
from __future__ import annotations

from collections import Counter
import inspect
import math

import torch
import torch.nn.functional as F

import methods as mt
import qops as q

A_CLIP, KV_CLIP = 0.9, 0.95
FROZEN = ("text.video", "text.action", "obsA", "obsKV", "futKV")


def a_base(r):
    return q.a_token(r, sym=True, clip=A_CLIP)


def a_fine(r):
    return q.a_token(r, sym=True, clip=1.0)


def k_base(r):
    return q.a_token(r, sym=False, clip=KV_CLIP)


def k_fine(r):
    return q.a_token(r, sym=False, clip=1.0)


def v_base(r):
    return q.a_group(r, r.shape[-1], clip=KV_CLIP)


def v_fine(r):
    return q.a_group(r, r.shape[-1], clip=1.0)


def two_pass(x, fine):
    first = fine(x)
    return first + fine(x - first)


def refine(x, rows, base, fine):
    """base(x) on every token (axis 1); the tokens in `rows` (indices or bool [T], None for none) get two fine passes."""
    out = base(x)
    if rows is None:
        return out
    idx = rows.nonzero().squeeze(1) if rows.dtype == torch.bool else rows
    if idx.numel() == 0:
        return out
    return out.index_copy(1, idx, two_pass(x.index_select(1, idx), fine))


def top(gaze, n):
    mask = torch.zeros(gaze.shape[0], dtype=torch.bool, device=gaze.device)
    if n > 0:
        mask[torch.topk(gaze, min(int(n), gaze.shape[0])).indices] = True
    return mask


def attention(query, key, heads, mask):
    """Attention probabilities [heads, Sq, Sk] of batch element 0 (FP32)."""
    qh = query[0].float().reshape(query.shape[1], heads, -1).transpose(0, 1)
    kh = key[0].float().reshape(key.shape[1], heads, -1).transpose(0, 1)
    scores = qh @ kh.transpose(1, 2) / math.sqrt(qh.shape[-1])
    if mask is not None:
        m = mask.to(scores.device)
        scores = scores.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else scores + m.float()
    return torch.softmax(scores, -1)


class Selection:
    """Row masks of one query from a spec (counts per class) and the gaze."""

    def __init__(self, st, spec, g_fut, g_obs):
        self.flags = {"text.video": bool(spec["text_video"]), "text.action": bool(spec["text_action"]),
                      "action": bool(spec["action"])}
        self.masks = dict(obsA=top(g_obs, spec["obsA"]), obsKV=top(g_obs, spec["obsKV"]), futKV=top(g_fut, spec["futKV"]))
        if len(spec["fut"]) != st.steps + 1:
            raise ValueError(f"spec['fut'] needs {st.steps + 1} counts")
        self.fut = torch.stack([top(g_fut, n) for n in spec["fut"]])
        self.cache = {}

    def rows(self, cls, n, step, device):
        if cls in self.flags:
            return torch.full((n,), self.flags[cls], dtype=torch.bool, device=device)
        mask = self.fut[step] if cls == "fut" else self.masks[cls]
        if mask.shape[0] != n:
            raise RuntimeError(f"{cls} has {mask.shape[0]} tokens, the operand has {n}")
        return mask


def spec_none(steps):
    return dict(text_video=False, text_action=False, action=False, obsA=0, obsKV=0, futKV=0, fut=[0] * (steps + 1))


class State:
    def __init__(self, steps, frame, video, arch, rotation_seed, device):
        self.steps, self.frame, self.video = int(steps), int(frame), int(video)
        self.arch, self.h, self.d = arch, arch["heads"], arch["head_dim"]
        self.rotation_seed, self.device = rotation_seed, device
        self.quant = False      # False: BF16 weights (pass-through or error injection); True: decoded 4-bit weights
        self.inject = None      # (class, "frozen" | "redrawn") in BF16 mode
        self.seed = 0
        self.gen = torch.Generator(device=device)
        self.signs = {}
        self.spec, self.sel, self.offline, self.online = None, None, None, False
        self.watch = False      # collect the attention gaze
        self.phase, self.n_video, self.n_action = None, 0, 0
        self.acc_action = torch.zeros(self.video, device=device)
        self.acc_hop = torch.zeros(self.video - self.frame, self.frame, device=device)
        self.hop_layers = 0
        self.kv_clean = None
        self.gaze = None
        self.trace = []
        self.seen, self.counts, self.shapes = set(), Counter(), {}
        self.fast = True        # reuse operands that are provably the same (bit-identical results, fewer GPU calls)
        self.shared = None      # (input tensor, stream, quantized operand): q/k/v read one input
        self.frozen = {}        # per text site: (input, output) of the last call; the context is the same at every step
        self.memo = True        # with fast: the text embedding and the cross-attention key norm return their last output
                                # when called on the same, unmodified tensor, so the text sites hit by identity
        self.static = None      # graph mode: {(segs, T, step): fixed index buffer or None}, rewritten by refresh()
        self.keys = set()       # operand keys met under the current spec
        self.graphing = False   # the phase calls run as CUDA graphs: no Python-side reuse decisions
        self.call = lambda kind, step, fn, kwargs: fn(**kwargs)

    # ---------------- selection
    def use(self, spec, online=False):
        """Install a spec with the offline gaze (zeros when no offline gaze is set: full or empty classes only)."""
        self.spec, self.online = spec, online
        self.keys, self.static = set(), None
        self.reset()

    def reset(self):
        """Start of an episode (or of an independent query): the offline gaze selects."""
        self.phase, self.n_video, self.n_action = None, 0, 0
        self.trace = []
        self.shared, self.frozen = None, {}
        if self.spec is None:
            self.sel = None
            return
        if self.offline is None:
            g_fut = torch.zeros(self.video - self.frame, device=self.device)
            g_obs = torch.zeros(self.frame, device=self.device)
        else:
            g_fut, g_obs = self.offline
        self.sel = Selection(self, self.spec, g_fut, g_obs)
        self.first = self.sel
        self.refresh()

    def step_index(self):
        if self.phase is None:
            raise RuntimeError("A video operand was quantized outside the video loop and the prefill")
        return self.phase[1]

    def _entry(self, segs, T, step):
        key = (segs, T, step)
        if key not in self.sel.cache:
            parts = [self.sel.rows(c, (T if b is None else b) - a, step, self.device) for c, a, b in segs]
            covered = sum(p.shape[0] for p in parts)
            if covered != T:
                raise RuntimeError(f"Segments {segs} cover {covered} of {T} tokens")
            mask = torch.cat(parts)
            self.sel.cache[key] = (mask, mask.nonzero().squeeze(1)) if bool(mask.any()) else (None, None)
        return self.sel.cache[key]

    def _step(self, segs):
        return self.step_index() if any(c == "fut" for c, _, _ in segs) else -1

    def mask(self, segs, T):
        """Refined rows of an operand with class segments `segs` (quantized mode), or None."""
        if self.sel is None:
            return None
        step = self._step(segs)
        self.keys.add((segs, T, step))
        return self._entry(segs, T, step)[0]

    def index(self, segs, T):
        """The refined rows as indices, computed once per selection (no per-call synchronization). In graph mode the
        fixed buffer of the operand, whose content refresh() keeps equal to the current selection."""
        if self.sel is None:
            return None
        if self.static is not None:
            key = (segs, T, self._step(segs))
            if key not in self.static:
                raise RuntimeError(f"Graph mode met an operand that the warm query did not: {key}")
            return self.static[key]
        self.mask(segs, T)
        return self.sel.cache[(segs, T, self._step(segs))][1]

    def freeze(self):
        """Fixed index buffers for every operand met under the current spec, filled from the current selection."""
        if self.sel is None or not self.keys:
            raise RuntimeError("freeze() needs a query under the current spec first")
        self.static = {}
        for key in self.keys:
            idx = self._entry(*key)[1]
            self.static[key] = None if idx is None else idx.clone()

    def refresh(self):
        """Rewrite the fixed index buffers in place from the current selection (the counts per class never change, so
        neither do the buffer sizes)."""
        if self.static is None or self.sel is None:
            return
        for key, buf in self.static.items():
            idx = self._entry(*key)[1]
            if (buf is None) != (idx is None) or (idx is not None and idx.shape != buf.shape):
                raise RuntimeError(f"The refined rows of {key} changed size under one spec")
            if buf is not None:
                buf.copy_(idx)

    def same(self, key, x):
        """The cached output of a text site if its input equals the one of the last call, else None."""
        hit = self.frozen.get(key)
        if hit is not None and (hit[0] is x or (hit[0].shape == x.shape and torch.equal(hit[0], x))):
            return hit[1]
        return None

    # ---------------- error injection (BF16 weights)
    def inject_range(self, segs, T):
        if self.inject is None:
            return None
        for c, a, b in segs:
            if c == self.inject[0]:
                return a, T if b is None else b
        return None

    def sign(self, key, shape):
        if self.inject[1] == "frozen":
            if key not in self.signs:
                self.signs[key] = self._draw(shape)
            if self.signs[key].shape != torch.Size(shape):
                raise RuntimeError(f"Frozen sign pattern of {key} changed shape")
            return self.signs[key]
        return self._draw(shape)

    def _draw(self, shape):
        return torch.randint(0, 2, tuple(shape), generator=self.gen, device=self.device).float() * 2 - 1

    # ---------------- operands
    def linear(self, x, module, xf, segs, name, stream=None):
        self.seen.add(name)
        T = x.shape[1]
        if self.quant:
            text = self.fast and not self.graphing and segs[0][0].startswith("text.")
            if text:
                out = self.same(name, x)
                if out is not None:
                    return out
            if self.fast and stream is not None and self.shared is not None and self.shared[0] is x and self.shared[1] == stream:
                operand = self.shared[2]
            else:
                operand = refine(xf.apply(x.float()), self.index(segs, T), a_base, a_fine).to(x.dtype)
                if stream is not None:
                    self.shared = (x, stream, operand)
            out = F.linear(operand, module.weight, module.bias)
            if text:
                self.frozen[name] = (x, out)
            return out
        rng = self.inject_range(segs, T)
        if rng is None:
            return F.linear(x, module.weight, module.bias)
        a, b = rng
        r = xf.apply(x[:, a:b].float())
        noisy = r + self.sign(f"A.{name}", r.shape) * (a_base(r) - r)
        x = torch.cat((x[:, :a], xf.inv(noisy).to(x.dtype), x[:, b:]), 1)
        return F.linear(x, module.weight, module.bias)

    def kv(self, x, kind, segs, key, site):
        """Quantized K or V [B, T, heads * head_dim], or None when the tensor stays as it is."""
        self.counts[site] += 1
        T = x.shape[1]
        base, fine = (k_base, k_fine) if kind == "K" else (v_base, v_fine)
        if self.quant:
            text = self.fast and not self.graphing and site == "text"
            if text:
                out = self.same(key, x)
                if out is not None:
                    return out
            r = q.fht(x.float().reshape(*x.shape[:-1], self.h, self.d))
            out = q.fht(refine(r, self.index(segs, T), base, fine)).reshape(x.shape).to(x.dtype)
            if text:
                self.frozen[key] = (x, out)
            return out
        rng = self.inject_range(segs, T)
        if rng is None:
            return None
        a, b = rng
        part = x[:, a:b]
        r = q.fht(part.float().reshape(*part.shape[:-1], self.h, self.d))
        noisy = r + self.sign(f"{key}.{self.inject[0]}", r.shape) * (base(r) - r)
        return torch.cat((x[:, :a], q.fht(noisy).reshape(part.shape).to(x.dtype), x[:, b:]), 1)

    def cache_kv(self):
        keys, values = self.kv_clean
        segs = (("obsKV", 0, self.frame), ("futKV", self.frame, None))
        out_k, out_v = [], []
        for i, (k, v) in enumerate(zip(keys, values)):
            nk, nv = self.kv(k, "K", segs, f"K.video.{i}", "future"), self.kv(v, "V", segs, f"V.video.{i}", "future")
            out_k.append(k if nk is None else nk)
            out_v.append(v if nv is None else nv)
        return out_k, out_v

    # ---------------- query phases and gaze
    def start_query(self):
        self.n_video = 0
        self.signs.clear()
        self.gen.manual_seed(int(self.seed))
        self.acc_hop.zero_()
        self.hop_layers = 0

    def observe(self, query, key, heads, mask):
        if not self.watch or self.phase is None:
            return
        if self.phase[0] == "action":
            self.acc_action += attention(query, key, heads, mask).sum((0, 1))[:self.video]
        elif self.phase[0] == "prefill":
            p = attention(query[:, self.frame:], key, heads, None if mask is None else mask[self.frame:])
            self.acc_hop += p.mean(0)[:, :self.frame]
            self.hop_layers += 1

    def finish_query(self):
        self.phase = None
        if not self.watch:
            return
        g_fut, g_dir = self.acc_action[self.frame:].clone(), self.acc_action[:self.frame].clone()
        hop = self.acc_hop / max(self.hop_layers, 1)
        g_obs = g_dir + g_fut @ hop
        self.gaze = dict(fut=g_fut, obs=g_obs, direct=g_dir, hop=hop)
        if self.online and self.spec is not None:
            self.shared, self.frozen = None, {}
            new = Selection(self, self.spec, g_fut, g_obs)
            self.trace.append(dict(previous=overlap(new.masks["futKV"], self.sel.masks["futKV"]),
                                   offline=overlap(new.masks["futKV"], self.first.masks["futKV"])))
            self.sel = new
            self.refresh()


def overlap(a, b):
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else None


def memo(st, module):
    """Return the last output when the module is called again on the very same tensor with neither the input nor the
    output modified in place since. The module is a pure function of its input, so the result is the one a new call
    would compute; the saving is that every reader downstream sees the same tensor object."""
    original = module.forward
    last = []

    def version(t):  # inference-mode tensors keep no version counter: identity alone (their readers never write them)
        return None if t.is_inference() else t._version

    def forward(x):
        if (st.fast and st.memo and not st.graphing and last and last[0] is x and last[1] == version(x)
                and last[3] == version(last[2])):
            return last[2]
        out = original(x)
        last[:] = [x, version(x), out, version(out)]
        return out
    module.forward = forward


def segments(st, role):
    if role.attn == "cross" and role.kind in ("k", "v"):
        return ((f"text.{role.expert}", 0, None),)
    if role.expert == "video":
        return (("obsA", 0, st.frame), ("fut", st.frame, None))
    return (("action", 0, None),)


def install(st, model, modules):
    """Replace the forwards of the block-level Linears, the cached-K/V sites, the three phase entry points and the
    attention of the MoT core. The model keeps its BF16 behaviour bit for bit until weights are loaded or an error
    is injected."""
    from fastwam.models.wan22 import mot as mot_module
    from fastwam.models.wan22.wan_video_dit import flash_attention, rope_apply
    installed = 0
    for name, module in modules.items():
        role = mt.parse(name)
        if role is None:
            continue
        xf = mt.xf_for("quarot", role, module.in_features, st.arch, st.rotation_seed, None, st.device)
        stream = f"{mt.stream_of(role)}.{role.block}.{role.attn}" if role.kind in ("q", "k", "v") else None
        module.forward = (lambda x, _m=module, _xf=xf, _s=segments(st, role), _n=name, _t=stream:
                          st.linear(x, _m, _xf, _s, _n, _t))
        installed += 1

    for expert, name in ((model.video_expert, "video"), (model.action_expert, "action")):
        for i, block in enumerate(expert.blocks):
            attn, segs = block.cross_attn, ((f"text.{name}", 0, None),)
            attn.norm_k.register_forward_hook(
                lambda _m, _i, out, _s=segs, _k=f"K.{name}.{i}.cross": st.kv(out, "K", _s, _k, "text"))
            attn.v.register_forward_hook(
                lambda _m, _i, out, _s=segs, _k=f"V.{name}.{i}.cross": st.kv(out, "V", _s, _k, "text"))

    for expert in (model.video_expert, model.action_expert):
        if getattr(expert, "text_embedding", None) is not None:
            memo(st, expert.text_embedding)
        for block in expert.blocks:
            memo(st, block.cross_attn.norm_k)

    def observed(attn, i):
        segs = (("obsKV", 0, None),)

        def forward(x, freqs, self_attn_mask=None):
            q_, k = attn.norm_q(attn.q(x)), attn.norm_k(attn.k(x))
            v = attn.v(x)
            q_, k = rope_apply(q_, freqs, attn.num_heads), rope_apply(k, freqs, attn.num_heads)
            if x.shape[1] != st.video:
                raise RuntimeError(f"Video self-attention expected {st.video} tokens, got {x.shape[1]}")
            nk = st.kv(k[:, :st.frame], "K", segs, f"K.video.{i}", "observation")
            nv = st.kv(v[:, :st.frame], "V", segs, f"V.video.{i}", "observation")
            if nk is not None:
                k = torch.cat((nk, k[:, st.frame:]), 1)
            if nv is not None:
                v = torch.cat((nv, v[:, st.frame:]), 1)
            return attn.o(flash_attention(q=q_, k=k, v=v, num_heads=attn.num_heads, ctx_mask=self_attn_mask))
        return forward

    for i, block in enumerate(model.video_expert.blocks):
        block.self_attn.forward = observed(block.self_attn, i)

    video_original = model._denoise_video
    video_signature = inspect.signature(video_original)

    def video(*args, **kwargs):
        if st.phase is not None and st.phase[0] == "action":
            raise RuntimeError(f"The previous query stopped after {st.n_action} of {st.steps} action steps")
        if st.phase is None or st.phase[0] != "video":
            st.start_query()
        if st.n_video >= st.steps:
            raise RuntimeError(f"More than {st.steps} video denoising steps in one query")
        st.phase = ("video", st.n_video)
        out = st.call("video", st.n_video, video_original, dict(video_signature.bind(*args, **kwargs).arguments))
        st.n_video += 1
        return out
    model._denoise_video = video

    prefill_original = model.mot.prefill_video_cache_tensor
    prefill_signature = inspect.signature(prefill_original)

    def prefill_body(**kwargs):
        keys, values = prefill_original(**kwargs)
        if keys[0].shape[1] != st.video:
            raise RuntimeError(f"Prefill cache holds {keys[0].shape[1]} tokens, expected {st.video}")
        st.kv_clean = (keys, values)
        return st.cache_kv()

    def prefill(*args, **kwargs):
        if st.n_video != st.steps:
            raise RuntimeError(f"Prefill after {st.n_video} video steps, expected {st.steps}")
        st.phase = ("prefill", st.steps)
        st.n_action = 0
        st.acc_action.zero_()  # only the action phase adds to it, so zeroing before the prefill changes nothing
        return st.call("prefill", st.steps, prefill_body, dict(prefill_signature.bind(*args, **kwargs).arguments))
    model.mot.prefill_video_cache_tensor = prefill

    action_original = model._denoise_action_with_video_cache
    signature = inspect.signature(action_original)

    def action(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        st.phase = ("action", st.n_action)
        st.shapes["action_tokens"] = int(bound.arguments["latents_action"].shape[1])
        st.shapes["context_tokens"] = int(bound.arguments["context"].shape[1])
        if not st.quant and st.inject is not None and st.inject[1] == "redrawn" and st.inject[0] in ("obsKV", "futKV"):
            bound.arguments["video_cache_k"], bound.arguments["video_cache_v"] = st.cache_kv()
        out = st.call("action", st.n_action, action_original, dict(bound.arguments))
        st.n_action += 1
        if st.n_action == st.steps:
            st.finish_query()
        return out
    model._denoise_action_with_video_cache = action

    flash_original = mot_module.flash_attention

    def flash(q=None, k=None, v=None, num_heads=None, ctx_mask=None, **kwargs):
        st.observe(q, k, num_heads, ctx_mask)
        return flash_original(q=q, k=k, v=v, num_heads=num_heads, ctx_mask=ctx_mask, **kwargs)
    mot_module.flash_attention = flash
    return installed


def load_weights(st, modules, index_of, bank_dir):
    """Decode the fitted 4-bit weights into the Linears; from here on the forwards quantize."""
    loaded = 0
    for name, module in modules.items():
        if mt.parse(name) is None:
            continue
        bank = torch.load(bank_dir / f"{index_of[name]:04d}.pt", map_location=st.device, weights_only=True)
        if bank["module"] != name or tuple(bank["shape"]) != tuple(module.weight.shape):
            raise RuntimeError(f"Bank does not match {name}")
        with torch.no_grad():
            module.weight.copy_(q.decode(bank).to(module.weight.dtype))
        loaded += 1
    st.quant, st.inject = True, None
    st.shared, st.frozen = None, {}
    return loaded


# ---------------- the reader metric
class Metric:
    """M = (1 - rho) (D / tau) (E^T diag(lam) E + diag(d)) + rho I on the predicted frames, flattened (c, f, h, w)."""

    def __init__(self, saved, device):
        self.E = saved["E"].to(device, torch.float32)
        self.lam = saved["lam"].to(device, torch.float32)
        self.dres = saved["d_res"].to(device, torch.float32)
        self.rho, self.D, self.tau = float(saved["rho"]), int(saved["D"]), float(saved["tau"])
        self.c = (1 - self.rho) * self.D / self.tau
        self.device = device

    def quad(self, v):
        v = v.reshape(-1).to(self.device, torch.float32)
        p = self.E @ v
        return float(self.c * ((self.lam * p.square()).sum() + (self.dres * v.square()).sum()) + self.rho * v.square().sum())

    def nmse(self, future, target):
        t = target[:, :, 1:].to(self.device, torch.float32)
        e = future[:, :, 1:].to(self.device, torch.float32) - t
        return self.quad(e) / float(t.square().sum().clamp_min(1e-12))


def token_gaze(saved, patch=2):
    """Offline gaze: the trace of the block of M that belongs to each future token, order (frame, row, column)."""
    rho, D, tau = float(saved["rho"]), int(saved["D"]), float(saved["tau"])
    _, c, f, h, w = saved["shape"]
    m = ((1 - rho) * D / tau * saved["diag"].double() + rho).reshape(c, f, h, w).sum(0)
    return m.reshape(f, h // patch, patch, w // patch, patch).sum((2, 4)).reshape(-1).float()


def errors(metric, action, future, teacher):
    a, z = action.float().cpu(), future.float().cpu()
    ta, tz = teacher["action"].float(), teacher["future"].float()
    return dict(A=float((a[:10, :6] - ta[:10, :6]).square().mean()),
                Z=float((z[:, :, 1:] - tz[:, :, 1:]).square().sum() / tz[:, :, 1:].square().sum()),
                ZM=metric.nmse(z, tz), gripper_flips=int((a[:10, 6].sign() != ta[:10, 6].sign()).sum()))


# ---------------- costs and the allocation (plain Python)
def costs(names_and_widths, arch, steps, frame, video, action_tokens, context_tokens):
    """Operand values (4-bit numbers) of one query, counted as a deployment computes them: the observed-frame rows
    once, operands shared by several Linears once. names_and_widths: [(module name, in_features)]."""
    tok, ctx, blocks = Counter(), Counter(), Counter()
    for name, width in names_and_widths:
        role = mt.parse(name)
        if role is None:
            continue
        if role.attn == "cross" and role.kind in ("k", "v"):
            if role.kind == "k":
                ctx[role.expert] += width
        elif not (role.attn == "self" and role.kind in ("k", "v")):
            tok[role.expert] += width
        if role.attn == "self" and role.kind == "q":
            blocks[role.expert] += 1
    kv_tok = {e: blocks[e] * 2 * arch["heads"] * arch["head_dim"] for e in blocks}
    fut = video - frame
    unit = {"obsA": dict(a=tok["video"], kv=0), "fut": dict(a=tok["video"], kv=0),
            "obsKV": dict(a=0, kv=kv_tok["video"]), "futKV": dict(a=0, kv=kv_tok["video"]),
            "action": dict(a=action_tokens * steps * tok["action"], kv=0)}
    for e in ("video", "action"):
        unit[f"text.{e}"] = dict(a=context_tokens * ctx[e], kv=context_tokens * kv_tok[e])
    base = dict(a=(frame + fut * (steps + 1)) * tok["video"] + unit["action"]["a"]
                + unit["text.video"]["a"] + unit["text.action"]["a"],
                kv=video * kv_tok["video"] + unit["text.video"]["kv"] + unit["text.action"]["kv"])
    return dict(unit=unit, base=base, token_pass=dict(tok), context_pass=dict(ctx), kv_token=kv_tok)


def solve(benefit, cost, g_fut, g_obs, fraction, steps):
    """Threshold rule: refine a unit iff benefit / cost >= mu, with mu set by the budget (a fraction of the operand
    values of a query). The action rows are always refined and charged first. The benefit of a class is split over its
    tokens in proportion to the gaze. Returns the spec (counts per class), mu and the accounting."""
    unit, base = cost["unit"], cost["base"]
    price = {k: v["a"] + v["kv"] for k, v in unit.items()}
    budget = fraction * (base["a"] + base["kv"]) - price["action"]
    if budget < 0:
        raise ValueError("The budget does not cover the action rows")
    gf = [max(float(x), 0.0) for x in g_fut]
    go = [max(float(x), 0.0) for x in g_obs]
    gf = [x / (sum(gf) or 1.0) for x in gf]
    go = [x / (sum(go) or 1.0) for x in go]
    cands = [(benefit[c] / price[c], price[c], c) for c in ("text.video", "text.action")]
    cands += [(benefit[c] * g / price[c], price[c], c) for c in ("obsA", "obsKV") for g in go]
    cands += [(benefit["futKV"] * g / price["futKV"], price["futKV"], "futKV") for g in gf]
    cands += [(benefit[f"fut.{s}"] * g / price["fut"], price["fut"], f"fut.{s}") for s in range(steps + 1) for g in gf]
    cands.sort(key=lambda t: -t[0])
    taken, spent, mu = Counter(), 0.0, None
    for ratio, c, cls in cands:
        if ratio <= 0:
            break
        if spent + c <= budget:
            taken[cls] += 1
            spent += c
            mu = ratio
    spec = dict(text_video=taken["text.video"] > 0, text_action=taken["text.action"] > 0, action=True,
                obsA=taken["obsA"], obsKV=taken["obsKV"], futKV=taken["futKV"],
                fut=[taken[f"fut.{s}"] for s in range(steps + 1)])
    return dict(spec=spec, mu=mu, accounting=accounting(spec, cost), budget_values=budget + price["action"])


def accounting(spec, cost):
    unit, base = cost["unit"], cost["base"]
    extra = dict(a=0.0, kv=0.0)
    for cls, n in (("text.video", int(spec["text_video"])), ("text.action", int(spec["text_action"])),
                   ("action", int(spec["action"])), ("obsA", spec["obsA"]), ("obsKV", spec["obsKV"]),
                   ("futKV", spec["futKV"]), ("fut", sum(spec["fut"]))):
        extra["a"] += n * unit[cls]["a"]
        extra["kv"] += n * unit[cls]["kv"]
    return dict(extra_a=extra["a"], extra_kv=extra["kv"], base_a=base["a"], base_kv=base["kv"],
                a_bits=4 * (1 + extra["a"] / base["a"]), kv_bits=4 * (1 + extra["kv"] / base["kv"]),
                fraction=(extra["a"] + extra["kv"]) / (base["a"] + base["kv"]))
