"""Cross-step activation quantizers for iterative samplers, and the D1 step recorder. No model imports.

A target Linear is called once per denoising step. The quantizer keeps per-(layer, input shape) state across
those calls within one query:

    Delta_s = z_s - xhat_{s-1}                (innovation, if enabled; else Delta_s = z_s)
    v_s     = Delta_s + h * w_{s-1}           (error feedback, if h != 0)
    q_s     = Q(v_s),  w_s = q_s - v_s        (per-token absmax grid, optional channel groups / dither)
    xhat_s  = xhat_{s-1} + q_s                (operand handed to the GEMM)

so xhat_s - z_s = w_s + h * w_{s-1} for every step. An input identical to the previous call of the same key
reuses that call's operand, as the deployed condition cache computes it once.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def grid(v, bits, group=None, noise=None):
    """Symmetric absmax grid per token (last dim) or per `group` channels; dequantized FP32.

    bits=4, group=None, noise=None is bitwise rotation_bank.A4 before its BF16 cast.
    """
    qmax = 2 ** (bits - 1) - 1
    v = v.float()
    width = v.shape[-1]
    grouped = bool(group) and group < width
    if grouped:
        pad = (-width) % group
        padded = F.pad(v, (0, pad)) if pad else v
        shaped = padded.reshape(*v.shape[:-1], padded.shape[-1] // group, group)
    else:
        shaped = v
    scale = shaped.abs().amax(-1, keepdim=True) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    ratio = shaped / safe
    if noise is not None:
        ratio = ratio + noise(ratio)
    q = torch.round(ratio).clamp(-qmax, qmax) * scale
    if grouped:
        q = q.reshape(*v.shape[:-1], -1)[..., :width]
    return q


class State:
    __slots__ = ('raw', 'operand', 'xhat', 'w', 'calls', 'static')

    def __init__(self):
        self.raw = self.operand = self.xhat = self.w = None
        self.calls = self.static = 0


def step(z, bits, group, innovation, feedback, state, noise=None):
    """One cross-step quantization; returns (xhat, w) with xhat - z = w + feedback * w_prev."""
    base = state.xhat if innovation and state.xhat is not None else None
    v = z if base is None else z - base
    if feedback and state.w is not None:
        v = v + feedback * state.w
    q = grid(v, bits, group, noise)
    xhat = q if base is None else base + q
    return xhat, (q - v) if feedback else None


class CrossStepQuantizer:
    """Per-arm operand producer. `transform(name, x)` gives the FP32 quantizer-coordinate input,
    `stream(name)` its stream; `prefill` = {'stream', 'call_index'} marks the single conditioning pass."""

    def __init__(self, spec, transform, stream, prefill=None, recorder=None):
        self.spec, self.transform, self.stream = dict(spec), transform, stream
        self.prefill, self.recorder = prefill, recorder
        self.innovation = bool(self.spec.get('innovation', False))
        self.feedback = float(self.spec.get('feedback', 0.0))
        self.group = self.spec.get('group')
        self.dither = bool(self.spec.get('dither', False))
        self.states, self.seed, self.generator = {}, 0, None

    def reset(self, seed=0):
        self.states.clear()
        self.seed, self.generator = int(seed), None

    def _noise(self, like):
        if self.generator is None or self.generator.device != like.device:
            self.generator = torch.Generator(device=like.device).manual_seed(self.seed)
        return torch.rand(like.shape, generator=self.generator, device=like.device, dtype=torch.float32) - 0.5

    def bits_for(self, name, call):
        stream = self.stream(name)
        if (self.prefill and self.spec.get('prefill_bits') and stream == self.prefill['stream']
                and call == self.prefill['call_index']):
            return int(self.spec['prefill_bits'])
        if stream == 'action' and self.spec.get('action_bits'):
            return int(self.spec['action_bits'])
        return int(self.spec['bits'])

    def operand(self, name, x):
        key = (name, tuple(x.shape))
        st = self.states.get(key)
        if st is None:
            st = self.states[key] = State()
        elif st.raw is not None and torch.equal(st.raw, x):
            st.calls += 1
            st.static += 1
            return st.operand
        call = st.calls
        z = self.transform(name, x)
        if self.recorder is not None:
            self.recorder(name, key, call, z)
        bits = self.bits_for(name, call)
        if bits >= 16:
            xhat, w = z, None
        else:
            xhat, w = step(z, bits, self.group, self.innovation, self.feedback, st,
                           self._noise if self.dither else None)
        st.operand = xhat.to(torch.bfloat16)
        st.raw = x.detach().clone()
        st.xhat = xhat if self.innovation else None
        st.w = w if self.feedback else None
        st.calls += 1
        return st.operand

    def call_counts(self):
        return {f'{name}|{shape[-2] if len(shape) > 1 else 1}': dict(calls=s.calls, static=s.static)
                for (name, shape), s in self.states.items()}


class StepRecorder:
    """D1: per (layer, shape, call) statistics of the quantizer-coordinate input z under the BF16 forward.

    Per token group: [|z|^2, |z - z_prev|^2, mean_t amax|z - z_prev| / amax|z|, <r, r_prev>, |r|^2, |r_prev|^2,
    has_prev], with r = A4 RTN residual of z. `groups(name, tokens)` returns [(label, slice over tokens)].
    """
    FIELDS = ('energy', 'innovation_energy', 'absmax_ratio', 'residual_dot', 'residual_sq', 'previous_residual_sq',
              'has_previous')

    def __init__(self, groups, stream):
        self.groups, self.stream = groups, stream
        self.prev, self.rows, self.meta = {}, [], {}

    def __call__(self, name, key, call, z):
        r = grid(z, 4) - z
        prev = self.prev.get(key)
        tokens = z.shape[-2] if z.dim() > 1 else 1
        groups = self.groups(name, tokens)
        values = []
        zero = z.new_zeros(())
        for _label, part in groups:
            zg, rg = z[..., part, :], r[..., part, :]
            if prev is None:
                ed = amr = rr = rp2 = zero
                has = zero
            else:
                d = zg - prev[0][..., part, :]
                rp = prev[1][..., part, :]
                ed = d.square().sum()
                amax_z, amax_d = zg.abs().amax(-1), d.abs().amax(-1)
                amr = torch.where(amax_z > 0, amax_d / torch.where(amax_z > 0, amax_z, torch.ones_like(amax_z)),
                                  torch.zeros_like(amax_z)).mean()
                rr, rp2 = (rg * rp).sum(), rp.square().sum()
                has = z.new_ones(())
            values += [zg.square().sum(), ed, amr, rr, rg.square().sum(), rp2, has]
        self.rows.append((key, call, torch.stack(values).float()))
        self.prev[key] = (z, r)
        self.meta[key] = dict(name=name, stream=self.stream(name), tokens=tokens, width=z.shape[-1],
                              groups=[label for label, _ in groups])

    def flush(self):
        """Return [(key, call, {label: {field: value}})] for the finished query and clear the per-query state."""
        if not self.rows:
            return []
        flat = torch.cat([v for _, _, v in self.rows]).cpu().tolist()
        out, at = [], 0
        for key, call, v in self.rows:
            labels = self.meta[key]['groups']
            n = len(self.FIELDS)
            block = flat[at:at + v.numel()]
            at += v.numel()
            out.append((key, call, {lab: dict(zip(self.FIELDS, block[i * n:(i + 1) * n]))
                                    for i, lab in enumerate(labels)}))
        self.rows.clear()
        self.prev.clear()
        return out
