"""Frozen Smooth, task-learned R128, and the existing C8 BF16/A4 VQ decoder."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import batched_transform
import micro_codebook as micro
from c8_train_banks import DecodedWeight, TrainableBanks


def stream_for(name):
    if name.startswith('video_expert.'):
        return 'video'
    if name.startswith('action_expert.'):
        return 'action'
    if name == 'proprio_encoder' or name.startswith('proprio_encoder.'):
        return 'proprio'
    raise ValueError('Unknown target stream: ' + name)


class A4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        scale = z.float().abs().amax(-1, keepdim=True) / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(z.float() / safe).clamp(-7, 7) * scale).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        # Match the original C8 identity STE; channel/absmax surrogates are not introduced.
        return grad.float()


class Rotations(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()
        streams = sorted({stream_for(n) for n, m in modules.items() if m.in_features >= 128})
        self.raw = torch.nn.ParameterDict({s: torch.nn.Parameter(torch.zeros(128, 128, device='cuda'))
                                           for s in streams})
        self.cache = {}

    def matrix(self, stream):
        if stream not in self.cache:
            a = self.raw[stream]
            k = a - a.t()
            eye = torch.eye(128, device=a.device, dtype=torch.float32)
            self.cache[stream] = torch.linalg.solve(eye - .5 * k, eye + .5 * k)
        return self.cache[stream]

    def refresh(self):
        self.cache.clear()

    def apply_after_h(self, z, stream):
        full = z.shape[-1] // 128 * 128
        if not full:
            return z
        with torch.autocast(device_type='cuda', enabled=False):
            head = z[..., :full].float().reshape(*z.shape[:-1], full // 128, 128)
            head = (head @ self.matrix(stream)).reshape(*z.shape[:-1], full)
            return torch.cat((head, z[..., full:]), -1) if full < z.shape[-1] else head

    def snapshot(self):
        return {s: a.detach().cpu().clone() for s, a in self.raw.items()}

    @torch.no_grad()
    def restore(self, values):
        if set(values) != set(self.raw):
            raise RuntimeError('Rotation stream set changed')
        for stream, value in values.items():
            if tuple(value.shape) != (128, 128) or not bool(torch.isfinite(value).all()):
                raise RuntimeError('Invalid saved rotation')
            self.raw[stream].copy_(value)
        self.refresh()

    def freeze(self):
        for value in self.raw.values():
            value.requires_grad_(False)
        self.refresh()

    def orthogonality(self):
        return {s: float((self.matrix(s).t() @ self.matrix(s)
                          - torch.eye(128, device='cuda')).abs().max()) for s in self.raw}


def transformed(x, transform, rotations, stream, weight=False):
    if weight:
        # Both operands use the same saved signs/H; Smooth is paired as W*S versus X/S.
        scaled = x.float() * transform['scale']
        full = scaled.shape[-1] // 128 * 128
        pieces = []
        if full:
            head = (scaled[..., :full] * transform['signs'][:full]).reshape(*scaled.shape[:-1], full // 128, 128)
            pieces.append(batched_transform.core._fwht(head).reshape(*scaled.shape[:-1], full))
        if full < scaled.shape[-1]:
            pieces.append(batched_transform.core._right_r(scaled[..., full:], transform['signs'][full:], True, 128))
        z = pieces[0] if len(pieces) == 1 else torch.cat(pieces, -1)
    else:
        z = batched_transform.transform_input(x, transform)
    return rotations.apply_after_h(z, stream)


class RotationBanks:
    def __init__(self, runtime, directory):
        self.modules = runtime.q.modules
        if len(self.modules) != 614:
            raise RuntimeError('Expected exactly 614 target Linears')
        for value in runtime.model.parameters():
            value.requires_grad_(False)
        runtime.q.disable()
        self.transforms = {}
        self.original = {n: m.forward for n, m in self.modules.items()}
        self.seen, self.moments, self.moment_rows = set(), {}, {}
        self.collect = False
        self.rotations = Rotations(self.modules)
        for index, (name, mod) in enumerate(self.modules.items()):
            bank = torch.load(Path(directory) / f'{index:04d}.pt', map_location='cuda', weights_only=True)
            if (bank.get('module') != name or bank.get('module_index') != index
                    or tuple(bank['encoded']['shape']) != tuple(mod.weight.shape)
                    or bank.get('cohort') != 'c8'):
                raise RuntimeError('Source bank identity differs: ' + name)
            self.transforms[name] = bank['transform']
            if not bank['transform']['hadamard'] or bank['transform']['block'] != 128:
                raise RuntimeError('Source transform must be frozen signed H128')
            mod.forward = self.make_forward(name, mod)

    def operand(self, x, name):
        return A4.apply(transformed(x, self.transforms[name], self.rotations, stream_for(name)))

    def weight(self, name):
        return transformed(self.modules[name].weight, self.transforms[name], self.rotations,
                           stream_for(name), weight=True)

    def make_forward(self, name, mod):
        def forward(x):
            self.seen.add(name)
            matrix = (self.rotations.matrix(stream_for(name)) if mod.in_features >= 128 else x.new_empty(0))

            def run(value, unused_matrix):
                qx = self.operand(value, name)
                if self.collect:
                    rows = qx.detach().float().reshape(-1, mod.in_features)
                    total = rows.square().sum(0)
                    self.moments[name] = self.moments.get(name, 0) + total
                    self.moment_rows[name] = self.moment_rows.get(name, 0) + rows.shape[0]
                weight = self.weight(name).to(torch.bfloat16)
                return F.linear(qx, weight, mod.bias)

            if torch.is_grad_enabled() and any(v.requires_grad for v in self.rotations.parameters()):
                return checkpoint(run, x, matrix, use_reentrant=False)
            return run(x, matrix)
        return forward

    def close(self):
        for name, mod in self.modules.items():
            mod.forward = self.original[name]


class VQBanks(TrainableBanks):
    def __init__(self, runtime, directory, rotations, transforms, protocol):
        self.modules = runtime.q.modules
        self.rotations, self.transforms = rotations, transforms
        self.forwards = {name: mod.forward for name, mod in self.modules.items()}
        self.encodings, self.books, self.logs, self.active = {}, {}, {}, {}
        self.seen = set()
        for index, (name, mod) in enumerate(self.modules.items()):
            bank = torch.load(Path(directory) / f'{index:04d}.pt', map_location='cuda', weights_only=True)
            if bank.get('protocol') != protocol or bank.get('module') != name or bank.get('module_index') != index:
                raise RuntimeError('R8 VQ bank binding differs')
            encoding = bank['encoded']
            micro.check_encoding(encoding)
            self.encodings[name] = encoding
            self.books[name] = torch.nn.Parameter(encoding['books'].float().clone())
            scales = encoding['row_scales'].float()
            self.active[name] = (scales > 0).float()
            self.logs[name] = torch.nn.Parameter(scales.clamp_min(torch.finfo(torch.float32).tiny).log())
            mod.weight.data = micro.decode_vq(encoding)

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                qx = A4.apply(transformed(x, self.transforms[_name], self.rotations, stream_for(_name)))
                weight = DecodedWeight.apply(self.books[_name], self.logs[_name], self.active[_name],
                    self.encodings[_name]['indices'], self.encodings[_name]['scale_exponents'], _mod.weight)
                return F.linear(qx, weight, _mod.bias)
            mod.forward = forward
        self.refresh()


