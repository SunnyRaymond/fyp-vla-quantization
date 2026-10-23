"""Clipping ratios on top of the frozen C8 bank training implementation."""
import math

import torch
import batched_transform
import train_banks as base


class ClippedA4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ratio, transform):
        z = batched_transform.transform_input(x, transform).float()
        maximum = z.abs().amax(dim=-1, keepdim=True)
        scale = maximum * ratio / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        u = z / safe
        q = u.round().clamp(-7, 7)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            inside = z.abs() <= maximum * ratio
            derivative = torch.where(inside, q - u, q)
            # Save the scale derivative in operand precision to bound checkpoint memory.
            ctx.save_for_backward(inside, derivative.to(torch.bfloat16), maximum / 7)
            ctx.normalizer = math.sqrt(z.numel() * 7)
        ctx.transform, ctx.dtype = transform, x.dtype
        return (q * scale).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        inside, derivative, base_scale = ctx.saved_tensors
        g = grad.float()
        ratio_grad = (g * derivative.float() * base_scale).sum() / ctx.normalizer
        input_grad = batched_transform.inverse_right_r(
            g * inside, ctx.transform['signs'], ctx.transform['hadamard'], ctx.transform['block'])
        return (input_grad / ctx.transform['scale']).to(ctx.dtype), ratio_grad, None


class ClipBanks(base.TrainableBanks):
    def __init__(self, runtime, bank_directory, masters, arm, clipping):
        super().__init__(runtime, bank_directory)
        self.restore(masters)
        self.train_weights = arm in ('weight_only', 'joint')
        self.train_clips = arm in ('clip_only', 'joint')
        self.minimum, self.maximum = clipping['minimum_ratio'], clipping['maximum_ratio']
        self.ratios = {name: torch.nn.Parameter(
            torch.tensor(clipping['initial_ratio'], device='cuda', dtype=torch.float32),
            requires_grad=self.train_clips) for name in self.modules}
        for name, mod in self.modules.items():
            self.books[name].requires_grad_(self.train_weights)
            self.logs[name].requires_grad_(self.train_weights)

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                if self.train_clips:
                    qx = ClippedA4.apply(x, self.ratios[_name], self.transforms[_name])
                else:
                    qx = base.A4Input.apply(x, self.transforms[_name])
                weight = _mod.weight
                if self.train_weights:
                    weight = base.DecodedWeight.apply(
                        self.books[_name], self.logs[_name], self.active[_name],
                        self.encodings[_name]['indices'], self.encodings[_name]['scale_exponents'], weight)
                return torch.nn.functional.linear(qx, weight, _mod.bias)

            mod.forward = forward

    def optimizer_groups(self, training):
        groups = []
        if self.train_weights:
            groups.append({'params': [p for name in self.modules for p in
                                     (self.books[name], self.logs[name])],
                           'lr': training['weight_learning_rate']})
        if self.train_clips:
            groups.append({'params': list(self.ratios.values()), 'lr': training['clip_learning_rate']})
        return groups

    @torch.no_grad()
    def project(self):
        for ratio in self.ratios.values():
            ratio.clamp_(self.minimum, self.maximum)
        if self.train_weights:
            self.refresh()

    def clip_snapshot(self):
        return {name: value.detach().cpu().clone() for name, value in self.ratios.items()}

    @torch.no_grad()
    def restore_clips(self, values):
        if set(values) != set(self.modules):
            raise RuntimeError('Clipping checkpoint module coverage differs')
        for name, value in values.items():
            if value.numel() != 1 or not self.minimum <= float(value) <= self.maximum:
                raise RuntimeError('Invalid clipping ratio')
            self.ratios[name].copy_(value)

    def clip_summary(self):
        values = [float(x.detach()) for x in self.ratios.values()]
        return {'count': len(values), 'min': min(values), 'max': max(values),
                'mean': sum(values) / len(values),
                'below_one': sum(x < 1 for x in values)}

    def coverage(self):
        groups = {}
        if self.train_weights:
            groups['weight'] = [(self.books[name], self.logs[name]) for name in self.modules]
        if self.train_clips:
            groups['clip'] = [(self.ratios[name],) for name in self.modules]
        result = {}
        for key, rows in groups.items():
            result[key] = {
                'connected': sum(all(p.grad is not None for p in row) for row in rows),
                'finite': sum(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                  for p in row) for row in rows),
                'nonzero': sum(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                   for p in row) for row in rows)}
        return result
