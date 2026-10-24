"""Activation-only clipping after the complete frozen Smooth/H/R transform."""
import math
import torch
import torch.nn.functional as F
from rotation_bank import transformed, stream_for, A4


class ClippedA4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, ratio):
        z = z.float()
        maximum = z.abs().amax(-1, keepdim=True)
        scale = maximum * ratio / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        u = z / safe
        q = u.round().clamp(-7, 7)
        inside = z.abs() <= maximum * ratio
        derivative = torch.where(inside, q - u, q)
        ctx.save_for_backward(inside, derivative.to(torch.bfloat16), maximum / 7)
        ctx.normalizer = math.sqrt(z.numel() * 7)
        return (q * scale).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        inside, derivative, base_scale = ctx.saved_tensors
        g = grad.float()
        return g * inside, (g * derivative.float() * base_scale).sum() / ctx.normalizer


class ClipBank:
    def __init__(self, bank):
        self.bank = bank
        self.ratios = {n: torch.nn.Parameter(torch.ones((), device='cuda')) for n in bank.modules}
        self.original_forwards = {n: m.forward for n, m in bank.modules.items()}
        for parameter in bank.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for name, module in bank.modules.items():
            def forward(x, _name=name, _module=module):
                bank.seen.add(_name)
                z = transformed(x, bank.transforms[_name], bank.rotations, stream_for(_name))
                return F.linear(ClippedA4.apply(z, self.ratios[_name]), _module.weight, _module.bias)
            module.forward = forward

    def parameters(self):
        return list(self.ratios.values())

    @torch.no_grad()
    def project(self):
        for ratio in self.ratios.values():
            ratio.clamp_(.5, 1.)

    def snapshot(self):
        return {n: p.detach().cpu().clone() for n, p in self.ratios.items()}

    @torch.no_grad()
    def restore(self, values):
        if set(values) != set(self.ratios):
            raise ValueError('Clipping module coverage differs')
        for n, p in self.ratios.items():
            if values[n].numel() != 1 or not .5 <= float(values[n]) <= 1:
                raise ValueError('Invalid clipping ratio')
            p.copy_(values[n])

    def summary(self):
        values = sorted(float(p.detach()) for p in self.parameters())
        return dict(count=len(values), min=values[0], max=values[-1], mean=sum(values)/len(values),
                    median=(values[306]+values[307])/2, below_one=sum(v<1 for v in values),
                    below_point9=sum(v<.9 for v in values), at_minimum=sum(v<=.5000001 for v in values))


def selfcheck():
    z = torch.tensor([[0., 1., -2., 7.], [0., 0., 0., 0.]], device='cuda', requires_grad=True)
    r = torch.tensor(1., device='cuda', requires_grad=True)
    if not torch.equal(ClippedA4.apply(z, r), A4.apply(z)):
        raise ValueError('r=1 A4 operand parity failed')
    r = torch.tensor(.5, device='cuda', requires_grad=True)
    y = ClippedA4.apply(z, r)
    expected = torch.tensor([[0., 1., -2., 3.5], [0., 0., 0., 0.]], device='cuda', dtype=torch.bfloat16)
    if not torch.equal(y, expected):
        raise ValueError('Clipped grid or zero-input check failed')
    y.float().sum().backward()
    if not torch.isfinite(r.grad) or not torch.isfinite(z.grad).all() or z.grad[0, 3] != 0:
        raise ValueError('Clipping gradient check failed')
    return dict(passed=True, r1_exact=True, clipping_grid=True, zero_input=True, finite_gradient=True)
