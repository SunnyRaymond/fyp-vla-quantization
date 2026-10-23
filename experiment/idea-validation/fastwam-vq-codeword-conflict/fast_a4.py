"""Reuse the project's bitwise-equivalent batched H128 for the same fake-A4 path."""
import torch
import batched_transform

IMPLEMENTATION = batched_transform.IMPLEMENTATION


def operand(x, transform):
    z = batched_transform.transform_input(x, transform).float()
    scale = z.abs().amax(dim=-1, keepdim=True) / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(z / safe).clamp(-7, 7) * scale).to(torch.bfloat16)


class A4Input(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, transform):
        ctx.transform, ctx.dtype = transform, x.dtype
        return operand(x, transform)

    @staticmethod
    def backward(ctx, grad):
        t = ctx.transform
        g = batched_transform.inverse_right_r(grad.float(), t['signs'], t['hadamard'], t['block'])
        return (g / t['scale']).to(ctx.dtype), None


def install_reader(weights):
    for name, module in weights.modules.items():
        def forward(x, _name=name, _module=module):
            weights.seen.add(_name)
            if weights.current == 'bf16':
                return weights.original[_name](x)
            return torch.nn.functional.linear(operand(x, weights.transforms[_name]), _module.weight, _module.bias)
        module.forward = forward
