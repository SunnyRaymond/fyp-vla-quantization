"""Per-input-channel D, with the completed B32 stored-weight/amax surrogate."""
import math

import torch
import batched_transform
from clip_banks import ClipBanks


def channel_scales(log2_scales, width):
    if log2_scales.shape != (width,):
        raise RuntimeError('One log2 scale per physical input channel is required')
    return torch.exp2(log2_scales)


def compensated_weight(weight, log2_scales):
    return (weight.float() * channel_scales(log2_scales, weight.shape[1])).to(torch.bfloat16)


def quantized_input(x, ratio, log2_scales, transform):
    z = batched_transform.transform_input(x, transform).float()
    d = channel_scales(log2_scales, z.shape[-1]); v = z / d
    maximum = v.abs().amax(-1, keepdim=True); scale = maximum * ratio / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale)); u = v / safe
    q = u.round().clamp(-7, 7)
    return (q * scale).to(torch.bfloat16)


def coupled_surrogate(g, inside, derivative, base_scale, d, ratio, max_sign, tie_count):
    width = g.shape[-1]; rows = g.numel() // width
    term = g * derivative.float() * base_scale
    ratio_grad = term.sum() / math.sqrt(g.numel() * 7)
    support = max_sign.float() / tie_count
    dm = (g * derivative.float()).sum(-1, keepdim=True) * ratio / 7
    z_grad = (g * inside + dm * support) / d
    per_channel = term * ratio * math.log(2)
    per_channel -= per_channel.sum(-1, keepdim=True) * support.abs()
    # Width=1 is the sole granularity change: retain row reduction and common-scale centering.
    scale_grad = per_channel.reshape(rows, width).sum(0) / math.sqrt(rows * 7)
    scale_grad -= scale_grad.mean()
    return z_grad, ratio_grad, scale_grad


class ChannelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ratio, log2_scales, weight, bias, transform):
        z = batched_transform.transform_input(x, transform).float()
        d = channel_scales(log2_scales, z.shape[-1]); v = z / d
        maximum = v.abs().amax(-1, keepdim=True); scale = maximum * ratio / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale)); u = v / safe
        q = u.round().clamp(-7, 7)
        result = torch.nn.functional.linear((q * scale).to(torch.bfloat16),
                                           compensated_weight(weight, log2_scales), bias)
        if any(ctx.needs_input_grad[:3]):
            inside = v.abs() <= maximum * ratio
            derivative = torch.where(inside, q - u, q).to(torch.bfloat16)
            maxima = v.abs() == maximum
            max_sign = torch.where(maxima, v.sign(), torch.zeros_like(v)).to(torch.int8)
            tie_count = maxima.sum(-1, keepdim=True)
            ctx.save_for_backward(inside, derivative, maximum / 7, d, ratio,
                                  log2_scales, weight, max_sign, tie_count)
        ctx.transform, ctx.dtype = transform, x.dtype
        return result

    @staticmethod
    def backward(ctx, grad):
        inside, derivative, base_scale, d, ratio, log2_scales, weight, max_sign, tie_count = ctx.saved_tensors
        g = torch.nn.functional.linear(grad, compensated_weight(weight, log2_scales).t()).float()
        z_grad, ratio_grad, scale_grad = coupled_surrogate(
            g, inside, derivative, base_scale, d, ratio, max_sign, tie_count)
        x_grad = batched_transform.inverse_right_r(z_grad, ctx.transform['signs'],
            ctx.transform['hadamard'], ctx.transform['block']) / ctx.transform['scale']
        return x_grad.to(ctx.dtype), ratio_grad, scale_grad, None, None, None


class ChannelBanks(ClipBanks):
    def __init__(self, runtime, bank_directory, masters, clipping, scaling):
        super().__init__(runtime, bank_directory, masters, 'clip_only', clipping)
        if scaling['group_size'] != 1:
            raise RuntimeError('This arm is per-channel only')
        self.shadows = {}; self.cached = False; self.static = {}
        self.lower, self.upper = scaling['minimum_log2'], scaling['maximum_log2']
        for name, mod in self.modules.items():
            self.shadows[name] = torch.nn.Parameter(torch.zeros(mod.weight.shape[1], device='cuda'))
            self.static[name] = {key: (self.encodings[name][key].data_ptr(), self.encodings[name][key]._version)
                                 for key in ('indices', 'scale_exponents')}

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                if self.cached:
                    operand = quantized_input(x, self.ratios[_name], self.shadows[_name], self.transforms[_name])
                    return torch.nn.functional.linear(operand, _mod.weight, _mod.bias)
                return ChannelLinear.apply(x, self.ratios[_name], self.shadows[_name],
                                           _mod.weight, _mod.bias, self.transforms[_name])
            mod.forward = forward

    def optimizer_groups(self, training, scaling):
        return [{'params': list(self.ratios.values()), 'lr': training['clip_learning_rate']},
                {'params': list(self.shadows.values()), 'lr': scaling['learning_rate']}]

    @torch.no_grad()
    def project(self):
        super().project()
        for value in self.shadows.values(): value.clamp_(self.lower, self.upper)

    def scale_snapshot(self):
        return {name: value.detach().cpu().clone() for name, value in self.shadows.items()}

    @torch.no_grad()
    def restore_scales(self, values):
        if set(values) != set(self.shadows): raise RuntimeError('Scale module coverage differs')
        for name, value in values.items():
            if (value.shape != self.shadows[name].shape or not bool(torch.isfinite(value).all())
                    or not bool(((value >= self.lower) & (value <= self.upper)).all())):
                raise RuntimeError('Invalid per-channel log2 scale')
            self.shadows[name].copy_(value)

    def scaling_summary(self):
        logs = torch.cat([v.detach() for v in self.shadows.values()]); d = torch.exp2(logs)
        return {'modules': len(self.shadows), 'group_size': 1, 'channels': logs.numel(),
                'changed_channels': int((d != 1).sum()),
                'channels_changed_gt_1pct': int(((d - 1).abs() > .01).sum()),
                'channels_changed_gt_5pct': int(((d - 1).abs() > .05).sum()),
                'changed_modules': sum(bool((v.detach() != 0).any()) for v in self.shadows.values()),
                'scale_min': float(d.min()), 'scale_median': float(d.median()), 'scale_max': float(d.max()),
                'mean_abs_log2': float(logs.abs().mean()), 'scale_metadata_dtype': 'float32_log2',
                'raw_scale_metadata_bytes': logs.numel() * 4, 'cached_compensated_weight_evaluation': self.cached,
                'dynamic_absmax_coupling': True, 'input_absmax_chain': True, 'amax_ties': 'equal_share'}

    def coverage(self):
        result = super().coverage(); values = list(self.shadows.values())
        result['channel'] = {'connected': sum(v.grad is not None for v in values),
            'finite': sum(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in values),
            'nonzero': sum(v.grad is not None and bool((v.grad != 0).any()) for v in values)}
        return result

    @torch.no_grad()
    def cache_compensated_weights(self):
        if self.cached: raise RuntimeError('Do not compensate weights twice')
        for value in list(self.ratios.values()) + list(self.shadows.values()): value.requires_grad_(False)
        errors = []; numerator = 0.; denominator = 0.
        for name, mod in self.modules.items():
            original = mod.weight; d = channel_scales(self.shadows[name], original.shape[1])
            adjusted = compensated_weight(original, self.shadows[name])
            error = (adjusted.float() / d - original.float()).square().sum(); energy = original.float().square().sum()
            n, e = float(error), float(energy); numerator += n; denominator += e
            errors.append({'module': name, 'nmse': n / max(e, 1e-30)})
            mod.weight.data = adjusted
        self.cached = True
        return {'scope': 'compensated BF16 weight restored to original coordinates versus original decoded BF16 baseline',
                'nmse': numerator / max(denominator, 1e-30),
                'maximum_module_nmse': max(v['nmse'] for v in errors), 'modules': len(errors), 'per_module': errors}

    def verify_frozen(self, masters):
        for name, master in masters.items():
            if (not torch.equal(self.books[name].detach().cpu(), master['books'])
                    or not torch.equal(self.logs[name].detach().cpu(), master['log_scales'])):
                raise RuntimeError('Frozen codebook or row gain changed')
            for key, identity in self.static[name].items():
                value = self.encodings[name][key]
                if (value.data_ptr(), value._version) != identity: raise RuntimeError('Frozen VQ indices/exponents changed')
        return {'codebooks_unchanged': True, 'indices_unchanged': True, 'row_gains_unchanged': True,
                'continuous_scale_metadata_separate': True, 'original_exponents_unchanged': True}
