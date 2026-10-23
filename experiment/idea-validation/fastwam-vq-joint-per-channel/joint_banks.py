"""Joint VQ masters plus the completed per-input-channel activation surrogate."""
import torch

import batched_transform
import train_banks
from clip_banks import ClipBanks
from channel_banks import (ChannelBanks, channel_scales, compensated_weight,
                           coupled_surrogate, quantized_input)


class JointChannelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ratio, log2_scales, weight, bias, transform):
        z = batched_transform.transform_input(x, transform).float()
        d = channel_scales(log2_scales, z.shape[-1])
        v = z / d
        maximum = v.abs().amax(-1, keepdim=True)
        scale = maximum * ratio / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        u = v / safe
        q = u.round().clamp(-7, 7)
        operand = (q * scale).to(torch.bfloat16)
        result = torch.nn.functional.linear(operand, compensated_weight(weight, log2_scales), bias)
        if any(ctx.needs_input_grad[:4]):
            inside = v.abs() <= maximum * ratio
            derivative = torch.where(inside, q - u, q).to(torch.bfloat16)
            maxima = v.abs() == maximum
            max_sign = torch.where(maxima, v.sign(), torch.zeros_like(v)).to(torch.int8)
            tie_count = maxima.sum(-1, keepdim=True)
            # Weight gradients need the actual rounded BF16 operand, not the STE residual.
            saved_operand = operand if ctx.needs_input_grad[3] else operand.new_empty(0)
            ctx.save_for_backward(inside, derivative, maximum / 7, d, ratio,
                                  log2_scales, weight, max_sign, tie_count, saved_operand)
        ctx.transform, ctx.dtype = transform, x.dtype
        return result

    @staticmethod
    def backward(ctx, grad):
        (inside, derivative, base_scale, d, ratio, log2_scales, weight,
         max_sign, tie_count, operand) = ctx.saved_tensors
        x_grad = ratio_grad = scale_grad = weight_grad = None
        if any(ctx.needs_input_grad[:3]):
            g = torch.nn.functional.linear(grad, compensated_weight(weight, log2_scales).t()).float()
            z_grad, ratio_grad, scale_grad = coupled_surrogate(
                g, inside, derivative, base_scale, d, ratio, max_sign, tie_count)
            x_grad = batched_transform.inverse_right_r(
                z_grad, ctx.transform['signs'], ctx.transform['hadamard'],
                ctx.transform['block']) / ctx.transform['scale']
            x_grad = x_grad.to(ctx.dtype)
        if ctx.needs_input_grad[3]:
            output_grad = grad.reshape(-1, grad.shape[-1])
            quantized = operand.reshape(-1, operand.shape[-1])
            compensated_grad = output_grad.t().matmul(quantized)
            # Adjoint of BF16(weight.float() * D), returned in the unscaled Wq coordinates.
            weight_grad = (compensated_grad.float() * d).to(weight.dtype)
        return x_grad, ratio_grad, scale_grad, weight_grad, None, None


class JointChannelBanks(ChannelBanks):
    def __init__(self, runtime, bank_directory, masters, clipping, scaling):
        super().__init__(runtime, bank_directory, masters, clipping, scaling)
        self.train_weights = True
        for name, mod in self.modules.items():
            self.books[name].requires_grad_(True)
            self.logs[name].requires_grad_(True)

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                if self.cached:
                    operand = quantized_input(x, self.ratios[_name], self.shadows[_name],
                                              self.transforms[_name])
                    return torch.nn.functional.linear(operand, _mod.weight, _mod.bias)
                decoded = train_banks.DecodedWeight.apply(
                    self.books[_name], self.logs[_name], self.active[_name],
                    self.encodings[_name]['indices'], self.encodings[_name]['scale_exponents'],
                    _mod.weight)
                return JointChannelLinear.apply(x, self.ratios[_name], self.shadows[_name],
                                                decoded, _mod.bias, self.transforms[_name])
            mod.forward = forward

    def optimizer_groups(self, training, scaling):
        return ClipBanks.optimizer_groups(self, training) + [
            {'params': list(self.shadows.values()), 'lr': scaling['learning_rate']}]

    @torch.no_grad()
    def cache_compensated_weights(self):
        for parameter in list(self.books.values()) + list(self.logs.values()):
            parameter.requires_grad_(False)
        result = super().cache_compensated_weights()
        result['scope'] = ('Compensated BF16 weight restored to original coordinates versus final '
                           'unscaled joint decoded Wq; not total full-precision weight-VQ error')
        return result

    def verify_joint(self, initial_masters):
        if set(initial_masters) != set(self.modules):
            raise RuntimeError('Initial master module coverage differs')
        changed_books = changed_gains = 0
        for name, initial in initial_masters.items():
            changed_books += not torch.equal(self.books[name].detach().cpu(), initial['books'])
            changed_gains += not torch.equal(self.logs[name].detach().cpu(), initial['log_scales'])
            for key, identity in self.static[name].items():
                value = self.encodings[name][key]
                if (value.data_ptr(), value._version) != identity:
                    raise RuntimeError('Frozen indices or original B32 exponents changed')
            if self.books[name].dtype != torch.float32 or self.logs[name].dtype != torch.float32:
                raise RuntimeError('Joint master parameters must remain FP32')
        if changed_books < 1 or changed_gains < 1:
            raise RuntimeError('Joint training must change both VQ books and log row gains')
        return {'indices_unchanged': True, 'original_exponents_unchanged': True,
                'continuous_scale_metadata_separate': True, 'module_masters_fp32': True,
                'codebooks_optimized': True, 'row_gains_optimized': True,
                'codebook_changed_modules': changed_books, 'row_gain_changed_modules': changed_gains}
