"""Explicit differentiable decoded-weight transport in the saved R0 coordinates."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from rotation_bank import A4, VQBanks, stream_for, transformed
from c8_train_banks import DecodedWeight


def right_blocks(value, matrix):
    full = value.shape[-1] // 128 * 128
    if not full:
        return value
    with torch.autocast(device_type='cuda', enabled=False):
        head = value[..., :full].float().reshape(*value.shape[:-1], full // 128, 128)
        head = (head @ matrix).reshape(*value.shape[:-1], full)
        return torch.cat((head, value[..., full:].float()), -1) if full < value.shape[-1] else head


def relative_input_inverse(current, anchor):
    # Centering makes the saved start exactly I despite the FP32 orthogonality residual.
    eye = torch.eye(128, dtype=current.dtype, device=current.device)
    return eye + torch.linalg.solve(current, anchor - current)


class TransportBanks(VQBanks):
    def __init__(self, runtime, directory, rotations, transforms, source_protocol):
        self.anchor = {}
        self.relative_cache, self.dense_cache = {}, {}
        self.compensated, self.activation_bits = False, 4
        self.cache_evaluation_weights = False
        super().__init__(runtime, directory, rotations, transforms, source_protocol)
        self.legacy = {name: mod.forward for name, mod in self.modules.items()}
        self.set_anchor(rotations.snapshot())
        for name, mod in self.modules.items():
            mod.forward = self.make_forward(name, mod)

    def refresh(self):
        # Release the previous endpoint's dense copy before decoding replacement weights.
        self.relative_cache.clear()
        self.dense_cache.clear()
        super().refresh()

    def set_anchor(self, raw):
        self.rotations.restore(raw)
        self.anchor = {s: self.rotations.matrix(s).detach().clone() for s in self.rotations.raw}
        self.relative_cache.clear()
        self.dense_cache.clear()

    def set_target(self, raw, *, compensated, activation_bits=4):
        if activation_bits not in (4, 16):
            raise ValueError('Only A4 or noA4 BF16 operands are supported')
        self.rotations.restore(raw)
        self.compensated, self.activation_bits = compensated, activation_bits
        self.relative_cache.clear()
        self.dense_cache.clear()

    def relative(self, stream):
        if stream not in self.relative_cache:
            self.relative_cache[stream] = relative_input_inverse(self.rotations.matrix(stream), self.anchor[stream])
        return self.relative_cache[stream]

    def query_reset(self):
        self.relative_cache.clear()

    def load_deployment(self, payload, expected_protocol):
        if (payload.get('protocol') != expected_protocol or payload.get('target_linears') != 614
                or payload.get('execution') != 'explicit_post_decode_BF16_weight_transport'
                or payload.get('updates') != 8 or payload.get('exposures') != 80
                or set(payload.get('module_masters', {})) != set(self.modules)):
            raise RuntimeError('Transport deployment binding differs')
        self.restore(payload['module_masters'])
        self.set_anchor(payload['anchor_rotations'])
        self.set_target(payload['rotations'], compensated=payload['compensated'], activation_bits=4)

    @torch.no_grad()
    def prepare_evaluation(self):
        self.dense_cache.clear()
        if self.compensated and self.cache_evaluation_weights:
            for name, mod in self.modules.items():
                if mod.in_features >= 128:
                    self.dense_cache[name] = right_blocks(mod.weight, self.relative(stream_for(name)).t()).to(torch.bfloat16)

    def make_forward(self, name, mod):
        stream = stream_for(name)

        def forward(x):
            # This branch is the archived uncompensated training graph, including its arithmetic.
            if not self.compensated and self.activation_bits == 4:
                return self.legacy[name](x)
            self.seen.add(name)
            matrix = self.rotations.matrix(stream) if mod.in_features >= 128 else x.new_empty(0)
            relative = self.relative(stream) if self.compensated and mod.in_features >= 128 else x.new_empty(0)

            def run(value, current, input_inverse, books, logs):
                # transformed() uses the same cached current matrix; explicit args preserve checkpoint dependencies.
                z = transformed(value, self.transforms[name], self.rotations, stream)
                qx = A4.apply(z) if self.activation_bits == 4 else z.to(torch.bfloat16)
                weight = DecodedWeight.apply(books, logs, self.active[name], self.encodings[name]['indices'],
                                             self.encodings[name]['scale_exponents'], mod.weight)
                if self.compensated and mod.in_features >= 128:
                    weight = self.dense_cache.get(name) if not torch.is_grad_enabled() and name in self.dense_cache else \
                        right_blocks(weight, input_inverse.t()).to(torch.bfloat16)
                return F.linear(qx, weight, mod.bias)

            if torch.is_grad_enabled() and self.compensated:
                return checkpoint(run, x, matrix, relative, self.books[name], self.logs[name], use_reentrant=False)
            return run(x, matrix, relative, self.books[name], self.logs[name])
        return forward
