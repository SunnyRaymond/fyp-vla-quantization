"""Fixed-index codebook training with deployed BF16 forward values."""
from pathlib import Path

import torch
import codebook
import micro_codebook as micro
import smooth_vq as core


class A4Input(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, transform):
        ctx.transform, ctx.dtype = transform, x.dtype
        z = core.transform_input(x, transform).float()
        scale = z.abs().amax(dim=-1, keepdim=True) / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(z / safe).clamp(-7, 7) * scale).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        t = ctx.transform
        # STE for A4; the fixed Smooth/Hadamard change of coordinates stays exact.
        g = core._right_r_inverse(grad.float(), t['signs'], t['hadamard'], t['block'])
        return (g / t['scale']).to(ctx.dtype), None


class DecodedWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, books, log_scales, active, indices, exponents, dense):
        ctx.save_for_backward(books, log_scales, active, indices, exponents)
        ctx.shape = tuple(dense.shape)
        return dense.detach().view_as(dense)

    @staticmethod
    def backward(ctx, grad):
        books, log_scales, active, indices, exponents = ctx.saved_tensors
        out_width, in_width = ctx.shape
        stored_books = books.to(torch.bfloat16).float()
        raw_gains = log_scales.exp() * active
        stored_gains = raw_gains.to(torch.bfloat16).float()
        book_grad = torch.zeros_like(books)
        scale_grad = torch.zeros_like(log_scales)
        padded = indices.shape[1] * 4
        group_index = torch.arange(indices.shape[1], device=indices.device) // 8
        # Bound temporary memory exactly as the previous row-scale trainer did.
        # Each micro exponent is applied only to its 32-row chunk (8 scale blocks).
        for start in range(0, out_width, 32):
            end = min(start + 32, out_width)
            exponent = exponents[start:end, group_index].to(torch.int32) - 127
            stored_scales = torch.ldexp(stored_gains[start:end, None], exponent)
            raw_scales = torch.ldexp(raw_gains[start:end, None], exponent)
            g = torch.nn.functional.pad(grad[start:end].float(), (0, padded - in_width))
            g = g.reshape(end - start, -1, 4)
            idx = indices[start:end].long()
            values = stored_books[0][idx[..., 0]] + stored_books[1][idx[..., 1]]
            scale_grad[start:end] = (g * values * raw_scales[..., None]).sum((1, 2))
            weighted = (g * stored_scales[..., None]).reshape(-1, 4)
            for book in range(2):
                book_grad[book].index_add_(0, idx[..., book].reshape(-1), weighted)
        return book_grad, scale_grad, None, None, None, None


class TrainableBanks:
    def __init__(self, runtime, bank_root, arm):
        self.modules = runtime.q.modules
        if len(self.modules) != 614:
            raise RuntimeError('Expected all 614 target Linears')
        self.encodings, self.transforms, self.books, self.logs, self.active = {}, {}, {}, {}, {}
        self.forwards = {name: mod.forward for name, mod in self.modules.items()}
        self.seen = set()
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)
        runtime.q.disable()
        for index, (name, mod) in enumerate(self.modules.items()):
            path = Path(bank_root) / 'results' / 'full' / 'banks' / arm / f'{index:04d}.pt'
            bank = torch.load(path, map_location='cuda', weights_only=True)
            if bank['module'] != name or tuple(bank['encoded']['shape']) != tuple(mod.weight.shape):
                raise RuntimeError(f'Bank identity differs at {name}')
            encoding = bank['encoded']
            micro.check_encoding(encoding)
            self.encodings[name], self.transforms[name] = encoding, bank['transform']
            self.books[name] = torch.nn.Parameter(encoding['books'].float().clone())
            scales = encoding['row_scales'].float()
            self.active[name] = (scales > 0).float()
            self.logs[name] = torch.nn.Parameter(scales.clamp_min(torch.finfo(torch.float32).tiny).log())
            mod.weight.data = micro.decode_vq(encoding)

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                qx = A4Input.apply(x, self.transforms[_name])
                weight = DecodedWeight.apply(self.books[_name], self.logs[_name], self.active[_name],
                                             self.encodings[_name]['indices'],
                                             self.encodings[_name]['scale_exponents'], _mod.weight)
                return torch.nn.functional.linear(qx, weight, _mod.bias)

            mod.forward = forward
            if index % 100 == 0:
                print(f'DISTILL_BANKS_LOADED {index}/614', flush=True)
        self.refresh()

    def parameters(self):
        return [value for name in self.modules for value in (self.books[name], self.logs[name])]

    def stored_encoding(self, name):
        value = dict(self.encodings[name])
        value['books'] = self.books[name].detach().to(torch.bfloat16)
        value['row_scales'] = (self.logs[name].detach().exp() * self.active[name]).to(torch.bfloat16)
        return value

    @torch.no_grad()
    def refresh(self):
        for name, mod in self.modules.items():
            encoding = self.stored_encoding(name)
            if not torch.isfinite(encoding['books']).all() or not torch.isfinite(encoding['row_scales']).all():
                raise RuntimeError(f'Nonfinite stored bank at {name}')
            if not torch.equal(encoding['indices'], self.encodings[name]['indices']) or not torch.equal(
                    encoding['scale_exponents'], self.encodings[name]['scale_exponents']):
                raise RuntimeError(f'Fixed micro indices or exponents changed at {name}')
            mod.weight.data = micro.decode_vq(encoding)

    def snapshot(self):
        return {name: {'books': self.books[name].detach().cpu().clone(),
                       'log_scales': self.logs[name].detach().cpu().clone()}
                for name in self.modules}

    @torch.no_grad()
    def restore(self, state):
        if set(state) != set(self.modules):
            raise RuntimeError('Checkpoint target coverage differs')
        for name in self.modules:
            self.books[name].copy_(state[name]['books'])
            self.logs[name].copy_(state[name]['log_scales'])
        self.refresh()

    def gradient_report(self):
        finite, connected, nonzero = [], [], []
        for name in self.modules:
            gradients = [p.grad for p in (self.books[name], self.logs[name])]
            present = [g for g in gradients if g is not None]
            if present:
                connected.append(name)
                if all(bool(torch.isfinite(g).all()) for g in present):
                    finite.append(name)
                if any(bool(g.abs().sum() > 0) for g in present):
                    nonzero.append(name)
        return {'targets': len(self.modules), 'connected': len(connected), 'finite': len(finite),
                'nonzero': len(nonzero), 'disconnected_modules': sorted(set(self.modules) - set(connected)),
                'zero_gradient_modules': sorted(set(connected) - set(nonzero))}


def self_check():
    """CPU/PBS parity and analytic micro STE-gradient check, including exponent blocks and a tail."""
    generator = torch.Generator().manual_seed(20261011)
    width = 37
    encoding = {'format': micro.FORMAT, 'block_size': 4, 'scale_block': 32, 'shape': (3, width),
                'books': torch.randn(2, 256, 4, generator=generator).to(torch.bfloat16),
                'row_scales': torch.tensor([0.25, 0.5, 0.0], dtype=torch.bfloat16),
                'scale_exponents': torch.tensor([[127, 127], [127, 128], [127, 127]], dtype=torch.uint8),
                'indices': torch.randint(0, 256, (3, 10, 2), generator=generator, dtype=torch.uint8)}
    micro.check_encoding(encoding)
    old_encoding = {'format': codebook.FORMAT, 'block_size': 4, 'shape': encoding['shape'],
                    'books': encoding['books'], 'row_scales': encoding['row_scales'],
                    'indices': encoding['indices']}
    e0 = dict(encoding, scale_exponents=torch.full_like(encoding['scale_exponents'], 127))
    assert torch.equal(micro.decode_vq(e0), codebook.decode_vq(old_encoding)), 'e0 parity must match row-scale decode'
    decoded = micro.decode_vq(encoding)
    old_dense = codebook.decode_vq(old_encoding)
    assert torch.equal(decoded[1, :32], old_dense[1, :32])
    assert torch.equal(decoded[1, 32:], (old_dense[1, 32:].float() * 2).to(torch.bfloat16)), 'nonzero exponent and tail'
    books = encoding['books'].float().requires_grad_()
    scales = encoding['row_scales'].float()
    active = (scales > 0).float()
    logs = scales.clamp_min(torch.finfo(torch.float32).tiny).log().requires_grad_()
    dense = micro.decode_vq(encoding)
    decoded = DecodedWeight.apply(books, logs, active, encoding['indices'], encoding['scale_exponents'], dense)
    assert torch.equal(decoded, dense)
    g = torch.randn(3, width, generator=generator).to(torch.bfloat16)
    (decoded * g).sum().backward()
    actual = (books.grad.clone(), logs.grad.clone())
    books.grad, logs.grad = None, None
    group_index = torch.arange(encoding['indices'].shape[1]) // 8
    exponents = encoding['scale_exponents'][:, group_index].to(torch.int32) - 127
    scale_factor = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)

    def ste_bf16(values):
        stored = values.to(torch.bfloat16).float()
        return values + (stored - values).detach()

    stored_books = ste_bf16(books)
    raw_gains = logs.exp() * active
    stored_gains = ste_bf16(raw_gains)
    sums = stored_books[0][encoding['indices'][..., 0].long()]
    sums = sums + stored_books[1][encoding['indices'][..., 1].long()]
    dense = (sums * stored_gains[:, None, None] * scale_factor[..., None]).reshape(3, -1)[:, :width]
    reference = ste_bf16(dense)
    assert torch.equal(reference, decoded.detach().float()), 'STE reference must preserve deployed micro BF16 values'
    (reference * g.float()).sum().backward()
    assert torch.allclose(actual[0], books.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual[1], logs.grad, atol=1e-6, rtol=1e-5)
    assert bool((books.grad.abs().sum((0, 2)) > 0).any()), 'micro books must receive finite gradients'
    assert bool(torch.isfinite(logs.grad).all()) and bool((logs.grad[:2].abs() > 0).all())
    x = torch.randn(2, 8, generator=generator).requires_grad_()
    transform = {'scale': torch.arange(1, 9).float(), 'signs': torch.tensor([1., -1.] * 4),
                 'hadamard': True, 'block': 8, 'input_dim': 8}
    y = A4Input.apply(x, transform)
    z = core.transform_input(x.detach(), transform)
    scale = z.abs().amax(-1, keepdim=True) / 7
    assert torch.equal(y, (torch.round(z / scale).clamp(-7, 7) * scale).to(torch.bfloat16))
    upstream = torch.randn(2, 8, generator=generator).to(torch.bfloat16)
    (y * upstream).sum().backward()
    actual_x = x.grad.clone()
    x.grad = None
    (core.transform_input(x, transform) * upstream.float()).sum().backward()
    assert torch.allclose(actual_x, x.grad, atol=1e-6, rtol=1e-5)
    print('MICRO_TRAINING_STE_SELF_CHECK_PASSED', flush=True)
