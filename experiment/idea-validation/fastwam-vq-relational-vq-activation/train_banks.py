"""Train fixed C8 microscale-VQ banks through the deployed BF16/A4 path."""
from pathlib import Path

import torch
import batched_transform
import micro_codebook as micro


BACKWARD_IMPLEMENTATION = 'bounded_vector_chunks_v2'
BACKWARD_MAX_VECTORS = 1 << 20


class A4Input(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, transform):
        ctx.transform, ctx.dtype = transform, x.dtype
        z = batched_transform.transform_input(x, transform).float()
        scale = z.abs().amax(dim=-1, keepdim=True) / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(z / safe).clamp(-7, 7) * scale).to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        transform = ctx.transform
        grad = batched_transform.inverse_right_r(
            grad.float(), transform['signs'], transform['hadamard'], transform['block'])
        return (grad / transform['scale']).to(ctx.dtype), None


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
        # Bound temporary tensors while batching enough vectors to amortize CUDA launches.
        chunk_rows = max(1, BACKWARD_MAX_VECTORS // indices.shape[1])
        for start in range(0, out_width, chunk_rows):
            end = min(start + chunk_rows, out_width)
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
    """Install one directly supplied C8 bank directory onto the 614 target Linears."""

    def __init__(self, runtime, bank_directory):
        self.modules = runtime.q.modules
        if len(self.modules) != 614:
            raise RuntimeError('Expected all 614 target Linears')
        self.encodings, self.transforms, self.books, self.logs, self.active = {}, {}, {}, {}, {}
        self.forwards = {name: mod.forward for name, mod in self.modules.items()}
        self.seen = set()
        bank_directory = Path(bank_directory)
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)
        runtime.q.disable()
        for index, (name, mod) in enumerate(self.modules.items()):
            path = bank_directory / f'{index:04d}.pt'
            bank = torch.load(path, map_location='cuda', weights_only=True)
            if (bank.get('protocol') != 'fastwam-vq-calibration-suites-v1'
                    or bank.get('cohort') != 'c8' or bank.get('module_index') != index
                    or bank.get('module') != name
                    or tuple(bank['encoded']['shape']) != tuple(mod.weight.shape)):
                raise RuntimeError(f'C8 bank identity differs at {name}')
            encoding = bank['encoded']
            micro.check_encoding(encoding)
            self.encodings[name], self.transforms[name] = encoding, bank['transform']
            self.books[name] = torch.nn.Parameter(encoding['books'].float().clone())
            scales = encoding['row_scales'].float()
            self.active[name] = (scales > 0).float()
            self.logs[name] = torch.nn.Parameter(
                scales.clamp_min(torch.finfo(torch.float32).tiny).log())
            mod.weight.data = micro.decode_vq(encoding)

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                qx = A4Input.apply(x, self.transforms[_name])
                weight = DecodedWeight.apply(
                    self.books[_name], self.logs[_name], self.active[_name],
                    self.encodings[_name]['indices'], self.encodings[_name]['scale_exponents'],
                    _mod.weight)
                return torch.nn.functional.linear(qx, weight, _mod.bias)

            mod.forward = forward
            if index % 100 == 0:
                print(f'C8_TRAIN_BANKS_LOADED {index}/614', flush=True)
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
            if (not bool(torch.isfinite(encoding['books']).all())
                    or not bool(torch.isfinite(encoding['row_scales']).all())):
                raise RuntimeError(f'Nonfinite stored bank at {name}')
            fixed = self.encodings[name]
            if (not torch.equal(encoding['indices'], fixed['indices'])
                    or not torch.equal(encoding['scale_exponents'], fixed['scale_exponents'])):
                raise RuntimeError(f'Fixed C8 indices or exponents changed at {name}')
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
            gradients = [self.books[name].grad, self.logs[name].grad]
            present = [grad for grad in gradients if grad is not None]
            if present:
                connected.append(name)
                if all(bool(torch.isfinite(grad).all()) for grad in present):
                    finite.append(name)
                if any(bool(grad.abs().sum() > 0) for grad in present):
                    nonzero.append(name)
        return {'targets': len(self.modules), 'connected': len(connected), 'finite': len(finite),
                'nonzero': len(nonzero), 'disconnected_modules': sorted(set(self.modules) - set(connected)),
                'zero_gradient_modules': sorted(set(connected) - set(nonzero))}
