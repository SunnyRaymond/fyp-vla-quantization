"""Frozen base model, sequential R128 learning, and fixed-index VQ calibration."""
from __future__ import annotations
from pathlib import Path
from common import (torch_module, cpu_tree, load_tensors, save_tensors, save_json,
                    target_manifest, restore_bf16, has_time)
from runtime import right_u, transform_weight


def rotation_orthogonality(matrix):
    """Check the stored matrix in FP64, independent of inference matmul precision."""
    torch = torch_module()
    with torch.no_grad(), torch.autocast(device_type=matrix.device.type, enabled=False):
        precise = matrix.to(torch.float64)
        identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=torch.float64)
        return float((precise.t() @ precise - identity).abs().max())


def a4_ste(value):
    torch = torch_module()
    class A4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            scale = x.float().abs().amax(-1, keepdim=True) / 7
            safe = torch.where(scale > 0, scale, torch.ones_like(scale))
            return (torch.round(x.float() / safe).clamp(-7, 7) * scale).to(torch.bfloat16)
        @staticmethod
        def backward(ctx, gradient):
            return gradient.float()
    return A4.apply(value)


def get_transforms(runtime, root, out):
    torch = torch_module()
    import smooth_vq
    path = Path(out) / 'state/transforms.pt'
    if path.exists():
        saved = load_tensors(path)
    else:
        statistics = load_tensors(Path(root) / 'results/teacher/statistics.pt')
        saved = {}
        for index, (name, module) in enumerate(runtime.target_linears.items()):
            saved[name] = cpu_tree(smooth_vq.make_transform(statistics[name]['absmax'].to('cuda'),
                module.weight, alpha=.5, hadamard=True, block=128, seed=20261024 + index))
        save_tensors(path, saved)
    return {name: {key: value.to('cuda') if torch.is_tensor(value) else value
                   for key, value in transform.items()} for name, transform in saved.items()}


class RotationState:
    def __init__(self, runtime, transforms, raw=None):
        torch = torch_module()
        self.runtime, self.transforms = runtime, transforms
        self.raw = torch.nn.Parameter(torch.zeros(128, 128, device='cuda', dtype=torch.float32))
        if raw is not None:
            self.raw.data.copy_(raw.to('cuda'))
        self._matrix = None

    def matrix(self):
        torch = torch_module()
        if self._matrix is None:
            skew = self.raw - self.raw.t()
            identity = torch.eye(128, device=self.raw.device, dtype=torch.float32)
            self._matrix = torch.linalg.solve(identity - .5 * skew, identity + .5 * skew)
        return self._matrix

    def prewarm(self):
        # Build Cayley outside all count-sensitive official block SAC regions.
        self._matrix = None
        self.matrix()

    def install(self):
        import batched_transform
        torch = torch_module()
        for name, module in self.runtime.target_linears.items():
            def forward(value, _name=name, _module=module):
                matrix = self.matrix()
                transformed = right_u(batched_transform.transform_input(value, self.transforms[_name]), matrix)
                activation = a4_ste(transformed)
                weight = transform_weight(_module.weight, self.transforms[_name], matrix).to(torch.bfloat16)
                with torch.autocast(device_type=activation.device.type, enabled=False):
                    return torch.nn.functional.linear(activation, weight, _module.bias)
            module.forward = forward

    def freeze(self):
        self.raw.requires_grad_(False)
        self.prewarm()
        return self.matrix().detach()


class VQState:
    def __init__(self, runtime, directory, transforms, matrix):
        torch = torch_module()
        self.runtime, self.transforms, self.matrix = runtime, transforms, matrix.detach()
        self.directory = Path(directory)
        self.encodings, self.books, self.logs, self.active = {}, {}, {}, {}
        for index, name in enumerate(runtime.target_linears):
            payload = load_tensors(self.directory / f'{index:04d}.pt')
            if payload['name'] != name:
                raise RuntimeError('VQ initialization module order differs')
            encoding = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in payload['encoding'].items()}
            self.encodings[name] = encoding
            self.books[name] = torch.nn.Parameter(encoding['books'].float().clone())
            gains = encoding['row_scales'].float()
            self.logs[name] = torch.nn.Parameter(gains.clamp_min(torch.finfo(torch.float32).tiny).log())
            self.active[name] = (gains > 0).float()

    def parameters(self):
        return [value for name in self.books for value in (self.books[name], self.logs[name])]

    def encoding(self, name):
        torch = torch_module()
        result = dict(self.encodings[name])
        result['books'] = self.books[name].detach().to(torch.bfloat16)
        result['row_scales'] = (self.logs[name].detach().exp() * self.active[name]).to(torch.bfloat16)
        return result

    def refresh(self):
        import micro_codebook
        torch = torch_module()
        with torch.no_grad():
            for name, module in self.runtime.target_linears.items():
                module.weight.data.copy_(micro_codebook.decode_vq(self.encoding(name)).to(torch.bfloat16))

    def install(self):
        import batched_transform
        from train_banks import DecodedWeight
        torch = torch_module()
        self.refresh()
        for name, module in self.runtime.target_linears.items():
            def forward(value, _name=name, _module=module):
                transformed = right_u(batched_transform.transform_input(value, self.transforms[_name]), self.matrix)
                activation = a4_ste(transformed)
                encoded = self.encodings[_name]
                weight = DecodedWeight.apply(self.books[_name], self.logs[_name], self.active[_name],
                    encoded['indices'], encoded['scale_exponents'], _module.weight)
                with torch.autocast(device_type=activation.device.type, enabled=False):
                    return torch.nn.functional.linear(activation, weight, _module.bias)
            module.forward = forward

    def snapshot(self):
        return {name: {'books': cpu_tree(self.books[name]), 'logs': cpu_tree(self.logs[name])}
                for name in self.books}

    def restore(self, saved):
        for name in self.books:
            self.books[name].data.copy_(saved[name]['books'].to('cuda'))
            self.logs[name].data.copy_(saved[name]['logs'].to('cuda'))
        self.refresh()


def prepare_vq(runtime, p, root, out, transforms, matrix, contexts, soft_deadline):
    """Fit in the final R0 coordinates, with per-module resumable banks."""
    torch = torch_module()
    import micro_codebook
    import batched_transform
    from common import endpoint
    out = Path(out)
    moment_path = out / 'state/rotation_moments.pt'
    if not moment_path.exists():
        partial_path = out / 'state/rotation_moments_partial.pt'
        partial = load_tensors(partial_path) if partial_path.exists() else {'completed': [], 'moments': {}}
        done, sums = set(partial['completed']), partial['moments']
        sums = {name: {'sum_sq': row['sum_sq'].to('cuda'), 'count': row['count']} for name, row in sums.items()}
        original = {name: module.forward for name, module in runtime.target_linears.items()}
        for name, module in runtime.target_linears.items():
            def forward(value, _name=name, _module=module):
                operand = a4_ste(right_u(batched_transform.transform_input(value, transforms[_name]), matrix))
                rows = operand.detach().float().reshape(-1, _module.in_features)
                item = sums.setdefault(_name, {'sum_sq': torch.zeros(_module.in_features, device='cuda'), 'count': 0})
                item['sum_sq'].add_(rows.square().sum(0))
                item['count'] += rows.shape[0]
                weight = transform_weight(_module.weight, transforms[_name], matrix).to(torch.bfloat16)
                return torch.nn.functional.linear(operand, weight, _module.bias)
            module.forward = forward
        try:
            for case, context in contexts.items():
                if case in done:
                    continue
                if not has_time(soft_deadline, 120):
                    break
                endpoint(runtime, context)
                done.add(case)
                if len(done) % 4 == 0:
                    save_tensors(partial_path, {'completed': sorted(done), 'moments': cpu_tree(sums)})
        finally:
            for name, module in runtime.target_linears.items():
                module.forward = original[name]
        save_tensors(partial_path, {'completed': sorted(done), 'moments': cpu_tree(sums)})
        if done != set(contexts):
            return False
        save_tensors(moment_path, {name: cpu_tree(row['sum_sq'] / row['count']) for name, row in sums.items()})
    moments = load_tensors(moment_path)
    bank_directory = out / 'initial_banks'
    bank_directory.mkdir(parents=True, exist_ok=True)
    for index, (name, module) in enumerate(runtime.target_linears.items()):
        destination = bank_directory / f'{index:04d}.pt'
        if destination.exists():
            continue
        if not has_time(soft_deadline, 90):
            return False
        weight = transform_weight(module.weight.detach(), transforms[name], matrix)
        encoding = micro_codebook.fit_vq(weight, second_moment=moments[name].to('cuda'),
            seed=20261024 + index, sample_limit=8192, iterations=8)
        micro_codebook.check_encoding(encoding)
        save_tensors(destination, {'name': name, 'encoding': cpu_tree(encoding)})
        save_json(out / 'progress.json', {'phase': 'vq_fit', 'completed_modules': index + 1,
                                         'expected_modules': len(runtime.target_linears)})
        print(f'OURS_VQ_FIT {index + 1}/{len(runtime.target_linears)}', flush=True)
    return True


def deploy(runtime, p, state, out):
    import os
    torch = torch_module()
    destination = Path(out) / 'deployment'
    entries, target_bytes = [], 128 * 128 * 4
    for index, (name, module) in enumerate(runtime.target_linears.items()):
        encoding, transform = cpu_tree(state.encoding(name)), cpu_tree(state.transforms[name])
        filename = f'bank/{index:04d}.pt'
        save_tensors(destination / filename, {'encoding': encoding, 'transform': transform})
        target_bytes += sum(v.numel() * v.element_size() for v in encoding.values() if torch.is_tensor(v))
        target_bytes += sum(v.numel() * v.element_size() for v in transform.values() if torch.is_tensor(v))
        entries.append({'name': name, 'shape': [module.out_features, module.in_features], 'file': filename})
    save_tensors(destination / 'rotation.pt', {'matrix': cpu_tree(state.matrix.float()),
                 'stream': 'cosmos', 'parameterization': 'H128_then_U'})
    target_numel = sum(module.weight.numel() for module in runtime.target_linears.values())
    bpw = 8 * target_bytes / target_numel
    target = float(p['quantization']['target_effective_bpw'])
    if abs(bpw - target) / target > p['quantization']['effective_bpw_max_relative_difference']:
        raise RuntimeError('Trained VQ artifact exceeds the frozen effective-bpw budget')
    from runtime import weight_storage
    storage = weight_storage(runtime, target_bytes)
    whole_bpw = 8 * storage['packed_storage_bytes'] / storage['whole_dit_parameter_numel']
    manifest = {'format': 'cosmos_vqwam_deployment_v1', 'protocol': p['protocol'],
        'source_job': os.environ['PBS_JOBID'], 'completed_updates': 8,
        'target_modules': target_manifest(runtime), 'modules': entries,
        'rotation_file': 'rotation.pt', 'reference_execution': 'decoded_bf16_fake_a4',
        'effective_bpw': bpw, 'whole_dit_effective_bpw': whole_bpw,
        'target_payload_bytes': target_bytes, 'storage': storage}
    save_json(destination / 'manifest.json', manifest)
    return manifest
