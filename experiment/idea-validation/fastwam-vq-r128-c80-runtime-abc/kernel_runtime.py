"""Inference-only integration; all construction and tensor I/O require PBS."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gc
from pathlib import Path


@contextmanager
def empty_factory(torch):
    """Keep parameters meta at registration, preserving real nonpersistent buffers."""
    from fastwam.models.wan22.helpers import loader
    register, to, load = torch.nn.Module.register_parameter, torch.nn.Module.to, loader._load_registered_model

    def register_meta(module, name, parameter):
        if parameter is not None and parameter.device.type != 'meta':
            parameter = torch.nn.Parameter(parameter.to('meta'), requires_grad=parameter.requires_grad)
        register(module, name, parameter)

    def defer_to(module, *args, **kwargs):
        if any(p.is_meta for p in module.parameters()):
            device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
            for child in module.modules():
                for name, value in child._buffers.items():
                    if value is not None:
                        cast = dtype if value.is_floating_point() or value.is_complex() else None
                        child._buffers[name] = value.to(device=device, dtype=cast)
            return module
        return to(module, *args, **kwargs)

    def architecture(path, model_name, torch_dtype, device, model_kwargs_override=None):
        choices = [r for r in loader.WAN22_MODEL_REGISTRY if r['model_name'] == model_name]
        if len(choices) != 1:
            raise ValueError('Ambiguous exported architecture: ' + model_name)
        row = choices[0]
        kwargs = dict(row.get('extra_kwargs', {}))
        kwargs.update(model_kwargs_override or {})
        return row['model_class'](**kwargs)

    torch.nn.Module.register_parameter, torch.nn.Module.to = register_meta, defer_to
    loader._load_registered_model = architecture
    try:
        yield
    finally:
        torch.nn.Module.register_parameter, torch.nn.Module.to = register, to
        loader._load_registered_model = load


def shared_gpu_state(state, torch):
    """Copy each aliased backing storage once; no dense target weight is present."""
    memo, result = {}, {}
    for name, tensor in state.items():
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in memo:
            view = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
            memo[key] = view.to('cuda')
        typed = memo[key].view(tensor.dtype)
        result[name] = torch.as_strided(typed, tensor.shape, tensor.stride(), tensor.storage_offset())
    return result


def _save_a4_mismatch_diagnostic(out, name, z, actual, expected):
    import json
    import torch

    width = z.shape[-1]
    actual_rows = actual.detach().contiguous().view(torch.int16).reshape(-1, width)
    expected_rows = expected.detach().contiguous().view(torch.int16).reshape(-1, width)
    mismatch = actual_rows != expected_rows
    mismatch_count = int(mismatch.sum().item())
    actual_float = actual.float()
    expected_float = expected.float()
    numeric_count = int((actual_float != expected_float).sum().item())
    actual_float_rows = actual_float.reshape(-1, width)
    expected_float_rows = expected_float.reshape(-1, width)
    zero_sign_count = int((mismatch & (actual_float_rows == 0) & (expected_float_rows == 0)).sum().item())
    mismatch_rows = mismatch.any(dim=1)
    row = int(torch.nonzero(mismatch_rows, as_tuple=False)[0, 0].item())
    cols = torch.nonzero(mismatch[row], as_tuple=False).flatten()[:32]

    z_rows = z.detach().reshape(-1, width)
    z_row = z_rows[row].float().contiguous()
    row_max = z_row.abs().amax()
    reference_scale = row_max / 7
    safe_scale = torch.where(reference_scale > 0, reference_scale, torch.ones_like(reference_scale))
    reference_scaled = z_row / safe_scale
    reference_rounded = torch.round(reference_scaled).clamp(-7, 7)
    abs_scaled = reference_scaled.abs()
    half_distance = ((abs_scaled - torch.floor(abs_scaled)) - 0.5).abs()

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "a4_mismatch_input.pt"
    torch.save({"layer": name, "row_index": row, "z": z_row.cpu()}, input_path)

    col_ids = cols.cpu().tolist()
    actual_bits = actual_rows[row, cols].cpu().tolist()
    expected_bits = expected_rows[row, cols].cpu().tolist()
    z_values = z_row[cols].cpu().tolist()
    scaled_values = reference_scaled[cols].cpu().tolist()
    rounded_values = reference_rounded[cols].cpu().tolist()
    distance_values = half_distance[cols].cpu().tolist()
    actual_values = actual.float().reshape(-1, width)[row, cols].cpu().tolist()
    expected_values = expected.float().reshape(-1, width)[row, cols].cpu().tolist()
    entries = [
        {
            "column": int(col),
            "z": float(value),
            "reference_scaled": float(scaled),
            "distance_to_half": float(distance),
            "reference_rounded": float(rounded),
            "fused_bf16_bits": f"0x{int(fused_bits) & 0xffff:04x}",
            "reference_bf16_bits": f"0x{reference_bits & 0xffff:04x}",
            "fused_value": float(fused_value),
            "reference_value": float(reference_value),
        }
        for col, value, scaled, distance, rounded, fused_bits, reference_bits,
        fused_value, reference_value in zip(
            col_ids, z_values, scaled_values, distance_values, rounded_values,
            actual_bits, expected_bits, actual_values, expected_values,
        )
    ]
    scale_mul = row_max * (1.0 / 7.0)
    report = {
        "layer": name,
        "input_shape": list(z.shape),
        "row_index": row,
        "feature_width": width,
        "mismatch_count": mismatch_count,
        "numeric_value_mismatch_count": numeric_count,
        "zero_sign_mismatch_count": zero_sign_count,
        "max_abs_difference": float((actual_float - expected_float).abs().max().item()),
        "row_max_abs": float(row_max.item()),
        "reference_scale_div7": float(reference_scale.item()),
        "reciprocal_multiply_scale": float(scale_mul.item()),
        "scale_bits_equal": bool(torch.equal(reference_scale.view(torch.int32), scale_mul.view(torch.int32))),
        "input_file": input_path.name,
        "first_mismatches": entries,
    }
    (out / "a4_mismatch.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return input_path


def make_runtime(out, protocol, bundle, direct=False):
    import torch
    import reference_profile as reference
    import export_runtime as export
    export.compute_guard()
    if not direct:
        runtime = reference.inference_runtime(out, protocol, bundle)
    else:
        # The exported state supplies all parameters. Cached tokenizer files remain required.
        old = bundle['assets']['resolved_config']['model'].get('skip_dit_load_from_pretrain')
        bundle['assets']['resolved_config']['model']['skip_dit_load_from_pretrain'] = True
        try:
            with empty_factory(torch):
                runtime = reference.inference_runtime(out, protocol, bundle)
        finally:
            bundle['assets']['resolved_config']['model']['skip_dit_load_from_pretrain'] = old
        runtime.direct_loader = True
    runtime.fused_diagnostic_dir = Path(out) / 'diagnostics'
    return runtime


class Banks:
    def parameters(self):
        return []

    def __init__(self, runtime, bundle, packed, fused):
        import torch
        import micro_codebook as micro
        import export_runtime as export
        import fused_vq
        self.modules, self.packed, self.fused = runtime.q.modules, packed, fused
        self.seen, self.calls, self.encodings, self.transforms = set(), Counter(), {}, {}
        self.prepared = {}
        self.diagnostic_dir = getattr(runtime, 'fused_diagnostic_dir', None)
        self.rotations = export.FixedRotations({k:v.to('cuda') for k,v in bundle['rotations'].items()})
        self.validate_input, self.validation = True, {}
        for name, module in self.modules.items():
            encoding = {k:v.to('cuda') if isinstance(v, torch.Tensor) else v
                        for k,v in bundle['quantized'][name].items() if k != 'module_index'}
            micro.check_encoding(encoding)
            self.transforms[name] = {k:v.to('cuda') if isinstance(v, torch.Tensor) else v
                                     for k,v in bundle['transforms'][name].items()}
            if packed:
                self.encodings[name] = encoding
                self.prepared[name] = fused_vq.prepare(encoding)
            else:
                module.weight = torch.nn.Parameter(micro.decode_vq(encoding), requires_grad=False)
            module.forward = self.forward(runtime, name, module)

    def operand(self, x, name):
        import torch
        import batched_transform
        import export_runtime as export
        import fused_input
        from rotation_bank import stream_for
        transform = self.transforms[name]
        z = fused_input.smooth_h(x, transform) if self.fused else batched_transform.transform_input(x, transform)
        if self.fused and self.validate_input and name not in self.validation:
            reference = batched_transform.transform_input(x, transform)
            error = float((z-reference).abs().max())
            if error != 0:
                raise ValueError(f'Fused Smooth/H differs at {name}: {error}')
        z = self.rotations.apply_after_h(z, stream_for(name))
        operand = fused_input.a4(z) if self.fused else export.a4(z)
        if self.fused and self.validate_input and name not in self.validation:
            expected = export.a4(z)
            if not torch.equal(operand.view(torch.int16), expected.view(torch.int16)):
                try:
                    path = _save_a4_mismatch_diagnostic(self.diagnostic_dir, name, z, operand, expected)
                    print(f'A4_MISMATCH_DIAGNOSTIC {path}', flush=True)
                except Exception as exc:
                    print(f'A4_MISMATCH_DIAGNOSTIC_FAILED {type(exc).__name__}: {exc}', flush=True)
                raise ValueError('Fused whole-row A4 differs at ' + name)
            self.validation[name] = dict(smooth_h_max_abs=0, a4_bitwise=True)
        return operand

    def forward(self, runtime, name, module):
        import torch
        import fused_vq
        @torch.no_grad()
        def run(x):
            self.seen.add(name)
            self.calls[name] += 1
            runtime.linear_calls += 1
            operand = self.operand(x, name)
            if self.packed:
                return fused_vq.prepared_linear(operand, self.prepared[name], module.bias)
            return torch.nn.functional.linear(operand, module.weight, module.bias)
        return run


def install(runtime, bundle, packed=False, fused=False, direct=False):
    import torch
    import export_runtime as export
    export.compute_guard()
    export._validate_header(bundle)
    if list(runtime.q.modules) != list(bundle['quantized']):
        raise ValueError('Quantized module order differs')
    if export._target_state_keys(runtime.model, runtime.q.modules) != bundle['excluded_model_state_keys']:
        raise ValueError('Target aliases differ')
    runtime.q.disable()
    for name, module in runtime.q.modules.items():
        if tuple(bundle['quantized'][name]['shape']) != (module.out_features, module.in_features):
            raise ValueError('Shape differs at ' + name)
        module.register_parameter('weight', None)
    if set(runtime.model.state_dict()) != set(bundle['model_state']):
        raise ValueError('Exported persistent state does not cover runtime')
    state = shared_gpu_state(bundle['model_state'], torch) if direct else bundle['model_state']
    runtime.model.load_state_dict(state, strict=True, assign=direct)
    del state
    runtime.model.to('cuda').eval()
    if any(t.is_meta for t in list(runtime.model.parameters()) + list(runtime.model.buffers())):
        raise ValueError('Unmaterialized parameter/buffer after direct load')
    runtime.processor.set_normalizer_from_stats(bundle['processor_stats'])
    if bundle['processor_state']:
        runtime.processor.load_state_dict(bundle['processor_state'], strict=True)
    runtime.processor.eval()
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.linear_calls = 0
    runtime.banks = Banks(runtime, bundle, packed, fused)
    runtime.current_arm = runtime.arm_name = 'reader_kl'
    runtime.install_metadata = dict(module_count=614, packed_gpu_resident=packed, native_vq_kernel=packed,
        native_integer_gemm=False, fused_input=fused, direct_loader=direct, new_training_updates=0,
        dense_target_weights=not packed, source_job=export.SOURCE_JOB)
    gc.collect()
    torch.cuda.empty_cache()
    return runtime.banks
