"""Inference-only integration; all construction and tensor I/O require PBS."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gc


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


def make_runtime(out, protocol, bundle, direct=False):
    import torch
    import reference_profile as reference
    import export_runtime as export
    export.compute_guard()
    if not direct:
        return reference.inference_runtime(out, protocol, bundle)
    # The exported state supplies all parameters. Cached tokenizer files remain required.
    old = bundle['assets']['resolved_config']['model'].get('skip_dit_load_from_pretrain')
    bundle['assets']['resolved_config']['model']['skip_dit_load_from_pretrain'] = True
    try:
        with empty_factory(torch):
            runtime = reference.inference_runtime(out, protocol, bundle)
    finally:
        bundle['assets']['resolved_config']['model']['skip_dit_load_from_pretrain'] = old
    runtime.direct_loader = True
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
