"""Inference-only C80 R128 MXVQ export; heavy operations require a PBS allocation.

The bundle contains complete persistent model state, processor statistics and
resolved configuration. Architecture/tokenizer code remains an external runtime
dependency. ``stream_decode`` is a residency diagnostic, not a fused kernel.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import socket

FORMAT = 'fastwam_c80_r128_reader_kl_inference_v1'
SOURCE_JOB = '25921795.pbs101'
MODULE_COUNT = 614
ENCODING_KEYS = ('books', 'row_scales', 'indices', 'scale_exponents')


def compute_guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or 'login' in host.lower() or not nodefile:
        raise RuntimeError('Approved PBS compute allocation required')
    nodes = {node.split('.')[0] for node in Path(nodefile).read_text().split()}
    if host not in nodes:
        raise RuntimeError('Hostname is outside the assigned PBS allocation')


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def _cpu_copy(value, torch, memo):
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        key = _storage_key(tensor)
        if key not in memo:
            # Copy storage once so aliased model keys/views do not inflate export.
            byte_view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            byte_view = byte_view.set_(tensor.untyped_storage(), 0,
                                      (tensor.untyped_storage().nbytes(),), (1,))
            copied = byte_view.cpu().clone() if tensor.device.type == 'cpu' else byte_view.cpu()
            # Keep the source storage alive; otherwise allocator pointer reuse can alias later temporaries.
            memo[key] = (copied, tensor.untyped_storage())
        typed = memo[key][0].view(tensor.dtype)
        return torch.as_strided(typed, tensor.shape, tensor.stride(), tensor.storage_offset())
    if isinstance(value, dict):
        return {key: _cpu_copy(item, torch, memo) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(item, torch, memo) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported inference-bundle value: {type(value).__name__}')


def _walk_tensors(value, torch, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_tensors(item, torch, f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_tensors(item, torch, f'{prefix}.{index}')


def inventory_from_descriptors(descriptors):
    """Pure-stdlib accounting, including aliased keys and padded storage."""
    seen = set()
    logical = unique = count = 0
    dtypes = {}
    for descriptor in descriptors:
        count += 1
        logical += descriptor['logical_bytes']
        key = descriptor['storage_key']
        if key not in seen:
            seen.add(key)
            unique += descriptor['storage_bytes']
            dtype = descriptor['dtype']
            dtypes[dtype] = dtypes.get(dtype, 0) + descriptor['storage_bytes']
    return dict(tensor_references=count, logical_tensor_bytes=logical,
                unique_storage_count=len(seen), unique_storage_bytes=unique,
                unique_storage_bytes_by_first_dtype=dtypes)


def tensor_inventory(value):
    compute_guard()
    import torch
    return inventory_from_descriptors(dict(storage_key=_storage_key(tensor),
        storage_bytes=tensor.untyped_storage().nbytes(),
        logical_bytes=tensor.numel() * tensor.element_size(), dtype=str(tensor.dtype))
        for _, tensor in _walk_tensors(value, torch))


def _target_state_keys(model, modules):
    target_ids = {id(module) for module in modules.values()}
    return sorted(f'{name}.weight' if name else 'weight'
                  for name, module in model.named_modules(remove_duplicate=False)
                  if id(module) in target_ids)


def _validate_header(bundle):
    if (bundle.get('format') != FORMAT or bundle.get('source_job') != SOURCE_JOB
            or bundle.get('arm') != 'reader_kl' or bundle.get('training_updates') != 0
            or len(bundle.get('quantized', {})) != MODULE_COUNT
            or set(bundle.get('rotations', {})) != {'action', 'video'}):
        raise ValueError('Expected the frozen final C80 R128 A+Z+KL inference bundle')
    excluded = set(bundle['excluded_model_state_keys'])
    if not excluded or excluded.intersection(bundle['model_state']):
        raise ValueError('Bundle retains a quantized dense weight or lacks its binding')


def minimal_inference_config(resolve):
    """Resolve selected original nodes while their root interpolation context exists."""
    train = dict(processor=resolve('data.train.processor'), video_size=resolve('data.train.video_size'))
    for key, default in (('concat_multi_camera', 'horizontal'), ('num_frames', 33),
                         ('action_video_freq_ratio', 1)):
        train[key] = resolve('data.train.' + key, default)
    evaluation = {}
    for key, default in (
            ('action_infer_mode', 'idm'), ('num_inference_steps', 10), ('sigma_shift', 1.),
            ('action_horizon', None), ('negative_prompt', ''), ('text_cfg_scale', 1.),
            ('rand_device', 'cpu'), ('tiled', False), ('compile_action_infer', False),
            ('visualize_future_video', False), ('use_action_ensembler', False),
            ('binarize_gripper', True), ('replan_steps', 10), ('num_steps_wait', 30),
            ('device', 'cuda')):
        evaluation[key] = resolve('EVALUATION.' + key, default)
    return dict(model=resolve('model'), data=dict(train=train), EVALUATION=evaluation,
                mixed_precision=resolve('mixed_precision', 'bf16'), seed=resolve('seed', None),
                eval_num_inference_steps=resolve('eval_num_inference_steps', 10))


def _deployment_assets(runtime):
    from omegaconf import OmegaConf

    def resolve(path, default=None):
        value = OmegaConf.select(runtime.cfg, path, default=default)
        return (OmegaConf.to_container(value, resolve=True, enum_to_str=True)
                if OmegaConf.is_config(value) else value)

    # Full training config contains Hydra-only output paths; resolve only deployed inputs.
    config = minimal_inference_config(resolve)
    stats_path = Path(str(resolve('EVALUATION.dataset_stats_path')))
    return dict(resolved_config_yaml=OmegaConf.to_yaml(OmegaConf.create(config), resolve=False),
                resolved_config=config,
                dataset_stats_json=stats_path.read_text(),
                architecture_dependency='FastWAM architecture, operators and processor/tokenizer code',
                training_checkpoint_required=False, source_bank_required=False)


def export_bundle(runtime, out, p):
    """Export after the original runtime's saved action/future anchor has passed.

    ``p`` is the profiling protocol. Source binding comes from ``runtime.protocol``.
    Returns paths and a storage report; no tensors/legacy banks are returned.
    """
    compute_guard()
    import torch
    modules, banks = runtime.q.modules, runtime.banks
    meta = runtime.install_metadata
    source = runtime.protocol['arm_sources']['reader_kl']
    if (runtime.current_arm != 'reader_kl' or len(modules) != MODULE_COUNT
            or meta.get('source_job') != SOURCE_JOB or source['source_job'] != SOURCE_JOB
            or meta.get('all_parameters_frozen') is not True
            or any(parameter.requires_grad for parameter in runtime.model.parameters())):
        raise ValueError('Export requires the verified frozen reader_kl source runtime')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path, manifest_path = out / 'inference_bundle.pt', out / 'storage.json'
    if path.exists() or path.with_suffix('.pt.tmp').exists():
        raise FileExistsError('Inference bundle already exists; choose a fresh export directory')
    excluded = _target_state_keys(runtime.model, modules)
    state = runtime.model.state_dict()
    if not set(excluded).issubset(state):
        raise ValueError('Quantized module aliases are absent from model state')
    memo = {}
    with torch.no_grad():
        bundle = dict(format=FORMAT, source_job=SOURCE_JOB, arm='reader_kl', training_updates=0,
            source_deployment=source['deployment'], source_protocol=source['source_protocol'],
            profile_protocol=p['protocol'],
            activation=dict(bits=4, signed_range=[-7, 7], dynamic_axis=-1,
                            scale='maxabs/7', clipping_ratio=1, output_dtype='bfloat16'),
            execution='BF16 GEMM on VQ-decoded weights; no native low-bit kernel',
            quantized={}, transforms={}, rotations={},
            excluded_model_state_keys=excluded,
            model_state=_cpu_copy({key: value for key, value in state.items() if key not in excluded}, torch, memo),
            processor_stats=_cpu_copy(runtime.processor.normalizer.stats, torch, memo),
            processor_state=_cpu_copy(runtime.processor.state_dict(), torch, memo)
                            if hasattr(runtime.processor, 'state_dict') else {},
            assets=_deployment_assets(runtime))
        for index, (name, mod) in enumerate(modules.items()):
            encoding = banks.stored_encoding(name)
            import micro_codebook as micro
            micro.check_encoding(encoding)
            bundle['quantized'][name] = dict(module_index=index, shape=tuple(encoding['shape']),
                format=encoding['format'], block_size=4, scale_block=32,
                **_cpu_copy({key: encoding[key] for key in ENCODING_KEYS}, torch, memo))
            transform = banks.transforms[name]
            if (transform['input_dim'] != mod.in_features or not transform['hadamard']
                    or transform['block'] != 128):
                raise ValueError('Frozen transform differs at ' + name)
            bundle['transforms'][name] = _cpu_copy({key: transform[key] for key in
                ('scale', 'signs', 'hadamard', 'block', 'input_dim')}, torch, memo)
        for stream in ('action', 'video'):
            matrix = banks.rotations.matrix(stream)
            if matrix.dtype != torch.float32 or tuple(matrix.shape) != (128, 128):
                raise ValueError('Expected the final fixed FP32 Cayley U')
            bundle['rotations'][stream] = _cpu_copy(matrix, torch, memo)
    _validate_header(bundle)
    target_numel = sum(mod.in_features * mod.out_features for mod in modules.values())
    original = tensor_inventory(state)
    parameter_names = {name for name, _ in runtime.model.named_parameters(remove_duplicate=False)}
    buffer_names = {name for name, _ in runtime.model.named_buffers(remove_duplicate=False)}
    original_parameters = tensor_inventory({key: value for key, value in state.items() if key in parameter_names})
    original_buffers = tensor_inventory({key: value for key, value in state.items() if key in buffer_names})
    nonquantized_parameters = tensor_inventory({key: value for key, value in bundle['model_state'].items()
                                              if key in parameter_names})
    encoding_inventory = tensor_inventory(bundle['quantized'])
    target_with_transforms = tensor_inventory({key: bundle[key] for key in ('quantized', 'transforms', 'rotations')})
    whole = tensor_inventory(bundle)
    categories = {key: tensor_inventory(bundle[key]) for key in
                  ('quantized', 'transforms', 'rotations', 'model_state', 'processor_stats', 'processor_state')}
    per_module = []
    for name, encoded in bundle['quantized'].items():
        rows, width = encoded['shape']
        payload = {key: encoded[key].numel() * encoded[key].element_size() for key in ENCODING_KEYS}
        per_module.append(dict(module=name, shape=[rows, width], weight_values=rows * width,
            payload_bytes=payload, encoded_bytes=sum(payload.values()),
            effective_encoding_bits_per_weight=8 * sum(payload.values()) / (rows * width)))
    metadata = {key: value for key, value in bundle.items() if key not in
                ('quantized', 'transforms', 'rotations', 'model_state', 'processor_stats', 'processor_state')}
    summary = dict(format=FORMAT, source_job=SOURCE_JOB, arm='reader_kl', target_modules=MODULE_COUNT,
        target_weight_values=target_numel, target_bf16_weight_bytes=target_numel * 2,
        encoding=encoding_inventory, target_with_transforms=target_with_transforms,
        original_dense_model_state=original, whole_bundle=whole, categories=categories,
        original_dense_model_parameters=original_parameters, original_persistent_model_buffers=original_buffers,
        unquantized_model_parameters=nonquantized_parameters,
        effective_encoding_bits_per_target_weight=8 * encoding_inventory['unique_storage_bytes'] / target_numel,
        effective_deployment_bits_per_target_weight=8 * target_with_transforms['unique_storage_bytes'] / target_numel,
        target_encoding_compression_ratio=target_numel * 2 / encoding_inventory['unique_storage_bytes'],
        excluded_dense_state_keys=excluded, per_module=per_module,
        plain_metadata_utf8_bytes=len(json.dumps(metadata, ensure_ascii=False).encode('utf-8')),
        accounting='unique underlying tensor storage counted once, including shared aliases and padding; '
                   'category totals can overlap if storage is shared; whole_bundle is globally deduplicated',
        claim_scope='Storage only; current bundle uses BF16 GEMM and dynamic fake-A4 operands')
    temp = path.with_suffix('.pt.tmp')
    torch.save(bundle, temp)
    temp.replace(path)
    summary['torch_file_bytes'] = path.stat().st_size
    summary['torch_archive_overhead_bytes'] = summary['torch_file_bytes'] - whole['unique_storage_bytes']
    summary['bundle_path'], summary['manifest_path'] = str(path), str(manifest_path)
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


def load_bundle(path):
    """CPU-map the tensor bundle inside a real compute allocation."""
    compute_guard()
    import torch
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    _validate_header(bundle)
    return bundle


class FixedRotations:
    def __init__(self, matrices):
        self.matrices = matrices

    def matrix(self, stream):
        return self.matrices[stream]

    def apply_after_h(self, value, stream):
        import torch
        full = value.shape[-1] // 128 * 128
        if not full:
            return value
        with torch.autocast(device_type='cuda', enabled=False):
            head = value[..., :full].float().reshape(*value.shape[:-1], full // 128, 128)
            head = (head @ self.matrix(stream)).reshape(*value.shape[:-1], full)
            return torch.cat((head, value[..., full:]), -1) if full < value.shape[-1] else head


def a4(value):
    """Original A4 arithmetic, separately visible to the profiling harness."""
    import torch
    scale = value.float().abs().amax(-1, keepdim=True) / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(value.float() / safe).clamp(-7, 7) * scale).to(torch.bfloat16)


class FrozenBanks:
    """No masters, autograd surrogate, optimizer, source banks or reader."""
    def __init__(self, runtime, bundle, mode):
        import torch
        import micro_codebook as micro
        self.modules, self.mode = runtime.q.modules, mode
        self.seen, self.forwards = set(), {}
        self.encodings, self.transforms = {}, {}
        device = next(runtime.model.parameters()).device
        self.rotations = FixedRotations({key: value.to(device=device) for key, value in bundle['rotations'].items()})
        for name, mod in self.modules.items():
            encoded = {key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                       for key, value in bundle['quantized'][name].items() if key != 'module_index'}
            micro.check_encoding(encoded)
            self.transforms[name] = {key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                                    for key, value in bundle['transforms'][name].items()}
            if mode == 'dense':
                mod.weight = torch.nn.Parameter(micro.decode_vq(encoded), requires_grad=False)
            else:
                self.encodings[name] = encoded
            mod.forward = self._forward(runtime, name, mod)

    def parameters(self):
        return []

    def operand(self, value, name):
        import batched_transform
        from rotation_bank import stream_for
        value = batched_transform.transform_input(value, self.transforms[name])
        value = self.rotations.apply_after_h(value, stream_for(name))
        return a4(value)

    def decode(self, name):
        import micro_codebook as micro
        return micro.decode_vq(self.encodings[name])

    def _forward(self, runtime, name, mod):
        import torch

        @torch.no_grad()
        def forward(value):
            self.seen.add(name)
            runtime.linear_calls += 1
            operand = self.operand(value, name)
            # shortcut: full decode per Linear is residency-only, replace with fused tile decode for speed.
            weight = mod.weight if self.mode == 'dense' else self.decode(name)
            return torch.nn.functional.linear(operand, weight, mod.bias)
        return forward


def install_bundle(runtime, bundle, mode='dense'):
    """Install on a fresh BF16 pilot Runtime or replace a verified legacy runtime.

    Dense mode keeps only decoded BF16 target weights on GPU, discarding encoded
    weights there. Stream mode keeps only packed weights at rest, with full-weight
    temporary decode; its latency is explicitly not a fused-kernel result.
    """
    compute_guard()
    import torch
    if mode not in ('dense', 'stream_decode'):
        raise ValueError('mode must be dense or stream_decode')
    if isinstance(bundle, (str, Path)):
        bundle = load_bundle(bundle)
    _validate_header(bundle)
    modules = runtime.q.modules
    if list(modules) != list(bundle['quantized']) or len(modules) != MODULE_COUNT:
        raise ValueError('Runtime target module names/order differ from inference bundle')
    expected_excluded = _target_state_keys(runtime.model, modules)
    if expected_excluded != bundle['excluded_model_state_keys']:
        raise ValueError('Runtime model aliases differ from inference bundle')
    for name, mod in modules.items():
        if tuple(bundle['quantized'][name]['shape']) != (mod.out_features, mod.in_features):
            raise ValueError('Target shape differs at ' + name)
    runtime.q.disable()
    runtime.banks, runtime.seen, runtime.forwards = None, set(), {}
    for module in modules.values():
        # Break old forward closures so masters/old encodings can be collected before decoding again.
        module.forward = torch.nn.Linear.forward.__get__(module, type(module))
        module.register_parameter('weight', None)
    gc.collect()
    torch.cuda.empty_cache()
    remaining = set(runtime.model.state_dict())
    if remaining != set(bundle['model_state']):
        raise ValueError('Unquantized persistent state differs from inference bundle')
    runtime.model.load_state_dict(bundle['model_state'], strict=True)
    runtime.processor.set_normalizer_from_stats(bundle['processor_stats'])
    if bundle['processor_state']:
        runtime.processor.load_state_dict(bundle['processor_state'], strict=True)
    runtime.processor.eval()
    runtime.linear_calls = 0
    banks = FrozenBanks(runtime, bundle, mode)
    runtime.banks, runtime.seen, runtime.forwards = banks, banks.seen, {}
    runtime.current_arm = runtime.arm_name = 'reader_kl'
    runtime.model.eval()
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.install_metadata = dict(source_job=SOURCE_JOB, arm='reader_kl', inference_only=True,
        all_parameters_frozen=True, new_training_updates=0, mode=mode, module_count=MODULE_COUNT,
        native_low_bit_kernel=False, native_int4_gemm_calls=0, kv_cache_dtype='bf16',
        packed_gpu_resident=mode == 'stream_decode', dense_target_weights=mode == 'dense',
        stream_decode_performance_claim=False, codebook_masters_resident=False,
        rotation_implementation='saved fixed FP32 Cayley U after original signed H128',
        processor_stats_loaded_from_bundle=True)
    gc.collect()
    torch.cuda.empty_cache()
    return banks
