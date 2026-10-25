"""Arm installation; only ours receives the VQ-WAM inference engine."""
from __future__ import annotations
from pathlib import Path
from common import torch_module, read_json, load_tensors, target_manifest, restore_bf16


def reference_a4(value):
    torch = torch_module()
    scale = value.float().abs().amax(-1, keepdim=True) / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(value.float() / safe).clamp(-7, 7) * scale).to(torch.bfloat16)


def right_u(value, matrix):
    torch = torch_module()
    full = value.shape[-1] // 128 * 128
    if not full:
        return value.float()
    with torch.autocast(device_type=value.device.type, enabled=False):
        head = (value[..., :full].float().reshape(*value.shape[:-1], full // 128, 128)
                @ matrix).reshape(*value.shape[:-1], full)
        return torch.cat((head, value[..., full:].float()), -1) if full < value.shape[-1] else head


def transform_weight(value, transform, matrix):
    import batched_transform
    scaled = value.float() * transform['scale']
    core = batched_transform.core
    full = scaled.shape[-1] // 128 * 128
    pieces = []
    if full:
        blocks = (scaled[..., :full] * transform['signs'][:full]).reshape(
            *scaled.shape[:-1], full // 128, 128)
        pieces.append(core._fwht(blocks).reshape(*scaled.shape[:-1], full))
    if full < scaled.shape[-1]:
        pieces.append(core._right_r(scaled[..., full:], transform['signs'][full:], True, 128))
    transformed = pieces[0] if len(pieces) == 1 else torch_module().cat(pieces, -1)
    return right_u(transformed, matrix)


def install_reference(runtime, p, artifact_dir):
    """Reference for the same trained artifact; this is not an extra experimental arm."""
    torch = torch_module()
    import batched_transform
    import micro_codebook as micro
    artifact_dir = Path(artifact_dir)
    manifest = read_json(artifact_dir / 'manifest.json')
    if (manifest.get('format') != 'cosmos_vqwam_deployment_v1'
            or manifest.get('protocol') != p['protocol'] or manifest.get('completed_updates') != 8
            or manifest.get('target_modules') != target_manifest(runtime)):
        raise RuntimeError('Ours deployment does not match the full scope/update8')
    runtime.controller._backup_weights()
    restore = getattr(runtime, 'restore_ours_vqwam', None)
    if callable(restore):
        restore()
    restore_bf16(runtime)
    matrix = load_tensors(artifact_dir / manifest['rotation_file'])['matrix'].to('cuda').float()
    holds = {}
    for row, (name, module) in zip(manifest['modules'], runtime.target_linears.items()):
        if row['name'] != name or row['shape'] != [module.out_features, module.in_features]:
            raise RuntimeError('Ours bank order/shape differs')
        payload = load_tensors(artifact_dir / row['file'])
        encoding = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in payload['encoding'].items()}
        transform = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in payload['transform'].items()}
        micro.check_encoding(encoding)
        module.weight.data.copy_(micro.decode_vq(encoding).to(torch.bfloat16))
        holds[name] = transform

        def forward(value, _module=module, _transform=transform):
            operand = right_u(batched_transform.transform_input(value, _transform), matrix)
            activation = reference_a4(operand)
            with torch.autocast(device_type=activation.device.type, enabled=False):
                return torch.nn.functional.linear(activation, _module.weight, _module.bias)
        module.forward = forward
    runtime._ours_reference_holds = (holds, matrix)
    metadata = {'arm': 'ours_vqwam', 'execution': 'decoded_bf16_fake_a4_reference',
                'native_int4_kernel_used': False, 'packed_gpu_resident': False,
                'artifact_dir': str(artifact_dir), 'target_module_count': 280,
                'same_module_effective_bpw': manifest['effective_bpw'],
                'whole_dit_effective_bpw': manifest['whole_dit_effective_bpw']}
    runtime.current_arm, runtime.quantization_evidence = 'ours_vqwam', metadata
    return metadata


def weight_storage(runtime, target_payload_bytes):
    target_ids = {id(m.weight) for m in runtime.target_linears.values() if m.weight is not None}
    seen, total, excluded, count = set(), 0, 0, 0
    for parameter in runtime.model.net.parameters():
        key = (str(parameter.device), parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes())
        if key in seen:
            continue
        seen.add(key)
        total += parameter.numel() * parameter.element_size()
        count += parameter.numel()
        if id(parameter) not in target_ids:
            excluded += parameter.numel() * parameter.element_size()
    return {'bf16_dit_payload_bytes': total, 'unquantized_dit_payload_bytes': excluded,
            'whole_dit_parameter_numel': count, 'packed_storage_bytes': int(target_payload_bytes + excluded),
            'storage_measurement': 'unique_DiT_tensor_payload; serialized_bytes_reported_separately'}


def install_arm(runtime, p, root, arm, optimized=True):
    root = Path(root)
    runtime.controller._backup_weights()
    restore_engine = getattr(runtime, 'restore_ours_vqwam', None)
    if callable(restore_engine):
        restore_engine()
    if arm in ('bf16', 'spinquant_w4a4'):
        restore_bf16(runtime)
        metadata = runtime.controller.set_arm(arm, artifact_dir=(
            p['reuse']['spinquant_artifact'] if arm != 'bf16' else None))
        if arm == 'bf16':
            payload = sum(m.weight.numel() * m.weight.element_size() for m in runtime.target_linears.values())
        else:
            summary = read_json(Path(p['reuse']['spinquant_artifact']) / 'summary.json')
            payload = summary['size_report']['spinquant']['bytes']
        metadata.update(weight_storage(runtime, payload))
        metadata['calibration_reused'] = arm != 'bf16'
        metadata['backend'] = 'official_bf16' if arm == 'bf16' else 'reused_decoded_bf16_fake_a4'
    elif arm in ('quarot_w4a4', 'svdquant_w4a4', 'viditq_w4a4'):
        from baseline_quant import install_baseline
        stage = {'quarot_w4a4': 'quarot', 'svdquant_w4a4': 'svdquant', 'viditq_w4a4': 'viditq'}[arm]
        directory = root / 'results' / stage
        restore_bf16(runtime)
        metadata = install_baseline(runtime, p, arm, directory)
        manifest = read_json(directory / 'manifest.json')
        payload = metadata.get('encoded_target_bytes', manifest.get('encoded_target_bytes'))
        if payload is None:
            report = manifest.get('storage', {})
            payload = report.get('tensor_payload_bytes', report.get('target_tensor_bytes'))
        if payload is None:
            raise RuntimeError('Baseline packed target payload accounting is missing')
        metadata.update(weight_storage(runtime, payload))
        metadata['backend'] = 'method_specific_eager_decoded_bf16_fake_a4'
    elif arm == 'ours_vqwam':
        directory = root / 'results/ours/deployment'
        if optimized:
            from validation import install_validated
            metadata = install_validated(runtime, p, root, directory)
        else:
            metadata = install_reference(runtime, p, directory)
        bundle_directory = root / 'results/ours/inference_bundle'
        storage_path = bundle_directory / 'storage.json'
        if not storage_path.exists():
            from export_bundle import export
            export(runtime, p, directory, bundle_directory)
        storage = read_json(storage_path)
        metadata['export_storage'] = storage
        metadata['packed_storage_bytes'] = storage['unique_tensor_payload_bytes']
        metadata['serialized_deployment_bytes'] = storage['deployment_files_bytes']
        metadata['backend'] = 'ours_C_decoder_buffer_BF16_GEMM'
    else:
        raise ValueError('Unknown requested arm: ' + arm)
    runtime.current_arm, runtime.quantization_evidence = arm, metadata
    runtime.query_count_by_arm.setdefault(arm, 0)
    return metadata
