"""CPU-only storage attribution for the frozen C80 R128 inference bundle.

The CLI must run inside the approved PBS allocation. It loads the bundle onto
CPU only; this script never initializes CUDA or invokes the model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import export_runtime


EXPECTED_UNIQUE_STORAGE_BYTES = 12_779_984_326
DEFAULT_BUNDLE = Path(
    '/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-inference-profile-20261023-r2/'
    'results/export/inference_bundle.pt'
)
MAX_REPORT_BYTES = 250_000

_COMPONENT_MARKERS = (
    ('action_policy', ('action_head', 'action_decoder', 'policy_head', 'act_head',
                       'action_model', 'action_expert', 'policy', 'actor', 'motor', 'action')),
    ('latent_autoencoder', ('vae', 'image_vae', 'video_vae', 'autoencoder', 'latent_autoencoder')),
    ('vision_encoder', ('vision_tower', 'vision_encoder', 'vision_model', 'visual_encoder',
                        'image_encoder', 'clip_vision', 'siglip', 'visual', 'vision', 'vit')),
    ('text_encoder', ('text_encoder', 'text_tower', 'text_model', 'language_model', 'llm', 't5', 'umt5',
                      'embed_tokens', 'token_embedding', 'word_embeddings', 'lm_head')),
    ('video_world_model', ('world_model', 'dynamics', 'video_decoder', 'video_model',
                           'future_model', 'predictor', 'denoiser', 'diffusion', 'dit',
                           'video', 'video_expert', 'latent')),
)
_PURPOSE = {
    'action_policy': 'Action or control prediction path; inferred from state key names.',
    'latent_autoencoder': 'Image or video latent encoding/decoding path; inferred from state key names.',
    'vision_encoder': 'Image or visual feature path; inferred from state key names.',
    'text_encoder': 'Text or token conditioning path; inferred from state key names.',
    'video_world_model': 'Video, future-latent, or dynamics path; inferred from state key names.',
    'unclassified': 'No architecture-specific marker in the state key; purpose is unclassified.',
}
_PARAMETER_LEAVES = {
    'weight', 'bias', 'scale', 'scaling', 'shift', 'offset', 'embedding',
    'running_mean', 'running_var', 'num_batches_tracked', 'pos_embed',
    'cls_token', 'mask_token', 'inv_freq', 'frequencies', 'mean', 'std',
}
_LAYER_MARKERS = {
    'layers', 'blocks', 'transformer_blocks', 'encoder_layers', 'decoder_layers',
    'layer', 'block', 'h',
}


def _path_parts(key: str) -> list[str]:
    return [part for part in re.split(r'[./]+', key.lower()) if part]


def classify_component(state_key: str) -> tuple[str, str | None]:
    """Return a conservative component guess and the matching path marker."""
    parts = _path_parts(state_key)
    for component, markers in _COMPONENT_MARKERS:
        for part in parts:
            if part in markers:
                return component, part
    return 'unclassified', None


def _module_path(state_key: str) -> str:
    parts = state_key.split('.')
    if len(parts) > 1 and parts[-1].lower() in _PARAMETER_LEAVES:
        return '.'.join(parts[:-1])
    return state_key


def _layer_id(module_path: str) -> str | None:
    parts = module_path.split('.')
    for index, part in enumerate(parts[:-1]):
        if part.lower() in _LAYER_MARKERS and parts[index + 1].isdigit():
            return '.'.join(parts[:index + 2])
        match = re.fullmatch(r'(?:layer|block|h)[_-](\d+)', part.lower())
        if match:
            return '.'.join(parts[:index + 1])
    return None


def inventory_from_descriptors(descriptors, expected_unique_bytes: int | None = None) -> dict:
    """Group tensor descriptors by underlying storage, counting aliases once.

    Descriptors are ordinary dictionaries so this accounting is testable without
    importing torch. ``storage_key`` identifies the backing storage, not a view.
    """
    groups = {}
    logical_total = 0
    reference_count = 0
    for raw in descriptors:
        row = dict(raw)
        required = ('state_key', 'storage_key', 'storage_bytes', 'logical_bytes', 'dtype')
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f'Missing descriptor fields: {missing}')
        row['state_key'] = str(row['state_key'])
        row['storage_bytes'] = int(row['storage_bytes'])
        row['logical_bytes'] = int(row['logical_bytes'])
        if row['storage_bytes'] < 0 or row['logical_bytes'] < 0:
            raise ValueError(f'Negative tensor byte count: {row["state_key"]}')
        row.setdefault('module', _module_path(row['state_key']))
        component, marker = classify_component(row['state_key'])
        row.setdefault('component', component)
        row.setdefault('component_marker', marker)
        row.setdefault('shape', [])
        row.setdefault('stride', [])
        row.setdefault('storage_offset', 0)
        row.setdefault('is_contiguous', row['logical_bytes'] == row['storage_bytes'])
        row['is_view'] = bool(row.get('is_view',
            row['logical_bytes'] != row['storage_bytes'] or row['storage_offset'] != 0
            or not row['is_contiguous']))
        groups.setdefault(row['storage_key'], []).append(row)
        logical_total += row['logical_bytes']
        reference_count += 1

    storage_rows = []
    module_totals = {}
    component_totals = {}
    dtype_bytes = {}
    reference_dtype_counts = {}
    for storage_key, refs in groups.items():
        refs.sort(key=lambda item: item['state_key'])
        owner = refs[0]
        storage_bytes = max(row['storage_bytes'] for row in refs)
        if any(row['storage_bytes'] != storage_bytes for row in refs):
            raise ValueError(f'Inconsistent backing sizes for storage aliases: {owner["state_key"]}')
        sid = len(storage_rows) + 1
        aliases = [dict(state_key=row['state_key'], module=row['module'],
                        component=row['component'], component_marker=row.get('component_marker'),
                        dtype=row['dtype'],
                        shape=row['shape'], stride=row['stride'],
                        storage_offset=row['storage_offset'], logical_bytes=row['logical_bytes'],
                        is_view=row['is_view']) for row in refs]
        storage = dict(storage_id=sid, owner_state_key=owner['state_key'],
                       owner_module=owner['module'], owner_component=owner['component'],
                       storage_bytes=storage_bytes,
                       logical_bytes_across_references=sum(row['logical_bytes'] for row in refs),
                       tensor_reference_count=len(refs), alias_reference_count=len(refs) - 1,
                       dtypes=sorted({str(row['dtype']) for row in refs}),
                       has_tensor_view=any(row['is_view'] for row in refs),
                       references=aliases)
        storage_rows.append(storage)
        dtype = str(owner['dtype'])
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + storage_bytes

        component = owner['component']
        c = component_totals.setdefault(component, dict(component=component,
            purpose_hint=_PURPOSE.get(component, _PURPOSE['unclassified']),
            unique_storage_bytes=0, unique_storage_count=0, tensor_key_count=0,
            logical_tensor_bytes=0, alias_reference_count=0, modules=set()))
        c['unique_storage_bytes'] += storage_bytes
        c['unique_storage_count'] += 1

        for row in refs:
            ref_component = component_totals.setdefault(row['component'], dict(component=row['component'],
                purpose_hint=_PURPOSE.get(row['component'], _PURPOSE['unclassified']),
                unique_storage_bytes=0, unique_storage_count=0, tensor_key_count=0,
                logical_tensor_bytes=0, alias_reference_count=0, modules=set()))
            ref_component['tensor_key_count'] += 1
            ref_component['logical_tensor_bytes'] += row['logical_bytes']
            ref_component['alias_reference_count'] += int(row['state_key'] != owner['state_key'])
            ref_component['modules'].add(row['module'])
            reference_dtype_counts[str(row['dtype'])] = reference_dtype_counts.get(str(row['dtype']), 0) + 1
            m = module_totals.setdefault(row['module'], dict(module=row['module'],
                component=row['component'], purpose_hint=_PURPOSE.get(row['component'], _PURPOSE['unclassified']),
                tensor_key_count=0, logical_tensor_bytes=0, unique_storage_bytes=0,
                unique_storage_count=0, alias_reference_count=0, layer_id=_layer_id(row['module'])))
            m['tensor_key_count'] += 1
            m['logical_tensor_bytes'] += row['logical_bytes']
            if row['state_key'] == owner['state_key']:
                m['unique_storage_bytes'] += storage_bytes
                m['unique_storage_count'] += 1
            else:
                m['alias_reference_count'] += 1

    for row in component_totals.values():
        row['module_count'] = len(row.pop('modules'))
    unique_total = sum(row['storage_bytes'] for row in storage_rows)
    if sum(row['unique_storage_bytes'] for row in component_totals.values()) != unique_total:
        raise AssertionError('Component unique-storage attribution does not sum to model_state storage')
    if sum(row['unique_storage_bytes'] for row in module_totals.values()) != unique_total:
        raise AssertionError('Module unique-storage attribution does not sum to model_state storage')
    if sum(row['tensor_key_count'] for row in module_totals.values()) != reference_count:
        raise AssertionError('Module tensor-reference counts do not sum to model_state keys')
    if expected_unique_bytes is not None and unique_total != int(expected_unique_bytes):
        raise ValueError(f'Unique storage total {unique_total} != expected {expected_unique_bytes}')

    component_rows = sorted(component_totals.values(),
        key=lambda row: (-row['unique_storage_bytes'], row['component']))
    module_rows = sorted(module_totals.values(), key=lambda row: row['module'])
    tensor_rows = sorted((ref for storage in storage_rows for ref in storage['references']),
        key=lambda row: (-row['logical_bytes'], row['state_key']))
    storage_rows.sort(key=lambda row: (-row['storage_bytes'], row['owner_state_key']))
    for index, row in enumerate(storage_rows, 1):
        row['storage_id'] = index
    # Rebuild reference -> storage attribution after sorting storage records.
    attribution = {}
    for storage in storage_rows:
        for reference in storage['references']:
            attribution[reference['state_key']] = dict(storage_id=storage['storage_id'],
                backing_storage_bytes=storage['storage_bytes'],
                unique_storage_owner=reference['state_key'] == storage['owner_state_key'])
    for row in tensor_rows:
        row.update(attribution[row['state_key']])

    return dict(tensor_key_count=reference_count, logical_tensor_bytes=logical_total,
        unique_storage_count=len(storage_rows), unique_storage_bytes=unique_total,
        alias_extra_reference_count=reference_count - len(storage_rows),
        unique_storage_bytes_by_owner_dtype=dict(sorted(dtype_bytes.items())),
        tensor_key_count_by_dtype=dict(sorted(reference_dtype_counts.items())),
        attribution_rule='Each backing storage is counted once and assigned to its lexicographically first state key.',
        components=component_rows, modules=module_rows, storages=storage_rows,
        top_storage_records=storage_rows[:50], top_tensor_references=tensor_rows[:50])


def _describe_tensor(state_key, tensor) -> dict:
    storage = tensor.untyped_storage()
    logical_bytes = int(tensor.numel()) * int(tensor.element_size())
    storage_bytes = int(storage.nbytes())
    return dict(state_key=state_key,
        storage_key=(str(tensor.device), int(storage.data_ptr()), storage_bytes),
        storage_bytes=storage_bytes, logical_bytes=logical_bytes, dtype=str(tensor.dtype),
        shape=[int(value) for value in tensor.shape],
        stride=[int(value) for value in tensor.stride()],
        storage_offset=int(tensor.storage_offset()), is_contiguous=bool(tensor.is_contiguous()))


def inspect_bundle(bundle, expected_unique_bytes: int | None = EXPECTED_UNIQUE_STORAGE_BYTES) -> dict:
    """Inspect an already CPU-loaded bundle, avoiding a second bundle read."""
    export_runtime.compute_guard()
    export_runtime._validate_header(bundle)
    state = bundle['model_state']
    report = inventory_from_descriptors((_describe_tensor(key, value) for key, value in state.items()))
    excluded = list(bundle.get('excluded_model_state_keys', ()))
    layer_counts = defaultdict(lambda: {'remaining_modules': set(), 'quantized_targets': set()})
    for row in report['modules']:
        if row['layer_id']:
            layer_counts[row['component']]['remaining_modules'].add(row['layer_id'])
    for module in bundle.get('quantized', {}):
        key = module
        component, _ = classify_component(key)
        layer = _layer_id(module)
        if layer:
            layer_counts[component]['quantized_targets'].add(module)
    model_state = {key: report[key] for key in (
        'tensor_key_count', 'logical_tensor_bytes', 'unique_storage_count',
        'unique_storage_bytes', 'alias_extra_reference_count', 'unique_storage_bytes_by_owner_dtype',
        'tensor_key_count_by_dtype')}
    quantized_by_component = defaultdict(set)
    for key in bundle.get('quantized', {}):
        component, _ = classify_component(key)
        quantized_by_component[component].add(key)
    component_rows = []
    for row in report['components']:
        item = dict(row)
        item['quantized_target_module_count'] = len(quantized_by_component[item['component']])
        component_rows.append(item)
    known_components = {row['component'] for row in component_rows}
    for component, targets in quantized_by_component.items():
        if component not in known_components:
            component_rows.append(dict(component=component,
                purpose_hint=_PURPOSE.get(component, _PURPOSE['unclassified']),
                unique_storage_bytes=0, unique_storage_count=0, tensor_key_count=0,
                logical_tensor_bytes=0, alias_reference_count=0, module_count=0,
                quantized_target_module_count=len(targets)))
    component_rows.sort(key=lambda row: (-row['unique_storage_bytes'], row['component']))
    summary = dict(schema_version=1, phase='pre', created_utc=datetime.now(timezone.utc).isoformat(),
        bundle_format=bundle.get('format'), source_job=bundle.get('source_job'), arm=bundle.get('arm'),
        model_state=model_state, expected_unique_storage_bytes=expected_unique_bytes,
        expected_bytes_match=(expected_unique_bytes is None or
                              model_state['unique_storage_bytes'] == expected_unique_bytes),
        attribution_rule=report['attribution_rule'], components=component_rows,
        top_components=component_rows[:20],
        top_storage_records=report['top_storage_records'],
        top_tensor_references=report['top_tensor_references'],
        layer_coverage=dict(static_remaining_state_module_count=len(report['modules']),
            static_remaining_layer_count=sum(len(value['remaining_modules']) for value in layer_counts.values()),
            components={key: dict(remaining_layers=len(value['remaining_modules']),
                                  quantized_target_layers=len({_layer_id(module) for module in value['quantized_targets']}),
                                  quantized_target_modules=len(value['quantized_targets']))
                        for key, value in sorted(layer_counts.items())},
            runtime_execution_coverage='not_recorded; static names do not establish execution or unused status'),
        quantization_boundary=dict(target_quantized_module_count=len(bundle.get('quantized', {})),
            excluded_dense_weight_key_count=len(excluded),
            excluded_keys_retained_in_model_state=sum(key in state for key in excluded),
            remaining_state='model_state keys absent from excluded_model_state_keys; tensor storage described below',
            execution='BF16 GEMM on VQ-decoded weights; no native low-bit kernel'),
        dtype_boundary=dict(unique_storage_bytes_by_owner_dtype=report['unique_storage_bytes_by_owner_dtype'],
            tensor_key_count_by_dtype=report['tensor_key_count_by_dtype'],
            all_tensor_keys_bfloat16=(set(report['tensor_key_count_by_dtype']) == {'torch.bfloat16'}),
            claim='Dtype is observed from tensor descriptors; state_dict can include persistent buffers as well as parameters.'),
        residency_scope='Unique backing storage in the CPU-mapped inference bundle model_state; not a CUDA allocator measurement.',
        protocol=None, bundle_path=None)
    details = dict(modules=report['modules'], storages=report['storages'])
    return {'summary': summary, 'details': details}


def install_runtime_probe(runtime):
    """Observe direct owners of remaining named parameters during existing forwards.

    The probe excludes the 614 quantized Linear modules and does not call forward.
    Call ``finalize_runtime_probe`` after the already-authorized anchor/profile.
    """
    model = runtime.model
    quantized = getattr(getattr(runtime, 'q', None), 'modules', {})
    quantized_ids = {id(module) for module in quantized.values()} if hasattr(quantized, 'values') else set()
    parameter_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    module_rows = {}
    for path, module in model.named_modules(remove_duplicate=False):
        entry = module_rows.setdefault(id(module), dict(module_paths=set(), direct_parameter_aliases=set(), module=module))
        entry['module_paths'].add(path or '<root>')
        if id(module) in quantized_ids:
            continue
        for local_name, _parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            key = f'{path}.{local_name}' if path else local_name
            if key in parameter_names:
                entry['direct_parameter_aliases'].add(key)
    records = []
    handles = []
    for entry in module_rows.values():
        if not entry['direct_parameter_aliases']:
            continue
        paths = sorted(entry['module_paths'])
        aliases = sorted(entry['direct_parameter_aliases'])
        components = sorted({classify_component(key)[0] for key in aliases})
        record = dict(module_paths=paths, direct_parameter_aliases=aliases,
                      components=components, calls=0)
        module = entry['module']
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, _output, item=record: item.__setitem__('calls', item['calls'] + 1)))
        records.append(record)
    return dict(handles=handles, records=records, target_modules_excluded=len(quantized_ids), closed=False)


def finalize_runtime_probe(probe) -> dict:
    """Remove hooks and return call/owner evidence without inferring unused weights."""
    if not probe.get('closed'):
        for handle in probe['handles']:
            handle.remove()
        probe['closed'] = True
    records = sorted((dict(row) for row in probe['records']),
                     key=lambda row: tuple(row['module_paths']))
    executed = [row for row in records if row['calls']]
    by_component = defaultdict(lambda: dict(owner_module_count=0, executed_owner_module_count=0,
                                            calls=0, direct_parameter_alias_count=0))
    for row in records:
        for component in row['components']:
            target = by_component[component]
            target['owner_module_count'] += 1
            target['executed_owner_module_count'] += int(row['calls'] > 0)
            target['calls'] += row['calls']
            target['direct_parameter_alias_count'] += len(row['direct_parameter_aliases'])
    return dict(status='captured', evidence='forward hooks on direct owners of remaining named parameters; no new forward was run',
        coverage_scope='candidate anchor and 3 warmups including graph capture/warmup; no extra forward',
        graph_replay_note='Python forward hooks count eager/capture execution only; CUDA graph replays do not trigger these hooks.',
        quantized_target_modules_excluded=probe['target_modules_excluded'],
        owner_module_count=len(records), executed_owner_module_count=len(executed),
        total_owner_module_calls=sum(row['calls'] for row in records),
        by_component={key: value for key, value in sorted(by_component.items())},
        executed_modules=executed,
        unobserved_owner_modules=[row for row in records if not row['calls']],
        interpretation='Unobserved during this bounded call window is not evidence that a parameter is unused.')


def _find_bundle_path(protocol: dict, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    wanted = {'inference_bundle', 'inference_bundle_path', 'bundle_path'}
    pending = [protocol]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in wanted:
                    if isinstance(child, str):
                        return Path(child)
                    if isinstance(child, dict):
                        for nested in ('path', 'file'):
                            if isinstance(child.get(nested), str):
                                return Path(child[nested])
                pending.append(child)
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return DEFAULT_BUNDLE


def _write_json(path: Path, payload: dict, max_bytes: int = 256 * 1024) -> int:
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    if len(data) >= max_bytes:
        raise ValueError(f'Report file exceeds {max_bytes} bytes: {path.name} ({len(data)} bytes)')
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)
    return len(data)


def write_runtime_probe(out, report: dict) -> dict:
    """Write complete owner-call details as bounded shards and return a compact receipt."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    owners = ([dict(status='executed', **row) for row in report.get('executed_modules', ())]
              + [dict(status='unobserved', **row) for row in report.get('unobserved_owner_modules', ())])
    owners.sort(key=lambda row: (tuple(row.get('module_paths', ())), row['status']))
    chunks = []
    current = []
    def shard_size(rows):
        # Reserve digits for the final part and count fields while chunking.
        payload = dict(schema_version=1, part=999999, part_count=999999, owners=rows)
        return len((json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    for owner in owners:
        candidate = current + [owner]
        if shard_size(candidate) >= 250_000:
            if not current:
                raise ValueError('One parameter owner record exceeds 250000 bytes')
            chunks.append(current)
            current = [owner]
            if shard_size(current) >= 250_000:
                raise ValueError('One parameter owner record exceeds 250000 bytes')
        else:
            current = candidate
    if current:
        chunks.append(current)

    files = []
    for part, rows in enumerate(chunks, 1):
        name = f'parameter_execution_{part:03d}.json'
        payload = dict(schema_version=1, part=part, part_count=len(chunks), owners=rows)
        byte_count = _write_json(out / name, payload, max_bytes=250_000)
        files.append(dict(path=name, owner_count=len(rows),
            executed_owner_count=sum(row['status'] == 'executed' for row in rows),
            unobserved_owner_count=sum(row['status'] == 'unobserved' for row in rows),
            bytes=byte_count))

    ranked = sorted(owners, key=lambda row: (-row.get('calls', 0), tuple(row.get('module_paths', ()))))[:20]
    compact = dict(schema_version=1, status=report.get('status'), evidence=report.get('evidence'),
        coverage_scope=report.get('coverage_scope'), graph_replay_note=report.get('graph_replay_note'),
        interpretation=report.get('interpretation'),
        quantized_target_modules_excluded=report.get('quantized_target_modules_excluded'),
        owner_module_count=report.get('owner_module_count'),
        executed_owner_module_count=report.get('executed_owner_module_count'),
        total_owner_module_calls=report.get('total_owner_module_calls'),
        by_component=report.get('by_component', {}),
        top_owners_by_calls=[dict(status=row['status'], module_paths=row['module_paths'],
            components=row['components'], calls=row['calls'],
            direct_parameter_alias_count=len(row['direct_parameter_aliases'])) for row in ranked],
        detail_files=files)
    _write_json(out / 'parameter_execution.json', compact, max_bytes=250_000)
    return compact


def write_inventory(results_dir, inspection: dict) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(inspection['summary'])
    details = inspection['details']
    files = []
    components = sorted({row['component'] for row in details['modules']} |
                        {row['owner_component'] for row in details['storages']})
    modules_by_component = defaultdict(list)
    storages_by_component = defaultdict(list)
    for row in details['modules']:
        modules_by_component[row['component']].append(row)
    for row in details['storages']:
        storages_by_component[row['owner_component']].append(row)
    for component in components:
        safe = re.sub(r'[^a-zA-Z0-9_-]+', '_', component)
        module_rows = modules_by_component[component]
        storage_rows = storages_by_component[component]
        chunks = []
        current = {'component': component, 'modules': [], 'storages': []}
        def flush():
            nonlocal current
            if current['modules'] or current['storages']:
                chunks.append(current)
            current = {'component': component, 'modules': [], 'storages': []}
        for kind, rows in (('modules', module_rows), ('storages', storage_rows)):
            for row in rows:
                current[kind].append(row)
                size = len((json.dumps(current, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
                if size >= MAX_REPORT_BYTES:
                    current[kind].pop()
                    flush()
                    current[kind].append(row)
                    if len((json.dumps(current, ensure_ascii=False, indent=2) + '\n').encode('utf-8')) >= MAX_REPORT_BYTES:
                        raise ValueError(f'One {kind} record exceeds 256 KiB for {component}')
        flush()
        for part, payload in enumerate(chunks, 1):
            name = f'storage_inventory_modules_{safe}_{part:03d}.json'
            byte_count = _write_json(results_dir / name, payload)
            files.append(dict(component=component, path=name, part=part,
                module_count=len(payload['modules']), storage_count=len(payload['storages']), bytes=byte_count))
    _write_json(results_dir / 'storage_inventory_modules.json',
        dict(schema_version=1, item_count=len(details['modules']), storage_count=len(details['storages']),
             components=summary['components'], files=files,
             meaning='Complete module aggregates and every backing-storage alias group are in the listed component files.'))
    summary['module_detail_index'] = 'storage_inventory_modules.json'
    summary['module_detail_file_count'] = len(files)
    summary_bytes = _write_json(results_dir / 'storage_inventory.json', summary)
    md = _render_markdown(summary)
    md_bytes = len(md.encode('utf-8'))
    if md_bytes >= 256 * 1024:
        raise ValueError(f'inventory.md exceeds 256 KiB ({md_bytes} bytes)')
    (results_dir / 'inventory.md').write_text(md, encoding='utf-8')
    return dict(storage_inventory_bytes=summary_bytes, inventory_md_bytes=md_bytes,
        module_detail_files=files)


def _render_markdown(summary: dict) -> str:
    state = summary['model_state']
    gib = 1024 ** 3
    lines = [
        '# C80 R128 remaining model-state inventory', '',
        f"- Phase: `{summary['phase']}`; bundle source: `{summary.get('source_job')}`.",
        f"- State tensor keys: {state['tensor_key_count']:,}; logical tensor bytes across key references: {state['logical_tensor_bytes']:,}.",
        f"- Unique resident backing storage: {state['unique_storage_bytes']:,} bytes ({state['unique_storage_bytes'] / gib:.6f} GiB) across {state['unique_storage_count']:,} storages.",
        f"- Repeated key references beyond one per storage: {state['alias_extra_reference_count']:,}; aliases/views are not extra resident storage.",
        f"- Expected unique bytes `{summary['expected_unique_storage_bytes']}`: match = `{summary['expected_bytes_match']}`.",
        '', '## Largest remaining components', '',
        '| Component | Unique storage bytes | Tensor keys | Modules | Quantized target modules | Purpose evidence |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for row in summary['top_components']:
        lines.append(f"| {row['component']} | {row['unique_storage_bytes']:,} | {row['tensor_key_count']:,} | {row['module_count']:,} | {row.get('quantized_target_module_count', 0):,} | {row['purpose_hint']} |")
    lines += ['', '## Quantization and execution boundary', '',
        f"- `{summary['quantization_boundary']['target_quantized_module_count']}` target modules are represented in the VQ payload; `{summary['quantization_boundary']['excluded_dense_weight_key_count']}` dense state keys are excluded from `model_state`.",
        '- This inventory describes the remaining `model_state` storage. Component purposes are conservative name-path classifications; `unclassified` entries stay unassigned rather than being called text-encoder weights.',
        '- Static inventory does not establish module execution. The optional runtime probe records calls only when attached to an already-authorized forward; unobserved modules are not labeled unused.',
        f"- Full module and alias groups: `{summary.get('module_detail_index')}`.", '']
    lines += ['## Largest backing storages', '',
        '| Storage bytes | Owner state key | Alias refs | View present |',
        '|---:|---|---:|---|']
    for row in summary['top_storage_records'][:12]:
        lines.append(f"| {row['storage_bytes']:,} | `{row['owner_state_key']}` | {row['tensor_reference_count']} | {row['has_tensor_view']} |")
    lines += ['', '## Largest component candidates for memory work', '',
        'These are resident storage opportunities by size. Any lower precision or offload change still needs parity and latency checks.', '']
    for row in summary['top_components'][:5]:
        lines.append(f"- `{row['component']}`: {row['unique_storage_bytes']:,} unique bytes; {row['purpose_hint']}")
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', help='Inference bundle path; defaults to the path in ARTIFACTS/protocol.json or the frozen profile path.')
    parser.add_argument('--phase', choices=('pre', 'post'), default='pre')
    args = parser.parse_args(argv)
    export_runtime.compute_guard()
    artifacts = Path(os.environ['ARTIFACTS'])
    protocol_path = artifacts / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    results_value = os.environ.get('RESULTS')
    if results_value:
        results = Path(results_value)
    elif protocol.get('remote_root'):
        results = Path(protocol['remote_root']) / 'results'
    else:
        raise RuntimeError('RESULTS is unset and protocol.json has no remote_root')
    bundle_path = _find_bundle_path(protocol, args.bundle)
    bundle = export_runtime.load_bundle(bundle_path)
    inspection = inspect_bundle(bundle)
    summary = inspection['summary']
    summary['phase'] = args.phase
    summary['pbs_jobid'] = os.environ['PBS_JOBID']
    summary['protocol'] = protocol.get('protocol')
    summary['bundle_path'] = str(bundle_path)
    summary['source_protocol_path'] = str(protocol_path)
    manifest_path = bundle_path.parent / 'storage.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        state_manifest = manifest.get('categories', {}).get('model_state', {})
        parameter_manifest = manifest.get('unquantized_model_parameters', {})
        summary['storage_manifest_validation'] = dict(path=str(manifest_path),
            model_state_unique_storage_bytes=state_manifest.get('unique_storage_bytes'),
            model_state_match=(state_manifest.get('unique_storage_bytes') ==
                               summary['model_state']['unique_storage_bytes']),
            unquantized_parameter_unique_storage_bytes=parameter_manifest.get('unique_storage_bytes'),
            unquantized_parameter_dtypes=parameter_manifest.get('unique_storage_bytes_by_first_dtype'),
            all_unquantized_parameter_storage_bfloat16=(
                set(parameter_manifest.get('unique_storage_bytes_by_first_dtype', {})) == {'torch.bfloat16'}))
    write_inventory(results, inspection)
    if not summary['expected_bytes_match']:
        raise SystemExit(f"Bundle unique storage mismatch: {summary['model_state']['unique_storage_bytes']} bytes")
    print(f"Inventory complete: {summary['model_state']['unique_storage_bytes']} unique model_state bytes; "
          f"{len(summary['components'])} components; outputs in {results}")


if __name__ == '__main__':
    main()
