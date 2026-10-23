"""Explicit payload accounting and a round-trip-checked sparse-support BF16 overlay."""
import json
import struct
from pathlib import Path
import torch

MAGIC = b'FWOVL001'


def write_overlay(path, header, values, directions):
    encoded = json.dumps(header, sort_keys=True, separators=(',', ':')).encode('utf-8')
    with Path(path).open('wb') as stream:
        stream.write(MAGIC); stream.write(struct.pack('<I', len(encoded))); stream.write(encoded)
        for name, value in values.items():
            packed = value[directions[name]['valid_coordinates']].contiguous().to(torch.bfloat16)
            stream.write(packed.view(torch.uint8).numpy().tobytes())


def read_overlay(path, expected_header, directions, names):
    out = {}
    with Path(path).open('rb') as stream:
        if stream.read(8) != MAGIC:
            raise RuntimeError('Overlay magic differs')
        length = struct.unpack('<I', stream.read(4))[0]
        if not 0 < length <= 262144:
            raise RuntimeError('Unbounded overlay header')
        if json.loads(stream.read(length)) != expected_header:
            raise RuntimeError('Overlay metadata differs')
        for name, meta in zip(names, expected_header['modules']):
            valid = directions[name]['valid_coordinates']
            if int(valid.sum()) != meta['scalar_count']:
                raise RuntimeError('Overlay support differs')
            payload = stream.read(meta['scalar_count'] * 2)
            if len(payload) != meta['scalar_count'] * 2:
                raise RuntimeError('Truncated overlay')
            value = torch.zeros(valid.shape, dtype=torch.bfloat16)
            value[valid] = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
            out[name] = value
        if stream.read(1):
            raise RuntimeError('Trailing overlay payload')
    return out


def tree_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size(), dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        pairs = {str(k): tree_bytes(v) for k, v in value.items()}
        return sum(v[0] for v in pairs.values()), {k: v[1] for k, v in pairs.items()}
    if isinstance(value, (tuple, list)):
        pairs = [tree_bytes(v) for v in value]
        return sum(v[0] for v in pairs), [v[1] for v in pairs]
    return 0, value


def accounting(p, directions, probes, selection, out):
    names = list(directions['modules'])
    groups = {g['module']: g for g in selection['groups']}
    header = dict(format='full-support-bf16-overlay-v1', base='action-only-epoch2',
        position_rule='retain-base-indices; book0 membership; row-major valid coordinates',
        modules=[dict(module_index=groups[n]['module_index'], book=0,
            codewords=sorted(groups[n]['codewords']), shape=directions['modules'][n]['shape'],
            scalar_count=int(directions['modules'][n]['valid_coordinates'].sum())) for n in names])
    # Selectors must recover positions from retained VQ indices, with no hidden position table.
    for name in names:
        d = directions['modules'][name]
        bank = torch.load(Path(p['source_root']) / 'results/full/banks' / p['initial_arm'] /
                          f"{groups[name]['module_index']:04d}.pt", map_location='cpu', weights_only=True)
        ids = bank['encoded']['indices'].reshape(-1, 2)[:, 0].long()
        selected = torch.isin(ids, torch.tensor(groups[name]['codewords'])).nonzero().flatten()
        if not torch.equal(selected, d['flat_ids']):
            raise RuntimeError('Stored code selectors do not recover the full overlay support')
    overlay_costs = {}
    for label, values in probes['values'].items():
        if '_shared_' in label:
            overlay_costs[label] = dict(added_bytes=0, reason='Overwrite same-size existing BF16 codebooks')
            continue
        path = out / f'{label}.overlay.bin'
        write_overlay(path, header, values, directions['modules'])
        restored = read_overlay(path, header, directions['modules'], names)
        if not all(torch.equal(restored[n], values[n]) for n in names):
            raise RuntimeError('BF16 deployment overlay does not round-trip exactly')
        overlay_costs[label] = dict(added_bytes=path.stat().st_size,
            payload_bytes=sum(v['scalar_count'] * 2 for v in header['modules']),
            metadata_bytes=path.stat().st_size - sum(v['scalar_count'] * 2 for v in header['modules']),
            selector_positions_verified=True, round_trip_bitwise_equal=True)
    total_weights = selected_module_weights = encoded_bytes = transform_bytes = metadata_bytes = 0
    components = dict(indices=0, books=0, row_scales=0, other=0)
    for i in range(614):
        bank = torch.load(Path(p['source_root']) / 'results/full/banks' / p['initial_arm'] / f'{i:04d}.pt',
                          map_location='cpu', weights_only=True)
        enc = dict(bank['encoded'])
        # The deployed A checkpoint stores books/scales as BF16, irrespective of optimizer master dtype.
        enc['books'] = enc['books'].to(torch.bfloat16)
        enc['row_scales'] = enc['row_scales'].to(torch.bfloat16)
        if enc['indices'].dtype != torch.uint8:
            raise RuntimeError('Unexpected VQ index representation')
        weights = int(enc['shape'][0]) * int(enc['shape'][1])
        total_weights += weights
        if bank['module'] in names:
            selected_module_weights += weights
        for key, value in enc.items():
            size, _ = tree_bytes(value)
            components[key if key in components else 'other'] += size
        size, description = tree_bytes(enc); encoded_bytes += size
        tsize, tdesc = tree_bytes(bank['transform']); transform_bytes += tsize
        metadata_bytes += len(json.dumps(dict(module=bank['module'], encoded=description,
            transform=tdesc), sort_keys=True, separators=(',', ':')).encode('utf-8')) + 4
        if i % 100 == 0:
            print('STORAGE_BANK', i, flush=True)
    baseline = encoded_bytes + transform_bytes + metadata_bytes
    support_weights = sum(v['scalar_count'] for v in header['modules'])
    for value in overlay_costs.values():
        extra = value['added_bytes']
        value.update(added_MiB=extra / 2**20, added_bpw_full614=8 * extra / total_weights,
            added_bpw_selected_modules=8 * extra / selected_module_weights,
            added_bpw_support=8 * extra / support_weights,
            total_effective_bpw_full614=8 * (baseline + extra) / total_weights)
    return dict(target_linears=614, total_target_weight_count=total_weights,
        selected_module_weight_count=selected_module_weights, support_weight_count=support_weights,
        base_encoded_tensor_bytes=encoded_bytes, base_transform_tensor_bytes=transform_bytes,
        base_explicit_metadata_bytes=metadata_bytes, base_total_bytes=baseline,
        base_tensor_components=components, base_effective_bpw=8 * baseline / total_weights,
        overlay_format=header, arms=overlay_costs,
        scope='Explicit packed target-weight representation; base indices retained for support recovery. '
              'Non-target policy weights/biases, training/master states and torch.save container excluded. '
              'No runtime VRAM, compressed-resident kernel or latency inference.')
