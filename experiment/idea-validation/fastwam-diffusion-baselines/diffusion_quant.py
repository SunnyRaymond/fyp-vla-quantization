"""Budgeted SVDQuant and ViDiT-Q reference arithmetic; Torch imports are lazy."""
from __future__ import annotations

import math

ARMS = ('svdquant_w4a4', 'viditq_w4a4')
IMPLEMENTATIONS = {ARMS[0]: 'svdquant_smoothed_low_rank_v1',
                   ARMS[1]: 'viditq_structured_hadamard_v1'}


def is_power_two(value):
    return value > 0 and value & (value - 1) == 0


def rotation_layout(width):
    if is_power_two(width):
        return 'full_power2'
    if width % 12 == 0 and is_power_two(width // 12):
        return 'full_h12_power2'
    return 'dyadic_tail'


def estimate_storage(shapes, rank, group=128):
    """Count the actual padded representation, including the A16 branch."""
    totals = {arm: dict(packed=0, scales=0, zero_points=0, smoothing=0,
                        rotation_signs=0, low_rank_factors=0) for arm in ARMS}
    parameters = 0
    for rows, columns in shapes:
        parameters += rows * columns
        groups = rows * math.ceil(columns / group)
        for arm in ARMS:
            totals[arm]['packed'] += groups * group // 2
            totals[arm]['scales'] += groups * 4
            totals[arm]['smoothing'] += columns * 4
        totals[ARMS[1]]['zero_points'] += groups
        totals[ARMS[1]]['rotation_signs'] += columns
        totals[ARMS[0]]['low_rank_factors'] += 2 * min(rank, rows, columns) * (rows + columns)
    return {arm: {'bytes_by_kind': parts, 'encoded_bytes': sum(parts.values()),
                  'parameters': parameters, 'bpw': 8 * sum(parts.values()) / parameters}
            for arm, parts in totals.items()}


def select_rank(shapes, protocol):
    quant = protocol['quantization']
    for rank in protocol['methods'][ARMS[0]]['rank_candidates_descending']:
        costs = estimate_storage(shapes, rank, quant['group_size'])
        first, second = (costs[arm]['bpw'] for arm in ARMS)
        gap = abs(first - second) / second
        if (max(first, second) <= quant['maximum_target_bpw']
                and gap <= quant['max_relative_target_bpw_gap']):
            return {'global_rank': rank, 'costs': costs, 'relative_bpw_gap': gap}
    raise RuntimeError('No nonzero SVD rank meets the frozen representation budget')


def fwht(value):
    import smooth_vq
    return smooth_vq._fwht(value)


_H12 = [
    [1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,1,-1,1,-1,-1,-1,1,1,1,-1,1],
    [1,1,1,-1,1,-1,-1,-1,1,1,1,-1],
    [1,-1,1,1,-1,1,-1,-1,-1,1,1,1],
    [1,1,-1,1,1,-1,1,-1,-1,-1,1,1],
    [1,1,1,-1,1,1,-1,1,-1,-1,-1,1],
    [1,1,1,1,-1,1,1,-1,1,-1,-1,-1],
    [1,-1,1,1,1,-1,1,1,-1,1,-1,-1],
    [1,-1,-1,1,1,1,-1,1,1,-1,1,-1],
    [1,-1,-1,-1,1,1,1,-1,1,1,-1,1],
    [1,1,-1,-1,-1,1,1,1,-1,1,1,-1],
    [1,-1,1,-1,-1,-1,1,1,1,-1,1,1],
]
_H12_CACHE = {}


def rotate(value, signs, inverse=False):
    """x @ diag(signs) @ H without materializing a dense channel matrix."""
    import torch
    x = value.float()
    if not inverse:
        x = x * signs
    width = x.shape[-1]
    layout = rotation_layout(width)
    if layout == 'full_power2':
        result = fwht(x)
    elif layout == 'full_h12_power2':
        key = str(x.device)
        if key not in _H12_CACHE:
            _H12_CACHE[key] = torch.tensor(_H12, device=x.device, dtype=torch.float32) / math.sqrt(12)
        h = _H12_CACHE[key]
        part = fwht(x.reshape(*x.shape[:-1], 12, width // 12))
        result = torch.nn.functional.linear(part.transpose(-1, -2), h if inverse else h.t())
        result = result.transpose(-1, -2).reshape_as(x)
    else:
        # A physical width such as proprio=7 has no real full Hadamard matrix.
        pieces, offset = [], 0
        while offset < width:
            span = 1 << ((width - offset).bit_length() - 1)
            pieces.append(fwht(x[..., offset:offset + span]))
            offset += span
        result = torch.cat(pieces, dim=-1)
    return result * signs if inverse else result


def quantize_weight(weight, group=128, affine=False):
    import torch
    import torch.nn.functional as functional
    rows, columns = weight.shape
    padded = math.ceil(columns / group) * group
    blocks = functional.pad(weight.float(), (0, padded - columns)).reshape(rows, -1, group)
    if affine:
        lower = blocks.amin(-1).clamp_max(0)
        upper = blocks.amax(-1).clamp_min(0)
        scales = (upper - lower) / 15
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        zeros = (-lower / safe).round().clamp(0, 15).to(torch.uint8)
        codes = (blocks / safe[..., None]).round().add(zeros[..., None]).clamp(0, 15).to(torch.uint8)
    else:
        scales = blocks.abs().amax(-1) / 7
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        codes = (blocks / safe[..., None]).round().clamp(-7, 7).add(8).to(torch.uint8)
        zeros = None
    codes = codes.reshape(rows, padded)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return {'packed': packed, 'scales': scales, 'zero_points': zeros,
            'shape': (rows, columns), 'group_size': group, 'affine': affine}


def decode_weight(encoded, dtype=None):
    import torch
    packed, scales = encoded['packed'], encoded['scales']
    rows, columns = encoded['shape']
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, -1)
    blocks = codes.float().reshape(rows, -1, encoded['group_size'])
    offset = encoded['zero_points'].float()[..., None] if encoded['affine'] else 8
    weight = ((blocks - offset) * scales[..., None]).reshape(rows, -1)[:, :columns]
    return weight.to(dtype or torch.bfloat16).contiguous()


def quantize_activation(value, affine=False):
    import torch
    x = value.float()
    if affine:
        lower = x.amin(-1, keepdim=True).clamp_max(0)
        upper = x.amax(-1, keepdim=True).clamp_min(0)
        scale = (upper - lower) / 15
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = (-lower / safe).round().clamp(0, 15)
        result = ((x / safe).round().add(zero).clamp(0, 15) - zero) * scale
    else:
        scale = x.abs().amax(-1, keepdim=True) / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        result = (x / safe).round().clamp(-7, 7) * scale
    return result.to(torch.bfloat16)


def fit_bank(weight, maximum, arm, protocol, rank, module_index):
    import torch
    seed = protocol['quantization']['seed'] + module_index * 17
    original = weight.detach().float()
    maximum = maximum.to(device=weight.device).clamp_min(1e-8)
    weight_max = original.abs().amax(0).clamp_min(1e-8)
    group = protocol['quantization']['group_size']
    bank = {'arm': arm, 'implementation': IMPLEMENTATIONS[arm]}
    if arm == ARMS[0]:
        alpha = protocol['methods'][arm]['smoothing_alpha']
        scale = maximum.pow(alpha) / weight_max.pow(1 - alpha)
        transformed = original * scale
        retained = min(rank, *weight.shape)
        config = protocol['methods'][arm]
        with torch.random.fork_rng(devices=[weight.device.index or 0]):
            torch.manual_seed(seed)
            u, singular, v = torch.svd_lowrank(
                transformed, q=min(retained + config['svd_oversampling'], *weight.shape),
                niter=config['svd_power_iterations'])
        root = singular[:retained].sqrt()
        up = (u[:, :retained] * root).to(torch.bfloat16)
        down = (root[:, None] * v[:, :retained].t()).to(torch.bfloat16)
        residual = transformed - up.float() @ down.float()
        bank.update(smoothing=scale, up=up, down=down, rank=retained,
                    encoded=quantize_weight(residual, group),
                    low_rank_energy_fraction=float(singular[:retained].square().sum()
                                                   / transformed.square().sum().clamp_min(1e-30)))
    else:
        alpha = protocol['methods'][arm]['alpha_code_convention']
        mask = weight_max.pow(alpha) / maximum.pow(1 - alpha)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        signs = (torch.randint(0, 2, (weight.shape[1],), generator=generator).to(
            device=weight.device, dtype=torch.int8) * 2 - 1)
        first = quantize_weight(original / mask, group, affine=True)
        prequantized = decode_weight(first, dtype=torch.float32)
        bank.update(smoothing=mask, signs=signs, rank=0,
                    rotation_layout=rotation_layout(weight.shape[1]),
                    encoded=quantize_weight(rotate(prequantized, signs), group, affine=True))
    if any(not bool(torch.isfinite(value).all()) for value in bank.values() if torch.is_tensor(value)):
        raise RuntimeError(f'Nonfinite prepared tensor in {arm}/{module_index}')
    return bank


def bank_bytes(bank):
    parts = {'packed': bank['encoded']['packed'], 'scales': bank['encoded']['scales'],
             'zero_points': bank['encoded']['zero_points'], 'smoothing': bank['smoothing'],
             'rotation_signs': bank.get('signs')}
    result = {key: value.numel() * value.element_size() if value is not None else 0
              for key, value in parts.items()}
    result['low_rank_factors'] = sum(bank[key].numel() * bank[key].element_size()
                                    for key in ('up', 'down') if key in bank)
    return result


def forward(value, weight, bias, bank):
    import torch.nn.functional as functional
    if bank['arm'] == ARMS[0]:
        normalized = value.float() / bank['smoothing']
        result = functional.linear(quantize_activation(normalized), weight, bias)
        low = functional.linear(normalized.to(bank['down'].dtype), bank['down'])
        return result + functional.linear(low, bank['up'])
    transformed = rotate(value.float() * bank['smoothing'], bank['signs'])
    return functional.linear(quantize_activation(transformed, affine=True), weight, bias)


def self_check(device='cuda'):
    """Small GPU numerical references; no model or test data enter this gate."""
    import torch
    generator = torch.Generator(device=device).manual_seed(271828)
    errors = {}
    for width in (7, 16, 24, 48, 128):
        x = torch.randn(3, width, device=device, generator=generator)
        signs = torch.where(torch.arange(width, device=device) % 2 == 0, 1, -1).to(torch.int8)
        dense = rotate(torch.eye(width, device=device), signs)
        error = max(float((rotate(x, signs) - x @ dense).abs().max()),
                    float((rotate(rotate(x, signs), signs, inverse=True) - x).abs().max()),
                    float((dense @ dense.t() - torch.eye(width, device=device)).abs().max()))
        if error > 2e-5:
            raise RuntimeError(f'Independent dense rotation reference failed: {width}/{error}')
        errors[str(width)] = error
    for affine in (False, True):
        w = torch.randn(17, 137, device=device, generator=generator)
        encoded = quantize_weight(w, affine=affine)
        decoded = decode_weight(encoded, dtype=torch.float32)
        grouped = torch.nn.functional.pad(w, (0, 119)).reshape(17, 2, 128)
        scale = encoded['scales'][..., None]
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        if affine:
            zero = encoded['zero_points'].float()[..., None]
            expected = ((grouped / safe).round().add(zero).clamp(0, 15) - zero) * scale
        else:
            expected = (grouped / safe).round().clamp(-7, 7) * scale
        if not torch.equal(decoded, expected.reshape(17, 256)[:, :137]):
            raise RuntimeError('Packed W4 roundtrip differs from independent arithmetic')
        zero_result = quantize_activation(torch.zeros(3, 7, device=device), affine)
        if not bool((zero_result == 0).all()):
            raise RuntimeError('Zero-valued activation control failed')
        x = torch.randn(3, 7, device=device, generator=generator)
        quantized = quantize_activation(x, affine)
        if not bool(torch.isfinite(quantized).all()) or torch.equal(quantized.float(), x):
            raise RuntimeError('A4 control is identity or nonfinite')
    x = torch.randn(4, 24, device=device, generator=generator).to(torch.bfloat16)
    w = torch.randn(32, 24, device=device, generator=generator).to(torch.bfloat16)
    bias = torch.randn(32, device=device, generator=generator).to(torch.bfloat16)
    config = {'quantization': {'seed': 123, 'group_size': 128},
              'methods': {ARMS[0]: {'smoothing_alpha': .5, 'svd_oversampling': 4,
                                    'svd_power_iterations': 2},
                          ARMS[1]: {'alpha_code_convention': .5665}}}
    for arm in ARMS:
        bank = fit_bank(w, x.float().abs().amax(0), arm, config, 4, 0)
        dense = decode_weight(bank['encoded'])
        actual = forward(x, dense, bias, bank)
        if arm == ARMS[0]:
            transformed = x.float() / bank['smoothing']
            scale = transformed.abs().amax(-1, keepdim=True) / 7
            operand = ((transformed / scale).round().clamp(-7, 7) * scale).to(torch.bfloat16)
            protected = torch.nn.functional.linear(
                torch.nn.functional.linear(transformed.to(torch.bfloat16), bank['down']), bank['up'])
            expected = torch.nn.functional.linear(operand, dense, bias) + protected
            if not bool((protected != 0).any()) or bank['up'].shape[1] != 4:
                raise RuntimeError('SVD high-precision branch control failed')
        else:
            matrix = rotate(torch.eye(24, device=device), bank['signs'])
            transformed = (x.float() * bank['smoothing']) @ matrix
            operand = quantize_activation(transformed, affine=True)
            expected = torch.nn.functional.linear(operand, dense, bias)
        error = float((actual.float() - expected.float()).abs().max())
        if error > .03125:
            raise RuntimeError(f'Method forward differs from independent dense reference: {arm}/{error}')
    return {'passed': True, 'rotation_dense_inverse_orthogonality_errors': errors,
            'packed_weight_independent_reference': True, 'a4_nonidentity_and_zero_checked': True,
            'svd_low_rank_branch_reference': True, 'vidit_dense_reference': True}
