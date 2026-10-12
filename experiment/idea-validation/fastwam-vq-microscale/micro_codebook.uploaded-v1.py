"""B32 power-of-two scales with the existing BF16, two-book d4 VQ fitter."""
from __future__ import annotations

import math

import torch
import codebook as base

FORMAT = 'fastwam_additive_vq_microscale_v1'
SCALE_BLOCK = 32


def block_scales(encoding):
    return torch.ldexp(encoding['row_scales'].float()[:, None],
                       encoding['scale_exponents'].to(torch.int32) - 127)


def scales_for_weight(weight):
    rows, width = base._check_weight(weight)
    values = weight.detach().float()
    row_scales = values.square().mean(1).sqrt().to(torch.bfloat16)
    padded = torch.nn.functional.pad(values, (0, (-width) % SCALE_BLOCK))
    counts = torch.full((padded.shape[1] // SCALE_BLOCK,), SCALE_BLOCK, device=values.device)
    counts[-1] = width - SCALE_BLOCK * (counts.numel() - 1)
    rms = (padded.reshape(rows, -1, SCALE_BLOCK).square().sum(-1) / counts).sqrt()
    safe_row = torch.where(row_scales.float() > 0, row_scales.float(), 1.)
    exponent = torch.round(torch.log2((rms / safe_row[:, None]).clamp_min(1e-12)))
    exponent = torch.where(row_scales[:, None] > 0, exponent, 0.)
    clipped = int(((exponent < -127) | (exponent > 127)).sum())
    stored = (exponent.clamp(-127, 127).to(torch.int32) + 127).to(torch.uint8)
    scales = torch.ldexp(row_scales.float()[:, None], stored.to(torch.int32) - 127)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError('Nonfinite decoded block scales')
    return row_scales, stored, scales, clipped


class MicroMetric(base._MetricView):
    """Same diagonal input metric, multiplied by each vector's actual S squared."""
    def __init__(self, weight, scales, second_moment):
        rows, width = weight.shape
        groups = (width + 3) // 4
        moment = second_moment.detach().to(device=weight.device, dtype=torch.float32)
        if moment.shape != (width,) or not bool(torch.isfinite(moment).all()) or bool((moment < 0).any()):
            raise ValueError('Expected finite nonnegative input second moment [in_features]')
        values = torch.nn.functional.pad(moment.clamp_min(1e-12), (0, groups * 4 - width))
        super().__init__('diag', values.reshape(1, groups, 4), rows, groups,
                         torch.ones(rows, device=weight.device))
        self.scales = scales
        trace = torch.zeros((), device=weight.device)
        for start in range(0, rows * groups, 16384):
            trace += self.chunk(start, min(start + 16384, rows * groups))[1].sum()
        normalizer = float(trace / (rows * width))
        if normalizer > 1e-12 and math.isfinite(normalizer):
            self.values = self.values / normalizer

    def _select(self, ids):
        rows = torch.div(ids, self.groups, rounding_mode='floor')
        groups = ids.remainder(self.groups)
        scale = self.scales[rows, torch.div(groups, SCALE_BLOCK // 4, rounding_mode='floor')]
        return 'diag', self.values[0, groups] * scale.square()[:, None]


def fit_vq(weight, *, second_moment, seed=20261006, sample_limit=8192, iterations=8):
    rows, width = base._check_weight(weight)
    if sample_limit < 1 or iterations < 2:
        raise ValueError('Invalid fitting budget')
    groups = (width + 3) // 4
    row_scales, exponents, scales, clipped = scales_for_weight(weight)
    expanded = scales.repeat_interleave(SCALE_BLOCK, dim=1)[:, :width]
    normalized = weight.detach().float() / torch.where(expanded > 0, expanded, 1.)
    vectors = torch.nn.functional.pad(normalized, (0, groups * 4 - width)).reshape(-1, 4).contiguous()
    metric = MicroMetric(weight, scales, second_moment)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    count = min(vectors.shape[0], sample_limit)
    ids = (torch.arange(vectors.shape[0], device=weight.device) if vectors.shape[0] <= sample_limit
           else torch.randint(vectors.shape[0], (count,), generator=generator).to(weight.device))
    sample, sample_metric = vectors[ids], metric.gather(ids)
    book0 = base._kmeans(sample, sample_metric, generator).to(torch.bfloat16).float()
    index0 = base._assign(sample, book0, sample_metric)
    residual = sample - book0[index0]
    book1 = base._kmeans(residual, sample_metric, generator).to(torch.bfloat16).float()
    index1 = base._assign(residual, book1, sample_metric)
    for _ in range(iterations):
        target = sample - book1[index1]
        index0 = base._assign(target, book0, sample_metric)
        book0 = base._update_centroids(target, index0, sample_metric, book0).to(torch.bfloat16).float()
        residual = sample - book0[index0]
        index1 = base._assign(residual, book1, sample_metric)
        book1 = base._update_centroids(residual, index1, sample_metric, book1).to(torch.bfloat16).float()
    books = torch.stack((book0.to(torch.bfloat16), book1.to(torch.bfloat16)))
    stored = books.float()
    index0 = base._assign(vectors, stored[0], metric)
    index1 = base._assign(vectors - stored[0][index0], stored[1], metric)
    for _ in range(2):
        index0 = base._assign(vectors - stored[1][index1], stored[0], metric)
        index1 = base._assign(vectors - stored[0][index0], stored[1], metric)
    return {'format': FORMAT, 'shape': (rows, width), 'block_size': 4, 'scale_block': SCALE_BLOCK,
            'books': books, 'row_scales': row_scales, 'scale_exponents': exponents,
            'indices': torch.stack((index0, index1), -1).reshape(rows, groups, 2).to(torch.uint8),
            'fit_stats': {'vectors': vectors.shape[0], 'sample_vectors': count, 'iterations': iterations,
                          'assignment_metric': 'input_second_moment_times_block_scale_squared',
                          'clipped_exponents': clipped}}


def check_encoding(encoding):
    if encoding.get('format') != FORMAT or encoding.get('scale_block') != 32 or encoding.get('block_size') != 4:
        raise ValueError('Unexpected encoding format')
    rows, width = encoding['shape']
    tensors = {key: encoding[key] for key in ('books', 'row_scales', 'indices', 'scale_exponents')}
    expected = {'books': ((2, 256, 4), torch.bfloat16), 'row_scales': ((rows,), torch.bfloat16),
                'indices': ((rows, (width + 3) // 4, 2), torch.uint8),
                'scale_exponents': ((rows, (width + 31) // 32), torch.uint8)}
    device = tensors['books'].device
    for key, (shape, dtype) in expected.items():
        tensor = tensors[key]
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
            raise ValueError(f'Invalid {key} shape/dtype/device')
    if bool((tensors['scale_exponents'] == 255).any()):
        raise ValueError('Reserved exponent encoding')
    if not bool(torch.isfinite(tensors['books']).all()) or not bool(torch.isfinite(tensors['row_scales']).all()):
        raise ValueError('Nonfinite book/row gain')
    if bool((tensors['row_scales'] < 0).any()):
        raise ValueError('Negative row gain')


def decode_vq(encoding):
    check_encoding(encoding)
    rows, width = encoding['shape']
    books, indices = encoding['books'].float(), encoding['indices'].long()
    vectors = books[0][indices[..., 0]] + books[1][indices[..., 1]]
    scales = block_scales(encoding).repeat_interleave(8, dim=1)[:, :indices.shape[1]]
    return (vectors * scales[..., None]).reshape(rows, -1)[:, :width].to(torch.bfloat16)


def encoded_nbytes(encoding):
    return sum(encoding[key].numel() * encoding[key].element_size()
               for key in ('books', 'row_scales', 'indices', 'scale_exponents'))


def self_check():
    device = torch.device('cuda')
    generator = torch.Generator(device='cpu').manual_seed(20261011)
    for width in (4, 32, 37):
        rows, groups = 3, (width + 3) // 4
        old = {'format': base.FORMAT, 'shape': (rows, width), 'block_size': 4,
               'books': torch.randn((2, 256, 4), generator=generator).to(device=device, dtype=torch.bfloat16),
               'row_scales': torch.tensor([0., .5, 2.], device=device, dtype=torch.bfloat16),
               'indices': torch.randint(256, (rows, groups, 2), generator=generator).to(device=device, dtype=torch.uint8)}
        micro = dict(old, format=FORMAT, scale_block=32,
                     scale_exponents=torch.full((rows, (width + 31) // 32), 127, device=device, dtype=torch.uint8))
        assert torch.equal(base.decode_vq(old), decode_vq(micro)), 'e0 decoder must match row-scale decoder'
        micro['scale_exponents'][1, 0] = 128
        assert torch.equal(decode_vq(micro)[1, :min(width, 32)], (base.decode_vq(old)[1, :min(width, 32)].float() * 2).to(torch.bfloat16))
    weight = torch.cat((torch.ones(1, 32), 4 * torch.ones(1, 32)), dim=1).to(device)
    row, exponents, scales, clipped = scales_for_weight(weight)
    assert exponents.tolist() == [[125, 127]] and clipped == 0
    metric = MicroMetric(weight, scales, torch.ones(64, device=device))
    first = metric.gather(torch.tensor([0, 8], device=device))[1]
    assert torch.allclose(first[1] / first[0], torch.full((4,), 16., device=device)), 'metric must include S squared'
    fit = fit_vq(torch.zeros((2, 5), device=device), second_moment=torch.ones(5, device=device),
                 sample_limit=16, iterations=2)
    assert torch.equal(decode_vq(fit), torch.zeros((2, 5), device=device, dtype=torch.bfloat16))
    print('MICROSCALE_SELF_CHECK_PASSED e0_parity tail zero metric_S2', flush=True)
