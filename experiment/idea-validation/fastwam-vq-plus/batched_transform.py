"""Batch independent H128 butterflies without changing the frozen transform."""
from __future__ import annotations

import smooth_vq as core
import torch

IMPLEMENTATION = 'batched_h128_v2'


def transform_input(value, transform):
    width = value.shape[-1]
    if width != transform['input_dim']:
        raise ValueError('input width does not match transform')
    if not transform['hadamard'] or transform['block'] != 128:
        return core.transform_input(value, transform)
    scaled = value.float() / transform['scale']
    signs = transform['signs']
    full = width // 128 * 128
    pieces = []
    if full:
        blocks = (scaled[..., :full] * signs[:full]).reshape(*value.shape[:-1], full // 128, 128)
        pieces.append(core._fwht(blocks).reshape(*value.shape[:-1], full))
    if full < width:
        pieces.append(core._right_r(scaled[..., full:], signs[full:], True, 128))
    if not pieces:
        return scaled
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
