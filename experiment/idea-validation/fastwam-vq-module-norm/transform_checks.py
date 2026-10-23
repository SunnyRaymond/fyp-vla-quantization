"""Small GPU equivalence fixtures and isolated timing, inside PBS only."""
from __future__ import annotations

import argparse
from pathlib import Path
import time


def run(out):
    import runner
    runner.guard()
    import torch
    import smooth_vq as old
    import batched_transform as new

    device = torch.device('cuda:0')
    generator = torch.Generator(device=device).manual_seed(20261012)

    def transform(width, hadamard=True, block=128):
        return {'input_dim': width, 'hadamard': hadamard, 'block': block,
                'scale': torch.linspace(1 / 16, 16, width, device=device),
                'signs': (torch.arange(width, device=device) % 2 * 2 - 1).float()}

    def operand(z):
        scale = z.abs().amax(dim=-1, keepdim=True) / 7
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(z / safe).clamp(-7, 7) * scale).to(torch.bfloat16)

    checked, maximum = 0, 0.0
    with torch.inference_mode():
        for width in (8, 127, 128, 129, 256, 257, 512, 768, 1024, 2048, 4096, 8192):
            t = transform(width)
            for dtype in (torch.float32, torch.bfloat16):
                for shape in ((width,), (3, width), (2, 3, width)):
                    storage = torch.randn(*shape[:-1], width * 2, device=device,
                                          generator=generator, dtype=dtype)
                    for value in (storage[..., ::2], storage[..., ::2].contiguous(),
                                  torch.zeros(shape, dtype=dtype, device=device)):
                        expected, actual = old.transform_input(value, t), new.transform_input(value, t)
                        maximum = max(maximum, float((expected - actual).abs().max()))
                        if not torch.equal(expected.view(torch.int32), actual.view(torch.int32)):
                            raise RuntimeError(f'FP32 transform is not bitwise equal: {width}/{dtype}/{shape}')
                        if not torch.equal(operand(expected).view(torch.int16), operand(actual).view(torch.int16)):
                            raise RuntimeError(f'A4 BF16 operand is not bitwise equal: {width}/{dtype}/{shape}')
                        old_inverse = old._right_r_inverse(value, t['signs'], t['hadamard'], t['block'])
                        new_inverse = new.inverse_right_r(value, t['signs'], t['hadamard'], t['block'])
                        if not torch.equal(old_inverse.view(torch.int32), new_inverse.view(torch.int32)):
                            raise RuntimeError(f'Inverse rotation is not bitwise equal: {width}/{dtype}/{shape}')
                        checked += 1
        for hadamard, block in ((False, 128), (True, 64)):
            value = torch.randn(3, 257, device=device, generator=generator)
            t = transform(257, hadamard=hadamard, block=block)
            if not torch.equal(old.transform_input(value, t).view(torch.int32),
                               new.transform_input(value, t).view(torch.int32)):
                raise RuntimeError('Fallback transform differs')
            checked += 1

        timing = []
        repeats = 8
        for tokens in (32, 512):
            value = torch.randn(1, tokens, 4096, device=device, generator=generator, dtype=torch.bfloat16)
            t = transform(4096)
            samples = {}
            for label, implementation in (('source_per_block_v1', old), (new.IMPLEMENTATION, new)):
                for _ in range(2):
                    implementation.transform_input(value, t)
                torch.cuda.synchronize(device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                for _ in range(repeats):
                    implementation.transform_input(value, t)
                end.record()
                torch.cuda.synchronize(device)
                samples[label] = {'wall_ms_per_call': (time.perf_counter() - wall) * 1000 / repeats,
                                  'cuda_interval_ms_per_call': start.elapsed_time(end) / repeats}
            timing.append({'shape': list(value.shape), 'repeats': repeats, 'samples': samples,
                           'wall_speedup': samples['source_per_block_v1']['wall_ms_per_call'] /
                                           samples[new.IMPLEMENTATION]['wall_ms_per_call']})
    receipt = {'passed': True, 'bitwise_equal': True, 'implementation': new.IMPLEMENTATION,
               'fixtures': checked, 'forward_inverse_checked': True, 'max_abs': maximum, 'device': torch.cuda.get_device_name(0),
               'isolated_transform_timing': timing,
               'timing_scope': 'Isolated transform only; end-to-end action query is measured by the workers'}
    runner.save(out, receipt)
    print(f'TRANSFORM_CHECK passed=true bitwise_equal=true fixtures={checked} max_abs={maximum}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
