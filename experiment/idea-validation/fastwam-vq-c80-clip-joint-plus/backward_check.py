"""Compare the original and batched VQ backward only inside a real GPU allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

# calibrate.pbs verifies qstat before entering the container, which has no PBS CLI.
from calibrate import guard


def original_backward(torch, ctx, grad):
    """Original 32-row implementation, retained only as the GPU comparison reference."""
    books, log_scales, active, indices, exponents = ctx.saved_tensors
    out_width, in_width = ctx.shape
    stored_books = books.to(torch.bfloat16).float()
    raw_gains = log_scales.exp() * active
    stored_gains = raw_gains.to(torch.bfloat16).float()
    book_grad = torch.zeros_like(books)
    scale_grad = torch.zeros_like(log_scales)
    padded = indices.shape[1] * 4
    group_index = torch.arange(indices.shape[1], device=indices.device) // 8
    for start in range(0, out_width, 32):
        end = min(start + 32, out_width)
        exponent = exponents[start:end, group_index].to(torch.int32) - 127
        stored_scales = torch.ldexp(stored_gains[start:end, None], exponent)
        raw_scales = torch.ldexp(raw_gains[start:end, None], exponent)
        g = torch.nn.functional.pad(grad[start:end].float(), (0, padded - in_width))
        g = g.reshape(end - start, -1, 4)
        idx = indices[start:end].long()
        values = stored_books[0][idx[..., 0]] + stored_books[1][idx[..., 1]]
        scale_grad[start:end] = (g * values * raw_scales[..., None]).sum((1, 2))
        weighted = (g * stored_scales[..., None]).reshape(-1, 4)
        for book in range(2):
            book_grad[book].index_add_(0, idx[..., book].reshape(-1), weighted)
    return book_grad, scale_grad, None, None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    guard()
    import torch
    import train_banks

    if (train_banks.BACKWARD_IMPLEMENTATION != 'bounded_vector_chunks_v2'
            or train_banks.BACKWARD_MAX_VECTORS != 1048576):
        raise RuntimeError('Unexpected weight backward implementation')
    torch.manual_seed(20261017)
    cases, timings = [], []

    def inputs(rows, width, dtype, noncontiguous=False):
        vectors = (width + 3) // 4
        books = torch.randn(2, 256, 4, device='cuda') * 0.1
        logs = torch.randn(rows, device='cuda') * 0.3
        active = torch.ones(rows, device='cuda')
        active[::7] = 0
        indices = torch.randint(0, 256, (rows, vectors, 2), device='cuda', dtype=torch.uint8)
        exponents = torch.randint(120, 130, (rows, (vectors + 7) // 8),
                                  device='cuda', dtype=torch.uint8)
        grad = torch.randn((width, rows) if noncontiguous else (rows, width),
                           device='cuda', dtype=dtype)
        if noncontiguous:
            grad = grad.T
        return SimpleNamespace(saved_tensors=(books, logs, active, indices, exponents),
                               shape=(rows, width)), grad

    def compare(label, ctx, grad):
        old = original_backward(torch, ctx, grad)
        new = train_banks.DecodedWeight.backward(ctx, grad)
        errors = {}
        for name, actual, expected in zip(('codebooks', 'log_row_gains'), new[:2], old[:2]):
            delta = actual - expected
            relative = float(torch.linalg.vector_norm(delta)
                             / torch.linalg.vector_norm(expected).clamp_min(1e-12))
            scaled_max = float(delta.abs().max() / expected.abs().max().clamp_min(1))
            finite = bool(torch.isfinite(actual).all())
            errors[name] = {'relative_l2': relative, 'scaled_max_abs': scaled_max,
                            'max_abs': float(delta.abs().max()), 'finite': finite,
                            'passed': finite and relative <= 5e-5 and scaled_max <= 2e-4}
        result = {'case': label, 'shape': list(ctx.shape), 'grad_dtype': str(grad.dtype),
                  'errors': errors, 'passed': all(value['passed'] for value in errors.values())}
        cases.append(result)
        del old, new

    for rows, width, dtype, nc in ((37, 135, torch.float32, True),
                                   (64, 128, torch.bfloat16, False),
                                   (2048, 2048, torch.bfloat16, False),
                                   (8192, 2048, torch.bfloat16, False),
                                   (2048, 8192, torch.bfloat16, False)):
        ctx, grad = inputs(rows, width, dtype, nc)
        compare(f'{rows}x{width}', ctx, grad)
        if rows >= 2048:
            measured = {}
            for name, function in (('original_32_rows', lambda: original_backward(torch, ctx, grad)),
                                   ('bounded_vector_chunks_v2', lambda: train_banks.DecodedWeight.backward(ctx, grad))):
                for _ in range(2):
                    output = function()
                    del output
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = function()
                    end.record()
                    end.synchronize()
                    elapsed.append(start.elapsed_time(end))
                    del output
                measured[name] = {'median_ms': statistics.median(elapsed), 'samples_ms': elapsed,
                                  'extra_peak_allocated_bytes': torch.cuda.max_memory_allocated() - baseline}
            measured.update(shape=[rows, width], speedup=measured['original_32_rows']['median_ms']
                            / measured['bounded_vector_chunks_v2']['median_ms'])
            timings.append(measured)
        del ctx, grad
        torch.cuda.empty_cache()

    correct = all(case['passed'] for case in cases)
    memory_safe = all(row['bounded_vector_chunks_v2']['extra_peak_allocated_bytes'] <= 256 * 1024**2
                      for row in timings)
    improved = (sum(row['bounded_vector_chunks_v2']['median_ms'] for row in timings)
                < sum(row['original_32_rows']['median_ms'] for row in timings))
    result = {'schema_version': 1, 'pbs_jobid': os.environ['PBS_JOBID'],
              'implementation': train_banks.BACKWARD_IMPLEMENTATION,
              'max_vectors_per_chunk': train_banks.BACKWARD_MAX_VECTORS,
              'device': torch.cuda.get_device_name(), 'cases': cases, 'timings': timings,
              'gradient_checks_passed': correct, 'memory_safe': memory_safe,
              'performance_improved': improved, 'passed': correct and memory_safe and improved,
              'tolerances': {'relative_l2': 5e-5, 'scaled_max_abs': 2e-4},
              'scope': 'isolated DecodedWeight backward on synthetic representative shapes; '
                       'FP32 accumulation order can differ; no end-to-end speedup claim'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + '.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(args.out)
    print('DECODED_WEIGHT_BACKWARD_CHECK ' + json.dumps(result, separators=(',', ':')), flush=True)
    if not result['passed']:
        raise RuntimeError('Weight backward accuracy, performance or memory gate failed')


if __name__ == '__main__':
    main()
