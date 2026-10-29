"""Phase-0 speed gate: real Linear shapes, INT8 vs BF16 GEMM, and three policy-call runtimes."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import statistics as stats
import time
from types import SimpleNamespace

from reference_profile import guard, memory, save, statistics

RUNTIME_KEYS = ('full_query_wall_ms', 'denoising_latency_ms', 'peak_allocated_gpu_GB',
                'peak_reserved_gpu_GB', 'baseline_allocated_gpu_GB')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stream_of(name):
    return name.split('.', 1)[0].replace('_expert', '').replace('_encoder', '')


def progress(p, mode, phase, **fields):
    row = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode, phase=phase,
               updated_epoch=time.time(), **fields)
    root = Path(p['remote_root']) / 'results'
    save(root / 'progress.json', row)
    save(root / mode / 'progress.json', row)
    print('SPEED_PROGRESS ' + json.dumps(row), flush=True)


class Panel:
    """The frozen held-out 80-query panel used by every historical runtime profile."""
    def __init__(self, p):
        self.settings = p['panel']
        self.plan_path = Path(self.settings['calibration_plan'])
        self.rows = {r['case_id']: r for r in read(self.plan_path)['inputs']}

    def datum(self, runtime, case):
        import numpy as np
        import torch
        import calibration_source
        torch.cuda.synchronize()
        begin = time.perf_counter()
        observation, metadata = calibration_source.load_observation(np, self.plan_path.parent, self.rows[case])
        runtime.description = metadata['description']
        value = runtime.datum(observation)
        torch.cuda.synchronize()
        return value, (time.perf_counter() - begin) * 1000

    def seed(self, case, index):
        return self.rows[case]['sampler_seeds'][index]


class CountingLinears:
    """Plain BF16 F.linear with the counters graph_runtime expects from engine banks."""
    def __init__(self, modules):
        self.modules, self.calls, self.seen = modules, Counter(), set()
        self.diagnostics, self.track, self.record = False, True, None
        for name, module in modules.items():
            module.forward = self._forward(name, module)

    def _forward(self, name, module):
        import torch.nn.functional as F

        def run(x):
            if self.track:
                self.calls[name] += 1
                self.seen.add(name)
                if self.record is not None:
                    self.record[(name, x.numel() // x.shape[-1], x.shape[-1],
                                 module.weight.shape[0], module.bias is not None)] += 1
            return F.linear(x, module.weight, module.bias)
        return run


def bf16_runtime(out):
    import runner
    runtime = runner.Runtime(out)
    runtime.arm('bf16')
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.model.eval()
    return runtime


def measure(runtime, p, panel, mode, setup, after_warmup=None):
    """Same warmup, panel and QueryMeter boundary as the historical reference_profile/profile_worker."""
    import torch
    out = Path(p['remote_root']) / 'results' / mode
    s = panel.settings
    progress(p, mode, 'warmup')
    value, _ = panel.datum(runtime, s['warmup_case_id'])
    for _ in range(s['warmup_queries']):
        runtime.infer(value, panel.seed(s['warmup_case_id'], 0), measure=False)
    del value
    torch.cuda.synchronize()
    extra = after_warmup() if after_warmup else {}
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = memory(torch)
    samples, actions = [], {}
    for case in s['case_ids']:
        value, preparation_ms = panel.datum(runtime, case)
        for index in s['seed_indices']:
            action, metrics = runtime.infer(value, panel.seed(case, index), measure=True)
            samples.append(dict(case_id=case, seed_index=index, preprocessing_wall_ms=preparation_ms, **metrics))
            actions[f'{case}_{index}'] = action.detach().cpu()
        del value
        progress(p, mode, 'measuring', queries=len(samples), expected=s['queries'])
    if len(samples) != s['queries']:
        raise RuntimeError(f'Expected {s["queries"]} measured queries, got {len(samples)}')
    save(out / 'samples.json', samples)
    torch.save(actions, out / 'actions.pt')
    props = torch.cuda.get_device_properties(0)
    summary = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode, complete=True,
                   queries=len(samples), metrics={k: statistics(samples, k) for k in RUNTIME_KEYS},
                   resident_after_warmup=resident, gpu=props.name, gpu_uuid=str(props.uuid),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
                   allow_tf32=torch.backends.cuda.matmul.allow_tf32, scope=s['scope'],
                   **setup, **(extra or {}))
    save(out / 'summary.json', summary)
    progress(p, mode, 'complete', queries=len(samples))
    return summary


def run_shapes(p, panel):
    """Record every target Linear call of one query, eager and with the query-local condition cache."""
    import torch
    import condition_cache
    mode, out = 'shapes', Path(p['remote_root']) / 'results' / 'shapes'
    progress(p, mode, 'loading')
    runtime = bf16_runtime(out)
    banks = CountingLinears(runtime.q.modules)
    case = panel.settings['shape_case_id']
    value, _ = panel.datum(runtime, case)
    seed = panel.seed(case, 0)
    runtime.infer(value, seed, measure=False)
    distributions = {}
    for label in ('eager', 'cached'):
        if label == 'cached':
            condition_cache.install(runtime.model, diagnostics=True, stable_buffers=True)
        banks.record = Counter()
        runtime.infer(value, seed, measure=False)
        torch.cuda.synchronize()
        distributions[label] = [dict(name=k[0], stream=stream_of(k[0]), M=k[1], K=k[2], N=k[3], bias=k[4], calls=v)
                                for k, v in sorted(banks.record.items())]
    banks.record = None
    result = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], complete=True,
                  modules=len(banks.modules), seen_modules=len(banks.seen),
                  calls_per_query={k: sum(r['calls'] for r in v) for k, v in distributions.items()},
                  case_id=case, seed_index=0, **distributions)
    save(out / 'shapes.json', result)
    save(out / 'summary.json', {k: v for k, v in result.items() if k not in distributions})
    progress(p, mode, 'complete', calls_per_query=result['calls_per_query'])


def _graph_time_us(torch, calls, warmup, timed):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            for call in calls:
                call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(warmup):
        graph.replay()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(timed):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / len(calls))
    del graph
    return stats.median(samples)


def run_gemm(p):
    """Per unique (M, K, N, bias): BF16 cuBLAS vs INT8 cuBLASLt, weights rotated to defeat L2 reuse."""
    import torch
    import torch.nn.functional as F
    g = p['gemm']
    mode, root = 'gemm', Path(p['remote_root']) / 'results'
    progress(p, mode, 'loading')
    # Microbenchmark only: no NaN fill of every output allocation; TF32 stays off.
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = read(root / 'shapes' / 'shapes.json')
    unique = {}
    for label in ('eager', 'cached'):
        for r in shapes[label]:
            row = unique.setdefault((r['M'], r['K'], r['N'], r['bias']),
                                    dict(M=r['M'], K=r['K'], N=r['N'], bias=r['bias'], streams=set(),
                                         eager_calls=0, cached_calls=0))
            row[label + '_calls'] += r['calls']
            row['streams'].add(r['stream'])
    pad = lambda v, m: -(-v // m) * m
    generator = torch.Generator(device='cuda').manual_seed(20261027)
    rows = []
    for index, row in enumerate(sorted(unique.values(), key=lambda r: (r['M'], r['K'], r['N']))):
        M, K, N = row['M'], row['K'], row['N']
        copies = min(g['graph_calls'], max(g['min_weight_copies'],
                     math.ceil(g['l2_flush_bytes'] / (2 * N * K))))
        # torch._int_mm needs M > 16 and K, N multiples of 8; pad only where required.
        Mi = M if M > 16 else g['int8_min_rows']
        Ki, Ni = pad(K, g['int8_multiple']), pad(N, g['int8_multiple'])
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16, generator=generator)
        weights = [torch.randn(N, K, device='cuda', dtype=torch.bfloat16, generator=generator) * .02
                   for _ in range(copies)]
        bias = torch.randn(N, device='cuda', dtype=torch.bfloat16, generator=generator) if row['bias'] else None
        x_pad = torch.zeros(Mi, Ki, device='cuda', dtype=torch.bfloat16)
        x_pad[:M, :K] = x
        a8 = torch.randint(-127, 128, (Mi, Ki), device='cuda', dtype=torch.int8, generator=generator)
        w8 = [torch.randint(-127, 128, (Ni, Ki), device='cuda', dtype=torch.int8, generator=generator)
              for _ in range(copies)]
        w_scale = torch.rand(Ni, device='cuda', dtype=torch.float32, generator=generator) * 1e-3
        bias_pad = None
        if bias is not None:
            bias_pad = torch.zeros(Ni, device='cuda', dtype=torch.float32)
            bias_pad[:N] = bias.float()

        def unfused(w):
            scale = x_pad.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
            q = torch.clamp(torch.round(x_pad.float() / scale), -127, 127).to(torch.int8)
            y = torch._int_mm(q, w.t()).float() * scale * w_scale
            if bias_pad is not None:
                y = y + bias_pad
            return y.to(torch.bfloat16)[:M, :N]

        variants = dict(
            bf16=[lambda w=w: F.linear(x, w, bias) for w in weights],
            int8=[lambda w=w: torch._int_mm(a8, w.t()) for w in w8],
            int8_unfused=[lambda w=w: unfused(w) for w in w8],
        )
        result = dict(M=M, K=K, N=N, bias=row['bias'], streams=sorted(row['streams']),
                      eager_calls=row['eager_calls'], cached_calls=row['cached_calls'],
                      weight_copies=copies, int8_shape=[Mi, Ki, Ni], int8_padded=(Mi, Ki, Ni) != (M, K, N))
        for name, calls in variants.items():
            try:
                result[name + '_us'] = _graph_time_us(torch, calls, g['warmup_replays'], g['timed_replays'])
            except Exception as exc:
                result[name + '_error'] = repr(exc)[:500]
        for name in ('int8', 'int8_unfused'):
            if name + '_us' in result and 'bf16_us' in result:
                result[name + '_speedup'] = result['bf16_us'] / result[name + '_us']
        rows.append(result)
        del x, weights, bias, x_pad, a8, w8, w_scale, bias_pad, variants
        torch.cuda.empty_cache()
        progress(p, mode, 'measuring', shapes=index + 1, expected=len(unique))
    props = torch.cuda.get_device_properties(0)
    summary = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode, complete=True,
                   shapes=rows, gpu=props.name, torch_version=torch.__version__,
                   cuda_version=torch.version.cuda, method=g)
    save(root / mode / 'summary.json', summary)
    progress(p, mode, 'complete', shapes=len(rows))


def run_bf16_eager(p, panel):
    mode = 'bf16_eager'
    progress(p, mode, 'loading')
    begin = time.perf_counter()
    runtime = bf16_runtime(Path(p['remote_root']) / 'results' / mode)
    measure(runtime, p, panel, mode, dict(load_seconds=time.perf_counter() - begin, cuda_graph=False,
                                          condition_cache=False, weights='bf16 dense'))


def run_q0_c_buffer(p, panel):
    """The recommended engine, installed exactly as the standard-LIBERO evaluation did."""
    from libero_engine_worker import build_runtime
    mode = 'q0_c_buffer'
    out = Path(p['remote_root']) / 'results' / mode
    progress(p, mode, 'loading')
    begin = time.perf_counter()
    runtime = build_runtime(p, out)
    measure(runtime, p, panel, mode, dict(load_and_anchor_seconds=time.perf_counter() - begin,
                                          cuda_graph=True, condition_cache=True,
                                          weights='packed MXVQ, layer BF16 decode', startup=read(out / 'startup.json')))


def run_bf16_cache_graph(p, panel):
    """Original BF16 weights with the engine's condition cache and three manual CUDA Graph regions."""
    import condition_cache
    import graph_runtime
    mode = 'bf16_cache_graph'
    progress(p, mode, 'loading')
    begin = time.perf_counter()
    runtime = bf16_runtime(Path(p['remote_root']) / 'results' / mode)
    banks = CountingLinears(runtime.q.modules)
    runtime.banks = banks
    cache = condition_cache.install(runtime.model, diagnostics=True, stable_buffers=True)
    optimization = SimpleNamespace(cache=cache, graph=None, compiled={'enabled': False, 'regions': {}})
    runtime.optimization = optimization
    graph_runtime.install(runtime, optimization, compiled=False)
    load_seconds = time.perf_counter() - begin

    def after_warmup():
        regions = optimization.graph.report()['regions']
        if set(regions) != {'video_step', 'video_prefill', 'action_step'}:
            raise RuntimeError(f'Expected three CUDA Graph regions, got {sorted(regions)}')
        if len(banks.seen) != len(banks.modules):
            raise RuntimeError(f'Warmup reached {len(banks.seen)}/{len(banks.modules)} target Linears')
        cache.diagnostics = False
        for slot in cache._slots.values():
            slot['input'] = None
        optimization.graph.reset_metrics()
        banks.calls.clear()
        regions = {name: dict({k: v for k, v in r.items() if k != 'captured_modules'},
                              captured_module_count=len(r.get('captured_modules', [])))
                   for name, r in regions.items()}
        metadata = {k: v for k, v in cache.metadata().items() if k != 'buffer_bytes_by_key'}
        return dict(graph_regions=regions, cache_after_warmup=metadata)

    summary = measure(runtime, p, panel, mode, dict(load_seconds=load_seconds, cuda_graph=True,
                                                    condition_cache=True, weights='bf16 dense'), after_warmup)
    calls = optimization.graph.module_call_counts()
    summary['logical_target_calls_per_query'] = sum(calls.values()) / summary['queries']
    summary['cache_diagnostic_mismatches'] = cache.metadata()['diagnostic_mismatches']
    save(Path(p['remote_root']) / 'results' / mode / 'summary.json', summary)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--mode', required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    if args.mode not in p['modes']:
        raise ValueError(args.mode)
    import torch
    from profile_worker import _freeze_numerics
    if torch.cuda.device_count() != 1 or p['allocation']['gpu_family'] not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected exactly one allocated A100')
    _freeze_numerics(torch)
    if args.mode == 'gemm':
        run_gemm(p)
        return
    panel = Panel(p)
    dict(shapes=run_shapes, bf16_eager=run_bf16_eager, q0_c_buffer=run_q0_c_buffer,
         bf16_cache_graph=run_bf16_cache_graph)[args.mode](p, panel)


if __name__ == '__main__':
    main()
