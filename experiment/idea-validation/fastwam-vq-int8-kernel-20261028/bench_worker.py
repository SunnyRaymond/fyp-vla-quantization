"""VQ->INT8 kernel: per-shape microbenchmark + tuning, then policy-call latency/memory and a small accuracy check.

Modes (one process each): kernel, bf16_cache_graph, h1b_bf16emu, h1b_int8.
"""
from __future__ import annotations

import argparse
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


def results(p, *parts):
    return Path(p['remote_root']).joinpath('results', *parts)


def progress(p, mode, phase, **fields):
    row = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode, phase=phase,
               updated_epoch=time.time(), **fields)
    save(results(p, 'progress.json'), row)
    save(results(p, mode, 'progress.json'), row)
    print('INT8K_PROGRESS ' + json.dumps(row), flush=True)


class Panel:
    """The frozen held-out 80-query panel of every historical runtime profile."""
    def __init__(self, p):
        self.settings = p['panel']
        self.plan_path = Path(self.settings['calibration_plan'])
        self.rows = {r['case_id']: r for r in read(self.plan_path)['inputs']}

    def datum(self, runtime, case):
        import numpy as np
        import calibration_source
        observation, metadata = calibration_source.load_observation(np, self.plan_path.parent, self.rows[case])
        runtime.description = metadata['description']
        return runtime.datum(observation)

    def seed(self, case, index):
        return self.rows[case]['sampler_seeds'][index]


def measure(runtime, p, panel, mode, setup, after_warmup=None):
    """80 queries with the historical warmup and QueryMeter boundary, then action+future on the accuracy subset."""
    import torch
    from profile_worker import _capture_anchor
    out, s = results(p, mode), panel.settings
    progress(p, mode, 'warmup')
    value = panel.datum(runtime, s['warmup_case_id'])
    for _ in range(s['warmup_queries']):
        runtime.infer(value, panel.seed(s['warmup_case_id'], 0), measure=False)
    del value
    torch.cuda.synchronize()
    extra = after_warmup() if after_warmup else {}
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = memory(torch)
    samples, actions = [], {}
    for case in s['case_ids']:
        value = panel.datum(runtime, case)
        for index in s['seed_indices']:
            action, metrics = runtime.infer(value, panel.seed(case, index), measure=True)
            samples.append(dict(case_id=case, seed_index=index, **metrics))
            actions[f'{case}_{index}'] = action.detach().cpu()
        del value
        progress(p, mode, 'measuring', queries=len(samples), expected=s['queries'])
    if len(samples) != s['queries']:
        raise RuntimeError(f'Expected {s["queries"]} measured queries, got {len(samples)}')
    process_peak = memory(torch)
    accuracy = {}
    for case in s['accuracy_case_ids']:
        value = panel.datum(runtime, case)
        for index in s['seed_indices']:
            action, _, future, _ = _capture_anchor(runtime, value, panel.seed(case, index), torch)
            accuracy[f'{case}_{index}'] = dict(action=action.detach().cpu(), future=future)
        del value
    save(out / 'samples.json', samples)
    torch.save(actions, out / 'actions.pt')
    torch.save(accuracy, out / 'accuracy.pt')
    props = torch.cuda.get_device_properties(0)
    summary = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode, complete=True,
                   queries=len(samples), metrics={k: statistics(samples, k) for k in RUNTIME_KEYS},
                   resident_after_warmup=resident, process_after_panel=process_peak,
                   accuracy_queries=len(accuracy), gpu=props.name, gpu_uuid=str(props.uuid),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
                   allow_tf32=torch.backends.cuda.matmul.allow_tf32, scope=s['scope'],
                   **setup, **(extra or {}))
    save(out / 'summary.json', summary)
    progress(p, mode, 'complete', queries=len(samples))
    return summary


# ------------------------------------------------------------------------------------------- kernel microbenchmark

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


def run_kernel(p):
    """Per unique (M, K, N, bias): BF16 cuBLAS, dense INT8 cuBLASLt, current decode+cuBLAS, our fused VQ->INT8
    (every tile candidate, bitwise-checked), and scalar W4A8 in the same template. Weights rotate to defeat L2."""
    import torch
    import torch.nn.functional as F
    import decoder_runtime
    import fused_vq
    import vq_int8
    g, mode = p['kernel'], 'kernel'
    progress(p, mode, 'loading')
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = read(g['shape_source'])['shapes']
    tuning = read(Path(p['engine']['source_results']) / 'tuning.json')
    decisions = {tuple(r['shape']): r['selected'] for r in tuning['shapes']}
    decoder = decoder_runtime.DecoderDispatcher(decisions, tuning, results(p, mode, 'reused_decoder.json'),
                                                fused_vq._decode_kernel)
    generator = torch.Generator(device='cuda').manual_seed(20261028)
    pad = lambda v, m: -(-v // m) * m
    rows, selection = [], {}
    for index, shape in enumerate(sorted(shapes, key=lambda r: (r['M'], r['K'], r['N']))):
        M, K, N, has_bias = shape['M'], shape['K'], shape['N'], shape['bias']
        copies = lambda nbytes: min(g['graph_calls'], max(g['min_weight_copies'], math.ceil(g['l2_flush_bytes'] / nbytes)))
        cycle = lambda items: [items[i % len(items)] for i in range(g['graph_calls'])]
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16, generator=generator)
        bias = torch.randn(N, device='cuda', dtype=torch.bfloat16, generator=generator) if has_bias else None
        q = torch.randint(-127, 128, (M, K), device='cuda', dtype=torch.int8, generator=generator)
        a_scale = torch.rand(M, device='cuda', generator=generator) * 1e-2
        bf16_w = [torch.randn(N, K, device='cuda', dtype=torch.bfloat16, generator=generator) * .02
                  for _ in range(copies(2 * N * K))]
        vq = [vq_int8.random_vq(N, K, 'cuda', generator) for _ in range(copies(N * K // 2 + 1))]
        w4 = [vq_int8.random_w4(N, K, 'cuda', generator) for _ in range(copies(N * K // 2 + 1))]
        result = dict(M=M, K=K, N=N, bias=has_bias, streams=shape.get('streams'), cached_calls=shape['cached_calls'],
                      eager_calls=shape['eager_calls'], copies=dict(bf16=len(bf16_w), vq=len(vq), w4=len(w4)))

        def timed(label, calls):
            try:
                result[label + '_us'] = _graph_time_us(torch, cycle(calls), g['warmup_replays'], g['timed_replays'])
            except Exception as exc:
                result[label + '_error'] = repr(exc)[:400]

        timed('bf16', [lambda w=w: F.linear(x, w, bias) for w in bf16_w])
        # Dense INT8 ceiling: _int_mm needs M > 16 and K, N multiples of 8 (padded only where required).
        Mi, Ki, Ni = (M if M > 16 else 32), pad(K, 8), pad(N, 8)
        a8 = torch.randint(-127, 128, (Mi, Ki), device='cuda', dtype=torch.int8, generator=generator)
        w8 = [torch.randint(-127, 128, (Ni, Ki), device='cuda', dtype=torch.int8, generator=generator)
              for _ in range(copies(N * K))]
        timed('int8_dense', [lambda w=w: torch._int_mm(a8, w.t()) for w in w8])
        # Current engine: tuned layer decode into a BF16 scratch, then BF16 cuBLAS.
        scratch = torch.empty(N * K, dtype=torch.bfloat16, device='cuda')
        legacy = [fused_vq.PreparedVQ(N, K, v.books.to(torch.bfloat16), v.indices, v.row_scales.to(torch.bfloat16),
                                      torch.full((N, -(-K // 32)), 127, dtype=torch.uint8, device='cuda')) for v in vq]
        timed('bf16_decode', [lambda v=v: F.linear(x, decoder(v, scratch), bias) for v in legacy])
        # Exact references for the first weight copy.
        ref_vq = vq_int8.reference_linear(q, a_scale, vq_int8.decode_vq_int(vq[0]), vq[0].row_scales, bias)
        ref_w4 = vq_int8.reference_linear(q, a_scale, vq_int8.decode_w4_int(w4[0]), w4[0].row_scales, bias)
        tried = {}
        for family, weights, fn, ref in (('vq', vq, vq_int8.vq_linear, ref_vq), ('w4', w4, vq_int8.w4_linear, ref_w4)):
            entries = []
            for cfg in vq_int8.candidates(M, K, N, scalar=family == 'w4'):
                entry = dict(config=cfg.name)
                try:
                    out = fn(q, a_scale, weights[0], bias, cfg)
                    torch.cuda.synchronize()
                    entry['bitwise'] = bool(torch.equal(out.view(torch.int16), ref.view(torch.int16)))
                    entry['relative_max'] = float((out.float() - ref.float()).abs().max()
                                                  / ref.float().abs().max().clamp_min(1e-30))
                    if entry['bitwise'] or entry['relative_max'] <= 2 ** -7:
                        entry['us'] = _graph_time_us(torch, cycle([lambda w=w, c=cfg: fn(q, a_scale, w, bias, c)
                                                                   for w in weights]),
                                                     g['warmup_replays'], g['timed_replays'])
                except Exception as exc:
                    entry['error'] = repr(exc)[:300]
                entries.append(entry)
            tried[family] = entries
            good = [e for e in entries if 'us' in e and e['bitwise']] or [e for e in entries if 'us' in e]
            if good:
                result[family + '_selected_bitwise'] = good[0]['bitwise']
                best = min(good, key=lambda e: e['us'])
                result[family + '_us'], result[family + '_config'] = best['us'], best['config']
                for variant in ('word', 'byte'):
                    sub = [e for e in good if e['config'].endswith(variant)]
                    if sub:
                        result[f'{family}_{variant}_us'] = min(e['us'] for e in sub)
                no_split = [e for e in good if '_s1_' in e['config']]
                if no_split:
                    result[family + '_nosplit_us'] = min(e['us'] for e in no_split)
            result[family + '_candidates'] = len(entries)
            result[family + '_bitwise_failures'] = [e['config'] for e in entries if e.get('bitwise') is False]
            result[family + '_errors'] = len([e for e in entries if 'error' in e])
        if 'vq_config' in result:
            selection[f'{M},{K},{N}'] = result['vq_config']
        # Activation side: FP32 operand -> INT8 + scale (ours) vs the current FP32 -> BF16 A8 values.
        from plus_engine import _a8_kernel
        import triton
        z = torch.randn(M, K, device='cuda', generator=generator)
        qb, sb, bf = torch.empty_like(q), torch.empty_like(a_scale), torch.empty(M, K, dtype=torch.bfloat16, device='cuda')
        block = triton.next_power_of_2(K)
        timed('quant_int8', [lambda: vq_int8.quant_a8(z, qb, sb)])
        timed('quant_bf16', [lambda: _a8_kernel[(M,)](z, bf, K, block, num_warps=8 if block >= 8192 else 4,
                                                      enable_fp_fusion=False)])
        q_ref, s_ref = vq_int8.reference_quant_a8(z)
        vq_int8.quant_a8(z, qb, sb)
        result['quant_int8_bitwise'] = bool(torch.equal(qb, q_ref) and torch.equal(sb.view(torch.int32), s_ref.view(torch.int32)))
        for name in ('int8_dense', 'bf16_decode', 'vq', 'w4'):
            if name + '_us' in result and 'bf16_us' in result:
                result[name + '_speedup'] = result['bf16_us'] / result[name + '_us']
        rows.append(result)
        save(results(p, mode, 'candidates', f'{M}_{K}_{N}.json'), tried)
        del x, bias, q, a_scale, bf16_w, vq, w4, a8, w8, scratch, legacy, ref_vq, ref_w4, z, qb, sb, bf
        torch.cuda.empty_cache()
        progress(p, mode, 'measuring', shapes=index + 1, expected=len(shapes))
    props = torch.cuda.get_device_properties(0)
    save(results(p, mode, 'selection.json'), dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'],
                                                  complete=True, selected=selection))
    save(results(p, mode, 'summary.json'), dict(
        protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'], mode=mode, complete=True, shapes=rows,
        gpu=props.name, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        triton_version=__import__('triton').__version__, method=g))
    progress(p, mode, 'complete', shapes=len(rows))


# ------------------------------------------------------------------------------------------- policy-call runtimes

class CountingLinears:
    """Plain BF16 F.linear with the counters graph_runtime expects from engine banks."""
    def __init__(self, modules):
        from collections import Counter
        self.modules, self.calls, self.seen = modules, Counter(), set()
        self.diagnostics, self.track = False, True
        for name, module in modules.items():
            module.forward = self._forward(name, module)

    def _forward(self, name, module):
        import torch.nn.functional as F

        def run(x):
            if self.track:
                self.calls[name] += 1
                self.seen.add(name)
            return F.linear(x, module.weight, module.bias)
        return run


def run_bf16_cache_graph(p, panel):
    """Original BF16 weights with the engine's condition cache and three CUDA Graph regions."""
    import runner
    import condition_cache
    import graph_runtime
    mode = 'bf16_cache_graph'
    progress(p, mode, 'loading')
    begin = time.perf_counter()
    runtime = runner.Runtime(results(p, mode))
    runtime.arm('bf16')
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.model.eval()
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
        cache.diagnostics = False
        for slot in cache._slots.values():
            slot['input'] = None
        optimization.graph.reset_metrics()
        banks.calls.clear()
        return {}

    measure(runtime, p, panel, mode, dict(load_seconds=load_seconds, cuda_graph=True, condition_cache=True,
                                          weights='bf16 dense'), after_warmup)


def run_h1b_bf16emu(p, panel):
    """The closed-loop engine of job 26033945: packed 1b codes, layer BF16 decode + cuBLAS, emulated A8."""
    from plus_engine import build_runtime
    mode = 'h1b_bf16emu'
    out = results(p, mode)
    progress(p, mode, 'loading')
    begin = time.perf_counter()
    cfg = dict(bundle=p['h1b']['bundle'], anchor=p['h1b']['anchor'], activation_bits=8, use_u=False)
    runtime = build_runtime(p, mode, cfg, out)
    measure(runtime, p, panel, mode, dict(load_and_anchor_seconds=time.perf_counter() - begin, cuda_graph=True,
                                          condition_cache=True, weights='packed 1b VQ, layer BF16 decode + cuBLAS',
                                          startup=read(out / 'startup.json')))


def run_h1b_int8(p, panel):
    import int8_engine
    mode = 'h1b_int8'
    out = results(p, mode)
    progress(p, mode, 'loading')
    begin = time.perf_counter()
    selection = read(results(p, 'kernel', 'selection.json'))['selected']
    runtime = int8_engine.build_runtime(p, out, selection)
    measure(runtime, p, panel, mode, dict(load_and_anchor_seconds=time.perf_counter() - begin, cuda_graph=True,
                                          condition_cache=True, weights='packed 1b VQ, fused VQ->INT8 MMA',
                                          startup=read(out / 'startup.json')))


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
    if args.mode == 'kernel':
        run_kernel(p)
        return
    panel = Panel(p)
    dict(bf16_cache_graph=run_bf16_cache_graph, h1b_bf16emu=run_h1b_bf16emu,
         h1b_int8=run_h1b_int8)[args.mode](p, panel)


if __name__ == '__main__':
    main()
