"""Aggregate the speed gate into summary.json and RESULTS.zh.md; runs inside the PBS allocation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from reference_profile import guard, save

RUNTIMES = ('bf16_eager', 'bf16_cache_graph', 'q0_c_buffer')
VARIANTS = ('bf16', 'int8', 'int8_unfused')


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def parity(torch, left_path, right_path):
    if not Path(left_path).exists() or not Path(right_path).exists():
        return None
    left = torch.load(left_path, map_location='cpu', weights_only=True)
    right = torch.load(right_path, map_location='cpu', weights_only=True)
    if set(left) != set(right):
        return dict(error='different query keys')
    errors = [float((left[k].float() - right[k].float()).abs().max()) for k in sorted(left)]
    return dict(queries=len(errors), max_abs=max(errors),
                exact_queries=sum(torch.equal(left[k], right[k]) for k in left))


def gemm_totals(shapes, gemm):
    """Per policy call GEMM time, weighting each shape by its condition-cache calls per query."""
    timing = {(r['M'], r['K'], r['N'], r['bias']): r for r in gemm['shapes']}
    totals = {v: 0.0 for v in VARIANTS}
    by_m, by_stream = defaultdict(lambda: {v: 0.0 for v in VARIANTS}), defaultdict(lambda: {v: 0.0 for v in VARIANTS})
    missing = []
    for r in shapes['cached']:
        t = timing[(r['M'], r['K'], r['N'], r['bias'])]
        for v in VARIANTS:
            if v + '_us' not in t:
                missing.append([r['M'], r['K'], r['N'], v])
                continue
            ms = r['calls'] * t[v + '_us'] / 1000
            totals[v] += ms
            by_m[f"M={r['M']}"][v] += ms
            by_stream[r['stream']][v] += ms
    return dict(per_query_ms=totals, by_m=dict(by_m), by_stream=dict(by_stream), missing=missing)


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    p = read(parser.parse_args().protocol)
    import torch
    root = Path(p['remote_root']) / 'results'
    artifacts = Path(os.environ['ARTIFACTS'])
    exits = {m: (artifacts / f'{m}.exit').read_text().strip() if (artifacts / f'{m}.exit').exists() else None
             for m in p['modes']}
    summaries = {m: read(root / m / 'summary.json') for m in p['modes']}
    complete = {m: bool(s and s.get('complete')) and exits[m] == '0' for m, s in summaries.items()}
    shapes = read(root / 'shapes' / 'shapes.json')
    gemm = summaries['gemm']
    totals = gemm_totals(shapes, gemm) if complete['shapes'] and complete['gemm'] else None

    runtimes = {}
    for mode in RUNTIMES:
        s = summaries[mode]
        if not complete[mode]:
            continue
        m = s['metrics']
        runtimes[mode] = dict(p50_ms=m['full_query_wall_ms']['p50'], mean_ms=m['full_query_wall_ms']['mean'],
                              p95_ms=m['full_query_wall_ms']['p95'],
                              denoising_p50_ms=m['denoising_latency_ms']['p50'],
                              peak_allocated_GB=m['peak_allocated_gpu_GB']['p50'],
                              peak_reserved_GB=m['peak_reserved_gpu_GB']['p50'])
    for mode, row in runtimes.items():
        for base in ('bf16_eager', 'bf16_cache_graph'):
            if base in runtimes:
                row[f'speedup_vs_{base}'] = runtimes[base]['p50_ms'] / row['p50_ms']
        ref = p['references'].get(mode)
        if ref:
            row['historical_p50_ms'], row['historical_peak_allocated_GB'] = ref['p50_ms'], ref['peak_allocated_GB']

    checks = dict(
        bf16_cache_graph_vs_bf16_eager=parity(torch, root / 'bf16_cache_graph/actions.pt', root / 'bf16_eager/actions.pt'),
        bf16_eager_vs_historical=parity(torch, root / 'bf16_eager/actions.pt', p['references']['bf16_eager']['actions']),
        q0_c_buffer_vs_historical=parity(torch, root / 'q0_c_buffer/actions.pt', p['references']['q0_c_buffer']['actions']),
    )
    summary = dict(protocol=p['protocol'], pbs_jobid=os.environ['PBS_JOBID'],
                   complete=all(complete.values()), modes_complete=sum(complete.values()),
                   mode_exit_codes=exits, calls_per_query=shapes and shapes['calls_per_query'],
                   gemm_per_query=totals, runtimes=runtimes, parity=checks,
                   bf16_cache_graph_logical_calls_per_query=(summaries['bf16_cache_graph'] or {}).get(
                       'logical_target_calls_per_query'),
                   new_training_updates=0, closed_loop_episodes=0)
    save(root / 'summary.json', summary)

    lines = ['# INT8 vs BF16 速度检查（阶段 0）', '',
             f"作业 `{summary['pbs_jobid']}`；完成模式 {summary['modes_complete']}/{len(p['modes'])}。"
             f"GPU：{(gemm or summaries['bf16_eager'] or {}).get('gpu', '—')}。", '']
    if shapes:
        lines += [f"每 query 目标 Linear 调用：eager {shapes['calls_per_query']['eager']}，"
                  f"condition cache 后 {shapes['calls_per_query']['cached']}。", '']
    if gemm:
        lines += ['## 逐形状 GEMM（CUDA Graph 中位数，权重轮换避开 L2）', '',
                  '| M | K | N | bias | stream | 调用/query（cache） | BF16 µs | INT8 µs | INT8 含外围 µs | INT8 加速 | 含外围加速 | INT8 是否补齐 |',
                  '|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
        for r in gemm['shapes']:
            lines.append(f"| {r['M']} | {r['K']} | {r['N']} | {'是' if r['bias'] else '否'} | {','.join(r['streams'])} | "
                         f"{r['cached_calls']} | {fmt(r.get('bf16_us'), 1)} | {fmt(r.get('int8_us'), 1)} | "
                         f"{fmt(r.get('int8_unfused_us'), 1)} | {fmt(r.get('int8_speedup'), 2)}× | "
                         f"{fmt(r.get('int8_unfused_speedup'), 2)}× | {'是' if r['int8_padded'] else '否'} |")
        errors = [(r['M'], r['K'], r['N'], k) for r in gemm['shapes'] for k in r if k.endswith('_error')]
        if errors:
            lines += ['', f'未测成功的变体：{errors}']
        lines.append('')
    if totals:
        t = totals['per_query_ms']
        lines += ['## 每次 policy call 的纯 GEMM 时间（按 cache 后调用次数加权）', '',
                  '| 分组 | BF16 ms | INT8 ms | INT8 含外围 ms | INT8 加速 | 含外围加速 |', '|---|---:|---:|---:|---:|---:|']
        groups = [('全部', t)] + sorted(totals['by_m'].items()) + sorted(totals['by_stream'].items())
        for name, g in groups:
            lines.append(f"| {name} | {fmt(g['bf16'])} | {fmt(g['int8'])} | {fmt(g['int8_unfused'])} | "
                         f"{fmt(g['bf16'] / g['int8'] if g['int8'] else None, 2)}× | "
                         f"{fmt(g['bf16'] / g['int8_unfused'] if g['int8_unfused'] else None, 2)}× |")
        lines.append('')
    lines += ['## 整次 policy call（同 80 个固定 query、3 次 warmup）', '',
              '| Runtime | p50 ms | mean ms | p95 ms | 去噪 p50 ms | peak allocated GB | peak reserved GB | 相对 BF16 eager | 相对 BF16+cache+Graph | 历史 p50 ms |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in RUNTIMES:
        r = runtimes.get(mode)
        if not r:
            lines.append(f'| {mode} | 未完成 | | | | | | | | |')
            continue
        lines.append(f"| {mode} | {fmt(r['p50_ms'], 1)} | {fmt(r['mean_ms'], 1)} | {fmt(r['p95_ms'], 1)} | "
                     f"{fmt(r['denoising_p50_ms'], 1)} | {fmt(r['peak_allocated_GB'])} | {fmt(r['peak_reserved_GB'])} | "
                     f"{fmt(r.get('speedup_vs_bf16_eager'), 2)}× | {fmt(r.get('speedup_vs_bf16_cache_graph'), 2)}× | "
                     f"{fmt(r.get('historical_p50_ms'), 1)} |")
    lines += ['', '## 输出一致性（80 个 action 的最大绝对差）', '']
    for name, value in checks.items():
        lines.append(f'- {name}: {json.dumps(value)}')
    (root / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SPEED_ANALYSIS ' + json.dumps(dict(complete=summary['complete'], runtimes=runtimes,
                                               gemm_per_query=totals and totals['per_query_ms'])), flush=True)


if __name__ == '__main__':
    main()
