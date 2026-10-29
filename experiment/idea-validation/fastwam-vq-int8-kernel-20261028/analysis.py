"""Aggregate the four modes into results/summary.json and RESULTS.zh.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNTIMES = ('bf16_cache_graph', 'h1b_bf16emu', 'h1b_int8')
GEMMS = ('bf16', 'int8_dense', 'bf16_decode', 'vq', 'w4')


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def action_metrics(torch, actual, reference):
    a = actual.float().reshape(-1, actual.shape[-1])
    r = reference.float().reshape(-1, reference.shape[-1])
    rmse = lambda x, y: float((x - y).square().mean().sqrt())
    return dict(first10motor_rmse=rmse(a[:10, :6], r[:10, :6]), full32motor_rmse=rmse(a[:, :6], r[:, :6]),
                first10gripper_rmse=rmse(a[:10, 6:], r[:10, 6:]), max_abs=float((a - r).abs().max()))


def mean_rows(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]} if rows else {}


def accuracy(root):
    import torch
    out = {}
    actions = {m: root / m / 'actions.pt' for m in RUNTIMES}
    subset = {m: root / m / 'accuracy.pt' for m in RUNTIMES}
    if not actions['bf16_cache_graph'].exists():
        return out
    ref = torch.load(actions['bf16_cache_graph'])
    ref_sub = torch.load(subset['bf16_cache_graph']) if subset['bf16_cache_graph'].exists() else {}
    loaded = {m: torch.load(actions[m]) for m in ('h1b_bf16emu', 'h1b_int8') if actions[m].exists()}
    for mode, acts in loaded.items():
        row = dict(panel80_vs_bf16=mean_rows([action_metrics(torch, acts[k], ref[k]) for k in ref]))
        if subset[mode].exists() and ref_sub:
            sub = torch.load(subset[mode])
            row['subset10_vs_bf16'] = mean_rows([
                dict(action_metrics(torch, sub[k]['action'], ref_sub[k]['action']),
                     future_nmse=float((sub[k]['future'].float() - ref_sub[k]['future'].float()).square().sum()
                                       / ref_sub[k]['future'].float().square().sum()))
                for k in ref_sub])
        out[mode] = row
    if len(loaded) == 2:
        a, b = loaded['h1b_int8'], loaded['h1b_bf16emu']
        out['int8_vs_bf16emu_panel80'] = mean_rows([action_metrics(torch, a[k], b[k]) for k in b])
        if subset['h1b_int8'].exists() and subset['h1b_bf16emu'].exists():
            sa, sb = torch.load(subset['h1b_int8']), torch.load(subset['h1b_bf16emu'])
            out['int8_vs_bf16emu_subset10_future_nmse'] = sum(
                float((sa[k]['future'].float() - sb[k]['future'].float()).square().sum()
                      / sb[k]['future'].float().square().sum()) for k in sb) / len(sb)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    p = json.loads(parser.parse_args().protocol.read_text(encoding='utf-8'))
    root = Path(p['remote_root']) / 'results'
    import os
    summaries = {m: read(root / m / 'summary.json') for m in p['modes']}
    complete = [m for m, s in summaries.items() if s and s.get('complete')]
    lines = [f"# VQ→INT8 融合 kernel：逐形状与整次 policy call", '',
             f"作业 `{os.environ.get('PBS_JOBID', '?')}`；完成模式 {len(complete)}/{len(p['modes'])}。"]
    weighted = {}
    kernel = summaries.get('kernel')
    if kernel:
        lines += ['', f"GPU {kernel['gpu']}，torch {kernel['torch_version']}，triton {kernel['triton_version']}。",
                  '', '## 逐形状（CUDA Graph 中位数 µs，权重轮换避开 L2；vq/w4 为全部候选中逐位正确的最快者）', '',
                  '| M | K | N | 调用/query | BF16 | INT8 dense | BF16 decode（现 engine） | VQ→INT8（ours） | word | byte | 无 split-K | 标量 W4A8 | 量化 INT8 | 量化 BF16 | ours 加速 |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        f = lambda r, k: f"{r[k]:.1f}" if k in r else '—'
        for r in kernel['shapes']:
            lines.append(f"| {r['M']} | {r['K']} | {r['N']} | {r['cached_calls']} | {f(r,'bf16_us')} | {f(r,'int8_dense_us')} | "
                         f"{f(r,'bf16_decode_us')} | {f(r,'vq_us')} | {f(r,'vq_word_us')} | {f(r,'vq_byte_us')} | "
                         f"{f(r,'vq_nosplit_us')} | {f(r,'w4_us')} | {f(r,'quant_int8_us')} | {f(r,'quant_bf16_us')} | "
                         f"{(str(round(r['vq_speedup'], 2)) + '×') if 'vq_speedup' in r else '—'} |")
        groups = {'全部': lambda r: True, 'video': lambda r: 'video' in (r.get('streams') or []),
                  'action': lambda r: 'action' in (r.get('streams') or []), 'M=294': lambda r: r['M'] == 294,
                  'M=32': lambda r: r['M'] == 32}
        lines += ['', '## 每次 policy call 的 GEMM 时间（按 cache 后调用次数加权，ms）', '',
                  '| 分组 | ' + ' | '.join(GEMMS) + ' | ours/BF16 |', '|---|' + '---:|' * (len(GEMMS) + 1)]
        for label, keep in groups.items():
            rows = [r for r in kernel['shapes'] if keep(r)]
            w = {g: sum(r['cached_calls'] * r[g + '_us'] for r in rows if g + '_us' in r) / 1000 for g in GEMMS}
            weighted[label] = w
            ratio = f"{w['bf16'] / w['vq']:.2f}×" if w['vq'] else '—'
            lines.append(f"| {label} | " + ' | '.join(f"{w[g]:.2f}" for g in GEMMS) + f" | {ratio} |")
        bad = [f"{r['M']},{r['K']},{r['N']}" for r in kernel['shapes'] if not r.get('vq_selected_bitwise', False)]
        lines += ['', f"逐位检查：VQ 选中配置非逐位的形状 {bad or '无'}；量化 kernel 逐位："
                  f"{all(r.get('quant_int8_bitwise') for r in kernel['shapes'])}。"]
    lines += ['', '## 整次 policy call（80 个固定 held-out query，单进程，同一 allocation）', '',
              '| Runtime | p50 ms | mean ms | p95 ms | 去噪 p50 ms | peak allocated GB | peak reserved GB | 相对 BF16+cache+Graph |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    base = summaries.get('bf16_cache_graph')
    for mode in RUNTIMES:
        s = summaries.get(mode)
        if not s:
            lines.append(f"| {mode} | 失败或未运行 |||||||")
            continue
        m = s['metrics']
        rel = base['metrics']['full_query_wall_ms']['p50'] / m['full_query_wall_ms']['p50'] if base else float('nan')
        lines.append(f"| {mode} | {m['full_query_wall_ms']['p50']:.1f} | {m['full_query_wall_ms']['mean']:.1f} | "
                     f"{m['full_query_wall_ms']['p95']:.1f} | {m['denoising_latency_ms']['p50']:.1f} | "
                     f"{m['peak_allocated_gpu_GB']['maximum']:.3f} | {m['peak_reserved_gpu_GB']['maximum']:.3f} | {rel:.2f}× |")
    r = p['references']
    lines += ['', f"历史参考（作业 {r['int8_speed_job']}）：BF16 eager {r['bf16_eager_p50_ms']} ms，"
              f"BF16+cache+Graph {r['bf16_cache_graph_p50_ms']} ms，Q0 c_buffer {r['q0_c_buffer_p50_ms']} ms。"]
    int8 = summaries.get('h1b_int8')
    if int8:
        st = int8['startup']
        inst = st['installation']
        lines += ['', '## INT8 engine 启动检查', '',
                  f"- 614 层 operand（INT8 + scale）与 eager 参考逐位一致：{st['operand_bitwise_modules']}/614",
                  f"- 614 层 GEMM 输出与 FP64 精确参考逐位一致：{st['output_bitwise_modules']}/614",
                  f"- 常驻 packed target：{inst['packed_target_bytes'] / 2**30:.3f} GiB；BF16 权重 scratch：0",
                  f"- 未调优形状：{st.get('untuned_shapes') or '无'}",
                  f"- 相对 BF16 模拟 anchor 的漂移：{json.dumps(st['anchor_drift_vs_bf16_emulated'])}"]
    acc = accuracy(root)
    if acc:
        lines += ['', '## 精度（相对 BF16；80 query 只有 action，10 query 子集含 future）', '',
                  '| Runtime | first10 motor | full32 motor | first10 gripper | 子集 future NMSE |', '|---|---:|---:|---:|---:|']
        for mode in ('h1b_bf16emu', 'h1b_int8'):
            if mode in acc:
                a, sub = acc[mode]['panel80_vs_bf16'], acc[mode].get('subset10_vs_bf16', {})
                lines.append(f"| {mode} | {a['first10motor_rmse']:.4f} | {a['full32motor_rmse']:.4f} | "
                             f"{a['first10gripper_rmse']:.4f} | {sub.get('future_nmse', float('nan')):.4f} |")
        if 'int8_vs_bf16emu_panel80' in acc:
            d = acc['int8_vs_bf16emu_panel80']
            lines += ['', f"INT8 vs BF16 模拟（同一 1b 码本）：first10 motor RMSE {d['first10motor_rmse']:.4f}，"
                      f"future NMSE {acc.get('int8_vs_bf16emu_subset10_future_nmse', float('nan')):.4f}。"]
    summary = dict(protocol=p['protocol'], pbs_jobid=os.environ.get('PBS_JOBID'), complete=len(complete) == len(p['modes']),
                   modes_complete=len(complete), modes={m: bool(s and s.get('complete')) for m, s in summaries.items()},
                   weighted_gemm_ms=weighted, accuracy=acc,
                   runtimes={m: summaries[m]['metrics'] for m in RUNTIMES if summaries.get(m)})
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (root / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
