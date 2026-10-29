"""Aggregate D1 statistics and D3 held-out arms into summary.json and RESULTS.zh.md (JSON only, no model)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

PROTOCOL = 'fastwam-temporal-a4-d1d3-20261028-v1'
SHOW = ('first10motor_rmse', 'future_latent_nmse', 'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse',
        'first10gripper_disagreement')
LOOPS = ('video_loop', 'action_loop')
TEMPORAL = ('A4_sd', 'A4_inno', 'A4_inno_sd', 'A4_inno_sd05')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def median(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def quartiles(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if len(values) < 4:
        return None
    q = statistics.quantiles(values, n=4)
    return [q[0], statistics.median(values), q[2]]


def category(entry, tg):
    meta, calls, recorded = entry['meta'], entry['calls'], len(entry['steps'])
    if calls['calls'] == 1:
        return 'single'
    if recorded == 1 and calls['static'] >= 1:
        return 'static'
    if meta['stream'] == 'video' and meta['tokens'] == tg['video_tokens'] and recorded >= 10:
        return 'video_loop'
    if meta['stream'] == 'action' and meta['tokens'] == tg['action_tokens'] and recorded >= 10:
        return 'action_loop'
    return 'other'


def step_stats(v):
    """Ratios for one key/step/group from the context sums."""
    if not v.get('has_previous'):
        return None
    energy, inno = v['energy'], v['innovation_energy']
    denom = math.sqrt(v['residual_sq'] * v['previous_residual_sq'])
    amr = v['absmax_ratio'] / v['has_previous']
    return dict(innovation_ratio=math.sqrt(inno / energy) if energy > 0 else None,
                absmax_ratio=amr,
                bits_gain=-math.log2(amr) if amr > 0 else None,
                residual_corr=v['residual_dot'] / denom if denom > 0 else None)


def d1_summary(raw, tg):
    by_cat = defaultdict(list)
    per_key = {}
    for k, entry in raw['keys'].items():
        cat = category(entry, tg)
        by_cat[cat].append(k)
        per_key[k] = {int(s): {lab: step_stats(v) for lab, v in groups.items()} for s, groups in entry['steps'].items()}
    out = dict(contexts=raw['contexts'], counts={c: len(v) for c, v in sorted(by_cat.items())},
               call_patterns={c: sorted({f"{raw['keys'][k]['calls']['calls']}/{raw['keys'][k]['calls']['static']}"
                                         for k in keys}) for c, keys in sorted(by_cat.items())})
    fields = ('innovation_ratio', 'absmax_ratio', 'bits_gain', 'residual_corr')
    out['by_step'] = {}
    for cat in LOOPS:
        rows = {}
        for s in range(1, 11):
            vals = [per_key[k].get(s, {}).get('all') for k in by_cat[cat]]
            vals = [v for v in vals if v]
            if vals:
                rows[s] = {f: median([v[f] for v in vals]) for f in fields}
                rows[s]['keys'] = len(vals)
        out['by_step'][cat] = rows
    out['by_group'] = {}
    for cat, labels in (('video_loop', ('all', 'frame0', 'future')), ('action_loop', ('all', 'first10', 'rest'))):
        out['by_group'][cat] = {}
        for lab in labels:
            vals = [per_key[k].get(s, {}).get(lab) for k in by_cat[cat] for s in range(1, 10)]
            vals = [v for v in vals if v]
            out['by_group'][cat][lab] = {f: quartiles([v[f] for v in vals]) for f in fields}
    pooled = [per_key[k][s]['all'] for cat in LOOPS for k in by_cat[cat] for s in range(1, 10)
              if per_key[k].get(s, {}).get('all')]
    out['loops_pooled'] = {f: quartiles([v[f] for v in pooled]) for f in fields}
    flops = {}
    for cat in LOOPS:
        num = den = 0.
        for k in by_cat[cat]:
            entry = raw['keys'][k]
            w = entry['meta']['tokens'] * entry['shape'][0] * entry['shape'][1]
            vals = [per_key[k][s]['all']['innovation_ratio'] for s in range(1, 10)
                    if per_key[k].get(s, {}).get('all') and per_key[k][s]['all']['innovation_ratio'] is not None]
            if vals:
                num += w * sum(vals) / len(vals)
                den += w
        flops[cat] = num / den if den else None
    out['flop_weighted_innovation_ratio'] = flops
    med_r = out['loops_pooled']['innovation_ratio'][1] if out['loops_pooled']['innovation_ratio'] else None
    med_c = out['loops_pooled']['residual_corr'][1] if out['loops_pooled']['residual_corr'] else None
    out['decision'] = dict(median_innovation_ratio=med_r, median_residual_corr=med_c,
                           go=bool((med_r is not None and med_r < .5) or (med_c is not None and med_c > .5)))
    return out


def context_means(records, metric):
    by = defaultdict(list)
    for r in records:
        by[r['case_id']].append(r[metric])
    return {c: sum(v) / len(v) for c, v in by.items()}


def paired(a, b, metric):
    x, y = context_means(a, metric), context_means(b, metric)
    better = sum(x[c] < y[c] for c in x)
    worse = sum(x[c] > y[c] for c in x)
    return dict(better=better, worse=worse, tied=len(x) - better - worse)


def run(cfg):
    root = Path(cfg['remote_root']) / 'results'
    arms = {a: read(root / 'arms' / f'{a}.json') for a in cfg['arms'] if (root / 'arms' / f'{a}.json').exists()}
    records = {a: [read(p) for p in sorted((root / 'endpoints' / a).glob('*.json'))] for a in arms}
    o = {a: v['overall'] for a, v in arms.items()}
    summary = dict(protocol=PROTOCOL, arms_complete=len(arms), arms_expected=len(cfg['arms']),
                   complete=len(arms) == len(cfg['arms']), overall=o,
                   by_suite={a: v['by_suite'] for a, v in arms.items()},
                   seconds_per_query={a: v['seconds_per_query'] for a, v in arms.items()},
                   static_reuse_per_query={a: v['static_reuse_per_query'] for a, v in arms.items()})
    if 'A4' in arms:
        summary['h_w16a4_reference'] = arms['A4'].get('h_w16a4_reference')
    workers = {w: read(root / f'worker_{w}' / 'startup.json') for w in cfg['workers']
               if (root / f'worker_{w}' / 'startup.json').exists()}
    summary['startup'] = workers
    summary['pbs_jobid'] = next(iter(workers.values()), {}).get('pbs_jobid')
    if 'A4' in o and 'A8' in o:
        gap = {}
        for a in o:
            gap[a] = {}
            for m in ('first10motor_rmse', 'future_latent_nmse'):
                span = o['A4'][m] - o['A8'][m]
                gap[a][m] = (o['A4'][m] - o[a][m]) / span if span > 0 else None
        summary['gap_closure_to_A8'] = gap
    pairs = {}
    for ref in ('A4', 'A4_g32'):
        if ref in records:
            pairs[ref] = {a: {m: paired(records[a], records[ref], m) for m in ('first10motor_rmse', 'future_latent_nmse')}
                          for a in records if a != ref and len(records[a]) == len(records[ref])}
    summary['paired_contexts'] = pairs
    d3 = None
    if all(a in o for a in ('A4', 'A8', 'A4_g32')) and any(a in o for a in TEMPORAL):
        candidates = [a for a in TEMPORAL if a in o]
        best = min(candidates, key=lambda a: o[a]['first10motor_rmse'] + o[a]['future_latent_nmse'])
        g = summary['gap_closure_to_A8'][best]
        d3 = dict(best_temporal_arm=best, gap_closure=g,
                  beats_g32={m: o[best][m] < o['A4_g32'][m] for m in ('first10motor_rmse', 'future_latent_nmse')})
        d3['go'] = bool(all(v is not None and v >= .5 for v in g.values()) and all(d3['beats_g32'].values()))
    summary['d3_decision'] = d3
    if 'A4' in o and 'A4_dither' in o:
        summary['mechanism_dither'] = {m: o['A4_dither'][m] - o['A4'][m] for m in ('first10motor_rmse', 'future_latent_nmse')}
    raw_path = root / 'd1' / 'raw.json'
    summary['d1'] = d1_summary(read(raw_path), cfg['d1']['token_groups']) if raw_path.exists() else None
    save(root / 'summary.json', summary)
    (root / 'RESULTS.zh.md').write_text(report(cfg, summary), encoding='utf-8')
    return summary


def fmt(v, digits=4):
    return '—' if v is None else f'{v:.{digits}f}'


def pct(v):
    return '—' if v is None else f'{100 * v:+.1f}%'


def report(cfg, s):
    o = s['overall']
    lines = ['# D1/D3：跨步 INT4 activation 量化器（BF16 权重，H 坐标）', '',
             f"作业 `{s.get('pbs_jobid')}`。held-out 40 contexts × 2 seeds，seed→context→suite→四 suite 等权；"
             f"完成臂 {s['arms_complete']}/{s['arms_expected']}。权重为原 BF16（Smooth+signs+H128 坐标），只换 activation operand。", '']
    ref = s.get('h_w16a4_reference')
    if ref:
        first = ref['first_endpoint']
        lines += [f"**历史对齐**：A4 臂即历史 H_W16A4（闭环 204/280）。case 3000 seed 0 最大相对差 {first['max_relative']:.2e}"
                  f"（指标逐位一致：{first['bitwise_metrics']}）；80 个 endpoint 的最大绝对差 motor "
                  f"{ref['max_abs_diff_all_endpoints']['first10motor_rmse']:.2e}、future "
                  f"{ref['max_abs_diff_all_endpoints']['future_latent_nmse']:.2e}。", '']
    lines += ['## D3 主表', '', '| 臂 | first10 motor | future NMSE | full32 motor | late22 motor | gripper RMSE | gripper 分歧 | 相对 A4 motor | 相对 A4 future | 补回 A4→A8 motor | 补回 A4→A8 future | s/query |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    gap = s.get('gap_closure_to_A8', {})
    for a in cfg['arms']:
        if a not in o:
            continue
        v = o[a]
        rel = {m: (v[m] / o['A4'][m] - 1) if 'A4' in o and o['A4'][m] else None for m in ('first10motor_rmse', 'future_latent_nmse')}
        g = gap.get(a, {})
        lines.append(f"| {a} | " + ' | '.join(fmt(v[m]) for m in SHOW)
                     + f" | {pct(rel['first10motor_rmse'])} | {pct(rel['future_latent_nmse'])} | "
                     f"{pct(g.get('first10motor_rmse'))} | {pct(g.get('future_latent_nmse'))} | {s['seconds_per_query'][a]:.2f} |")
    for name, r in cfg['references'].items():
        lines.append(f"| 参照 {name} | {fmt(r['first10motor_rmse'])} | {fmt(r['future_latent_nmse'])} |  |  |  |  |  |  |  |  |  |")
    lines.append('')
    d3 = s.get('d3_decision')
    if d3:
        lines += [f"**D3 判据**（预注册：最佳跨步臂在 motor 与 future 上都补回 ≥50% 的 A4→A8 差距，且都低于 A4_g32）："
                  f"最佳跨步臂 `{d3['best_temporal_arm']}`，补回 motor {pct(d3['gap_closure']['first10motor_rmse'])}、"
                  f"future {pct(d3['gap_closure']['future_latent_nmse'])}；低于 g32：{d3['beats_g32']}；**go = {d3['go']}**。", '']
    if s.get('mechanism_dither'):
        md = s['mechanism_dither']
        verdict = ("支持：RTN 误差的跨步相干累积存在" if md['future_latent_nmse'] < 0
                   else "不支持：打散跨步相关没有带来收益，相干累积不是主要机制")
        lines += [f"**机制对照**：A4_dither − A4 = motor {md['first10motor_rmse']:+.4f}、future {md['future_latent_nmse']:+.4f}"
                  f"（dither 每步误差能量约为 RTN 的 2 倍但跨步不相关；预注册判据为 future 差值 < 0）。结论：{verdict}。", '']
    for ref_arm, table in s.get('paired_contexts', {}).items():
        lines += [f'## 逐 context 配对（40 contexts，两 seed 平均；better = 左边误差更低），对照 {ref_arm}', '',
                  '| 臂 | motor better/worse/tied | future better/worse/tied |', '|---|---:|---:|']
        for a, v in table.items():
            m, f = v['first10motor_rmse'], v['future_latent_nmse']
            lines.append(f"| {a} | {m['better']}/{m['worse']}/{m['tied']} | {f['better']}/{f['worse']}/{f['tied']} |")
        lines.append('')
    lines += ['## 逐 suite（first10 motor / future NMSE）', '', '| 臂 | ' + ' | '.join(cfg['suites']) + ' |',
              '|---|' + '---:|' * len(cfg['suites'])]
    for a in cfg['arms']:
        if a in s['by_suite']:
            b = s['by_suite'][a]
            lines.append(f'| {a} | ' + ' | '.join(f"{b[x]['first10motor_rmse']:.4f} / {b[x]['future_latent_nmse']:.4f}"
                                                   for x in cfg['suites']) + ' |')
    lines.append('')
    d1 = s.get('d1')
    if d1:
        lines += ['## D1：去噪步之间的 activation 结构（BF16 前向，10 个 C80 context）', '',
                  f"Linear×shape 分类：{d1['counts']}；调用/静态复用次数模式：{d1['call_patterns']}。", '',
                  '量的定义：innovation 比 = ‖z_s − z_{s−1}‖/‖z_s‖；absmax 比 = 每 token 的 amax|z_s − z_{s−1}| / amax|z_s| 的均值（直接决定 innovation 编码后的 A4 步长）；'
                  '等效增益位数 = −log2(absmax 比)；残差相关 = 相邻两步 A4 RTN 残差的余弦。表中为该步所有 Linear 的中位数。', '',
                  '| 步 s | video innovation 比 | video absmax 比 | video 增益位数 | video 残差相关 | action innovation 比 | action absmax 比 | action 增益位数 | action 残差相关 |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for step in range(1, 11):
            row = [d1['by_step'].get(c, {}).get(str(step)) or d1['by_step'].get(c, {}).get(step) for c in LOOPS]
            cells = []
            for r in row:
                cells += [fmt(r[f], 3) if r else '—' for f in ('innovation_ratio', 'absmax_ratio', 'bits_gain', 'residual_corr')]
            label = f'{step}（prefill）' if step == 10 else str(step)
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines += ['', '按 token 组（步 1–9 汇总，四分位 p25 / 中位 / p75）：', '',
                  '| 循环 | 组 | innovation 比 | absmax 比 | 残差相关 |', '|---|---|---|---|---|']
        q = lambda v: '—' if not v else ' / '.join(f'{x:.3f}' for x in v)
        for cat, groups in d1['by_group'].items():
            for lab, v in groups.items():
                lines.append(f"| {cat} | {lab} | {q(v['innovation_ratio'])} | {q(v['absmax_ratio'])} | {q(v['residual_corr'])} |")
        dd = d1['decision']
        lines += ['', f"两循环合并（步 1–9）：innovation 比中位 {fmt(dd['median_innovation_ratio'], 3)}，残差相关中位 "
                  f"{fmt(dd['median_residual_corr'], 3)}；按 GEMM FLOP 加权的 innovation 比 {d1['flop_weighted_innovation_ratio']}。"
                  f"**D1 判据**（innovation 比 < 0.5 或残差相关 > 0.5）：**go = {dd['go']}**。", '']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    cfg = read(args.protocol)
    if cfg['protocol'] != PROTOCOL:
        raise ValueError('Protocol identity differs')
    s = run(cfg)
    if not s['complete']:
        raise SystemExit(f"Only {s['arms_complete']}/{s['arms_expected']} arms complete")


if __name__ == '__main__':
    main()
