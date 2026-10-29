"""Aggregate phase 1a: reused Q0_A4 plus Q0_A16 / Q0_A8 / Q0_H_INT8_A8 into summary.json and RESULTS.zh.md."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from worker import ARMS, METRICS, PROTOCOL, aggregate, read, save

ALL = ('Q0_A4',) + ARMS
KEY = ('first10motor_rmse', 'future_latent_nmse', 'full32motor_rmse', 'late22motor_rmse',
       'first10gripper_rmse', 'first10gripper_disagreement')
LABEL = {'first10motor_rmse': 'first10 motor RMSE', 'first10motor_mse': 'first10 motor MSE',
         'future_latent_nmse': 'future NMSE', 'full32motor_rmse': 'full32 motor RMSE',
         'late22motor_rmse': 'late22 motor RMSE', 'first10gripper_rmse': 'first10 gripper RMSE',
         'first10gripper_disagreement': 'gripper 判定不一致率', 'mixed_az_loss': 'A+Z mixed'}
PAIRS = (('Q0_A8', 'Q0_A4'), ('Q0_A8', 'Q0_A16'), ('Q0_H_INT8_A8', 'Q0_A8'))


def change(a, b):
    return 100 * (a / b - 1) if b else None


def endpoint_dir(arm, root, cfg):
    return Path(cfg['q0_reference']['endpoints']) if arm == 'Q0_A4' else root / 'endpoints' / arm


def context_means(arm, root, cfg, metric):
    ev, d = cfg['evaluation'], endpoint_dir(arm, root, cfg)
    return {c: sum(read(d / f'{c}_{s}.json')[metric] for s in ev['seed_indices']) / len(ev['seed_indices'])
            for c in ev['case_ids']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    cfg = read(parser.parse_args().protocol)
    ev = cfg['evaluation']
    root = Path(cfg['remote_root']) / 'results'
    worker = read(root / 'worker_0/summary.json')
    if not worker.get('complete') or worker.get('anchor_max_abs', 1) > cfg['gates']['anchor_max_abs']:
        raise ValueError('Worker completion or source anchor gate failed')
    arms = {a: read(root / f'arms/{a}.json')['heldout'] for a in ARMS}
    reused = [read(endpoint_dir('Q0_A4', root, cfg) / f'{c}_{s}.json') for c in ev['case_ids'] for s in ev['seed_indices']]
    arms['Q0_A4'] = aggregate(reused, ev['case_ids'], ev['seed_indices'], cfg['suites'])
    if any(v['queries'] != 80 for v in arms.values()):
        raise ValueError('Every arm needs 80 heldout queries')
    overall = {a: arms[a]['overall'] for a in ALL}
    q0 = overall['Q0_A4']
    reuse_diff = max(abs(q0[m] - cfg['q0_reference']['overall'][m]) for m in METRICS)

    decomposition = {}
    for m in ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'full32motor_rmse'):
        a4, a16, a8 = q0[m], overall['Q0_A16'][m], overall['Q0_A8'][m]
        decomposition[m] = dict(
            a8_recovered_fraction=(a4 - a8) / (a4 - a16) if a4 != a16 else None,
            a8_vs_a16_percent=change(a8, a16),
            h_int8_cost_vs_a8_percent=change(overall['Q0_H_INT8_A8'][m], a8))

    paired = {}
    for left, right in PAIRS:
        paired[f'{left}_vs_{right}'] = {}
        for m in ('first10motor_rmse', 'future_latent_nmse'):
            x, y = context_means(left, root, cfg, m), context_means(right, root, cfg, m)
            paired[f'{left}_vs_{right}'][m] = dict(
                better=sum(x[c] < y[c] for c in x), worse=sum(x[c] > y[c] for c in x),
                tied=sum(x[c] == y[c] for c in x), contexts=len(x))

    weights = read(root / 'weight_diagnostic.json')
    summary = dict(protocol=PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), status='complete', complete=True,
                   arms_complete=len(ARMS), reused_arms=['Q0_A4'], heldout_queries_per_arm=80,
                   new_teacher_queries=0, training_updates=0, closedloop_episodes=0, overall=overall,
                   by_suite={a: arms[a]['by_suite'] for a in ALL},
                   change_vs_Q0_A4_percent={a: {m: change(overall[a][m], q0[m]) for m in METRICS} for a in ARMS},
                   decomposition=decomposition, paired_contexts=paired, weight_diagnostic=weights,
                   q0_a4_reuse=dict(source=cfg['q0_reference']['source'], overall_max_abs_vs_recorded=reuse_diff),
                   startup=read(root / 'worker_0/startup.json'), counters=worker['counters'], scope=cfg['scope'])
    save(root / 'summary.json', summary)

    f = lambda v: '—' if v is None else f'{v:.6f}'
    p = lambda v: '—' if v is None else f'{v:+.2f}%'
    lines = ['# 阶段 1a：Q0 的 A16 / A8 / 只用 H + INT8 行格点（固定输入）', '',
             f'作业 `{summary["pbs_jobid"]}`。held-out 40 contexts × 2 seeds，每臂 80 query；先 seed、再 context、再 suite、'
             '四 suite 等权。Q0_A4 复用 Q-Clip 作业的 80 条 Q0 endpoint，未重跑。decoded BF16 权重 + fake-quant activation。', '',
             '## 主表', '',
             '| 臂 | ' + ' | '.join(LABEL[m] for m in KEY) + ' |', '|---|' + '---:|' * len(KEY)]
    for a in ALL:
        lines.append(f'| {a} | ' + ' | '.join(f(overall[a][m]) for m in KEY) + ' |')
    lines += ['', '## 相对 Q0_A4 的变化', '',
              '| 臂 | ' + ' | '.join(LABEL[m] for m in KEY) + ' |', '|---|' + '---:|' * len(KEY)]
    for a in ARMS:
        lines.append(f'| {a} | ' + ' | '.join(p(summary['change_vs_Q0_A4_percent'][a][m]) for m in KEY) + ' |')
    lines += ['', '## 分解', '',
              '| 指标 | A4→A8 回收比例 (A4−A8)/(A4−A16) | A8 相对 A16 | 只用 H + INT8 行格点相对 Q0_A8 |',
              '|---|---:|---:|---:|']
    for m, d in decomposition.items():
        frac = d['a8_recovered_fraction']
        lines.append(f'| {LABEL[m]} | {"—" if frac is None else f"{100 * frac:.1f}%"} | '
                     f'{p(d["a8_vs_a16_percent"])} | {p(d["h_int8_cost_vs_a8_percent"])} |')
    lines += ['', '## 权重级诊断（614 层，按原权重能量归一化）', '',
              f'- VQ 本身误差 NMSE：{weights["vq_nmse"]:.3e}',
              f'- INT8 行取整误差 NMSE：R0 坐标 {weights["int8_row_r0_nmse"]:.3e}，H 坐标 {weights["int8_row_h_nmse"]:.3e}',
              f'- 逐层 INT8(H)/VQ 误差比：中位 {weights["int8_h_over_vq_median"]:.3e}，最大 {weights["int8_h_over_vq_max"]:.3e}',
              '', '## 逐 context 配对（40 contexts，先平均两 seeds；better = 左边误差更低）', '',
              '| 对比 | motor better/worse/tied | future better/worse/tied |', '|---|---:|---:|']
    for k, v in paired.items():
        mm, zz = v['first10motor_rmse'], v['future_latent_nmse']
        lines.append(f'| {k} | {mm["better"]}/{mm["worse"]}/{mm["tied"]} | {zz["better"]}/{zz["worse"]}/{zz["tied"]} |')
    lines += ['', '## 逐 suite（first10 motor RMSE / future NMSE）', '',
              '| 臂 | ' + ' | '.join(cfg['suites']) + ' |', '|---|' + '---:|' * len(cfg['suites'])]
    for a in ALL:
        cells = [f"{summary['by_suite'][a][s]['first10motor_rmse']:.5f} / {summary['by_suite'][a][s]['future_latent_nmse']:.5f}"
                 for s in cfg['suites']]
        lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 来源绑定', '',
              f'- 启动时 Q0 case 2000 anchor 最大差：{worker["anchor_max_abs"]:g}。',
              f'- 复用的 Q0_A4 聚合与 Q-Clip 记录值最大差：{reuse_diff:.3g}。',
              '- 坐标改写检查（随机输入单层，相对误差）：' + ', '.join(
                  f'{k}={v:.2e}' for k, v in summary['startup']['transform_check'].items() if k != 'module') + '。',
              '- 每臂的 case 2000 eager anchor 存于 `results/anchors/<arm>.pt`，供 engine 评测器启动检查。']
    (root / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('A8ACC_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
