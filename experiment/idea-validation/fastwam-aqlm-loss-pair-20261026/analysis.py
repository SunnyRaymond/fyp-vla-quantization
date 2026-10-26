"""Aggregate all frozen paired fixed-input records; never submits evaluation."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics

from worker import PROTOCOL, ARMS, CASES, HELDOUT, read, save, validate


def aggregate(rows, suites):
    if not rows:
        raise ValueError('No evaluation records')
    keys = tuple(rows[0]['metrics'])
    contexts = {}
    for row in rows:
        contexts.setdefault(row['case_id'], []).append(row)
        if set(row['metrics']) != set(keys) or any(not math.isfinite(v) for v in row['metrics'].values()):
            raise ValueError('Metric schema or finite gate failed')
    means = {}
    for case, values in contexts.items():
        if len({v['suite'] for v in values}) != 1:
            raise ValueError('Context suite changed across seeds')
        means[case] = dict(suite=values[0]['suite'],
            metrics={k:statistics.mean(v['metrics'][k] for v in values) for k in keys})
    by_suite = {}
    for suite in suites:
        values = [v['metrics'] for v in means.values() if v['suite'] == suite]
        if not values:
            raise ValueError('A suite is missing')
        by_suite[suite] = {k:statistics.mean(v[k] for v in values) for k in keys}
    return dict(queries=len(rows),contexts=len(means),by_context=means,by_suite=by_suite,
        mean={k:statistics.mean(by_suite[s][k] for s in suites) for k in keys})


def paired(control, ours):
    if set(control['by_context']) != set(ours['by_context']):
        raise ValueError('Unpaired context coverage')
    result = {}
    for key, baseline in control['mean'].items():
        deltas=[ours['by_context'][c]['metrics'][key]-control['by_context'][c]['metrics'][key]
                for c in control['by_context']]
        value=ours['mean'][key]
        result[key]=dict(reconstruction=baseline,ours=value,delta=value-baseline,
            delta_percent=100*(value-baseline)/baseline if baseline else None,
            median_paired_delta=statistics.median(deltas),
            wins=sum(d < -1e-12 for d in deltas),losses=sum(d > 1e-12 for d in deltas),
            ties=sum(abs(d) <= 1e-12 for d in deltas))
    return result


def main():
    import sampler
    sampler._require_compute_allocation()
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args()
    p=read(args.protocol)
    validate(p)
    root=Path(p['remote_root'])/'results'
    arms,aggregates={},{}
    prepare=read(root/'prepare/summary.json')
    if not prepare.get('complete') or prepare['fit_banks'] != 614 or prepare['calibration_contexts'] != 80:
        raise ValueError('Official shared AQLM initialization is incomplete')
    for arm in ARMS:
        directory=root/'arms'/arm
        summary=read(directory/'summary.json')
        if (summary['protocol'] != PROTOCOL or summary['arm'] != arm or not summary['complete']
                or summary['updates'] != 8 or summary['exposures'] != 80
                or summary['target_linears'] != 614 or summary['closed_loop_episodes'] != 0):
            raise ValueError('Arm is incomplete')
        setup=read(directory/'setup_checks.json')
        training=read(directory/'training.json')
        if not setup['passed'] or len(training['history']) != 8:
            raise ValueError('Setup or training history incomplete')
        for row in training['history']:
            if row['coverage']['connected'] != 614 or row['coverage']['finite'] != 614 or not row['coverage']['indices_frozen']:
                raise ValueError('Training gradient/frozen-indices gate failed')
        arms[arm]=summary
        aggregates[arm]={}
        for split,cases,seeds in (('calibration',CASES,[0]),('heldout',HELDOUT,[0,1])):
            rows=[]
            for case in cases:
                for seed in seeds:
                    row=read(directory/'evaluation'/split/f'{case}_{seed}.json')
                    if (row['protocol'] != PROTOCOL or row['arm'] != arm or row['case_id'] != case
                            or row['seed_index'] != seed or row['updates'] != 8):
                        raise ValueError('Frozen evaluation record identity differs')
                    rows.append(row)
            actual={path.name for path in (directory/'evaluation'/split).glob('*.json')}
            if actual != {f'{c}_{s}.json' for c in cases for s in seeds}:
                raise ValueError('Extra or missing fixed-input evaluation records')
            aggregates[arm][split]=aggregate(rows,p['suites'])
    histories=[read(root/'arms'/a/'training.json')['history'] for a in ARMS]
    if [r['cases'] for r in histories[0]] != [r['cases'] for r in histories[1]]:
        raise ValueError('Paired training schedules differ')
    initial=[torch.load(root/'arms'/a/'initial_anchor.pt',map_location='cpu',weights_only=True) for a in ARMS]
    initial_error=max(float((initial[0][key].float()-initial[1][key].float()).abs().max()) for key in ('action','future'))
    if initial_error > 1e-6:
        raise ValueError('Arm initializations do not replay identically')
    comparison={split:paired(aggregates['reconstruction'][split],aggregates['ours'][split])
                for split in ('calibration','heldout')}
    save(root/'comparison.json',dict(protocol=PROTOCOL,paired=comparison,initial_max_abs=initial_error,
        aggregates=aggregates,storage=prepare,closed_loop_episodes=0))
    summary=dict(protocol=PROTOCOL,status='complete',complete=True,pbs_jobid=os.environ['PBS_JOBID'],
        arms=arms,target_linears=614,initial_max_abs=initial_error,paired=comparison,
        effective_bpw=prepare['effective_bpw'],weight_encoding_bpw=prepare['weight_encoding_bpw'],
        fit_profile_counts=prepare.get('fit_profile_counts',{}),
        closed_loop_episodes=0,boundary='fixed-input decoded-BF16/fake-A4; no closed-loop or native performance evidence')
    save(root/'summary.json',summary)
    lines=['# AQLM 两臂 fixed-input 消融','',
        '官方 input-aware AQLM 初始化共用；仅 loss 不同，indices 与底模冻结。',
        'Control 为官方 Linear-output reconstruction objective 的 Fast-WAM adaptation，不是 block fine-tuning。',
        f"含共同 preprocessing 的 target-weight effective bpw：{prepare['effective_bpw']:.6f}。",'',
        f"共享初始化来源：{prepare.get('fit_profile_counts',{})}；既有各预算 banks 原样保留，其余采用用户指定 minimal_v3 预算。",'',
        '| Held-out 指标 | Reconstruction | Ours | 相对变化 | Context 胜/负/平 |',
        '|---|---:|---:|---:|---|']
    for key in ('first10motor_rmse','full32motor_rmse','late22motor_rmse','first10gripper_rmse','future_latent_nmse','mixed_loss'):
        row=comparison['heldout'][key]
        delta='n/a' if row['delta_percent'] is None else f"{row['delta_percent']:+.2f}%"
        lines.append(f"| {key} | {row['reconstruction']:.8g} | {row['ours']:.8g} | {delta} | {row['wins']}/{row['losses']}/{row['ties']} |")
    lines += ['', 'Held-out：40 contexts × 2 seeds；先在 context 内平均 seeds，再按四个 suites 等权汇总。',
        '表中 mixed_loss = action MSE + lambda × future NMSE，不含 reader KL；两臂按同一指标比较。',
        '每臂另外完整报告 C80 calibration。固定 final update8，无 held-out checkpoint selection。',
        '闭环 episodes = 0；停止在 280-episode 闭环之前。结果不代表 native-kernel latency 或 closed-loop success。','']
    (root/'report.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print('AQLM_FIXED_INPUT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
