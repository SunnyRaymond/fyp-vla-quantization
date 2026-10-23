"""Paired full-cohort module-budget attribution and explicit byte tradeoffs; CPU only."""
import csv
import json
import math
import os
from pathlib import Path
import statistics
from common import PROTOCOL, contexts, guard, save
from evaluate import variants, key

METRICS = ('future_nmse', 'teacher_first10', 'teacher_full32', 'teacher_late22',
           'teacher_gripper', 'native_first10')
TEACHER = dict(teacher_first10='first10motor_rmse', teacher_full32='full32motor_rmse',
               teacher_late22='late22motor_rmse', teacher_gripper='first10gripper_rmse')


def values(row, baseline=False):
    reader = row['baseline_teacher' if baseline else 'teacher']
    native = row['baseline_native' if baseline else 'native']
    return dict(future_nmse=row['baseline_future_nmse' if baseline else 'future_nmse'],
        native_first10=native['first10motor_rmse'], **{k: reader[v] for k, v in TEACHER.items()})


def aggregate(rows, metrics=METRICS):
    groups = {}
    for row in rows:
        groups.setdefault(str(row['case_id']), []).append(row)
    cases = {c: {m: statistics.mean(r[m] for r in group) for m in metrics} for c, group in groups.items()}
    return dict(case_count=len(cases), context_count=len(rows), per_case=cases,
        mean={m: statistics.mean(v[m] for v in cases.values()) for m in metrics},
        median={m: statistics.median(v[m] for v in cases.values()) for m in metrics})


def contrast(left, right, np, seed):
    if set(left['per_case']) != set(right['per_case']):
        raise RuntimeError('Unpaired cases')
    cases = sorted(left['per_case'], key=int)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cases), size=(5000, len(cases)))
    result = {}
    for m in sorted(set(left['mean']) & set(right['mean'])):
        delta = {c: left['per_case'][c][m] - right['per_case'][c][m] for c in cases}
        vector = np.array(list(delta.values()), dtype=float)
        interval = np.quantile(vector[draws].mean(1), [.025, .975]).tolist()
        result[m] = dict(mean_delta=float(vector.mean()), median_delta=float(np.median(vector)),
            min_delta=float(vector.min()), max_delta=float(vector.max()),
            improved=int((vector < 0).sum()), worsened=int((vector > 0).sum()), tied=int((vector == 0).sum()),
            descriptive_case_bootstrap_95=interval, per_case_delta=delta)
    return result


def main():
    guard()
    import numpy as np
    root = Path(os.environ['ROOT']); p = json.loads((root / 'protocol.json').read_text())
    prepared = json.loads((root / 'results/prepare/summary.json').read_text())
    norms = json.loads((root / 'results/prepare/norms.json').read_text())
    storage = json.loads((root / 'results/prepare/storage.json').read_text())
    labels = [v[0] for v in variants(p)]
    rows, baseline, seen, reused = [], {}, set(), 0
    for shard in range(4):
        directory = root / 'results' / f'evaluate-{shard}'
        receipt = json.loads((directory / 'summary.json').read_text())
        if (receipt['protocol'] != PROTOCOL or receipt['status'] != 'complete' or receipt['shard'] != shard
                or receipt['row_count'] != [192, 192, 168, 168][shard]):
            raise RuntimeError('Incomplete evaluation shard')
        if json.loads((directory / 'step_norms.json').read_text()) != norms:
            raise RuntimeError('Module norms differ across GPU shards')
        for c in (v for v in contexts(p) if v['shard'] == shard):
            for label in labels:
                r = json.loads((directory / f'{key(c, label)}.json').read_text())
                identity = c['case_id'], c['seed_index'], label
                if (r['protocol'] != PROTOCOL or r['context'] != c or r['label'] != label or identity in seen
                        or not r['teacher_reader_unquantized'] or not r['future_injection_verified']):
                    raise RuntimeError('Condition identity or reader changed')
                seen.add(identity); reused += int(r['reused'])
                v, b = values(r), values(r, True)
                if not all(math.isfinite(x) and x >= 0 for x in [*v.values(), *b.values()]):
                    raise RuntimeError('Nonfinite or invalid fidelity metric')
                ck = c['case_id'], c['seed_index']
                if ck in baseline and baseline[ck]['metrics'] != b:
                    raise RuntimeError('Baseline differs between conditions')
                baseline[ck] = dict(context=c, metrics=b)
                rows.append(dict(**c, label=label, reused=r['reused'], **v))
    if len(rows) != 720 or len(baseline) != 60 or reused != prepared['reused_condition_count']:
        raise RuntimeError('Full720-condition cohort with all bitwise reusable conditions required')
    # A no-additional-bpw checkpoint is a necessary cost reference; reuse its reader records.
    reference = Path(p['future_reader_root'])
    manifest = json.loads((reference / 'input_manifest.json').read_text())
    entries = {(e['case_id'], e['seed_index']): e for e in manifest['contexts']}
    zrows = []
    for c in contexts(p):
        e = entries[c['case_id'], c['seed_index']]
        z = json.loads((reference / 'results' / f"teacher-{e['shard']}" /
            f"case_{c['case_id']}_seed_{c['seed_index']}_action_latent.json").read_text())
        if (z['reader'] != 'bf16' or z['donor'] != 'action_latent' or not z['future_injection_verified']
                or z['sampler_seed'] != e['sampler_seed'] or z['action_dit_calls'] != 10 or z['video_dit_calls'] != 0):
            raise RuntimeError('Zero-extra-byte Z reference differs')
        zrows.append(dict(**c, **{k: z[v] for k, v in TEACHER.items()}))
    tables, comparisons = {}, {}
    seed = p['sample_seed'] + 1024
    for split in ('calibration', 'selection', 'test'):
        expected = {'calibration': (8, 8), 'selection': (4, 8), 'test': (22, 44)}[split]
        arms = {label: aggregate([r for r in rows if r['split'] == split and r['label'] == label]) for label in labels}
        arms['baseline_a'] = aggregate([dict(**v['context'], **v['metrics']) for v in baseline.values()
                                        if v['context']['split'] == split])
        arms['zero_cost_z'] = aggregate([r for r in zrows if r['split'] == split], tuple(TEACHER))
        for label, arm in arms.items():
            if (arm['case_count'], arm['context_count']) != expected:
                raise RuntimeError('Split denominator changed')
            tables[f'{split}/{label}'] = arm
        pairs = []
        for profile in p['allocation_profiles']:
            for level in range(2):
                pairs.extend([(f'{profile}_untied_{level}', f'{profile}_shared_{level}'),
                              (f'{profile}_untied_{level}', f'{profile}_random_{level}')])
                for kind in ('shared', 'untied', 'random'):
                    pairs.extend([(f'{profile}_{kind}_{level}', 'baseline_a'),
                                  (f'{profile}_{kind}_{level}', 'zero_cost_z')])
        for level in range(2):
            pairs.extend((f'ubudget_{kind}_{level}', f'sbudget_{kind}_{level}') for kind in ('shared', 'untied', 'random'))
        for left, right in pairs:
            comparisons[f'{split}/{left}-minus-{right}'] = contrast(arms[left], arms[right], np, seed)
    previous = json.loads((Path(p['previous_root']) / 'results/analysis/analysis.json').read_text())
    attribution = {}
    for level in range(2):
        attribution[str(level)] = {}
        for m in METRICS:
            cell = {f'{profile}/{kind}': tables[f'test/{profile}_{kind}_{level}']['mean'][m]
                    for profile in p['allocation_profiles'] for kind in ('shared', 'untied')}
            original = cell['ubudget/untied'] - cell['sbudget/shared']
            direction_u = cell['ubudget/untied'] - cell['ubudget/shared']
            allocation_s = cell['ubudget/shared'] - cell['sbudget/shared']
            direction_s = cell['sbudget/untied'] - cell['sbudget/shared']
            allocation_u = cell['ubudget/untied'] - cell['sbudget/untied']
            if not math.isclose(original, direction_u + allocation_s, abs_tol=1e-12):
                raise RuntimeError('Factorial attribution identity failed')
            old_gap = previous['tables'][f'test/untied_{level}']['mean'][m] - previous['tables'][f'test/shared_{level}']['mean'][m]
            attribution[str(level)][m] = dict(matched_diagonal_gap=original,
                previous_original_gap=old_gap, snap_change_in_diagonal_gap=original-old_gap,
                direction_at_ubudget=direction_u,
                allocation_at_shared=allocation_s, direction_at_sbudget=direction_s,
                allocation_at_untied=allocation_u, interaction=direction_u - direction_s)
    result = dict(protocol=PROTOCOL, phase='analyze', status='complete', job_id=os.environ['PBS_JOBID'],
        row_count=720, total_contexts=60, test_case_count=22, test_context_count=44,
        reused_conditions=reused, newly_evaluated_conditions=720-reused, new_queries=2*(720-reused), new_gradient_queries=0,
        all_module_norm_receipts_match=True, finite_nonnegative_metrics=True,
        tables=tables, comparisons=comparisons, test_attribution=attribution, storage=storage,
        inference_scope='Internal development cases; case-bootstrap intervals descriptive, not simultaneous or independent confirmation. No p-values or outlier removal.',
        software=dict(numpy=np.__version__, bootstrap_repeats=5000, bootstrap_seed=seed), claim_scope=p['claim_scope'])
    out = root / 'results/analysis'; out.mkdir(parents=True, exist_ok=True)
    save(out / 'analysis.json', result)
    with (out / 'paired_contexts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    primary = {k: {m: {q: x for q, x in v.items() if q != 'per_case_delta'} for m, v in value.items()}
               for k, value in comparisons.items() if k.startswith('test/') and
               ('untied_0-minus-' in k or 'ubudget_shared_0-minus-sbudget_shared_0' in k)}
    summary = {k: v for k, v in result.items() if k not in ('tables', 'comparisons')}
    summary['primary_test_contrasts'] = primary
    save(Path(os.environ['ARTIFACTS']) / 'summary.json', summary)
    compact = dict(summary, tables={k: {q: v for q, v in value.items() if q != 'per_case'}
                                    for k, value in tables.items()},
        test_contrasts={k: {m: {q: x for q, x in v.items() if q != 'per_case_delta'}
                           for m, v in value.items()} for k, value in comparisons.items() if k.startswith('test/')})
    save(out / 'compact.json', compact)
    lines = ['# Module-matched finite intervention', '',
        f'完整60 contexts、720条件；{reused}条件经过bitwise权重核对后复用，{720-reused}条件新评估。Test22cases/44contexts。',
        '主分析epsilon0.005；0.015为稳定性检查。逐case平均seeds后等权。所有数值越低越好。', '',
        '| Test arm | Future NMSE | Fixed-reader first10 | Full32 | Late22 | Gripper | Native first10 | Added MiB | Added bpw over614 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in ['baseline_a', 'zero_cost_z'] + labels:
        v = tables[f'test/{label}']['mean']; cost = storage['arms'].get(label, dict(added_MiB=0, added_bpw_full614=0))
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v[m]:.9g}' if m in v else '—' for m in METRICS) +
                     f" | {cost['added_MiB']:.6g} | {cost['added_bpw_full614']:.9g} |")
    lines += ['', '| Test paired contrast | Future delta [descriptive95%] | Future wins | Motor delta [descriptive95%] | Motor wins |',
              '|---|---:|---:|---:|---:|']
    for name, v in comparisons.items():
        if not name.startswith('test/'):
            continue
        f = v.get('future_nmse'); m = v['teacher_first10']
        render = lambda x: f"{x['mean_delta']:.9g} [{x['descriptive_case_bootstrap_95'][0]:.9g}, {x['descriptive_case_bootstrap_95'][1]:.9g}]"
        lines.append(f"| {name[5:]} | {render(f) if f else '—'} | {str(f['improved'])+'/22' if f else '—'} | {render(m)} | {m['improved']}/22 |")
    lines += ['', '完整calibration/selection/per-case表及代数分解保留在analysis.json，不能把交互强行压成唯一归因百分比。',
        f"基础614目标weights={storage['total_target_weight_count']}，基础明确表示bytes={storage['base_total_bytes']}，effective bpw={storage['base_effective_bpw']:.9g}。",
        storage['scope'], '', result['inference_scope'], '', p['claim_scope']]
    (out / 'REPORT.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Paired-case signs and heterogeneity are visible without a normality test or dropped outliers.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for level, ax in enumerate(axes):
        for profile in p['allocation_profiles']:
            v = comparisons[f'test/{profile}_untied_{level}-minus-{profile}_shared_{level}']
            f, m = v['future_nmse']['per_case_delta'], v['teacher_first10']['per_case_delta']
            ax.scatter([f[c] for c in f], [m[c] for c in f], label=profile, alpha=.75, s=24)
        ax.axhline(0, color='grey', linewidth=.8); ax.axvline(0, color='grey', linewidth=.8)
        ax.set(title=f"epsilon {p['intervention_step_norms'][level]}", xlabel='Untied - shared: future NMSE', ylabel='Untied - shared: motor RMSE')
        ax.legend()
    fig.savefig(out / 'paired_effects.png', dpi=180); fig.savefig(out / 'paired_effects.svg'); plt.close(fig)
    print(f'MODULE_MATCHED_ANALYSIS_COMPLETE rows=720 test_cases=22 new_queries={2*(720-reused)}', flush=True)


if __name__ == '__main__':
    main()
