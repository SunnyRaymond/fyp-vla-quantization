"""Full-cohort, case-paired analysis; CPU allocation only, no model forwards."""
from pathlib import Path
import argparse
import csv
import json
import os
import statistics
from common import PROTOCOL, DONORS, contexts, stem, aggregate, paired, guard, save

METRICS = ('first10motor_rmse', 'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse')


def collect(p, phase):
    root, rows = Path(p['remote_root']), []
    readers = [('teacher', 'bf16')] if phase == 'teacher' else [('cross_a', 'action_only'), ('cross_z', 'action_latent')]
    for prefix, reader in readers:
        donors = DONORS if prefix == 'teacher' else ('action_only', 'action_latent')
        for shard in (0, 1):
            d = root / 'results' / f'{prefix}-{shard}'
            s = json.loads((d / 'summary.json').read_text())
            if (s['protocol'] != PROTOCOL or s['status'] != 'complete' or s['reader'] != reader
                    or s['shard'] != shard or s['row_count'] != 30 * len(donors) or s['context_count'] != 30):
                raise RuntimeError('Cannot analyze a partial/mismatched shard')
        for c in contexts(p):
            d = root / 'results' / f"{prefix}-{c['shard']}"
            for donor in donors:
                r = json.loads((d / f'{stem(c)}_{donor}.json').read_text())
                if (r['context'] != c or r['protocol'] != PROTOCOL or r['reader'] != reader or r['donor'] != donor
                        or not r['future_injection_verified'] or r['video_dit_calls'] != 0 or r['action_dit_calls'] != 10
                        or not (d / f'{stem(c)}_{donor}.pt').is_file()):
                    raise RuntimeError('Intervention row identity/implementation check failed')
                if reader == donor and r['replay_parity'] is None:
                    raise RuntimeError('Missing diagonal parity')
                rows.append(r)
    if len(rows) != 240:
        raise RuntimeError('Full phase must contain240 rows')
    return rows


def summarize(p, rows, phase):
    tables, comparisons = {}, {}
    readers = sorted({r['reader'] for r in rows})
    for split in ('calibration', 'selection', 'test'):
        for reader in readers:
            for donor in sorted({r['donor'] for r in rows}):
                subset = [r for r in rows if r['split'] == split and r['reader'] == reader and r['donor'] == donor]
                if not subset:
                    continue
                name = f'{split}/{reader}/{donor}'
                tables[name] = {m: aggregate(subset, m) for m in METRICS}
                expected_cases, expected_contexts = {'calibration': (8, 8), 'selection': (4, 8), 'test': (22, 44)}[split]
                if any(v['case_count'] != expected_cases or v['context_count'] != expected_contexts for v in tables[name].values()):
                    raise RuntimeError('Split denominator changed')
            for left, right in [('action_latent', 'action_only')] + ([('action_only', 'parent'), ('action_latent', 'parent')] if phase == 'teacher' else []):
                a = [r for r in rows if r['split'] == split and r['reader'] == reader and r['donor'] == left]
                b = [r for r in rows if r['split'] == split and r['reader'] == reader and r['donor'] == right]
                comparisons[f'{split}/{reader}/{left}-minus-{right}'] = {m: paired(a, b, m) for m in METRICS}
    gate = None
    if phase == 'teacher':
        primary = comparisons['test/bf16/action_latent-minus-action_only']['first10motor_rmse']
        gate = primary['mean_delta'] < 0 and primary['improved'] >= 12
    return dict(protocol=PROTOCOL, phase='analyze_' + phase, status='complete', job_id=os.environ.get('PBS_JOBID'),
                row_count=len(rows), total_contexts=60, test_case_count=22, test_context_count=44,
                model_forward_count=0, tables=tables, comparisons=comparisons, cross_gate_passed=gate,
                cross_gate_definition=p['cross_gate'], claim_scope=p['claim_scope'])


def main():
    guard()
    a = argparse.ArgumentParser()
    a.add_argument('--protocol', type=Path, required=True)
    a.add_argument('--phase', choices=('teacher', 'cross'), required=True)
    args = a.parse_args()
    p = json.loads(args.protocol.read_text())
    rows = collect(p, args.phase)
    result = summarize(p, rows, args.phase)
    out = Path(p['remote_root']) / 'results' / ('analysis_' + args.phase)
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'analysis.json', result)
    fields = ['split', 'reader', 'donor', 'case_id', 'seed_index', 'sampler_seed', 'domain', 'dimension'] + list(METRICS)
    with (out / 'paired_contexts.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Fixed-reader future intervention', '', '完整60 contexts；test22 cases/44 contexts。误差均相对同输入同seed的BF16 action。', '',
             '| Split | Reader | Future donor | first10 | full32 | late22 | gripper |', '|---|---|---|---:|---:|---:|---:|']
    for name, metrics in result['tables'].items():
        lines.append('| ' + name.replace('/', ' | ') + ' | ' + ' | '.join(f"{metrics[m]['mean']:.9f}" for m in METRICS) + ' |')
    lines += ['', '配对差值为前者减后者；负值表示改善。', '', '| Contrast | First10 mean delta | Improved | Worsened | Median delta |', '|---|---:|---:|---:|---:|']
    for name, values in result['comparisons'].items():
        if not name.startswith('test/'):
            continue
        v = values['first10motor_rmse']
        lines.append(f"| {name} | {v['mean_delta']:.9f} | {v['improved']}/22 | {v['worsened']}/22 | {statistics.median(v['per_case_delta'].values()):.9f} |")
    if args.phase == 'teacher':
        lines += ['', f"Cross progression gate: {result['cross_gate_passed']}. {p['cross_gate']}"]
    parity = [r['replay_parity'] for r in rows if r['replay_parity'] is not None]
    result['replay_parity_count'] = len(parity)
    result['replay_max_abs'] = max(v['max_abs'] for v in parity)
    save(out / 'analysis.json', result)
    lines += ['', f"Replay checks={len(parity)}, maximum absolute action difference={result['replay_max_abs']:.9g}.", '', p['claim_scope']]
    (out / 'REPORT.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('ANALYSIS_COMPLETE', args.phase, flush=True)


if __name__ == '__main__':
    main()
