"""Aggregate the paired C8 fixed-input latent-anchor-KL evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

PROTOCOL = 'fastwam-vq-c8-latent-anchor-kl-v1'
BASELINE_ARM = 'baseline'
CANDIDATE_ARM = 'anchor_kl'
SPLITS = ('calibration', 'heldout')
COMMON_METRICS = (
    'first10motor_rmse', 'full32motor_rmse', 'late22motor_rmse',
    'first10gripper_rmse', 'future_latent_nmse',
)
PRIMARY_METRICS = ('first10motor_rmse', 'future_latent_nmse')
CANDIDATE_DIAGNOSTICS = (
    'action_mse', 'latent_nmse', 'latent_anchor_kl',
    'baseline_objective', 'composite_objective',
)
BASELINE_COPY_FIELDS = ('new_protocol', 'source_path')


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False,
                                allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get('protocol') != PROTOCOL:
        raise ValueError('Unexpected C8 latent-anchor protocol')
    baseline = protocol.get('baseline_protocol', {})
    baseline_cache = protocol.get('baseline_cache_protocol', {})
    source = baseline_cache.get('source_clipping_protocol', {})
    training = protocol.get('training', {})
    expected = protocol.get('expected', {})
    calibration = protocol.get('calibration', {})
    evaluation = protocol.get('evaluation', {})
    baseline_config = baseline.get('arm_configs', {}).get('rho1.5', {})
    if (protocol.get('baseline_arm') != 'rho1.5' or protocol.get('baseline_updates') != 8
            or baseline.get('remote_root') != protocol.get('baseline_root')
            or baseline_config.get('rho') != 1.5 or baseline_config.get('epochs') != 1
            or baseline_config.get('updates') != 8
            or baseline_cache.get('protocol') != 'fastwam-vq-c80-activation-clipping-v2-original-heldout'
            or baseline_cache.get('remote_root') != protocol.get('baseline_cache_root')
            or source.get('parent_root') != protocol.get('baseline_root')
            or source.get('parent_arm') != 'rho1.5' or source.get('parent_updates') != 8):
        raise ValueError('Cached baseline is not the original C8 rho1.5 eight-update configuration')
    if (training.get('rho') != 1.5 or training.get('epochs') != 1 or training.get('updates') != 8
            or training.get('activation_clipping') is not False
            or training.get('per_channel_scaling') is not False
            or 'codebooks' not in training.get('trainables', '').lower()
            or 'row gains' not in training.get('trainables', '').lower()):
        raise ValueError('Anchor-KL must retain the original C8 banks and train only books/row gains for eight updates')
    if (len(calibration.get('case_ids', [])) != 8 or calibration.get('seed_index') != 0
            or len(evaluation.get('case_ids', [])) != 40
            or evaluation.get('seed_indices') != [0, 1]
            or expected.get('calibration_queries_per_arm') != 8
            or expected.get('heldout_queries_per_arm') != 80
            or expected.get('baseline_new_model_queries') != 0
            or expected.get('new_teacher_queries') != 0):
        raise ValueError('Expected frozen C8 8-query and heldout 40x2-query cohorts with reused baselines')
    if (any(type(value) is not int for value in calibration['case_ids'] + evaluation['case_ids'])
            or len(set(calibration['case_ids'])) != 8 or len(set(evaluation['case_ids'])) != 40
            or set(calibration['case_ids']) & set(evaluation['case_ids'])):
        raise ValueError('Calibration and heldout case identities must be unique and disjoint')
    return protocol


def expected_slots(protocol, split):
    if split == 'calibration':
        cases = protocol['calibration']['case_ids']
        seeds = [protocol['calibration']['seed_index']]
    elif split == 'heldout':
        cases, seeds = protocol['evaluation']['case_ids'], protocol['evaluation']['seed_indices']
    else:
        raise ValueError(f'Unknown evaluation split: {split}')
    return [(case, seed) for case in cases for seed in seeds]


def _validate_plan_row(record, case_id, seed_index):
    row = record.get('plan_row')
    if (type(case_id) is not int or type(seed_index) is not int
            or not isinstance(row, dict) or row.get('case_id') != case_id
            or row.get('suite') != record.get('suite')
            or not isinstance(row.get('sampler_seeds'), list)
            or seed_index >= len(row['sampler_seeds'])
            or row['sampler_seeds'][seed_index] != record.get('sampler_seed')):
        raise ValueError(f'Plan row or sampler seed binding differs for {case_id}_{seed_index}')
    if not isinstance(record.get('suite'), str) or not record['suite']:
        raise ValueError(f'Missing suite for {case_id}_{seed_index}')
    return row


def _validate_record(record, protocol, split, arm, case_id, seed_index, source_protocol):
    if not isinstance(record, dict):
        raise ValueError(f'Expected a JSON object for {split}/{arm}/{case_id}_{seed_index}')
    updates = 0 if arm == BASELINE_ARM else 8
    expected_protocol = source_protocol if arm == BASELINE_ARM else protocol['protocol']
    if (record.get('protocol') != expected_protocol or record.get('arm') != arm
            or type(record.get('updates')) is not int or record.get('updates') != updates
            or type(record.get('case_id')) is not int or record.get('case_id') != case_id
            or type(record.get('seed_index')) is not int or record.get('seed_index') != seed_index):
        raise ValueError(f'Protocol, arm, update or slot identity differs for {split}/{arm}/{case_id}_{seed_index}')
    if record.get('split', split) != split:
        raise ValueError(f'Split identity differs for {split}/{arm}/{case_id}_{seed_index}')
    _validate_plan_row(record, case_id, seed_index)
    for metric in COMMON_METRICS:
        value = record.get(metric)
        if not _finite(value) or value < 0:
            raise ValueError(f'Missing or invalid {metric} for {split}/{arm}/{case_id}_{seed_index}')
    if arm == BASELINE_ARM:
        source = protocol['baseline_cache_protocol']['source_clipping_protocol']
        record_source = record.get('source_clipping_protocol')
        if record_source is not None and (not isinstance(record_source, dict) or any(
                record_source.get(key) != source[key]
                for key in ('parent_root', 'parent_arm', 'parent_updates'))):
            raise ValueError('Baseline query is not bound to the frozen C8 parent checkpoint')
    else:
        for metric in CANDIDATE_DIAGNOSTICS:
            if metric in record and (not _finite(record[metric]) or record[metric] < 0):
                raise ValueError(f'Invalid optional candidate diagnostic {metric}')
        action = record.get('action')
        if action is not None and (not isinstance(action, list) or len(action) != 32
                                   or any(not isinstance(step, list) or len(step) != 7
                                          or any(not _finite(value) for value in step)
                                          for step in action)):
            raise ValueError(f'Candidate action must be a finite 32x7 list for {case_id}_{seed_index}')
    return record


def _load_baseline_copy(protocol, results_root, split, case_id, seed_index):
    cache = (Path(protocol['baseline_cache_root']) / 'results' / split / BASELINE_ARM
             / 'queries' / f'{case_id}_{seed_index}.json')
    copied = (Path(results_root) / BASELINE_ARM / split / 'queries'
              / f'{case_id}_{seed_index}.json')
    original, record = read_json(cache), read_json(copied)
    if (record.get('new_protocol') != protocol['protocol']
            or record.get('source_path') != str(cache)):
        raise ValueError(f'Baseline copy is missing its protocol/path provenance: {case_id}_{seed_index}')
    preserved = {key: value for key, value in record.items() if key not in BASELINE_COPY_FIELDS}
    if preserved != original:
        raise ValueError(f'Copied baseline record was altered: {case_id}_{seed_index}')
    _validate_record(record, protocol, split, BASELINE_ARM, case_id, seed_index,
                     protocol['baseline_cache_protocol']['protocol'])
    return record


def _load_candidate(protocol, results_root, split, case_id, seed_index):
    path = (Path(results_root) / CANDIDATE_ARM / split / 'queries'
            / f'{case_id}_{seed_index}.json')
    record = read_json(path)
    return _validate_record(record, protocol, split, CANDIDATE_ARM, case_id, seed_index,
                            protocol['protocol'])


def load_records(protocol, results_root):
    baseline, candidate = {}, {}
    for split in SPLITS:
        baseline[split], candidate[split] = {}, {}
        for case_id, seed_index in expected_slots(protocol, split):
            key = (case_id, seed_index)
            baseline_record = _load_baseline_copy(protocol, results_root, split, *key)
            candidate_record = _load_candidate(protocol, results_root, split, *key)
            if candidate_record['plan_row'] != baseline_record['plan_row']:
                raise ValueError(f'Paired plan rows differ for {split}/{case_id}_{seed_index}')
            if candidate_record['suite'] != baseline_record['suite']:
                raise ValueError(f'Paired suite differs for {split}/{case_id}_{seed_index}')
            baseline[split][key] = baseline_record
            candidate[split][key] = candidate_record
        if len(baseline[split]) != len(expected_slots(protocol, split)):
            raise ValueError(f'Incomplete {split} baseline/candidate coverage')
    return baseline, candidate


def _case_metric_values(records, split):
    grouped = {}
    for (case_id, seed_index), record in records[split].items():
        grouped.setdefault(case_id, []).append((seed_index, record))
    result = {}
    for case_id, rows in grouped.items():
        rows.sort(key=lambda value: value[0])
        suites = {row['suite'] for _, row in rows}
        if len(suites) != 1:
            raise ValueError(f'Paired case {case_id} changes suite between seeds')
        result[case_id] = {
            'suite': rows[0][1]['suite'],
            'seeds': [seed for seed, _ in rows],
            'metrics': {metric: statistics.mean(record[metric] for _, record in rows)
                        for metric in COMMON_METRICS},
            'records': [record for _, record in rows],
        }
    return result


def _metric_summary(baseline_cases, candidate_cases, case_ids, metric):
    base = [baseline_cases[case]['metrics'][metric] for case in case_ids]
    cand = [candidate_cases[case]['metrics'][metric] for case in case_ids]
    deltas = [right - left for left, right in zip(base, cand)]
    wins = sum(delta < 0 for delta in deltas)
    losses = sum(delta > 0 for delta in deltas)
    return {
        'baseline': {'mean_case_value': statistics.mean(base),
                     'median_case_value': statistics.median(base)},
        'anchor_kl': {'mean_case_value': statistics.mean(cand),
                      'median_case_value': statistics.median(cand)},
        'paired': {'mean_difference_anchor_minus_baseline': statistics.mean(deltas),
                   'median_difference_anchor_minus_baseline': statistics.median(deltas),
                   'wins': wins, 'losses': losses, 'ties': len(deltas) - wins - losses,
                   'cases': len(deltas), 'lower_is_better': True},
    }


def _optional_case_summary(case_records, metric):
    """Summarize an anchor-only diagnostic only when every paired query supplied it."""
    values = {}
    for case_id, case in case_records.items():
        per_seed = [record.get(metric) for record in case['records']]
        if any(not _finite(value) for value in per_seed):
            return {'available': False, 'reason': f'candidate records do not all contain {metric}'}
        values[case_id] = statistics.mean(per_seed)
    suites = sorted({case['suite'] for case in case_records.values()})
    by_suite = {}
    for suite in suites:
        selected = [values[case_id] for case_id, case in case_records.items()
                    if case['suite'] == suite]
        by_suite[suite] = {'cases': len(selected), 'mean_case_value': statistics.mean(selected),
                           'median_case_value': statistics.median(selected)}
    return {'available': True, 'mean_equal_suite': statistics.mean(
                row['mean_case_value'] for row in by_suite.values()),
            'median_of_suite_medians': statistics.median(
                row['median_case_value'] for row in by_suite.values()),
            'by_suite': by_suite}


def _summarize_split(protocol, split, baseline_records, candidate_records):
    slots = expected_slots(protocol, split)
    if (set(baseline_records) != set(slots) or set(candidate_records) != set(slots)):
        raise ValueError(f'{split} comparison must contain exactly {len(slots)} paired query records')
    baseline_cases = _case_metric_values({split: baseline_records}, split)
    candidate_cases = _case_metric_values({split: candidate_records}, split)
    suite_names = sorted({case['suite'] for case in baseline_cases.values()})
    expected_cases_per_suite = 2 if split == 'calibration' else 10
    if (len(suite_names) != 4 or any(sum(case['suite'] == suite for case in baseline_cases.values())
                                    != expected_cases_per_suite for suite in suite_names)):
        raise ValueError(f'{split} must contain four suites with {expected_cases_per_suite} cases each')

    suites = {}
    for suite in suite_names:
        case_ids = sorted(case_id for case_id, case in baseline_cases.items()
                          if case['suite'] == suite)
        suites[suite] = {
            'cases': len(case_ids),
            'metrics': {metric: _metric_summary(baseline_cases, candidate_cases, case_ids, metric)
                        for metric in COMMON_METRICS},
        }
    overall = {}
    for metric in COMMON_METRICS:
        per_suite = [suites[suite]['metrics'][metric] for suite in suite_names]
        paired_differences = [row['paired']['mean_difference_anchor_minus_baseline']
                              for row in per_suite]
        wins = sum(row['paired']['wins'] for row in per_suite)
        losses = sum(row['paired']['losses'] for row in per_suite)
        ties = sum(row['paired']['ties'] for row in per_suite)
        overall[metric] = {
            'baseline': {
                'mean_equal_suite': statistics.mean(row['baseline']['mean_case_value'] for row in per_suite),
                'median_of_suite_medians': statistics.median(
                    row['baseline']['median_case_value'] for row in per_suite),
            },
            'anchor_kl': {
                'mean_equal_suite': statistics.mean(row['anchor_kl']['mean_case_value'] for row in per_suite),
                'median_of_suite_medians': statistics.median(
                    row['anchor_kl']['median_case_value'] for row in per_suite),
            },
            'paired': {
                'mean_difference_equal_suite': statistics.mean(paired_differences),
                'median_of_suite_median_differences': statistics.median(
                    row['paired']['median_difference_anchor_minus_baseline'] for row in per_suite),
                'wins': wins, 'losses': losses, 'ties': ties,
                'cases': wins + losses + ties, 'win_loss_tie_unit': 'case mean over seeds',
                'lower_is_better': True,
            },
        }

    reconstructed_objective = {}
    rho_c = protocol['training']['rho'] * protocol['training']['normalization_c']
    for arm, records in ((BASELINE_ARM, baseline_records), (CANDIDATE_ARM, candidate_records)):
        by_case = {}
        for (case_id, seed), record in records.items():
            value = record['first10motor_rmse'] ** 2 + rho_c * record['future_latent_nmse']
            by_case.setdefault(case_id, []).append((seed, value))
        case_values = {case: statistics.mean(value for _, value in rows)
                       for case, rows in by_case.items()}
        suite_means = [statistics.mean(case_values[case_id] for case_id, case in baseline_cases.items()
                                       if case['suite'] == suite)
                       for suite in suite_names]
        reconstructed_objective[arm] = {
            'mean_equal_suite': statistics.mean(suite_means),
            'median_of_suite_means': statistics.median(suite_means),
            'formula': 'first10motor_rmse^2 + rho * normalization_c * future_latent_nmse',
            'rounding_note': 'RMSE is squared after rounding in the stored record; this is approximate.',
        }

    return {
        'query_records_per_arm': len(slots),
        'cases': len(baseline_cases),
        'seeds_per_case': 1 if split == 'calibration' else 2,
        'overall': overall,
        'suites': suites,
        'primary_metrics': list(PRIMARY_METRICS),
        'gripper_metric_role': 'diagnostic only',
        'case_means': {
            str(case_id): {'suite': baseline_cases[case_id]['suite'],
                           'seeds': baseline_cases[case_id]['seeds'],
                           'baseline': baseline_cases[case_id]['metrics'],
                           'anchor_kl': candidate_cases[case_id]['metrics']}
            for case_id in sorted(baseline_cases)
        },
        'reconstructed_common_objective': reconstructed_objective,
        'candidate_only_diagnostics': {
            metric: _optional_case_summary(candidate_cases, metric)
            for metric in CANDIDATE_DIAGNOSTICS
        },
        'baseline_anchor_kl_unavailable': {
            'available': False,
            'reason': 'Cached baseline records do not contain full future tensors; no new baseline model forwards are allowed.',
        },
        'baseline_composite_objective_unavailable': {
            'available': False,
            'reason': 'The cached baseline has no teacher-anchor KL term or full future tensor; it is not recomputed.',
        },
    }


def summarize(protocol, baseline, candidate, prepare, candidate_summary, pbs_jobid):
    splits = {split: _summarize_split(protocol, split, baseline[split], candidate[split])
             for split in SPLITS}
    return {
        'protocol': protocol['protocol'], 'complete': True, 'pbs_jobid': pbs_jobid,
        'baseline_records': sum(len(baseline[split]) for split in SPLITS),
        'candidate_records': sum(len(candidate[split]) for split in SPLITS),
        'baseline_new_model_queries': 0,
        'candidate_new_teacher_queries': candidate_summary['new_teacher_queries'],
        'candidate_evaluation_queries': candidate_summary.get('evaluation_queries'),
        'candidate_new_model_queries': candidate_summary.get('new_model_queries'),
        'baseline_binding': {
            'protocol': protocol['baseline_cache_protocol']['protocol'],
            'cache_root': protocol['baseline_cache_root'],
            'source_parent_root': protocol['baseline_cache_protocol']['source_clipping_protocol']['parent_root'],
            'source_parent_arm': protocol['baseline_cache_protocol']['source_clipping_protocol']['parent_arm'],
            'source_parent_updates': protocol['baseline_cache_protocol']['source_clipping_protocol']['parent_updates'],
            'baseline_arm': protocol['baseline_arm'], 'baseline_updates': protocol['baseline_updates'],
            'updates_interpretation': '0 additional C80 updates; the reused parent baseline already has C8 rho1.5 updates=8',
            'learned_clipping_or_scaling': False,
        },
        'preparation': {key: prepare[key] for key in ('beta', 'temperature', 'stats', 'normalization_c')},
        'candidate_training': {
            'arm': CANDIDATE_ARM, 'epochs': protocol['training']['epochs'],
            'updates': protocol['training']['updates'], 'rho': protocol['training']['rho'],
            'trainables': protocol['training']['trainables'],
            'activation_clipping': protocol['training']['activation_clipping'],
            'per_channel_scaling': protocol['training']['per_channel_scaling'],
            'initialization': protocol['training']['initialization'],
        },
        'splits': splits,
        'interpretation': (
            'Descriptive paired fixed-input results only. Seeds are averaged within each case, then cases within each suite, '
            'and the four suite means receive equal weight. Win/loss/tie counts use paired case means, not individual seeds. '
            'These repeated queries on the same fixed LIBERO cases are not independent tasks; no p-values or closed-loop '
            'inference are reported. Baseline records are reused cache entries; no baseline model or teacher was rerun.'),
    }


def comparison_view(summary):
    return {
        'protocol': summary['protocol'], 'complete': summary['complete'],
        'pbs_jobid': summary['pbs_jobid'],
        'baseline_records': summary['baseline_records'],
        'candidate_records': summary['candidate_records'],
        'primary_metrics': list(PRIMARY_METRICS),
        'splits': {split: {
            'overall': {metric: summary['splits'][split]['overall'][metric]
                        for metric in COMMON_METRICS},
            'suites': {suite: {'metrics': {metric: row['metrics'][metric]
                                           for metric in COMMON_METRICS}}
                       for suite, row in summary['splits'][split]['suites'].items()},
        } for split in SPLITS},
        'interpretation': summary['interpretation'],
    }


def _fmt(value):
    return '—' if value is None else f'{value:.6g}'


def render_report(summary):
    lines = [
        '# C8 latent-anchor KL：固定输入配对结果', '',
        f"状态：complete；baseline cached queries：{summary['baseline_records']}/88；"
        f"anchor-KL queries：{summary['candidate_records']}/88。",
        'Baseline 新模型 forward：0；新 teacher query：'
        f"{summary['candidate_new_teacher_queries']}。", '',
        '对照是原始 C8 A+Z rho=1.5、1 epoch/8 updates 的冻结结果；目录名包含 activation-clipping '
        '仅表示缓存所在的旧实验目录。Baseline query 的额外 updates=0，父 checkpoint 已完成 C8 rho=1.5 的 8 updates；'
        '不把该缓存解释为新增 clipping 训练。Candidate 从同一原始 C8 bank 初始化，只训练 codebooks 与 log row gains，'
        '普通 dynamic A4，无 learned clipping/scaling。', '',
        '每个 case 先平均 seed，再在 suite 内平均 case，最后四个 suite 等权。胜负按 case-level paired mean 计数；'
        '低值为胜。Gripper RMSE 仅作诊断。', '',
    ]
    for split in SPLITS:
        data = summary['splits'][split]
        lines += [f"## {split}（{data['cases']} cases，{data['query_records_per_arm']} queries/arm）", '',
                  '| Scope | Metric | Baseline | Anchor-KL | Δ (candidate−baseline) | W/L/T cases |',
                  '|---|---|---:|---:|---:|---:|']
        rows = [('Overall · four suites equally weighted', data['overall'])]
        rows.extend((suite, value['metrics']) for suite, value in data['suites'].items())
        for scope, metrics in rows:
            for metric in COMMON_METRICS:
                item = metrics[metric]
                if scope == 'Overall · four suites equally weighted':
                    left = item['baseline']['mean_equal_suite']
                    right = item['anchor_kl']['mean_equal_suite']
                    delta = item['paired']['mean_difference_equal_suite']
                else:
                    left = item['baseline']['mean_case_value']
                    right = item['anchor_kl']['mean_case_value']
                    delta = item['paired']['mean_difference_anchor_minus_baseline']
                pair = item['paired']
                wlt = f"{pair['wins']}/{pair['losses']}/{pair['ties']}"
                lines.append(f'| {scope} | {metric} | {_fmt(left)} | {_fmt(right)} | {_fmt(delta)} | {wlt} |')
        lines.append('')
        objective = data['reconstructed_common_objective']
        lines += ['共同目标诊断（由记录中的 `first10motor_rmse² + rho × normalization_c × future_latent_nmse` 重建；'
                  'RMSE 已舍入，因此 MSE 部分有舍入误差）：',
                  f"- Baseline：{_fmt(objective['baseline']['mean_equal_suite'])}",
                  f"- Anchor-KL：{_fmt(objective['anchor_kl']['mean_equal_suite'])}",
                  'Baseline 的 `latent_anchor_kl` 与 `composite_objective` 标为 unavailable：缓存没有完整 future tensor，'
                  '本轮不做 baseline 重放或补充 forward。', '']
        diagnostics = data['candidate_only_diagnostics']
        lines += ['Candidate-only diagnostics（均值按相同 case/suite 层级汇总）：']
        for metric in CANDIDATE_DIAGNOSTICS:
            item = diagnostics[metric]
            value = item.get('mean_equal_suite') if item['available'] else None
            suffix = '' if item['available'] else f"；unavailable：{item['reason']}"
            lines.append(f'- `{metric}`：{_fmt(value)}{suffix}')
        lines.append('')
    lines += [summary['interpretation'], '']
    return '\n'.join(lines)


def write_outputs(results_root, summary):
    output = Path(results_root) / 'aggregate'
    _write_json(output / 'summary.json', summary)
    _write_json(output / 'comparison.json', comparison_view(summary))
    _write_text(Path(results_root) / 'ANALYSIS.zh.md', render_report(summary))


def validate_prepare_summary(protocol, prepare):
    if (prepare.get('complete') is not True or prepare.get('passed') is not True
            or prepare.get('protocol') != protocol['protocol']
            or prepare.get('heldout_used') is not False
            or prepare.get('calibration_case_ids') != protocol['calibration']['case_ids']
            or prepare.get('coefficient_frozen') is not True
            or not _finite(prepare.get('beta')) or prepare['beta'] <= 0
            or not _finite(prepare.get('temperature')) or prepare['temperature'] <= 0
            or not isinstance(prepare.get('stats'), dict)
            or prepare.get('normalization_c') != protocol['training']['normalization_c']):
        raise ValueError('Prepare summary must bind the protocol, use only the exact C8 calibration IDs, and freeze coefficients')


def run(protocol_path, results_root=None):
    from anchor_selfcheck import require_allocation

    require_allocation()
    protocol = validate_protocol(read_json(protocol_path))
    results_root = Path(results_root or (Path(protocol['remote_root']) / 'results'))
    prepare = read_json(results_root / 'prepare' / 'summary.json')
    validate_prepare_summary(protocol, prepare)
    candidate_summary = read_json(results_root / CANDIDATE_ARM / 'summary.json')
    if (candidate_summary.get('complete') is not True or candidate_summary.get('updates') != 8
            or candidate_summary.get('evaluation_queries') != 88
            or candidate_summary.get('new_teacher_queries') != 0
            or candidate_summary.get('baseline_new_model_queries') != 0):
        raise ValueError('Candidate summary must confirm complete update8 coverage and zero new baseline/teacher queries')
    if (candidate_summary.get('protocol', protocol['protocol']) != protocol['protocol']
            or candidate_summary.get('arm', CANDIDATE_ARM) != CANDIDATE_ARM):
        raise ValueError('Candidate summary is bound to a different protocol or arm')
    baseline, candidate = load_records(protocol, results_root)
    pbs_jobid = os.environ['PBS_JOBID']
    summary = summarize(protocol, baseline, candidate, prepare, candidate_summary, pbs_jobid)
    if (summary['baseline_records'] != 88 or summary['candidate_records'] != 88
            or summary['candidate_evaluation_queries'] != 88
            or summary['candidate_new_teacher_queries'] != 0):
        raise ValueError('Final aggregate does not cover the exact 88+88 query contract')
    write_outputs(results_root, summary)
    print('C8_LATENT_ANCHOR_AGGREGATE complete baseline_records=88 candidate_records=88', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    run(args.protocol, args.results)


if __name__ == '__main__':
    main()
