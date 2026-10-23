"""Aggregate the frozen fixed-input VQ plus activation KL experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

from relational_selfcheck import require_allocation


METRICS = (
    'action_kl', 'latent_kl', 'action_temporal_kl',
    'latent_spatiotemporal_kl', 'softbin_objective',
    'relational_objective', 'first10motor_rmse', 'future_latent_nmse',
)
NEW_ARMS = ('vq_kl', 'vq_kl_activation_kl')
OLD_ARMS = ('softbin', 'relational')
ARMS = NEW_ARMS + OLD_ARMS
FULL_ARM = 'vq_kl_activation_kl'
COMPARISONS = {
    'vq_kl_activation_kl_vs_relational': 'relational',
    'vq_kl_activation_kl_vs_softbin': 'softbin',
    'vq_kl_activation_kl_vs_vq_kl': 'vq_kl',
}
HISTORICAL_MSE_METRICS = (
    'action_kl', 'latent_kl', 'first10motor_rmse', 'future_latent_nmse',
)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_source_parity(protocol, source_protocol):
    require(protocol['protocol'] == 'fastwam-vq-relational-vq-activation-v1',
            'Unexpected frozen protocol identifier')
    require(tuple(protocol['arms']) == NEW_ARMS, 'New arm order or identity differs')
    vq = protocol['vq_training']
    require(vq['case_ids'] == [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070],
            'Frozen C8 update cases differ')
    require(vq['updates'] == 8 and vq['learning_rate'] == 0.0005
            and vq['gradient_norm_clip'] == 1.0
            and vq['optimizer_reset_after_update1'] is True,
            'Frozen C8 training settings differ')
    for section, keys in {
        'calibration': ('case_ids', 'seed_index', 'update_order', 'c8_reuse_ids'),
        'evaluation': ('case_ids', 'seed_indices', 'cases_per_suite', 'aggregation'),
        'training': ('rho', 'normalization_c'),
    }.items():
        for key in keys:
            require(protocol[section][key] == source_protocol[section][key],
                    f'Inherited {section}.{key} differs from relational source')
    for section in ('scaling', 'clipping', 'softbin', 'relational'):
        require(protocol[section] == source_protocol[section],
                f'Inherited {section} definition differs from relational source')
    require(len(protocol['calibration']['case_ids']) == 80
            and len(protocol['evaluation']['case_ids']) == 40
            and protocol['evaluation']['seed_indices'] == [0, 1],
            'Frozen C80/heldout cohort size differs')


def validate_history(receipt, phase, expected_order, skipped=False):
    order = receipt[f'{phase}_update_order']
    history = receipt[f'{phase}_history']
    require(order == expected_order, f'{phase} update order differs from frozen protocol')
    require(len(history) == len(expected_order), f'{phase} history length differs')
    for index, (row, case_id) in enumerate(zip(history, expected_order), start=1):
        require(row.get('update') == index and row.get('case_id') == case_id,
                f'{phase} history does not match its exact update order')

    gradient = receipt['gradient_coverage'][phase]
    frozen = receipt['frozen_checks'][phase]
    require(gradient.get('passed') is True and frozen.get('passed') is True,
            f'{phase} gradient coverage or frozen checks did not pass')
    if skipped:
        require(gradient.get('skipped') is True and frozen.get('skipped') is True,
                f'Skipped {phase} must be explicitly marked skipped')


def validate_new_summary(root, arm, summary, protocol):
    vq_count = protocol['vq_training']['updates']
    activation_count = 0 if arm == 'vq_kl' else protocol['training']['additional_updates']
    require(summary.get('complete') is True and summary.get('pbs_jobid'),
            f'{arm} training unit is incomplete or lacks PBS evidence')
    require(summary.get('vq_updates') == vq_count
            and summary.get('activation_updates') == activation_count
            and summary.get('updates') == (vq_count if activation_count == 0 else activation_count),
            f'{arm} update counts differ from protocol')
    require(summary.get('evaluation_queries') == 160
            and summary.get('new_teacher_queries') == 0,
            f'{arm} endpoint or teacher-query count differs')

    receipt = summary['training_receipt']
    require(receipt.get('complete') is True, f'{arm} training receipt is incomplete')
    require(receipt.get('vq_updates') == vq_count
            and receipt.get('activation_updates') == activation_count,
            f'{arm} receipt stage counts differ')
    validate_history(receipt, 'vq', protocol['vq_training']['case_ids'])
    activation_order = [] if activation_count == 0 else protocol['calibration']['update_order']
    validate_history(receipt, 'activation', activation_order, skipped=activation_count == 0)
    parity = receipt['parity']
    require(parity.get('passed') is True and parity.get('initial') is not None
            and parity.get('deployment') is not None,
            f'{arm} parity evidence is incomplete')
    return {
        'pbs_jobid': summary['pbs_jobid'], 'updates': summary['updates'],
        'vq_updates': vq_count, 'activation_updates': activation_count,
        'evaluation_queries': summary['evaluation_queries'],
        'new_teacher_queries': summary['new_teacher_queries'],
        'receipt_complete': receipt['complete'],
        'gradient_coverage_passed': True, 'frozen_checks_passed': True,
        'parity_passed': True,
    }


def expected_slots(protocol, split):
    if split == 'calibration':
        return [(case_id, protocol['calibration']['seed_index'])
                for case_id in protocol['calibration']['case_ids']]
    return [(case_id, seed) for case_id in protocol['evaluation']['case_ids']
            for seed in protocol['evaluation']['seed_indices']]


def load_records(root, arm, split, protocol, source_protocol_id, expected_updates):
    is_new = arm in NEW_ARMS
    data_root = root if is_new else Path(protocol['source_relational_root'])
    records = []
    for case_id, seed in expected_slots(protocol, split):
        path = data_root / 'results' / split / arm / 'queries' / f'{case_id}_{seed}.json'
        row = read_json(path)
        expected_protocol = protocol['protocol'] if is_new else source_protocol_id
        require(row.get('protocol') == expected_protocol and row.get('arm') == arm
                and row.get('split') == split and row.get('updates') == expected_updates,
                f'Endpoint identity differs in {path}')
        require(row.get('case_id') == case_id and row.get('seed_index') == seed
                and row.get('plan_row') is not None and row.get('sampler_seed') is not None,
                f'Endpoint is missing frozen pairing identity in {path}')
        if is_new:
            activation_updates = 0 if arm == 'vq_kl' else protocol['training']['additional_updates']
            require(row.get('vq_updates') == protocol['vq_training']['updates']
                    and row.get('activation_updates') == activation_updates,
                    f'Endpoint stage counts differ in {path}')
        for metric in METRICS:
            value = row.get(metric)
            require(isinstance(value, (int, float)) and math.isfinite(value),
                    f'Invalid {metric} in {path}')
        records.append(row)
    keys = [(row['case_id'], row['seed_index']) for row in records]
    require(len(records) == len(expected_slots(protocol, split)) and len(set(keys)) == len(keys),
            f'{arm}/{split} does not cover the exact frozen query cohort')
    return records


def summarize(records):
    by_case = {}
    for row in records:
        by_case.setdefault(row['case_id'], []).append(row)
    suites = {}
    for case_id, rows in by_case.items():
        suite = rows[0]['suite']
        require(all(row['suite'] == suite for row in rows), 'Suite differs between seeds')
        suites.setdefault(suite, []).append({
            metric: statistics.mean(row[metric] for row in rows) for metric in METRICS
        })
    require(len(suites) == 4, 'Exactly four suites are required')
    expected_cases_per_suite = 20 if len(by_case) == 80 else 10
    for suite, case_rows in suites.items():
        require(len(case_rows) == expected_cases_per_suite,
                f'{suite} has unexpected frozen case coverage')
    by_suite = {
        suite: {metric: statistics.mean(row[metric] for row in case_rows) for metric in METRICS}
        for suite, case_rows in sorted(suites.items())
    }
    return {
        'queries': len(records), 'cases': len(by_case), 'suites': len(suites),
        'overall': {metric: statistics.mean(values[metric] for values in by_suite.values())
                    for metric in METRICS},
        'by_suite': by_suite,
    }


def percent_change(candidate, baseline):
    return None if baseline == 0 else 100.0 * (candidate / baseline - 1.0)


def paired_comparison(records_by_arm, candidate_arm, baseline_arm, split, protocol):
    candidate = {(row['case_id'], row['seed_index']): row
                 for row in records_by_arm[split, candidate_arm]}
    baseline = {(row['case_id'], row['seed_index']): row
                for row in records_by_arm[split, baseline_arm]}
    case_rows = []
    for case_id in protocol['calibration']['case_ids'] if split == 'calibration' else protocol['evaluation']['case_ids']:
        seeds = [protocol['calibration']['seed_index']] if split == 'calibration' else protocol['evaluation']['seed_indices']
        c_rows, b_rows = [], []
        for seed in seeds:
            c_row, b_row = candidate[case_id, seed], baseline[case_id, seed]
            for key in ('plan_row', 'suite', 'sampler_seed'):
                require(c_row[key] == b_row[key],
                        f'Paired {key} differs for {case_id}/{seed}: {candidate_arm} vs {baseline_arm}')
            c_rows.append(c_row)
            b_rows.append(b_row)
        suite = c_rows[0]['suite']
        case_rows.append({
            'case_id': case_id, 'suite': suite,
            **{metric: statistics.mean(row[metric] for row in c_rows)
               - statistics.mean(row[metric] for row in b_rows) for metric in METRICS},
        })
    per_metric = {}
    for metric in METRICS:
        deltas = [row[metric] for row in case_rows]
        per_metric[metric] = {
            'mean_case_delta': statistics.mean(deltas),
            'median_case_delta': statistics.median(deltas),
            'improved_cases': sum(delta < 0 for delta in deltas),
            'worsened_cases': sum(delta > 0 for delta in deltas),
            'tied_cases': sum(delta == 0 for delta in deltas),
        }
    suite_rows = {}
    for suite in sorted({row['suite'] for row in case_rows}):
        rows = [row for row in case_rows if row['suite'] == suite]
        suite_rows[suite] = {
            'cases': len(rows),
            'metrics': {
                metric: {
                    'mean_case_delta': statistics.mean(row[metric] for row in rows),
                    'relative_change_pct': percent_change(
                        statistics.mean(records_by_arm[split, candidate_arm][i][metric]
                                        for i in range(len(records_by_arm[split, candidate_arm]))
                                        if records_by_arm[split, candidate_arm][i]['suite'] == suite),
                        statistics.mean(records_by_arm[split, baseline_arm][i][metric]
                                        for i in range(len(records_by_arm[split, baseline_arm]))
                                        if records_by_arm[split, baseline_arm][i]['suite'] == suite),
                    ),
                    'improved_cases': sum(row[metric] < 0 for row in rows),
                } for metric in METRICS
            },
        }
    return {'case_count': len(case_rows), 'cases': case_rows,
            'summary': per_metric, 'by_suite': suite_rows}


def load_historical_mse(path):
    if not path.exists():
        return {'available': False}
    summary = read_json(path)
    overall = summary.get('overall', {})
    require(summary.get('complete') is True, 'Historical MSE reference is incomplete')
    require(all(metric in overall for metric in HISTORICAL_MSE_METRICS),
            'Historical MSE reference lacks one of its original four metrics')
    return {
        'available': True, 'protocol': summary.get('protocol'),
        'pbs_jobid': summary.get('pbs_jobid'), 'updates': summary.get('updates'),
        'metrics': {metric: overall[metric] for metric in HISTORICAL_MSE_METRICS},
        'note': 'Existing MSE endpoint only; no relational or composite objective is reconstructed.',
    }


def load_historical_vq_mse_reference(protocol, summaries):
    baseline_root = Path(protocol['control_root'])
    protocol_path = baseline_root / 'protocol.json'
    baseline_protocol = read_json(protocol_path)
    for key in ('case_ids', 'seed_indices', 'aggregation'):
        require(baseline_protocol['evaluation'][key] == protocol['evaluation'][key],
                f'Historical VQ-MSE evaluation.{key} differs from the frozen cohort')
    require(baseline_protocol['calibration']['case_ids'] == protocol['calibration']['case_ids']
            and baseline_protocol['calibration']['seed_index'] == protocol['calibration']['seed_index'],
            'Historical VQ-MSE calibration cohort differs from the frozen cohort')

    parent = baseline_protocol['source_clipping_protocol']
    current_parent = protocol['source_clipping_protocol']
    require(parent['parent_updates'] == current_parent['parent_updates'] == 8
            and parent['parent_arm'] == current_parent['parent_arm'] == 'rho1.5'
            and parent['parent_root'] == current_parent['parent_root'],
            'Historical VQ-MSE parent is not the same-budget rho1.5 C8 source')

    reference, metadata = {}, {}
    for split, expected_cases in (('calibration', 80), ('heldout', 40)):
        path = baseline_root / 'results' / split / 'baseline' / 'summary.json'
        summary = read_json(path)
        require(summary.get('complete') is True
                and summary.get('protocol') == baseline_protocol['protocol']
                and summary.get('arm') == 'baseline'
                and summary.get('updates') == 0
                and summary.get('queries') == 80
                and summary.get('cases') == expected_cases,
                f'Historical VQ-MSE {split} summary is incomplete or mismatched')
        values = {}
        for metric in ('first10motor_rmse', 'future_latent_nmse'):
            value = summary['overall'].get(metric)
            require(isinstance(value, (int, float)) and math.isfinite(value),
                    f'Historical VQ-MSE {split} summary lacks finite {metric}')
            values[metric] = value
        reference[split] = values
        metadata[split] = {
            'source_path': str(path), 'complete': summary['complete'],
            'protocol': summary['protocol'], 'arm': summary['arm'],
            'updates': summary['updates'], 'pbs_jobid': summary.get('pbs_jobid'),
            'queries': summary['queries'], 'cases': summary['cases'],
        }

    require(metadata['calibration']['pbs_jobid'] == metadata['heldout']['pbs_jobid'],
            'Historical VQ-MSE split summaries come from different jobs')
    changes = {
        split: {
            metric: percent_change(summaries[split, 'vq_kl']['overall'][metric],
                                   reference[split][metric])
            for metric in ('first10motor_rmse', 'future_latent_nmse')
        }
        for split in ('calibration', 'heldout')
    }
    return {
        'available': True,
        'source_protocol_file': str(protocol_path),
        'source_root': str(baseline_root),
        'source_protocol': baseline_protocol['protocol'],
        'parent': {
            'root': parent['parent_root'], 'arm': parent['parent_arm'],
            'updates': parent['parent_updates'],
            'training_job': parent.get('parent_training_job'),
            'initialization': 'MSE-aware rho1.5 C8 VQ checkpoint; D=1/r=1; no additional activation training',
        },
        'summary_metadata': metadata,
        'metrics': reference,
        'vq_kl_relative_change_pct': changes,
        'source_endpoint_reused': True,
        'additional_forward_queries': 0,
        'additional_training_updates': 0,
        'heldout_selection': False,
        'note': 'Same-budget VQ8 upstream reference; summary updates=0 means no extra activation updates. No KL relation or composite metrics are available or reconstructed.',
    }


def metric_table(split, summaries, changes):
    columns = ['metric'] + list(ARMS) + list(COMPARISONS)
    lines = [f'## {"C80 calibration" if split == "calibration" else "Held-out"} values', '',
             '| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join(['---'] + ['---:'] * (len(columns) - 1)) + ' |']
    for metric in METRICS:
        values = [f'{summaries[split, arm]["overall"][metric]:.8g}' for arm in ARMS]
        pcts = [changes[split][name][metric] for name in COMPARISONS]
        values += ['n/a' if value is None else f'{value:+.2f}%' for value in pcts]
        lines.append('| ' + metric + ' | ' + ' | '.join(values) + ' |')
    return lines


def write_report(path, protocol, source_scale, source_prepare, summaries,
                 changes, paired, job_evidence, historical_mse, historical_vq_mse):
    alpha_a = source_scale['action_kl_coefficient']
    alpha_z = source_scale['latent_kl_coefficient']
    beta_a = source_prepare['coefficients']['action_beta']
    beta_z = source_prepare['coefficients']['latent_beta']
    rho_c = protocol['training']['rho'] * protocol['training']['normalization_c']
    report = [
        '# VQ relational KL + activation 固定输入结果', '',
        '本实验只测 fixed-input teacher/student 输出，不执行环境闭环。原始共享 teacher 全部复用；新增 teacher query 为 0。每个新 arm 在 C80 calibration 80 cases 和 held-out 40 cases × 2 seeds 上评估，旧 softbin/relational 端点直接复用。', '',
        '公平性约束：四组按同一 case、seed、plan_row 和 sampler_seed 配对；先在 case 内平均 seed，再对四个 suite 等权。held-out 的配对单位是 40 个 case，不把 80 条 query 当成独立样本。C8 初始化沿用原 bank masters，没有套用 MSE-aware masters。', '',
        '训练分两阶段：`vq_kl` 将 action+latent VQ 上游原 C8 的 8 updates 从 MSE 改为冻结 softbin 与 relational KL；`vq_kl_activation_kl` 在该 C8 KL 结果上再按当前 activation 代码训练 D/r 80 updates。目标沿用冻结定义：', '',
        '`L = αa·(action_kl + βa·action_temporal_kl) + (ρc)·αz·(latent_kl + βz·latent_spatiotemporal_kl)`', '',
        f'冻结值来自 source_relational/source_dual：αa={alpha_a:.10g}, αz={alpha_z:.10g}, βa={beta_a:.10g}, βz={beta_z:.10g}, ρc={rho_c:.10g}。温度、softbin 定义和 scaling/clipping 沿用 source_relational；没有新 dev，也没有 held-out 调参。', '',
        '下表的百分比均为同一 metric 下 `vq_kl_activation_kl` 相对相应对照组的变化（负值表示下降）。`softbin_objective` 与 `relational_objective` 分别在同名 metric 内跨 arm 比较；不把两种不同 objective 的原始值互相比较。', '',
    ]
    report += metric_table('calibration', summaries, changes)
    report += ['']
    report += metric_table('heldout', summaries, changes)
    report += ['', '## Held-out 按 case 配对', '',
               '| Comparison | Metric | Improved / 40 | Worsened / 40 | Tied / 40 | Median case delta |',
               '|---|---|---:|---:|---:|---:|']
    for name in COMPARISONS:
        for metric in METRICS:
            item = paired[name]['heldout']['summary'][metric]
            report.append(f"| {name} | {metric} | {item['improved_cases']}/40 | {item['worsened_cases']}/40 | {item['tied_cases']}/40 | {item['median_case_delta']:+.6g} |")
    report += ['', '## Held-out suite 异质性', '',
               '以下为主要比较 `vq_kl_activation_kl` vs `relational` 的 suite 内相对变化；四个 suite 分开呈现，不推断总体显著性。', '',
               '| Metric | ' + ' | '.join(paired['vq_kl_activation_kl_vs_relational']['heldout']['by_suite']) + ' |',
               '|---|' + '---:|' * len(paired['vq_kl_activation_kl_vs_relational']['heldout']['by_suite'])]
    suite_data = paired['vq_kl_activation_kl_vs_relational']['heldout']['by_suite']
    for metric in METRICS:
        cells = []
        for suite in suite_data:
            item = suite_data[suite]['metrics'][metric]
            pct = item['relative_change_pct']
            cells.append('n/a' if pct is None else f'{pct:+.2f}% ({item["improved_cases"]}/{suite_data[suite]["cases"]})')
        report.append('| ' + metric + ' | ' + ' | '.join(cells) + ' |')
    report += ['', '## 训练与复用证据', '',
               '| Group | PBS job | VQ updates | Activation updates | Endpoint queries | New teacher queries | Receipt / checks |',
               '|---|---|---:|---:|---:|---:|---|']
    for arm in ARMS:
        ev = job_evidence[arm]
        report.append(f"| {arm} | {ev['pbs_jobid']} | {ev['vq_updates']} | {ev['activation_updates']} | {ev['evaluation_queries']} | {ev['new_teacher_queries']} | {ev['evidence']} |")
    report += ['', '两条新 arm 合计新增 320 个 student endpoint query；旧 softbin/relational 的 320 个 endpoint query 复用。四组共享既有 teacher 输入。每个新 arm 只有一条训练轨迹，因此 case 计数用于描述配对一致性，不代表独立训练重复或一般化显著性。', '',
               '## 历史 MSE activation final80 参照', '']
    if historical_mse['available']:
        report += [f"历史 MSE 端点 `{historical_mse.get('pbs_jobid')}`（{historical_mse.get('updates')} updates）只报告它原有的四项：", '',
                   '| Metric | Value |', '|---|---:|']
        report += [f'| {metric} | {historical_mse["metrics"][metric]:.8g} |'
                   for metric in HISTORICAL_MSE_METRICS]
        report += ['', '该参照没有 relational 或 composite 输出，本报告不补造这些指标。', '']
    else:
        report += ['指定 source 中没有可读的历史 MSE summary；不构造替代值。', '']
    report += ['## 同预算 VQ8 上游对照', '',
               '以下复用原 MSE-aware rho1.5 C8 VQ final8、D=1/r=1 的固定输入 summary。summary 中 `updates=0` 表示没有额外 activation 更新；其 parent 是 rho1.5 C8 VQ checkpoint（8 updates）。因此这是上游 VQ 阶段自身的同预算参照，和上面的 MSE activation final80 参照分开。这里只读既有 summary JSON，没有新增 forward、训练或 held-out 选参。', '',
               f"source root：`{historical_vq_mse['source_root']}`；protocol：`{historical_vq_mse['source_protocol']}`；PBS job：`{historical_vq_mse['summary_metadata']['heldout']['pbs_jobid']}`。", '',
               '| Split | Metric | MSE-aware VQ8 baseline | VQ-KL final8 | VQ-KL vs baseline |',
               '|---|---|---:|---:|---:|']
    for split in ('calibration', 'heldout'):
        for metric in ('first10motor_rmse', 'future_latent_nmse'):
            baseline_value = historical_vq_mse['metrics'][split][metric]
            candidate_value = summaries[split, 'vq_kl']['overall'][metric]
            delta = historical_vq_mse['vq_kl_relative_change_pct'][split][metric]
            report.append(f'| {split} | {metric} | {baseline_value:.8g} | {candidate_value:.8g} | {delta:+.2f}% |')
    report += ['', '该历史 baseline 只有这两项诊断值；没有 KL relation 或 composite 指标，本报告不补造，也不据此声称这些目标的优劣。该对照仅说明固定输入下 VQ 上游阶段的变化，不代表 activation final80 的比较。', '']
    report += ['范围限制：单次训练轨迹、冻结 held-out cohort 的 fixed-input 证据；不报告 statistical significance、未见任务泛化或 task success。', '',
               '方法流程参考：本报告沿用 `experimental-design` workflow 组织冻结 cohort、配对单位与证据边界。参见 Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents* (2026), [arXiv](https://doi.org/10.48550/arXiv.2609.00065)。本实验 loss 与系数来自冻结 source protocol/prepare artifacts。', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(report), encoding='utf-8')


def run(args):
    protocol_path = Path(args.protocol)
    protocol = read_json(protocol_path)
    root = Path(protocol['remote_root'])
    source_root = Path(protocol['source_relational_root'])
    source_protocol = read_json(source_root / 'protocol.json')
    require_source_parity(protocol, source_protocol)
    source_prepare = read_json(source_root / 'results/prepare/summary.json')
    require(source_prepare.get('complete') is True and source_prepare.get('passed') is True
            and source_prepare.get('heldout_used') is not True,
            'Frozen teacher-only KL definition validation is missing')
    source_scale = read_json(Path(protocol['source_dual_root'])
                             / 'results/prepare/loss_scaling.json')

    job_evidence = {}
    for arm in NEW_ARMS:
        summary = read_json(root / 'results' / arm / 'summary.json')
        job_evidence[arm] = validate_new_summary(root, arm, summary, protocol)
    for arm in OLD_ARMS:
        summary = read_json(source_root / 'results' / arm / 'summary.json')
        require(summary.get('complete') is True and summary.get('updates') == 80
                and summary.get('evaluation_queries') == 160
                and summary.get('new_teacher_queries', 0) == 0,
                f'Reused {arm} training endpoint is incomplete')
        job_evidence[arm] = {
            'pbs_jobid': summary.get('pbs_jobid', 'source summary'),
            'updates': 80, 'vq_updates': 0, 'activation_updates': 80,
            'evaluation_queries': 160,
            'new_teacher_queries': summary.get('new_teacher_queries', 0),
            'receipt_complete': True, 'gradient_coverage_passed': 'source receipt',
            'frozen_checks_passed': 'source receipt', 'parity_passed': 'source receipt',
        }
        job_evidence[arm]['evidence'] = 'reused complete source endpoint'
    for arm in NEW_ARMS:
        job_evidence[arm]['evidence'] = 'complete; gradient/frozen/parity checks passed'

    source_protocol_id = source_protocol['protocol']
    records_by_arm = {}
    summaries = {}
    for split in ('calibration', 'heldout'):
        for arm in ARMS:
            updates = (8 if arm == 'vq_kl' else 80)
            rows = load_records(root, arm, split, protocol, source_protocol_id, updates)
            records_by_arm[split, arm] = rows
            summaries[split, arm] = summarize(rows)
        reference = {(row['case_id'], row['seed_index']): row
                     for row in records_by_arm[split, 'relational']}
        for arm in ('vq_kl', 'vq_kl_activation_kl', 'softbin'):
            for row in records_by_arm[split, arm]:
                key = (row['case_id'], row['seed_index'])
                base = reference[key]
                for field in ('plan_row', 'suite', 'sampler_seed'):
                    require(row[field] == base[field],
                            f'{field} differs between {arm} and relational for {key}')

    changes, pair_details = {}, {}
    for split in ('calibration', 'heldout'):
        changes[split] = {}
        for name, baseline_arm in COMPARISONS.items():
            changes[split][name] = {
                metric: percent_change(summaries[split, FULL_ARM]['overall'][metric],
                                       summaries[split, baseline_arm]['overall'][metric])
                for metric in METRICS
            }
            pair_details.setdefault(name, {})[split] = paired_comparison(
                records_by_arm, FULL_ARM, baseline_arm, split, protocol)

    historical_mse = load_historical_mse(
        Path(protocol['source_dual_root']) / 'results/heldout/mse_nmse/summary.json')
    historical_vq_mse = load_historical_vq_mse_reference(protocol, summaries)
    frozen = {
        'alpha_action': source_scale['action_kl_coefficient'],
        'alpha_latent': source_scale['latent_kl_coefficient'],
        'beta_action': source_prepare['coefficients']['action_beta'],
        'beta_latent': source_prepare['coefficients']['latent_beta'],
        'rho_times_c': protocol['training']['rho'] * protocol['training']['normalization_c'],
        'relational_config': source_prepare['relational_config'],
        'coefficients': source_prepare['coefficients'],
    }
    all_summaries = {
        split: {arm: summaries[split, arm] for arm in ARMS}
        for split in ('calibration', 'heldout')
    }
    comparison = {
        'protocol': protocol['protocol'], 'complete': True,
        'pbs_jobid': os.environ['PBS_JOBID'],
        'arms': list(ARMS), 'metrics': list(METRICS), 'splits': all_summaries,
        'percent_change_full_vs_comparator': changes,
        'frozen_objective': frozen,
        'evidence': {
            'new_endpoint_queries': 320, 'reused_source_endpoint_queries': 320,
            'new_teacher_queries': 0, 'new_dev_cases': 0,
            'case_seed_pairing_checked': True, 'heldout_selection': False,
            'closed_loop_tested': False,
        },
        'training_units': job_evidence,
        'historical_mse_reference': historical_mse,
        'historical_vq_mse_reference': historical_vq_mse,
    }
    primary = 'vq_kl_activation_kl_vs_relational'
    decision = {
        'protocol': protocol['protocol'], 'complete': True,
        'heldout_used_for_training_or_selection': False,
        'source_objective_and_pairing_checks_passed': True,
        'primary_comparison': primary,
        'heldout_relative_change_pct': changes['heldout'][primary],
        'heldout_case_counts': pair_details[primary]['heldout']['summary'],
        'one_training_trajectory_per_arm': True,
        'closed_loop_tested': False, 'automatic_next_training': False,
        'scope': 'paired fixed-input evidence on one frozen held-out cohort; no significance, unseen-task generalization, or task-success claim',
    }
    save_json(root / 'results/comparison.json', comparison)
    save_json(root / 'results/paired_details.json', {
        'protocol': protocol['protocol'], 'unit_of_pairing': 'case after averaging seeds',
        'heldout_independent_case_count': 40, 'comparisons': pair_details,
    })
    save_json(root / 'results/decision.json', decision)
    write_report(root / 'results/ANALYSIS.zh.md', protocol, source_scale,
                 source_prepare, summaries, changes, pair_details,
                 job_evidence, historical_mse, historical_vq_mse)
    save_json(root / 'results/aggregate/summary.json', {
        'protocol': protocol['protocol'], 'complete': True,
        'pbs_jobid': os.environ['PBS_JOBID'],
        'calibration_records': 320, 'heldout_records': 320,
        'records_per_arm_per_split': 80, 'new_endpoint_records': 320,
        'reused_source_endpoint_records': 320, 'new_teacher_queries': 0,
        'training_units': job_evidence, 'decision': decision,
    })
    print('VQ_RELATIONAL_ACTIVATION_AGGREGATE_COMPLETE', flush=True)
    return comparison


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
