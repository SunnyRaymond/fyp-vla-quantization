"""Evaluate the frozen C80 joint checkpoint; all model work is PBS-only."""
import argparse
import json
import math
import os
import sys
from pathlib import Path
import sys


def query_pairs(protocol, split):
    spec = protocol['calibration' if split == 'calibration' else 'evaluation']
    seeds = [0] if split == 'calibration' else [0, 1]
    return [(case, seed) for case in spec['case_ids'] for seed in seeds]


def require_exact_coverage(records, protocol, split):
    expected = set(query_pairs(protocol, split))
    keys = [(record.get('case_id'), record.get('seed_index')) for record in records]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise RuntimeError('Exact C80 query coverage required for ' + split)


def _check_metrics(record, metric_names):
    for metric in metric_names:
        value = record.get(metric)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise RuntimeError('Invalid metric ' + metric)


def candidate_binding(protocol, row, split, seed, source_job_id):
    return {
        'protocol': protocol['protocol'], 'arm': protocol['arm'], 'split': split,
        'updates': protocol['joint_source']['updates'], 'case_id': row['case_id'],
        'suite': row['suite'], 'plan_row': row, 'seed_index': seed,
        'sampler_seed': row['sampler_seeds'][seed],
        'source_checkpoint': protocol['joint_source']['checkpoint'],
        'source_pbs_jobid': source_job_id,
    }


def validate_candidate_record(record, protocol, row, split, seed, source_job_id,
                              metric_names):
    expected = candidate_binding(protocol, row, split, seed, source_job_id)
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Candidate endpoint binding differs')
    if not isinstance(record.get('pbs_jobid'), str) or not record['pbs_jobid']:
        raise RuntimeError('Candidate endpoint is missing its evaluation PBS job ID')
    _check_metrics(record, metric_names)


def aggregate_endpoints(records, source_protocol, split):
    require_exact_coverage(records, source_protocol, split)
    import c80_reference as c80
    stats = c80.aggregate_records(records, source_protocol, split)
    _check_metrics(stats['overall'], c80.METRICS)
    for suite in c80.ref.SUITES:
        _check_metrics(stats['by_suite'][suite], c80.METRICS)
    return stats


def paired_case_counts(records_by_arm, candidate, control, metric):
    values = {}
    for arm, records in records_by_arm.items():
        per_case = {}
        for record in records:
            per_case.setdefault(record['case_id'], []).append(record[metric])
        values[arm] = {case: sum(items) / len(items) for case, items in per_case.items()}
    if set(values[candidate]) != set(values[control]):
        raise RuntimeError('Paired case cohorts differ')
    deltas = [values[candidate][case] - values[control][case]
              for case in sorted(values[candidate])]
    return {'cases': len(deltas), 'improved': sum(value < 0 for value in deltas),
            'worsened': sum(value > 0 for value in deltas),
            'tied': sum(value == 0 for value in deltas)}


def percent_changes(candidate, controls, metrics):
    result = {}
    for control, values in controls.items():
        result[control] = {}
        for metric in metrics:
            denominator = values[metric]
            result[control][metric] = (100 * (candidate[metric] / denominator - 1)
                                       if denominator else None)
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _source_contract(protocol, joint_contract):
    root = Path(protocol['remote_root'])
    source_c80 = protocol['source_c80_protocol']
    control_protocol = _read_json(Path(protocol['control_root']) / 'protocol.json')
    if control_protocol != source_c80:
        raise RuntimeError('Original C80 control protocol differs from the consumer snapshot')
    if (protocol.get('arm') != 'joint_c80'
            or protocol.get('arms') != ['baseline', 'clip_only', 'joint_c80']
            or protocol.get('joint_source', {}).get('updates') != 80
            or any(protocol.get(key) != source_c80.get(key)
                   for key in ('calibration', 'training', 'clipping', 'evaluation'))):
        raise RuntimeError('Consumer cohort, training, clipping, or evaluation contract differs')

    joint = protocol['joint_source']
    source_protocol = _read_json(Path(joint['root']) / 'protocol.json')
    evidence = _read_json(root / 'control' / 'source_evidence.json')
    snapshot = evidence.get('joint_source', {})
    joint_dir = joint['root'].rstrip('/') + '/results/training/joint/'
    if (source_protocol.get('protocol') != joint['protocol']
            or source_protocol.get('remote_root') != joint['root']
            or source_protocol.get('source_c80_protocol') != source_c80
            or any(source_protocol.get(key) != protocol.get(key)
                   for key in ('calibration', 'training', 'clipping'))
            or joint.get('checkpoint') != joint_dir + 'deployment.pt'
            or joint.get('receipt') != joint_dir + 'training_receipt.json'
            or joint.get('reference_action') != joint_dir + 'reference_action.json'
            or snapshot.get('protocol') != source_protocol):
        raise RuntimeError('Frozen joint source protocol snapshot differs')
    joint_contract.validate_protocol(source_protocol)

    receipt = _read_json(joint['receipt'])
    if (snapshot.get('receipt') != receipt
            or not snapshot.get('final_job_id')
            or receipt.get('pbs_jobid') != snapshot.get('final_job_id')):
        raise RuntimeError('Joint source receipt and controller job binding differ')
    joint_contract.validate_receipt(receipt, source_protocol, 'joint')
    return root, source_c80, source_protocol, receipt, snapshot


def _load_checkpoint(torch, protocol, source_protocol, receipt, joint_contract):
    joint = protocol['joint_source']
    deployment = torch.load(joint['checkpoint'], map_location='cpu', weights_only=True)
    joint_contract.validate_deployment(deployment, source_protocol, 'joint')
    masters, ratios = deployment.get('module_masters'), deployment.get('ratios')
    if (not isinstance(masters, dict) or len(masters) != 614
            or not isinstance(ratios, dict) or len(ratios) != 614
            or set(masters) != set(ratios)):
        raise RuntimeError('Joint deployment must contain all 614 masters and ratios')
    for value in masters.values():
        if (not isinstance(value, dict) or set(value) != {'books', 'log_scales'}
                or value['books'].dtype != torch.float32
                or value['log_scales'].dtype != torch.float32):
            raise RuntimeError('Joint deployment masters must be FP32 books and row gains')
    if any(value.dtype != torch.float32 for value in ratios.values()):
        raise RuntimeError('Joint deployment ratios must be FP32')
    if receipt.get('updates') != 80:
        raise RuntimeError('Joint source is not the final update80 checkpoint')
    return deployment


def _load_plan(protocol, source_protocol):
    plan = _read_json(protocol['calibration']['plan_file'])
    calibration = protocol['calibration']['case_ids']
    evaluation = protocol['evaluation']['case_ids']
    if (plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c80') != calibration):
        raise RuntimeError('Frozen C80 calibration plan differs')
    rows = {int(row['case_id']): row for row in plan['inputs']}
    required = set(calibration) | set(evaluation)
    if not required <= set(rows):
        raise RuntimeError('Frozen C80 plan is missing evaluation rows')
    if (calibration != source_protocol['calibration']['case_ids']
            or evaluation != source_protocol['evaluation']['case_ids']):
        raise RuntimeError('Consumer C80 cohorts differ from the source protocol')
    for case in required:
        row = rows[case]
        if (row.get('domain') != 'original' or len(row.get('sampler_seeds', [])) != 2
                or row.get('suite') not in ('libero_spatial', 'libero_object',
                                             'libero_goal', 'libero_10')):
            raise RuntimeError('C80 plan row domain, suite, or seeds differ')
    return plan, rows


def teacher_path(control_root, clipping_root, split, case, seed):
    directory = (Path(control_root) / 'results' / 'teacher' / 'calibration'
                 if split == 'calibration'
                 else Path(clipping_root) / 'results' / 'teacher')
    return directory / f'{case}_{seed}.pt'


def _load_teacher(torch, control_root, clipping_root, rows, split, case, seed):
    saved = torch.load(teacher_path(control_root, clipping_root, split, case, seed),
                       map_location='cpu', weights_only=True)
    row = rows[case]
    if (saved.get('plan_row') != row
            or saved.get('sampler_seed') != row['sampler_seeds'][seed]):
        raise RuntimeError('Shared teacher plan row or sampler seed differs')
    action, future = saved.get('action'), saved.get('future')
    if (tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
            or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())):
        raise RuntimeError('Shared teacher tensors are invalid')
    return saved


def run_evaluate(protocol, c80, ref, joint_contract):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from metrics_reference import metrics
    from clip_banks import ClipBanks

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    root, source_c80, source_protocol, receipt, snapshot = _source_contract(
        protocol, joint_contract)
    deployment = _load_checkpoint(torch, protocol, source_protocol, receipt, joint_contract)
    _, rows = _load_plan(protocol, source_c80)

    comparison = _read_json(Path(protocol['control_root']) / 'results' / 'comparison.json')
    if (comparison.get('protocol') != source_c80['protocol']
            or comparison.get('complete') is not True
            or comparison.get('evaluation_queries') != 320):
        raise RuntimeError('Complete original baseline/clip_only C80 controls are required')
    for split, directory, expected_ids, expected_seeds in (
            ('calibration', Path(protocol['control_root']) / 'results' / 'teacher' / 'calibration',
             protocol['calibration']['case_ids'], [0]),
            ('heldout', Path(protocol['clipping_root']) / 'results' / 'teacher',
             protocol['evaluation']['case_ids'], [0, 1])):
        teacher_receipt = _read_json(directory / 'receipt.json')
        if (teacher_receipt.get('complete') is not True or teacher_receipt.get('queries') != 80
                or teacher_receipt.get('case_ids') != expected_ids
                or teacher_receipt.get('seed_indices') != expected_seeds):
            raise RuntimeError('Shared ' + split + ' teacher coverage differs')

    data_root, source_rows, _, baseline, baseline_reference = ref.load_inputs(
        source_c80['source_clipping_protocol'], torch)
    if (baseline.get('arm') != 'rho1.5' or baseline.get('updates') != 8
            or len(baseline.get('module_masters', {})) != 614):
        raise RuntimeError('Original C8 rho1.5 baseline checkpoint differs')
    if any(source_rows.get(case) != rows[case] for case in set(protocol['calibration']['case_ids'])
           | set(protocol['evaluation']['case_ids'])):
        raise RuntimeError('C80 plan rows differ from the original baseline plan')

    joint_reference = _read_json(protocol['joint_source']['reference_action'])
    first = protocol['calibration']['case_ids'][0]
    first_row = rows[first]
    if (baseline_reference.get('case_id') != first
            or baseline_reference.get('seed_index') != 0
            or baseline_reference.get('sampler_seed') != first_row['sampler_seeds'][0]
            or baseline_reference.get('plan_row') != first_row):
        raise RuntimeError('Original baseline reference action is not bound to C80 case 2000 seed 0')
    if (joint_reference.get('protocol') != source_protocol['protocol']
            or joint_reference.get('arm') != 'joint' or joint_reference.get('case_id') != first
            or joint_reference.get('seed_index') != 0
            or joint_reference.get('sampler_seed') != first_row['sampler_seeds'][0]
            or joint_reference.get('plan_row') != first_row):
        raise RuntimeError('Joint reference action is not bound to C80 case 2000 seed 0')

    out = root / 'results' / 'evaluate'
    out.mkdir(parents=True, exist_ok=True)
    state = {
        'protocol': protocol['protocol'], 'unit': 'evaluate', 'pbs_jobid': os.environ['PBS_JOBID'],
        'complete': False, 'phase': 'loading', 'training_updates_this_task': 0,
        'new_teacher_queries': 0, 'new_evaluation_endpoint_records_this_job': 0,
        'reused_evaluation_endpoint_records_this_job': 0,
        'new_model_query_executions_this_job': 0, 'new_parity_queries_this_job': 0,
        'evaluation_queries': 160, 'all_parameters_frozen': True,
        'source_checkpoint': protocol['joint_source']['checkpoint'],
        'source_pbs_jobid': snapshot['final_job_id'],
    }

    def progress(**values):
        state.update(values)
        ref.save_json(out / 'progress.json', state)
        ref.save_json(out / ('executions_' + os.environ['PBS_JOBID'] + '.json'), state)
        print('JOINT_RMSE_PROGRESS ' + json.dumps(state, allow_nan=False), flush=True)

    deadline = ref.Deadline()
    runtime = None
    progress()
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        banks = ClipBanks(runtime, protocol['calibration']['bank_directory'],
                          baseline['module_masters'], 'clip_only', source_c80['clipping'])
        if (len(banks.modules) != 614
                or set(banks.modules) != set(deployment['module_masters'])):
            raise RuntimeError('Runtime target Linear coverage differs from the joint checkpoint')

        def freeze_all():
            for value in runtime.model.parameters():
                value.requires_grad_(False)
            for group in (banks.books, banks.logs, banks.ratios):
                for value in group.values():
                    value.requires_grad_(False)
            if (any(value.requires_grad for value in runtime.model.parameters())
                    or any(value.requires_grad for group in (banks.books, banks.logs, banks.ratios)
                           for value in group.values())):
                raise RuntimeError('Every model, codebook, row gain, and ratio must be frozen')

        data = {}

        def datum(case):
            if case not in data:
                observation, metadata = calibration_source.load_observation(np, data_root, rows[case])
                data[case] = runtime.datum(observation), metadata
            return data[case]

        def query(case, seed):
            deadline.check()
            value, metadata = datum(case)
            result = ref.query(runtime, sampler, torch, value, metadata['description'],
                               rows[case]['sampler_seeds'][seed], banks)
            state['new_model_query_executions_this_job'] += 1
            return result

        freeze_all()
        baseline_result = query(first, 0)
        state['new_parity_queries_this_job'] += 1
        baseline_max_abs = float((baseline_result[0].float()
                                  - torch.tensor(baseline_reference['action'], dtype=torch.float32)).abs().max())
        if baseline_max_abs != 0:
            raise RuntimeError('Original baseline case 2000 seed 0 parity is not exact: '
                               + str(baseline_max_abs))
        progress(phase='joint_restore', baseline_parity_max_abs=baseline_max_abs)

        banks.restore(deployment['module_masters'])
        banks.restore_clips(deployment['ratios'])
        freeze_all()
        if (set(banks.ratios) != set(deployment['ratios'])
                or any(value.requires_grad for group in (banks.books, banks.logs, banks.ratios)
                       for value in group.values())):
            raise RuntimeError('Restored joint deployment binding or frozen state differs')
        joint_first = query(first, 0)
        state['new_parity_queries_this_job'] += 1
        joint_max_abs = float((joint_first[0].float()
                               - torch.tensor(joint_reference['action'], dtype=torch.float32)).abs().max())
        if not math.isfinite(joint_max_abs) or joint_max_abs > 1e-6:
            raise RuntimeError('Restored joint case 2000 seed 0 reference parity exceeds 1e-6: '
                               + str(joint_max_abs))
        progress(phase='evaluating', baseline_parity_max_abs=baseline_max_abs,
                 joint_reference_max_abs=joint_max_abs)

        split_summaries = {}
        completed = 0
        for split in ('calibration', 'heldout'):
            records = []
            for case, seed in c80.queries(source_c80, split):
                row = rows[case]
                path = root / 'results' / split / protocol['arm'] / 'queries' / f'{case}_{seed}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_file():
                    record = _read_json(path)
                    validate_candidate_record(record, protocol, row, split, seed,
                                              snapshot['final_job_id'], c80.METRICS)
                    state['reused_evaluation_endpoint_records_this_job'] += 1
                else:
                    teacher = _load_teacher(torch, protocol['control_root'],
                                            protocol['clipping_root'], rows,
                                            split, case, seed)
                    if split == 'calibration' and case == first and seed == 0:
                        action, future, counts = joint_first
                    else:
                        action, future, counts = query(case, seed)
                    record = {
                        **candidate_binding(protocol, row, split, seed, snapshot['final_job_id']),
                        'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts,
                        **metrics(torch, latent_reference, action, future, teacher, protocol['training']),
                    }
                    validate_candidate_record(record, protocol, row, split, seed,
                                              snapshot['final_job_id'], c80.METRICS)
                    ref.save_json(path, record)
                    state['new_evaluation_endpoint_records_this_job'] += 1
                records.append(record)
                completed += 1
                progress(phase='evaluating', split=split, completed_evaluation_queries=completed,
                         last_case=case, last_seed=seed)

            stats = aggregate_endpoints(records, source_c80, split)
            summary = {
                'protocol': protocol['protocol'], 'source_protocol': source_c80['protocol'],
                'arm': protocol['arm'], 'split': split, 'updates': 80, 'complete': True,
                'source_checkpoint': protocol['joint_source']['checkpoint'],
                'source_pbs_jobid': snapshot['final_job_id'],
                'pbs_jobid': os.environ['PBS_JOBID'], 'all_parameters_frozen': True,
                'baseline_parity_max_abs': baseline_max_abs,
                'joint_reference_max_abs': joint_max_abs, **stats,
            }
            ref.save_json(root / 'results' / split / protocol['arm'] / 'summary.json', summary)
            split_summaries[split] = summary

        final = {
            **state, 'complete': True, 'phase': 'complete',
            'new_evaluation_endpoint_records': state['new_evaluation_endpoint_records_this_job'],
            'new_training_updates': 0, 'training_updates_this_task': 0,
            'new_teacher_queries': 0, 'evaluation_queries': 160,
            'baseline_parity_max_abs': baseline_max_abs,
            'joint_reference_max_abs': joint_max_abs,
            'split_summaries': split_summaries,
            'evidence_scope': 'fixed-input decoded-BF16 VQ + A4; no closed-loop or native-kernel claim',
        }
        ref.save_json(out / 'summary.json', final)
        progress(phase='complete', complete=True, completed_evaluation_queries=160)
        print('JOINT_RMSE_EVALUATION_COMPLETE', flush=True)
        return 0
    except TimeoutError as exc:
        progress(phase='checkpointed', complete=False, reason=str(exc),
                 completed_evaluation_queries=state.get('completed_evaluation_queries', 0))
        return 75
    except Exception as exc:
        progress(phase='failed', complete=False,
                 error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if runtime is not None and runtime.env is not None:
            try:
                runtime.env.close()
            except Exception as exc:
                print('JOINT_RMSE_CLEANUP_ERROR ' + type(exc).__name__ + ': ' + str(exc),
                      file=sys.stderr, flush=True)


def _validate_controls_and_candidate(protocol, c80, root, source_c80,
                                     snapshot, rows):
    controls_root = Path(protocol['control_root'])
    old = _read_json(controls_root / 'results' / 'comparison.json')
    if (old.get('complete') is not True or old.get('protocol') != source_c80['protocol']
            or old.get('evaluation_queries') != 320):
        raise RuntimeError('Original 320 baseline/clip_only control endpoints are incomplete')

    arms, records_by_split = {}, {}
    for split in ('calibration', 'heldout'):
        arms[split], records_by_split[split] = {}, {}
        for arm in ('baseline', 'clip_only', protocol['arm']):
            origin = root if arm == protocol['arm'] else controls_root
            records = []
            for case, seed in query_pairs(source_c80, split):
                record = _read_json(origin / 'results' / split / arm / 'queries' / f'{case}_{seed}.json')
                if arm == protocol['arm']:
                    validate_candidate_record(record, protocol, rows[case], split, seed,
                                              snapshot['final_job_id'], c80.METRICS)
                else:
                    c80.validate_binding(record, source_c80, rows[case], split, arm, seed)
                    _check_metrics(record, c80.METRICS)
                records.append(record)
            stats = aggregate_endpoints(records, source_c80, split)
            summary = _read_json(origin / 'results' / split / arm / 'summary.json')
            expected_protocol = protocol['protocol'] if arm == protocol['arm'] else source_c80['protocol']
            expected_updates = 80 if arm in ('clip_only', protocol['arm']) else 0
            if (summary.get('complete') is not True or summary.get('protocol') != expected_protocol
                    or summary.get('split') != split or summary.get('arm') != arm
                    or summary.get('updates') != expected_updates
                    or any(summary.get(key) != value for key, value in stats.items())):
                raise RuntimeError('Saved ' + split + '/' + arm + ' summary differs from its exact endpoints')
            if arm == protocol['arm'] and (
                    summary.get('source_checkpoint') != protocol['joint_source']['checkpoint']
                    or summary.get('source_pbs_jobid') != snapshot['final_job_id']
                    or summary.get('all_parameters_frozen') is not True):
                raise RuntimeError('Candidate aggregate summary source binding differs')
            arms[split][arm] = summary
            records_by_split[split][arm] = records
    return arms, records_by_split


def aggregate(protocol, c80, joint_contract, ref):
    root, source_c80, _, _, snapshot = _source_contract(protocol, joint_contract)
    _, rows = _load_plan(protocol, source_c80)
    evaluation = _read_json(root / 'results' / 'evaluate' / 'summary.json')
    if (evaluation.get('complete') is not True or evaluation.get('evaluation_queries') != 160
            or evaluation.get('training_updates_this_task') != 0
            or evaluation.get('new_teacher_queries') != 0
            or evaluation.get('all_parameters_frozen') is not True
            or evaluation.get('source_checkpoint') != protocol['joint_source']['checkpoint']
            or evaluation.get('source_pbs_jobid') != snapshot['final_job_id']
            or evaluation.get('baseline_parity_max_abs') != 0
            or isinstance(evaluation.get('joint_reference_max_abs'), bool)
            or not isinstance(evaluation.get('joint_reference_max_abs'), (int, float))
            or not math.isfinite(evaluation['joint_reference_max_abs'])
            or evaluation['joint_reference_max_abs'] > 1e-6):
        raise RuntimeError('Complete frozen 160-endpoint joint evaluation is required')
    arms, records_by_split = _validate_controls_and_candidate(
        protocol, c80, root, source_c80, snapshot, rows)

    changes, paired = {}, {}
    primary_metrics = ('first10motor_rmse', 'future_latent_nmse')
    for split in ('calibration', 'heldout'):
        candidate = arms[split][protocol['arm']]['overall']
        controls = {arm: arms[split][arm]['overall'] for arm in ('baseline', 'clip_only')}
        changes[split] = percent_changes(candidate, controls, c80.METRICS)
        paired[split] = {
            control: {metric: paired_case_counts(records_by_split[split], protocol['arm'],
                                                   control, metric)
                      for metric in primary_metrics}
            for control in ('baseline', 'clip_only')
        }

    result = {
        'protocol': protocol['protocol'], 'complete': True,
        'source_protocol': source_c80['protocol'],
        'pbs_jobid': os.environ['PBS_JOBID'], 'source_checkpoint': protocol['joint_source']['checkpoint'],
        'source_pbs_jobid': snapshot['final_job_id'], 'source_updates': 80,
        'new_training_updates': 0, 'training_updates_this_task': 0,
        'new_teacher_queries': 0, 'evaluation_queries': 480,
        'new_evaluation_endpoint_records': 160,
        'reused_control_endpoint_records': 320,
        'arms': arms, 'percent_change_vs_controls': changes,
        'paired_case_counts_after_seed_average': paired,
        'baseline_parity_max_abs': evaluation['baseline_parity_max_abs'],
        'joint_reference_max_abs': evaluation['joint_reference_max_abs'],
        'all_parameters_frozen': True,
        'training_updates_this_task_zero': True,
        'claim_scope': 'fixed-input decoded-BF16 VQ + A4; no closed-loop or native-kernel claim',
    }
    out = root / 'results'
    ref.save_json(out / 'comparison.json', result)

    metrics = tuple(c80.METRICS)
    lines = [
        '# C80 joint checkpoint RMSE evaluation', '',
        '直接评估已完成的 joint C80 update80 checkpoint，没有在本任务重训。',
        'Calibration 为原80 contexts × 1 seed；held-out 为原40 contexts × 2 seeds；复用原 teacher 与 baseline/clip_only 共320条端点。',
        '所有指标均为 fixed-input decoded-BF16 VQ + A4 证据，不代表闭环或 native kernel 性能。', '',
        '| Split / arm | ' + ' | '.join(metrics) + ' |',
        '|---|' + '|'.join('---:' for _ in metrics) + '|',
    ]
    for split in ('calibration', 'heldout'):
        for arm in ('baseline', 'clip_only', protocol['arm']):
            values = arms[split][arm]['overall']
            lines.append('| ' + split + ' / ' + arm + ' | '
                         + ' | '.join(f'{values[metric]:.9f}' for metric in metrics) + ' |')
    lines += ['', 'joint 相对对照变化，负值表示该指标数值降低：', '']
    for split in ('calibration', 'heldout'):
        for control in ('baseline', 'clip_only'):
            formatted = ', '.join(
                f'{metric} {changes[split][control][metric]:+.3f}%'
                if changes[split][control][metric] is not None
                else f'{metric} undefined (control mean is zero)'
                for metric in metrics)
            lines.append(f'- {split} vs {control}: {formatted}。')
    lines += ['', 'Held-out 按每个 case 先平均两个 seed 后计数（只对 First10 motor RMSE 与 future latent NMSE 计数）：', '']
    for control in ('baseline', 'clip_only'):
        for metric in primary_metrics:
            counts = paired['heldout'][control][metric]
            lines.append(f'- vs {control}, {metric}: {counts["improved"]} improved / '
                         f'{counts["worsened"]} worsened / {counts["tied"]} tied '
                         f'({counts["cases"]} cases)。')
    lines += [
        '', 'Late22、Full32 与 gripper 仅作诊断，不作为 gate。',
        f'本任务训练更新数为 0；读取的 source joint checkpoint 来自 {snapshot["final_job_id"]}。',
    ]
    (out / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('JOINT_RMSE_AGGREGATE_COMPLETE', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=('evaluate', 'aggregate'), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args(argv)

    import c80_reference as c80
    c80.guard()
    protocol = _read_json(args.protocol)
    import clipping_reference as ref
    import joint_contract
    if args.unit == 'aggregate':
        aggregate(protocol, c80, joint_contract, ref)
        return 0
    return run_evaluate(protocol, c80, ref, joint_contract)


if __name__ == '__main__':
    raise SystemExit(main())
