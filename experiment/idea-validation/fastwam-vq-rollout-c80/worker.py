"""Train and evaluate the frozen rollout C80 joint arm inside its PBS allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import common


UPDATES = 80
TARGET_LINEARS = 614
C80_CONSUMER_PROTOCOL = 'fastwam-vq-c80-joint-rmse-v1'
METRICS = ('first10motor_mse', 'first10motor_rmse', 'full32motor_rmse',
           'late22motor_rmse', 'first10gripper_rmse', 'future_latent_nmse', 'mixed_loss')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def _metric_values(record, names=METRICS):
    for name in names:
        value = record.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise RuntimeError('Invalid metric ' + name)


def query_pairs(protocol):
    return [(case, seed) for case in protocol['evaluation']['case_ids']
            for seed in protocol['evaluation']['seed_indices']]


def require_exact_coverage(records, protocol):
    expected = set(query_pairs(protocol))
    keys = [(r.get('case_id'), r.get('seed_index')) for r in records]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise RuntimeError('Exactly 40 held-out cases x 2 seeds are required')


def candidate_binding(protocol, row, seed, checkpoint, source_job_id):
    return {
        'protocol': protocol['protocol'], 'arm': protocol['arm'], 'split': 'heldout',
        'updates': UPDATES, 'case_id': row['case_id'], 'suite': row['suite'],
        'plan_row': row, 'seed_index': seed, 'sampler_seed': row['sampler_seeds'][seed],
        'source_checkpoint': str(checkpoint), 'source_jobid': source_job_id,
    }


def validate_candidate_record(record, protocol, row, seed, checkpoint, source_job_id):
    expected = candidate_binding(protocol, row, seed, checkpoint, source_job_id)
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Held-out record plan/seed/checkpoint binding differs')
    if not isinstance(record.get('pbs_jobid'), str) or not record['pbs_jobid']:
        raise RuntimeError('Held-out record is missing its PBS job ID')
    _metric_values(record)


def validate_plan_document(plan, expected):
    inputs = plan.get('inputs')
    if plan.get('status') != 'complete' or not isinstance(inputs, list) or len(inputs) != 120:
        raise RuntimeError('Completed 120-row rollout plan is required')
    actual = {int(row['case_id']): row for row in inputs}
    if (len(actual) != 120 or list(actual) != list(expected) or actual != expected
            or plan.get('calibration_cohorts', {}).get('c80') != list(range(2000, 2080))):
        raise RuntimeError('Rollout plan rows differ from common.expected_rows')


def validate_prepare_summary(summary, protocol):
    parity = summary.get('source_parity_max_abs')
    if (summary.get('protocol') != protocol['protocol'] or summary.get('complete') is not True
            or summary.get('contexts') != 80 or summary.get('visited_contexts') != 40
            or summary.get('anchor_contexts') != 40 or summary.get('new_teacher_queries') != 40
            or summary.get('teacher_queries') != 80 or summary.get('collector_queries') != 40
            or summary.get('teacher_seed_indices') != [0]
            or summary.get('case_ids') != list(range(2000, 2080))
            or summary.get('render_parity_passed') is not True
            or isinstance(parity, bool) or not isinstance(parity, (int, float))
            or not math.isfinite(parity) or parity > 1e-6
            or not isinstance(summary.get('pbs_jobid'), str) or not summary['pbs_jobid']):
        raise RuntimeError('Prepare summary does not match the frozen 80/40/40 teacher/render contract')


def validate_teacher(torch, path, row, seed):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('plan_row') != row or saved.get('sampler_seed') != row['sampler_seeds'][seed]:
        raise RuntimeError('Teacher plan row or sampler seed binding differs: ' + str(path))
    action, future = saved.get('action'), saved.get('future')
    if (tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
            or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())):
        raise RuntimeError('Invalid teacher tensor: ' + str(path))
    return saved


def _save_progress(root, artifact, state):
    state['jobid'] = state['pbs_jobid'] = os.environ.get('PBS_JOBID', 'missing')
    common.save(root / 'results/train_eval/progress.json', state)
    common.save(root / ('results/train_eval/executions_' + state['jobid'] + '.json'), state)
    if artifact:
        common.save(artifact / 'train_eval_summary.json', state)
    print('ROLLOUT_C80_TRAIN_EVAL_PROGRESS ' + json.dumps(state, allow_nan=False), flush=True)


def _plan_rows(p):
    expected = common.expected_rows(p)
    plan = common.read(Path(p['remote_root']) / 'data/plan.json')
    validate_plan_document(plan, expected)
    for case in p['calibration']['case_ids']:
        if case not in expected or len(expected[case].get('sampler_seeds', [])) != 2:
            raise RuntimeError('Training plan is missing a paired-seed calibration row')
    return expected


def _validate_teacher_inputs(p, rows):
    root = Path(p['remote_root'])
    prepare = common.read(root / 'results/prepare/summary.json')
    receipt = common.read(root / 'results/teacher/receipt.json')
    validate_prepare_summary(prepare, p)
    validate_prepare_summary(receipt, p)
    if receipt['pbs_jobid'] != prepare['pbs_jobid']:
        raise RuntimeError('Prepare and teacher receipts come from different PBS jobs')
    import torch

    teachers = {}
    for case in p['calibration']['case_ids']:
        row = rows[case]
        path = common.teacher_path(p, row, 0)
        teachers[case] = validate_teacher(torch, path, row, 0)
    return teachers


def _validate_backward(p, train_banks):
    receipt = common.read(Path(p['remote_root']) / 'results/backward_check.json')
    if (getattr(train_banks, 'BACKWARD_IMPLEMENTATION', None) != 'bounded_vector_chunks_v2'
            or getattr(train_banks, 'BACKWARD_MAX_VECTORS', None) != 1048576
            or receipt.get('passed') is not True
            or receipt.get('implementation') != 'bounded_vector_chunks_v2'
            or receipt.get('max_vectors_per_chunk') != 1048576):
        raise RuntimeError('PBS bounded_vector_chunks_v2 backward receipt is missing or differs')
    return receipt


def _resume_source(root, job_id, checkpoint_path):
    if not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', job_id or ''):
        raise RuntimeError('Invalid --resume-from PBS job ID')
    artifact = root / 'artifacts' / job_id
    state = common.read(artifact / 'train_eval_summary.json')
    if ((artifact / 'exit_code.txt').read_text().strip() != '75'
            or not (artifact / 'TRAINING_CHECKPOINTED').is_file()
            or (artifact / 'TRAINING_COMPLETE').exists()
            or state.get('phase') != 'checkpointed' or state.get('complete') is not False
            or state.get('jobid') != job_id or state.get('checkpoint') != str(checkpoint_path)):
        raise RuntimeError('Resume source is not a verified terminal checkpointed PBS attempt')
    return state


def _freeze_all(runtime, banks):
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    for group in (banks.books, banks.logs, banks.ratios):
        for parameter in group.values():
            parameter.requires_grad_(False)
    if (any(p.requires_grad for p in runtime.model.parameters())
            or any(p.requires_grad for group in (banks.books, banks.logs, banks.ratios)
                   for p in group.values())):
        raise RuntimeError('All model and bank parameters must be frozen for endpoint evaluation')


def aggregate_records(records, c80, source_protocol):
    require_exact_coverage(records, source_protocol)
    stats = c80.aggregate_records(records, source_protocol, 'heldout')
    _metric_values(stats['overall'])
    for suite in SUITES:
        _metric_values(stats['by_suite'][suite])
    return stats


def _paired_counts(records_by_arm, candidate, control):
    output = {}
    for metric in METRICS:
        means = {}
        for arm in (candidate, control):
            by_case = {}
            for row in records_by_arm[arm]:
                by_case.setdefault(row['case_id'], []).append(row[metric])
            means[arm] = {case: sum(values) / len(values) for case, values in by_case.items()}
        if set(means[candidate]) != set(means[control]) or len(means[candidate]) != 40:
            raise RuntimeError('Paired held-out cohorts differ')
        delta = [means[candidate][case] - means[control][case]
                 for case in sorted(means[candidate])]
        output[metric] = {'cases': 40, 'improved': sum(x < 0 for x in delta),
                          'worsened': sum(x > 0 for x in delta),
                          'tied': sum(x == 0 for x in delta)}
    return output


def _control_root(protocol, arm):
    if arm == 'joint_c80':
        return Path(protocol['reference_root'])
    if arm in ('clip_only', 'baseline'):
        return Path(protocol['source_c80_protocol']['remote_root'])
    raise RuntimeError('Unknown C80 control arm: ' + str(arm))


def _control_records(control_root, comparison, arm, rows, source_protocol, c80):
    summary = comparison['arms']['heldout'].get(arm)
    if not isinstance(summary, dict) or summary.get('queries') != 80:
        raise RuntimeError('Original comparison lacks exact 80-query ' + arm + ' arm')
    records = []
    for case, seed in c80.queries(source_protocol, 'heldout'):
        path = control_root / 'results/heldout' / arm / 'queries' / f'{case}_{seed}.json'
        record = common.read(path)
        row = rows[case]
        if (record.get('case_id') != case or record.get('seed_index') != seed
                or record.get('suite') != row['suite'] or record.get('plan_row') != row
                or record.get('sampler_seed') != row['sampler_seeds'][seed]):
            raise RuntimeError('Original control endpoint is not paired to the rollout plan')
        _metric_values(record)
        records.append(record)
    return records


def _validate_control_summary(expected, observed, arm, source_protocol):
    expected_protocol = (C80_CONSUMER_PROTOCOL if arm == 'joint_c80'
                         else source_protocol['protocol'])
    if (expected.get('protocol') != expected_protocol or expected.get('complete') is not True
            or expected.get('arm') != arm or expected.get('split') != 'heldout'
            or expected.get('queries') != 80 or expected.get('cases') != 40):
        raise RuntimeError('Original comparison summary differs for ' + arm)
    _metric_values(expected.get('overall', {}))
    for metric in METRICS:
        if not math.isclose(expected['overall'][metric], observed['overall'][metric],
                            rel_tol=1e-7, abs_tol=1e-9):
            raise RuntimeError('Original comparison statistics differ for ' + arm)
    for suite in SUITES:
        _metric_values(expected.get('by_suite', {}).get(suite, {}))
        for metric in METRICS:
            if not math.isclose(expected['by_suite'][suite][metric],
                                observed['by_suite'][suite][metric],
                                rel_tol=1e-7, abs_tol=1e-9):
                raise RuntimeError('Original comparison suite statistics differ for ' + arm)


def _write_comparison(p, c80, rows, candidate):
    ref_root = Path(p['reference_root'])
    source = common.read(ref_root / 'results/comparison.json')
    if (source.get('complete') is not True
            or source.get('protocol') != C80_CONSUMER_PROTOCOL):
        raise RuntimeError('Original C80 comparison is missing or belongs to another protocol')
    records = {'rollout_joint': candidate}
    for arm in ('joint_c80', 'clip_only', 'baseline'):
        records[arm] = _control_records(_control_root(p, arm), source, arm, rows,
                                        p['source_c80_protocol'], c80)
    summaries = {arm: aggregate_records(values, c80, p['source_c80_protocol'])
                 for arm, values in records.items()}
    for arm in ('joint_c80', 'clip_only', 'baseline'):
        _validate_control_summary(source['arms']['heldout'][arm], summaries[arm], arm,
                                  p['source_c80_protocol'])
    controls = {}
    changes, paired = {}, {}
    for arm in ('joint_c80', 'clip_only', 'baseline'):
        controls[arm] = {}
        changes[arm] = {}
        for metric in METRICS:
            value = summaries[arm]['overall'][metric]
            controls[arm][metric] = value
            changes[arm][metric] = (100 * (summaries['rollout_joint']['overall'][metric] / value - 1)
                                    if value else None)
        paired[arm] = _paired_counts(records, 'rollout_joint', arm)
    result = {
        'protocol': p['protocol'], 'complete': True, 'source_comparison': str(ref_root / 'results/comparison.json'),
        'pbs_jobid': os.environ['PBS_JOBID'], 'training_updates': 80,
        'evaluation_queries': 80, 'new_teacher_queries': 0, 'all_parameters_frozen': True,
        'arms': {'heldout': summaries}, 'percent_change_vs_controls': {'heldout': changes},
        'paired_case_counts_after_seed_average': {'heldout': paired},
        'claim_scope': 'fixed-input decoded-BF16 VQ + A4; no closed-loop or native-kernel claim',
    }
    common.save(Path(p['remote_root']) / 'results/comparison.json', result)
    lines = [
        '# Rollout C80 joint held-out comparison', '',
        'rollout_joint 使用 anchor 原 teacher 与 visited-state teacher 训练80次；评估仅为原40 held-out contexts × 2 seeds。',
        '所有指标为 fixed-input decoded-BF16 VQ + A4；不表示 closed-loop 或 native-kernel 性能。', '',
        '| Arm | ' + ' | '.join(METRICS) + ' |', '|---|' + '|'.join('---:' for _ in METRICS) + '|',
    ]
    for arm in ('baseline', 'clip_only', 'joint_c80', 'rollout_joint'):
        values = summaries[arm]['overall']
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{values[m]:.9f}' for m in METRICS) + ' |')
    lines += ['', 'rollout_joint 相对原 C80 arms 的百分比变化（负数表示误差降低）：', '']
    for arm in ('joint_c80', 'clip_only', 'baseline'):
        lines.append('- ' + arm + ': ' + ', '.join(
            f'{metric} {changes[arm][metric]:+.3f}%' if changes[arm][metric] is not None
            else f'{metric} undefined (control mean is zero)' for metric in METRICS) + '。')
    lines += ['', '逐 case 先平均两个 seed 后的配对计数：', '']
    for arm in ('joint_c80', 'clip_only', 'baseline'):
        for metric in ('first10motor_rmse', 'future_latent_nmse'):
            count = paired[arm][metric]
            lines.append(f'- vs {arm}, {metric}: {count["improved"]} improved / '
                         f'{count["worsened"]} worsened / {count["tied"]} tied (40 cases)。')
    (Path(p['remote_root']) / 'results/RESULTS.zh.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    return result


def run_train_eval(p, resume_from=None):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import clipping_reference as ref
    import c80_reference as c80
    import latent_reference
    import train_banks
    from clip_banks import ClipBanks
    from metrics_reference import metrics
    from checkpointing import (atomic_torch_save, capture_rng, optimizer_to_device,
                               restore_rng, to_cpu)

    root = Path(p['remote_root'])
    artifact = Path(os.environ['ARTIFACTS'])
    out = root / 'results/training/rollout_joint'
    checkpoint_path = out / 'checkpoint.pt'
    out.mkdir(parents=True, exist_ok=True)
    state = {
        'protocol': p['protocol'], 'arm': p['arm'], 'unit': 'train_eval',
        'phase': 'validating_sources', 'status': 'running', 'complete': False,
        'updates': 0, 'updates_this_job': 0, 'trainingupdates80': 80,
        'evalqueries80': 80, 'allfrozen': False, 'new_teacher_queries': 0,
        'history': [], 'update_order': p['calibration']['update_order'],
        'checkpoint': str(checkpoint_path), 'resume_from': resume_from, 'resume_lineage': [],
    }
    job_start_updates = 0
    _save_progress(root, artifact, state)

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    _validate_backward(p, train_banks)
    source_protocol = p['source_c80_protocol']
    if common.read(Path(source_protocol['remote_root']) / 'protocol.json') != source_protocol:
        raise RuntimeError('Embedded source C80 protocol differs from its frozen file')
    source_clip = source_protocol['source_clipping_protocol']
    if common.read(Path(source_protocol['clipping_root']) / 'protocol.json') != source_clip:
        raise RuntimeError('Embedded source clipping protocol differs from its frozen file')
    rows = _plan_rows(p)
    teachers = _validate_teacher_inputs(p, rows)

    data_root, source_rows, _, baseline, baseline_reference = ref.load_inputs(source_clip, torch)
    if (baseline.get('arm') != 'rho1.5' or baseline.get('updates') != 8
            or len(baseline.get('module_masters', {})) != TARGET_LINEARS
            or baseline.get('normalization_c') != p['training']['normalization_c']):
        raise RuntimeError('C8 rho1.5 update8 baseline differs')
    anchor = p['rollout']['anchor_ids'][0]
    row0 = rows[anchor]
    if (anchor != 2000 or source_rows.get(anchor) != row0
            or baseline_reference.get('case_id') != anchor
            or baseline_reference.get('seed_index') != 0
            or baseline_reference.get('sampler_seed') != row0['sampler_seeds'][0]
            or baseline_reference.get('plan_row') != row0):
        raise RuntimeError('Baseline reference action must bind original anchor 2000 seed 0')
    if (len(p['calibration']['case_ids']) != UPDATES
            or len(p['calibration']['update_order']) != UPDATES
            or p['channel_scaling'] is not False):
        raise RuntimeError('Frozen 80-context update order or no-channel-scaling contract differs')

    deadline = ref.Deadline()
    runtime = None
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        banks = ClipBanks(runtime, p['calibration']['bank_directory'],
                          baseline['module_masters'], 'joint', p['clipping'])
        if (len(banks.modules) != TARGET_LINEARS or banks.train_weights is not True
                or banks.train_clips is not True or set(banks.modules) != set(baseline['module_masters'])):
            raise RuntimeError('Joint arm must train all 614 VQ masters and clipping ratios')
        groups = banks.optimizer_groups(p['training'])
        parameters = [parameter for group in groups for parameter in group['params']]
        expected_parameters = [parameter for name in banks.modules for parameter in
                               (banks.books[name], banks.logs[name], banks.ratios[name])]
        if (len(groups) != 2 or [group['lr'] for group in groups] != [0.0005, 0.01]
                or {id(v) for v in parameters} != {id(v) for v in expected_parameters}
                or any(not v.requires_grad for v in expected_parameters)
                or any(v.requires_grad for v in runtime.model.parameters())):
            raise RuntimeError('Only the requested VQ masters and clipping ratios may train')
        if any(float(value) != 1.0 for value in banks.ratios.values()):
            raise RuntimeError('Initial clipping ratios must all equal one')

        data = {}

        def datum(case):
            if case not in data:
                observation, metadata = common.load_observation(np, p, rows[case])
                data[case] = runtime.datum(observation), metadata
            return data[case]

        def progress(**values):
            state.update(values)
            _save_progress(root, artifact, state)

        def query(case, seed):
            value, metadata = datum(case)
            return ref.query(runtime, sampler, torch, value, metadata['description'],
                             rows[case]['sampler_seeds'][seed], banks)

        # Check the untrained C8 reference before a resume can restore trained masters.
        value, metadata = datum(anchor)
        runtime.description = metadata['description']
        initial_action, _, _ = query(anchor, 0)
        initial_parity = float((initial_action.float()
                                - torch.tensor(baseline_reference['action'], dtype=torch.float32)).abs().max())
        if not math.isfinite(initial_parity) or initial_parity > 1e-6:
            raise RuntimeError('Initial C8 baseline reference action parity exceeds 1e-6')

        optimizer = torch.optim.Adam(groups)
        history, updates = [], 0
        resume_lineage = []
        if checkpoint_path.exists():
            if not resume_from:
                raise RuntimeError('Existing checkpoint requires explicit verified --resume-from')
            previous = _resume_source(root, resume_from, checkpoint_path)
            resume_lineage = list(previous.get('resume_lineage', [])) + [resume_from]
            state['resume_lineage'] = resume_lineage
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            history, updates = saved['history'], saved['updates']
            order = p['calibration']['update_order']
            if (saved.get('protocol') != p or saved.get('arm') != p['arm']
                    or saved.get('update_order') != order or not isinstance(updates, int)
                    or not 0 <= updates <= UPDATES or len(history) != updates
                    or [item.get('case_id') for item in history] != order[:updates]
                    or [item.get('update') for item in history] != list(range(1, updates + 1))
                    or previous.get('updates') != updates):
                raise RuntimeError('Checkpoint protocol/history is not the verified exact prefix')
            banks.restore(saved['module_masters'])
            banks.restore_clips(saved['ratios'])
            optimizer.load_state_dict(saved['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda')
            restore_rng(torch, saved['rng'])
            job_start_updates = updates
        elif resume_from:
            raise RuntimeError('--resume-from was given but checkpoint is missing')
        if not checkpoint_path.exists():
            atomic_torch_save(torch, checkpoint_path, {
                'protocol': p, 'arm': p['arm'], 'updates': 0,
                'update_order': p['calibration']['update_order'], 'history': [],
                'module_masters': banks.snapshot(), 'ratios': banks.clip_snapshot(),
                'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
            })

        progress(phase='training', updates=updates, history=history,
                 initial_baseline_reference_max_abs=initial_parity,
                 updates_this_job=0)

        while updates < UPDATES:
            deadline.check(max([item.get('seconds', 0) for item in history[-3:]], default=300) * 1.15)
            case = p['calibration']['update_order'][updates]
            value, metadata = datum(case)
            runtime.description = metadata['description']
            teacher = teachers[case]
            banks.seen.clear()
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            action, counts, future = sampler.differentiable_action(
                runtime, value, rows[case]['sampler_seeds'][0], checkpointed=True, return_future=True)
            action_mse = (action[:10, :6].float()
                          - teacher['action'][:10, :6].to('cuda').float()).square().mean()
            latent_nmse = latent_reference.future_nmse_loss(torch, future, teacher['future'].to('cuda'))
            loss = (action_mse + p['training']['rho']
                    * p['training']['normalization_c'] * latent_nmse)
            if (len(banks.seen) != TARGET_LINEARS or not loss.requires_grad
                    or not bool(torch.isfinite(loss))):
                raise RuntimeError('Training coverage or loss check failed')
            loss.backward()
            coverage = banks.coverage()
            for group in ('weight', 'clip'):
                item = coverage.get(group, {})
                if (item.get('connected') != TARGET_LINEARS or item.get('finite') != TARGET_LINEARS
                        or item.get('nonzero', 0) < 1):
                    raise RuntimeError(f'Gradient coverage failed for {group}: {item}')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if updates == 0 and p['training']['optimizer_reset_after_update1']:
                optimizer.state.clear()
            banks.project()
            torch.cuda.synchronize()
            updates += 1
            history.append({
                'update': updates, 'case_id': case, 'seed_index': 0,
                'action_mse': float(action_mse), 'future_latent_nmse': float(latent_nmse),
                'loss': float(loss), 'gradient_norm': float(norm), 'coverage': coverage,
                'seconds': time.monotonic() - started, 'pbs_jobid': os.environ['PBS_JOBID'], **counts,
            })
            del action, future, action_mse, latent_nmse, loss
            atomic_torch_save(torch, checkpoint_path, {
                'protocol': p, 'arm': p['arm'], 'updates': updates,
                'update_order': p['calibration']['update_order'], 'history': history,
                'module_masters': banks.snapshot(), 'ratios': banks.clip_snapshot(),
                'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
            })
            progress(phase='training', updates=updates, updates_this_job=updates - job_start_updates,
                     history=history, last_update=history[-1], checkpoint=str(checkpoint_path))

        deadline.check(300)
        masters, ratios = banks.snapshot(), banks.clip_snapshot()
        if (len(masters) != TARGET_LINEARS or len(ratios) != TARGET_LINEARS
                or any(t.dtype != torch.float32 for v in masters.values()
                       for t in (v['books'], v['log_scales']))
                or any(v.dtype != torch.float32 for v in ratios.values())):
            raise RuntimeError('Final deployment must contain 614 FP32 books/log_scales and ratios')
        deployment = {
            'schema_version': 1, 'protocol': p['protocol'], 'arm': p['arm'],
            'updates': UPDATES, 'epochs': 1, 'target_linears': TARGET_LINEARS,
            'module_masters': masters, 'ratios': ratios,
            'normalization_c': p['training']['normalization_c'], 'rho': p['training']['rho'],
            'case_ids': p['calibration']['case_ids'], 'update_order': p['calibration']['update_order'],
            'channel_scaling': False,
        }
        if (deployment['arm'] != 'rollout_joint' or deployment['updates'] != UPDATES
                or set(deployment['module_masters']) != set(deployment['ratios'])):
            raise RuntimeError('Rollout joint deployment metadata differs')
        atomic_torch_save(torch, out / 'deployment.pt', deployment)
        progress(phase='reference_parity', updates=updates, history=history,
                 deployment=str(out / 'deployment.pt'))

        # Recompute and save the final deployed action at the original anchor.
        value, metadata = datum(anchor)
        runtime.description = metadata['description']
        banks.seen.clear()
        with torch.no_grad():
            action, counts, _ = sampler.differentiable_action(
                runtime, value, row0['sampler_seeds'][0], checkpointed=False, return_future=True)
        actual_action, _ = runtime.infer(value, row0['sampler_seeds'][0], measure=False)
        final_parity = float((actual_action.detach().float().cpu()
                              - action.detach().float().cpu()).abs().max())
        if (len(banks.seen) != TARGET_LINEARS or tuple(action.shape) != (32, 7)
                or not math.isfinite(final_parity) or final_parity > 1e-6):
            raise RuntimeError('Final sampler/runtime.infer parity exceeds 1e-6')
        reference_action = {
            'protocol': p['protocol'], 'arm': p['arm'], 'case_id': anchor, 'seed_index': 0,
            'sampler_seed': row0['sampler_seeds'][0], 'action': action.detach().float().cpu().tolist(),
            'target_linears': len(banks.seen), 'counts': counts,
            'actual_infer_parity': {'max_abs': final_parity, 'passed': True}, 'plan_row': row0,
            'initial_baseline_reference_max_abs': initial_parity,
        }
        common.save(out / 'reference_action.json', reference_action)
        banks.restore(masters)
        banks.restore_clips(ratios)
        _freeze_all(runtime, banks)
        if any(v.requires_grad for runtime_group in (banks.books, banks.logs, banks.ratios)
               for v in runtime_group.values()):
            raise RuntimeError('Deployment parameters are not frozen for evaluation')

        eval_root = root / 'results/heldout/rollout_joint'
        query_root = eval_root / 'queries'
        query_root.mkdir(parents=True, exist_ok=True)
        evaluation_records = []
        completed = 0
        for case, seed in query_pairs(p):
            deadline.check(180)
            row = rows[case]
            path = query_root / f'{case}_{seed}.json'
            if path.is_file():
                record = common.read(path)
                allowed_jobs = set(resume_lineage) | {os.environ['PBS_JOBID']}
                source_jobid = record.get('source_jobid')
                if (source_jobid not in allowed_jobs or record.get('pbs_jobid') not in allowed_jobs):
                    raise RuntimeError('Reused held-out endpoint is outside the verified resume lineage')
                validate_candidate_record(record, p, row, seed, out / 'deployment.pt', source_jobid)
            else:
                teacher = validate_teacher(torch, common.teacher_path(p, row, seed), row, seed)
                action, future, counts = query(case, seed)
                record = {
                    **candidate_binding(p, row, seed, out / 'deployment.pt', os.environ['PBS_JOBID']),
                    'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts,
                    'teacher_path': str(common.teacher_path(p, row, seed)),
                    **metrics(torch, latent_reference, action, future, teacher, p['training']),
                }
                validate_candidate_record(record, p, row, seed, out / 'deployment.pt', os.environ['PBS_JOBID'])
                common.save(path, record)
            evaluation_records.append(record)
            completed += 1
            progress(phase='evaluating', updates=updates, history=history,
                     completed_evaluation_queries=completed, evalqueries80=80,
                     last_case=case, last_seed=seed, allfrozen=True)

        require_exact_coverage(evaluation_records, p)
        stats = aggregate_records(evaluation_records, c80, source_protocol)
        heldout_summary = {
            'protocol': p['protocol'], 'arm': p['arm'], 'split': 'heldout', 'complete': True,
            'updates': UPDATES, 'queries': 80, 'new_teacher_queries': 0,
            'pbs_jobid': os.environ['PBS_JOBID'], 'all_parameters_frozen': True,
            'initial_baseline_reference_max_abs': initial_parity,
            'final_reference_action_max_abs': final_parity, **stats,
        }
        common.save(eval_root / 'summary.json', heldout_summary)
        comparison = _write_comparison(p, c80, rows, evaluation_records)
        final = {
            **state, 'status': 'complete', 'phase': 'complete', 'complete': True,
            'updates': UPDATES, 'trainingupdates80': 80, 'evaluation_queries': 80,
            'training_updates': 80, 'updates_this_job': updates - job_start_updates,
            'evalqueries80': 80, 'new_teacher_queries': 0, 'allfrozen': True,
            'all_parameters_frozen': True, 'resume_lineage': resume_lineage,
            'history': history, 'update_order': p['calibration']['update_order'],
            'heldout_summary': heldout_summary, 'comparison': comparison,
            'initial_baseline_reference_max_abs': initial_parity,
            'final_reference_action_max_abs': final_parity,
            'deployment': str(out / 'deployment.pt'),
            'reference_action': str(out / 'reference_action.json'),
            'evidence_scope': 'fixed-input decoded-BF16 VQ + A4; no closed-loop or native-kernel claim',
        }
        _save_progress(root, artifact, final)
        common.save(root / 'results/train_eval/summary.json', final)
        common.save(artifact / 'train_eval_summary.json', final)
        (artifact / 'TRAINING_COMPLETE').write_text('complete\n', encoding='utf-8')
        print('ROLLOUT_C80_TRAIN_EVAL_COMPLETE', flush=True)
        return 0
    except TimeoutError as exc:
        state.update({'phase': 'checkpointed', 'status': 'checkpointed', 'complete': False,
                      'reason': str(exc), 'checkpoint': str(checkpoint_path),
                      'updates': locals().get('updates', state.get('updates', 0)),
                      'history': locals().get('history', state.get('history', [])),
                      'completed_evaluation_queries': state.get('completed_evaluation_queries', 0)})
        _save_progress(root, artifact, state)
        common.save(root / 'results/train_eval/summary.json', state)
        (artifact / 'TRAINING_CHECKPOINTED').write_text('checkpointed\n', encoding='utf-8')
        return 75
    except Exception as exc:
        state.update({'phase': 'failed', 'status': 'failed', 'complete': False,
                      'error': type(exc).__name__ + ': ' + str(exc),
                      'updates': locals().get('updates', state.get('updates', 0)),
                      'history': locals().get('history', state.get('history', []))})
        _save_progress(root, artifact, state)
        common.save(root / 'results/train_eval/summary.json', state)
        raise
    finally:
        if runtime is not None and getattr(runtime, 'env', None) is not None:
            try:
                runtime.env.close()
            except Exception as exc:
                print('ROLLOUT_C80_CLEANUP_ERROR ' + type(exc).__name__ + ': ' + str(exc),
                      file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=('train_eval',), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--resume-from')
    args = parser.parse_args(argv)
    p = common.validate(common.read(args.protocol))
    common.guard()
    if not os.environ.get('ARTIFACTS'):
        raise RuntimeError('PBS ARTIFACTS path is required')
    return run_train_eval(p, args.resume_from)


if __name__ == '__main__':
    raise SystemExit(main())
