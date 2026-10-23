"""Train and evaluate the fresh joint per-channel arm inside its PBS allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80
import joint_rmse_reference as joint_ref
import per_channel_reference as per_ref


def validate_record(record, p, row, split, seed):
    expected = {'protocol': p['protocol'], 'arm': p['arm'], 'split': split,
        'updates': 80, 'case_id': row['case_id'], 'suite': row['suite'],
        'plan_row': row, 'seed_index': seed, 'sampler_seed': row['sampler_seeds'][seed],
        'weight_path': 'joint_per_channel_compensated_BF16_weights'}
    if any(record.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Saved endpoint binding differs')
    for metric in c80.METRICS:
        value = record.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError('Invalid endpoint metric: ' + metric)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _source_contract(p, validate_records=False):
    root = Path(p['remote_root'])
    control = Path(p['control_root'])
    per_root = Path(p['per_channel_root'])
    joint_root = Path(p['joint_control_root'])
    if _json(control / 'protocol.json') != p['source_control_protocol']:
        raise RuntimeError('Original C80 control protocol differs')
    actual_per = _json(per_root / 'protocol.json')
    if any(actual_per.get(k) != v for k, v in p['source_per_channel_protocol'].items()):
        raise RuntimeError('Completed per-channel protocol differs')
    if _json(joint_root / 'protocol.json') != p['source_joint_protocol']:
        raise RuntimeError('Completed joint-C80 consumer protocol differs')
    control_result = _json(control / 'results/comparison.json')
    per_result = _json(per_root / 'results/comparison.json')
    joint_result = _json(joint_root / 'results/comparison.json')
    if (control_result.get('complete') is not True or control_result.get('protocol') != p['source_control_protocol']['protocol']
            or control_result.get('evaluation_queries') != 320):
        raise RuntimeError('Original baseline/clip-only C80 endpoints are incomplete')
    if (per_result.get('complete') is not True or per_result.get('protocol') != p['source_per_channel_protocol']['protocol']
            or per_result.get('evaluation_queries') != p['source_per_channel_protocol']['expected']['total_endpoint_records']):
        raise RuntimeError('Completed per-channel aggregate is incomplete')
    if (joint_result.get('complete') is not True or joint_result.get('protocol') != p['source_joint_protocol']['protocol']
            or joint_result.get('evaluation_queries') != 480):
        raise RuntimeError('Completed joint-C80 consumer aggregate is incomplete')
    plan = _json(p['calibration']['plan_file'])
    if plan.get('status') != 'complete' or plan.get('calibration_cohorts', {}).get('c80') != p['calibration']['case_ids']:
        raise RuntimeError('Frozen C80 plan differs')
    rows = {int(r['case_id']): r for r in plan['inputs']}
    required = set(p['calibration']['case_ids']) | set(p['evaluation']['case_ids'])
    if not required <= set(rows):
        raise RuntimeError('Frozen C80 plan rows are incomplete')
    for split, path, cases, seeds in (
        ('calibration', control / 'results/teacher/calibration/receipt.json', p['calibration']['case_ids'], [0]),
        ('heldout', Path(p['clipping_root']) / 'results/teacher/receipt.json', p['evaluation']['case_ids'], [0, 1])):
        receipt = _json(path)
        if not receipt.get('complete') or receipt.get('queries') != 80 or receipt.get('case_ids') != cases or receipt.get('seed_indices') != seeds:
            raise RuntimeError('Shared teacher coverage differs for ' + split)
    # Require every source's stored terminal summaries as well as its aggregate receipt.
    if validate_records:
        _validate_source_summaries(p, rows)
    return root, rows


def _validate_source_summaries(p, rows):
    for split in ('calibration', 'heldout'):
        for arm in ('baseline', 'clip_only'):
            records = [_json(Path(p['control_root']) / 'results' / split / arm / 'queries' / f'{case}_{seed}.json')
                       for case, seed in c80.queries(p['source_control_protocol'], split)]
            for r in records:
                c80.validate_binding(r, p['source_control_protocol'], rows[r['case_id']], split, arm, r['seed_index'])
            stats = c80.aggregate_records(records, p, split)
            saved = _json(Path(p['control_root']) / 'results' / split / arm / 'summary.json')
            expected_updates = 80 if arm == 'clip_only' else 0
            if (saved.get('complete') is not True or saved.get('protocol') != p['source_control_protocol']['protocol']
                    or saved.get('arm') != arm or saved.get('split') != split
                    or saved.get('updates') != expected_updates
                    or any(saved.get(k) != v for k, v in stats.items())):
                raise RuntimeError('C80 control summary differs from bound endpoint rows')

        origin = Path(p['per_channel_root'])
        per_p = p['source_per_channel_protocol']
        per_records = [_json(origin / 'results' / split / 'per_channel_clip' / 'queries' / f'{case}_{seed}.json')
                       for case, seed in c80.queries(per_p, split)]
        for r in per_records:
            per_ref.validate_record(r, per_p, rows[r['case_id']], split, r['seed_index'])
        stats = c80.aggregate_records(per_records, p, split)
        saved = _json(origin / 'results' / split / 'per_channel_clip' / 'summary.json')
        if (saved.get('complete') is not True or saved.get('protocol') != per_p['protocol']
                or saved.get('arm') != 'per_channel_clip' or saved.get('split') != split
                or saved.get('updates') != 80 or any(saved.get(k) != v for k, v in stats.items())):
            raise RuntimeError('Per-channel summary differs from bound endpoint rows')

        joint_p = p['source_joint_protocol']; joint_root = Path(p['joint_control_root'])
        joint_summary = _json(joint_root / 'results' / split / 'joint_c80' / 'summary.json')
        evaluation = _json(joint_root / 'results/evaluate/summary.json')
        source_receipt = _json(joint_p['joint_source']['receipt'])
        source_job = evaluation.get('source_pbs_jobid')
        if (source_job != source_receipt.get('pbs_jobid') or source_receipt.get('complete') is not True
                or source_receipt.get('updates') != 80
                or evaluation.get('source_checkpoint') != joint_p['joint_source']['checkpoint']):
            raise RuntimeError('Joint-C80 endpoints do not bind to the verified training checkpoint')
        joint_records = [_json(joint_root / 'results' / split / 'joint_c80' / 'queries' / f'{case}_{seed}.json')
                         for case, seed in c80.queries(joint_p, split)]
        for r in joint_records:
            joint_ref.validate_candidate_record(r, joint_p, rows[r['case_id']], split, r['seed_index'],
                                                source_job, c80.METRICS)
        joint_stats = joint_ref.aggregate_endpoints(joint_records, joint_p, split)
        if (not joint_summary.get('complete') or joint_summary.get('protocol') != joint_p['protocol']
                or joint_summary.get('arm') != 'joint_c80' or joint_summary.get('split') != split
                or joint_summary.get('updates') != 80 or joint_summary.get('pbs_jobid') != evaluation.get('pbs_jobid')
                or joint_summary.get('source_pbs_jobid') != source_job
                or joint_summary.get('source_checkpoint') != joint_p['joint_source']['checkpoint']
                or any(joint_summary.get(k) != v for k, v in joint_stats.items())):
            raise RuntimeError('Joint-C80 summary differs from bound endpoint rows')


def run(p):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from metrics_reference import metrics
    from joint_banks import JointChannelBanks
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, optimizer_to_device, to_cpu

    sampler._require_compute_allocation()
    artifacts = Path(os.environ['ARTIFACTS'])
    job_id = os.environ['PBS_JOBID']
    checks = {
        'clip_selfcheck.json': ('passed',),
        'channel_selfcheck.json': ('passed', 'independent_reference_passed', 'old_surrogate_rejected'),
        'joint_selfcheck.json': ('passed', 'weight_gradient_reference_passed',
                                 'frozen_parameter_gradients_preserved', 'master_gradient_connected'),
    }
    check_results = {}
    for filename, keys in checks.items():
        check = _json(artifacts / filename)
        if check.get('pbs_jobid') != job_id or not all(check.get(k) is True for k in keys):
            raise RuntimeError('Required GPU selfcheck failed: ' + filename)
        check_results[filename] = check
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    root, rows = _source_contract(p)
    data_root, _, _, deployment, reference_action = ref.load_inputs(p['source_clipping_protocol'], torch)
    runtime = None
    out = root / 'results/train_eval'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / 'checkpoint.pt'
    if checkpoint_path.exists():
        raise RuntimeError('Existing checkpoint requires explicit reconciliation; fresh joint initialization only')
    state = {'protocol': p['protocol'], 'unit': 'train_eval', 'pbs_jobid': job_id,
             'phase': 'loading', 'complete': False, 'updates': 0,
             'new_teacher_queries': 0, 'new_evaluation_queries_this_job': 0,
             'new_parity_queries_this_job': 0}

    def progress(**values):
        state.update(values)
        ref.save_json(out / 'progress.json', state)
        ref.save_json(out / ('executions_' + job_id + '.json'), state)
        print('JOINT_CHANNEL_PROGRESS ' + json.dumps(state), flush=True)

    deadline = ref.Deadline()
    progress()
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval(); runtime.arm('bf16'); data = {}
        banks = JointChannelBanks(runtime, p['calibration']['bank_directory'], deployment['module_masters'],
                                  p['clipping'], p['scaling'])
        initial_masters = banks.snapshot()
        groups = banks.optimizer_groups(p['training'], p['scaling'])
        parameters = [v for group in groups for v in group['params']]
        expected_ids = {id(v) for v in (list(banks.books.values()) + list(banks.logs.values())
                                       + list(banks.ratios.values()) + list(banks.shadows.values()))}
        if {id(v) for v in parameters} != expected_ids or len(parameters) != 614 * 2 + 614 * 2:
            raise RuntimeError('Optimizer must contain exactly books, row gains, ratios and per-channel D')
        if any(v.requires_grad for v in runtime.model.parameters()):
            raise RuntimeError('Non-target model parameters must remain frozen')
        if any(torch.is_tensor(tensor) and tensor.requires_grad
               for encoding in banks.encodings.values() for tensor in encoding.values()):
            raise RuntimeError('VQ indices and original exponents must remain frozen')

        def datum(case):
            if case not in data:
                obs, meta = calibration_source.load_observation(np, data_root, rows[case])
                data[case] = runtime.datum(obs), meta
            return data[case]

        def query(case, seed):
            deadline.check()
            value, meta = datum(case)
            return ref.query(runtime, sampler, torch, value, meta['description'],
                             rows[case]['sampler_seeds'][seed], banks)

        def teacher(split, case, seed):
            directory = (Path(p['control_root']) / 'results/teacher/calibration'
                         if split == 'calibration' else Path(p['clipping_root']) / 'results/teacher')
            saved = torch.load(directory / f'{case}_{seed}.pt', map_location='cpu', weights_only=True)
            if saved.get('plan_row') != rows[case] or saved.get('sampler_seed') != rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Shared teacher plan row or seed differs')
            return saved

        first = p['calibration']['case_ids'][0]
        initial = query(first, 0); state['new_parity_queries_this_job'] += 1
        initial_parity = float((initial[0].float() - torch.tensor(reference_action['action'], dtype=torch.float32)).abs().max())
        if initial_parity != 0:
            raise RuntimeError('D=1 and ratio=1 must reproduce the C8 baseline exactly')
        progress(initial_parity_max_abs=initial_parity, initialization='fresh C8 rho1.5 update8; D=1; ratio=1')
        optimizer = torch.optim.Adam(groups)
        updates, history = 0, []
        while updates < 80:
            deadline.check(max([h['seconds'] for h in history[-3:]], default=300) * 1.15)
            case = p['calibration']['update_order'][updates]
            value, meta = datum(case); runtime.description = meta['description']; banks.seen.clear()
            target = teacher('calibration', case, 0)
            optimizer.zero_grad(set_to_none=True); started = time.monotonic()
            action, counts, future = sampler.differentiable_action(runtime, value,
                rows[case]['sampler_seeds'][0], checkpointed=True, return_future=True)
            action_mse = (action[:10, :6].float() - target['action'][:10, :6].to('cuda').float()).square().mean()
            latent_nmse = latent_reference.future_nmse_loss(torch, future, target['future'].to('cuda'))
            loss = action_mse + p['training']['rho'] * p['training']['normalization_c'] * latent_nmse
            if len(banks.seen) != 614 or not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise RuntimeError('Training loss/module coverage failed')
            loss.backward(); coverage = banks.coverage()
            if any(v['connected'] != 614 or v['finite'] != 614 or v['nonzero'] < 1 for v in coverage.values()):
                raise RuntimeError('Joint parameter gradients are disconnected or nonfinite')
            norm = torch.nn.utils.clip_grad_norm_(parameters, p['training']['gradient_norm_clip'], error_if_nonfinite=True)
            optimizer.step()
            if updates == 0 and p['training']['optimizer_reset_after_update1']:
                optimizer.state.clear()
            banks.project(); torch.cuda.synchronize(); updates += 1
            history.append({'update': updates, 'case_id': case, 'seed_index': 0,
                'action_mse': float(action_mse), 'future_latent_nmse': float(latent_nmse),
                'loss': float(loss), 'gradient_norm': float(norm), 'coverage': coverage,
                'seconds': time.monotonic() - started, 'pbs_jobid': job_id, **counts})
            del action, future, action_mse, latent_nmse, loss
            atomic_torch_save(torch, checkpoint_path, {'protocol': p, 'updates': updates, 'history': history,
                'module_masters': banks.snapshot(), 'ratios': banks.clip_snapshot(),
                'shadow_log2_scales': banks.scale_snapshot(), 'optimizer': to_cpu(optimizer.state_dict()),
                'rng': capture_rng(torch)})
            progress(phase='training', updates=updates, last_update=history[-1])

        progress(phase='folding_parity', updates=80)
        uncached = query(first, 0); state['new_parity_queries_this_job'] += 1
        final_masters = banks.snapshot()
        rounding = banks.cache_compensated_weights()
        folded = query(first, 0); state['new_parity_queries_this_job'] += 1
        parity = {'action_max_abs': float((uncached[0].float() - folded[0].float()).abs().max()),
                  'future_max_abs': float((uncached[1].float() - folded[1].float()).abs().max())}
        if any(value != 0 for value in parity.values()):
            raise RuntimeError('Full cached and uncached action/future queries differ')
        final_parameters = (list(banks.books.values()) + list(banks.logs.values())
                            + list(banks.ratios.values()) + list(banks.shadows.values()))
        if any(value.requires_grad for value in final_parameters):
            raise RuntimeError('Books, row gains, ratios and per-channel D must be frozen for final evaluation')
        frozen = banks.verify_joint(initial_masters)
        deployment_value = {'protocol': p, 'updates': 80, 'module_masters': final_masters,
            'ratios': banks.clip_snapshot(), 'log2_scales': banks.scale_snapshot(),
            'scale_metadata_dtype': 'float32_log2', 'initialization': 'C8 rho1.5 update8, ratio=1, log2D=0',
            'source_checkpoint': str(Path(p['baseline_root']) / 'results/training/rho1.5/deployment.pt'),
            'weight_compensation_rounding': {k: v for k, v in rounding.items() if k != 'per_module'}}
        atomic_torch_save(torch, out / 'deployment.pt', deployment_value)
        training = {'protocol': p['protocol'], 'arm': p['arm'], 'complete': True, 'updates': 80,
            'history': history, 'update_order': p['calibration']['update_order'],
            'initial_parity_max_abs': initial_parity, 'cached_query_parity': parity,
            'weight_compensation_rounding': deployment_value['weight_compensation_rounding'],
            'gradient_selfchecks': check_results, 'pbs_jobid': job_id, **frozen}
        ref.save_json(out / 'training_receipt.json', training)
        for split in ('calibration', 'heldout'):
            records = []
            for case, seed in c80.queries(p, split):
                target = teacher(split, case, seed)
                if split == 'calibration' and case == first:
                    action, future, counts = folded
                else:
                    action, future, counts = query(case, seed)
                    state['new_evaluation_queries_this_job'] += 1
                record = {'protocol': p['protocol'], 'arm': p['arm'], 'split': split, 'updates': 80,
                    'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                    'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed],
                    'weight_path': 'joint_per_channel_compensated_BF16_weights', 'pbs_jobid': job_id,
                    'counts': counts, **metrics(torch, latent_reference, action, future, target, p['training'])}
                validate_record(record, p, rows[case], split, seed)
                ref.save_json(root / 'results' / split / p['arm'] / 'queries' / f'{case}_{seed}.json', record)
                records.append(record)
                progress(phase='evaluation', split=split, queries=len(records), last_case=case, last_seed=seed)
            ref.save_json(root / 'results' / split / p['arm'] / 'summary.json', {
                'protocol': p['protocol'], 'complete': True, 'arm': p['arm'], 'split': split,
                'updates': 80, 'pbs_jobid': job_id, **c80.aggregate_records(records, p, split)})
        ref.save_json(out / 'summary.json', {**state, 'complete': True, 'phase': 'complete',
            'updates': 80, 'evaluation_queries': 160, 'cached_query_parity': parity, **frozen})
        progress(phase='complete', complete=True, updates=80)
        return 0
    except TimeoutError as exc:
        progress(phase='checkpointed', reason=str(exc))
        return 75
    except Exception as exc:
        progress(phase='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


def _percent(candidate, control, metric):
    denominator = control[metric]
    return 100 * (candidate[metric] / denominator - 1) if denominator else None


def _paired(records_a, records_b, metric):
    def means(records):
        groups = {}
        for record in records:
            groups.setdefault(record['case_id'], []).append(record[metric])
        return {case: sum(vals) / len(vals) for case, vals in groups.items()}
    a, b = means(records_a), means(records_b)
    if set(a) != set(b):
        raise RuntimeError('Paired held-out case cohorts differ')
    deltas = [a[case] - b[case] for case in sorted(a)]
    return {'cases': len(deltas), 'improved': sum(v < 0 for v in deltas),
            'worsened': sum(v > 0 for v in deltas), 'tied': sum(v == 0 for v in deltas)}


def aggregate(p):
    root, rows = _source_contract(p, validate_records=True)
    arm_roots = {'baseline': Path(p['control_root']), 'clip_only': Path(p['control_root']),
                 'joint_c80': Path(p['joint_control_root']),
                 'per_channel_clip': Path(p['per_channel_root']), p['arm']: root}
    arms, records_by_arm, changes, paired, suite_changes = {}, {}, {}, {}, {}
    for split in ('calibration', 'heldout'):
        arms[split], records_by_arm[split] = {}, {}
        for arm, origin in arm_roots.items():
            records = [_json(origin / 'results' / split / arm / 'queries' / f'{case}_{seed}.json')
                       for case, seed in c80.queries(p, split)]
            for record in records:
                row = rows[record['case_id']]
                if arm == p['arm']:
                    validate_record(record, p, row, split, record['seed_index'])
                elif arm in ('baseline', 'clip_only'):
                    c80.validate_binding(record, p['source_control_protocol'], row, split, arm, record['seed_index'])
                elif arm == 'per_channel_clip':
                    per_ref.validate_record(record, p['source_per_channel_protocol'], row, split, record['seed_index'])
                else:
                    evaluation = _json(Path(p['joint_control_root']) / 'results/evaluate/summary.json')
                    joint_ref.validate_candidate_record(record, p['source_joint_protocol'], row, split,
                        record['seed_index'], evaluation['source_pbs_jobid'], c80.METRICS)
            stats = c80.aggregate_records(records, p, split)
            summary = _json(origin / 'results' / split / arm / 'summary.json')
            expected_protocol = p['protocol'] if arm == p['arm'] else (
                p['source_joint_protocol']['protocol'] if arm == 'joint_c80' else
                p['source_per_channel_protocol']['protocol'] if arm == 'per_channel_clip' else
                p['source_control_protocol']['protocol'])
            if (summary.get('complete') is not True or summary.get('protocol') != expected_protocol
                    or summary.get('arm') != arm or summary.get('split') != split
                    or summary.get('updates') != (0 if arm == 'baseline' else 80)
                    or any(summary.get(k) != v for k, v in stats.items())):
                raise RuntimeError('Saved ' + split + '/' + arm + ' summary differs from its bound records')
            arms[split][arm] = summary; records_by_arm[split][arm] = records
        changes[split], paired[split], suite_changes[split] = {}, {}, {}
        for control in ('per_channel_clip', 'joint_c80'):
            a, b = arms[split][p['arm']], arms[split][control]
            changes[split][control] = {m: _percent(a['overall'], b['overall'], m) for m in c80.METRICS}
            paired[split][control] = {m: _paired(records_by_arm[split][p['arm']], records_by_arm[split][control], m)
                                      for m in ('first10motor_rmse', 'future_latent_nmse')}
            suite_changes[split][control] = {suite: {m: _percent(a['by_suite'][suite], b['by_suite'][suite], m)
                for m in c80.METRICS} for suite in a['by_suite']}
    receipt = _json(root / 'results/train_eval/training_receipt.json')
    required = ('indices_unchanged', 'original_exponents_unchanged', 'continuous_scale_metadata_separate',
                'module_masters_fp32', 'codebooks_optimized', 'row_gains_optimized')
    if (receipt.get('complete') is not True or receipt.get('updates') != 80
            or not all(receipt.get(k) is True for k in required)
            or [h['case_id'] for h in receipt.get('history', [])] != p['calibration']['update_order']
            or any(v != 0 for v in receipt.get('cached_query_parity', {}).values())):
        raise RuntimeError('Exact80 joint training and full cached parity are required')
    if p['expected']['new_endpoint_records'] != 160 or p['expected']['reused_endpoint_records'] != 640 or p['expected']['total_endpoint_records'] != 800:
        raise RuntimeError('Endpoint count contract must be 640 reused + 160 new = 800')
    result = {'protocol': p['protocol'], 'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
        'training_updates': 80, 'evaluation_queries': 800, 'new_teacher_queries': 0,
        'new_endpoint_records': 160, 'reused_endpoint_records': 640, 'total_endpoint_records': 800,
        'arms': arms, 'percent_change_vs_controls': changes,
        'paired_case_counts_after_seed_average': paired,
        'suite_percent_change_vs_controls': suite_changes, 'training': receipt, 'scope': p['scope']}
    ref.save_json(root / 'results/comparison.json', result)
    lines = ['# Joint per-channel C80 evaluation', '',
        '新 arm 从 C8 rho1.5 update8 初始化，在原 C80、原 loss、teacher 与 seed 上联合优化 VQ books、log row gains、614 个 Linear 的 per-channel D 向量与 ratios。',
        '没有沿用已训练 per-channel checkpoint；held-out 为 40 contexts × 2 paired seeds。4 组旧端点共 640 records，新 arm 160 records，总计 800。', '',
        '| Held-out arm | First10 motor RMSE | Future latent NMSE |', '|---|---:|---:|']
    for arm in arms['heldout']:
        m = arms['heldout'][arm]['overall']
        lines.append(f"| {arm} | {m['first10motor_rmse']:.9f} | {m['future_latent_nmse']:.9f} |")
    lines += ['', 'joint_per_channel 相对 per_channel_clip 与 joint_c80 的变化（负值表示数值降低）：', '']
    for control in ('per_channel_clip', 'joint_c80'):
        vals = changes['heldout'][control]
        lines.append(f"- vs {control}: " + ', '.join(f'{m} {v:+.3f}%' if v is not None else f'{m} undefined' for m, v in vals.items()))
        for metric, counts in paired['heldout'][control].items():
            lines.append(f"  - {metric}: {counts['improved']} improved / {counts['worsened']} worsened / {counts['tied']} tied across {counts['cases']} cases after seed averaging.")
    lines += ['', 'Held-out 四个 suite 的 primary metric 差异：']
    for control in ('per_channel_clip', 'joint_c80'):
        for suite, values in suite_changes['heldout'][control].items():
            action = f"{values['first10motor_rmse']:+.3f}%" if values['first10motor_rmse'] is not None else 'undefined'
            latent = f"{values['future_latent_nmse']:+.3f}%" if values['future_latent_nmse'] is not None else 'undefined'
            lines.append(f"- vs {control}, {suite}: first10motor_rmse {action}，future_latent_nmse {latent}。")
    lines += ['', '每个 suite 的全指标、Calibration 差异与指标绝对值一并写入 comparison.json。Late22/full32/gripper 为诊断项。',
        '结果范围是 fixed-input decoded-BF16 VQ + fake A4 数值诊断，不代表 closed-loop 或 native-kernel 性能。']
    (root / 'results/RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (root / 'results/PIPELINE_COMPLETE').write_text('complete\n', encoding='utf-8')
    print('JOINT_CHANNEL_AGGREGATE_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=('train_eval', 'aggregate'), required=True)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    c80.guard()
    p = _json(args.protocol)
    if args.unit == 'aggregate':
        aggregate(p)
        return 0
    return run(p)


if __name__ == '__main__':
    raise SystemExit(main())
