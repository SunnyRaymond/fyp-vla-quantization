"""Paired C8 clipping experiment; all model work requires a real PBS allocation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import time


HERE = Path(__file__).resolve().parent
METRICS = ('first10motor_rmse', 'full32motor_rmse', 'late22motor_rmse',
           'first10gripper_rmse', 'future_latent_nmse')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def aggregate(records, protocol):
    expected = {(case, seed) for case in protocol['evaluation']['case_ids']
                for seed in protocol['evaluation']['seed_indices']}
    actual = [(r['case_id'], r['seed_index']) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Evaluation must contain the exact 40 x 2 paired queries')
    by_suite = {}
    for suite in SUITES:
        rows = [r for r in records if r['suite'] == suite]
        cases = sorted({r['case_id'] for r in rows})
        if len(cases) != 10 or len(rows) != 20:
            raise ValueError('Each suite must contain exactly ten cases and two seeds')
        by_suite[suite] = {metric: sum(
            sum(r[metric] for r in rows if r['case_id'] == case) / 2 for case in cases) / 10
            for metric in METRICS}
    overall = {metric: sum(row[metric] for row in by_suite.values()) / 4 for metric in METRICS}
    if not all(math.isfinite(value) and value >= 0 for value in overall.values()):
        raise ValueError('Evaluation metrics must be finite and nonnegative')
    return {'queries': len(records), 'cases': 40, 'overall': overall, 'by_suite': by_suite}


def gate(summaries):
    required = {'baseline', 'weight_only', 'clip_only', 'joint'}
    if set(summaries) != required or any(not s.get('complete') or s.get('queries') != 80
                                         for s in summaries.values()):
        raise ValueError('All four complete 80-query arms are required')
    metrics = ('first10motor_rmse', 'future_latent_nmse')
    improvements = {}
    for arm in ('weight_only', 'clip_only', 'joint'):
        improvements[arm] = {m: 100 * (summaries[arm]['overall'][m]
                                       / summaries['baseline']['overall'][m] - 1)
                             for m in metrics}
    clip_better = all(improvements['clip_only'][m] < 0 for m in metrics)
    joint_better = all(improvements['joint'][m] < 0 and
                       summaries['joint']['overall'][m] < summaries['weight_only']['overall'][m]
                       for m in metrics)
    return {'advance_to_channel_scaling': clip_better or joint_better,
            'clip_only_both_improved': clip_better,
            'joint_beats_baseline_and_weight_only_on_both': joint_better,
            'percent_change_vs_baseline': improvements,
            'scope': 'fixed-cohort engineering screen; no significance or closed-loop claim'}


class Deadline:
    def __init__(self):
        self.end = float(os.environ['FW_JOB_START_EPOCH']) + float(
            os.environ.get('FW_SOFT_DEADLINE_SECONDS', '6840'))
        self.interrupted = False
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

    def stop(self, *_):
        self.interrupted = True

    def check(self, estimate=90):
        if self.interrupted or time.time() + estimate + 120 >= self.end:
            raise TimeoutError('Soft deadline reached; completed updates/queries are reusable')


def load_inputs(protocol, torch):
    parent = Path(protocol['parent_root'])
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    reference = parent / 'results/reference'
    reference_receipt = json.loads((reference / 'reference_receipt.json').read_text())
    cache = torch.load(reference / 'reference_cache.pt', map_location='cpu', weights_only=True)
    training_dir = parent / 'results/training' / protocol['parent_arm']
    receipt = json.loads((training_dir / 'training_receipt.json').read_text())
    deployment = torch.load(training_dir / 'deployment.pt', map_location='cpu', weights_only=True)
    reference_action = json.loads((training_dir / 'reference_action.json').read_text())
    if (not receipt.get('complete') or receipt.get('updates') != protocol['parent_updates']
            or deployment.get('arm') != 'rho1.5' or deployment.get('updates') != 8
            or deployment.get('target_linears') != 614
            or deployment.get('case_ids') != protocol['calibration']['case_ids']
            or reference_receipt.get('complete') is not True
            or reference_receipt.get('passed') is not True
            or len(deployment.get('module_masters', {})) != 614):
        raise RuntimeError('Parent must be the completed C8 rho1.5 update8 checkpoint')
    for source in (deployment, cache['normalization']):
        if not math.isclose(source['normalization_c'], protocol['training']['normalization_c'],
                            rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError('Frozen C8 normalization differs')
    if parent_protocol['calibration']['case_ids'] != protocol['calibration']['case_ids']:
        raise RuntimeError('C8 calibration inputs changed')
    plan_path = Path(protocol['calibration']['plan_file'])
    plan = json.loads(plan_path.read_text())
    if (plan.get('status') != 'complete' or plan.get('calibration_cohorts', {}).get('c8')
            != protocol['calibration']['case_ids']):
        raise RuntimeError('Frozen input plan is unavailable or changed')
    by_id = {int(row['case_id']): row for row in plan['inputs']}
    for saved in cache['rows']:
        if saved['plan_row'] != by_id[saved['case_id']]:
            raise RuntimeError('C8 row differs from the teacher reference')
    calibration = [by_id[c] for c in protocol['calibration']['case_ids']]
    test = [by_id[c] for c in protocol['evaluation']['case_ids']]
    if Counter(r['suite'] for r in test) != Counter({s: 10 for s in SUITES}):
        raise RuntimeError('Held-out inputs do not cover four suites equally')
    calibration_demos = {(r['suite'], r['task_id'], r['demo_id']) for r in calibration}
    if any((r['suite'], r['task_id'], r['demo_id']) in calibration_demos for r in test):
        raise RuntimeError('Held-out demonstration leaks into calibration')
    if any(r.get('domain') != 'original' or len(r['sampler_seeds']) != 2 for r in test):
        raise RuntimeError('Held-out input domain or paired seeds differ')
    return plan_path.parent, by_id, cache, deployment, reference_action


def query(runtime, sampler, torch, datum, description, seed, banks=None):
    runtime.description = description
    if banks:
        banks.seen.clear()
    with torch.no_grad():
        action, counts, future = sampler.differentiable_action(
            runtime, datum, seed, checkpointed=False, return_future=True)
    if banks and len(banks.seen) != 614:
        raise RuntimeError('Query must reach all 614 target Linears')
    if tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28):
        raise RuntimeError('Action/future output shape differs')
    if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all()):
        raise RuntimeError('Nonfinite model output')
    return action.detach().cpu(), future.detach().cpu(), counts


def validate_prepare(root, protocol, source_job):
    """Validate completed numerical work after the known post-result cleanup failure."""
    import re
    import subprocess
    if not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', source_job):
        raise ValueError('Invalid source PBS job ID')
    observed = subprocess.run(['qstat', '-xf', source_job], capture_output=True,
                              text=True, check=True, timeout=35).stdout
    if not re.search(r'^\s*job_state\s*=\s*[FX]\s*$', observed, re.M):
        raise RuntimeError('Source GPU job is not terminal')
    if not re.search(r'^\s*Exit_status\s*=\s*1\s*$', observed, re.M):
        raise RuntimeError('Source failure differs from the recorded cleanup failure')
    artifacts = root / 'artifacts' / source_job
    if (artifacts / 'exit_code.txt').read_text().strip() != '1':
        raise RuntimeError('Original artifact exit code differs')
    with (artifacts / 'job.log').open('rb') as stream:
        stream.seek(max(0, (artifacts / 'job.log').stat().st_size - 16384))
        tail = stream.read().decode('utf-8', errors='replace')
    if ('ARM_COMPLETE ' not in tail
            or "AttributeError: 'Runtime' object has no attribute 'close'" not in tail):
        raise RuntimeError('Only the known post-ARM_COMPLETE cleanup error may be repaired')
    summary = json.loads((root / 'results/baseline/summary.json').read_text())
    teacher = json.loads((root / 'results/teacher/receipt.json').read_text())
    selfcheck = json.loads((artifacts / 'clip_selfcheck.json').read_text())
    if (summary.get('complete') is not True or summary.get('queries') != 80
            or summary.get('pbs_jobid') != source_job or summary.get('updates') != 0
            or summary.get('initial_parity_max_abs') != 0.
            or summary.get('clipping', {}).get('count') != 614
            or summary['clipping']['min'] != 1. or summary['clipping']['max'] != 1.
            or not teacher.get('complete') or teacher.get('queries') != 80
            or teacher.get('pbs_jobid') != source_job
            or teacher.get('case_ids') != protocol['evaluation']['case_ids']
            or teacher.get('seed_indices') != [0, 1]
            or not selfcheck.get('passed')):
        raise RuntimeError('Completed baseline/teacher/parity/selfcheck evidence is incomplete')
    plan = json.loads(Path(protocol['calibration']['plan_file']).read_text())
    by_id = {row['case_id']: row for row in plan['inputs']}
    records = []
    for case in protocol['evaluation']['case_ids']:
        for seed in protocol['evaluation']['seed_indices']:
            record = json.loads((root / 'results/baseline/queries' / f'{case}_{seed}.json').read_text())
            if (record.get('arm') != 'baseline' or record.get('updates') != 0
                    or record.get('case_id') != case or record.get('seed_index') != seed
                    or record.get('sampler_seed') != by_id[case]['sampler_seeds'][seed]
                    or record.get('suite') != by_id[case]['suite']
                    or not (root / 'results/teacher' / f'{case}_{seed}.pt').is_file()):
                raise RuntimeError('A paired baseline/teacher query is missing or misbound')
            records.append(record)
    recomputed = aggregate(records, protocol)
    if any(recomputed[key] != summary[key] for key in ('queries', 'cases', 'overall', 'by_suite')):
        raise RuntimeError('Baseline metrics do not reproduce from all 80 query records')
    receipt = {'protocol': protocol['protocol'], 'complete': True, 'queries': 80,
               'purpose': 'validation_only', 'pbs_jobid': os.environ['PBS_JOBID'],
               'source_job_id': source_job, 'source_exit_code': 1,
               'failure': 'Runtime.close AttributeError after complete numerical results',
               'new_model_forwards': 0, 'recomputed_from_query_records': True,
               'original_failure_artifacts_preserved': True}
    save_json(root / 'results/baseline/validation.json', receipt)
    print('PREPARE_VALIDATION_COMPLETE ' + json.dumps(receipt), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare', 'arm', 'aggregate', 'validate-prepare'), required=True)
    parser.add_argument('--arm', choices=('weight_only', 'clip_only', 'joint'))
    parser.add_argument('--source-job')
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    root = Path(protocol['remote_root'])
    import sampler
    sampler._require_compute_allocation()
    if args.phase == 'validate-prepare':
        return validate_prepare(root, protocol, args.source_job)
    if args.phase == 'aggregate':
        summaries = {a: json.loads((root / 'results' / a / 'summary.json').read_text())
                     for a in protocol['arms']}
        result = {'protocol': protocol['protocol'], 'complete': True,
                  'pbs_jobid': os.environ['PBS_JOBID'], 'arms': summaries, 'gate': gate(summaries)}
        save_json(root / 'results/comparison.json', result)
        lines = ['# C8 rho1.5 activation clipping 四组 held-out 结果', '',
                 '每组40 contexts x 2 seeds = 80 paired queries；四suite等权。无closed-loop episodes。', '',
                 '| Arm | 额外updates | First10 motor RMSE | Full32 motor RMSE | Late22 motor RMSE | Future latent NMSE |',
                 '|---|---:|---:|---:|---:|---:|']
        for arm in protocol['arms']:
            summary = summaries[arm]
            m = summary['overall']
            lines.append(f"| {arm} | {summary['updates']} | {m['first10motor_rmse']:.9f} | "
                         f"{m['full32motor_rmse']:.9f} | {m['late22motor_rmse']:.9f} | "
                         f"{m['future_latent_nmse']:.9f} |")
        lines += ['', '相对baseline变化（负值为改善）：', '',
                  '| Arm | First10 motor RMSE变化 | Future latent NMSE变化 |', '|---|---:|---:|']
        for arm, changes in result['gate']['percent_change_vs_baseline'].items():
            lines.append(f"| {arm} | {changes['first10motor_rmse']:+.3f}% | "
                         f"{changes['future_latent_nmse']:+.3f}% |")
        lines += ['', 'Clipping参数：', '', '| Arm | Min | Mean | Max | Ratios < 1 |', '|---|---:|---:|---:|---:|']
        for arm, summary in summaries.items():
            c = summary['clipping']
            lines.append(f"| {arm} | {c['min']:.6f} | {c['mean']:.6f} | {c['max']:.6f} | {c['below_one']}/614 |")
        lines += ['', f"预设channel scaling推进gate：{result['gate']['advance_to_channel_scaling']}。",
                  'Gate仅为固定held-out cohort的工程筛选；没有统计显著性、closed-loop成功率或native INT4 latency结论。',
                  '从同一C8 rho1.5 update8 checkpoint初始化；后三组固定额外8updates，未按held-out挑选checkpoint。']
        (root / 'results/RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (root / 'results/PIPELINE_COMPLETE').write_text('complete\n')
        print('COMPARISON_COMPLETE ' + json.dumps(result), flush=True)
        return 0

    import torch
    import numpy as np
    import runner
    import calibration_source
    import latent_reference
    from clip_banks import ClipBanks
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, optimizer_to_device, to_cpu

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    deadline = Deadline()
    data_root, by_id, calibration_cache, deployment, reference_action = load_inputs(protocol, torch)
    arm = 'baseline' if args.phase == 'prepare' else args.arm
    if not arm:
        parser.error('--arm is required for the arm phase')
    out = root / 'results' / arm
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / 'progress.json'
    state = {'protocol': protocol['protocol'], 'arm': arm, 'phase': 'loading',
             'updates': 0, 'queries': 0, 'complete': False, 'pbs_jobid': os.environ['PBS_JOBID']}

    def progress(**values):
        state.update(values)
        save_json(progress_path, state)
        print('CLIP_PROGRESS ' + json.dumps(state), flush=True)

    runtime = None
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        data = {}

        def datum(case):
            if case not in data:
                observation, metadata = calibration_source.load_observation(np, data_root, by_id[case])
                data[case] = runtime.datum(observation), metadata
            return data[case]

        teacher_dir = root / 'results/teacher'
        teacher_dir.mkdir(parents=True, exist_ok=True)
        if args.phase == 'prepare':
            progress(phase='teacher')
            for case in protocol['evaluation']['case_ids']:
                value, metadata = datum(case)
                for seed_index in protocol['evaluation']['seed_indices']:
                    seed = by_id[case]['sampler_seeds'][seed_index]
                    path = teacher_dir / f'{case}_{seed_index}.pt'
                    if path.is_file():
                        saved = torch.load(path, map_location='cpu', weights_only=True)
                        if saved['plan_row'] != by_id[case] or saved['sampler_seed'] != seed:
                            raise RuntimeError('Cached held-out teacher binding differs')
                        continue
                    deadline.check()
                    action, future, counts = query(runtime, sampler, torch, value,
                                                   metadata['description'], seed)
                    atomic_torch_save(torch, path, {'protocol': protocol['protocol'],
                        'case_id': case, 'seed_index': seed_index, 'sampler_seed': seed,
                        'plan_row': by_id[case], 'action': action, 'future': future, 'counts': counts})
                    progress(phase='teacher', teacher_last_case=case, teacher_last_seed=seed_index)
            save_json(teacher_dir / 'receipt.json', {'protocol': protocol['protocol'],
                      'complete': True, 'queries': 80, 'case_ids': protocol['evaluation']['case_ids'],
                      'seed_indices': [0, 1], 'pbs_jobid': os.environ['PBS_JOBID']})
        else:
            teacher_receipt = json.loads((teacher_dir / 'receipt.json').read_text())
            if not teacher_receipt.get('complete') or teacher_receipt.get('queries') != 80:
                raise RuntimeError('Shared held-out BF16 teacher is incomplete')

        banks = ClipBanks(runtime, protocol['calibration']['bank_directory'],
                          deployment['module_masters'], arm, protocol['clipping'])
        progress(phase='initial_parity')
        first = protocol['calibration']['case_ids'][0]
        value, metadata = datum(first)
        deadline.check()
        action, _, _ = query(runtime, sampler, torch, value, metadata['description'],
                             by_id[first]['sampler_seeds'][0], banks)
        expected = torch.tensor(reference_action['action'], dtype=torch.float32)
        initial_max_abs = float((action.float() - expected).abs().max())
        if initial_max_abs > 1e-6:
            raise RuntimeError(f'Common C8 rho1.5 initialization parity failed: {initial_max_abs}')
        progress(initial_parity_max_abs=initial_max_abs)
        training = protocol['training']
        groups = banks.optimizer_groups(training)
        optimizer = torch.optim.Adam(groups) if groups else None
        checkpoint_path = out / 'checkpoint.pt'
        updates, history = 0, []
        if checkpoint_path.is_file():
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if payload['protocol'] != protocol or payload['arm'] != arm:
                raise RuntimeError('Resume checkpoint protocol differs')
            updates, history = payload['updates'], payload['history']
            banks.restore(payload['module_masters'])
            banks.restore_clips(payload['ratios'])
            optimizer.load_state_dict(payload['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda')
            restore_rng(torch, payload['rng'])

        def checkpoint():
            atomic_torch_save(torch, checkpoint_path, {'protocol': protocol, 'arm': arm,
                'updates': updates, 'history': history, 'module_masters': banks.snapshot(),
                'ratios': banks.clip_snapshot(), 'optimizer': to_cpu(optimizer.state_dict()),
                'rng': capture_rng(torch)})

        if arm != 'baseline':
            parameters = [p for group in groups for p in group['params']]
            while updates < training['additional_updates']:
                estimate = max([r['seconds'] for r in history[-3:]], default=900) * 1.15
                deadline.check(estimate)
                case = protocol['calibration']['case_ids'][updates]
                value, metadata = datum(case)
                runtime.description = metadata['description']
                banks.seen.clear()
                optimizer.zero_grad(set_to_none=True)
                started = time.monotonic()
                action, counts, future = sampler.differentiable_action(
                    runtime, value, by_id[case]['sampler_seeds'][0],
                    checkpointed=True, return_future=True)
                teacher = calibration_cache['teacher'][str(case)]
                action_mse = (action[:10, :6].float() - teacher['action'][:10, :6].to('cuda').float()).square().mean()
                latent_nmse = latent_reference.future_nmse_loss(torch, future, teacher['future'].to('cuda'))
                loss = action_mse + training['rho'] * training['normalization_c'] * latent_nmse
                if len(banks.seen) != 614 or not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError('Training loss or 614-module coverage failed')
                loss.backward()
                coverage = banks.coverage()
                if any(r['connected'] != 614 or r['finite'] != 614 or r['nonzero'] < 1
                       for r in coverage.values()):
                    raise RuntimeError(f'Trainable parameter gradients disconnected/nonfinite: {coverage}')
                norm = torch.nn.utils.clip_grad_norm_(parameters, training['gradient_norm_clip'],
                                                       error_if_nonfinite=True)
                optimizer.step()
                if updates == 0 and training['optimizer_reset_after_update1']:
                    optimizer.state.clear()
                banks.project()
                torch.cuda.synchronize()
                updates += 1
                history.append({'update': updates, 'case_id': case, 'action_mse': float(action_mse),
                    'future_latent_nmse': float(latent_nmse), 'loss': float(loss),
                    'gradient_norm': float(norm), 'coverage': coverage,
                    'seconds': time.monotonic() - started, **counts})
                del action, future, loss, action_mse, latent_nmse
                checkpoint()
                progress(phase='training', updates=updates, last_update=history[-1],
                         clipping=banks.clip_summary())
            atomic_torch_save(torch, out / 'deployment.pt', {'protocol': protocol, 'arm': arm,
                'updates': updates, 'module_masters': banks.snapshot(), 'ratios': banks.clip_snapshot()})
            save_json(out / 'training_receipt.json', {'protocol': protocol['protocol'], 'arm': arm,
                'complete': True, 'updates': updates, 'calibration_ids': protocol['calibration']['case_ids'],
                'initial_parity_max_abs': initial_max_abs, 'history': history, 'clipping': banks.clip_summary(),
                'parent_training_job': protocol['parent_training_job'], 'pbs_jobid': os.environ['PBS_JOBID']})

        records = []
        for case in protocol['evaluation']['case_ids']:
            value, metadata = datum(case)
            for seed_index in protocol['evaluation']['seed_indices']:
                seed = by_id[case]['sampler_seeds'][seed_index]
                record_path = out / 'queries' / f'{case}_{seed_index}.json'
                if record_path.is_file():
                    record = json.loads(record_path.read_text())
                    if (record['arm'] != arm or record['sampler_seed'] != seed
                            or record['updates'] != updates):
                        raise RuntimeError('Evaluation query binding differs')
                else:
                    deadline.check()
                    teacher = torch.load(teacher_dir / f'{case}_{seed_index}.pt',
                                         map_location='cpu', weights_only=True)
                    if teacher['plan_row'] != by_id[case] or teacher['sampler_seed'] != seed:
                        raise RuntimeError('Shared teacher query does not match evaluation input')
                    action, future, counts = query(runtime, sampler, torch, value,
                                                   metadata['description'], seed, banks)
                    difference = action.float() - teacher['action'].float()
                    rmse = lambda tensor: float(tensor.square().mean().sqrt())
                    record = {'protocol': protocol['protocol'], 'arm': arm, 'case_id': case,
                        'suite': by_id[case]['suite'], 'seed_index': seed_index, 'sampler_seed': seed,
                        'updates': updates, 'first10motor_rmse': rmse(difference[:10, :6]),
                        'full32motor_rmse': rmse(difference[:, :6]),
                        'late22motor_rmse': rmse(difference[10:, :6]),
                        'first10gripper_rmse': rmse(difference[:10, 6]),
                        'future_latent_nmse': float(latent_reference.future_nmse_loss(torch, future, teacher['future'])),
                        'counts': counts}
                    save_json(record_path, record)
                records.append(record)
                progress(phase='evaluation', updates=updates, queries=len(records),
                         last_case=case, last_seed=seed_index)
        summary = {'protocol': protocol['protocol'], 'arm': arm, 'complete': True,
                   'updates': updates, 'initial_parity_max_abs': initial_max_abs,
                   'clipping': banks.clip_summary(), 'pbs_jobid': os.environ['PBS_JOBID'],
                   **aggregate(records, protocol)}
        save_json(out / 'summary.json', summary)
        progress(phase='complete', complete=True, queries=80)
        print('ARM_COMPLETE ' + json.dumps(summary), flush=True)
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


if __name__ == '__main__':
    raise SystemExit(main())
