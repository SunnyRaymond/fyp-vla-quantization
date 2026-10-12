"""Reuse the frozen SpinQuant bank on the action-aware VQ paired observations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import time
import traceback

ARMS = ('spinquant_native_w4a4',)
METRICS = ('motor_rmse_first10_vs_bf16', 'gripper_rmse_first10_vs_bf16',
           'motor_rmse_32_vs_bf16')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or 'login' in host.lower():
        raise RuntimeError('Approved PBS compute allocation required')
    if host not in {n.split('.')[0] for n in Path(nodefile).read_text().split()}:
        raise RuntimeError('Host is absent from PBS_NODEFILE')


def aggregate(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record['case_id'], []).append(record)
    if not grouped or any(len(rows) != 2 for rows in grouped.values()):
        raise RuntimeError('Expected exactly two paired sampler seeds per case')
    per_case = {str(case): {metric: statistics.mean(row[metric] for row in rows)
                           for metric in METRICS} for case, rows in grouped.items()}
    return {'case_count': len(grouped), 'context_count': len(records), 'per_case': per_case,
            **{metric: statistics.mean(row[metric] for row in per_case.values()) for metric in METRICS}}


def run(args):
    guard()
    import torch
    import rotation
    import runner as pilot
    import run_diagnostic as source
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    if p['protocol'] != 'fastwam-spinquant-rmse-v1' or tuple(p['arms']) != ARMS:
        raise RuntimeError('Unexpected frozen SpinQuant RMSE protocol')
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / 'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text()) != p:
        raise RuntimeError('Cannot resume with a changed protocol')
    save(frozen, p)
    artifact = Path(os.environ['ARTIFACTS'])
    job = os.environ['PBS_JOBID']
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['soft_deadline_seconds']
    root = Path(p['source_root'])
    summary = json.loads((root / 'results/full/summary.json').read_text())
    if summary['status'] != 'complete' or summary['pbs_jobid'] != p['source_full_job']:
        raise RuntimeError('Source VQ campaign is not the declared completed campaign')
    plan = {row['case_id']: row for row in json.loads((root / 'plan.json').read_text())['inputs']}
    splits = {case: split for split in ('selection', 'test') for case in p[split + '_cases']}
    expected = {(arm, case, seed) for arm in ARMS for case in splits for seed in p['seed_indices']}
    cached = {}
    winner_label = f"row_action_vq_a4_lambda_{summary['selected_action_lambda']:g}"
    for line in (root / 'results/full/actions.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        if row['case_id'] not in splits:
            continue
        if row['arm'] == winner_label:
            row['arm'] = 'row_action_vq_a4'
        if row['arm'] not in p['reference_arms']:
            continue
        key = (row['arm'], row['case_id'], row['seed_index'])
        if key in cached or row['sampler_seed'] != plan[row['case_id']]['sampler_seeds'][row['seed_index']]:
            raise RuntimeError('Duplicate or unpaired cached action')
        if row['domain'] != plan[row['case_id']]['domain'] or row['dimension'] != plan[row['case_id']]['dimension']:
            raise RuntimeError('Cached domain/dimension differs from frozen plan')
        action = torch.tensor(row['action'], dtype=torch.float32)
        if tuple(action.shape) != (32, 7) or not bool(torch.isfinite(action).all()):
            raise RuntimeError('Invalid cached reference action')
        cached[key] = {**row, **source.action_metrics(torch, action, action)}
        cached[key]['action'] = action
    for arm in p['reference_arms']:
        for case in splits:
            for seed in p['seed_indices']:
                row = cached[arm, case, seed]
                row.update(source.action_metrics(torch, row['action'], cached['bf16', case, seed]['action']))
    records = {}
    query_dir = args.out / 'query_records'
    query_dir.mkdir(exist_ok=True)
    for path in query_dir.glob('*.json'):
        row = json.loads(path.read_text())
        key = (row['arm'], row['case_id'], row['seed_index'])
        case, seed = row['case_id'], row['seed_index']
        if (key not in expected or key in records or row['protocol'] != p['protocol']
                or row['input_identity'] != plan[case]
                or row['sampler_seed'] != plan[case]['sampler_seeds'][seed]):
            raise RuntimeError('Saved query identity differs from frozen protocol')
        action = torch.tensor(row['action'], dtype=torch.float32)
        if tuple(action.shape) != (32, 7) or not bool(torch.isfinite(action).all()):
            raise RuntimeError('Invalid resumed action')
        row.update(source.action_metrics(torch, action, cached['bf16', case, seed]['action']))
        records[key] = row

    runtime, hooks = None, []
    seen, query_seconds = set(), []
    torch.backends.cuda.matmul.allow_tf32 = False

    def progress(stage, **extra):
        value = {'status': 'running', 'protocol': p['protocol'], 'job_id': job, 'stage': stage,
                 'new_queries': len(records), 'expected_queries': len(expected), **extra}
        save(args.out / 'progress.json', value)
        print('SPIN_RMSE_PROGRESS ' + json.dumps(value), flush=True)

    def checkpoint():
        value = {'status': 'checkpointed', 'protocol': p['protocol'], 'job_id': job,
                 'new_queries': len(records), 'expected_queries': len(expected), 'requires_resume': True}
        save(args.out / 'checkpoint.json', value)
        save(artifact / 'summary.json', value)
        print('SPIN_RMSE_CHECKPOINT ' + json.dumps(value), flush=True)

    try:
        progress('model_loading')
        runtime = pilot.Runtime(artifact)
        runtime.arm('bf16')
        if 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Frozen reference hardware class requires A100')
        first_case = p['test_cases'][0]
        observation, metadata = source.load_case(root, plan[first_case])
        runtime.description = metadata['description']
        action, _ = runtime.infer(runtime.datum(observation), plan[first_case]['sampler_seeds'][0], measure=False)
        parity = float((action.detach().float().cpu() - cached['bf16', first_case, 0]['action']).abs().max())
        save(artifact / 'bf16_replay.json', {'case_id': first_case, 'max_abs': parity, 'tolerance': 1e-6})
        if parity > 1e-6:
            raise RuntimeError(f'BF16 replay does not match cached reference: {parity}')
        bank = rotation.apply_native_bank(runtime.q, p['spinquant_bank'])
        if (bank['arm'] != 'spinquant_adapted_w4a4' or bank['rotation_mode'] != 'learned'
                or bank['module_count'] != 614
                or bank['experiment_metadata']['training']['selected_step'] != 200):
            raise RuntimeError('Unexpected frozen learned SpinQuant bank')
        save(artifact / 'bank_reuse.json', {'bank': p['spinquant_bank'], 'manifest': bank,
             'new_training_steps': 0, 'new_calibration_cases': 0, 'base_model_loads': 1})
        for name, module in runtime.q.modules.items():
            hooks.append(module.register_forward_hook(lambda _m, _a, _o, n=name: seen.add(n)))
        for arm in ARMS:
            remaining = [key for key in sorted(expected) if key[0] == arm and key not in records]
            if not remaining:
                continue
            if time.time() > deadline - 180:
                checkpoint()
                return
            progress('arm_setup', arm=arm)
            runtime.q.enable('w4a4')
            runtime.current_arm = arm
            for _, case, seed in remaining:
                reserve = max(180, 2 * statistics.mean(query_seconds[-4:])) if query_seconds else 180
                if time.time() > deadline - reserve:
                    checkpoint()
                    return
                progress('query', arm=arm, case_id=case, seed_index=seed)
                observation, metadata = source.load_case(root, plan[case])
                runtime.description = metadata['description']
                seen.clear()
                native_before = runtime.q._native_int4_gemm_calls
                started = time.perf_counter()
                action, _ = runtime.infer(runtime.datum(observation), plan[case]['sampler_seeds'][seed], measure=False)
                action = action.detach().float().cpu()
                seconds = time.perf_counter() - started
                native_calls = runtime.q._native_int4_gemm_calls - native_before
                if len(seen) != 614 or native_calls <= 0 or not runtime.q._native_int4_ptx_verified:
                    raise RuntimeError('Query failed target coverage/native INT4 evidence')
                record = {'protocol': p['protocol'], 'job_id': job, 'arm': arm, 'case_id': case,
                          'seed_index': seed, 'sampler_seed': plan[case]['sampler_seeds'][seed],
                          'input_identity': plan[case], 'split': splits[case], 'domain': plan[case]['domain'],
                          'dimension': plan[case]['dimension'], 'action': action.tolist(),
                          **source.action_metrics(torch, action, cached['bf16', case, seed]['action']),
                          'observed_module_count': len(seen), 'native_int4_calls': native_calls,
                          'kv_cache_dtype': 'bf16', 'episodes': 0, 'predicted_actions_executed': 0,
                          'wall_seconds_diagnostic_only': seconds}
                save(query_dir / f'{arm}_{case}_{seed}.json', record)
                records[arm, case, seed] = record
                query_seconds.append(seconds)
        if set(records) != expected:
            raise RuntimeError('Frozen paired query coverage is incomplete')
        arms = {}
        for arm in p['reference_arms'] + list(ARMS):
            rows = [row for key, row in (records if arm in ARMS else cached).items() if key[0] == arm]
            arms[arm] = {split: aggregate([row for row in rows if row['case_id'] in p[split + '_cases']])
                         for split in ('selection', 'test')}
            arms[arm]['test_groups'] = {domain: aggregate([row for row in rows
                if row['case_id'] in p['test_cases'] and row['domain'] == domain]) for domain in ('original', 'plus')}
        primary = METRICS[0]
        comparisons = {}
        for arm in ARMS:
            comparisons[arm] = {}
            for ref in p['reference_arms'][1:]:
                deltas = [arms[arm]['test']['per_case'][str(case)][primary] - arms[ref]['test']['per_case'][str(case)][primary]
                          for case in p['test_cases']]
                comparisons[arm][ref] = {'mean_rmse_difference': statistics.mean(deltas),
                                       'improved_cases': sum(delta < 0 for delta in deltas), 'cases': len(deltas)}
        value = {'status': 'complete', 'protocol': p['protocol'], 'phase': 'full', 'job_id': job,
                 'source_full_job': p['source_full_job'], 'spinquant_bank': p['spinquant_bank'],
                 'new_queries': len(records), 'target_linears': 614, 'arms': arms,
                 'paired_comparisons': comparisons,
                 'bf16_replay_max_abs': parity, 'base_model_loads_this_allocation': 1,
                 'new_training_steps': 0, 'new_calibration_cases': 0, 'evaluation_episodes': 0,
                 'claim_scope': p['claim_scope']}
        save(args.out / 'summary.json', value)
        save(artifact / 'summary.json', value)
        lines = ['# SpinQuant 与 action-aware VQ：同输入 action RMSE', '',
                 '22 个 test observations × 2 sampler seeds；先在 case 内平均，再对 case 等权平均。', '',
                 '| 配置 | Selection motor RMSE | Test motor RMSE | Test gripper RMSE | Test 32-step motor RMSE |',
                 '|---|---:|---:|---:|---:|']
        for arm, item in arms.items():
            lines.append(f"| {arm} | {item['selection'][primary]:.6f} | {item['test'][primary]:.6f} | "
                         f"{item['test'][METRICS[1]]:.6f} | {item['test'][METRICS[2]]:.6f} |")
        lines += ['', 'SpinQuant 直接使用原 closed-loop 的 native INT4 实现；本轮没有 decoded-BF16 SpinQuant 对照。',
                  '同一个200-step learned bank、原packed codes/scales/rotation，均未重新拟合；KV BF16。',
                  '这些是已有 observation cohort 上的描述性复用比较，训练数据和量化方法不同；不表示成功率或 native speedup。']
        (args.out / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        progress('complete')
        save(args.out / 'progress.json', {'status': 'complete', 'job_id': job, 'new_queries': len(records)})
        print('SPIN_RMSE_COMPLETE queries=52 test_contexts=44', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        if runtime is not None:
            if hasattr(runtime.q, '_rotation_native_state'):
                rotation.discard_native_bank(runtime.q)
            runtime.q.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        traceback.print_exc()
        if os.environ.get('PBS_JOBID') and 'login' not in socket.gethostname().lower():
            save(args.out / 'failure.json', {'job_id': os.environ['PBS_JOBID'],
                 'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
