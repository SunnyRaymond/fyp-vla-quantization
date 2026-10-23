"""Prepare the frozen C8 teacher/initial-bank references inside a PBS allocation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import time


PROTOCOL = 'fastwam-vq-c8-action-latent-plus-v1'
SOURCE_PROTOCOL = 'fastwam-vq-calibration-suites-v1'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)


def _allocation_guard() -> str:
    host = socket.gethostname().split('.', 1)[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError('A genuine PBS compute allocation is required')
    allocated = {line.split('.', 1)[0] for line in Path(nodefile).read_text(encoding='utf-8').split()}
    if 'login' in host.lower() or host not in allocated:
        raise RuntimeError(f'Host {host} is not part of the active PBS allocation')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('A GPU allocation is required for C8 reference preparation')
    return host


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _save_torch(torch, path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _binding(protocol: dict) -> dict:
    calibration = protocol.get('calibration')
    if not isinstance(calibration, dict):
        raise ValueError('Protocol calibration object is missing')
    root = Path(calibration['root']).resolve()
    plan_file = Path(calibration['plan_file'])
    if not plan_file.is_absolute():
        plan_file = root / plan_file
    plan_file = plan_file.resolve()
    bank_directory = Path(calibration['bank_directory']).resolve()
    case_ids = calibration.get('case_ids')
    if (protocol.get('protocol') != PROTOCOL or not protocol.get('remote_root')
            or not isinstance(case_ids, list) or len(case_ids) != 8
            or len(set(case_ids)) != 8 or any(type(case) is not int for case in case_ids)
            or not calibration.get('source_job_id') or calibration.get('cohort', 'c8') != 'c8'
            or calibration.get('domain', 'original') != 'original'
            or calibration.get('seed_index', 0) != 0 or calibration.get('contexts', 8) != 8
            or calibration.get('per_suite', 2) != 2 or calibration.get('suites', 4) != 4):
        raise ValueError('Protocol must bind the exact eight C8 inputs and source job')
    return {'protocol': protocol['protocol'], 'remote_root': protocol['remote_root'],
            'calibration': {'root': str(root), 'plan_file': str(plan_file),
                            'case_ids': list(case_ids), 'source_job_id': calibration['source_job_id'],
                            'bank_directory': str(bank_directory), 'cohort': 'c8',
                            'domain': 'original', 'seed_index': 0}}


def _load_c8_inputs(protocol: dict, binding: dict):
    calibration = binding['calibration']
    root, plan_path = Path(calibration['root']), Path(calibration['plan_file'])
    if not plan_path.is_file():
        raise FileNotFoundError(f'Frozen C8 plan is missing: {plan_path}')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('protocol') != SOURCE_PROTOCOL or plan.get('status') != 'complete':
        raise RuntimeError('The frozen C8 input plan is not complete')
    case_ids = calibration['case_ids']
    if plan.get('calibration_cohorts', {}).get('c8') != case_ids:
        raise RuntimeError('Protocol case_ids differ from calibration_cohorts.c8')
    rows = {int(row['case_id']): row for row in plan.get('inputs', [])}
    if len(rows) != len(plan.get('inputs', [])) or not set(case_ids).issubset(rows):
        raise RuntimeError('Frozen C8 plan has duplicate or missing input rows')
    selected = [rows[case] for case in case_ids]
    suites = Counter(row.get('suite') for row in selected)
    if suites != Counter({suite: 2 for suite in SUITES}):
        raise RuntimeError(f'C8 must contain exactly two rows per suite: {dict(suites)}')
    for row in selected:
        seeds = row.get('sampler_seeds')
        if (row.get('domain') != 'original' or row.get('dimension') != 'original'
                or row.get('variant') != 'original' or not isinstance(seeds, list)
                or len(seeds) != 2 or any(type(seed) is not int for seed in seeds)):
            raise RuntimeError(f'Invalid frozen C8 plan row: {row.get("case_id")}')

    summary_path = root / 'results' / 'c8' / 'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(f'Completed C8 source summary is missing: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected_bank = Path(calibration['bank_directory'])
    if (summary.get('status') != 'complete' or summary.get('protocol') != SOURCE_PROTOCOL
            or summary.get('cohort') != 'c8' or summary.get('pbs_jobid') != calibration['source_job_id']
            or summary.get('target_linears') != 614 or summary.get('fitted_linears') != 614
            or summary.get('calibration_case_ids') != case_ids
            or Path(summary.get('bank_directory', '')).resolve() != expected_bank):
        raise RuntimeError('C8 source summary, job, cases, banks, or 614 coverage differs')
    if not expected_bank.is_dir():
        raise FileNotFoundError(f'C8 bank directory is missing: {expected_bank}')
    if sum(1 for path in expected_bank.glob('*.pt') if path.is_file()) != 614:
        raise RuntimeError('C8 initial bank directory must contain exactly 614 module files')
    return plan_path.parent, selected, summary


def _tensor(torch, value, shape, label):
    if (not torch.is_tensor(value) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise RuntimeError(f'{label} must be finite with shape {tuple(shape)}')
    return value.detach().cpu().clone()


def _parity(torch, actual, expected) -> dict:
    delta = actual.detach().float().cpu() - expected.detach().float().cpu()
    return {'max_abs': float(delta.abs().max()), 'rmse': float(delta.square().mean().sqrt())}


def _query(runtime, sampler, datum, description: str, seed: int, *, banks=None):
    runtime.description = description
    if banks is not None:
        banks.seen.clear()
    action, counts, future = sampler.differentiable_action(
        runtime, datum, seed, checkpointed=False, return_future=True)
    if banks is not None and len(banks.seen) != 614:
        raise RuntimeError(f'Sampler reached only {len(banks.seen)}/614 C8 Linears')
    return action, counts, future


def _actual_infer(runtime, datum, description: str, seed: int, *, banks=None):
    runtime.description = description
    if banks is not None:
        banks.seen.clear()
    action, _ = runtime.infer(datum, seed, measure=False)
    if banks is not None and len(banks.seen) != 614:
        raise RuntimeError(f'pilot.infer reached only {len(banks.seen)}/614 C8 Linears')
    return action


def _validate_existing(torch, out: Path, binding: dict) -> bool:
    cache_path, receipt_path = out / 'reference_cache.pt', out / 'reference_receipt.json'
    cache_tmp, receipt_tmp = out / 'reference_cache.pt.tmp', out / 'reference_receipt.json.tmp'
    if any(path.exists() for path in (cache_tmp, receipt_tmp)):
        raise FileExistsError('An incomplete C8 reference write exists; refusing to replace it')
    if not cache_path.exists() and not receipt_path.exists():
        return False
    if not cache_path.is_file() or not receipt_path.is_file():
        raise FileExistsError('An incomplete C8 reference exists; refusing to overwrite it')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    if (receipt.get('passed') is not True or receipt.get('complete') is not True
            or receipt.get('status') != 'complete' or receipt.get('binding') != binding
            or cache.get('binding') != binding or cache.get('schema_version') != 1
            or cache.get('context_count') != 8 or cache.get('target_linears') != 614
            or set(cache.get('teacher', {})) != {str(case) for case in binding['calibration']['case_ids']}
            or set(cache.get('parent', {})) != {str(case) for case in binding['calibration']['case_ids']}
            or not isinstance(cache.get('normalization', {}).get('normalization_c'), (int, float))
            or not math.isfinite(float(cache.get('normalization', {}).get('normalization_c', float('nan'))))
            or len(cache.get('rows', [])) != 8
            or [row.get('case_id') for row in cache.get('rows', [])]
            != binding['calibration']['case_ids']
            or receipt.get('source_c8') != cache.get('source_c8')
            or cache.get('source_c8', {}).get('suite_case_counts') != {suite: 2 for suite in SUITES}
            or cache.get('counts', {}).get('training_performed') is not False):
        raise RuntimeError('Existing C8 reference cache/receipt is incomplete or bound to another protocol')
    for case in binding['calibration']['case_ids']:
        for name, records in (('teacher', cache['teacher']), ('parent', cache['parent'])):
            record = records[str(case)]
            _tensor(torch, record.get('action'), ACTION_SHAPE, f'existing {name} action {case}')
            _tensor(torch, record.get('future'), FUTURE_SHAPE, f'existing {name} future {case}')
    if any(cache.get('parity', {}).get(name, {}).get(str(case), {}).get('passed') is not True
           for name in ('teacher', 'parent') for case in binding['calibration']['case_ids']):
        raise RuntimeError('Existing C8 cache is missing a passed pilot.infer parity receipt')
    print('C8_REFERENCE_REUSED ' + json.dumps({'cache': str(cache_path), 'new_model_forwards': 0}), flush=True)
    return True


def run(protocol_path: Path, out: Path) -> None:
    host = _allocation_guard()
    import torch
    import numpy as np
    import calibration_source as source
    import sampler
    import runner as pilot
    import latent_reference
    from train_banks import TrainableBanks

    sampler._require_compute_allocation()
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    binding = _binding(protocol)
    data_root, rows, source_summary = _load_c8_inputs(protocol, binding)
    out.mkdir(parents=True, exist_ok=True)
    if _validate_existing(torch, out, binding):
        return
    started = time.monotonic()
    runtime = None
    try:
        runtime = pilot.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        if len(runtime.q.modules) != 614:
            raise RuntimeError('Pilot runtime does not expose all 614 target Linears')

        rows_payload, teacher, parent, parity = [], {}, {}, {'teacher': {}, 'parent': {}}
        datum_by_case = {}
        for row in rows:
            case = int(row['case_id'])
            observation, metadata = source.load_observation(np, data_root, row)
            if metadata.get('case_id') != case or not metadata.get('description'):
                raise RuntimeError(f'C8 metadata binding is invalid for case {case}')
            seed = int(row['sampler_seeds'][0])
            datum = runtime.datum(observation)
            datum_by_case[case] = datum
            teacher_action, counts, teacher_future = _query(
                runtime, sampler, datum, metadata['description'], seed)
            actual_action = _actual_infer(runtime, datum, metadata['description'], seed)
            teacher_parity = _parity(torch, actual_action, teacher_action)
            passed = (teacher_parity['max_abs'] <= 0.002 and teacher_parity['rmse'] <= 0.001)
            parity['teacher'][str(case)] = {'pilot_infer_vs_sampler': teacher_parity,
                                             'passed': passed, **counts}
            if not passed:
                raise RuntimeError(f'BF16 pilot.infer/sampler parity failed for case {case}')
            _tensor(torch, teacher_action, ACTION_SHAPE, f'BF16 teacher action {case}')
            _tensor(torch, teacher_future, FUTURE_SHAPE, f'BF16 teacher future {case}')
            teacher[str(case)] = {'case_id': case, 'suite': row['suite'], 'seed_index': 0,
                                  'sampler_seed': seed,
                                  'action': teacher_action.detach().cpu().clone(),
                                  'future': teacher_future.detach().cpu().clone()}
            rows_payload.append({'case_id': case, 'suite': row['suite'], 'seed_index': 0,
                                 'sampler_seed': seed, 'plan_row': row, 'metadata': metadata})
            del actual_action, teacher_action, teacher_future

        banks = TrainableBanks(runtime, protocol['calibration']['bank_directory'])
        for parameter in banks.parameters():
            parameter.requires_grad_(False)
        runtime.current_arm = 'micro_scale_vq_a4'
        for row_record in rows_payload:
            case = row_record['case_id']
            datum = datum_by_case[case]
            description, seed = row_record['metadata']['description'], row_record['sampler_seed']
            parent_action, counts, parent_future = _query(
                runtime, sampler, datum, description, seed, banks=banks)
            actual_action = _actual_infer(runtime, datum, description, seed, banks=banks)
            parent_parity = _parity(torch, actual_action, parent_action)
            passed = (parent_parity['max_abs'] <= 0.002 and parent_parity['rmse'] <= 0.001)
            parity['parent'][str(case)] = {'pilot_infer_vs_sampler': parent_parity,
                                            'passed': passed, **counts}
            if not passed:
                raise RuntimeError(f'C8 parent pilot.infer/sampler parity failed for case {case}')
            _tensor(torch, parent_action, ACTION_SHAPE, f'C8 parent action {case}')
            _tensor(torch, parent_future, FUTURE_SHAPE, f'C8 parent future {case}')
            parent[str(case)] = {'case_id': case, 'suite': row_record['suite'], 'seed_index': 0,
                                 'sampler_seed': seed,
                                 'action': parent_action.detach().cpu().clone(),
                                 'future': parent_future.detach().cpu().clone()}
            del actual_action, parent_action, parent_future

        calibration_rows = []
        for row in rows_payload:
            key = str(row['case_id'])
            action_mse = (parent[key]['action'][:10, :6].float()
                          - teacher[key]['action'][:10, :6].float()).square().mean()
            future_nmse = latent_reference.future_nmse_loss(
                torch, parent[key]['future'], teacher[key]['future'])
            calibration_rows.append({'case_id': row['case_id'], 'seed_index': 0,
                                     'split': 'calibration', 'action_mse': float(action_mse),
                                     'future_nmse': float(future_nmse)})
        mean_action = sum(row['action_mse'] for row in calibration_rows) / 8
        mean_future = sum(row['future_nmse'] for row in calibration_rows) / 8
        normalization_c = mean_action / mean_future if mean_future > 0 else float('nan')
        if not math.isfinite(normalization_c) or normalization_c <= 0:
            raise RuntimeError('C8-derived normalization_c is not finite and positive')
        normalization = {'policy': 'normalization_c = mean_C8_parent_action_mse / mean_C8_parent_future_nmse',
                         'calibration_ids': binding['calibration']['case_ids'],
                         'mean_cal_initial_action_mse': mean_action,
                         'mean_cal_initial_future_nmse': mean_future,
                         'normalization_c': normalization_c, 'per_case': calibration_rows}
        counts = {'teacher_sampler_contexts': 8, 'parent_sampler_contexts': 8,
                  'actual_pilot_infer_calls': 16, 'teacher_forward_count': 8,
                  'parent_forward_count': 8, 'model_forward_count': 32,
                  'target_linears': 614, 'gradient_updates': 0, 'training_performed': False}
        cache = {'schema_version': 1, 'protocol': PROTOCOL, 'binding': binding,
                 'source_c8': {'protocol': source_summary['protocol'], 'cohort': 'c8',
                               'source_job_id': source_summary['pbs_jobid'],
                               'case_ids': binding['calibration']['case_ids'],
                               'suite_case_counts': {suite: 2 for suite in SUITES},
                               'bank_directory': binding['calibration']['bank_directory']},
                 'context_count': 8, 'target_linears': 614, 'rows': rows_payload,
                 'teacher': teacher, 'parent': parent, 'normalization': normalization,
                 'parity': parity, 'counts': counts,
                 'prepared_host': host, 'elapsed_seconds': time.monotonic() - started}
        _save_torch(torch, out / 'reference_cache.pt', cache)
        receipt = {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'complete',
                   'complete': True, 'passed': True, 'binding': binding,
                   'source_c8': cache['source_c8'], 'context_count': 8,
                   'target_linears': 614, 'suite_case_counts': {suite: 2 for suite in SUITES},
                   'parity': parity, 'counts': counts, 'normalization': normalization,
                   'new_model_forwards': counts['model_forward_count'],
                   'no_training': True, 'elapsed_seconds': cache['elapsed_seconds']}
        _save_json(out / 'reference_receipt.json', receipt)
        print('C8_REFERENCE_COMPLETE ' + json.dumps(receipt), flush=True)
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.out)
