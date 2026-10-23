"""Train one frozen C8 action+future VQ arm and save a replayable action."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

from prepare_c8 import (ACTION_SHAPE, FUTURE_SHAPE, PROTOCOL, SUITES,
                        _binding, _save_json, _tensor)


def _allocation_guard() -> str:
    import socket
    host = socket.gethostname().split('.', 1)[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError('A genuine PBS compute allocation is required')
    allocated = {line.split('.', 1)[0] for line in Path(nodefile).read_text(encoding='utf-8').split()}
    if 'login' in host.lower() or host not in allocated:
        raise RuntimeError(f'Host {host} is not part of the active PBS allocation')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('A GPU allocation is required for C8 training')
    return host


def _atomic_torch_save(torch, path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_arm(protocol: dict, arm: str) -> dict:
    if protocol.get('protocol') != PROTOCOL:
        raise ValueError('Unexpected C8 action+latent protocol')
    if protocol.get('arms') != ['rho0.1', 'rho1.5']:
        raise ValueError('The frozen C8 protocol must contain exactly rho0.1 and rho1.5')
    config = protocol.get('arm_configs', {}).get(arm)
    expected = {'rho0.1': (0.1, 2), 'rho1.5': (1.5, 1)}
    if arm not in expected or not isinstance(config, dict):
        raise ValueError(f'Unknown frozen arm: {arm}')
    rho, epochs = expected[arm]
    if (float(config.get('rho', float('nan'))) != rho
            or type(config.get('epochs')) is not int or config['epochs'] != epochs):
        raise ValueError(f'{arm} rho/epochs differ from the frozen first-round choice')
    if config.get('updates', epochs * 8) != epochs * 8:
        raise ValueError(f'{arm} update count differs from eight C8 contexts per epoch')
    training = protocol.get('training', {})
    if (training.get('learning_rate') != 0.0005
            or training.get('gradient_norm_clip') != 1.0
            or training.get('optimizer') != 'Adam'
            or training.get('optimizer_reset_after_update1') is not True
            or training.get('reference_max_abs_tolerance') != 0.002
            or training.get('reference_rmse_tolerance') != 0.001):
        raise ValueError('Frozen Adam learning rate, clipping, or update-1 reset differs')
    return {'arm': arm, 'rho': rho, 'epochs': epochs,
            'learning_rate': 0.0005, 'gradient_norm_clip': 1.0,
            'optimizer_reset_after_update1': True}


def _load_reference(torch, reference: Path, binding: dict, case_ids: list[int], latent_reference):
    cache_path, receipt_path = reference / 'reference_cache.pt', reference / 'reference_receipt.json'
    if not cache_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError('A complete reference_cache.pt and reference_receipt.json are required')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    keys = {str(case) for case in case_ids}
    if (receipt.get('status') != 'complete' or receipt.get('complete') is not True
            or receipt.get('passed') is not True or receipt.get('binding') != binding
            or receipt.get('target_linears') != 614 or receipt.get('context_count') != 8
            or receipt.get('no_training') is not True
            or cache.get('schema_version') != 1 or cache.get('binding') != binding
            or cache.get('context_count') != 8 or cache.get('target_linears') != 614
            or set(cache.get('teacher', {})) != keys or set(cache.get('parent', {})) != keys):
        raise RuntimeError('C8 reference cache/receipt is incomplete or bound to another source')
    if (receipt.get('source_c8') != cache.get('source_c8')
            or cache.get('source_c8', {}).get('case_ids') != case_ids
            or cache.get('source_c8', {}).get('suite_case_counts') != {suite: 2 for suite in SUITES}
            or cache.get('source_c8', {}).get('protocol') != 'fastwam-vq-calibration-suites-v1'
            or cache.get('source_c8', {}).get('cohort') != 'c8'
            or cache.get('source_c8', {}).get('source_job_id')
            != binding['calibration']['source_job_id']
            or cache.get('source_c8', {}).get('bank_directory')
            != binding['calibration']['bank_directory']):
        raise RuntimeError('C8 reference receipt has a different source/cohort binding')
    parity = cache.get('parity', {})
    if any(parity.get(name, {}).get(str(case), {}).get('passed') is not True
           for name in ('teacher', 'parent') for case in case_ids):
        raise RuntimeError('C8 reference cache is missing a passed actual-infer parity receipt')
    rows = cache.get('rows')
    if not isinstance(rows, list) or [row.get('case_id') for row in rows] != case_ids:
        raise RuntimeError('C8 reference rows differ from the frozen case order')
    by_case = {}
    for row in rows:
        case, key = row['case_id'], str(row['case_id'])
        plan_row = row.get('plan_row', {})
        seed = plan_row.get('sampler_seeds', [None])[0]
        if (row.get('suite') not in SUITES or row.get('seed_index') != 0
                or row.get('sampler_seed') != seed or type(seed) is not int
                or plan_row.get('case_id') != case or plan_row.get('suite') != row['suite']):
            raise RuntimeError(f'C8 reference row identity differs at {case}')
        for name, records in (('teacher', cache['teacher']), ('parent', cache['parent'])):
            record = records[key]
            if (record.get('case_id') != case or record.get('suite') != row['suite']
                    or record.get('seed_index') != 0 or record.get('sampler_seed') != seed):
                raise RuntimeError(f'C8 {name} identity differs at {case}')
            _tensor(torch, record.get('action'), ACTION_SHAPE, f'{name} action {case}')
            _tensor(torch, record.get('future'), FUTURE_SHAPE, f'{name} future {case}')
        by_case[case] = row
    if cache.get('source_c8', {}).get('case_ids') != case_ids:
        raise RuntimeError('C8 source summary case IDs differ')

    calibration_rows = []
    for case in case_ids:
        key = str(case)
        action_mse = (cache['parent'][key]['action'][:10, :6].float()
                      - cache['teacher'][key]['action'][:10, :6].float()).square().mean()
        future_nmse = latent_reference.future_nmse_loss(
            torch, cache['parent'][key]['future'], cache['teacher'][key]['future'])
        calibration_rows.append({'case_id': case, 'seed_index': 0, 'split': 'calibration',
                                 'action_mse': float(action_mse), 'future_nmse': float(future_nmse)})
    mean_action = sum(row['action_mse'] for row in calibration_rows) / 8
    mean_future = sum(row['future_nmse'] for row in calibration_rows) / 8
    normalization_c = mean_action / mean_future if mean_future > 0 else float('nan')
    stored_c = cache.get('normalization', {}).get('normalization_c')
    if (not math.isfinite(normalization_c) or normalization_c <= 0
            or not isinstance(stored_c, (int, float))
            or not math.isclose(normalization_c, float(stored_c), rel_tol=1e-8, abs_tol=1e-12)):
        raise RuntimeError('C8 normalization_c does not recompute from the exact parent/teacher cache')
    if not math.isclose(float(receipt.get('normalization', {}).get('normalization_c', float('nan'))),
                        normalization_c, rel_tol=1e-8, abs_tol=1e-12):
        raise RuntimeError('Reference receipt normalization_c differs from the cache')
    return cache, by_case, float(normalization_c)


def _parity(torch, actual, expected) -> dict:
    delta = actual.detach().float().cpu() - expected.detach().float().cpu()
    return {'max_abs': float(delta.abs().max()), 'rmse': float(delta.square().mean().sqrt())}


def _source_c8(cache: dict) -> dict:
    source = cache['source_c8']
    return {'protocol': source['protocol'], 'cohort': source['cohort'],
            'source_job_id': source['source_job_id'], 'case_ids': list(source['case_ids']),
            'suite_case_counts': dict(source['suite_case_counts']),
            'bank_directory': source['bank_directory']}


def _training_receipt(binding, state, cache, *, status: str, complete: bool,
                      target_linears: int = 0, reference_action_counts=None) -> dict:
    report = state.get('gradient_coverage') or {'targets': target_linears, 'connected': 0,
                                                'finite': 0, 'nonzero': 0}
    return {'schema_version': 1, 'protocol': PROTOCOL, 'status': status, 'complete': complete,
            'binding': binding, 'reference_binding': binding['reference_binding'],
            'arm': binding['arm'], 'rho': binding['rho'],
            'epochs': binding['epochs'], 'updates': state.get('updates_completed', 0),
            'target_linears': target_linears, 'case_ids': binding['source_c8']['case_ids'],
            'source_c8': binding['source_c8'],
            'normalization_c': binding['normalization_c'], 'latent_weight': binding['latent_weight'],
            'optimizer_reset_after_update1': state.get('optimizer_reset_after_update1', False),
            'gradient_coverage': {'target_linears': report.get('targets', target_linears),
                                  'connected_linears': report.get('connected', 0),
                                  'finite_linears': report.get('finite', 0),
                                  'nonzero_gradient_linears': report.get('nonzero', 0)},
            'counts': {'reference_contexts': cache['context_count'],
                       'gradient_updates': state.get('updates_completed', 0),
                       'reference_action_sampler_forwards': (reference_action_counts or {}).get('sampler', 0),
                       'reference_action_pilot_infer_calls': (reference_action_counts or {}).get('pilot_infer', 0),
                       'teacher_forwards': 0, 'training_performed': state.get('updates_completed', 0) > 0},
            'checkpoint': state.get('checkpoint_path'),
            'epochs_completed': state.get('epochs_completed', 0),
            'next_update': state.get('updates_completed', 0) + 1}


def _validate_completed(torch, out: Path, receipt: dict, binding: dict,
                        protocol_binding: dict, total_updates: int, case_ids: list[int], seed: int) -> None:
    if (receipt.get('status') != 'complete' or receipt.get('updates') != total_updates
            or receipt.get('epochs') != binding['epochs'] or receipt.get('epochs_completed') != binding['epochs']
            or receipt.get('target_linears') != 614 or receipt.get('optimizer_reset_after_update1') is not True
            or receipt.get('source_c8') != binding['source_c8']
            or receipt.get('normalization_c') != binding['normalization_c']
            or receipt.get('latent_weight') != binding['latent_weight']):
        raise RuntimeError('Existing complete training receipt fails the frozen completion gate')
    deployment = torch.load(out / 'deployment.pt', map_location='cpu', weights_only=True)
    if (deployment.get('schema_version') != 1 or deployment.get('protocol') != PROTOCOL
            or deployment.get('arm') != binding['arm'] or deployment.get('rho') != binding['rho']
            or deployment.get('epochs') != binding['epochs'] or deployment.get('updates') != total_updates
            or deployment.get('target_linears') != 614 or deployment.get('case_ids') != case_ids
            or deployment.get('source_c8') != binding['source_c8']
            or deployment.get('reference_binding') != protocol_binding
            or deployment.get('normalization_c') != binding['normalization_c']
            or deployment.get('latent_weight') != binding['latent_weight']):
        raise RuntimeError('Existing deployment is incomplete or has a different C8 binding')
    masters = deployment.get('module_masters')
    if not isinstance(masters, dict) or len(masters) != 614:
        raise RuntimeError('Existing deployment does not cover all 614 master modules')
    for name, master in masters.items():
        books, scales = master.get('books'), master.get('log_scales')
        if (not torch.is_tensor(books) or tuple(books.shape) != (2, 256, 4)
                or books.dtype != torch.float32 or not bool(torch.isfinite(books).all())
                or not torch.is_tensor(scales) or scales.ndim != 1 or scales.dtype != torch.float32
                or not bool(torch.isfinite(scales).all())):
            raise RuntimeError(f'Existing deployment has invalid FP32 masters for {name}')
    action = json.loads((out / 'reference_action.json').read_text(encoding='utf-8'))
    values = action.get('action')
    tensor = torch.tensor(values, dtype=torch.float32)
    parity = action.get('actual_infer_parity', {})
    if (action.get('schema_version') != 1 or action.get('protocol') != PROTOCOL
            or action.get('arm') != binding['arm'] or action.get('rho') != binding['rho']
            or action.get('case_id') != case_ids[0] or action.get('seed_index') != 0
            or action.get('sampler_seed') != seed or tuple(tensor.shape) != ACTION_SHAPE
            or action.get('source_c8') != binding['source_c8']
            or action.get('counts', {}).get('target_linears') != 614
            or action.get('recipe', {}).get('normalization_c') != binding['normalization_c']
            or action.get('recipe', {}).get('latent_weight') != binding['latent_weight']
            or not bool(torch.isfinite(tensor).all())
            or not math.isfinite(float(parity.get('max_abs', float('nan'))))
            or float(parity['max_abs']) > 1e-6):
        raise RuntimeError('Existing reference action fails its same-input/same-seed parity gate')


def run(protocol_path: Path, reference: Path, arm: str, out: Path) -> int:
    host = _allocation_guard()
    import torch
    import numpy as np
    import calibration_source as source
    import sampler
    import runner as pilot
    import latent_reference
    import prepare_c8
    from train_banks import TrainableBanks
    from checkpointing import (atomic_torch_save, capture_rng, optimizer_to_device,
                               restore_rng, to_cpu)

    sampler._require_compute_allocation()
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    config = _validate_arm(protocol, arm)
    protocol_binding = _binding(protocol)
    calibration = protocol_binding['calibration']
    case_ids = calibration['case_ids']
    reference_cache, reference_rows, normalization_c = _load_reference(
        torch, reference, protocol_binding, case_ids, latent_reference)
    latent_weight = config['rho'] * normalization_c
    binding = {'reference_binding': protocol_binding, **config,
               'source_c8': _source_c8(reference_cache), 'case_ids': case_ids,
               'normalization_c': normalization_c, 'latent_weight': latent_weight}
    if set(binding['source_c8']['suite_case_counts'].values()) != {2}:
        raise RuntimeError('C8 reference must contain exactly two cases per suite')
    plan_path = Path(calibration['plan_file'])
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if (plan.get('protocol') != 'fastwam-vq-calibration-suites-v1'
            or plan.get('status') != 'complete'
            or plan.get('calibration_cohorts', {}).get('c8') != case_ids):
        raise RuntimeError('Current plan file no longer matches the frozen C8 reference')
    current_rows = {int(row['case_id']): row for row in plan.get('inputs', [])}
    reference_rows_by_case = {int(row['case_id']): row['plan_row'] for row in reference_cache['rows']}
    if any(current_rows.get(case) != reference_rows_by_case.get(case) for case in case_ids):
        raise RuntimeError('Current C8 plan rows changed after reference preparation')
    data_root = plan_path.parent

    out.mkdir(parents=True, exist_ok=True)
    receipt_path, checkpoint_path = out / 'training_receipt.json', out / 'checkpoint.pt'
    existing_receipt = None
    if receipt_path.exists():
        existing_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if existing_receipt.get('binding') != binding:
            raise RuntimeError('Existing training receipt has a different frozen binding')
        if existing_receipt.get('complete') is True:
            if (not checkpoint_path.is_file() or not (out / 'deployment.pt').is_file()
                    or not (out / 'reference_action.json').is_file()):
                raise RuntimeError('Existing complete receipt is missing final outputs')
            _validate_completed(torch, out, existing_receipt, binding, protocol_binding,
                                config['epochs'] * len(case_ids), case_ids,
                                current_rows[case_ids[0]]['sampler_seeds'][0])
            print('C8_TRAINING_REUSED ' + json.dumps({'arm': arm, 'updates': existing_receipt['updates'],
                                                      'new_model_forwards': 0}), flush=True)
            return 0
        if existing_receipt.get('status') != 'checkpointed' or not checkpoint_path.is_file():
            raise RuntimeError('Existing incomplete training output has no resumable checkpoint')
    elif checkpoint_path.exists():
        # Recover a crash between the atomic checkpoint write and its JSON receipt.
        pass
    for temporary in (checkpoint_path.with_suffix('.pt.tmp'),):
        if temporary.exists():
            raise FileExistsError('An incomplete checkpoint write exists; refusing to replace it')

    started = time.monotonic()
    job_id = os.environ['PBS_JOBID']
    interrupted = []
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupted.append(signum))
    allocation_start_raw = os.environ.get('FW_JOB_START_EPOCH')
    if allocation_start_raw is None:
        raise RuntimeError('FW_JOB_START_EPOCH must be shared by every PBS phase')
    allocation_start = float(allocation_start_raw)
    soft_seconds = float(os.environ.get(
        'FW_SOFT_DEADLINE_SECONDS', protocol.get('training', {}).get(
            'soft_deadline_seconds', protocol.get('soft_deadline_seconds', 84_600))))
    save_reserve = float(protocol.get('training', {}).get('save_reserve_seconds', 300))
    if (not math.isfinite(allocation_start) or not math.isfinite(soft_seconds) or soft_seconds <= 0
            or not math.isfinite(save_reserve) or save_reserve < 0):
        raise ValueError('Invalid PBS soft-deadline or checkpoint reserve')
    deadline = allocation_start + soft_seconds
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = None
    try:
        runtime = pilot.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        banks = TrainableBanks(runtime, calibration['bank_directory'])
        parameters = banks.parameters()
        optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])
        total_updates = config['epochs'] * len(case_ids)
        contract = {'binding': binding, 'reference_receipt': str(reference / 'reference_receipt.json'),
                    'reference_cache_schema': reference_cache['schema_version'],
                    'source_job_id': calibration['source_job_id'], 'case_ids': case_ids}

        payload = None
        if checkpoint_path.is_file():
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if payload.get('schema_version') != 1 or payload.get('contract') != contract:
                raise RuntimeError('Existing checkpoint has a different training/reference binding')
            state = payload['run_state']
            expected_cursor = (state['updates_completed'] // len(case_ids),
                               state['updates_completed'] % len(case_ids))
            if (state.get('updates_completed', -1) < 0 or state['updates_completed'] > total_updates
                    or state.get('cursor') != list(expected_cursor)
                    or state.get('optimizer_reset_after_update1') != (state['updates_completed'] >= 1)):
                raise RuntimeError('Checkpoint update cursor or Adam reset state is invalid')
            banks.restore(payload['module_masters'])
            optimizer.load_state_dict(payload['optimizer_state'])
            optimizer_to_device(torch, optimizer, 'cuda')
            restore_rng(torch, payload['rng_state'])
            if existing_receipt is None:
                _save_json(receipt_path, _training_receipt(binding, state, reference_cache,
                                                           status='checkpointed', complete=False,
                                                           target_linears=614))
        else:
            if existing_receipt is not None:
                raise RuntimeError('Checkpointed receipt is missing its checkpoint')
            state = {'updates_completed': 0, 'epochs_completed': 0, 'cursor': [0, 0],
                     'optimizer_reset_after_update1': False, 'gradient_coverage': None,
                     'training_history': [], 'operation_seconds': [], 'parity': None,
                     'checkpoint_path': str(checkpoint_path), 'created_job_id': job_id,
                     'resume_lineage': []}

        state['resume_lineage'].append({'job_id': job_id,
                                        'resume_from': 'checkpoint.pt' if payload is not None else None})
        del payload
        data_by_case = {}
        plan_rows = {int(row['case_id']): row['plan_row'] for row in reference_cache['rows']}
        cached_metadata = {int(row['case_id']): row['metadata'] for row in reference_cache['rows']}

        def datum(case: int):
            if case not in data_by_case:
                row = plan_rows[case]
                observation, metadata = source.load_observation(np, data_root, row)
                if metadata != cached_metadata[case]:
                    raise RuntimeError(f'C8 metadata changed since reference preparation: {case}')
                data_by_case[case] = (runtime.datum(observation), metadata)
            return data_by_case[case]

        def persist(status='checkpointed', complete=False):
            snapshot = {'schema_version': 1, 'contract': contract,
                        'module_masters': banks.snapshot(),
                        'optimizer_state': to_cpu(optimizer.state_dict()),
                        'rng_state': capture_rng(torch),
                        'run_state': {**state, 'last_job_id': job_id,
                                      'cumulative_wall_seconds': state.get('cumulative_wall_seconds', 0.0)
                                      + time.monotonic() - started}}
            atomic_torch_save(torch, checkpoint_path, snapshot)
            state['checkpoint_path'] = str(checkpoint_path)
            _save_json(receipt_path, _training_receipt(binding, state, reference_cache,
                                                       status=status, complete=complete,
                                                       target_linears=614))

        def should_pause(finalize=False):
            durations = state['operation_seconds'][-4:]
            estimate = max(durations) * 1.2 if durations else float(
                protocol.get('training', {}).get('first_update_estimate_seconds', 1800))
            if finalize:
                estimate = min(estimate, float(protocol.get('training', {}).get(
                    'reference_action_estimate_seconds', 300)))
            if interrupted or time.time() + estimate + save_reserve >= deadline:
                persist(status='checkpointed', complete=False)
                print('C8_TRAINING_CHECKPOINTED ' + json.dumps({
                    'arm': arm, 'updates': state['updates_completed'],
                    'epochs_completed': state['epochs_completed'], 'cursor': state['cursor'],
                    'checkpoint': str(checkpoint_path),
                    'reason': 'termination signal' if interrupted else 'PBS soft deadline'}), flush=True)
                return True
            return False

        if state['updates_completed'] == 0 and state['parity'] is None:
            if should_pause():
                return 75
            case = case_ids[0]
            value, metadata = datum(case)
            runtime.description, runtime.current_arm = metadata['description'], arm
            banks.seen.clear()
            action, counts, future = sampler.differentiable_action(
                runtime, value, plan_rows[case]['sampler_seeds'][0],
                checkpointed=True, return_future=True)
            if len(banks.seen) != 614:
                raise RuntimeError(f'Initial gradient forward reached only {len(banks.seen)}/614 Linears')
            parent = reference_cache['parent'][str(case)]
            action_parity = _parity(torch, action, parent['action'])
            future_parity = _parity(torch, future, parent['future'])
            parity = {'case_id': case, 'action': action_parity, 'future': future_parity,
                      'passed': (action_parity['max_abs'] <= 0.002 and action_parity['rmse'] <= 0.001
                                 and future_parity['max_abs'] <= 0.002 and future_parity['rmse'] <= 0.001),
                      **counts}
            if not parity['passed']:
                raise RuntimeError(f'Checkpointed C8 parent forward parity failed: {parity}')
            state['parity'] = parity
            del action, future

        while state['updates_completed'] < total_updates:
            if should_pause():
                return 75
            update_index = state['updates_completed']
            epoch_index, case_index = divmod(update_index, len(case_ids))
            if state['cursor'] != [epoch_index, case_index]:
                raise RuntimeError('Training cursor no longer matches the frozen case/epoch order')
            case = case_ids[case_index]
            value, metadata = datum(case)
            runtime.description, runtime.current_arm = metadata['description'], arm
            banks.seen.clear()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            action, counts, future = sampler.differentiable_action(
                runtime, value, plan_rows[case]['sampler_seeds'][0],
                checkpointed=True, return_future=True)
            if len(banks.seen) != 614:
                raise RuntimeError(f'Gradient forward reached only {len(banks.seen)}/614 Linears')
            if not future.requires_grad:
                raise RuntimeError('Student future is disconnected from the full gradient path')
            teacher = reference_cache['teacher'][str(case)]
            teacher_action = teacher['action'].to(device='cuda', dtype=torch.float32)
            teacher_future = teacher['future'].to(device='cuda')
            action_mse = (action[:10, :6].float() - teacher_action[:10, :6]).square().mean()
            future_nmse = latent_reference.future_nmse_loss(torch, future, teacher_future)
            scaled_future = latent_weight * future_nmse
            total_loss = action_mse + scaled_future
            if (not bool(torch.isfinite(total_loss)) or not bool(torch.isfinite(action_mse))
                    or not bool(torch.isfinite(future_nmse)) or not future_nmse.requires_grad):
                raise RuntimeError('Action+future loss is nonfinite or disconnected')
            values = {'action_mse': float(action_mse.detach()),
                      'future_nmse': float(future_nmse.detach()),
                      'scaled_future_loss': float(scaled_future.detach()),
                      'total_loss': float(total_loss.detach()), 'latent_weight': latent_weight}
            total_loss.backward()
            report = banks.gradient_report()
            if report['connected'] != 614 or report['finite'] != 614 or report['nonzero'] < 1:
                _save_json(out / 'gradient_coverage_failure.json', report)
                raise RuntimeError('Gradient coverage must connect and finitize all 614 Linears')
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, config['gradient_norm_clip'], error_if_nonfinite=True)
            optimizer.step()
            reset_now = latent_reference.reset_adam_after_update1(optimizer, state)
            banks.refresh()
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            state['operation_seconds'].append(duration)
            state['gradient_coverage'] = report
            state['training_history'].append({
                'update': update_index + 1, 'epoch': epoch_index + 1,
                'case_id': case, **values, 'gradient_norm_before_clip': float(gradient_norm),
                'gradient_seconds': duration, 'optimizer_reset_performed_this_update': reset_now,
                'optimizer_reset_after_update1': state['optimizer_reset_after_update1'],
                **counts, 'connected_linears': report['connected'],
                'finite_linears': report['finite'], 'nonzero_gradient_linears': report['nonzero']})
            state['updates_completed'] += 1
            state['epochs_completed'] = state['updates_completed'] // len(case_ids)
            state['cursor'] = [state['updates_completed'] // len(case_ids),
                               state['updates_completed'] % len(case_ids)]
            del action, future, teacher_action, teacher_future, action_mse, future_nmse, scaled_future, total_loss
            persist(status='checkpointed', complete=False)
            print('C8_TRAIN_UPDATE ' + json.dumps(state['training_history'][-1]), flush=True)

        if state['updates_completed'] != total_updates or not state['optimizer_reset_after_update1']:
            raise RuntimeError('Training cannot complete before the exact frozen update/reset contract')
        if should_pause(finalize=True):
            return 75

        first_case = case_ids[0]
        value, metadata = datum(first_case)
        runtime.description, runtime.current_arm = metadata['description'], arm
        banks.seen.clear()
        reference_action, counts, _ = sampler.differentiable_action(
            runtime, value, plan_rows[first_case]['sampler_seeds'][0],
            checkpointed=False, return_future=False)
        if len(banks.seen) != 614:
            raise RuntimeError(f'Reference action reached only {len(banks.seen)}/614 Linears')
        actual_action, _ = runtime.infer(value, plan_rows[first_case]['sampler_seeds'][0], measure=False)
        parity = _parity(torch, actual_action, reference_action)
        if parity['max_abs'] > 1e-6:
            raise RuntimeError(f'Trained sampler/pilot.infer reference parity exceeds 1e-6: {parity}')
        _tensor(torch, reference_action, ACTION_SHAPE, 'trained C8 reference action')

        source_c8 = binding['source_c8']
        fixed = {'books': 'FP32 master; BF16 deployed',
                 'row_scales': 'FP32 log master; BF16 deployed',
                 'indices': 'frozen uint8', 'scale_exponents': 'frozen uint8; scale_block=32',
                 'transform': {'name': 'smooth05_hadamard', 'alpha': 0.5, 'hadamard_block': 128},
                 'activation': {'name': 'dynamic A4', 'qmax': 7, 'row_scale': 'absmax'},
                 'biases': 'frozen', 'non_target_model_parameters': 'frozen'}
        deployment = {'schema_version': 1, 'protocol': PROTOCOL, 'arm': arm,
                      'rho': config['rho'], 'epochs': config['epochs'], 'updates': total_updates,
                      'target_linears': 614, 'case_ids': case_ids, 'source_c8': source_c8,
                      'normalization_c': normalization_c, 'latent_weight': latent_weight,
                      'module_masters': banks.snapshot(), 'fixed': fixed,
                      'reference_binding': protocol_binding}
        _atomic_torch_save(torch, out / 'deployment.pt', deployment)
        reference_action_payload = {
            'schema_version': 1, 'protocol': PROTOCOL, 'arm': arm, 'rho': config['rho'],
            'case_id': first_case, 'seed_index': 0,
            'sampler_seed': plan_rows[first_case]['sampler_seeds'][0],
            'action': reference_action.detach().float().cpu().tolist(),
            'counts': {'target_linears': 614, **counts},
            'actual_infer_parity': parity,
            'recipe': {'loss': 'mean((action[:10,:6]-teacher[:10,:6])^2) + latent_weight * future_nmse',
                       'normalization_c': normalization_c, 'latent_weight': latent_weight,
                       'future_nmse': 'global NMSE over clean generated frames 1..T-1; clamped frame 0 excluded',
                       'epochs': config['epochs'], 'updates': total_updates,
                       'optimizer_reset_after_update1': True},
            'source_c8': source_c8, 'plan_row': plan_rows[first_case], 'metadata': metadata}
        _save_json(out / 'reference_action.json', reference_action_payload)
        receipt = _training_receipt(binding, state, reference_cache, status='complete', complete=True,
                                    target_linears=614,
                                    reference_action_counts={'sampler': 1, 'pilot_infer': 1})
        receipt.update({'updates': total_updates, 'epochs_completed': config['epochs'],
                        'target_linears': 614, 'gradient_coverage': {
                            'target_linears': 614, 'connected_linears': state['gradient_coverage']['connected'],
                            'finite_linears': state['gradient_coverage']['finite'],
                            'nonzero_gradient_linears': state['gradient_coverage']['nonzero']},
                        'counts': {'reference_contexts': 8, 'gradient_updates': total_updates,
                                   'reference_action_sampler_forwards': 1,
                                   'reference_action_pilot_infer_calls': 1,
                                   'teacher_forwards': 0, 'training_performed': True},
                        'last_job_id': job_id, 'resume_lineage': state['resume_lineage'],
                        'checkpoint': str(checkpoint_path),
                        'deployment': str(out / 'deployment.pt'),
                        'reference_action': str(out / 'reference_action.json'),
                        'reference_action_parity': parity,
                        'wall_seconds': time.monotonic() - started,
                        'cumulative_wall_seconds': state.get('cumulative_wall_seconds', 0.0)
                        + time.monotonic() - started})
        _save_json(receipt_path, receipt)
        print('C8_TRAINING_COMPLETE ' + json.dumps(receipt), flush=True)
        return 0
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.protocol, args.reference, args.arm, args.out))
