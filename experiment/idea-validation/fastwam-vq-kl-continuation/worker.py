"""Continue the frozen C80 KL arms from one completed per-channel checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80


ARMS = ('mse_nmse', 'action_kl', 'latent_kl', 'dual_kl')
SNAPSHOTS = (0, 20, 40, 60, 80)


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('Required file is missing: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def objective(values, unit, p, scale):
    action = (scale['action_kl_coefficient'] * values['action_kl']
              if unit in ('action_kl', 'dual_kl') else values['action_mse'])
    latent = (scale['latent_kl_coefficient'] * values['latent_kl']
              if unit in ('latent_kl', 'dual_kl') else values['latent_nmse'])
    return action + p['training']['rho'] * p['training']['normalization_c'] * latent


def record_values(record):
    return {'action_mse': record['first10motor_mse'],
            'latent_nmse': record['future_latent_nmse'],
            'action_kl': record['action_kl'], 'latent_kl': record['latent_kl']}


def validate_config(cfg, decision):
    required = ('protocol', 'remote_root', 'source_root', 'start_checkpoint', 'start_job',
                'updates', 'learning_rate_factor', 'snapshots', 'arms', 'decision_file')
    if any(key not in cfg for key in required):
        raise RuntimeError('Continuation protocol is missing a required field')
    if (not isinstance(cfg['protocol'], str) or not cfg['protocol'] or cfg['updates'] != 80
            or cfg['snapshots'] != [20, 40, 60, 80] or set(cfg['arms']) != set(ARMS)
            or len(cfg['arms']) != len(ARMS)):
        raise RuntimeError('Continuation protocol scope differs from the frozen contract')
    factor = cfg['learning_rate_factor']
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
        raise RuntimeError('A finite positive C80-selected learning-rate factor is required')
    if (decision.get('complete') is not True or decision.get('heldout_used') is not False
            or decision.get('sampling_scope') != 'C80'):
        raise RuntimeError('The learning-rate decision must be complete and C80-only')
    chosen = decision.get('learning_rate_factor')
    if (isinstance(chosen, bool) or not isinstance(chosen, (int, float)) or not math.isfinite(chosen)
            or not math.isclose(float(chosen), float(factor), rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError('Protocol learning-rate factor differs from its C80 decision file')


def run(cfg):
    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from channel_banks import ChannelBanks
    from softbin_loss import losses
    from metrics_reference import metrics
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, optimizer_to_device, to_cpu

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)

    root = Path(cfg['remote_root'])
    source_root = Path(cfg['source_root'])
    old = read_json(source_root / 'protocol.json')
    if old.get('protocol') != 'fastwam-vq-dual-kl-v1' or old.get('remote_root') != str(source_root):
        raise RuntimeError('source_root must contain the completed dual-KL protocol')
    decision = read_json(cfg['decision_file'])
    validate_config(cfg, decision)
    source_root_checkpoint = Path(old['source_per_channel_root']) / 'results/train_eval/checkpoint.pt'
    if Path(cfg['start_checkpoint']) != source_root_checkpoint:
        raise RuntimeError('start_checkpoint must be the original per-channel final80 checkpoint')
    if old['source_per_channel_job'] != cfg['start_job']:
        raise RuntimeError('Warm-start PBS job differs from the old dual-KL protocol')
    if (old['training']['additional_updates'] != 80
            or not old['training']['optimizer_reset_after_update1']
            or not math.isclose(old['training']['clip_learning_rate'], 0.01, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(old['scaling']['learning_rate'], 0.02, rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError('Original update, optimizer-reset, or learning-rate contract changed')
    if old['calibration']['case_ids'] != list(range(2000, 2080)) or len(set(old['calibration']['update_order'])) != 80:
        raise RuntimeError('Original C80 calibration cohort or update order changed')
    if old['evaluation']['case_ids'] != list(range(3000, 3040)) or old['evaluation']['seed_indices'] != [0, 1]:
        raise RuntimeError('Original paired held-out cohort changed')

    artifact = Path(os.environ['ARTIFACTS'])
    source_protocol = read_json(artifact / 'source_protocol.json')
    if read_json(Path(old['source_per_channel_root']) / 'protocol.json') != source_protocol:
        raise RuntimeError('Original per-channel protocol differs from the old artifact')
    for key in ('training', 'calibration', 'evaluation', 'clipping', 'scaling'):
        if key == 'evaluation':
            if any(old[key][k] != source_protocol[key][k] for k in ('case_ids', 'seed_indices', 'aggregation')):
                raise RuntimeError('Original held-out definition differs')
        elif old[key] != source_protocol[key]:
            raise RuntimeError('Old dual-KL frozen protocol differs from its per-channel source: ' + key)

    gradient_check = read_json(artifact / 'channel_selfcheck.json')
    kl_check = read_json(artifact / 'softbin_selfcheck.json')
    if (not all(gradient_check.get(k) is True for k in
                ('passed', 'independent_reference_passed', 'old_surrogate_rejected'))
            or kl_check.get('passed') is not True):
        raise RuntimeError('Independent channel-gradient and soft-bin checks are required')
    if any(v.get('pbs_jobid') != os.environ['PBS_JOBID'] for v in (gradient_check, kl_check)):
        raise RuntimeError('Self-check results must come from this PBS allocation')

    stats = read_json(source_root / 'results/fit_stats/summary.json')
    if (stats.get('complete') is not True or stats.get('heldout_used') is not False
            or stats.get('calibration_case_ids') != old['calibration']['case_ids']):
        raise RuntimeError('Frozen calibration-only soft-bin statistics are required')
    scale = read_json(source_root / 'results/prepare/loss_scaling.json')
    if (scale.get('complete') is not True or scale.get('heldout_used') is not False
            or scale.get('case_ids') != old['calibration']['case_ids']):
        raise RuntimeError('Original calibration-only loss coefficients are required')
    if not math.isclose(scale['original_latent_weight'],
                        old['training']['rho'] * old['training']['normalization_c'],
                        rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError('Original latent loss coefficient changed')

    source_receipt = read_json(Path(old['source_per_channel_root']) / 'results/train_eval/training_receipt.json')
    start = torch.load(cfg['start_checkpoint'], map_location='cpu', weights_only=True)
    expected_order = old['calibration']['update_order']
    if (source_receipt.get('complete') is not True or source_receipt.get('updates') != 80
            or source_receipt.get('pbs_jobid') != cfg['start_job']
            or source_receipt.get('update_order') != expected_order
            or start.get('updates') != 80
            or [h['case_id'] for h in start.get('history', [])] != expected_order
            or start.get('protocol') != source_protocol):
        raise RuntimeError('The exact completed source80 checkpoint and interleaved C80 history are required')
    start_ratios, start_scales = start['ratios'], start['shadow_log2_scales']
    del start

    teacher_dirs = {'calibration': Path(old['control_root']) / 'results/teacher/calibration',
                    'heldout': Path(old['clipping_root']) / 'results/teacher'}
    for split, directory in teacher_dirs.items():
        receipt = read_json(directory / 'receipt.json')
        expected = old['calibration' if split == 'calibration' else 'evaluation']['case_ids']
        seeds = [0] if split == 'calibration' else [0, 1]
        if (receipt.get('complete') is not True or receipt.get('queries') != 80
                or receipt.get('case_ids') != expected or receipt.get('seed_indices') != seeds):
            raise RuntimeError('The shared teacher coverage differs for ' + split)

    data_root, rows, cache, deployment, reference_action = ref.load_inputs(old['source_clipping_protocol'], torch)
    unit = cfg['selected_unit']
    if unit not in ARMS:
        raise RuntimeError('Selected arm is outside the four frozen loss objectives')
    out = root / 'results' / unit
    out.mkdir(parents=True, exist_ok=True)
    previous_progress = read_json(out / 'progress.json') if (out / 'progress.json').is_file() else {}
    state = {'protocol': cfg['protocol'], 'unit': unit,
             'pbs_jobid': os.environ['PBS_JOBID'], 'warm_start_job': cfg['start_job'],
             'complete': False, 'phase': 'loading', 'base_updates': 80,
             'additional_updates': 0, 'total_updates': 80,
             'new_teacher_queries': 0, 'new_model_queries': 0}

    def progress(**values):
        state.update(values)
        ref.save_json(out / 'progress.json', state)
        print('KL_CONTINUATION_PROGRESS ' + json.dumps(state), flush=True)

    def count_model_query(purpose):
        state['new_model_queries'] = int(state.get('new_model_queries', 0)) + 1
        key = purpose + '_query_count'
        state[key] = int(state.get(key, 0)) + 1
        ref.save_json(out / 'progress.json', state)

    runtime = None
    deadline = ref.Deadline()
    progress()
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        data = {}
        banks = ChannelBanks(runtime, old['calibration']['bank_directory'], deployment['module_masters'],
                             old['clipping'], old['scaling'])
        if (len(banks.ratios) != 614
                or sum(value.numel() for value in banks.shadows.values()) != 1_806_863):
            raise RuntimeError('The frozen 614-ratio and 1,806,863-channel parameter inventory changed')

        def datum(case):
            if case not in data:
                obs, meta = calibration_source.load_observation(np, data_root, rows[case])
                data[case] = runtime.datum(obs), meta
            return data[case]

        def query(case, seed, *, purpose='model'):
            deadline.check(90)
            value, meta = datum(case)
            runtime.description = meta['description']
            banks.seen.clear()
            result = ref.query(runtime, sampler, torch, value, meta['description'],
                               rows[case]['sampler_seeds'][seed], banks)
            count_model_query(purpose)
            return result

        def teacher(split, case, seed):
            saved = torch.load(teacher_dirs[split] / f'{case}_{seed}.pt',
                               map_location='cpu', weights_only=True)
            if saved['plan_row'] != rows[case] or saved['sampler_seed'] != rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Shared teacher binding differs')
            return saved

        def terms(action, future, target):
            return losses(torch, action, future, target['action'].to(action.device),
                          target['future'].to(future.device), stats, old['softbin'])

        def saved_objective(record, selected_unit):
            values = {key: torch.tensor(value, dtype=torch.float32)
                      for key, value in record_values(record).items()}
            return float(objective(values, selected_unit, old, scale))

        def make_record(unit, split, case, seed, updates, action, future, target, counts):
            values = terms(action, future, target)
            measured = metrics(torch, latent_reference, action, future, target, old['training'])
            return {'protocol': cfg['protocol'], 'unit': unit, 'arm': unit, 'split': split,
                    'updates': updates, 'snapshot_step': updates if split == 'calibration' else None,
                    'base_updates': 80, 'additional_updates': updates, 'total_updates': 80 + updates,
                    'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                    'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed],
                    'pbs_jobid': os.environ['PBS_JOBID'], 'warm_start_job': cfg['start_job'],
                    'counts': counts, **measured,
                    'action_kl': float(values['action_kl'].detach()),
                    'latent_kl': float(values['latent_kl'].detach()),
                    'objective_loss': float(objective(values, unit, old, scale).detach()),
                    'action': action.float().tolist(),
                    'commands': runtime.command(action.clone()).tolist(),
                    'teacher_action': target['action'].float().tolist(),
                    'teacher_commands': runtime.command(target['action'].clone()).tolist()}

        def validate_new_record(record, unit, step, split, case, seed):
            expected = {'protocol': cfg['protocol'], 'unit': unit, 'arm': unit, 'split': split,
                        'updates': step, 'snapshot_step': step if split == 'calibration' else None,
                        'plan_row': rows[case], 'case_id': case, 'seed_index': seed,
                        'sampler_seed': rows[case]['sampler_seeds'][seed]}
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Saved continuation endpoint binding differs')
            expected_objective = saved_objective(record, unit)
            if record.get('objective_loss') != expected_objective:
                raise RuntimeError('Saved endpoint objective differs from the selected arm formula')

        def snapshot_dir(unit, step):
            return out / 'snapshots' / f'step_{step:03d}' / 'queries'

        def read_snapshot(unit, step):
            records = []
            for case, seed in c80.queries(old, 'calibration'):
                record = read_json(snapshot_dir(unit, step) / f'{case}_{seed}.json')
                validate_new_record(record, unit, step, 'calibration', case, seed)
                records.append(record)
            if len(records) != 80:
                raise RuntimeError('C80 snapshot must contain exactly 80 records')
            return records

        def write_snapshot(unit, step, *, reuse_source=False):
            query_dir = snapshot_dir(unit, step)
            records = []
            queries = c80.queries(old, 'calibration')
            for index, (case, seed) in enumerate(queries, 1):
                path = query_dir / f'{case}_{seed}.json'
                if path.is_file():
                    record = read_json(path)
                    validate_new_record(record, unit, step, 'calibration', case, seed)
                elif reuse_source:
                    source_path = source_root / 'results/calibration/mse_nmse/queries' / f'{case}_{seed}.json'
                    original = read_json(source_path)
                    expected = {'protocol': old['protocol'], 'arm': 'mse_nmse', 'split': 'calibration',
                                'updates': 80, 'plan_row': rows[case], 'case_id': case,
                                'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed]}
                    if any(original.get(key) != value for key, value in expected.items()):
                        raise RuntimeError('Source MSE endpoint binding differs')
                    record = dict(original)
                    values = record_values(original)
                    record.update({'protocol': cfg['protocol'], 'unit': unit, 'arm': unit,
                                   'updates': 0, 'snapshot_step': 0, 'base_updates': 80,
                                   'additional_updates': 0, 'total_updates': 80,
                                   'warm_start_job': cfg['start_job'],
                                   'source_pbs_jobid': original.get('pbs_jobid'),
                                   'source_objective_loss': original.get('objective_loss'),
                                   'pbs_jobid': os.environ['PBS_JOBID'],
                                   'objective_loss': saved_objective(original, unit)})
                    if not all(key in record for key in ('action', 'commands', 'teacher_action',
                                                         'teacher_commands', 'plan_row', 'counts')):
                        raise RuntimeError('Source MSE endpoint is missing arrays or provenance')
                    ref.save_json(path, record)
                else:
                    target = teacher('calibration', case, seed)
                    action, future, counts = query(case, seed, purpose='snapshot')
                    record = make_record(unit, 'calibration', case, seed, step,
                                         action, future, target, counts)
                    ref.save_json(path, record)
                records.append(record)
                progress(phase='snapshot_' + str(step), updates=step,
                         snapshot_step=step, snapshot_queries=index,
                         snapshot_total_queries=len(queries), last_case=case)
            if len(records) != 80:
                raise RuntimeError('Every C80 snapshot must contain exactly 80 records')
            aggregate = c80.aggregate_records(records, old, 'calibration')
            for metric in ('action_kl', 'latent_kl', 'objective_loss'):
                aggregate['overall'][metric] = sum(r[metric] for r in records) / 80
                for suite in ref.SUITES:
                    rr = [r for r in records if r['suite'] == suite]
                    aggregate['by_suite'][suite][metric] = sum(r[metric] for r in rr) / len(rr)
            ref.save_json(query_dir.parent / 'summary.json',
                          {'protocol': cfg['protocol'], 'complete': True, 'unit': unit,
                           'split': 'calibration', 'snapshot_step': step, 'base_updates': 80,
                           'additional_updates': step, 'total_updates': 80 + step,
                           'pbs_jobid': os.environ['PBS_JOBID'], 'source_step0_reused': reuse_source,
                           **aggregate})
            return records

        def endpoint_path(split, case, seed):
            return root / 'results' / split / unit / 'queries' / f'{case}_{seed}.json'

        def copy_final_calibration(records):
            for record in records:
                path = endpoint_path('calibration', record['case_id'], record['seed_index'])
                if path.is_file():
                    existing = read_json(path)
                    validate_new_record(existing, unit, 80, 'calibration',
                                        record['case_id'], record['seed_index'])
                    if existing != record:
                        raise RuntimeError('Final calibration endpoint differs from snapshot80')
                else:
                    ref.save_json(path, record)

        work_training = dict(old['training'])
        work_scaling = dict(old['scaling'])
        factor = float(cfg['learning_rate_factor'])
        work_training['clip_learning_rate'] *= factor
        work_scaling['learning_rate'] *= factor
        groups = banks.optimizer_groups(work_training, work_scaling)
        parameters = [v for group in groups for v in group['params']]
        if {id(v) for v in parameters} != {id(v) for v in list(banks.ratios.values()) + list(banks.shadows.values())}:
            raise RuntimeError('Only original ratios and per-channel log2 scales may train')
        if any(v.requires_grad for v in runtime.model.parameters()) or any(
                v.requires_grad for v in list(banks.books.values()) + list(banks.logs.values())):
            raise RuntimeError('Weights, books, indices and row gains must remain frozen')

        unit = state['unit']
        out = root / 'results' / unit
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out / 'checkpoint.pt'
        optimizer = torch.optim.Adam(groups)
        updates, history, initial_probe = 0, [], None
        if checkpoint_path.is_file():
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if (saved.get('protocol') != cfg or saved.get('unit') != unit
                    or saved.get('source_checkpoint') != str(source_root_checkpoint)
                    or saved.get('learning_rate_factor') != factor
                    or saved.get('learning_rates') != {
                        'clip': work_training['clip_learning_rate'],
                        'channel': work_scaling['learning_rate']}
                    or saved.get('source_optimizer_discarded') is not True):
                raise RuntimeError('Continuation resume contract differs')
            updates, history = saved['updates'], saved['history']
            if [h['case_id'] for h in history] != expected_order[:updates]:
                raise RuntimeError('Continuation update order changed')
            banks.restore_clips(saved['ratios'])
            banks.restore_scales(saved['shadow_log2_scales'])
            optimizer.load_state_dict(saved['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda')
            restore_rng(torch, saved['rng'])
            initial_probe = saved.get('initial_probe')
            state.update(saved.get('counts', {}))
            if (previous_progress.get('protocol') == cfg['protocol']
                    and previous_progress.get('unit') == unit):
                for key, value in previous_progress.items():
                    if (key.endswith('_count') or key.endswith('_queries')) and isinstance(value, int):
                        state[key] = max(int(state.get(key, 0)), value)
            if saved.get('first_train_forward_parity') is not None:
                state['first_train_forward_parity'] = saved['first_train_forward_parity']
        else:
            banks.restore_clips(start_ratios)
            banks.restore_scales(start_scales)
            initial_probe = None
        if updates < 0 or updates > 80:
            raise RuntimeError('Invalid continuation update count')

        def save_checkpoint():
            atomic_torch_save(torch, checkpoint_path, {
                'protocol': cfg, 'unit': unit, 'source_checkpoint': str(source_root_checkpoint),
                'warm_start_job': cfg['start_job'], 'source_optimizer_discarded': True,
                'learning_rate_factor': factor, 'learning_rates': {
                    'clip': work_training['clip_learning_rate'],
                    'channel': work_scaling['learning_rate']},
                'updates': updates, 'history': history, 'ratios': banks.clip_snapshot(),
                'shadow_log2_scales': banks.scale_snapshot(),
                'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
                'initial_probe': to_cpu(initial_probe) if initial_probe is not None else None,
                'first_train_forward_parity': state.get('first_train_forward_parity'),
                'counts': {k: v for k, v in state.items() if k.endswith('_count') or k.endswith('_queries')},
            })

        def ensure_initial_probe():
            nonlocal initial_probe
            if initial_probe is None:
                case = old['calibration']['case_ids'][0]
                progress(phase='source_endpoint_check', updates=updates, case_id=case)
                initial = query(case, 0, purpose='initial_source_endpoint')
                target = teacher('calibration', case, 0)
                values = terms(initial[0], initial[1], target)
                measured = metrics(torch, latent_reference, initial[0], initial[1], target, old['training'])
                source_endpoint = read_json(source_root / 'results/calibration/mse_nmse/queries' / f'{case}_0.json')
                expected_source = {'protocol': old['protocol'], 'arm': 'mse_nmse', 'split': 'calibration',
                                   'updates': 80, 'plan_row': rows[case], 'case_id': case,
                                   'seed_index': 0, 'sampler_seed': rows[case]['sampler_seeds'][0]}
                if any(source_endpoint.get(k) != v for k, v in expected_source.items()):
                    raise RuntimeError('Initial source MSE endpoint binding differs')
                if any(measured[key] != source_endpoint[key] for key in c80.METRICS):
                    raise RuntimeError('Warm-start no-grad case 2000 differs from source MSE endpoint metrics')
                source_scalar = saved_objective(source_endpoint, 'mse_nmse')
                observed_scalar = float(objective(values, 'mse_nmse', old, scale))
                if observed_scalar != source_scalar or source_endpoint.get('objective_loss') != source_scalar:
                    raise RuntimeError('Warm-start no-grad case 2000 objective differs from source MSE endpoint')
                initial_probe = {'case_id': case, 'seed_index': 0,
                                 'action': initial[0].detach().cpu(), 'future': initial[1].detach().cpu(),
                                 'source_mse_objective': source_scalar}
                state['initial_source_endpoint_queries'] = 1
                state['source_endpoint_objective_max_abs'] = abs(observed_scalar - source_scalar)
                save_checkpoint()
                progress(phase='source_endpoint_checked', updates=updates,
                         source_endpoint_objective_max_abs=0.0)

        def ensure_after_cpu():
            if not history or history[-1].get('after_cpu_objective') is not None:
                return
            row = history[-1]
            case, seed = row['case_id'], row['seed_index']
            progress(phase='step_after_cpu', updates=updates, case_id=case)
            target = teacher('calibration', case, seed)
            after_action, after_future, after_counts = query(case, seed, purpose='step_after_cpu')
            after_values = terms(after_action, after_future, target)
            after_objective = float(objective(after_values, unit, old, scale))
            row.update({'after_cpu_objective': after_objective,
                        'true_step_delta': after_objective - row['before_cpu_objective'],
                        'after_counts': after_counts,
                        'after_cpu_action_mse': float(after_values['action_mse']),
                        'after_cpu_latent_nmse': float(after_values['latent_nmse']),
                        'after_cpu_action_kl': float(after_values['action_kl']),
                        'after_cpu_latent_kl': float(after_values['latent_kl'])})
            save_checkpoint()
            progress(phase='step_after_cpu_complete', updates=updates,
                     case_id=case, true_step_delta=row['true_step_delta'])

        def ensure_snapshot(step):
            return write_snapshot(unit, step, reuse_source=(step == 0))

        ensure_initial_probe()
        progress(phase='snapshot_0', updates=updates)
        ensure_snapshot(0)
        read_snapshot(unit, 0)
        state['snapshot_reused_queries'] = 80
        save_checkpoint()

        ensure_after_cpu()
        if updates in (20, 40, 60) and not (out / 'snapshots' / f'step_{updates:03d}' / 'summary.json').is_file():
            progress(phase='snapshot_' + str(updates), updates=updates)
            ensure_snapshot(updates)
            save_checkpoint()

        while updates < 80:
            if history:
                ensure_after_cpu()
            if updates in (20, 40, 60) and not (out / 'snapshots' / f'step_{updates:03d}' / 'summary.json').is_file():
                ensure_snapshot(updates)
                save_checkpoint()
            deadline.check(max([h['seconds'] for h in history[-3:]], default=300) * 1.15 + 180)
            case = expected_order[updates]
            value, meta = datum(case)
            runtime.description = meta['description']
            banks.seen.clear()
            target = teacher('calibration', case, 0)
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            action, counts, future = sampler.differentiable_action(
                runtime, value, rows[case]['sampler_seeds'][0],
                checkpointed=True, return_future=True)
            count_model_query('training_checkpointed')
            values = terms(action, future, target)
            loss = objective(values, unit, old, scale)
            if (len(banks.seen) != 614 or not loss.requires_grad or not bool(torch.isfinite(loss))):
                raise RuntimeError('Training objective or 614-module coverage failed')
            if updates == 0:
                if initial_probe is None or initial_probe['case_id'] != case:
                    raise RuntimeError('First update must match the warm-start case2000 probe')
                checkpointed_parity = {
                    'action_max_abs': float((action.detach().cpu().float() - initial_probe['action'].float()).abs().max()),
                    'future_max_abs': float((future.detach().cpu().float() - initial_probe['future'].float()).abs().max())}
                if any(value != 0 for value in checkpointed_parity.values()):
                    raise RuntimeError('First checkpointed/no-grad action+future parity failed')
                state['first_train_forward_parity'] = checkpointed_parity
            before_gpu = float(loss.detach())
            before_cpu_values = terms(action.detach().cpu(), future.detach().cpu(), target)
            before_cpu = float(objective(before_cpu_values, unit, old, scale))
            loss.backward()
            coverage = banks.coverage()
            if any(v['connected'] != 614 or v['finite'] != 614 or v['nonzero'] < 1
                   for v in coverage.values()):
                raise RuntimeError('Ratio or channel gradients disconnected or nonfinite')
            norm = torch.nn.utils.clip_grad_norm_(parameters,
                                                  old['training']['gradient_norm_clip'],
                                                  error_if_nonfinite=True)
            optimizer.step()
            if updates == 0 and old['training']['optimizer_reset_after_update1']:
                optimizer.state.clear()
            banks.project()
            torch.cuda.synchronize()
            updates += 1
            history.append({'update': updates, 'case_id': case, 'seed_index': 0,
                            'before_gpu_objective': before_gpu,
                            'before_cpu_objective': before_cpu,
                            'before_gpu_minus_cpu': before_gpu - before_cpu,
                            'action_mse': float(values['action_mse'].detach()),
                            'latent_nmse': float(values['latent_nmse'].detach()),
                            'action_kl': float(values['action_kl'].detach()),
                            'latent_kl': float(values['latent_kl'].detach()),
                            'loss': before_gpu, 'gradient_norm': float(norm),
                            'coverage': coverage, 'seconds': time.monotonic() - started,
                            'pbs_jobid': os.environ['PBS_JOBID'], 'base_updates': 80,
                            'additional_updates': updates, 'total_updates': 80 + updates,
                            **counts})
            del action, future, values, loss, before_cpu_values
            state.update(phase='training', updates=updates,
                         additional_updates=updates, total_updates=80 + updates)
            save_checkpoint()
            progress(phase='training', updates=updates, last_update=history[-1])
            ensure_after_cpu()
            if updates in (20, 40, 60):
                progress(phase='snapshot_' + str(updates), updates=updates)
                ensure_snapshot(updates)
                save_checkpoint()

        ensure_after_cpu()
        progress(phase='deployment_parity', updates=80)
        first = old['calibration']['case_ids'][0]
        uncached = query(first, 0, purpose='deployment_parity')
        progress(phase='deployment_parity_uncached', updates=80)
        rounding = banks.cache_compensated_weights()
        cached = query(first, 0, purpose='deployment_parity')
        progress(phase='deployment_parity_cached', updates=80)
        folded = {'action_max_abs': float((uncached[0].float() - cached[0].float()).abs().max()),
                  'future_max_abs': float((uncached[1].float() - cached[1].float()).abs().max())}
        if any(value != 0 for value in folded.values()):
            raise RuntimeError('Cached deployment differs from uncached final80 action/future')
        runtime.current_arm = unit
        runtime.description = datum(first)[1]['description']
        actual, _ = runtime.infer(datum(first)[0], rows[first]['sampler_seeds'][0], measure=False)
        count_model_query('runtime_infer')
        progress(phase='deployment_parity_runtime_infer', updates=80)
        actual_parity = float((actual.float().cpu() - cached[0].float()).abs().max())
        if actual_parity != 0:
            raise RuntimeError('Actual runtime.infer action differs from cached query')
        del actual, uncached, cached
        frozen = banks.verify_frozen(deployment['module_masters'])
        state.update(deployment_folded_parity=folded,
                     actual_infer_action_parity_max_abs=actual_parity,
                     deployment_parity_queries=3)
        atomic_torch_save(torch, out / 'deployment.pt', {
            'protocol': cfg, 'unit': unit, 'updates': 80, 'base_updates': 80,
            'additional_updates': 80, 'total_updates': 160,
            'baseline_checkpoint': str(source_root_checkpoint),
            'warm_start_job': cfg['start_job'], 'source_optimizer_discarded': True,
            'learning_rate_factor': factor, 'ratios': banks.clip_snapshot(),
            'log2_scales': banks.scale_snapshot(), 'scale_metadata_dtype': 'float32_log2'})
        ref.save_json(out / 'weight_rounding.json', rounding)

        ensure_snapshot(80)
        final_records = read_snapshot(unit, 80)
        copy_final_calibration(final_records)
        endpoint_count = 80
        calibration_summary = c80.aggregate_records(final_records, old, 'calibration')
        for metric in ('action_kl', 'latent_kl', 'objective_loss'):
            calibration_summary['overall'][metric] = sum(r[metric] for r in final_records) / 80
            for suite in ref.SUITES:
                rr = [r for r in final_records if r['suite'] == suite]
                calibration_summary['by_suite'][suite][metric] = sum(r[metric] for r in rr) / len(rr)
        ref.save_json(root / 'results/calibration' / unit / 'summary.json', {
            'protocol': cfg['protocol'], 'complete': True, 'unit': unit,
            'split': 'calibration', 'updates': 80, 'base_updates': 80,
            'additional_updates': 80, 'total_updates': 160,
            'reused_snapshot_step': 80, 'pbs_jobid': os.environ['PBS_JOBID'],
            **calibration_summary})
        for split in ('heldout',):
            records = []
            for case, seed in c80.queries(old, split):
                path = endpoint_path(split, case, seed)
                if path.is_file():
                    record = read_json(path)
                    validate_new_record(record, unit, 80, split, case, seed)
                else:
                    target = teacher(split, case, seed)
                    action, future, counts = query(case, seed, purpose='final_evaluation')
                    record = make_record(unit, split, case, seed, 80,
                                         action, future, target, counts)
                    ref.save_json(path, record)
                records.append(record)
                progress(phase='final_evaluation', split=split,
                         queries=len(records), last_case=case, last_seed=seed)
            if len(records) != 80:
                raise RuntimeError('Original held-out evaluation must contain exactly 40x2 records')
            summary = c80.aggregate_records(records, old, split)
            for metric in ('action_kl', 'latent_kl', 'objective_loss'):
                summary['overall'][metric] = sum(r[metric] for r in records) / len(records)
                for suite in ref.SUITES:
                    rr = [r for r in records if r['suite'] == suite]
                    summary['by_suite'][suite][metric] = sum(r[metric] for r in rr) / len(rr)
            ref.save_json(root / 'results' / split / unit / 'summary.json', {
                'protocol': cfg['protocol'], 'complete': True, 'unit': unit,
                'split': split, 'updates': 80, 'base_updates': 80,
                'additional_updates': 80, 'total_updates': 160,
                'pbs_jobid': os.environ['PBS_JOBID'], **summary})
            endpoint_count += len(records)

        snapshot_counts = {}
        for step in SNAPSHOTS:
            snap = read_json(out / 'snapshots' / f'step_{step:03d}' / 'summary.json')
            if snap.get('complete') is not True or snap.get('queries') != 80:
                raise RuntimeError('All five complete C80 snapshots are required')
            read_snapshot(unit, step)
            snapshot_counts[str(step)] = snap['queries']
        counts = {'base_updates': 80, 'additional_updates': 80, 'total_updates': 160,
                  'training_checkpointed_forwards': 80,
                  'training_checkpointed_forward_attempts': state.get('training_checkpointed_query_count', 0),
                  'step_before_gpu_objectives': 80, 'step_before_cpu_objectives': 80,
                  'step_after_cpu_queries': state.get('step_after_cpu_query_count', 0),
                  'snapshot_queries_by_step': snapshot_counts,
                  'snapshot_reused_source_queries': 80,
                  'snapshot_new_model_queries': state.get('snapshot_query_count', 0),
                  'final_calibration_queries': 80, 'final_calibration_queries_reused_from_snapshot80': 80,
                  'final_heldout_queries': state.get('final_evaluation_query_count', 0),
                  'final_evaluation_queries': endpoint_count,
                  'deployment_parity_queries': state.get('deployment_parity_query_count', 0)
                      + state.get('runtime_infer_query_count', 0),
                  'initial_source_endpoint_queries': state.get('initial_source_endpoint_query_count', 0),
                  'new_model_queries': state['new_model_queries'], 'new_teacher_queries': 0}
        if (updates != 80 or endpoint_count != 160
                or any(value != 80 for value in snapshot_counts.values())
                or counts['step_after_cpu_queries'] < 80
                or counts['snapshot_new_model_queries'] < 320
                or counts['final_heldout_queries'] < 80
                or counts['initial_source_endpoint_queries'] < 1
                or counts['deployment_parity_queries'] < 3
                or sum(row.get('after_cpu_objective') is not None for row in history) != 80):
            raise RuntimeError('Completion gate requires 80 updates, 160 final records, and five complete C80 snapshots')

        training = {'protocol': cfg['protocol'], 'complete': True, 'unit': unit,
                    'pbs_jobid': os.environ['PBS_JOBID'], 'warm_start_job': cfg['start_job'],
                    'source_checkpoint': str(source_root_checkpoint), 'source_updates': 80,
                    'updates': 80, 'base_updates': 80, 'additional_updates': 80,
                    'total_updates': 160, 'update_order': expected_order,
                    'learning_rate_factor': factor,
                    'learning_rates': {'clip': work_training['clip_learning_rate'],
                                       'channel': work_scaling['learning_rate']},
                    'learning_rate_decision': {'complete': decision['complete'],
                                               'heldout_used': decision['heldout_used'],
                                               'sampling_scope': decision['sampling_scope'],
                                               'learning_rate_factor': decision['learning_rate_factor']},
                    'source_optimizer_discarded': True,
                    'heldout_used_for_selection': False,
                    'optimizer_reset_after_update1': True,
                    'history': history, 'counts': counts,
                    'initial_source_endpoint_objective_max_abs': 0.0,
                    'first_train_forward_parity': state['first_train_forward_parity'],
                    'cached_query_parity': folded,
                    'actual_infer_action_parity_max_abs': actual_parity,
                    'clipping': banks.clip_summary(), 'scaling': banks.scaling_summary(),
                    'softbin_selfcheck': kl_check, 'gradient_selfcheck': gradient_check,
                    'gradient_coupling_complete': True, 'input_absmax_chain': True, **frozen}
        ref.save_json(out / 'training_receipt.json', training)
        summary = {**state, 'complete': True, 'phase': 'complete', 'updates': 80,
                   'base_updates': 80, 'additional_updates': 80, 'total_updates': 160,
                   'evaluation_queries': endpoint_count, 'snapshots': snapshot_counts,
                   'counts': counts, 'training_receipt': training,
                   'source_optimizer_discarded': True, 'heldout_used_for_selection': False,
                   'new_teacher_queries': 0, **frozen}
        ref.save_json(out / 'summary.json', summary)
        progress(phase='complete', complete=True, updates=80, **counts)
        print('KL_CONTINUATION_WORKER_COMPLETE', flush=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--unit', choices=ARMS, required=True)
    args = parser.parse_args()
    c80.guard()
    cfg = read_json(args.protocol)
    if args.unit not in cfg.get('arms', []):
        raise RuntimeError('Requested arm is absent from the continuation protocol')
    cfg['selected_unit'] = args.unit
    return run(cfg)


if __name__ == '__main__':
    raise SystemExit(main())
