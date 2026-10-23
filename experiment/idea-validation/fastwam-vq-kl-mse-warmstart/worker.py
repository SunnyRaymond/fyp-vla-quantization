"""Apply the completed MSE-start dual-KL diagnostic candidate once, then evaluate it."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time


DIRECTIONS = ('saved_adam', 'fresh_adam')
FRACTIONS = (1.0, 0.5, 0.25)
METRICS = ('action_mse', 'latent_nmse', 'action_kl', 'latent_kl')


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('Required file is missing: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    temp.replace(path)


def compute_guard():
    job = os.environ.get('PBS_JOBID')
    nodefile = os.environ.get('PBS_NODEFILE')
    host = socket.gethostname().split('.')[0]
    if not job or not nodefile or not Path(nodefile).is_file() or 'login' in host.lower():
        raise RuntimeError('A real PBS compute allocation is required')
    nodes = {value.split('.')[0] for value in Path(nodefile).read_text(encoding='utf-8').split()}
    if host not in nodes:
        raise RuntimeError('Current host is not part of the PBS allocation')


def label_for(direction, fraction):
    return direction + '_f' + str(float(fraction)).replace('.', 'p')


def mean(values):
    values = list(values)
    if not values:
        raise RuntimeError('Cannot summarize an empty C80 set')
    return sum(float(value) for value in values) / len(values)


def close(a, b, *, rel=1e-6, abs_=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def record_values(record):
    return {
        'action_mse': record['action_mse'] if 'action_mse' in record else record['first10motor_mse'],
        'latent_nmse': record['latent_nmse'] if 'latent_nmse' in record else record['future_latent_nmse'],
        'action_kl': record['action_kl'],
        'latent_kl': record['latent_kl'],
    }


def check_endpoint_binding(record, protocol, arm, split, case, seed, row):
    expected = {'protocol': protocol, 'arm': arm, 'split': split, 'updates': 80,
                'case_id': case, 'suite': row['suite'], 'plan_row': row,
                'seed_index': seed, 'sampler_seed': row['sampler_seeds'][seed]}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Source C80 endpoint binding differs: ' + str((arm, split, case, seed)))


def run(cfg, unit):
    compute_guard()
    if unit != 'dual_kl' or cfg.get('protocol') != 'fastwam-vq-kl-mse-warmstart-v1':
        raise RuntimeError('Warmstart protocol or unit is invalid')
    if (cfg.get('arm') != unit or cfg.get('checkpoint_arm') != 'mse_nmse'
            or cfg.get('diagnostic_unit') != 'mse_dual'):
        raise RuntimeError('Warmstart arm/checkpoint/diagnostic binding changed')
    if cfg.get('directions') != list(DIRECTIONS) or cfg.get('fractions') != list(FRACTIONS):
        raise RuntimeError('The frozen six diagnostic candidate set changed')
    if cfg.get('gradient_queries_new') != 0 or cfg.get('gradient_queries_reused') != 80:
        raise RuntimeError('The full MSE-start diagnostic gradient must be reused exactly')
    if cfg.get('maximum_refinement_updates') != 1:
        raise RuntimeError('At most one refinement update is allowed')
    if cfg.get('calibration_case_ids') != list(range(2000, 2080)):
        raise RuntimeError('The original C80 calibration cohort changed')
    if (cfg.get('heldout_case_ids') != list(range(3000, 3040))
            or cfg.get('heldout_seed_indices') != [0, 1]):
        raise RuntimeError('The original paired held-out cohort changed')

    import torch
    import numpy as np
    import sampler
    import runner
    import calibration_source
    import clipping_reference as ref
    import c80_reference as c80
    import latent_reference
    from channel_banks import ChannelBanks
    from checkpointing import atomic_torch_save, capture_rng, optimizer_to_device, to_cpu
    from direction_utils import adam_delta
    from metrics_reference import metrics
    from softbin_loss import losses

    sampler._require_compute_allocation()
    torch.set_num_threads(8)
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False

    root = Path(cfg['remote_root'])
    source = Path(cfg['source_root'])
    source_mse = Path(cfg['source_mse_root'])
    diag_root = Path(cfg['diagnostic_root'])
    if len({root, source, source_mse, diag_root}) != 4:
        raise RuntimeError('New output and immutable source roots must be distinct')
    old = read_json(source / 'protocol.json')
    if old.get('protocol') != 'fastwam-vq-dual-kl-v1' or Path(old.get('remote_root', '')) != source:
        raise RuntimeError('source_root must be the original detailed dual-KL protocol')
    if Path(old['source_per_channel_root']) != source_mse:
        raise RuntimeError('source_mse_root differs from the immutable MSE source')
    if (old['calibration']['case_ids'] != cfg['calibration_case_ids']
            or old['evaluation']['case_ids'] != cfg['heldout_case_ids']
            or old['evaluation']['seed_indices'] != cfg['heldout_seed_indices']):
        raise RuntimeError('The original C80 and paired held-out contract changed')
    expected_order = old['calibration']['update_order']
    if (len(expected_order) != 80 or len(set(expected_order)) != 80
            or set(expected_order) != set(cfg['calibration_case_ids'])):
        raise RuntimeError('The original exact80 optimizer history contract changed')

    artifact = Path(os.environ['ARTIFACTS'])
    source_protocol = read_json(artifact / 'source_protocol.json')
    source_protocol_file = read_json(source_mse / 'protocol.json')
    if source_protocol != source_protocol_file or source_protocol.get('protocol') != 'fastwam-vq-per-channel-scaling-clipping-v1':
        raise RuntimeError('The copied original MSE protocol differs from the immutable source')
    source_checkpoint = source_mse / 'results/train_eval/checkpoint.pt'
    source_receipt = read_json(source_mse / 'results/train_eval/training_receipt.json')
    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('protocol') != source_protocol or saved.get('updates') != 80
            or saved.get('optimizer') is None or source_receipt.get('complete') is not True
            or source_receipt.get('updates') != 80 or source_receipt.get('pbs_jobid') != cfg['source_mse_job']
            or source_receipt.get('update_order') != source_protocol.get('calibration', {}).get('update_order')
            or source_receipt.get('update_order') != expected_order
            or [entry.get('case_id') for entry in saved.get('history', [])] != expected_order):
        raise RuntimeError('The original MSE final80 checkpoint/history/receipt binding differs')
    if any(entry.get('pbs_jobid') != cfg['source_mse_job'] for entry in saved['history']):
        raise RuntimeError('MSE source history contains a different PBS job')

    diag_protocol = read_json(diag_root / 'protocol.json')
    diag_state = read_json(diag_root / 'results/mse_dual/summary.json')
    direction_summary = read_json(diag_root / 'results/mse_dual/direction_summary.json')
    diag_gradient_path = diag_root / 'results/mse_dual/gradient_state.pt'
    diag_gradient = torch.load(diag_gradient_path, map_location='cpu', weights_only=True)
    if (diag_protocol.get('protocol') != 'fastwam-vq-kl-direction-v1'
            or Path(diag_protocol.get('source_root', '')) != source
            or diag_protocol.get('case_ids') != cfg['calibration_case_ids']
            or diag_protocol.get('seed_indices') != [0]
            or diag_protocol.get('units', {}).get('mse_dual') !=
               {'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl'}):
        raise RuntimeError('MSE-start dual-KL diagnostic protocol binding differs')
    if (diag_state.get('complete') is not True or diag_state.get('unit') != 'mse_dual'
            or diag_state.get('checkpoint_arm') != 'mse_nmse' or diag_state.get('loss') != 'dual_kl'
            or diag_state.get('pbs_jobid') != cfg['diagnostic_job']
            or diag_state.get('heldout_used') is not False or diag_state.get('gradient_query_count') != 80
            or diag_gradient.get('protocol') != diag_protocol or diag_gradient.get('unit') != 'mse_dual'
            or len(diag_gradient.get('full_gradient', [])) == 0
            or [row.get('case_id') for row in diag_gradient.get('baseline', [])] != cfg['calibration_case_ids']):
        raise RuntimeError('The completed exact80 MSE-start dual-KL gradient diagnostic is required')
    if not isinstance(direction_summary, list) or len(direction_summary) != 6:
        raise RuntimeError('All six completed diagnostic C80 candidate means are required')
    if diag_state.get('direction_conditions') != direction_summary:
        raise RuntimeError('Diagnostic summary and direction means differ')
    by_condition = {(row['direction'], float(row['fraction'])): row for row in direction_summary}
    if set(by_condition) != {(direction, fraction) for direction in DIRECTIONS for fraction in FRACTIONS}:
        raise RuntimeError('Diagnostic directions/fractions differ from the frozen six candidates')
    if any(row.get('queries') != 80 or not math.isfinite(float(row.get('objective', float('nan'))))
           for row in direction_summary):
        raise RuntimeError('Every diagnostic candidate must have a finite full-C80 objective')
    baseline_rows = diag_gradient['baseline']
    baseline_by_case = {row['case_id']: row for row in baseline_rows}
    if set(baseline_by_case) != set(cfg['calibration_case_ids']):
        raise RuntimeError('Diagnostic no-step baseline does not cover C80 exactly')

    scale = read_json(source / 'results/prepare/loss_scaling.json')
    stats = read_json(source / 'results/fit_stats/summary.json')
    if (scale.get('complete') is not True or scale.get('heldout_used') is not False
            or scale.get('case_ids') != cfg['calibration_case_ids']
            or stats.get('complete') is not True or stats.get('heldout_used') is not False
            or stats.get('calibration_case_ids') != cfg['calibration_case_ids']):
        raise RuntimeError('Calibration-only frozen loss scaling/statistics are required')
    if not math.isclose(scale['original_latent_weight'],
                        old['training']['rho'] * old['training']['normalization_c'],
                        rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError('The original latent-loss weight changed')

    first = cfg['calibration_case_ids'][0]
    source_arm = 'mse_nmse'
    source_endpoint_root = source / 'results/calibration' / source_arm / 'queries'
    source_plan = read_json(old['calibration']['plan_file'])
    rows_by_case = {row['case_id']: row for row in source_plan['inputs']}
    source_endpoints = {}
    for case in cfg['calibration_case_ids']:
        endpoint = read_json(source_endpoint_root / f'{case}_0.json')
        check_endpoint_binding(endpoint, old['protocol'], source_arm, 'calibration', case, 0,
                               rows_by_case[case])
        if not all(key in endpoint for key in ('action', 'commands', 'teacher_action', 'teacher_commands')):
            raise RuntimeError('Source C80 endpoint is missing raw action/command matrices')
        source_endpoints[case] = endpoint
    if not close(mean(row['objective'] for row in baseline_rows),
                 diag_state['baseline_metrics']['objective']):
        raise RuntimeError('Diagnostic baseline C80 objective differs from its complete per-case records')
    for case, diagnostic_row in baseline_by_case.items():
        for key, value in record_values(source_endpoints[case]).items():
            if not close(diagnostic_row[key], value):
                raise RuntimeError('Diagnostic baseline differs from original MSE C80 endpoint: '
                                   + str(case) + '/' + key)

    teacher_dirs = {'calibration': Path(old['control_root']) / 'results/teacher/calibration',
                    'heldout': Path(old['clipping_root']) / 'results/teacher'}
    teacher_receipt = read_json(teacher_dirs['calibration'] / 'receipt.json')
    if (teacher_receipt.get('complete') is not True or teacher_receipt.get('queries') != 80
            or teacher_receipt.get('case_ids') != cfg['calibration_case_ids']
            or teacher_receipt.get('seed_indices') != [0]):
        raise RuntimeError('Shared calibration teacher targets differ')

    out = root / 'results/dual_kl'
    out.mkdir(parents=True, exist_ok=True)
    accounting_path = out / 'accounting.json'
    accounting = read_json(accounting_path) if accounting_path.is_file() else {
        'protocol': cfg['protocol'], 'unit': unit, 'model_queries': 0,
        'source_replay_queries': 0, 'calibration_queries': 0,
        'deployment_parity_queries': 0, 'heldout_queries': 0}
    if accounting.get('protocol') != cfg['protocol'] or accounting.get('unit') != unit:
        raise RuntimeError('Saved forward accounting belongs to a different warmstart protocol')
    state = {'protocol': cfg['protocol'], 'unit': unit, 'arm': unit, 'pbs_jobid': os.environ['PBS_JOBID'],
             'phase': 'loading', 'complete': False}

    def progress(**values):
        state.update(values)
        write_json(out / 'progress.json', {**state, 'forward_accounting': accounting})
        print('MSE_WARMSTART_PROGRESS ' + json.dumps(state, allow_nan=False), flush=True)

    def charge(counter):
        accounting['model_queries'] += 1
        accounting[counter] = accounting.get(counter, 0) + 1
        accounting['last_pbs_jobid'] = os.environ['PBS_JOBID']
        state['new_model_queries'] = accounting['model_queries']
        write_json(accounting_path, accounting)

    data_root, rows, cache, deployment, _ = ref.load_inputs(old['source_clipping_protocol'], torch)
    if not (set(cfg['calibration_case_ids']) | set(cfg['heldout_case_ids'])) <= set(rows):
        raise RuntimeError('Frozen source plan is missing C80 or held-out rows')
    runtime = None
    deadline = ref.Deadline()
    started = time.monotonic()
    source_parity = {'passed': False}
    deployment_parity = {'passed': False}
    frozen_checks = {'passed': False}
    selection = {}
    chosen = {}
    refinement_updates = 0
    total_updates = 80
    try:
        progress(phase='loading', source_updates=80, gradient_queries_new=0,
                 gradient_queries_reused=80, new_teacher_queries=0)
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        banks = ChannelBanks(runtime, old['calibration']['bank_directory'],
                             deployment['module_masters'], old['clipping'], old['scaling'])
        banks.restore_clips(saved['ratios'])
        banks.restore_scales(saved.get('shadow_log2_scales', saved.get('log2_scales')))
        groups = banks.optimizer_groups(old['training'], old['scaling'])
        for group, name in zip(groups, ('ratio', 'channel')):
            group['name'] = name
        parameters = [parameter for group in groups for parameter in group['params']]
        if (len(groups) != 2 or [group['name'] for group in groups] != ['ratio', 'channel']
                or sum(parameter.numel() for parameter in groups[0]['params']) != 614
                or sum(parameter.numel() for parameter in groups[1]['params']) != 1806863):
            raise RuntimeError('Trainable scope/order must be 614 ratios then 1,806,863 channel scales')
        if (any(parameter.requires_grad for parameter in runtime.model.parameters())
                or any(parameter.requires_grad for parameter in list(banks.books.values()) + list(banks.logs.values()))):
            raise RuntimeError('Backbone, VQ books/indices, gains and exponents must remain frozen')
        originals = [parameter.detach().clone() for parameter in parameters]
        data = {}

        def datum(case):
            if case not in data:
                observation, metadata = calibration_source.load_observation(np, data_root, rows[case])
                data[case] = runtime.datum(observation), metadata
            return data[case]

        def target(split, case, seed):
            value = torch.load(teacher_dirs[split] / f'{case}_{seed}.pt',
                               map_location='cpu', weights_only=True)
            if (value['plan_row'] != rows[case]
                    or value['sampler_seed'] != rows[case]['sampler_seeds'][seed]):
                raise RuntimeError('Shared teacher target binding differs')
            return value

        def terms(action, future, teacher_value):
            return losses(torch, action, future, teacher_value['action'].to(action.device),
                          teacher_value['future'].to(future.device), stats, old['softbin'])

        def objective(values):
            return (scale['action_kl_coefficient'] * values['action_kl']
                    + old['training']['rho'] * old['training']['normalization_c']
                    * scale['latent_kl_coefficient'] * values['latent_kl'])

        def make_record(split, case, seed, action, future, teacher_value, counts, *, replay=False):
            values = terms(action, future, teacher_value)
            measured = metrics(torch, latent_reference, action, future, teacher_value, old['training'])
            value_map = {key: float(value.detach()) for key, value in values.items()}
            command_matrix = runtime.command(action.detach().clone())
            teacher_command_matrix = runtime.command(teacher_value['action'].clone())
            diagnostic_condition = chosen['condition'] if 'chosen' in selection else None
            record = {
                'protocol': cfg['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'],
                'unit': unit, 'arm': unit, 'split': split, 'case_id': case,
                'suite': rows[case]['suite'], 'plan_row': rows[case],
                'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed],
                'source_updates': 80, 'refinement_updates': refinement_updates,
                'total_updates': total_updates, 'updates': total_updates, 'counts': counts,
                'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl',
                'source_record_reused': False,
                'source_record_protocol': old['protocol'], 'source_record_arm': source_arm,
                'source_record_pbs_jobid': source_endpoints[first].get('pbs_jobid'),
                'source_checkpoint_protocol': source_protocol['protocol'],
                'source_checkpoint_pbs_jobid': cfg['source_mse_job'],
                'diagnostic_protocol': diag_protocol['protocol'],
                'diagnostic_unit': 'mse_dual', 'diagnostic_pbs_jobid': cfg['diagnostic_job'],
                'diagnostic_condition': diagnostic_condition,
                'diagnostic_replay_verified': bool(replay),
                **measured, **value_map, 'objective_loss': float(objective(values).detach()),
                'action': action.detach().float().cpu().tolist(),
                'commands': command_matrix.tolist(),
                'teacher_action': teacher_value['action'].float().tolist(),
                'teacher_commands': teacher_command_matrix.tolist(),
            }
            return record

        def validate_new_record(record, split, case, seed):
            expected = {'protocol': cfg['protocol'], 'unit': unit, 'arm': unit, 'split': split,
                        'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                        'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed],
                        'source_updates': 80, 'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl'}
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Saved warmstart endpoint binding differs: ' + str((split, case, seed)))
            if (record.get('refinement_updates') not in (0, 1)
                    or record.get('updates') != 80 + record['refinement_updates']
                    or record.get('total_updates') != record['updates']):
                raise RuntimeError('Saved warmstart endpoint update accounting differs')
            if any(not math.isfinite(float(record.get(key, float('nan'))))
                   for key in (*METRICS, 'objective_loss')):
                raise RuntimeError('Saved warmstart endpoint has nonfinite loss metrics')
            if (record.get('source_record_reused') is not False
                    or record.get('source_record_protocol') != old['protocol']
                    or record.get('source_record_arm') != source_arm
                    or record.get('source_checkpoint_pbs_jobid') != cfg['source_mse_job']
                    or record.get('diagnostic_unit') != 'mse_dual'
                    or record.get('diagnostic_pbs_jobid') != cfg['diagnostic_job']
                    or record.get('diagnostic_condition') != chosen.get('condition')):
                raise RuntimeError('Saved warmstart endpoint provenance differs')
            if not all(key in record for key in ('action', 'commands', 'teacher_action', 'teacher_commands')):
                raise RuntimeError('Saved warmstart endpoint lacks raw action/command matrices')

        def query(case, seed, counter):
            deadline.check(90)
            value, metadata = datum(case)
            result = ref.query(runtime, sampler, torch, value, metadata['description'],
                               rows[case]['sampler_seeds'][seed], banks)
            charge(counter)
            return result

        def restore():
            with torch.no_grad():
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)

        def set_delta(delta, fraction):
            expected = []
            index = 0
            with torch.no_grad():
                for group in groups:
                    lower, upper = (0.5, 1.0) if group['name'] == 'ratio' else (-1.0, 1.0)
                    for parameter in group['params']:
                        candidate = (originals[index] + float(fraction) * delta[index]).clamp(lower, upper)
                        parameter.copy_(candidate)
                        expected.append(candidate.detach().clone())
                        index += 1
            return expected

        def source_values(record):
            return record_values(record)

        def verify_source_replay():
            endpoint = source_endpoints[first]
            action, future, counts = query(first, 0, 'source_replay_queries')
            teacher_value = target('calibration', first, 0)
            replay = make_record('calibration', first, 0, action, future, teacher_value, counts)
            differences = {}
            for key in METRICS:
                differences[key] = abs(float(replay[key]) - float(source_values(endpoint)[key]))
                if not close(replay[key], source_values(endpoint)[key]):
                    raise RuntimeError('Source case2000 loss replay differs: ' + key)
            source_action = np.asarray(endpoint['action'], dtype=np.float32)
            current_action = np.asarray(replay['action'], dtype=np.float32)
            source_commands = np.asarray(endpoint['commands'], dtype=np.float32)
            current_commands = np.asarray(replay['commands'], dtype=np.float32)
            action_abs = float(np.max(np.abs(current_action - source_action)))
            command_abs = float(np.max(np.abs(current_commands - source_commands)))
            if action_abs != 0.0 or command_abs != 0.0:
                raise RuntimeError('Source case2000 action/command replay is not exact')
            replay.update({'source_record_reused': False,
                           'source_record_protocol': endpoint['protocol'],
                           'source_record_arm': endpoint['arm'],
                           'source_record_pbs_jobid': endpoint.get('pbs_jobid'),
                           'diagnostic_unit': 'mse_dual', 'diagnostic_pbs_jobid': cfg['diagnostic_job'],
                           'diagnostic_replay_verified': True})
            source_parity.update({'passed': True, 'case_id': first,
                                  'source_endpoint_protocol': endpoint['protocol'],
                                  'source_endpoint_arm': endpoint['arm'],
                                  'source_endpoint_pbs_jobid': endpoint.get('pbs_jobid'),
                                  'loss_abs_differences': differences,
                                  'action_max_abs': action_abs,
                                  'commands_max_abs': command_abs})
            del action, future
            return replay

        # Replay and record construction happen before any full endpoint evaluation so
        # command conversion or metric API errors fail before selected vectors are saved.
        replay_record = verify_source_replay()
        del replay_record

        gradient = [value.to('cuda') for value in diag_gradient['full_gradient']]
        if (len(gradient) != len(parameters)
                or any(value.shape != parameter.shape or not bool(torch.isfinite(value).all())
                       for value, parameter in zip(gradient, parameters))):
            raise RuntimeError('Reused MSE-start dual-KL gradient tensor order/shapes are invalid')

        baseline_mean = float(diag_state['baseline_metrics']['objective'])
        if not math.isfinite(baseline_mean):
            raise RuntimeError('Diagnostic baseline objective is nonfinite')
        candidates = []
        for direction in DIRECTIONS:
            for fraction in FRACTIONS:
                row = by_condition[(direction, fraction)]
                candidates.append({'condition': label_for(direction, fraction),
                                   'direction': direction, 'fraction': float(fraction),
                                   'optimizer': direction, 'objective_mean': float(row['objective']),
                                   'queries': 80, 'source': 'reused_complete_diagnostic_C80',
                                   'diagnostic_pbs_jobid': cfg['diagnostic_job'],
                                   'refinement_updates': 1})
        baseline = {'condition': 'no_step', 'direction': None, 'fraction': 0.0,
                    'optimizer': 'source_checkpoint', 'objective_mean': baseline_mean,
                    'queries': 80, 'source': 'reused_source_C80_baseline',
                    'diagnostic_pbs_jobid': cfg['diagnostic_job'], 'refinement_updates': 0}
        conditions = [baseline] + candidates
        chosen = min(conditions, key=lambda item: (item['objective_mean'], item['condition']))
        selection = {'complete': True, 'heldout_used': False, 'sampling_scope': 'C80',
                     'method': 'reused_mse_dual_full_C80_actual_loss_selection',
                     'objective_coefficients': {
                         'action_kl': float(scale['action_kl_coefficient']),
                         'latent_kl': float(scale['latent_kl_coefficient']),
                         'latent_weight': float(old['training']['rho'] * old['training']['normalization_c'])},
                     'baseline_objective_mean': baseline_mean, 'baseline': baseline,
                     'candidates': candidates, 'conditions': conditions,
                     'chosen': dict(chosen), 'selected_from': 'baseline_plus_6_diagnostic_candidates',
                     'candidate_count': 6, 'condition_count': 7,
                     'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl',
                     'diagnostic_unit': 'mse_dual', 'diagnostic_pbs_jobid': cfg['diagnostic_job']}
        selection_path = out / 'selection.json'
        if selection_path.is_file():
            prior_selection = read_json(selection_path)
            if prior_selection != selection:
                raise RuntimeError('Persisted warmstart C80 selection differs from frozen diagnostic selection')
        else:
            write_json(selection_path, selection)
        progress(phase='selected', selected_condition=chosen['condition'],
                 selected_objective=chosen['objective_mean'])

        heldout_receipt = read_json(teacher_dirs['heldout'] / 'receipt.json')
        if (heldout_receipt.get('complete') is not True or heldout_receipt.get('queries') != 80
                or heldout_receipt.get('case_ids') != cfg['heldout_case_ids']
                or heldout_receipt.get('seed_indices') != cfg['heldout_seed_indices']):
            raise RuntimeError('Shared held-out teacher target coverage differs')

        # Rebuild from the immutable MSE final80 checkpoint on every restart. This
        # makes resumption idempotent and prevents applying the selected update twice.
        restore()
        optimizer = torch.optim.Adam(groups)
        optimizer_family = chosen['optimizer']
        optimizer_max_abs = 0.0
        refinement_updates = 0
        if chosen['condition'] == 'no_step':
            optimizer.load_state_dict(saved['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda')
        else:
            direction = chosen['direction']
            fraction = float(chosen['fraction'])
            delta = adam_delta(torch, gradient, groups,
                               saved['optimizer'] if direction == 'saved_adam' else None)
            expected_values = set_delta(delta, fraction)
            optimizer = torch.optim.Adam(groups)
            if direction == 'saved_adam':
                optimizer.load_state_dict(saved['optimizer'])
                optimizer_to_device(torch, optimizer, 'cuda')
            for group in optimizer.param_groups:
                group['lr'] = float(group['lr']) * fraction
            # set_delta generated a clone of the expected candidate; restore before
            # stepping so the optimizer acts on the original MSE parameters once.
            restore()
            for parameter, grad in zip(parameters, gradient):
                parameter.grad = grad.detach().clone()
            optimizer.step()
            banks.project()
            optimizer_max_abs = float(torch.stack([
                (parameter.detach() - expected).abs().max()
                for parameter, expected in zip(parameters, expected_values)]).max())
            if optimizer_max_abs > 1e-6:
                raise RuntimeError('Adam step differs from the diagnostic pure candidate: '
                                   + str(optimizer_max_abs))
            with torch.no_grad():
                for parameter, expected in zip(parameters, expected_values):
                    parameter.copy_(expected)
            for parameter in parameters:
                parameter.grad = None
            refinement_updates = 1
            del delta
        total_updates = 80 + refinement_updates

        from checkpointing import capture_rng
        history = list(saved.get('history', []))
        if refinement_updates:
            history.append({'update': 81, 'case_id': 'mse_dual_selected_C80_direction',
                            'seed_index': None, 'gradient_queries_reused': 80,
                            'direction': chosen['direction'], 'fraction': float(chosen['fraction']),
                            'optimizer_family': optimizer_family,
                            'objective_mean_selected': chosen['objective_mean'],
                            'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl',
                            'pbs_jobid': os.environ['PBS_JOBID']})
        checkpoint_path = out / 'checkpoint.pt'
        atomic_torch_save(torch, checkpoint_path, {
            'protocol': cfg, 'unit': unit, 'arm': unit, 'checkpoint_arm': 'mse_nmse',
            'loss': 'dual_kl', 'source_checkpoint': str(source_checkpoint),
            'source_checkpoint_pbs_jobid': cfg['source_mse_job'],
            'source_updates': 80, 'refinement_updates': refinement_updates,
            'updates': total_updates, 'history': history,
            'ratios': banks.clip_snapshot(), 'shadow_log2_scales': banks.scale_snapshot(),
            'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
            'selection': selection, 'gradient_source': {
                'diagnostic_root': str(diag_root), 'diagnostic_unit': 'mse_dual',
                'diagnostic_pbs_jobid': cfg['diagnostic_job'], 'gradient_queries_new': 0,
                'gradient_queries_reused': 80}, 'heldout_used': False})

        # Validate the saved-vector deployment and actual inference paths before C80/heldout.
        uncached_action, uncached_future, _ = query(first, 0, 'deployment_parity_queries')
        banks.cache_compensated_weights()
        cached_action, cached_future, _ = query(first, 0, 'deployment_parity_queries')
        uncached_action_cpu = uncached_action.detach().float().cpu()
        cached_action_cpu = cached_action.detach().float().cpu()
        uncached_future_cpu = uncached_future.detach().float().cpu()
        cached_future_cpu = cached_future.detach().float().cpu()
        cached_parity = {'action_max_abs': float((uncached_action_cpu - cached_action_cpu).abs().max()),
                         'future_max_abs': float((uncached_future_cpu - cached_future_cpu).abs().max())}
        runtime.current_arm = unit
        runtime.description = datum(first)[1]['description']
        actual, _ = runtime.infer(datum(first)[0], rows[first]['sampler_seeds'][0], measure=False)
        charge('deployment_parity_queries')
        infer_diff = float((actual.detach().float().cpu() - cached_action_cpu).abs().max())
        deployment_parity = {'passed': all(value == 0.0 for value in cached_parity.values()) and infer_diff == 0.0,
                             'uncached_cached': cached_parity,
                             'actual_infer_action_max_abs': infer_diff,
                             'source_case_id': first, 'cached_compensation': True}
        if deployment_parity['passed'] is not True:
            raise RuntimeError('Uncached/cached action+future and actual inference parity failed')
        frozen = banks.verify_frozen(deployment['module_masters'])
        frozen_checks = {'passed': all(frozen.get(key) is True for key in
                                       ('codebooks_unchanged', 'indices_unchanged', 'row_gains_unchanged',
                                        'original_exponents_unchanged', 'continuous_scale_metadata_separate')),
                         **frozen}
        if frozen_checks['passed'] is not True:
            raise RuntimeError('Frozen VQ/master verification failed')
        atomic_torch_save(torch, out / 'deployment.pt', {
            'protocol': cfg, 'unit': unit, 'arm': unit, 'checkpoint_arm': 'mse_nmse',
            'loss': 'dual_kl', 'source_updates': 80, 'refinement_updates': refinement_updates,
            'updates': total_updates, 'selection': selection,
            'ratios': banks.clip_snapshot(), 'log2_scales': banks.scale_snapshot(),
            'scale_metadata_dtype': 'float32_log2', 'dynamic_fake_a4': True, 'bf16_kv': True,
            'cached_query_parity': cached_parity, 'actual_infer_action_parity_max_abs': infer_diff,
            'frozen_checks': frozen_checks})
        del uncached_action, uncached_future, cached_action, cached_future, actual

        def endpoint_path(split, case, seed):
            return root / 'results' / split / unit / 'queries' / f'{case}_{seed}.json'

        for split in ('calibration', 'heldout'):
            queries = c80.queries(old, split)
            if len(queries) != 80:
                raise RuntimeError('Each final split must contain exactly 80 queries')
            records = []
            for index, (case, seed) in enumerate(queries, 1):
                deadline.check(90)
                path = endpoint_path(split, case, seed)
                if path.is_file():
                    record = read_json(path)
                    validate_new_record(record, split, case, seed)
                    if split == 'calibration' and record.get('diagnostic_replay_verified') is not True:
                        raise RuntimeError('Cached selected C80 endpoint lacks diagnostic replay verification')
                else:
                    action, future, counts = query(
                        case, seed, 'calibration_queries' if split == 'calibration' else 'heldout_queries')
                    teacher_value = target(split, case, seed)
                    record = make_record(split, case, seed, action, future, teacher_value, counts,
                                         replay=(split == 'calibration'))
                    if split == 'calibration':
                        if chosen['condition'] == 'no_step':
                            reference = baseline_by_case[case]
                            reference_values = {key: float(reference[key]) for key in METRICS}
                        else:
                            label = chosen['condition']
                            diag_case = read_json(diag_root / 'results/mse_dual/directions' /
                                                  label / f'{case}_0.json')
                            if (diag_case.get('protocol') != diag_protocol['protocol']
                                    or diag_case.get('unit') != 'mse_dual'
                                    or diag_case.get('direction') != chosen['direction']
                                    or float(diag_case.get('fraction')) != float(chosen['fraction'])
                                    or diag_case.get('case_id') != case
                                    or diag_case.get('seed_index') != seed
                                    or diag_case.get('sampler_seed') != rows[case]['sampler_seeds'][seed]):
                                raise RuntimeError('Selected diagnostic C80 case binding differs')
                            reference_values = {key: float(diag_case[key]) for key in METRICS}
                        diffs = {}
                        for key in METRICS:
                            diffs[key] = abs(float(record[key]) - reference_values[key])
                            if not close(record[key], reference_values[key]):
                                raise RuntimeError('Selected C80 endpoint differs from the original MSE-start diagnostic: '
                                                   + str(case) + '/' + key)
                        record['diagnostic_replay_abs_differences'] = diffs
                        record['diagnostic_replay_verified'] = True
                        record['diagnostic_condition'] = chosen['condition']
                        record['diagnostic_pbs_jobid'] = cfg['diagnostic_job']
                    write_json(path, record)
                    del action, future
                records.append(record)
                progress(phase='endpoint_evaluation', split=split, queries=index,
                         last_case=case, last_seed=seed)
            if len(records) != 80:
                raise RuntimeError('Final split is not a complete 80-query cohort')
            if (split == 'calibration'
                    and not close(mean(record['objective_loss'] for record in records),
                                  chosen['objective_mean'])):
                raise RuntimeError('Selected C80 objective mean differs from the diagnostic choice')
            summary_values = c80.aggregate_records(records, old, split)
            for key in ('action_kl', 'latent_kl', 'objective_loss'):
                summary_values['overall'][key] = mean(record[key] for record in records)
                for suite in ref.SUITES:
                    suite_records = [record for record in records if record['suite'] == suite]
                    summary_values['by_suite'][suite][key] = mean(record[key] for record in suite_records)
            write_json(root / 'results' / split / unit / 'summary.json', {
                'protocol': cfg['protocol'], 'complete': True, 'unit': unit, 'arm': unit,
                'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl', 'split': split,
                'pbs_jobid': os.environ['PBS_JOBID'], 'source_updates': 80,
                'refinement_updates': refinement_updates, 'total_updates': total_updates,
                'updates': total_updates, 'selection': selection, **summary_values})

        receipt = {'protocol': cfg['protocol'], 'complete': True,
                   'pbs_jobid': os.environ['PBS_JOBID'], 'unit': unit, 'arm': unit,
                   'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl',
                   'source_updates': 80, 'refinement_updates': refinement_updates,
                   'total_updates': total_updates, 'updates': total_updates,
                   'source_checkpoint': str(source_checkpoint),
                   'source_checkpoint_pbs_jobid': cfg['source_mse_job'],
                   'diagnostic_root': str(diag_root), 'diagnostic_unit': 'mse_dual',
                   'diagnostic_pbs_jobid': cfg['diagnostic_job'],
                   'matched_comparison_reference': {'root': cfg['matched_root'],
                                                    'mse_job_id': cfg['matched_mse_job'],
                                                    'dual_job_id': cfg['matched_dual_job']},
                   'gradient_query_budget': 80, 'gradient_queries_new': 0,
                   'gradient_queries_reused': 80,
                   'gradient_source': {'kind': 'completed_diagnostic_full_C80',
                                       'path': str(diag_gradient_path),
                                       'diagnostic_unit': 'mse_dual',
                                       'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl',
                                       'diagnostic_pbs_jobid': cfg['diagnostic_job']},
                   'calibration_queries': 80, 'heldout_queries': 80, 'evaluation_queries': 160,
                   'selection': selection,
                   'source_parity_passed': source_parity['passed'],
                   'source_original_endpoint_replay': source_parity,
                   'deployment_parity_passed': deployment_parity['passed'],
                   'deployment_parity': deployment_parity,
                   'frozen_verified': frozen_checks['passed'], 'frozen_checks': frozen_checks,
                   'new_teacher_queries': 0, 'new_model_queries': accounting['model_queries'],
                   'forward_accounting': accounting,
                   'checkpoint': str(checkpoint_path), 'deployment': str(out / 'deployment.pt'),
                   'heldout_used': False, 'seconds_inside_worker': time.monotonic() - started}
        write_json(out / 'training_receipt.json', receipt)
        write_json(out / 'summary.json', {**receipt, 'phase': 'complete'})
        progress(phase='complete', complete=True, source_updates=80,
                 refinement_updates=refinement_updates, total_updates=total_updates)
        print('MSE_WARMSTART_COMPLETE', flush=True)
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
    parser.add_argument('--unit', choices=('dual_kl',), required=True)
    args = parser.parse_args()
    return run(read_json(args.protocol), args.unit)


if __name__ == '__main__':
    raise SystemExit(main())
