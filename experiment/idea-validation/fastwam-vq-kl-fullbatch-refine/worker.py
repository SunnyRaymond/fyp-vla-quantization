"""One C80-mean-gradient refinement update, selected only by full-C80 loss."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time


ARMS = ('mse_nmse', 'dual_kl')
FRACTIONS = (1.0, 0.5, 0.25)
DIRECTIONS = ('saved_adam', 'fresh_adam')
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


def objective(values, unit, old, scale):
    action = (scale['action_kl_coefficient'] * values['action_kl']
              if unit == 'dual_kl' else values['action_mse'])
    latent = (scale['latent_kl_coefficient'] * values['latent_kl']
              if unit == 'dual_kl' else values['latent_nmse'])
    return action + old['training']['rho'] * old['training']['normalization_c'] * latent


def record_values(record):
    return {'action_mse': record['action_mse'] if 'action_mse' in record else record['first10motor_mse'],
            'latent_nmse': record['latent_nmse'] if 'latent_nmse' in record else record['future_latent_nmse'],
            'action_kl': record['action_kl'], 'latent_kl': record['latent_kl']}


def mean(values):
    values = list(values)
    if not values:
        raise RuntimeError('Cannot summarize an empty C80 set')
    return sum(float(value) for value in values) / len(values)


def close(a, b, *, rel=1e-6, abs_=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def check_record_binding(record, protocol, arm, split, case, seed, row):
    expected = {'protocol': protocol, 'arm': arm, 'split': split, 'updates': 80,
                'case_id': case, 'suite': row['suite'], 'plan_row': row,
                'seed_index': seed, 'sampler_seed': row['sampler_seeds'][seed]}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Source C80 endpoint binding differs: ' + str((arm, split, case, seed)))


def run(cfg, unit):
    compute_guard()
    if cfg.get('protocol') != 'fastwam-vq-kl-fullbatch-refine-v1' or unit not in ARMS:
        raise RuntimeError('Fullbatch refinement protocol or arm is invalid')
    if cfg.get('arms') != list(ARMS) or cfg.get('fractions') != list(FRACTIONS):
        raise RuntimeError('The frozen arm or candidate set changed')
    if cfg.get('directions') != list(DIRECTIONS) or cfg.get('gradient_query_budget_per_arm') != 80:
        raise RuntimeError('The frozen gradient direction or query budget changed')
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
    from checkpointing import (atomic_torch_save, optimizer_to_device, to_cpu)
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
    if root in (source, source_mse, diag_root) or source == source_mse:
        raise RuntimeError('Output and immutable source roots must be distinct')
    old = read_json(source / 'protocol.json')
    if old.get('protocol') != 'fastwam-vq-dual-kl-v1' or Path(old.get('remote_root', '')) != source:
        raise RuntimeError('source_root must be the completed original dual-KL protocol')
    if Path(old['source_per_channel_root']) != source_mse:
        raise RuntimeError('source_mse_root differs from the original MSE/NMSE training source')
    if old['calibration']['case_ids'] != cfg['calibration_case_ids']:
        raise RuntimeError('Detailed source C80 contract differs from the new protocol')
    if (old['evaluation']['case_ids'] != cfg['heldout_case_ids']
            or old['evaluation']['seed_indices'] != cfg['heldout_seed_indices']):
        raise RuntimeError('Detailed source held-out contract differs from the new protocol')
    scale = read_json(source / 'results/prepare/loss_scaling.json')
    stats = read_json(source / 'results/fit_stats/summary.json')
    if (scale.get('complete') is not True or scale.get('heldout_used') is not False
            or scale.get('case_ids') != cfg['calibration_case_ids']
            or stats.get('complete') is not True or stats.get('heldout_used') is not False
            or stats.get('calibration_case_ids') != cfg['calibration_case_ids']):
        raise RuntimeError('Calibration-only frozen loss scaling and statistics are required')
    expected_latent_weight = old['training']['rho'] * old['training']['normalization_c']
    if not math.isclose(scale['original_latent_weight'], expected_latent_weight, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError('Original latent objective weight changed')
    decision = read_json(cfg['decision_file'])
    if (decision.get('complete') is not True or decision.get('heldout_used') is not False
            or decision.get('sampling_scope') != 'C80'
            or decision.get('method') != 'full_C80_gradient_actual_loss_selection'):
        raise RuntimeError('The complete heldout-free C80 decision is required')

    expected_order = old['calibration']['update_order']
    if (len(expected_order) != 80 or len(set(expected_order)) != 80
            or set(expected_order) != set(cfg['calibration_case_ids'])):
        raise RuntimeError('The original exact80 training receipt contract changed')
    source_checkpoint = (source_mse / 'results/train_eval/checkpoint.pt' if unit == 'mse_nmse'
                        else source / 'results/dual_kl/checkpoint.pt')
    source_receipt_path = (source_mse / 'results/train_eval/training_receipt.json' if unit == 'mse_nmse'
                           else source / 'results/dual_kl/training_receipt.json')
    source_receipt = read_json(source_receipt_path)
    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
    if unit == 'mse_nmse':
        artifact = Path(os.environ['ARTIFACTS'])
        original_mse_protocol = read_json(artifact / 'source_protocol.json')
        if (read_json(source_mse / 'protocol.json') != original_mse_protocol
                or saved.get('protocol') != original_mse_protocol):
            raise RuntimeError('MSE checkpoint protocol differs from the completed source protocol')
        source_job = old['source_per_channel_job']
    else:
        if saved.get('protocol') != old or saved.get('unit') != 'dual_kl':
            raise RuntimeError('Dual-KL checkpoint binding differs from the old source protocol')
        source_job = source_receipt.get('pbs_jobid')
    if (source_receipt.get('complete') is not True or source_receipt.get('updates') != 80
            or saved.get('updates') != 80 or saved.get('optimizer') is None
            or [item['case_id'] for item in saved.get('history', [])] != expected_order):
        raise RuntimeError('An immutable complete exact80 source checkpoint is required')
    if unit == 'mse_nmse' and source_receipt.get('pbs_jobid') != source_job:
        raise RuntimeError('MSE source checkpoint job receipt differs')
    expected_source_job = cfg['source_mse_job'] if unit == 'mse_nmse' else cfg['source_dual_job']
    if source_job != expected_source_job:
        raise RuntimeError('Source checkpoint PBS job differs from the new protocol')
    if unit == 'dual_kl' and source_receipt.get('unit') not in (None, 'dual_kl'):
        raise RuntimeError('Dual-KL source receipt belongs to another unit')

    diag_protocol = diag_state = direction_summary = None
    if unit == 'dual_kl':
        diag_protocol = read_json(diag_root / 'protocol.json')
        diag_state = read_json(diag_root / 'results/dual_dual/summary.json')
        direction_summary = read_json(diag_root / 'results/dual_dual/direction_summary.json')
        diag_gradient = torch.load(diag_root / 'results/dual_dual/gradient_state.pt',
                                   map_location='cpu', weights_only=True)
        if (diag_protocol.get('protocol') != 'fastwam-vq-kl-direction-v1'
                or Path(diag_protocol.get('source_root', '')) != source
                or diag_protocol.get('case_ids') != cfg['calibration_case_ids']
                or diag_protocol.get('seed_indices') != [0]
                or diag_protocol.get('units', {}).get('dual_dual') !=
                   {'checkpoint_arm': 'dual_kl', 'loss': 'dual_kl'}):
            raise RuntimeError('Completed dual-dual diagnostic protocol binding differs')
        if (diag_state.get('complete') is not True or diag_state.get('unit') != 'dual_dual'
                or diag_state.get('checkpoint_arm') != 'dual_kl' or diag_state.get('loss') != 'dual_kl'
                or diag_state.get('pbs_jobid') != cfg.get('diagnostic_artifact_job')
                or diag_state.get('heldout_used') is not False
                or diag_state.get('gradient_query_count') != 80
                or diag_gradient.get('protocol') != diag_protocol
                or diag_gradient.get('unit') != 'dual_dual'
                or [value['case_id'] for value in diag_gradient.get('baseline', [])] != cfg['calibration_case_ids']
                or len(diag_gradient.get('full_gradient', [])) == 0):
            raise RuntimeError('Completed 80-query dual-dual gradient diagnostic is required')
        diag_baseline_by_case = {row['case_id']: row for row in diag_gradient['baseline']}
        if set(diag_baseline_by_case) != set(cfg['calibration_case_ids']):
            raise RuntimeError('Diagnostic baseline does not cover the complete C80 cohort')
        if (not isinstance(direction_summary, list) or len(direction_summary) != 6
                or diag_state.get('direction_conditions') != direction_summary):
            raise RuntimeError('All six complete C80 direction means are required')
        diag_conditions = {(row['direction'], float(row['fraction'])): row for row in direction_summary}
        if set(diag_conditions) != {(d, f) for d in DIRECTIONS for f in FRACTIONS}:
            raise RuntimeError('Diagnostic candidates do not match the frozen six conditions')
        if any(row.get('queries') != 80 or not math.isfinite(float(row.get('objective', float('nan'))))
               for row in direction_summary):
            raise RuntimeError('Diagnostic selection means must each cover all 80 C80 cases')
    else:
        diag_gradient = None

    first = cfg['calibration_case_ids'][0]
    source_arm = unit
    source_endpoint_dir = source / 'results/calibration' / source_arm / 'queries'
    source_plan = read_json(old['calibration']['plan_file'])
    source_plan_rows = {row['case_id']: row for row in source_plan['inputs']}
    source_endpoints = {}
    for case in cfg['calibration_case_ids']:
        record = read_json(source_endpoint_dir / f'{case}_0.json')
        check_record_binding(record, old['protocol'], source_arm, 'calibration', case, 0,
                             source_plan_rows[case])
        if record.get('plan_row') != source_plan_rows[case]:
            raise RuntimeError('Source C80 endpoint plan binding differs')
        source_endpoints[case] = record

    # Confirm the source training arm and shared teacher files before loading model weights.
    teacher_dirs = {'calibration': Path(old['control_root']) / 'results/teacher/calibration',
                    'heldout': Path(old['clipping_root']) / 'results/teacher'}
    for split in ('calibration',):
        directory = teacher_dirs[split]
        receipt_path = directory / 'receipt.json'
        teacher_receipt = read_json(receipt_path)
        expected_cases = cfg['calibration_case_ids']
        expected_seeds = [0]
        if (teacher_receipt.get('complete') is not True or teacher_receipt.get('queries') != 80
                or teacher_receipt.get('case_ids') != expected_cases
                or teacher_receipt.get('seed_indices') != expected_seeds):
            raise RuntimeError('Shared teacher target coverage differs for ' + split)

    out = root / 'results' / unit
    out.mkdir(parents=True, exist_ok=True)
    state = {'protocol': cfg['protocol'], 'unit': unit, 'pbs_jobid': os.environ['PBS_JOBID'],
             'phase': 'loading', 'complete': False, 'new_teacher_queries': 0}
    accounting_path = out / 'accounting.json'
    accounting = read_json(accounting_path) if accounting_path.is_file() else {
        'protocol': cfg['protocol'], 'unit': unit, 'model_queries': 0, 'source_replay_queries': 0,
        'gradient_queries': 0, 'source_checkpointed_queries': 0, 'candidate_queries': 0, 'deployment_parity_queries': 0,
        'final_calibration_queries': 0, 'heldout_queries': 0}
    if accounting.get('protocol') != cfg['protocol'] or accounting.get('unit') != unit:
        raise RuntimeError('Saved forward accounting belongs to a different protocol')

    def progress(**values):
        state.update(values)
        write_json(out / 'progress.json', {**state, 'accounting': accounting})
        print('KL_FULLBATCH_PROGRESS ' + json.dumps(state, allow_nan=False), flush=True)

    def charge(counter):
        accounting['model_queries'] += 1
        accounting[counter] = accounting.get(counter, 0) + 1
        accounting['last_pbs_jobid'] = os.environ['PBS_JOBID']
        write_json(accounting_path, accounting)
        state['new_model_queries'] = accounting['model_queries']

    data_root, rows, cache, deployment, reference_action = ref.load_inputs(old['source_clipping_protocol'], torch)
    if not (set(cfg['calibration_case_ids']) | set(cfg['heldout_case_ids'])) <= set(rows):
        raise RuntimeError('Frozen plan is missing requested C80 rows')
    runtime = None
    deadline = ref.Deadline()
    started = time.monotonic()
    gradient_new = 0 if unit == 'dual_kl' else 80
    gradient_reused = 80 if unit == 'dual_kl' else 0
    gradient_file = out / 'gradient_state.pt'
    candidate_root = out / 'candidates'
    source_parity = {'passed': False}
    deployment_parity = {'passed': False}
    frozen_checks = {'passed': False}
    try:
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
        if (sum(parameter.numel() for parameter in groups[0]['params']) != 614
                or sum(parameter.numel() for parameter in groups[1]['params']) != 1806863
                or len(groups) != 2 or [group['name'] for group in groups] != ['ratio', 'channel']):
            raise RuntimeError('Trainable parameter scope/order must be 614 ratios then 1,806,863 channel scales')
        if (any(parameter.requires_grad for parameter in runtime.model.parameters())
                or any(parameter.requires_grad for parameter in list(banks.books.values()) + list(banks.logs.values()))):
            raise RuntimeError('Backbone, VQ books, indices, gains and exponents must remain frozen')
        originals = [parameter.detach().clone() for parameter in parameters]
        refinement_updates = 0
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
                raise RuntimeError('Shared teacher binding differs')
            return value

        def terms(action, future, teacher_value):
            return losses(torch, action, future, teacher_value['action'].to(action.device),
                          teacher_value['future'].to(future.device), stats, old['softbin'])

        def source_metric_values(record):
            return {'action_mse': record['action_mse'] if 'action_mse' in record else record['first10motor_mse'],
                    'latent_nmse': record['latent_nmse'] if 'latent_nmse' in record else record['future_latent_nmse'],
                    'action_kl': record['action_kl'], 'latent_kl': record['latent_kl']}

        def check_source_metrics(observed, reference, case):
            ref_values = source_metric_values(reference)
            diffs = {}
            for key in METRICS:
                value = float(observed[key].detach()) if torch.is_tensor(observed[key]) else float(observed[key])
                diffs[key] = abs(value - float(ref_values[key]))
                if not close(value, ref_values[key]):
                    raise RuntimeError('Source C80 metric replay differs: ' + str(case) + '/' + key)
            return diffs

        def make_record(split, case, seed, action, future, teacher_value, counts, *, condition=None,
                        source_record_reused=False, source_record=None, refinement_override=None):
            values = terms(action, future, teacher_value)
            measured = metrics(torch, latent_reference, action, future, teacher_value, old['training'])
            value_map = {key: float(value.detach()) for key, value in values.items()}
            loss = float(objective(values, unit, old, scale).detach())
            record_refinement = refinement_updates if refinement_override is None else refinement_override
            record = {'protocol': cfg['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'],
                      'unit': unit, 'arm': unit, 'split': split, 'updates': 80 + int(record_refinement),
                      'source_updates': 80, 'refinement_updates': int(record_refinement),
                      'total_updates': 80 + int(record_refinement), 'case_id': case,
                      'suite': rows[case]['suite'], 'plan_row': rows[case], 'seed_index': seed,
                      'sampler_seed': rows[case]['sampler_seeds'][seed], 'counts': counts,
                      'source_record_reused': bool(source_record_reused),
                      'source_record_protocol': source_record.get('protocol') if source_record_reused else None,
                      'source_record_arm': source_record.get('arm') if source_record_reused else None,
                      **measured, **value_map, 'objective_loss': loss,
                      'action': action.detach().float().cpu().tolist(),
                      'commands': runtime.command(action.detach().clone()).tolist(),
                      'teacher_action': teacher_value['action'].float().cpu().tolist(),
                      'teacher_commands': runtime.command(teacher_value['action'].clone()).tolist()}
            if condition is not None:
                record['condition'] = condition
            return record

        def validate_new_record(record, split, case, seed):
            expected = {'protocol': cfg['protocol'], 'unit': unit, 'arm': unit, 'split': split,
                        'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                        'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed],
                        'source_updates': 80}
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Saved endpoint binding differs: ' + str((split, case, seed)))
            if record.get('refinement_updates') not in (0, 1) or record.get('updates') != 80 + record['refinement_updates']:
                raise RuntimeError('Saved endpoint update accounting differs')
            if record.get('source_record_reused') is True and record['refinement_updates'] != 0:
                raise RuntimeError('A refined endpoint cannot claim source-record reuse')
            if any(not math.isfinite(float(record.get(key, float('nan'))))
                   for key in (*METRICS, 'objective_loss')):
                raise RuntimeError('Saved endpoint contains nonfinite loss metrics')
            if record.get('source_record_reused') is True and not all(
                    record.get(key) is not None for key in
                    ('source_record_protocol', 'source_record_arm', 'source_pbs_jobid')):
                raise RuntimeError('Reused source endpoint lacks provenance')

        def query(case, seed, purpose_counter):
            deadline.check(90)
            value, metadata = datum(case)
            result = ref.query(runtime, sampler, torch, value, metadata['description'],
                               rows[case]['sampler_seeds'][seed], banks)
            charge(purpose_counter)
            return result

        def differentiable(case, purpose_counter):
            deadline.check(120)
            value, metadata = datum(case)
            runtime.description = metadata['description']
            banks.seen.clear()
            action, counts, future = sampler.differentiable_action(
                runtime, value, rows[case]['sampler_seeds'][0], checkpointed=True, return_future=True)
            charge(purpose_counter)
            if len(banks.seen) != 614 or counts != {'video_steps': 10, 'action_steps': 10}:
                raise RuntimeError('Checkpointed C80 forward did not visit exactly 614 frozen modules')
            return action, future, counts

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

        def copy_source_record(source_record, split, case, seed, condition):
            reused = dict(source_record)
            values = source_metric_values(source_record)
            loss = float(objective({key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()},
                                   unit, old, scale))
            reused.update({'protocol': cfg['protocol'],
                           'unit': unit, 'arm': unit, 'split': split, 'updates': 80,
                           'source_updates': 80, 'refinement_updates': 0, 'total_updates': 80,
                           'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                           'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed],
                           'source_record_reused': True, 'source_record_protocol': source_record['protocol'],
                           'source_record_arm': source_record['arm'], 'source_pbs_jobid': source_record.get('pbs_jobid'),
                           'curated_by_pbs_jobid': os.environ['PBS_JOBID'],
                           'action_mse': float(values['action_mse']), 'latent_nmse': float(values['latent_nmse']),
                           'objective_loss': loss, 'condition': condition})
            if 'counts' not in reused:
                reused['counts'] = {'source_record_reused': True}
            return reused

        # Same-state source replay validates the original endpoint and the exact C80 case2000 forward.
        source_record = source_endpoints[first]
        initial_action, initial_future, initial_counts = query(first, 0, 'source_replay_queries')
        initial_target = target('calibration', first, 0)
        initial_values = terms(initial_action, initial_future, initial_target)
        initial_scalars = {key: float(value.detach()) for key, value in initial_values.items()}
        scalar_diffs = check_source_metrics(initial_scalars, source_record, first)
        if 'action' not in source_record:
            raise RuntimeError('Source C80 endpoint must retain its action output for endpoint replay')
        initial_reference_action = torch.tensor(source_record['action'], dtype=torch.float32)
        source_action_parity = float((initial_action.float().cpu() - initial_reference_action).abs().max())
        if source_action_parity != 0:
            raise RuntimeError('Original source action parity failed')
        source_record_replay = make_record('calibration', first, 0, initial_action, initial_future,
                                          initial_target, initial_counts)
        if any(source_record_replay[key] != source_record[key] for key in ('commands', 'teacher_commands')):
            raise RuntimeError('Original source command serialization parity failed')
        checkpointed_action = checkpointed_future = None
        if unit == 'mse_nmse':
            gradient_state = None
        else:
            checkpointed_action, checkpointed_future, _ = differentiable(first, 'source_checkpointed_queries')
        if unit == 'dual_kl':
            grad_action_diff = float((checkpointed_action.detach().float().cpu() - initial_action.float().cpu()).abs().max())
            grad_future_diff = float((checkpointed_future.detach().float().cpu() - initial_future.float().cpu()).abs().max())
            if grad_action_diff != 0 or grad_future_diff != 0:
                raise RuntimeError('Source no-grad/checkpointed action and future parity failed')
            del checkpointed_action, checkpointed_future
        source_parity = {'passed': True, 'case_id': first, 'scalar_metric_abs_differences': scalar_diffs,
                         'action_reference_max_abs': source_action_parity,
                         'command_serialization_exact': True,
                         'no_grad_checkpointed_action_max_abs': None if unit == 'mse_nmse' else grad_action_diff,
                         'no_grad_checkpointed_future_max_abs': None if unit == 'mse_nmse' else grad_future_diff}
        if unit == 'dual_kl':
            del initial_action, initial_future

        if unit == 'mse_nmse':
            full_gradient = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
            baseline_rows = []
            gradient_cases = []
            if gradient_file.is_file():
                prior = torch.load(gradient_file, map_location='cpu', weights_only=True)
                if (prior.get('protocol') != cfg or prior.get('unit') != unit
                        or prior.get('case_ids') != cfg['calibration_case_ids']):
                    raise RuntimeError('Saved MSE gradient checkpoint belongs to a different protocol')
                baseline_rows, gradient_cases = prior['baseline_rows'], prior['gradient_cases']
                if prior.get('source_parity'):
                    source_parity.update(prior['source_parity'])
                if [row['case_id'] for row in baseline_rows] != [row['case_id'] for row in gradient_cases]:
                    raise RuntimeError('MSE gradient prefix and metric prefix differ')
                if [row['case_id'] for row in gradient_cases] != cfg['calibration_case_ids'][:len(gradient_cases)]:
                    raise RuntimeError('MSE gradient checkpoint case order differs')
                full_gradient = [value.to('cuda') for value in prior['full_gradient']]
                if (len(full_gradient) != len(parameters)
                        or any(value.shape != parameter.shape or not bool(torch.isfinite(value).all())
                               for value, parameter in zip(full_gradient, parameters))):
                    raise RuntimeError('Saved MSE full-C80 gradient tensors differ from source parameter order')
                for row in baseline_rows:
                    for key, value in source_metric_values(source_endpoints[row['case_id']]).items():
                        if not close(row[key], value):
                            raise RuntimeError('Saved MSE gradient metric prefix differs from source C80')
            else:
                full_gradient = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
            cached_gradient_cases = len(gradient_cases)
            progress(phase='mse_full_gradient', gradient_cases=len(gradient_cases),
                     new_model_queries=accounting['model_queries'])
            for case in cfg['calibration_case_ids'][len(gradient_cases):]:
                deadline.check(150)
                action, future, counts = differentiable(case, 'gradient_queries')
                if case == first:
                    action_diff = float((action.detach().float().cpu() - initial_action.float().cpu()).abs().max())
                    future_diff = float((future.detach().float().cpu() - initial_future.float().cpu()).abs().max())
                    if action_diff != 0 or future_diff != 0:
                        raise RuntimeError('Source no-grad/checkpointed action and future parity failed')
                    source_parity['no_grad_checkpointed_action_max_abs'] = action_diff
                    source_parity['no_grad_checkpointed_future_max_abs'] = future_diff
                teacher_value = target('calibration', case, 0)
                values = terms(action, future, teacher_value)
                source_values = source_metric_values(source_endpoints[case])
                diffs = check_source_metrics(values, source_endpoints[case], case)
                for parameter in parameters:
                    parameter.grad = None
                loss = objective(values, unit, old, scale)
                if not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError('MSE/NMSE full-C80 objective is invalid')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
                if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                       for parameter in parameters):
                    raise RuntimeError('MSE/NMSE full-C80 gradient coverage is incomplete or nonfinite')
                for total, parameter in zip(full_gradient, parameters):
                    total.add_(parameter.grad.detach().float(), alpha=1.0 / 80.0)
                baseline_rows.append({'case_id': case, **source_values,
                                      'objective': float(objective(values, unit, old, scale).detach())})
                gradient_cases.append({'case_id': case, 'metric_abs_differences': diffs,
                                       'gradient_norm': float(norm), 'counts': counts})
                for parameter in parameters:
                    parameter.grad = None
                del action, future, values, loss
                if len(gradient_cases) % 5 == 0 or len(gradient_cases) == 80:
                    atomic_torch_save(torch, gradient_file, {
                        'protocol': cfg, 'unit': unit, 'case_ids': cfg['calibration_case_ids'],
                        'baseline_rows': baseline_rows, 'gradient_cases': gradient_cases,
                        'full_gradient': [value.detach().cpu() for value in full_gradient],
                        'gradient_queries_materialized': len(gradient_cases), 'heldout_used': False,
                        'source_parity': source_parity})
                progress(phase='mse_full_gradient', gradient_cases=len(gradient_cases), last_case=case,
                         new_model_queries=accounting['model_queries'])
            if len(gradient_cases) != 80 or len(baseline_rows) != 80:
                raise RuntimeError('MSE/NMSE requires exactly 80 full-C80 backward queries')
            if (source_parity.get('no_grad_checkpointed_action_max_abs') != 0.0
                    or source_parity.get('no_grad_checkpointed_future_max_abs') != 0.0):
                raise RuntimeError('Same-state source action/future parity is missing or nonzero')
            if not gradient_file.is_file():
                atomic_torch_save(torch, gradient_file, {
                    'protocol': cfg, 'unit': unit, 'case_ids': cfg['calibration_case_ids'],
                    'baseline_rows': baseline_rows, 'gradient_cases': gradient_cases,
                    'full_gradient': [value.detach().cpu() for value in full_gradient],
                    'gradient_queries_materialized': 80, 'heldout_used': False,
                    'source_parity': source_parity})
            gradient_new, gradient_reused = 80, 0
            gradient_prefix_queries_reused = cached_gradient_cases
            gradient_queries_computed_this_attempt = 80 - cached_gradient_cases
            gradient_source = {'gradient_queries_new': 80, 'gradient_queries_reused': 0,
                               'cached_gradient_cases': cached_gradient_cases,
                               'gradient_queries_computed_this_attempt': gradient_queries_computed_this_attempt,
                               'gradient_state': str(gradient_file), 'case_count': 80}
        else:
            full_gradient = [value.to('cuda') for value in diag_gradient['full_gradient']]
            if (len(full_gradient) != len(parameters)
                    or any(value.shape != parameter.shape or not bool(torch.isfinite(value).all())
                           for value, parameter in zip(full_gradient, parameters))):
                raise RuntimeError('Reused dual-KL full-C80 gradient tensors differ from source parameter order')
            gradient_source = {'gradient_queries_new': 0, 'gradient_queries_reused': 80,
                               'cached_gradient_cases': 80,
                               'gradient_queries_computed_this_attempt': 0,
                               'gradient_state': str(diag_root / 'results/dual_dual/gradient_state.pt'),
                               'case_count': 80, 'diagnostic_pbs_jobid': diag_state['pbs_jobid']}
            gradient_prefix_queries_reused = 0
            gradient_queries_computed_this_attempt = 0

        if unit == 'mse_nmse':
            del initial_action, initial_future

        baseline_records = []
        baseline_path = candidate_root / 'no_step'
        for case in cfg['calibration_case_ids']:
            path = baseline_path / f'{case}_0.json'
            if path.is_file():
                record = read_json(path)
                if (record.get('protocol') != cfg['protocol'] or record.get('unit') != unit
                        or record.get('condition') != 'no_step' or record.get('case_id') != case
                        or record.get('source_record_reused') is not True):
                    raise RuntimeError('Saved no-step source record binding differs')
                validate_new_record(record, 'calibration', case, 0)
            else:
                source_value = source_endpoints[case]
                record = copy_source_record(source_value, 'calibration', case, 0, 'no_step')
                write_json(path, record)
            validate_new_record(record, 'calibration', case, 0)
            baseline_records.append(record)
        baseline_mean = mean(record['objective_loss'] for record in baseline_records)

        candidate_means = []
        if unit == 'mse_nmse':
            from direction_utils import adam_delta
            for direction in DIRECTIONS:
                delta = adam_delta(torch, full_gradient, groups,
                                   saved['optimizer'] if direction == 'saved_adam' else None)
                for fraction in FRACTIONS:
                    label = label_for(direction, fraction)
                    expected_values = set_delta(delta, fraction)
                    records = []
                    try:
                        for case in cfg['calibration_case_ids']:
                            deadline.check(90)
                            path = candidate_root / label / f'{case}_0.json'
                            if path.is_file():
                                record = read_json(path)
                                if (record.get('protocol') != cfg['protocol'] or record.get('unit') != unit
                                        or record.get('condition') != label or record.get('case_id') != case
                                        or record.get('source_record_reused') is not False
                                        or record.get('candidate_direction') != direction
                                        or float(record.get('candidate_fraction')) != float(fraction)
                                        or record.get('refinement_updates') != 1):
                                    raise RuntimeError('Saved MSE candidate record binding differs')
                                validate_new_record(record, 'calibration', case, 0)
                            else:
                                action, future, counts = query(case, 0, 'candidate_queries')
                                record = make_record('calibration', case, 0, action, future,
                                                     target('calibration', case, 0), counts,
                                                     condition=label, source_record_reused=False,
                                                     refinement_override=1)
                                record.update({'candidate_direction': direction, 'candidate_fraction': fraction,
                                               'optimizer_family': direction})
                                write_json(path, record)
                                del action, future
                            validate_new_record(record, 'calibration', case, 0)
                            records.append(record)
                            progress(phase='mse_candidate_evaluation', condition=label,
                                     candidate_queries=len(records), last_case=case)
                    finally:
                        restore()
                    candidate_means.append({'condition': label, 'direction': direction,
                        'fraction': float(fraction), 'optimizer': direction,
                        'objective_mean': mean(record['objective_loss'] for record in records),
                        'queries': 80, 'source': 'new_actual_loss_C80',
                        'refinement_updates': 1})
                del delta
        else:
            for direction in DIRECTIONS:
                for fraction in FRACTIONS:
                    label = label_for(direction, fraction)
                    summary = next(value for value in direction_summary
                                   if value['direction'] == direction and float(value['fraction']) == fraction)
                    candidate_means.append({'condition': label, 'direction': direction,
                        'fraction': float(fraction), 'optimizer': direction,
                        'objective_mean': float(summary['objective']), 'queries': 80,
                        'source': 'reused_complete_diagnostic_C80',
                        'diagnostic_mean': float(summary['objective']), 'refinement_updates': 1})
        conditions = [{'condition': 'no_step', 'direction': None, 'fraction': 0.0,
                       'optimizer': 'source_checkpoint', 'objective_mean': baseline_mean,
                       'queries': 80, 'source': 'source_C80_endpoint_records',
                       'refinement_updates': 0}] + candidate_means
        if len(candidate_means) != 6 or len(conditions) != 7:
            raise RuntimeError('Selection requires the no-step baseline and all six frozen candidates')
        if unit == 'dual_kl':
            if not close(baseline_mean, diag_state['baseline_metrics']['objective']):
                raise RuntimeError('Reused diagnostic baseline differs from original dual-KL C80 endpoints')
        chosen = min(conditions, key=lambda value: (value['objective_mean'], value['condition']))
        selection = {'complete': True, 'heldout_used': False, 'sampling_scope': 'C80',
                     'method': 'full_C80_gradient_actual_loss_selection',
                     'baseline_objective_mean': baseline_mean, 'baseline': conditions[0],
                     'candidates': candidate_means, 'conditions': conditions,
                     'chosen': dict(chosen), 'selected_from': 'baseline_plus_6_candidates',
                     'candidate_count': 6, 'condition_count': 7}
        write_json(out / 'selection.json', selection)
        progress(phase='selected', selected_condition=chosen['condition'],
                 selected_objective=chosen['objective_mean'])
        heldout_receipt = read_json(teacher_dirs['heldout'] / 'receipt.json')
        if (heldout_receipt.get('complete') is not True or heldout_receipt.get('queries') != 80
                or heldout_receipt.get('case_ids') != cfg['heldout_case_ids']
                or heldout_receipt.get('seed_indices') != cfg['heldout_seed_indices']):
            raise RuntimeError('Shared held-out teacher target coverage differs')

        # Always rebuild from the immutable source checkpoint before materializing the one chosen step.
        restore()
        optimizer = torch.optim.Adam(groups)
        optimizer_family = chosen['optimizer']
        refinement_updates = 0
        if chosen['condition'] == 'no_step':
            optimizer.load_state_dict(saved['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda')
        else:
            direction = chosen['direction']
            fraction = float(chosen['fraction'])
            delta = adam_delta(torch, full_gradient, groups,
                               saved['optimizer'] if direction == 'saved_adam' else None)
            expected_values = set_delta(delta, fraction)
            optimizer = torch.optim.Adam(groups)
            if direction == 'saved_adam':
                optimizer.load_state_dict(saved['optimizer'])
                optimizer_to_device(torch, optimizer, 'cuda')
            for group in optimizer.param_groups:
                group['lr'] = float(group['lr']) * fraction
            restore()
            for parameter, gradient in zip(parameters, full_gradient):
                parameter.grad = gradient.detach().clone()
            optimizer.step()
            banks.project()
            optimizer_max_abs = float(torch.stack([
                (parameter.detach() - expected).abs().max()
                for parameter, expected in zip(parameters, expected_values)]).max())
            if optimizer_max_abs > 1e-6:
                raise RuntimeError('Torch Adam step differs from selected pure Adam candidate: ' + str(optimizer_max_abs))
            with torch.no_grad():
                for parameter, expected in zip(parameters, expected_values):
                    parameter.copy_(expected)
            for parameter in parameters:
                parameter.grad = None
            refinement_updates = 1
            del delta
        total_updates = 80 + refinement_updates

        selected_label = chosen['condition']
        if selected_label == 'no_step':
            restore()

        # Save the refined training and deployment state before any endpoint evaluation.
        from checkpointing import capture_rng
        refinement_history = list(saved.get('history', []))
        if refinement_updates:
            refinement_history.append({'update': 81, 'case_id': 'full_C80_mean_gradient',
                'seed_index': None, 'cases': 80, 'objective_mean_selected': chosen['objective_mean'],
                'direction': chosen['direction'], 'fraction': float(chosen['fraction']),
                'optimizer_family': optimizer_family, 'gradient_queries': 80,
                'pbs_jobid': os.environ['PBS_JOBID']})
        checkpoint_path = out / 'checkpoint.pt'
        atomic_torch_save(torch, checkpoint_path, {'protocol': cfg, 'unit': unit,
            'source_checkpoint': str(source_checkpoint), 'source_updates': 80,
            'refinement_updates': refinement_updates, 'updates': total_updates,
            'history': refinement_history, 'ratios': banks.clip_snapshot(),
            'shadow_log2_scales': banks.scale_snapshot(), 'optimizer': to_cpu(optimizer.state_dict()),
            'rng': capture_rng(torch), 'selection': selection, 'gradient_source': gradient_source,
            'heldout_used': False})

        # Exact live candidate coordinates are also the deployment coordinates.
        uncached = query(first, 0, 'deployment_parity_queries')
        rounding = banks.cache_compensated_weights()
        cached = query(first, 0, 'deployment_parity_queries')
        uncached_action = uncached[0].detach().float().cpu()
        cached_action = cached[0].detach().float().cpu()
        uncached_future = uncached[1].detach().float().cpu()
        cached_future = cached[1].detach().float().cpu()
        cached_parity = {'action_max_abs': float((uncached_action - cached_action).abs().max()),
                         'future_max_abs': float((uncached_future - cached_future).abs().max())}
        runtime.current_arm = unit
        runtime.description = datum(first)[1]['description']
        actual, _ = runtime.infer(datum(first)[0], rows[first]['sampler_seeds'][0], measure=False)
        charge('deployment_parity_queries')
        actual_parity = float((actual.detach().float().cpu() - cached_action).abs().max())
        deployment_parity = {'passed': all(value == 0 for value in cached_parity.values()) and actual_parity == 0,
                             'uncached_cached': cached_parity,
                             'actual_infer_action_max_abs': actual_parity,
                             'source_case_id': first, 'cached_compensation': True}
        if not deployment_parity['passed']:
            raise RuntimeError('Uncached/cached action+future or actual infer parity failed')
        frozen = banks.verify_frozen(deployment['module_masters'])
        frozen_checks = {'passed': all(frozen.get(key) is True for key in
                                       ('codebooks_unchanged', 'indices_unchanged', 'row_gains_unchanged',
                                        'original_exponents_unchanged', 'continuous_scale_metadata_separate')),
                         **frozen}
        if not frozen_checks['passed']:
            raise RuntimeError('Frozen VQ and weight masters changed')
        atomic_torch_save(torch, out / 'deployment.pt', {'protocol': cfg, 'unit': unit,
            'source_checkpoint': str(source_checkpoint), 'source_updates': 80,
            'refinement_updates': refinement_updates, 'updates': total_updates,
            'selection': selection, 'ratios': banks.clip_snapshot(),
            'log2_scales': banks.scale_snapshot(), 'scale_metadata_dtype': 'float32_log2',
            'dynamic_fake_a4': True, 'bf16_kv': True,
            'cached_query_parity': cached_parity, 'actual_infer_action_parity_max_abs': actual_parity,
            'frozen_checks': frozen_checks})
        del uncached, cached, actual

        def endpoint_path(split, case, seed):
            return root / 'results' / split / unit / 'queries' / f'{case}_{seed}.json'

        def validate_and_append(split, case, seed, record, records):
            validate_new_record(record, split, case, seed)
            records.append(record)

        # MSE C80 records are the actual candidate evaluations already used for selection.
        # Dual-KL C80 is rerun on the selected vector and checked against diagnostic per-case metrics.
        for split in ('calibration', 'heldout'):
            records = []
            queries = c80.queries(old, split)
            if len(queries) != 80:
                raise RuntimeError('Final endpoint set must contain exactly 80 queries')
            for index, (case, seed) in enumerate(queries, 1):
                deadline.check(90)
                path = endpoint_path(split, case, seed)
                if path.is_file():
                    record = read_json(path)
                    validate_new_record(record, split, case, seed)
                elif split == 'calibration' and unit == 'mse_nmse':
                    candidate_record = read_json(candidate_root / selected_label / f'{case}_0.json')
                    record = dict(candidate_record)
                    record.update({'split': 'calibration', 'updates': total_updates,
                                   'source_updates': 80, 'refinement_updates': refinement_updates,
                                   'total_updates': total_updates, 'pbs_jobid': candidate_record.get('pbs_jobid', os.environ['PBS_JOBID'])})
                    write_json(path, record)
                elif split == 'calibration' and unit == 'dual_kl' and selected_label == 'no_step':
                    record = copy_source_record(source_endpoints[case], 'calibration', case, seed, 'no_step')
                    reference = diag_baseline_by_case[case]
                    for key, value in source_metric_values(record).items():
                        if not close(value, reference[key]):
                            raise RuntimeError('Reused dual-KL source endpoint differs from diagnostic baseline: '
                                               + str(case) + '/' + key)
                    record['diagnostic_replay_verified'] = True
                    write_json(path, record)
                else:
                    action, future, counts = query(case, seed,
                        'heldout_queries' if split == 'heldout' else 'final_calibration_queries')
                    teacher_value = target(split, case, seed)
                    record = make_record(split, case, seed, action, future, teacher_value, counts,
                                         source_record_reused=False)
                    if split == 'calibration' and unit == 'dual_kl':
                        label = selected_label
                        diag_case = read_json(diag_root / 'results/dual_dual/directions' / label /
                                              f'{case}_0.json')
                        if (diag_case.get('protocol') != diag_protocol['protocol']
                                or diag_case.get('unit') != 'dual_dual'
                                or diag_case.get('direction') != chosen['direction']
                                or float(diag_case.get('fraction')) != float(chosen['fraction'])
                                or diag_case.get('case_id') != case):
                            raise RuntimeError('Selected diagnostic per-case condition binding differs')
                        reference_values = {key: diag_case[key] for key in METRICS}
                        for key in METRICS:
                            if not close(record[key], reference_values[key]):
                                raise RuntimeError('Selected dual-KL C80 replay differs from diagnostic: '
                                                   + str(case) + '/' + key)
                    if split == 'calibration' and unit == 'dual_kl':
                        record['diagnostic_replay_verified'] = True
                    write_json(path, record)
                    del action, future
                if split == 'calibration' and unit == 'dual_kl':
                    if not record.get('diagnostic_replay_verified'):
                        raise RuntimeError('Dual-KL selected C80 endpoint lacks diagnostic replay verification')
                validate_and_append(split, case, seed, record, records)
                progress(phase='endpoint_evaluation', split=split, queries=index,
                         last_case=case, last_seed=seed)
            if len(records) != 80:
                raise RuntimeError('Final split does not contain exactly 80 query records')
            summary_values = c80.aggregate_records(records, old, split)
            for key in ('action_kl', 'latent_kl', 'objective_loss'):
                summary_values['overall'][key] = mean(record[key] for record in records)
                for suite in ref.SUITES:
                    suite_records = [record for record in records if record['suite'] == suite]
                    summary_values['by_suite'][suite][key] = mean(record[key] for record in suite_records)
            write_json(root / 'results' / split / unit / 'summary.json', {
                'protocol': cfg['protocol'], 'complete': True, 'unit': unit, 'arm': unit,
                'split': split, 'pbs_jobid': os.environ['PBS_JOBID'],
                'source_updates': 80, 'refinement_updates': refinement_updates,
                'total_updates': total_updates, 'updates': total_updates,
                'source_record_reused_queries': sum(record.get('source_record_reused') is True for record in records),
                **summary_values})

        receipt = {'protocol': cfg['protocol'], 'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
                   'unit': unit, 'source_updates': 80, 'refinement_updates': refinement_updates,
                   'total_updates': total_updates, 'updates': total_updates,
                   'source_checkpoint': str(source_checkpoint), 'source_checkpoint_jobid': source_job,
                   'gradient_query_budget': 80, 'gradient_queries_new': gradient_new,
                   'gradient_queries_reused': gradient_reused, 'gradient_source': gradient_source,
                   'gradient_prefix_queries_reused': gradient_prefix_queries_reused,
                   'gradient_queries_computed_this_attempt': gradient_queries_computed_this_attempt,
                   'calibration_queries': 80, 'heldout_queries': 80, 'evaluation_queries': 160,
                   'selection': selection, 'source_parity_passed': source_parity['passed'],
                   'source_original_endpoint_replay': source_parity,
                   'deployment_parity_passed': deployment_parity['passed'],
                   'deployment_parity': deployment_parity, 'frozen_verified': frozen_checks['passed'],
                   'frozen_checks': frozen_checks, 'optimizer_family': optimizer_family,
                   'selected_adam_step_max_abs': optimizer_max_abs if refinement_updates else 0.0,
                   'new_teacher_queries': 0, 'new_model_queries': accounting['model_queries'],
                   'forward_accounting': accounting, 'candidate_records_root': str(candidate_root),
                   'checkpoint': str(checkpoint_path), 'deployment': str(out / 'deployment.pt'),
                   'heldout_used': False, 'seconds_inside_worker': time.monotonic() - started}
        write_json(out / 'training_receipt.json', receipt)
        write_json(out / 'summary.json', {**receipt, 'phase': 'complete'})
        progress(phase='complete', complete=True, source_updates=80,
                 refinement_updates=refinement_updates, total_updates=total_updates)
        print('KL_FULLBATCH_COMPLETE', flush=True)
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
    cfg = read_json(args.protocol)
    return run(cfg, args.unit)


if __name__ == '__main__':
    raise SystemExit(main())
