"""C80-selected, fixed-D ratio-only refinement for the original MSE80 checkpoint."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import socket
import time


ARMS = ('mse_nmse', 'dual_kl')
DIRECTIONS = ('saved_adam', 'fresh_adam')
FRACTIONS = (1.0, 0.5, 0.25)
METRICS = ('action_mse', 'latent_nmse', 'action_kl', 'latent_kl')


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('Required file is missing: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


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


def close(a, b, *, rel=1e-6, abs_=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def mean(values):
    values = list(values)
    if not values:
        raise RuntimeError('Cannot summarize an empty C80 cohort')
    return sum(float(value) for value in values) / len(values)


def label_for(direction, fraction):
    return direction + '_f' + str(float(fraction)).replace('.', 'p')


def record_values(record):
    return {
        'action_mse': record['action_mse'] if 'action_mse' in record else record['first10motor_mse'],
        'latent_nmse': record['latent_nmse'] if 'latent_nmse' in record else record['future_latent_nmse'],
        'action_kl': record['action_kl'],
        'latent_kl': record['latent_kl'],
    }


def objective(values, unit, old, scale):
    weight = old['training']['rho'] * old['training']['normalization_c']
    if unit == 'mse_nmse':
        return values['action_mse'] + weight * values['latent_nmse']
    return (scale['action_kl_coefficient'] * values['action_kl']
            + weight * scale['latent_kl_coefficient'] * values['latent_kl'])


def check_source_record(record, old, case, row):
    expected = {'protocol': old['protocol'], 'arm': 'mse_nmse', 'split': 'calibration',
                'updates': 80, 'case_id': case, 'suite': row['suite'], 'plan_row': row,
                'seed_index': 0, 'sampler_seed': row['sampler_seeds'][0]}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Original MSE C80 endpoint binding differs: ' + str(case))
    if not all(key in record for key in ('action', 'commands', 'teacher_action', 'teacher_commands')):
        raise RuntimeError('Original MSE C80 endpoint lacks raw matrices: ' + str(case))
    if any(not math.isfinite(float(record_values(record)[key])) for key in METRICS):
        raise RuntimeError('Original MSE C80 endpoint contains a nonfinite loss: ' + str(case))


def sliced_ratio_optimizer(source_optimizer, ratio_count):
    groups = source_optimizer.get('param_groups', [])
    if (len(groups) != 2 or len(groups[0].get('params', [])) != ratio_count
            or len(groups[1].get('params', [])) != ratio_count):
        raise RuntimeError('MSE checkpoint must contain ratio then channel Adam groups')
    ratio_group = copy.deepcopy(groups[0])
    ratio_ids = ratio_group['params']
    state = source_optimizer.get('state', {})
    return {'state': {key: copy.deepcopy(state[key]) for key in ratio_ids if key in state},
            'param_groups': [ratio_group]}


def run(cfg, unit):
    compute_guard()
    if cfg.get('protocol') != 'fastwam-vq-kl-ratio-refine-v1' or unit not in ARMS:
        raise RuntimeError('Ratio-refine protocol or unit is invalid')
    if (cfg.get('arms') != list(ARMS) or cfg.get('directions') != list(DIRECTIONS)
            or cfg.get('fractions') != list(FRACTIONS)
            or cfg.get('gradient_query_budget_per_arm') != 80
            or cfg.get('gradient_queries_new_per_arm') != 0
            or cfg.get('gradient_queries_reused_per_arm') != 80
            or cfg.get('maximum_refinement_updates') != 1
            or cfg.get('checkpoint_arm') != 'mse_nmse'):
        raise RuntimeError('Frozen arms, candidates, gradient budget, or update limit changed')
    if cfg.get('calibration_case_ids') != list(range(2000, 2080)):
        raise RuntimeError('The original C80 cohort changed')
    if (cfg.get('heldout_case_ids') != list(range(3000, 3040))
            or cfg.get('heldout_seed_indices') != [0, 1]):
        raise RuntimeError('The original paired held-out cohort changed')
    if cfg.get('fixed_D') != 'Retain exact original MSE80 per-channel log2D values; never reset or update D':
        raise RuntimeError('The frozen per-channel D contract changed')

    import numpy as np
    import torch
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
    source_root = Path(cfg['source_root'])
    source_mse_root = Path(cfg['source_mse_root'])
    if len({root, source_root, source_mse_root}) != 3:
        raise RuntimeError('Output, dual-KL source, and MSE source roots must be distinct')
    old = read_json(source_root / 'protocol.json')
    if (old.get('protocol') != 'fastwam-vq-dual-kl-v1'
            or Path(old.get('remote_root', '')) != source_root
            or Path(old.get('source_per_channel_root', '')) != source_mse_root):
        raise RuntimeError('Detailed dual-KL source protocol binding differs')
    if (old['calibration']['case_ids'] != cfg['calibration_case_ids']
            or old['evaluation']['case_ids'] != cfg['heldout_case_ids']
            or old['evaluation']['seed_indices'] != cfg['heldout_seed_indices']):
        raise RuntimeError('Source calibration or held-out cohort differs from the ratio-refine protocol')
    update_order = old['calibration']['update_order']
    if (len(update_order) != 80 or len(set(update_order)) != 80
            or set(update_order) != set(cfg['calibration_case_ids'])):
        raise RuntimeError('Original MSE80 update order is incomplete')

    artifact = Path(os.environ['ARTIFACTS'])
    source_protocol = read_json(artifact / 'source_protocol.json')
    if (source_protocol != read_json(source_mse_root / 'protocol.json')
            or source_protocol.get('protocol') != 'fastwam-vq-per-channel-scaling-clipping-v1'):
        raise RuntimeError('Copied and immutable MSE source protocols differ')
    source_checkpoint = source_mse_root / 'results/train_eval/checkpoint.pt'
    source_receipt = read_json(source_mse_root / 'results/train_eval/training_receipt.json')
    saved = torch.load(source_checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('protocol') != source_protocol or saved.get('updates') != 80
            or saved.get('optimizer') is None or source_receipt.get('complete') is not True
            or source_receipt.get('updates') != 80
            or source_receipt.get('pbs_jobid') != cfg['source_mse_job']
            or source_receipt.get('update_order') != update_order
            or [row.get('case_id') for row in saved.get('history', [])] != update_order
            or any(row.get('pbs_jobid') != cfg['source_mse_job'] for row in saved.get('history', []))):
        raise RuntimeError('Immutable original MSE final80 checkpoint/history/receipt differs')
    if 'ratios' not in saved or not (saved.get('shadow_log2_scales') is not None or saved.get('log2_scales') is not None):
        raise RuntimeError('Original MSE checkpoint lacks ratio or per-channel D values')

    decision = read_json(cfg['decision_file'])
    if (decision.get('complete') is not True or decision.get('heldout_used') is not False
            or decision.get('sampling_scope') != 'C80'
            or decision.get('method') != 'fixed_D_ratio_only_update_subspace'):
        raise RuntimeError('Complete heldout-free ratio-only decision is required')

    scaling = read_json(source_root / 'results/prepare/loss_scaling.json')
    stats = read_json(source_root / 'results/fit_stats/summary.json')
    if (scaling.get('complete') is not True or scaling.get('heldout_used') is not False
            or scaling.get('case_ids') != cfg['calibration_case_ids']
            or stats.get('complete') is not True or stats.get('heldout_used') is not False
            or stats.get('calibration_case_ids') != cfg['calibration_case_ids']):
        raise RuntimeError('Frozen calibration-only KL scaling/statistics are required')
    if not close(scaling['original_latent_weight'],
                 old['training']['rho'] * old['training']['normalization_c'], rel=0.0, abs_=1e-15):
        raise RuntimeError('Original latent objective coefficient changed')

    gradient_path = Path(cfg['gradient_sources'][unit])
    expected_gradient_job = cfg['gradient_source_jobs'][unit]
    if unit == 'mse_nmse':
        gradient_protocol = read_json(Path(cfg['matched_dense_root']) / 'protocol.json')
        gradient_summary = read_json(Path(cfg['matched_dense_root']) / 'results/mse_nmse/summary.json')
        gradient_state = torch.load(gradient_path, map_location='cpu', weights_only=True)
        if (gradient_summary.get('complete') is not True
                or gradient_summary.get('pbs_jobid') != expected_gradient_job
                or gradient_state.get('protocol') != gradient_protocol
                or gradient_state.get('unit') != 'mse_nmse'
                or gradient_state.get('case_ids') != cfg['calibration_case_ids']
                or gradient_state.get('gradient_queries_materialized') != 80
                or gradient_state.get('heldout_used') is not False
                or not gradient_state.get('source_parity', {}).get('passed')):
            raise RuntimeError('Completed MSE full-C80 gradient cache binding differs')
        baseline_key, cases_key = 'baseline_rows', 'gradient_cases'
        baseline_rows = gradient_state.get(baseline_key, [])
        gradient_cases = gradient_state.get(cases_key, [])
        if len(baseline_rows) != 80 or len(gradient_cases) != 80:
            raise RuntimeError('MSE gradient cache does not contain complete baseline and case rows')
    else:
        diagnostic_root = Path(cfg['diagnostic_root'])
        gradient_protocol = read_json(diagnostic_root / 'protocol.json')
        gradient_summary = read_json(diagnostic_root / 'results/mse_dual/summary.json')
        gradient_state = torch.load(gradient_path, map_location='cpu', weights_only=True)
        if (gradient_protocol.get('protocol') != 'fastwam-vq-kl-direction-v1'
                or Path(gradient_protocol.get('source_root', '')) != source_root
                or gradient_protocol.get('case_ids') != cfg['calibration_case_ids']
                or gradient_protocol.get('units', {}).get('mse_dual') !=
                   {'checkpoint_arm': 'mse_nmse', 'loss': 'dual_kl'}
                or gradient_summary.get('complete') is not True
                or gradient_summary.get('unit') != 'mse_dual'
                or gradient_summary.get('checkpoint_arm') != 'mse_nmse'
                or gradient_summary.get('loss') != 'dual_kl'
                or gradient_summary.get('pbs_jobid') != expected_gradient_job
                or gradient_summary.get('gradient_query_count') != 80
                or gradient_state.get('protocol') != gradient_protocol
                or gradient_state.get('unit') != 'mse_dual'
                or gradient_state.get('heldout_used') is not False):
            raise RuntimeError('Completed MSE-start dual-KL gradient cache binding differs')
        baseline_key, cases_key = 'baseline', 'parity'
        baseline_rows = gradient_state.get(baseline_key, [])
        gradient_cases = gradient_state.get(cases_key, [])
        if len(baseline_rows) != 80 or len(gradient_cases) != 80:
            raise RuntimeError('Dual-KL gradient cache does not contain complete baseline and parity rows')
    if ([row.get('case_id') for row in baseline_rows] != cfg['calibration_case_ids']
            or [row.get('case_id') for row in gradient_cases] != cfg['calibration_case_ids']
            or len(gradient_state.get('full_gradient', [])) != 1228):
        raise RuntimeError('Cached gradient case order or ratio+channel tensor count differs')
    full_gradient_cpu = gradient_state['full_gradient']
    if any(not bool(torch.isfinite(value).all()) for value in full_gradient_cpu):
        raise RuntimeError('Cached full-C80 gradient contains nonfinite values')

    first = cfg['calibration_case_ids'][0]
    source_endpoint_root = source_root / 'results/calibration/mse_nmse/queries'
    source_endpoints = {}
    for case in cfg['calibration_case_ids']:
        endpoint = read_json(source_endpoint_root / f'{case}_0.json')
        check_source_record(endpoint, old, case, endpoint['plan_row'])
        source_endpoints[case] = endpoint
    for gradient_row, case in zip(baseline_rows, cfg['calibration_case_ids']):
        source_values = record_values(source_endpoints[case])
        if any(not close(gradient_row[key], source_values[key]) for key in METRICS):
            raise RuntimeError('Cached gradient baseline is not bound to original MSE80 C80: ' + str(case))
    for row in gradient_cases:
        if unit == 'mse_nmse':
            if row.get('case_id') not in cfg['calibration_case_ids']:
                raise RuntimeError('Cached MSE gradient case binding differs')
        elif row.get('action_max_abs') is None or row.get('future_max_abs') is None:
            raise RuntimeError('Cached dual-KL source parity row lacks action/future differences')

    source_optimizer = sliced_ratio_optimizer(saved['optimizer'], 614)
    source_ratio_group = source_optimizer['param_groups'][0]
    if (not close(source_ratio_group.get('lr'), 0.01, rel=0.0, abs_=1e-12)
            or tuple(source_ratio_group.get('betas', ())) != (0.9, 0.999)
            or not close(source_ratio_group.get('eps'), 1e-8, rel=0.0, abs_=1e-15)
            or source_ratio_group.get('weight_decay', 0) != 0
            or source_ratio_group.get('amsgrad', False)
            or source_ratio_group.get('maximize', False)):
        raise RuntimeError('Ratio Adam must be the source standard Adam group at lr=.01')

    teacher_dirs = {'calibration': Path(old['control_root']) / 'results/teacher/calibration',
                    'heldout': Path(old['clipping_root']) / 'results/teacher'}
    calibration_teacher_receipt = read_json(teacher_dirs['calibration'] / 'receipt.json')
    if (calibration_teacher_receipt.get('complete') is not True
            or calibration_teacher_receipt.get('queries') != 80
            or calibration_teacher_receipt.get('case_ids') != cfg['calibration_case_ids']
            or calibration_teacher_receipt.get('seed_indices') != [0]):
        raise RuntimeError('Shared calibration teacher target coverage differs')

    data_root, rows, cache, deployment, _ = ref.load_inputs(old['source_clipping_protocol'], torch)
    if not (set(cfg['calibration_case_ids']) | set(cfg['heldout_case_ids'])) <= set(rows):
        raise RuntimeError('Frozen source plan lacks required C80 or held-out rows')
    for case in cfg['calibration_case_ids']:
        check_source_record(source_endpoints[case], old, case, rows[case])
    out = root / 'results' / unit
    out.mkdir(parents=True, exist_ok=True)
    accounting_path = out / 'accounting.json'
    accounting = read_json(accounting_path) if accounting_path.is_file() else {
        'protocol': cfg['protocol'], 'unit': unit, 'model_queries': 0,
        'source_replay_queries': 0, 'candidate_queries': 0,
        'source_cache_parity_queries': 0, 'selected_uncached_parity_queries': 0,
        'selected_cached_parity_queries': 0, 'actual_infer_parity_queries': 0,
        'heldout_queries': 0}
    if accounting.get('protocol') != cfg['protocol'] or accounting.get('unit') != unit:
        raise RuntimeError('Persisted forward accounting belongs to another experiment')
    queries_at_start = int(accounting.get('model_queries', 0))
    state = {'protocol': cfg['protocol'], 'unit': unit, 'pbs_jobid': os.environ['PBS_JOBID'],
             'phase': 'loading', 'complete': False, 'gradient_queries_new': 0,
             'gradient_queries_reused': 80, 'new_backward_queries': 0,
             'new_teacher_queries': 0, 'new_model_queries': 0}

    def progress(**values):
        state.update(values)
        write_json(out / 'progress.json', {**state, 'forward_accounting': accounting})
        print('RATIO_REFINE_PROGRESS ' + json.dumps(state, allow_nan=False), flush=True)

    def charge(counter):
        accounting['model_queries'] = accounting.get('model_queries', 0) + 1
        accounting[counter] = accounting.get(counter, 0) + 1
        accounting['last_pbs_jobid'] = os.environ['PBS_JOBID']
        state['new_model_queries'] = accounting['model_queries']
        write_json(accounting_path, accounting)

    runtime = None
    source_parity = {'passed': False}
    deployment_parity = {'passed': False}
    frozen_checks = {'passed': False}
    selection = {}
    chosen = {}
    refinement_updates = 0
    total_updates = 80
    attempt_reused_candidate_records = 0
    attempt_reused_endpoint_records = 0
    attempt_reused_source_records = 0
    deadline = ref.Deadline()
    started = time.monotonic()
    try:
        progress(phase='loading_source_state')
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        banks = ChannelBanks(runtime, old['calibration']['bank_directory'],
                             deployment['module_masters'], old['clipping'], old['scaling'])
        banks.restore_clips(saved['ratios'])
        source_logs = saved.get('shadow_log2_scales', saved.get('log2_scales'))
        banks.restore_scales(source_logs)
        for value in banks.shadows.values():
            value.requires_grad_(False)
        groups = banks.optimizer_groups(old['training'], old['scaling'])
        if len(groups) != 2:
            raise RuntimeError('Expected source optimizer groups: ratios followed by fixed D')
        for group, name in zip(groups, ('ratio', 'channel')):
            group['name'] = name
        ratio_params = list(groups[0]['params'])
        ratio_group = {'params': ratio_params, 'lr': float(source_ratio_group['lr'])}
        if (sum(parameter.numel() for parameter in ratio_params) != 614
                or sum(parameter.numel() for parameter in groups[1]['params']) != 1806863
                or len(ratio_params) != 614
                or len(full_gradient_cpu) != 1228
                or any(parameter.requires_grad for parameter in runtime.model.parameters())
                or any(parameter.requires_grad for parameter in list(banks.books.values()) + list(banks.logs.values()))):
            raise RuntimeError('Only 614 ratios may be optimized; model/VQ and 1,806,863 D coordinates are frozen')
        if not torch.equal(torch.cat([value.detach().reshape(-1).cpu() for value in banks.shadows.values()]),
                           torch.cat([value.detach().reshape(-1).cpu() for value in source_logs.values()])):
            raise RuntimeError('Restored MSE80 D values differ from immutable checkpoint')
        originals = [parameter.detach().clone() for parameter in ratio_params]
        original_d = {name: value.detach().clone() for name, value in banks.shadows.items()}
        gradient = [value.to('cuda', dtype=torch.float32) for value in full_gradient_cpu[:len(ratio_params)]]
        if (len(gradient) != len(ratio_params)
                or any(value.shape != parameter.shape or not bool(torch.isfinite(value).all())
                       for value, parameter in zip(gradient, ratio_params))):
            raise RuntimeError('The first 614 cached gradients do not match ratio parameter order/shapes')

        data = {}

        def datum(case):
            if case not in data:
                observation, metadata = calibration_source.load_observation(np, data_root, rows[case])
                data[case] = runtime.datum(observation), metadata
            return data[case]

        def target(split, case, seed):
            saved_target = torch.load(teacher_dirs[split] / f'{case}_{seed}.pt',
                                      map_location='cpu', weights_only=True)
            if (saved_target['plan_row'] != rows[case]
                    or saved_target['sampler_seed'] != rows[case]['sampler_seeds'][seed]):
                raise RuntimeError('Shared teacher target plan/seed binding differs')
            return saved_target

        def terms(action, future, saved_target):
            return losses(torch, action, future, saved_target['action'].to(action.device),
                          saved_target['future'].to(future.device), stats, old['softbin'])

        def values_for(action, future, saved_target):
            return {key: float(value.detach()) for key, value in terms(action, future, saved_target).items()}

        def make_record(split, case, seed, action, future, saved_target, counts,
                        condition, update_count, *, reused=False, source_record=None):
            values = terms(action, future, saved_target)
            measured = metrics(torch, latent_reference, action, future, saved_target, old['training'])
            value_map = {key: float(value.detach()) for key, value in values.items()}
            action_commands = runtime.command(action.detach().clone())
            teacher_commands = runtime.command(saved_target['action'].clone())
            record = {
                'protocol': cfg['protocol'], 'unit': unit, 'arm': unit,
                'split': split, 'case_id': case, 'suite': rows[case]['suite'],
                'plan_row': rows[case], 'seed_index': seed,
                'sampler_seed': rows[case]['sampler_seeds'][seed],
                'pbs_jobid': os.environ['PBS_JOBID'],
                'source_updates': 80, 'refinement_updates': update_count,
                'total_updates': 80 + update_count, 'updates': 80 + update_count,
                'checkpoint_arm': 'mse_nmse', 'loss': unit,
                'source_checkpoint_pbs_jobid': cfg['source_mse_job'],
                'gradient_source_pbs_jobid': expected_gradient_job,
                'ratio_only': True, 'D_unchanged': True,
                'condition': condition, 'source_record_reused': reused,
                'counts': counts,
                **measured, **value_map,
                'objective_loss': float(objective(value_map, unit, old, scaling)),
                'action': action.detach().float().cpu().tolist(),
                'commands': action_commands.tolist(),
                'teacher_action': saved_target['action'].float().tolist(),
                'teacher_commands': teacher_commands.tolist(),
            }
            if reused:
                record.update({
                    'pbs_jobid': source_record.get('pbs_jobid'),
                    'source_record_protocol': source_record.get('protocol'),
                    'source_record_arm': source_record.get('arm'),
                    'source_record_pbs_jobid': source_record.get('pbs_jobid'),
                    'source_record_path': str(source_endpoint_root / f'{case}_0.json'),
                    'curated_by_pbs_jobid': os.environ['PBS_JOBID'],
                })
            return record

        def source_reuse_record(case):
            source = dict(source_endpoints[case])
            values = record_values(source)
            source.update({
                'protocol': cfg['protocol'], 'unit': unit, 'arm': unit,
                'split': 'calibration', 'source_updates': 80,
                'refinement_updates': 0, 'total_updates': 80, 'updates': 80,
                'checkpoint_arm': 'mse_nmse', 'loss': unit,
                'source_checkpoint_pbs_jobid': cfg['source_mse_job'],
                'gradient_source_pbs_jobid': expected_gradient_job,
                'ratio_only': True, 'D_unchanged': True,
                'condition': 'no_step', 'source_record_reused': True,
                'source_record_protocol': source_endpoints[case].get('protocol'),
                'source_record_arm': source_endpoints[case].get('arm'),
                'source_record_pbs_jobid': source_endpoints[case].get('pbs_jobid'),
                'source_record_path': str(source_endpoint_root / f'{case}_0.json'),
                'curated_by_pbs_jobid': os.environ['PBS_JOBID'],
                'objective_loss': float(objective(values, unit, old, scaling)),
                **values,
            })
            return source

        def validate_record(record, split, case, seed, condition):
            expected = {'protocol': cfg['protocol'], 'unit': unit, 'arm': unit,
                        'split': split, 'case_id': case, 'suite': rows[case]['suite'],
                        'plan_row': rows[case], 'seed_index': seed,
                        'sampler_seed': rows[case]['sampler_seeds'][seed],
                        'source_updates': 80, 'checkpoint_arm': 'mse_nmse',
                        'loss': unit, 'condition': condition, 'ratio_only': True,
                        'D_unchanged': True}
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Saved ratio-refine query binding differs: ' + str((split, case, seed, condition)))
            expected_updates = 0 if condition == 'no_step' else 1
            if (record.get('refinement_updates') != expected_updates
                    or record.get('total_updates') != 80 + expected_updates
                    or record.get('updates') != 80 + expected_updates):
                raise RuntimeError('Saved ratio-refine update accounting differs')
            values = record_values(record)
            if any(not math.isfinite(float(values[key])) for key in METRICS) or not math.isfinite(
                    float(record.get('objective_loss', float('nan')))):
                raise RuntimeError('Saved ratio-refine record contains nonfinite losses')
            if not all(key in record for key in ('action', 'commands', 'teacher_action', 'teacher_commands')):
                raise RuntimeError('Saved ratio-refine record lacks required raw matrices')
            if not close(record['objective_loss'], objective(values, unit, old, scaling)):
                raise RuntimeError('Saved ratio-refine objective does not match its loss terms')
            if condition == 'no_step' and split == 'calibration':
                if (record.get('source_record_reused') is not True
                        or record.get('source_record_pbs_jobid') != record.get('pbs_jobid')
                        or record.get('source_record_protocol') != old['protocol']
                        or record.get('source_record_arm') != 'mse_nmse'
                        or record.get('curated_by_pbs_jobid') is None):
                    raise RuntimeError('No-step output lacks original endpoint provenance')
            elif record.get('source_record_reused') is not False or not record.get('pbs_jobid'):
                raise RuntimeError('Measured candidate/heldout record provenance differs')

        def query(case, seed, counter):
            deadline.check(90)
            value, metadata = datum(case)
            banks.seen.clear()
            with torch.no_grad():
                result = ref.query(runtime, sampler, torch, value, metadata['description'],
                                   rows[case]['sampler_seeds'][seed], banks)
            charge(counter)
            if len(banks.seen) != 614:
                raise RuntimeError('Query did not cover all 614 dynamic A4 modules')
            return result

        def restore_ratios():
            with torch.no_grad():
                for parameter, original in zip(ratio_params, originals):
                    parameter.copy_(original)

        def set_candidate(delta, fraction):
            expected = []
            lower, upper = old['clipping']['minimum_ratio'], old['clipping']['maximum_ratio']
            with torch.no_grad():
                for parameter, original, change in zip(ratio_params, originals, delta):
                    value = (original + float(fraction) * change).clamp(lower, upper)
                    parameter.copy_(value)
                    expected.append(value.detach().clone())
            return expected

        # Validate same-state output against the original MSE80 C80 endpoint before caching.
        progress(phase='source_endpoint_replay')
        source_record = source_endpoints[first]
        source_action, source_future, source_counts = query(first, 0, 'source_replay_queries')
        source_target = target('calibration', first, 0)
        source_live = make_record('calibration', first, 0, source_action, source_future,
                                  source_target, source_counts, 'no_step', 0)
        source_diffs = {}
        for key in METRICS:
            source_diffs[key] = abs(float(source_live[key]) - float(record_values(source_record)[key]))
            if not close(source_live[key], record_values(source_record)[key]):
                raise RuntimeError('Source MSE80 first-case loss replay differs: ' + key)
        source_action_diff = float((source_action.detach().float().cpu()
                                     - torch.as_tensor(source_record['action'], dtype=torch.float32)).abs().max())
        source_command = runtime.command(source_action.detach().clone())
        source_command_diff = float(np.max(np.abs(np.asarray(source_command, dtype=np.float32)
                                                   - np.asarray(source_record['commands'], dtype=np.float32))))
        if source_action_diff != 0.0 or source_command_diff != 0.0:
            raise RuntimeError('Source MSE80 first-case action/command replay is not exact')
        source_parity = {'passed': True, 'case_id': first,
                         'source_endpoint_protocol': source_record['protocol'],
                         'source_endpoint_arm': source_record['arm'],
                         'source_endpoint_pbs_jobid': source_record.get('pbs_jobid'),
                         'loss_abs_differences': source_diffs,
                         'action_max_abs': source_action_diff,
                         'commands_max_abs': source_command_diff,
                         'gradient_baseline_cases_verified': 80,
                         'gradient_source_unit': gradient_state.get('unit'),
                         'gradient_source_pbs_jobid': expected_gradient_job}
        source_uncached_action = source_action.detach().float().cpu().clone()
        source_uncached_future = source_future.detach().float().cpu().clone()
        del source_live, source_action, source_future

        # The source baseline is reused; six ratio-only candidates are measured afresh.
        candidate_root = out / 'candidates'
        candidates = []
        baseline_records = []
        baseline_mean = mean(objective(record_values(source_endpoints[case]), unit, old, scaling)
                             for case in cfg['calibration_case_ids'])
        no_step_dir = candidate_root / 'no_step'
        for case in cfg['calibration_case_ids']:
            path = no_step_dir / f'{case}_0.json'
            if path.is_file():
                record = read_json(path)
                validate_record(record, 'calibration', case, 0, 'no_step')
                attempt_reused_candidate_records += 1
            else:
                record = source_reuse_record(case)
                validate_record(record, 'calibration', case, 0, 'no_step')
                write_json(path, record)
                attempt_reused_source_records += 1
            baseline_records.append(record)
        if not close(mean(record['objective_loss'] for record in baseline_records), baseline_mean):
            raise RuntimeError('No-step source C80 objective mean is inconsistent')

        for direction in DIRECTIONS:
            delta = adam_delta(torch, gradient, [ratio_group],
                               source_optimizer if direction == 'saved_adam' else None)
            for fraction in FRACTIONS:
                label = label_for(direction, fraction)
                expected_values = set_candidate(delta, fraction)
                records = []
                condition_dir = candidate_root / label
                maximum_abs = None
                try:
                    for case in cfg['calibration_case_ids']:
                        deadline.check(90)
                        path = condition_dir / f'{case}_0.json'
                        if path.is_file():
                            record = read_json(path)
                            validate_record(record, 'calibration', case, 0, label)
                            if (record.get('direction') != direction
                                    or float(record.get('fraction')) != float(fraction)):
                                raise RuntimeError('Cached ratio-only candidate metadata differs')
                            attempt_reused_candidate_records += 1
                        else:
                            action, future, counts = query(case, 0, 'candidate_queries')
                            teacher_value = target('calibration', case, 0)
                            record = make_record('calibration', case, 0, action, future,
                                                 teacher_value, counts, label, 1)
                            record.update({'direction': direction, 'fraction': float(fraction),
                                           'optimizer': direction,
                                           'candidate_source': 'fixed_D_ratio_only_actual_C80'})
                            validate_record(record, 'calibration', case, 0, label)
                            write_json(path, record)
                            del action, future
                        records.append(record)
                        progress(phase='candidate_C80_evaluation', condition=label,
                                 candidate_queries=len(records), last_case=case)
                    if len(records) == 80:
                        maximum_abs = float(torch.stack([
                            (parameter.detach() - expected).abs().max()
                            for parameter, expected in zip(ratio_params, expected_values)]).max())
                finally:
                    restore_ratios()
                if len(records) != 80:
                    raise RuntimeError('A ratio-only candidate is missing C80 records')
                if maximum_abs is None or maximum_abs != 0.0:
                    raise RuntimeError('Candidate parameter vector drifted during evaluation')
                candidates.append({
                    'condition': label, 'direction': direction,
                    'fraction': float(fraction), 'optimizer': direction,
                    'objective_mean': mean(record['objective_loss'] for record in records),
                    'queries': 80, 'source': 'fixed_D_ratio_only_actual_C80',
                    'pbs_jobids': sorted({record['pbs_jobid'] for record in records}),
                    'refinement_updates': 1,
                })
            del delta

        baseline = {'condition': 'no_step', 'direction': None, 'fraction': 0.0,
                    'optimizer': 'source_checkpoint', 'objective_mean': baseline_mean,
                    'queries': 80, 'source': 'reused_original_MSE80_C80_endpoints',
                    'pbs_jobids': sorted({record['pbs_jobid'] for record in baseline_records}),
                    'refinement_updates': 0}
        conditions = [baseline] + candidates
        if len(candidates) != 6 or len(conditions) != 7:
            raise RuntimeError('Selection requires baseline and all six ratio-only conditions')
        chosen = min(conditions, key=lambda row: (
            row['objective_mean'], row['condition'] != 'no_step', row['condition']))
        selection = {
            'complete': True, 'heldout_used': False, 'sampling_scope': 'C80',
            'method': 'ratio_only_full_C80_gradient_actual_loss_selection',
            'selected_from': 'baseline_plus_6_ratio_only_candidates',
            'objective_coefficients': ({'action_mse': 1.0,
                                        'latent_nmse': float(old['training']['rho'] * old['training']['normalization_c'])}
                                       if unit == 'mse_nmse' else
                                       {'action_kl': float(scaling['action_kl_coefficient']),
                                        'latent_kl': float(scaling['latent_kl_coefficient']),
                                        'latent_weight': float(old['training']['rho'] * old['training']['normalization_c'])}),
            'baseline_objective_mean': baseline_mean, 'baseline': baseline,
            'candidates': candidates, 'conditions': conditions,
            'chosen': dict(chosen), 'candidate_count': 6, 'condition_count': 7,
            'ratio_only': True, 'D_unchanged': True,
            'gradient_source_unit': gradient_state.get('unit'),
            'gradient_source_pbs_jobid': expected_gradient_job,
        }
        selection_path = out / 'selection.json'
        if selection_path.is_file():
            if read_json(selection_path) != selection:
                raise RuntimeError('Persisted ratio-only C80 selection changed on resume')
        else:
            write_json(selection_path, selection)
        progress(phase='selected', selected_condition=chosen['condition'],
                 selected_objective=chosen['objective_mean'])

        # Rebuild from the immutable MSE80 checkpoint each attempt, then materialize once.
        restore_ratios()
        optimizer = torch.optim.Adam([ratio_group])
        optimizer_family = chosen['optimizer']
        optimizer_max_abs = 0.0
        refinement_updates = 0
        selected_uncached_action = selected_uncached_future = None
        if chosen['condition'] == 'no_step':
            optimizer.load_state_dict(source_optimizer)
            optimizer_to_device(torch, optimizer, 'cuda')
        else:
            direction, fraction = chosen['direction'], float(chosen['fraction'])
            delta = adam_delta(torch, gradient, [ratio_group],
                               source_optimizer if direction == 'saved_adam' else None)
            expected_values = set_candidate(delta, fraction)
            optimizer = torch.optim.Adam([ratio_group])
            if direction == 'saved_adam':
                optimizer.load_state_dict(source_optimizer)
                optimizer_to_device(torch, optimizer, 'cuda')
            optimizer.param_groups[0]['lr'] = float(optimizer.param_groups[0]['lr']) * fraction
            restore_ratios()
            for parameter, grad in zip(ratio_params, gradient):
                parameter.grad = grad.detach().clone()
            optimizer.step()
            banks.project()
            optimizer_max_abs = float(torch.stack([
                (parameter.detach() - expected).abs().max()
                for parameter, expected in zip(ratio_params, expected_values)]).max())
            if optimizer_max_abs > 1e-6:
                raise RuntimeError('One real ratio Adam step differs from helper candidate: '
                                   + str(optimizer_max_abs))
            with torch.no_grad():
                for parameter, expected in zip(ratio_params, expected_values):
                    parameter.copy_(expected)
            for parameter in ratio_params:
                parameter.grad = None
            refinement_updates = 1
            del delta
        total_updates = 80 + refinement_updates
        if not all(torch.equal(banks.shadows[name].detach(), value)
                   for name, value in original_d.items()):
            raise RuntimeError('Per-channel D changed during ratio-only materialization')

        if chosen['condition'] != 'no_step':
            selected_uncached_action, selected_uncached_future, _ = query(
                first, 0, 'selected_uncached_parity_queries')
            selected_uncached_action = selected_uncached_action.detach().float().cpu().clone()
            selected_uncached_future = selected_uncached_future.detach().float().cpu().clone()
        selected_ratios = [parameter.detach().clone() for parameter in ratio_params]
        banks.cache_compensated_weights()

        # Caching must preserve source and selected outputs because D never changes.
        if chosen['condition'] == 'no_step':
            selected_uncached_action, selected_uncached_future = source_uncached_action, source_uncached_future
        restore_ratios()
        source_cached_action, source_cached_future, _ = query(first, 0, 'source_cache_parity_queries')
        source_cache_parity = {
            'action_max_abs': float((source_cached_action.detach().float().cpu() - source_uncached_action).abs().max()),
            'future_max_abs': float((source_cached_future.detach().float().cpu() - source_uncached_future).abs().max()),
        }
        if any(value != 0.0 for value in source_cache_parity.values()):
            raise RuntimeError('Source uncached/cached action+future parity failed')
        with torch.no_grad():
            for parameter, selected in zip(ratio_params, selected_ratios):
                parameter.copy_(selected)
        selected_cached_action, selected_cached_future, _ = query(
            first, 0, 'selected_cached_parity_queries')
        selected_cache_parity = {
            'action_max_abs': float((selected_cached_action.detach().float().cpu() - selected_uncached_action).abs().max()),
            'future_max_abs': float((selected_cached_future.detach().float().cpu() - selected_uncached_future).abs().max()),
        }
        runtime.current_arm = unit
        runtime.description = datum(first)[1]['description']
        actual_action, _ = runtime.infer(datum(first)[0], rows[first]['sampler_seeds'][0], measure=False)
        charge('actual_infer_parity_queries')
        actual_infer_diff = float((actual_action.detach().float().cpu()
                                   - selected_cached_action.detach().float().cpu()).abs().max())
        deployment_parity = {
            'passed': all(value == 0.0 for value in selected_cache_parity.values()) and actual_infer_diff == 0.0,
            'source_uncached_cached': source_cache_parity,
            'selected_uncached_cached': selected_cache_parity,
            'actual_infer_action_max_abs': actual_infer_diff,
            'source_case_id': first, 'cached_compensation': True,
        }
        if not deployment_parity['passed']:
            raise RuntimeError('Selected cached/formal inference action+future parity failed')
        source_parity['cached_action_future_max_abs'] = source_cache_parity
        source_parity['passed'] = source_parity['passed'] and all(value == 0.0 for value in source_cache_parity.values())

        frozen = banks.verify_frozen(deployment['module_masters'])
        d_same = all(torch.equal(banks.shadows[name].detach(), value)
                     for name, value in original_d.items())
        frozen_checks = {
            'passed': d_same and all(frozen.get(key) is True for key in (
                'codebooks_unchanged', 'indices_unchanged', 'row_gains_unchanged',
                'original_exponents_unchanged', 'continuous_scale_metadata_separate'))
                and all(not parameter.requires_grad for parameter in ratio_params)
                and all(not parameter.requires_grad for parameter in banks.shadows.values())
                and all(not parameter.requires_grad for parameter in runtime.model.parameters()),
            'D_unchanged': d_same, 'ratio_parameters_frozen_after_selection': all(
                not parameter.requires_grad for parameter in ratio_params),
            'channel_parameters_frozen': all(not parameter.requires_grad for parameter in banks.shadows.values()),
            'backbone_frozen': all(not parameter.requires_grad for parameter in runtime.model.parameters()),
            **frozen,
        }
        if not frozen_checks['passed']:
            raise RuntimeError('Ratio-only scope or frozen state verification failed')

        from checkpointing import capture_rng
        history = list(saved.get('history', []))
        if refinement_updates:
            history.append({'update': 81, 'case_id': 'ratio_only_full_C80_gradient',
                            'seed_index': None, 'gradient_queries_reused': 80,
                            'direction': chosen['direction'], 'fraction': chosen['fraction'],
                            'optimizer_family': optimizer_family,
                            'objective_mean_selected': chosen['objective_mean'],
                            'ratio_only': True, 'D_unchanged': True,
                            'pbs_jobid': os.environ['PBS_JOBID']})
        checkpoint_path = out / 'checkpoint.pt'
        atomic_torch_save(torch, checkpoint_path, {
            'protocol': cfg, 'unit': unit, 'arm': unit, 'checkpoint_arm': 'mse_nmse',
            'source_checkpoint': str(source_checkpoint),
            'source_checkpoint_pbs_jobid': cfg['source_mse_job'],
            'source_updates': 80, 'refinement_updates': refinement_updates,
            'updates': total_updates, 'history': history,
            'ratios': banks.clip_snapshot(), 'shadow_log2_scales': banks.scale_snapshot(),
            'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
            'selection': selection, 'gradient_source': {
                'path': str(gradient_path), 'unit': gradient_state.get('unit'),
                'pbs_jobid': expected_gradient_job, 'gradient_queries_new': 0,
                'gradient_queries_reused': 80},
            'ratio_only': True, 'D_unchanged': True, 'heldout_used': False,
        })
        atomic_torch_save(torch, out / 'deployment.pt', {
            'protocol': cfg, 'unit': unit, 'arm': unit, 'checkpoint_arm': 'mse_nmse',
            'source_updates': 80, 'refinement_updates': refinement_updates,
            'updates': total_updates, 'selection': selection,
            'ratios': banks.clip_snapshot(), 'log2_scales': banks.scale_snapshot(),
            'scale_metadata_dtype': 'float32_log2', 'dynamic_fake_a4': True, 'bf16_kv': True,
            'ratio_only': True, 'D_unchanged': True,
            'deployment_parity': deployment_parity, 'frozen_checks': frozen_checks,
        })

        heldout_receipt = read_json(teacher_dirs['heldout'] / 'receipt.json')
        if (heldout_receipt.get('complete') is not True
                or heldout_receipt.get('queries') != 80
                or heldout_receipt.get('case_ids') != cfg['heldout_case_ids']
                or heldout_receipt.get('seed_indices') != cfg['heldout_seed_indices']):
            raise RuntimeError('Shared held-out teacher target coverage differs')

        selected_condition = chosen['condition']
        selected_label = selected_condition
        endpoint_root = root / 'results'
        for split in ('calibration', 'heldout'):
            queries = c80.queries(old, split)
            if len(queries) != 80:
                raise RuntimeError('Each final split must contain exactly 80 frozen queries')
            records = []
            for index, (case, seed) in enumerate(queries, 1):
                deadline.check(90)
                endpoint_path = endpoint_root / split / unit / 'queries' / f'{case}_{seed}.json'
                if endpoint_path.is_file():
                    record = read_json(endpoint_path)
                    validate_record(record, split, case, seed, selected_condition)
                    attempt_reused_endpoint_records += 1
                elif split == 'calibration':
                    selected_path = candidate_root / selected_label / f'{case}_0.json'
                    record = read_json(selected_path)
                    validate_record(record, 'calibration', case, 0, selected_condition)
                    record = dict(record)
                    record['curated_by_pbs_jobid'] = os.environ['PBS_JOBID']
                    record['endpoint_source'] = 'selected_C80_candidate_record_reused'
                    write_json(endpoint_path, record)
                else:
                    action, future, counts = query(case, seed, 'heldout_queries')
                    teacher_value = target('heldout', case, seed)
                    record = make_record(split, case, seed, action, future, teacher_value,
                                         counts, selected_condition, refinement_updates)
                    record['direction'] = chosen['direction']
                    record['fraction'] = chosen['fraction']
                    record['optimizer'] = chosen['optimizer']
                    record['endpoint_source'] = 'new_selected_ratio_only_heldout_query'
                    validate_record(record, split, case, seed, selected_condition)
                    write_json(endpoint_path, record)
                    del action, future
                records.append(record)
                progress(phase='final_endpoint_evaluation', split=split, queries=index,
                         last_case=case, last_seed=seed)
            summary_values = c80.aggregate_records(records, old, split)
            display_keys = {'action_mse': 'first10motor_mse',
                            'latent_nmse': 'future_latent_nmse',
                            'action_kl': 'action_kl', 'latent_kl': 'latent_kl'}
            for key, display_key in display_keys.items():
                summary_values['overall'][display_key] = mean(record_values(record)[key]
                                                              for record in records)
                for suite in ref.SUITES:
                    suite_records = [record for record in records if record['suite'] == suite]
                    summary_values['by_suite'][suite][display_key] = mean(
                        record_values(record)[key] for record in suite_records)
            summary_values['overall']['objective_loss'] = mean(record['objective_loss'] for record in records)
            for suite in ref.SUITES:
                suite_records = [record for record in records if record['suite'] == suite]
                summary_values['by_suite'][suite]['objective_loss'] = mean(
                    record['objective_loss'] for record in suite_records)
            if split == 'calibration' and not close(summary_values['overall']['objective_loss'],
                                                    chosen['objective_mean']):
                raise RuntimeError('Reused chosen C80 records differ from selected objective mean')
            write_json(root / 'results' / split / unit / 'summary.json', {
                'protocol': cfg['protocol'], 'complete': True, 'unit': unit, 'arm': unit,
                'checkpoint_arm': 'mse_nmse', 'loss': unit, 'split': split,
                'pbs_jobid': os.environ['PBS_JOBID'], 'source_updates': 80,
                'refinement_updates': refinement_updates, 'total_updates': total_updates,
                'updates': total_updates, 'selection': selection, **summary_values})

        if not source_parity['passed'] or not deployment_parity['passed'] or not frozen_checks['passed']:
            raise RuntimeError('A required parity or frozen-state gate did not pass')
        receipt = {
            'protocol': cfg['protocol'], 'complete': True,
            'pbs_jobid': os.environ['PBS_JOBID'], 'unit': unit, 'arm': unit,
            'checkpoint_arm': 'mse_nmse', 'loss': unit,
            'source_checkpoint': str(source_checkpoint),
            'source_checkpoint_pbs_jobid': cfg['source_mse_job'],
            'source_updates': 80, 'refinement_updates': refinement_updates,
            'total_updates': total_updates, 'updates': total_updates,
            'gradient_query_budget': 80, 'gradient_queries_new': 0,
            'gradient_queries_reused': 80, 'new_backward_queries': 0,
            'gradient_source': {'path': str(gradient_path), 'unit': gradient_state.get('unit'),
                                'pbs_jobid': expected_gradient_job,
                                'source_gradient_protocol': gradient_protocol.get('protocol')},
            'calibration_queries': 80, 'heldout_queries': 80, 'evaluation_queries': 160,
            'new_teacher_queries': 0,
            'new_model_queries': accounting['model_queries'],
            'new_model_queries_this_attempt': accounting['model_queries'] - queries_at_start,
            'candidate_queries_actual': accounting.get('candidate_queries', 0),
            'candidate_records_reused_this_attempt': attempt_reused_candidate_records,
            'endpoint_records_reused_this_attempt': attempt_reused_endpoint_records,
            'source_records_reused_this_attempt': attempt_reused_source_records,
            'ratio_only': True, 'D_unchanged': d_same,
            'trainable_ratio_coordinates': 614,
            'frozen_channel_coordinates': 1806863,
            'selection': selection,
            'source_parity_passed': source_parity['passed'],
            'source_parity': source_parity,
            'deployment_parity_passed': deployment_parity['passed'],
            'deployment_parity': deployment_parity,
            'frozen_verified': frozen_checks['passed'],
            'frozen_checks': frozen_checks,
            'optimizer_family': optimizer_family,
            'selected_adam_step_max_abs': optimizer_max_abs,
            'forward_accounting': accounting,
            'candidate_records_root': str(candidate_root),
            'checkpoint': str(checkpoint_path), 'deployment': str(out / 'deployment.pt'),
            'heldout_used': False, 'seconds_inside_worker': time.monotonic() - started,
        }
        write_json(out / 'training_receipt.json', receipt)
        write_json(out / 'summary.json', {**receipt, 'phase': 'complete'})
        progress(phase='complete', complete=True, source_updates=80,
                 refinement_updates=refinement_updates, total_updates=total_updates)
        print('RATIO_REFINE_COMPLETE', flush=True)
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
    parser.add_argument('--unit', choices=ARMS, default=os.environ.get('FW_REFINE_UNIT'))
    args = parser.parse_args()
    if args.unit not in ARMS:
        parser.error('--unit or FW_REFINE_UNIT must be mse_nmse or dual_kl')
    return run(read_json(args.protocol), args.unit)


if __name__ == '__main__':
    raise SystemExit(main())
