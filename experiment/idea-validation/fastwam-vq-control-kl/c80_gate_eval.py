"""Finite evaluation of reversible full-C80 D/r proposals."""
import argparse
import math
import os
from pathlib import Path
import sys

from control_common import Deadline, guard, read, save


VARIANTS = ('clean', 'noisy')
COUNTS = {'video_steps': 10, 'action_steps': 10}
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)
NOISE_SHAPE = (1, 32, 7)
FRACTIONS = (1.0, 0.1)
UNIT_OBJECTIVES = {
    'c80_eval_response_0': 'action_response_kl',
    'c80_eval_response_1': 'action_response_kl',
    'c80_eval_latent_0': 'latent_control_kl',
    'c80_eval_latent_1': 'latent_control_kl',
}


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _record_is_reusable(record, binding, objective):
    if not isinstance(record, dict) or record.get('complete') is not True:
        return False
    if (record.get('binding') != binding or not record.get('created_pbs_jobid')
            or not _finite_number(record.get('case_loss'))):
        return False
    components = record.get('loss_components')
    required = {'pair'} if objective == 'action_response_kl' else set(VARIANTS)
    if (not isinstance(components, dict) or set(components) != required
            or not all(_finite_number(value) for value in components.values())):
        return False
    expected_loss = (components['pair'] if objective == 'action_response_kl'
                     else 0.5 * (components['clean'] + components['noisy']))
    if not math.isclose(float(record['case_loss']), expected_loss,
                        rel_tol=0.0, abs_tol=1e-12):
        return False
    if record.get('native_noise_parity') != {v: True for v in VARIANTS}:
        return False
    expected_counts = {v: COUNTS for v in VARIANTS}
    if record.get('runtime_counts') != expected_counts:
        return False
    if objective == 'latent_control_kl':
        if record.get('reader_noise_parity') != {v: True for v in VARIANTS}:
            return False
        expected_reader = {v: {'video_prefill_calls': 1, 'action_scheduler_steps': 10}
                           for v in VARIANTS}
        if record.get('reader_counts') != expected_reader:
            return False
    return True


def _render_candidate_summaries(fractions, proposals, records):
    result = []
    for fraction in fractions:
        if fraction not in proposals:
            continue
        saved = records[fraction]
        case_ids = sorted(saved)
        loss_sum = sum(float(saved[index]['case_loss']) for index in case_ids)
        result.append({**proposals[fraction]['summary_binding'],
                       'fraction': fraction, 'case_ids': case_ids,
                       'cases': len(case_ids), 'loss_sum': loss_sum,
                       'mean_loss': loss_sum / len(case_ids) if case_ids else None})
    return result


def _refresh_progress(state, proposals, fraction_records, fractions, cases, objective):
    state['candidates'] = _render_candidate_summaries(
        fractions, proposals, fraction_records)
    coverage = [set(fraction_records.get(float(fraction), {}))
                for fraction in fractions]
    state['completed_cases'] = len(set.intersection(*coverage)) if coverage else 0
    state['native_and_reader_initial_noise_parity_valid'] = all(
        len(fraction_records.get(float(fraction), {})) == len(cases)
        and all(record['native_noise_parity'] == {v: True for v in VARIANTS}
                and (objective != 'latent_control_kl'
                     or record['reader_noise_parity'] == {v: True for v in VARIANTS})
                for record in fraction_records.get(float(fraction), {}).values())
        for fraction in fractions)


def _record_binding(config, protocol, unit, objective, shard, case, identities,
                    proposal, calibration_unit):
    return {
        'protocol': protocol,
        'gate_protocol': config['protocol'],
        'config': config,
        'unit': unit,
        'objective': objective,
        'shard': shard,
        'case': {'case_id': int(case['case_id']), 'suite': case['suite'],
                 'shard': int(case['shard']), 'seed': int(case['sampler_seed']),
                 'description': case['description']},
        'seed': int(case['sampler_seed']),
        'input_identity': identities,
        'source_cache': {'unit': calibration_unit, 'arm': 'per_channel_clip',
                         'teacher_arm': 'bf16', 'variants': list(VARIANTS)},
        'proposal': proposal['record_binding'],
    }


def _validate_candidate(torch, value, path, direction, candidate_row, objective,
                        fraction, expected_ids, modules):
    expected_keys = {'protocol', 'parent_protocol', 'direction_unit',
                     'direction_pbs_jobid', 'objective', 'fraction', 'case_ids',
                     'ratios', 'log2_scales'}
    if (not isinstance(value, dict) or set(value) != expected_keys
            or value.get('protocol') != direction['gate_protocol']
            or value.get('parent_protocol') != direction['protocol']
            or value.get('direction_unit') != direction['unit']
            or value.get('direction_pbs_jobid') != direction['pbs_jobid']
            or value.get('objective') != objective
            or float(value.get('fraction', -1)) != fraction
            or value.get('case_ids') != expected_ids):
        raise ValueError('Candidate file does not match its completed direction receipt')
    if (not path.is_absolute() or str(candidate_row.get('path')) != str(path)
            or set(value.get('ratios', {})) != set(modules)
            or set(value.get('log2_scales', {})) != set(modules)):
        raise ValueError('Candidate path or 614-module parameter coverage differs')
    for key in ('ratios', 'log2_scales'):
        mapping = value[key]
        for name, tensor in mapping.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or tensor.dtype != torch.float32 or tensor.requires_grad
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError(f'Candidate {key} must contain finite CPU FP32 tensors: {name}')
    return value


def _load_native_pair(torch, root, protocol, calibration_unit, case, identities):
    from calibration_worker import _pair_binding
    from diagnostic_runtime import validate_binding

    records = {}
    for arm in ('bf16', 'per_channel_clip'):
        for variant in VARIANTS:
            path = root / 'results' / calibration_unit / 'records' / (
                f"{int(case['case_id'])}_{arm}_{variant}.pt")
            saved = torch.load(path, map_location='cpu', weights_only=True)
            binding = {**_pair_binding(protocol, calibration_unit, case, arm, variant),
                       'input_identity': identities[variant]}
            validate_binding(saved, binding)
            if saved.get('counts') != COUNTS:
                raise ValueError(f'Cached calibration query counts differ: {path.name}')
            for key, shape in (('action', ACTION_SHAPE), ('future', FUTURE_SHAPE),
                               ('action_initial_noise', NOISE_SHAPE)):
                tensor = torch.as_tensor(saved.get(key)).detach().float().cpu()
                if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'Cached calibration {key} differs: {path.name}')
                saved[key] = tensor
            records[arm, variant] = saved
    for variant in VARIANTS:
        if not torch.equal(records['bf16', variant]['action_initial_noise'],
                           records['per_channel_clip', variant]['action_initial_noise']):
            raise ValueError(f'Original C80 initial action noise differs for {case["case_id"]}/{variant}')
    return records


def _reader_pass(torch, reader, runtime, datum, seed, future, expected_noise,
                 deadline, counters):
    if deadline.near(300):
        raise TimeoutError('C80 finite evaluation reached its soft deadline')
    counters['reader_new_pass_attempted_count'] += 1
    action, counts = reader(runtime, datum, seed, future, checkpointed=False)
    counters['reader_new_pass_count'] += 1
    action = action.detach().float().cpu()
    noise = torch.as_tensor(counts.get('action_initial_noise')).detach().float().cpu()
    if (tuple(action.shape) != ACTION_SHAPE or not bool(torch.isfinite(action).all())
            or tuple(noise.shape) != NOISE_SHAPE or not torch.equal(noise, expected_noise)
            or counts.get('video_prefill_calls') != 1
            or counts.get('action_scheduler_steps') != 10):
        raise RuntimeError('Frozen BF16 reader action, counts, or initial noise parity differs')
    return action, {'video_prefill_calls': 1, 'action_scheduler_steps': 10}


def _read_case_inputs(torch, np, switching, root, case):
    from calibration_worker import _load_case_inputs

    metadata = root / 'data' / 'calibration_pairs' / str(int(case['case_id'])) / 'metadata.json'
    return _load_case_inputs(torch, np, switching, root, case, metadata)


def _validate_direction_receipt(receipt, protocol, config, unit, objective, expected_ids):
    if (receipt.get('protocol') != protocol or receipt.get('gate_protocol') != config['protocol']
            or receipt.get('unit') != unit or receipt.get('objective') != objective
            or receipt.get('complete') is not True or not receipt.get('pbs_jobid')
            or receipt.get('case_ids') != expected_ids):
        raise ValueError('Completed C80 direction receipt does not match this evaluation')
    rows = receipt.get('candidates')
    fractions = [float(value) for value in config['fractions']]
    if (not isinstance(rows, list) or len(rows) != len(fractions)
            or {float(row.get('fraction', -1)) for row in rows} != set(fractions)):
        raise ValueError('Direction receipt must contain both frozen candidate fractions')
    if any(not _finite_number(row.get(key)) for row in rows
           for key in ('surrogate_g_dot_delta', 'projected_step_l2')):
        raise ValueError('Direction proposal metrics must be finite')
    return rows


def _validate_calibration_receipt(receipt, protocol, unit, shard, cases):
    ids = [int(case['case_id']) for case in cases]
    records = receipt.get('records')
    if (receipt.get('protocol') != protocol or receipt.get('unit') != unit
            or receipt.get('complete') is not True or receipt.get('status') != 'complete'
            or receipt.get('shard') != shard or receipt.get('expected') != 40
            or receipt.get('completed') != 40 or not isinstance(records, list)
            or len(records) != 40):
        raise ValueError('Original calibration shard is incomplete or mismatched')
    by_id = {int(row['case_id']): row for row in records}
    if set(by_id) != set(ids) or len(by_id) != len(records):
        raise ValueError('Original calibration shard case IDs differ')
    for case_id in ids:
        row = by_id[case_id]
        if (row.get('shard') != shard or row.get('native_queries_expected') != 8
                or row.get('native_queries_saved') != 8
                or row.get('initial_noise_parity_valid') is not True):
            raise ValueError(f'Original calibration case receipt is incomplete: {case_id}')
    if (receipt.get('counts_valid') is not True
            or receipt.get('initial_noise_parity_all_arms_per_variant') is not True):
        raise ValueError('Original calibration shard count/noise gate failed')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=tuple(UNIT_OBJECTIVES), required=True)
    args = parser.parse_args(argv)
    guard()

    p = read(Path(__file__).with_name('protocol.json'))
    config = read(Path(__file__).with_name('c80_gate_protocol.json'))
    unit, objective = args.unit, UNIT_OBJECTIVES[args.unit]
    shard = int(unit[-1])
    root = Path(p['remote_root'])
    out = root / 'results' / unit
    job_id = os.environ['PBS_JOBID']
    state = {
        'protocol': p['protocol'], 'gate_protocol': config.get('protocol'), 'unit': unit,
        'pbs_jobid': job_id, 'complete': False, 'objective': objective, 'shard': shard,
        'expected_cases': 40, 'completed_cases': 0, 'case_ids': [], 'candidates': [],
        'native_and_reader_initial_noise_parity_valid': False,
        'bf16_reader_native_parity_all_zero': 'not-required'
        if objective == 'action_response_kl' else False,
        'original_parameters_restored': False, 'frozen_vq': None,
        'training_updates': 0, 'heldout_inputs_used': False,
        'sensor_evaluation_inputs_used': False,
        'runtime_new_query_count': 0, 'runtime_new_query_attempted_count': 0,
        'reader_new_pass_count': 0, 'reader_new_pass_attempted_count': 0,
        'case_records_saved': 0, 'case_records_reused': 0, 'case_records_rejected': 0,
        'source_calibration_unit': f'calibration_{shard}',
        'direction_unit': 'c80_direction_response' if objective == 'action_response_kl'
                          else 'c80_direction_latent',
        'failure': None,
    }
    save(out / 'summary.json', state)

    switching = None
    reader = None
    original_ratios = original_scales = None
    failure = None
    proposals, fraction_records = {}, {}
    counters = state
    try:
        if (config.get('parent_protocol') != p.get('protocol')
                or config.get('protocol') != 'control-kl-c80-parameter-gate-v1'
                or config.get('fractions') != list(FRACTIONS)
                or config.get('objectives') != ['action_response_kl', 'latent_control_kl']
                or config.get('objective_keys', {}).get('response') != 'action_response_kl'
                or config.get('objective_keys', {}).get('latent') != 'latent_control_kl'
                or config.get('training_updates') != 0
                or config.get('heldout_inputs_used') is not False
                or config.get('sensor_evaluation_inputs_used') is not False):
            raise ValueError('Frozen C80 gate protocol differs')
        from aggregate import completed

        gate = completed(root, 'parameter_gate')
        if (gate.get('protocol') != p['protocol'] or gate.get('unit') != 'parameter_gate'
                or gate.get('original_parameters_restored') is not True
                or gate.get('frozen_reader_and_model_unchanged') is not True
                or gate.get('training_updates') != 0
                or gate.get('heldout_inputs_used') is not False
                or gate.get('sensor_evaluation_inputs_used') is not False):
            raise RuntimeError('The completed reversible parameter gate did not pass')

        expected_ids = [int(value) for value in config['case_ids']]
        parent_ids = [int(value) for value in p['source_relational_protocol']['calibration']['case_ids']]
        if expected_ids != parent_ids or len(expected_ids) != 80:
            raise ValueError('Frozen full-C80 case IDs differ from the parent protocol')
        direction_unit = state['direction_unit']
        direction = completed(root, direction_unit)
        direction_rows = _validate_direction_receipt(
            direction, p['protocol'], config, direction_unit, objective, expected_ids)
        proposal_rows = {float(row['fraction']): row for row in direction_rows}

        calibration_unit = state['source_calibration_unit']
        calibration_receipt = completed(root, calibration_unit)
        pair_summary = completed(root, 'calibration_pairs')
        relp = p['source_relational_protocol']
        definition = read(root / 'data' / 'calibration_pairs' / 'definition.json')
        original_plan = read(Path(relp['calibration']['plan_file']))
        original_rows = {int(row['case_id']): row for row in original_plan.get('inputs', [])
                         if int(row['case_id']) in set(expected_ids)}
        if set(original_rows) != set(expected_ids):
            raise ValueError('Original complete C80 plan rows do not cover all cases')
        from calibration_worker import _pair_catalog
        cases = _pair_catalog(p['protocol'], definition, pair_summary, original_rows,
                              expected_ids, shard)
        cases = sorted(cases, key=lambda case: int(case['case_id']))
        if (len(cases) != 40 or any(int(case['case_id']) not in expected_ids
                                    or int(case['shard']) != shard for case in cases)):
            raise ValueError('Calibration shard case IDs differ from the frozen case catalog')
        _validate_calibration_receipt(calibration_receipt, p['protocol'],
                                      calibration_unit, shard, cases)
        state['case_ids'] = [int(case['case_id']) for case in cases]
        state['direction_pbs_jobid'] = direction['pbs_jobid']

        import numpy as np
        import torch
        from control_losses import action_response_kl, latent_control_kl
        from diagnostic_runtime import SwitchingRuntime
        from frozen_reader import BF16Reader

        deadline = Deadline()
        stats = read(Path(relp['source_dual_root']) / 'results' / 'fit_stats' / 'summary.json')
        switching = SwitchingRuntime(p, out)
        runtime, banks = switching.runtime, switching.banks
        modules = runtime.q.modules
        if len(modules) != 614:
            raise RuntimeError('Expected all 614 target Linears')
        if set(switching.deployments['per_channel_clip']['ratios']) != set(modules):
            raise RuntimeError('Original per_channel_clip deployment module coverage differs')
        original_ratios = {name: value.detach().cpu().clone() for name, value in
                           switching.deployments['per_channel_clip']['ratios'].items()}
        original_scales = {name: value.detach().cpu().clone() for name, value in
                           switching.deployments['per_channel_clip']['log2_scales'].items()}
        switching.activate('per_channel_clip')
        if not banks.cached:
            raise RuntimeError('Original per_channel_clip deployment did not activate cached weights')
        for name in modules:
            if (not torch.equal(banks.ratios[name].detach().cpu(), original_ratios[name])
                    or not torch.equal(banks.shadows[name].detach().cpu(), original_scales[name])):
                raise RuntimeError('Activated per_channel_clip parameters differ from its deployment')
        reader = BF16Reader(runtime, switching.original_weights)

        candidate_ids = expected_ids
        for fraction in FRACTIONS:
            row = proposal_rows[fraction]
            path = Path(row['path'])
            candidate = torch.load(path, map_location='cpu', weights_only=True)
            candidate = _validate_candidate(torch, candidate, path, direction, row,
                                            objective, fraction, candidate_ids, modules)
            for name in modules:
                if (tuple(candidate['ratios'][name].shape) != tuple(banks.ratios[name].shape)
                        or tuple(candidate['log2_scales'][name].shape) != tuple(banks.shadows[name].shape)):
                    raise ValueError(f'Candidate parameter shape differs for {name}')
            proposal_binding = {
                'direction_unit': direction_unit,
                'direction_pbs_jobid': direction['pbs_jobid'],
                'objective': objective, 'fraction': fraction,
                'candidate_path': str(path), 'case_ids': candidate_ids,
                'surrogate_g_dot_delta': float(row['surrogate_g_dot_delta']),
                'projected_step_l2': float(row['projected_step_l2']),
            }
            proposals[fraction] = {
                'tensors': candidate,
                'record_binding': proposal_binding,
                'summary_binding': {key: value for key, value in proposal_binding.items()
                                    if key not in ('candidate_path', 'case_ids')},
            }
            fraction_records[fraction] = {}

        state['candidates'] = _render_candidate_summaries(
            FRACTIONS, proposals, fraction_records)
        reader_native_parity = []
        weight_signatures = {
            name: (value.data_ptr(), value._version, value.dtype, tuple(value.shape),
                   value.device.type)
            for name, value in switching.original_weights.items()
        }
        target_parameter_ids = {id(module.weight) for module in modules.values()}
        other_parameter_signatures = {
            name: (id(value), value.data_ptr(), value._version, value.dtype,
                   tuple(value.shape), value.device.type)
            for name, value in runtime.model.named_parameters()
            if id(value) not in target_parameter_ids
        }

        def restore_original_parameters():
            banks.restore_clips(original_ratios)
            banks.restore_scales(original_scales)
            switching.install_compensated_weights()
            banks.cached = True
            if not all(torch.equal(banks.ratios[name].detach().cpu(), original_ratios[name])
                       and torch.equal(banks.shadows[name].detach().cpu(), original_scales[name])
                       for name in modules):
                raise RuntimeError('Original MSE D/r values were not restored exactly')

        def install_candidate(candidate):
            banks.restore_clips(candidate['ratios'])
            banks.restore_scales(candidate['log2_scales'])
            switching.install_compensated_weights()
            banks.cached = True

        if objective == 'latent_control_kl':
            first_case = cases[0]
            datums, identities, _ = _read_case_inputs(torch, np, switching, root, first_case)
            native = _load_native_pair(torch, root, p['protocol'], calibration_unit,
                                       first_case, identities)
            runtime.description = first_case['description']
            for variant in VARIANTS:
                cached = native['bf16', variant]
                action, reader_counts = _reader_pass(
                    torch, reader, runtime, datums[variant], int(first_case['sampler_seed']),
                    cached['future'], cached['action_initial_noise'], deadline, counters)
                error = float((action - cached['action']).abs().max())
                if error != 0.0:
                    raise RuntimeError('BF16Reader does not reproduce the cached BF16 action')
                reader_native_parity.append({'variant': variant, 'action_max_abs': error,
                                             'initial_noise_max_abs': 0.0,
                                             **reader_counts})
            state['bf16_reader_native_parity'] = reader_native_parity
            state['bf16_reader_native_parity_all_zero'] = True

        _refresh_progress(state, proposals, fraction_records, FRACTIONS, cases, objective)
        save(out / 'summary.json', state)
        for fraction in FRACTIONS:
            proposal = proposals[fraction]
            install_candidate(proposal['tensors'])
            try:
                for case in cases:
                    datums, identities, _ = _read_case_inputs(torch, np, switching, root, case)
                    native = _load_native_pair(torch, root, p['protocol'], calibration_unit,
                                               case, identities)
                    runtime.description = case['description']
                    binding = _record_binding(config, p['protocol'], unit, objective, shard,
                                              case, identities, proposal, calibration_unit)
                    record_path = out / 'records' / (
                        f"{int(case['case_id'])}_f{fraction:.1f}.json")
                    cached_record = None
                    if record_path.is_file():
                        try:
                            existing = read(record_path)
                            if _record_is_reusable(existing, binding, objective):
                                cached_record = existing
                        except (OSError, ValueError, TypeError):
                            cached_record = None
                        if cached_record is None:
                            counters['case_records_rejected'] += 1
                    if cached_record is not None:
                        counters['case_records_reused'] += 1
                        fraction_records[fraction][int(case['case_id'])] = cached_record
                        _refresh_progress(state, proposals, fraction_records,
                                          FRACTIONS, cases, objective)
                        save(out / 'summary.json', state)
                        continue

                    outputs, runtime_counts, native_noise_parity = {}, {}, {}
                    reader_counts, reader_noise_parity = {}, {}
                    loss_components = {}
                    for variant in VARIANTS:
                        if deadline.near(300):
                            raise TimeoutError('C80 finite evaluation reached its soft deadline')
                        counters['runtime_new_query_attempted_count'] += 1
                        action, future, counts, noise = switching.query(
                            datums[variant], int(case['sampler_seed']), case['description'])
                        counters['runtime_new_query_count'] += 1
                        action = torch.as_tensor(action).detach().float().cpu()
                        future = torch.as_tensor(future).detach().float().cpu()
                        noise = torch.as_tensor(noise).detach().float().cpu()
                        if (counts != COUNTS or tuple(action.shape) != ACTION_SHAPE
                                or tuple(future.shape) != FUTURE_SHAPE
                                or tuple(noise.shape) != NOISE_SHAPE
                                or not bool(torch.isfinite(action).all())
                                or not bool(torch.isfinite(future).all())
                                or not bool(torch.isfinite(noise).all())):
                            raise RuntimeError('Candidate native query coverage/counts differ')
                        parity = torch.equal(noise, native['bf16', variant]['action_initial_noise'])
                        if not parity:
                            raise RuntimeError('Candidate/BF16 initial action noise differs')
                        outputs[variant] = {'action': action, 'future': future}
                        runtime_counts[variant] = dict(counts)
                        native_noise_parity[variant] = True

                        if objective == 'latent_control_kl':
                            reader_action, small_counts = _reader_pass(
                                torch, reader, runtime, datums[variant],
                                int(case['sampler_seed']), future,
                                native['bf16', variant]['action_initial_noise'],
                                deadline, counters)
                            loss = latent_control_kl(
                                torch, reader_action, native['bf16', variant]['action'], stats)
                            loss_components[variant] = float(loss.detach())
                            reader_counts[variant] = small_counts
                            reader_noise_parity[variant] = True

                    if objective == 'action_response_kl':
                        loss = action_response_kl(
                            torch, outputs['clean']['action'], outputs['noisy']['action'],
                            native['bf16', 'clean']['action'],
                            native['bf16', 'noisy']['action'], stats)
                        loss_components['pair'] = float(loss.detach())
                        case_loss = loss_components['pair']
                    else:
                        case_loss = 0.5 * (loss_components['clean'] + loss_components['noisy'])
                    if not _finite_number(case_loss):
                        raise RuntimeError('Candidate finite-evaluation loss is not finite')
                    record = {
                        'complete': True, 'binding': binding,
                        'created_pbs_jobid': job_id, 'case_loss': case_loss,
                        'loss_components': loss_components,
                        'runtime_counts': runtime_counts,
                        'native_noise_parity': native_noise_parity,
                        'reader_counts': reader_counts,
                        'reader_noise_parity': reader_noise_parity,
                    }

                    save(record_path, record)
                    counters['case_records_saved'] += 1
                    fraction_records[fraction][int(case['case_id'])] = record
                    _refresh_progress(state, proposals, fraction_records,
                                      FRACTIONS, cases, objective)
                    save(out / 'summary.json', state)
            finally:
                restore_original_parameters()

        baseline_path = Path(p['source_closed_protocol']['sources']['baseline_deployment'])
        baseline = torch.load(baseline_path, map_location='cpu', weights_only=True)
        state['frozen_vq'] = banks.verify_frozen(baseline['module_masters'])
        state['frozen_reader_and_model_unchanged'] = (
            all((value.data_ptr(), value._version, value.dtype, tuple(value.shape),
                 value.device.type) == weight_signatures[name]
                for name, value in switching.original_weights.items())
            and all((id(value), value.data_ptr(), value._version, value.dtype,
                     tuple(value.shape), value.device.type) == other_parameter_signatures[name]
                    for name, value in runtime.model.named_parameters()
                    if id(value) not in target_parameter_ids))
        if state['frozen_reader_and_model_unchanged'] is not True:
            raise RuntimeError('Retained BF16 reader or frozen model parameters changed')

    except Exception as exc:
        failure = (exc, exc.__traceback__)
        state['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if switching is not None and original_ratios is not None and original_scales is not None:
                banks = switching.banks
                banks.restore_clips(original_ratios)
                banks.restore_scales(original_scales)
                switching.install_compensated_weights()
                banks.cached = True
                state['original_parameters_restored'] = all(
                    torch.equal(banks.ratios[name].detach().cpu(), original_ratios[name])
                    and torch.equal(banks.shadows[name].detach().cpu(), original_scales[name])
                    for name in switching.runtime.q.modules)
                if not state['original_parameters_restored']:
                    raise RuntimeError('Final original MSE D/r rollback failed')
        except Exception as exc:
            if failure is None:
                failure = (exc, exc.__traceback__)
            state['failure'] = (state['failure'] + '; ' if state['failure'] else '') + \
                f'{type(exc).__name__}: {exc}'
        try:
            if switching is not None:
                switching.close()
        except Exception as exc:
            if failure is None:
                failure = (exc, exc.__traceback__)
            state['failure'] = (state['failure'] + '; ' if state['failure'] else '') + \
                f'{type(exc).__name__}: {exc}'

        if proposals:
            _refresh_progress(state, proposals, fraction_records,
                              FRACTIONS, cases, objective)
        fractions = list(FRACTIONS)
        complete_candidates = (len(state['candidates']) == len(fractions)
                               and all(row.get('cases') == 40
                                       and row.get('case_ids') == state['case_ids']
                                       for row in state['candidates']))
        state['complete'] = bool(
            failure is None and complete_candidates and state['completed_cases'] == 40
            and state['native_and_reader_initial_noise_parity_valid'] is True
            and state['bf16_reader_native_parity_all_zero'] in (True, 'not-required')
            and state['original_parameters_restored'] is True
            and isinstance(state.get('frozen_vq'), dict)
            and all(state['frozen_vq'].values())
            and state.get('frozen_reader_and_model_unchanged') is True
            and state['training_updates'] == 0
            and state['heldout_inputs_used'] is False
            and state['sensor_evaluation_inputs_used'] is False)
        if failure is not None:
            state['complete'] = False
        save(out / 'summary.json', state)

    if failure is not None:
        raise failure[0].with_traceback(failure[1])
    if state['complete'] is not True:
        raise RuntimeError('C80 finite evaluation ended without complete coverage')
    print({'unit': unit, 'complete': True, 'completed_cases': state['completed_cases'],
           'runtime_new_query_count': state['runtime_new_query_count'],
           'case_records_saved': state['case_records_saved'],
           'case_records_reused': state['case_records_reused']}, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
