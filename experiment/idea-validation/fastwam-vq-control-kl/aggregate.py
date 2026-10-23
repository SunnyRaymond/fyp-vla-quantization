"""CPU summaries of frozen same-state endpoints and reader interventions."""
import argparse
import os
from pathlib import Path
import statistics

from control_common import guard, read, save

STUDENTS = ('per_channel_clip', 'softbin', 'relational')
ENDPOINT_KEYS = ('motor_rmse', 'action_kl', 'latent_kl', 'latent_nmse', 'action_temporal_kl', 'latent_spatiotemporal_kl', 'softbin_objective', 'relational_objective', 'command_rmse', 'command_chunk_bias_rms', 'command_prefix_cumsum_max_abs')
RESPONSE_KEYS = ('action_response_kl', 'normalized_response_error_rmse', 'teacher_response_rms', 'command_rmse', 'command_chunk_bias_rms', 'command_prefix_cumsum_max_abs')
READER_KEYS = ('first10_motor_rmse', 'first10_command_motor_rmse', 'softbin_first10_motor_kl', 'command_prefix_first10_rmse', 'command_prefix_first10_max_abs')


def means(records, keys):
    if not records:
        raise ValueError('Cannot summarize an empty cohort')
    suites = sorted({r['suite'] for r in records})
    by_suite = {s: {k: statistics.fmean(r[k] for r in records if r['suite'] == s)
                    for k in keys} for s in suites}
    return {'cases': len(records), 'by_suite': by_suite,
            'equal_suite_mean': {k: statistics.fmean(by_suite[s][k] for s in suites) for k in keys}}


def command_errors(torch, candidate, target):
    delta = torch.as_tensor(candidate).float()[:10, :6] - torch.as_tensor(target).float()[:10, :6]
    return {'command_rmse': float(delta.square().mean().sqrt()),
            'command_chunk_bias_rms': float(delta.mean(0).square().mean().sqrt()),
            'command_prefix_cumsum_max_abs': float(delta.cumsum(0).abs().max())}


def completed(root, unit):
    value = read(root/'results'/unit/'summary.json')
    if value.get('complete') is not True:
        raise RuntimeError('Incomplete prerequisite: '+unit)
    artifact = root/'artifacts'/value['pbs_jobid']
    if (artifact/'exit_code.txt').read_text().strip() != '0' or not (artifact/'PHASE_COMPLETE').is_file():
        raise RuntimeError('Prerequisite has no successful completion receipt: '+unit)
    return value


def loss_definition(p):
    relp = p['source_relational_protocol']
    definition = read(Path(relp['remote_root'])/'results/prepare/summary.json')
    scaling = read(Path(relp['source_dual_root'])/'results/prepare/loss_scaling.json')
    alpha_a, alpha_z = scaling['action_kl_coefficient'], scaling['latent_kl_coefficient']
    latent_weight = relp['training']['rho']*relp['training']['normalization_c']
    return definition, alpha_a, alpha_z, latent_weight


def endpoint_metrics(torch, q, t, stats, definition):
    from softbin_loss import losses
    from relational_loss import relational_losses
    config, alpha_a, alpha_z, latent_weight = definition
    beta = config['coefficients']
    values = losses(torch, q['action'], q['future'], t['action'], t['future'], stats)
    values.update(relational_losses(torch, q['action'], q['future'], t['action'], t['future'], stats, config['relational_config']))
    soft = alpha_a*values['action_kl']+latent_weight*alpha_z*values['latent_kl']
    composite = soft+alpha_a*beta['action_beta']*values['action_temporal_kl']+latent_weight*alpha_z*beta['latent_beta']*values['latent_spatiotemporal_kl']
    return {'motor_rmse': float(values['action_mse'].sqrt()),
            **{k: float(values[k]) for k in ('action_kl', 'latent_kl', 'latent_nmse', 'action_temporal_kl', 'latent_spatiotemporal_kl')},
            'softbin_objective': float(soft), 'relational_objective': float(composite),
            **command_errors(torch, q['commands'], t['commands'])}


def response_metrics(torch, qc, qn, tc, tn, stats):
    from control_losses import action_response_kl
    error = (qn['action']-qc['action']-tn['action']+tc['action'])[:10, :6].float()
    qdelta = torch.as_tensor(qn['commands'])-torch.as_tensor(qc['commands'])
    tdelta = torch.as_tensor(tn['commands'])-torch.as_tensor(tc['commands'])
    return {'action_response_kl': float(action_response_kl(torch, qc['action'], qn['action'], tc['action'], tn['action'], stats)),
            'normalized_response_error_rmse': float(error.square().mean().sqrt()),
            'teacher_response_rms': float((tn['action']-tc['action'])[:10, :6].square().mean().sqrt()),
            **command_errors(torch, qdelta, tdelta)}


def sensor(p, root, cases, stats, prefix='sensor', expected_per_shard=20):
    import torch
    definition = loss_definition(p)
    for unit in (prefix+'_0', prefix+'_1'):
        receipt = completed(root, unit)
        if receipt.get('completed') != expected_per_shard:
            raise RuntimeError('Expected full case coverage in '+unit)
    endpoints, responses = [], []
    for case in cases:
        index, unit = case['index'], prefix+'_'+str(case['shard'])
        records = {}
        for arm in ('bf16', *STUDENTS):
            for variant in ('clean', 'noisy'):
                stem = root/'results'/unit/'records'/f'{index}_{arm}_{variant}'
                r, meta = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True), read(stem.with_suffix('.json'))
                if (r['index'] != index or r['arm'] != arm or r['variant'] != variant
                        or r['seed'] != case['sampler_seed'] or r['protocol'] != p['protocol']):
                    raise ValueError('Endpoint input binding differs')
                r['commands'] = meta['commands']
                records[arm, variant] = r
        for arm in STUDENTS:
            for variant in ('clean', 'noisy'):
                q, t = records[arm, variant], records['bf16', variant]
                if q['input_identity'] != t['input_identity'] or not torch.equal(q['action_initial_noise'], t['action_initial_noise']):
                    raise ValueError('Student and teacher did not use the same input and action noise')
                endpoints.append({'index': index, 'suite': case['row']['suite'], 'arm': arm, 'variant': variant,
                    'bf16_original_success': case.get('original_success', {}).get('bf16'),
                    'student_original_success': case.get('original_success', {}).get(arm),
                    'diagnostic_discordant': case.get('diagnostic_discordant', False),
                    **endpoint_metrics(torch, q, t, stats, definition)})
            qc, qn, tc, tn = (records[arm, 'clean'], records[arm, 'noisy'], records['bf16', 'clean'], records['bf16', 'noisy'])
            responses.append({'index': index, 'suite': case['row']['suite'], 'arm': arm,
                'bf16_original_success': case.get('original_success', {}).get('bf16'),
                'diagnostic_discordant': case.get('diagnostic_discordant', False),
                **response_metrics(torch, qc, qn, tc, tn, stats)})
    return {'endpoint_records': len(cases)*8, 'student_endpoint_metrics': len(endpoints), 'response_records': len(responses),
            'endpoints': {a: {v: means([r for r in endpoints if r['arm'] == a and r['variant'] == v], ENDPOINT_KEYS)
                              for v in ('clean', 'noisy')} for a in STUDENTS},
            'responses': {a: means([r for r in responses if r['arm'] == a], RESPONSE_KEYS) for a in STUDENTS}}, {'endpoints': endpoints, 'responses': responses}


def calibration_cases(root):
    plan = read(root/'data/calibration_pairs/definition.json')
    if len(plan['cases']) != 80 or plan['training_use'] is not False:
        raise ValueError('All80 diagnostic-only calibration pairs required')
    return [{**c, 'index': c['case_id']} for c in plan['cases']]


def calibration(p, root, cases, stats):
    result, details = sensor(p, root, calibration_cases(root), stats, prefix='calibration', expected_per_shard=40)
    result['calibration_contexts'] = 80
    return result, details


def readers(p, root, cases, stats):
    rows = []
    for unit in ('reader_0', 'reader_1'):
        receipt = completed(root, unit)
        if receipt.get('core_records_saved') != 280 or not receipt.get('bf16_self_parity_all_zero') or not receipt.get('q_self_parity_all_zero'):
            raise RuntimeError('Reader coverage/parity failed: '+unit)
    matrix = [('bf16', 'bf16'), *[('bf16', a) for a in STUDENTS], *[(a, 'bf16') for a in STUDENTS]]
    for case in cases:
        for variant in ('clean', 'noisy'):
            for reader, donor in matrix:
                unit = 'reader_'+str(case['shard'])
                r = read(root/'results'/unit/'records'/f"{case['index']}_{reader}_from_{donor}_{variant}.json")
                if (r['index'] != case['index'] or r['seed'] != case['sampler_seed'] or r['protocol'] != p['protocol']):
                    raise ValueError('Reader binding differs')
                rows.append({'index': case['index'], 'suite': case['row']['suite'], 'variant': variant,
                    'reader': reader, 'donor': donor, **{k: r['metrics'][k] for k in READER_KEYS}})
    comparison = {f'{reader}_from_{donor}': {v: means([r for r in rows if r['reader'] == reader and r['donor'] == donor and r['variant'] == v], READER_KEYS)
                  for v in ('clean', 'noisy')} for reader, donor in matrix}
    return {'reader_core_records': len(rows), 'comparison': comparison,
            'gradient_gate': read(root/'results/reader_0/gradient_gate.json')}, {'records': rows}


def episode_means(records, keys):
    indices = sorted({r['index'] for r in records})
    if not indices:
        raise ValueError('Cannot summarize empty trajectory records')
    rows = [{'index': index, 'states': sum(r['index'] == index for r in records),
             **{k: statistics.fmean(r[k] for r in records if r['index'] == index) for k in keys}}
            for index in indices]
    return {'episodes': len(rows), 'states': len(records), 'by_episode': rows,
            'episode_mean': {k: statistics.fmean(r[k] for r in rows) for k in keys}}


def trajectories(p, root, cases, stats):
    import torch
    case_map = {c['index']: c for c in cases}
    definition = loss_definition(p)
    endpoints, responses, reader_rows = [], [], []
    total_states = 0
    for unit in ('trajectory_0', 'trajectory_1'):
        receipt = completed(root, unit)
        if not receipt.get('source_replay_passed_all') or not receipt.get('native_and_reader_initial_noise_parity_valid'):
            raise RuntimeError('Trajectory source/noise parity failed: '+unit)
        for state in receipt['states']:
            index, collector, step = state['index'], state['collector'], state['step']
            case = case_map[index]
            base = {'index': index, 'collector': collector, 'step': step,
                    'suite': case['row']['suite'], 'bf16_original_success': case['original_success']['bf16']}
            stem = f'{collector}_{index}_step{step:03d}'
            records = {}
            for variant in ('clean', 'noisy'):
                for arm in ('bf16', *STUDENTS):
                    path = root/'results'/unit/'records'/f'{stem}_{variant}_native_{arm}.pt'
                    r = torch.load(path, map_location='cpu', weights_only=True)
                    if (r['protocol'] != p['protocol'] or r['index'] != index or r['step'] != step
                            or r['seed'] != state['seed'] or r['arm'] != arm or r['variant'] != variant):
                        raise ValueError('Trajectory native input binding differs')
                    records[arm, variant] = r
                    reader_path = root/'results'/unit/'records'/f'{stem}_{variant}_reader_bf16_from_{arm}.json'
                    reader = read(reader_path)
                    if (reader['protocol'] != p['protocol'] or reader['seed'] != state['seed']
                            or reader['input_identity'] != r['input_identity']
                            or reader['action_initial_noise_max_abs_vs_native'] != 0):
                        raise ValueError('Trajectory reader input binding differs')
                    reader_rows.append({**base, 'variant': variant, 'donor': arm,
                                        **{k: reader['metrics'][k] for k in READER_KEYS}})
                for arm in STUDENTS:
                    q, t = records[arm, variant], records['bf16', variant]
                    if q['input_identity'] != t['input_identity'] or not torch.equal(q['action_initial_noise'], t['action_initial_noise']):
                        raise ValueError('Trajectory student/teacher input differs')
                    endpoints.append({**base, 'arm': arm, 'variant': variant,
                                      **endpoint_metrics(torch, q, t, stats, definition)})
            for arm in STUDENTS:
                responses.append({**base, 'arm': arm, **response_metrics(torch,
                    records[arm, 'clean'], records[arm, 'noisy'],
                    records['bf16', 'clean'], records['bf16', 'noisy'], stats)})
            total_states += 1
    if {r['index'] for r in responses} != {c['index'] for c in cases if c['diagnostic_discordant']}:
        raise ValueError('Trajectory results must cover all five frozen discordant episodes')
    by_collector = {}
    for collector in ('per_channel_clip', 'relational'):
        by_collector[collector] = {
            'endpoints': {a: {v: episode_means([r for r in endpoints if r['collector'] == collector and r['arm'] == a and r['variant'] == v], ENDPOINT_KEYS)
                             for v in ('clean', 'noisy')} for a in STUDENTS},
            'responses': {a: episode_means([r for r in responses if r['collector'] == collector and r['arm'] == a], RESPONSE_KEYS) for a in STUDENTS},
            'bf16_reader': {a: {v: episode_means([r for r in reader_rows if r['collector'] == collector and r['donor'] == a and r['variant'] == v], READER_KEYS)
                               for v in ('clean', 'noisy')} for a in ('bf16', *STUDENTS)}}
    return {'discordant_episodes': 5, 'collectors': 2, 'actual_captured_states': total_states,
            'by_collector': by_collector,
            'aggregation': 'Average actual states within each case, then average the five cases; collectors remain separate; selected-case mechanism evidence only'}, {
                'endpoints': endpoints, 'responses': responses, 'reader_records': reader_rows}


def observed_features(torch, future):
    if tuple(future.shape) != (1, 48, 3, 14, 28):
        raise ValueError('Expected frozen Fast-WAM observed/future latent shape')
    frame = future[0, :, 0].float()
    parts = []
    for view in frame.split(14, dim=-1):
        parts.extend((view.mean((1, 2)), view.std((1, 2), correction=0)))
    return torch.cat(parts)


def coverage(p, root, cases, stats):
    import torch
    ccases = calibration_cases(root)
    features, rows = {}, []

    def add(source, case, variant, path, step=0, collector='none'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['protocol'] != p['protocol'] or saved['index'] != case['index']:
            raise ValueError('Coverage representation source differs')
        feature = observed_features(torch, saved['future'])
        if not bool(torch.isfinite(feature).all()):
            raise ValueError('Non-finite observed-frame representation')
        key = source, collector, case['index'], step, variant
        if key in features:
            raise ValueError('Duplicate coverage observation')
        features[key] = feature
        rows.append({'source': source, 'collector': collector, 'index': case['index'],
                     'step': step, 'variant': variant, 'suite': case['row']['suite'], 'feature_key': key})

    for prefix, selected in (('calibration', ccases), ('sensor', cases)):
        for shard in (0, 1):
            completed(root, prefix+'_'+str(shard))
        for case in selected:
            directory = root/'results'/f"{prefix}_{case['shard']}"/'records'
            for variant in ('clean', 'noisy'):
                add(prefix, case, variant, directory/f"{case['index']}_bf16_{variant}.pt")
    case_map = {c['index']: c for c in cases}
    for shard in (0, 1):
        unit = 'trajectory_'+str(shard)
        receipt = completed(root, unit)
        for state in receipt['states']:
            for variant in ('clean', 'noisy'):
                stem = f"{state['collector']}_{state['index']}_step{state['step']:03d}_{variant}_native_bf16.pt"
                add('trajectory', case_map[state['index']], variant,
                    root/'results'/unit/'records'/stem, state['step'], state['collector'])
    reference = torch.stack([features['calibration', 'none', c['index'], 0, 'clean'] for c in ccases])
    mean, scale = reference.mean(0), reference.std(0, correction=0).clamp_min(.02)
    reference = (reference-mean)/scale
    reference_ids = [c['index'] for c in ccases]
    for row in rows:
        key = row.pop('feature_key')
        value = (features[key]-mean)/scale
        distances = (reference-value).square().mean(-1).sqrt()
        # Exclude the same calibration context for both variants; clean references use leave-one-out.
        if row['source'] == 'calibration':
            distances[reference_ids.index(row['index'])] = torch.inf
        row['nearest_c80_clean_distance'] = float(distances.min())
        pair_key = (*key[:-1], 'clean')
        row['clean_pair_distance'] = float(((features[key]-features[pair_key])/scale).square().mean().sqrt())
    groups = sorted({(r['source'], r['collector'], r['variant']) for r in rows})
    comparison = {'/'.join(group): episode_means([r for r in rows if (r['source'], r['collector'], r['variant']) == group],
                    ('nearest_c80_clean_distance', 'clean_pair_distance')) for group in groups}
    return {'reference_contexts': 80, 'feature_dimensions': 192,
            'feature': 'BF16 observed VAE frame0; each camera channel spatial mean and population std; no predicted future frames',
            'normalization': 'mean/population std fitted only on original C80 clean BF16 observed-frame features; std floor0.02',
            'comparison': comparison,
            'scope': 'descriptive representation-distance proxy; not a proof of distribution support, success or a loss-selection gate; no additional model forwards'}, {'records': rows}


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=('aggregate_sensor', 'aggregate_reader', 'aggregate_trajectory', 'aggregate_calibration', 'aggregate_coverage'), required=True)
    args = parser.parse_args()
    p = read(Path(__file__).with_name('protocol.json'))
    root = Path(p['remote_root'])
    cases = read(root/'data/sensor_plan.json')['cases']
    stats = read(Path(p['source_relational_protocol']['source_dual_root'])/'results/fit_stats/summary.json')
    functions = {'aggregate_sensor': sensor, 'aggregate_reader': readers, 'aggregate_trajectory': trajectories,
                 'aggregate_calibration': calibration, 'aggregate_coverage': coverage}
    result, details = functions[args.unit](p, root, cases, stats)
    summary = {'protocol': p['protocol'], 'unit': args.unit, 'pbs_jobid': os.environ['PBS_JOBID'],
               'complete': True, 'sensor_cases': 40, 'training_updates': 0, **result,
               'boundary': 'same-state mechanism diagnosis on an already observed evaluation cohort; case is the unit; no new generalization or closed-loop success claim; cumulative commands are not physical displacement'}
    save(root/'results'/args.unit/'details.json', details)
    save(root/'results'/args.unit/'summary.json', summary)
    print({'unit': args.unit, 'complete': True, 'sensor_cases': 40}, flush=True)


if __name__ == '__main__':
    main()
