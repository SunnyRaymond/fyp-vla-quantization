"""Read completed paired episodes on CPU; freeze all sensor cases and discordances."""
import math
import os
from pathlib import Path

from control_common import guard, read, save


def first_chunk_metrics(candidate, teacher):
    error = [[float(a[d]) - float(b[d]) for d in range(6)]
             for a, b in zip(candidate[:10], teacher[:10])]
    if len(error) != 10 or not all(math.isfinite(x) for row in error for x in row):
        raise ValueError('Finite first10 x6 actions required')
    bias = [sum(row[d] for row in error) / 10 for d in range(6)]
    return {'motor_rmse': math.sqrt(sum(x*x for row in error for x in row) / 60),
            'chunk_bias': bias, 'chunk_bias_rms': math.sqrt(sum(x*x for x in bias) / 6),
            'max_prefix_command_error': max(abs(sum(error[t][d] for t in range(k)))
                                            for k in range(1, 11) for d in range(6))}


def main():
    guard()
    import source_aggregate as source
    p = read(Path(__file__).with_name('protocol.json'))
    root = Path(p['remote_root'])
    old = p['source_closed_protocol']
    manifest, rows = source.load_manifest(old['cohort']['reference_manifest'])
    references, _ = source._load_references(old, manifest, rows)
    _, new, _, _, checkpointed = source.load_workers(Path(p['source_closed_root'])/'results', old, manifest, rows)
    if checkpointed:
        raise RuntimeError('Source closed-loop evaluation is incomplete')
    arms = {'bf16': {i: references['bf16', i] for i in range(280)},
            'per_channel_clip': {i: references['per_channel_clip', i] for i in range(280)}, **new}
    selected = [i for i, row in enumerate(rows) if row['dimension'] == 'sensor_noise']
    if len(selected) != 40:
        raise RuntimeError('Frozen sensor dimension must contain exactly40 cases')
    discordant = [i for i in selected if arms['per_channel_clip'][i]['success'] and not arms['relational'][i]['success']]
    reverse = [i for i in selected if arms['relational'][i]['success'] and not arms['per_channel_clip'][i]['success']]
    if len(discordant) != 5 or reverse:
        raise RuntimeError('The reported sensor discordances do not reproduce')
    cases, metrics = [], []
    for ordinal, index in enumerate(selected):
        row = rows[index]
        record = {'index': index, 'ordinal': ordinal, 'shard': ordinal % 2, 'row': row,
                  'diagnostic_discordant': index in discordant,
                  'environment_seed': 800000000 + index*1000, 'sampler_seed': 900000000 + index*1000,
                  'original_success': {a: arms[a][index]['success'] for a in p['diagnostic']['arms']}}
        cases.append(record)
        action_records = {}
        for arm in p['diagnostic']['arms']:
            r = arms[arm][index]
            if r['env_seed'] != record['environment_seed'] or r['sampler_seed'] != record['sampler_seed']:
                raise RuntimeError('Original paired episode seed changed')
            chunks = r['first_action_chunks']
            if not chunks:
                raise RuntimeError('Missing original first action')
            action_records[arm] = {'action': chunks[0], 'success': r['success'], 'steps': r['steps']}
        save(root/'data/source_actions'/f'{index}.json', action_records)
        for arm in p['diagnostic']['arms'][1:]:
            metrics.append({'index': index, 'suite': row['suite'], 'arm': arm,
                            'discordant': index in discordant,
                            **first_chunk_metrics(action_records[arm]['action'], action_records['bf16']['action'])})
    plan = {'protocol': p['protocol'], 'complete': True, 'source_evaluation_job': p['source_closed_job'],
            'cases': cases, 'sensor_indices': selected, 'discordant_indices': discordant,
            'discordant_cases': [{**next(r for r in cases if r['index'] == index), 'capture_shard': n % 2}
                                 for n, index in enumerate(discordant)],
            'training_use': False, 'selection_scope': 'all40 sensor episodes; all5 MSE-only successes for mechanistic diagnosis'}
    save(root/'data/sensor_plan.json', plan)
    save(root/'results/prepare/first_chunk_metrics.json', {'complete': True, 'records': metrics,
         'boundary': 'same frozen initial observations and seeds only; normalized motor coordinates; cumulative command proxy is not measured displacement'})
    summary = {'protocol': p['protocol'], 'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
               'unit': 'prepare', 'sensor_cases': 40, 'discordant_cases': 5, 'reverse_discordant_cases': 0,
               'discordant_indices': discordant,
               'discordant_details': [{'index': r['index'], 'suite': r['row']['suite'],
                                      'subtype': r['row'].get('subtype'), 'variant_id': r['row']['variant_id']}
                                     for r in plan['discordant_cases']],
               'new_model_queries': 0, 'new_teacher_queries': 0, 'training_updates': 0,
               'frozen_heldout_untouched': True, 'diagnostic_states_not_training': True}
    save(root/'results/prepare/summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
