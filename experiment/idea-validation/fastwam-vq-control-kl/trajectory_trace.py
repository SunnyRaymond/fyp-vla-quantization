"""Export a small, complete state table from the verified CPU trajectory report."""
import os
from pathlib import Path

from aggregate import completed, STUDENTS
from control_common import guard, read, save


def unique_records(records, suffix):
    keys = ('collector', 'index', 'step', *suffix)
    result = {tuple(row[key] for key in keys): row for row in records}
    if len(result) != len(records):
        raise ValueError('Duplicate trajectory metric records')
    return result


def main():
    guard()
    p = read(Path(__file__).with_name('protocol.json'))
    root = Path(p['remote_root'])
    receipt = completed(root, 'aggregate_trajectory')
    details = read(root / 'results/aggregate_trajectory/details.json')
    endpoints = unique_records(details['endpoints'], ('arm', 'variant'))
    responses = unique_records(details['responses'], ('arm',))
    readers = unique_records(details['reader_records'], ('donor', 'variant'))
    states = [state for shard in (0, 1)
              for state in completed(root, f'trajectory_{shard}')['states']]
    if len(states) != receipt['actual_captured_states']:
        raise ValueError('Compact table must cover all actual captured states')
    rows = []
    for state in states:
        key = state['collector'], state['index'], state['step']
        arms = {}
        for arm in STUDENTS:
            noisy = endpoints[key + (arm, 'noisy')]
            arms[arm] = {
                **{name: noisy[name] for name in (
                    'motor_rmse', 'action_kl', 'latent_nmse',
                    'command_chunk_bias_rms', 'command_prefix_cumsum_max_abs')},
                'clean_motor_rmse': endpoints[key + (arm, 'clean')]['motor_rmse'],
                'response_kl': responses[key + (arm,)]['action_response_kl'],
                'response_bias_rms': responses[key + (arm,)]['command_chunk_bias_rms'],
                'reader_control_kl': readers[key + (arm, 'noisy')]['softbin_first10_motor_kl'],
                'reader_motor_rmse': readers[key + (arm, 'noisy')]['first10_motor_rmse']}
        rows.append({'collector': key[0], 'index': key[1], 'step': key[2],
                     'bf16_original_success': noisy['bf16_original_success'], 'arms': arms})
    save(root / 'results/trajectory_trace/summary.json', {
        'protocol': p['protocol'], 'unit': 'trajectory_trace',
        'pbs_jobid': os.environ['PBS_JOBID'], 'complete': True,
        'source_aggregate_job': receipt['pbs_jobid'],
        'actual_captured_states': len(rows), 'records': rows,
        'new_model_forwards': 0, 'training_updates': 0,
        'scope': 'all five selected discordant cases and both collectors; repeated states are diagnostic, never training or independent episodes'})


if __name__ == '__main__':
    main()
