"""Independently verify downloaded episode evidence and paired development gate."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import screen_runner as screen
from paired_stats import binomial_interval, paired_summary


def verify_job(job, targets, split):
    summaries = list(job.glob(f'evaluation/{split}/*/summary.json'))
    assert len(summaries) == 1, (job, summaries)
    path = summaries[0]
    summary = json.loads(path.read_text())
    assert summary['status'] == 'complete' and summary['failed'] == 0
    manifest = json.loads((targets / f'episode_manifest_{split}.json').read_text())
    entries = screen._validate_manifest(manifest, targets, split)
    expected = {row['episode_id']: row for row in entries}
    assert len(summary['episodes']) == screen.SPLIT_SIZES[split]
    seen = set()
    for row in summary['episodes']:
        episode_id = row['episode_id']
        assert episode_id not in seen
        seen.add(episode_id)
        target = expected[episode_id]
        for key in ('dataset_index', 'env_seed', 'cem_seed', 'layout', 'target_fingerprint'):
            assert row[key] == target[key], (episode_id, key)
        assert row['protocol'] == screen._protocol_config()
        assert row['mode'] == summary['mode'] and row['status'] == 'complete'
        assert screen._stable_identity(row['checkpoint_identity']) == screen._stable_identity(manifest['checkpoint_identity'])
        folder = path.parent / 'episodes' / f"episode_{row['local_index']:03d}"
        assert json.loads(folder.with_suffix('.json').read_text()) == row
        with np.load(folder / 'trajectory.npz') as arrays:
            states = arrays['states']
            actions = arrays['actions_executed']
            planned = arrays['planned_actions_normalized']
            assert np.isfinite(states).all() and np.isfinite(actions).all()
            assert np.isfinite(planned).all()
            rounds = row['mpc_points_visited']
            assert 1 <= rounds <= 12
            steps = row['executed_env_steps']
            assert steps == rounds * 25 <= 300
            assert actions.shape == (steps, 2)
            assert states.shape == (steps + 1, 2)
            assert planned.shape == (1, rounds * 5, 10)
            assert row['planned_model_actions'] == rounds * 5
            assert row['action_len_model_actions'] == (rounds * 5 if row['success'] else None)
            assert np.array_equal(arrays['actions_executed_prefix'], actions)
            assert np.array_equal(states[0], target['target']['state_0'][0])
            distance = float(np.linalg.norm(states[-1, :2] - target['target']['state_g'][0, :2]))
            assert abs(distance - row['goal_xy_distance']) < 1e-8
            assert bool(distance < 4.5) == row['success']
        if summary['record_pools']:
            with (folder / 'workload.pkl').open('rb') as stream:
                workload = pickle.load(stream)
            assert len(workload['cases']) == min(rounds, 2) * 2 == row['pool_count']
            for pool in workload['cases']:
                assert pool['episode_id'] == episode_id
                candidates = np.asarray(pool['candidates'])
                scores = np.asarray(pool['reference_scores'])
                assert candidates.shape == (300, 5, 10) and scores.shape == (300,)
                assert np.isfinite(candidates).all() and np.isfinite(scores).all()
                assert pool['cem_iteration'] in (1, 5)
                assert pool['mpc_point'] in (0, 1)
    assert seen == set(expected)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, nargs='+', required=True)
    parser.add_argument('--split', choices=tuple(screen.SPLIT_SIZES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summaries = [verify_job(job, args.targets, args.split) for job in args.jobs]
    by_mode = {summary['mode']: summary for summary in summaries}
    assert len(by_mode) == len(summaries)
    outcomes = {
        mode: {row['episode_id']: row['success'] for row in summary['episodes']}
        for mode, summary in by_mode.items()
    }
    counts = {mode: sum(rows.values()) for mode, rows in outcomes.items()}
    result = {'verified': True, 'split': args.split, 'success_counts': counts,
              'sample_size': screen.SPLIT_SIZES[args.split], 'outcomes': outcomes}
    result['success_rate_ci95'] = {mode: binomial_interval(count, result['sample_size'])
                                   for mode, count in counts.items()}
    pairs = {}
    for left in outcomes:
        for right in outcomes:
            if left == right:
                continue
            wins = sum(outcomes[left][key] and not outcomes[right][key] for key in outcomes[left])
            losses = sum(outcomes[right][key] and not outcomes[left][key] for key in outcomes[left])
            pairs[f'{left}_minus_{right}'] = paired_summary(wins, losses, len(outcomes[left]))
    result['paired_comparisons'] = pairs
    if args.split == 'development':
        assert set(counts) == {'FP32', 'all_W4', 'all_W8'}
        result['resource_gate'] = {
            'reference_adequate': counts['FP32'] >= 5,
            'w8_sanity': counts['all_W8'] >= counts['FP32'] - 1,
            'quantization_gap': counts['FP32'] - counts['all_W4'] >= 2,
        }
        result['proceed_to_test'] = all(result['resource_gate'].values())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
