"""Create fresh small CAL/DEV reference trajectories using the verified adapter."""
import argparse
import json
from pathlib import Path
import time
from allocation_guard import require_allocation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    allocation = require_allocation()
    import numpy as np
    import smoke_runner as smoke
    import screen_runner as screen
    started = time.monotonic()
    runtime = smoke._runtime(args.root)
    targets = args.base / 'targets'
    definitions = {'newcal': (42, 600000), 'newdev': (46, 700000)}
    for split, (offset, namespace) in definitions.items():
        screen.SPLIT_SIZES[split] = 4
        screen.SPLIT_NAMESPACES[split] = namespace
        screen.SPLIT_DATASET_OFFSETS[split] = offset
    oldprints = set()
    for path in (args.root/'rankcal_screen/targets').glob('episode_manifest_*.json'):
        prior = json.loads(path.read_text())
        screen._validate_runtime_identity(prior, runtime)
        oldprints.update(row['target_fingerprint'] for row in prior['episodes'])
    assert len(oldprints) == 40
    groups = smoke._linear_groups(runtime['model'])
    label, metadata = screen._configure_mode(runtime, 'FP32', None, groups)
    records = {}
    for split in definitions:
        manifest = screen._prepare_targets(runtime, targets, split)
        manifest['dataset_index_strategy'] = 'new CAL 42-45; new DEV 46-49; old 0-41 excluded'
        screen._atomic_json(targets / f'episode_manifest_{split}.json', manifest)
        _, _, episodes = screen._load_targets(targets, split, runtime)
        assert not oldprints.intersection(row['target_fingerprint'] for row in episodes)
        pools, rows = [], []
        for episode in episodes:
            result, _, cases = screen._run_episode(runtime, episode, args.output/split/f"episode_{episode['local_index']:03d}", True, label, metadata)
            assert result['status'] == 'complete', result
            # Keep one predeclared MPC point per episode for bounded search cost.
            chosen = [case for case in cases if case['mpc_point'] == 0]
            assert sorted(case['cem_iteration'] for case in chosen) == [1, 5]
            pools.extend(chosen)
            rows.append({key: result[key] for key in ('episode_id', 'dataset_index', 'env_seed', 'cem_seed', 'target_fingerprint', 'success', 'executed_env_steps')})
            with np.load(args.output/split/f"episode_{episode['local_index']:03d}"/'trajectory.npz') as arrays:
                distance = float(np.linalg.norm(arrays['states'][-1, :2] - episode['target']['state_g'][0, :2]))
                assert abs(distance-result['goal_xy_distance']) < 1e-8
                assert bool(distance < 4.5) == result['success']
        screen._atomic_pickle(args.base/'pools'/f'{split}.pkl', {'split': split, 'cases': pools, 'planner': screen._protocol_config(), 'runtime_identity': smoke._checkpoint_identity(runtime)})
        records[split] = {'episode_count': len(rows), 'pool_count': len(pools), 'episodes': rows}
    screen._atomic_json(args.output/'summary.json', {'status': 'complete', 'allocation': allocation, 'splits': records, 'old_target_overlap': 0, 'elapsed_seconds': time.monotonic()-started, 'claim': 'FP32 collection only; no new method success evidence'})


if __name__ == '__main__':
    main()
