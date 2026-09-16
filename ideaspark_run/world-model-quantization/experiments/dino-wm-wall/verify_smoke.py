"""Independently verify the compact smoke artifacts, without GPU or checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', type=Path)
    args = parser.parse_args()
    root = args.artifacts
    check = root / 'check'
    audit = json.loads((check / 'check_audit.json').read_text())
    assert audit['status'] == 'complete' and audit['case_count'] == 2
    assert audit['checkpoint_identity']['recorded_epoch'] == 65
    assert audit['checkpoint_identity']['execution'] == 'emulation_only'
    arrays = np.load(check / 'shared_pool_scores.npz')
    sites = json.loads((check / 'site_manifest.json').read_text())['groups']
    assert len(sites) == 18
    family_counts = {}
    for family, count in [('encoder', 12), ('predictor', 6)]:
        rows = [g for g in sites if g['family'] == family]
        assert len(rows) == count
        costs = {(g['numel'], g['scale_count'], g['logical_weight_bytes_W4'],
                  g['logical_weight_bytes_W8']) for g in rows}
        assert len(costs) == 1, f'Unequal allocation cost within {family}'
        family_counts[family] = {'groups': count, 'cost_per_group': list(costs)[0]}
    numeric = []
    for case in audit['cases']:
        cid = case['case_id']
        prefix = f'case_{cid:02d}_'
        assert case['replay_deterministic'] and case['replay_env_steps'] == 25
        assert case['fp32_noop_exact'] and case['fp32_restore_exact']
        assert case['closed_loop'] == 'complete'
        assert case['original_cem_bridge']['exact_equal']
        replay = np.load(check / f'case_{cid:02d}' / 'replay_25steps.npz')
        assert replay['actions'].shape == (1, 25, 2)
        assert np.any(replay['actions'] != 0) and np.isfinite(replay['states']).all()
        assert replay['states'].shape[1] == 26
        reference = arrays[prefix + 'FP32_scores']
        assert reference.shape == (300,) and np.isfinite(reference).all()
        assert np.array_equal(reference, arrays[prefix + 'NOOP_scores'])
        ref_order = np.argsort(reference, kind='stable')
        for mode in ['FP32', 'NOOP', 'all_W4', 'all_W8']:
            scores = arrays[prefix + mode + '_scores']
            assert scores.shape == reference.shape and np.isfinite(scores).all()
            elite = np.argsort(scores, kind='stable')[:30]
            assert np.array_equal(elite, arrays[prefix + mode + '_elite_indices'])
            mean_action = arrays[prefix + 'candidates'][elite].mean(axis=0)
            assert np.allclose(mean_action, arrays[prefix + mode + '_elite_mean_action'], atol=1e-7)
            numeric.append({'case': cid, 'mode': mode,
                            'score_max_abs_diff': float(np.max(np.abs(scores-reference))),
                            'elite_overlap': len(set(elite) & set(ref_order[:30])) / 30})
    assert len({c['env_seed'] for c in audit['cases']}) == 2
    assert len({str((c['initial_state'], c['goal_state'], c['layout'])) for c in audit['cases']}) == 2
    controller = json.loads((root / 'controller.json').read_text())
    assert controller['status'] == 'completed'
    groups = {}
    for count in [1, 2]:
        rows = [json.loads((root / f'workers_{count}' / f'worker_{w}' / 'bench.json').read_text())
                for w in range(count)]
        records = sum(row['records'] for row in rows)
        assert records == 24
        for row in rows:
            assert row['status'] == 'complete' and row['workers'] == count
            for repeat in row['scores']:
                for case in repeat['cases']:
                    ref = arrays[f"case_{case['case_id']:02d}_FP32_scores"]
                    actual = np.asarray(case['scores'])
                    assert np.isfinite(actual).all()
                    assert np.allclose(ref, actual, rtol=0, atol=1e-6)
                    assert np.array_equal(np.argsort(ref, kind='stable')[:30],
                                          np.argsort(actual, kind='stable')[:30])
        elapsed = max(r['end_monotonic'] for r in rows) - min(r['start_monotonic'] for r in rows)
        groups[str(count)] = {'records': records, 'elapsed_seconds': elapsed,
                              'records_per_second': records / elapsed,
                              'peak_allocated_gib_per_worker': [r['cuda_peak_allocated_gib'] for r in rows],
                              'peak_reserved_gib_per_worker': [r['cuda_peak_reserved_gib'] for r in rows]}
    speedup = groups['2']['records_per_second'] / groups['1']['records_per_second']
    summary = {'smoke_verified': True, 'family_cost_check': family_counts,
               'numeric_checks': numeric, 'benchmarks': groups,
               'two_worker_throughput_ratio': speedup,
               'recommended_workers_per_gpu': 2 if speedup >= 1.1 else 1,
               'scope': 'engineering_smoke_only_not_RankCal_effectiveness',
               'check_seconds': audit['elapsed_seconds'],
               'suite_seconds': controller['elapsed_seconds']}
    (root / 'verification.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
