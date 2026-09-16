"""Recompute frozen semantic screen metrics from raw arrays on CCDS CPU."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import numpy as np

    source = args.input.resolve()
    summary = json.loads((source / 'summary.json').read_text())
    raw_path = source / 'raw_scores_actions.npz'
    arms = ['FP32', 'W4A32', 'FullInput-A8', 'Semantic-A8', 'Permuted-A8', 'PerChannel-A8']
    with np.load(raw_path, allow_pickle=False) as data:
        if data['schema'].item() != 'semantic-input-scales-raw-v1':
            raise ValueError('Raw schema mismatch')
        if data['arm_names'].tolist() != arms or data['dataset_indices'].tolist() != list(range(112, 118)):
            raise ValueError('Arm or data identity mismatch')
        actions, scores = data['actions'].astype(np.float64), data['scores'].astype(np.float64)
        completed, saved_top = data['completed'].copy(), data['top_indices'].copy()
    if actions.shape != (6, 64, 5, 10) or scores.shape != (6, 6, 64):
        raise ValueError('Raw shape mismatch')
    if not np.isfinite(actions).all() or not np.isfinite(scores).all() or not completed.all():
        raise ValueError('Incomplete/nonfinite experiment')
    errors = []
    if summary.get('status') != 'complete':
        errors.append('runtime experiment incomplete')
    noop = summary.get('no_op', {})
    if not all(noop.get(key) for key in ('exact_array_equal', 'stable_top6_equal', 'removed_without_residue')):
        errors.append('runtime no-op gate not verified')
    primary, full, regret = (np.zeros((6, 6)) for _ in range(3))
    for ep in range(6):
        top = [np.lexsort((np.arange(64), scores[ep, a]))[:6] for a in range(6)]
        if not np.array_equal(np.asarray(top), saved_top[ep]):
            errors.append(f'stable top6 mismatch episode {ep}')
        for a in range(6):
            delta = actions[ep, top[a]].mean(0) - actions[ep, top[0]].mean(0)
            primary[ep, a] = np.square(delta[0]).mean()
            full[ep, a] = np.square(delta).mean()
            r = scores[ep, 0, top[a]].mean() - scores[ep, 0, top[0]].mean()
            if r < -1e-12:
                errors.append(f'negative regret episode {ep} arm {arms[a]}')
            regret[ep, a] = max(0., r)
    means = primary.mean(0)
    comparisons = {}
    for a, arm in enumerate(arms):
        runtime_value = summary['metrics']['primary_mean_by_arm'][arm]
        agrees = bool(np.isclose(means[a], runtime_value, rtol=2e-6, atol=1e-10))
        comparisons[arm] = {'runtime': runtime_value, 'independent': float(means[a]), 'agrees': agrees}
        if not agrees:
            errors.append(f'primary aggregate mismatch: {arm}')
    improvements, counts = {}, {}
    for control in (2, 4):
        improvements[arms[control]] = None if means[control] <= 1e-12 else float((means[control]-means[3])/means[control])
        counts[arms[control]] = int((primary[:, 3] < primary[:, control]-1e-12).sum())
    near_zero = bool(means[2] <= 1e-12 or means[4] <= 1e-12)
    passed = not near_zero and all(x is not None and x >= .05 for x in improvements.values()) and all(x >= 4 for x in counts.values())
    verdict = 'no_binding_locus' if near_zero else 'preliminary_go' if passed else 'mechanism_no_go'
    if summary['metrics']['decision_gate'] != verdict:
        errors.append('runtime versus independent verdict disagreement')
    report = {
        'schema': 'semantic-independent-verification-v1', 'allocation': allocation,
        'source_job_dir': str(source), 'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'engineering_pass': not errors, 'errors': errors,
        'decision': 'implementation_failure' if errors else verdict,
        'primary_mean_by_arm': dict(zip(arms, means.tolist())),
        'primary_episode_arm': primary.tolist(),
        'full_horizon_mse_mean_by_arm': dict(zip(arms, full.mean(0).tolist())),
        'regret_mean_by_arm': dict(zip(arms, regret.mean(0).tolist())),
        'improvement_vs_controls': improvements, 'improved_episode_counts': counts,
        'runtime_comparison': comparisons,
        'scope': 'Raw score/action metrics replay only. Source, hooks, observer and weight checks rely on runtime records; no independent model inference. FP fidelity is not environment success.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('engineering_pass', 'decision', 'improvement_vs_controls', 'improved_episode_counts')}))


if __name__ == '__main__':
    main()
