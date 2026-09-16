"""Independent raw-score replay on CCDS CPU allocation; no model inference."""
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
    errors = []
    raw_path = source / 'raw_scores.npz'
    with np.load(raw_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    expected = {
        'actions': (6, 300, 50), 'scores_fp32': (6, 300),
        'scores_rtn': (6, 300), 'scores_rtn_w8': (6, 300),
        'scores_independent': (6, 3, 2, 300),
        'scores_antithetic': (6, 3, 2, 300), 'completed': (6,),
    }
    for key, shape in expected.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f'Raw shape mismatch: {key}')
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'Nonfinite raw data: {key}')
    if not arrays['completed'].all() or summary.get('status') != 'complete':
        errors.append('incomplete experiment')
    if not np.array_equal(arrays['scores_independent'][:, :, 0], arrays['scores_antithetic'][:, :, 0]):
        errors.append('pair arms do not share first member exactly')
    if summary.get('parameters', {}).get('dataset_indices') != list(range(96, 102)):
        errors.append('wrong source index range')
    if summary.get('model_structure', {}).get('num_hist') != 1:
        errors.append('unverified model structure')
    if not summary.get('gates', {}).get('engineering', {}).get('passed'):
        errors.append('runtime engineering gate did not pass')
    if not summary.get('dataset_mapping', {}).get('fresh_disjoint_from_excluded'):
        errors.append('underlying source separation not verified')
    if summary.get('source_identity', {}).get('checkpoint_sha256') != '8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b':
        errors.append('checkpoint identity mismatch')

    def evaluate(ep, scores):
        reference = arrays['scores_fp32'][ep].astype(np.float64)
        actions = arrays['actions'][ep].astype(np.float64)
        ref_idx = np.lexsort((np.arange(300), reference))[:30]
        idx = np.lexsort((np.arange(300), scores))[:30]
        regret = float(reference[idx].mean() - reference[ref_idx].mean())
        if regret < -1e-12:
            raise ValueError('Negative top-k regret violates ordering')
        mse = float(np.mean((actions[idx].mean(0) - actions[ref_idx].mean(0)) ** 2))
        return max(0.0, regret), mse

    metrics = {}
    for arm, key in [('rtn', 'scores_rtn'), ('rtn_w8', 'scores_rtn_w8'),
                     ('independent', 'scores_independent'), ('antithetic', 'scores_antithetic')]:
        rows = []
        for ep in range(6):
            if arm in ('rtn', 'rtn_w8'):
                rows.append([evaluate(ep, arrays[key][ep].astype(np.float64))] * 3)
            else:
                # Match the frozen implementation's FP32 score averaging;
                # metric arithmetic uses FP64 after the selection scores exist.
                rows.append([evaluate(ep, arrays[key][ep, seed].mean(0).astype(np.float64)) for seed in range(3)])
        values = np.asarray(rows)
        metrics[arm] = {
            'regret_episode_seed': values[:, :, 0].tolist(),
            'action_mse_episode_seed': values[:, :, 1].tolist(),
            'regret_mean': float(values[:, :, 0].mean()),
            'action_mse_mean': float(values[:, :, 1].mean()),
        }
    anti, ind, rtn, w8 = (metrics[k] for k in ('antithetic', 'independent', 'rtn', 'rtn_w8'))
    ar, ir = (np.array(x['regret_episode_seed']) for x in (anti, ind))
    ratio = None if ind['regret_mean'] <= 1e-12 else (ind['regret_mean'] - anti['regret_mean']) / ind['regret_mean']
    gates = {
        'headroom': ind['regret_mean'] > 1e-12,
        'relative_regret_improvement_ge_5pct': ratio is not None and ratio >= .05,
        'improved_episodes_ge_4': int((ar.mean(1) < ir.mean(1)).sum()) >= 4,
        'improved_rounding_seeds_ge_2': int((ar.mean(0) < ir.mean(0)).sum()) >= 2,
        'regret_not_worse_than_rtn': anti['regret_mean'] <= rtn['regret_mean'] + 1e-12,
        'action_mse_not_worse_than_independent': anti['action_mse_mean'] <= ind['action_mse_mean'] + 1e-12,
        'action_mse_not_worse_than_rtn': anti['action_mse_mean'] <= rtn['action_mse_mean'] + 1e-12,
    }
    numerical_w8_dominance = w8['regret_mean'] <= anti['regret_mean'] + 1e-12 and w8['action_mse_mean'] <= anti['action_mse_mean'] + 1e-12
    comparison = {}
    for arm in metrics:
        saved = summary['metrics']['aggregate'][arm]
        comparison[arm] = {}
        for metric, saved_key in [('regret_mean', 'selected_top30_fp_score_regret'), ('action_mse_mean', 'elite_mean_action_mse')]:
            left, right = metrics[arm][metric], saved[saved_key]['mean_over_episodes']
            # Runtime action-mean arithmetic was FP32; recomputation is FP64.
            agrees = bool(np.isclose(left, right, rtol=2e-6, atol=1e-10))
            comparison[arm][metric] = {'runtime': right, 'independent': left, 'agrees': agrees}
            if not agrees:
                errors.append(f'runtime aggregate mismatch: {arm}/{metric}')
    expected_mechanism = 'preliminary_go' if all(gates.values()) else 'mechanism_no_go'
    if summary.get('mechanism_verdict') != expected_mechanism:
        errors.append('runtime versus independent mechanism verdict disagreement')
    report = {
        'schema': 'antithetic-independent-score-verification-v1',
        'allocation': allocation, 'source_job_dir': str(source),
        'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'engineering_pass': not errors, 'errors': errors,
        'metrics': metrics, 'gates': gates,
        'runtime_comparison': comparison,
        'relative_regret_improvement': ratio,
        'improved_episodes': int((ar.mean(1) < ir.mean(1)).sum()),
        'improved_rounding_seeds': int((ar.mean(0) < ir.mean(0)).sum()),
        'mechanism_verdict': ('implementation_failure' if errors else 'preliminary_go' if all(gates.values()) else 'mechanism_no_go'),
        'w8_numerically_dominates': numerical_w8_dominance,
        'scope': 'Independent raw score/action recomputation only; quantizer/restore/source checks rely on saved runtime records, not independent model re-inference. FP fidelity is not environment success.',
        'statistical_unit': 'six episodes; three rounding seeds are repeated measurements',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('engineering_pass', 'mechanism_verdict', 'relative_regret_improvement', 'improved_episodes', 'improved_rounding_seeds', 'w8_numerically_dominates')}))


if __name__ == '__main__':
    main()
