"""Independent CPU replay of the frozen gradient screen; no model inference."""
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

    raw_path = args.input / 'raw_gradient.npz'
    engineering = json.loads((args.input / 'engineering.json').read_text())
    summary = json.loads((args.input / 'summary.json').read_text())
    with np.load(raw_path, allow_pickle=False) as raw:
        if raw['schema'].item() != 'action-gradient-geometry-raw-v1':
            raise ValueError('Raw schema mismatch')
        arrays = {k: raw[k].copy() for k in (
            'actions', 'scores', 'gradients', 'fp_step_scores', 'base_scores',
            'fd_values', 'completed', 'dataset_indices', 'source_episode_ids', 'arm_names')}
    expected_shapes = {
        'actions': (6, 64, 2, 10), 'scores': (6, 2, 64),
        'gradients': (6, 2, 4, 2, 10), 'fp_step_scores': (6, 2, 4),
        'base_scores': (6, 4), 'fd_values': (6, 2, 2),
        'completed': (6,), 'dataset_indices': (6,), 'source_episode_ids': (6,),
        'arm_names': (2,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f'{key}: expected {shape}, got {arrays[key].shape}')
    if not arrays['completed'].all():
        raise ValueError('Incomplete raw record')
    if arrays['dataset_indices'].tolist() != list(range(118, 124)):
        raise ValueError('Frozen dataset indices mismatch')
    if len(set(arrays['source_episode_ids'].tolist())) != 6:
        raise ValueError('Duplicate underlying episode IDs')
    if arrays['arm_names'].tolist() != ['FP32', 'predictor_W4']:
        raise ValueError('Arm identity mismatch')
    runtime_checks = {
        'complete': engineering.get('status') == 'complete' and summary.get('status') == 'complete',
        'six_episodes': engineering.get('completed_episodes') == 6,
        'restored': engineering.get('weights_restored_exactly') is True,
        'encoder_unchanged': engineering.get('predictor_only_encoder_untouched') is True,
        'V100': 'v100' in summary.get('gpu', {}).get('name', '').casefold(),
        'dataset_identity': summary.get('dataset_mapping', {}).get('source_episode_ids') == arrays['source_episode_ids'].tolist(),
        'disjoint_episodes': summary.get('dataset_mapping', {}).get('fresh_disjoint_from_excluded') is True,
        'checkpoint': summary.get('runtime_identity', {}).get('checkpoint_sha256') == '8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b',
        'source_commit': summary.get('runtime_identity', {}).get('source_commit') == '0a9492fa12044b852ae9e001cc74604b79c8bb0c',
        'dinov2_commit': summary.get('runtime_identity', {}).get('dinov2_source_commit') == '7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    }
    for key in ('actions', 'scores', 'gradients', 'fp_step_scores', 'base_scores', 'fd_values'):
        arrays[key] = arrays[key].astype(np.float64)
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'Nonfinite {key}')

    def ranks(values):
        order = np.argsort(values, kind='stable')
        result = np.empty(len(values), dtype=np.float64)
        first = 0
        while first < len(values):
            last = first + 1
            while last < len(values) and values[order[last]] == values[order[first]]:
                last += 1
            result[order[first:last]] = (first + last - 1) / 2
            first = last
        return result

    def rho(x, y):
        a, b = ranks(x), ranks(y)
        if np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    records = []
    for ep in range(6):
        scores = arrays['scores'][ep]
        std_fp = float(np.std(scores[0]))
        correlation = rho(scores[0], scores[1])
        error = (scores[1] - scores[1].mean()) - (scores[0] - scores[0].mean())
        nrmse = float(np.sqrt(np.mean(error**2)) / std_fp) if std_fp > 1e-8 else None
        global_binding = (correlation is not None and nrmse is not None
                          and correlation >= .9 - 1e-12 and nrmse <= .25 + 1e-12)
        gradients = arrays['gradients'][ep].reshape(2, 4, 20)
        norms = np.linalg.norm(gradients, axis=-1)
        valid_norms = bool((norms > 1e-8).all())
        cosines = ((gradients[0] * gradients[1]).sum(-1) / (norms[0] * norms[1])) if valid_norms else None
        gains = arrays['base_scores'][ep][None, :] - arrays['fp_step_scores'][ep]
        local_binding = valid_norms and bool((gains[0] > 1e-7).all())
        bad_geometry = (local_binding and float(np.median(cosines)) <= .5 + 1e-12
                        and float(gains[1].mean()) <= .5 * float(gains[0].mean()) + 1e-12)
        fd = (arrays['fd_values'][ep, :, 0] - arrays['fd_values'][ep, :, 1]) / .01
        dot = norms[:, 0]
        fd_error = np.abs(fd - dot)
        fd_tolerance = .1 * np.abs(dot) + 1e-4
        fd_pass = bool((fd_error <= fd_tolerance + 1e-12).all())
        records.append({
            'dataset_index': int(arrays['dataset_indices'][ep]),
            'source_episode_id': str(arrays['source_episode_ids'][ep]),
            'spearman': correlation, 'centered_score_nrmse': nrmse,
            'global_binding': global_binding, 'local_binding': local_binding,
            'gradient_norms': norms.tolist(),
            'gradient_cosines': None if cosines is None else cosines.tolist(),
            'median_cosine': None if cosines is None else float(np.median(cosines)),
            'fp_objective_step_gains': gains.tolist(),
            'bad_local_geometry': bool(bad_geometry),
            'joint_signal': bool(global_binding and bad_geometry),
            'fd_derivative': fd.tolist(), 'autograd_dot': dot.tolist(),
            'fd_abs_error': fd_error.tolist(), 'fd_tolerance': fd_tolerance.tolist(),
            'fd_pass': fd_pass,
        })
    identifiable = sum(r['global_binding'] and r['local_binding'] for r in records)
    signals = sum(r['joint_signal'] for r in records)
    engineering_pass = all(r['fd_pass'] for r in records) and all(runtime_checks.values())
    decision = ('implementation_inconclusive' if not engineering_pass else
                'conditional_signal' if signals >= 4 else
                'inconclusive_binding' if identifiable < 4 else 'mechanism_no_go')
    report = {
        'schema': 'action-gradient-independent-verification-v1',
        'allocation': allocation, 'source_job_dir': str(args.input),
        'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'raw_schema_finite_complete_pass': True, 'fd_engineering_pass': all(r['fd_pass'] for r in records),
        'combined_engineering_pass': engineering_pass,
        'runtime_checks': runtime_checks,
        'identifiable_episodes': int(identifiable), 'joint_signal_episodes': int(signals),
        'decision': decision, 'episodes': records,
        'scope': 'FP64 independent raw replay. Runtime source, restore and graph checks must separately pass. Neither environmental gradients nor task success are measured.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'decision': decision, 'identifiable': identifiable, 'joint_signal': signals}))


if __name__ == '__main__':
    main()
