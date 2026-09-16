"""Independent, FP64 raw-vector replay. Execute only on CCDS CPU allocation."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recover-summary-overflow', action='store_true')
    args = parser.parse_args()
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import numpy as np

    path = args.input / 'raw_flow.npz'
    summary = json.loads((args.input / 'summary.json').read_text(encoding='utf-8'))
    recovery = None
    if args.recover_summary_overflow:
        # Preserve the original failed summary; this is a separate evidence path.
        if args.input.name != '64763' or summary.get('error') != 'RuntimeError: summary.json exceeds 64 KiB (110959 bytes)':
            raise ValueError('Recovery only covers the audited 64763 serialization failure')
        trace = (args.input / 'run.log').read_text()
        if 'line 1255, in _run' not in trace or '_write_small_summary(output / "summary.json", summary)' not in trace:
            raise ValueError('Trace is not the audited final serialization call')
        source = (args.input / 'flow_screen.py').read_text()
        if source.splitlines()[1254].strip() != '_write_small_summary(output / "summary.json", summary)':
            raise ValueError('Snapshot line does not match the failure trace')
        allocation_original = json.loads((args.input / 'allocation.json').read_text())
        if allocation_original.get('job_id') != '64763' or allocation_original.get('verified') is not True:
            raise ValueError('Original allocation evidence mismatch')
        gpu_text = (args.input / 'gpu_identity.csv').read_text().strip()
        if 'v100' not in gpu_text.casefold():
            raise ValueError('Original GPU is not V100')
        assets = args.input.parents[1] / 'smolvla'
        manifest = json.loads((assets / 'manifest.json').read_text())
        asset_brief = json.loads((args.input / 'asset_identity_brief.json').read_text())
        identity_sha = hashlib.sha256((assets / 'identity.json').read_bytes()).hexdigest()
        if identity_sha != asset_brief['full_identity_sha256']:
            raise ValueError('Current asset identity differs from GPU-saved identity')
        for sample in manifest['samples']:
            actual_sha = hashlib.sha256((assets / sample['sample_path']).read_bytes()).hexdigest()
            if actual_sha != sample['sha256']:
                raise ValueError('Prepared sample SHA mismatch')
        expected_episode_ids = [f"task{s['task_index']}:episode{s['episode_index']}:frame{s['frame_index']}" for s in manifest['samples']]
        recovery = {
            'original_job_status': 'FAILED 1:0 at final summary serialization',
            'original_summary_unchanged': True, 'gpu_evidence': gpu_text,
            'original_allocation': allocation_original,
            'runner_sha256': hashlib.sha256((args.input / 'flow_screen.py').read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256((assets / 'manifest.json').read_bytes()).hexdigest(),
            'full_asset_identity_sha256': identity_sha,
            'gpu_asset_identity_binding_pass': True,
            'passed_runtime_gates_evidence': 'Control-flow inference from preserved source and final-write exception after strict load, no-op, finite outputs, bypass checks and restore; detailed in-memory records were lost.',
            'not_recovered': ['GPU-calculated numeric metrics', 'per-layer quantizer details', 'numeric bypass digest', 'complete runtime summary'],
            'model_inference_repeated': False,
        }
    with np.load(path, allow_pickle=False) as raw:
        velocities = raw['velocities'].astype(np.float64)
        actions = raw['actions'].astype(np.float64)
        completed = raw['completed'].copy()
        arms = raw['arm_names'].tolist()
        noise = raw['noise'].copy()
        episode_ids = raw['episode_ids'].tolist()
        schema = raw['schema'].item()
    if schema != 'flow-geometry-drift-raw-v1':
        raise ValueError('Unexpected raw schema')
    if arms != ['FP32', 'backbone_W4', 'expert_W4']:
        raise ValueError('Unexpected arm order')
    if velocities.shape != (12, 2, 3, 10, 50, 32) or actions.shape != (12, 2, 3, 50, 7):
        raise ValueError('Raw shape mismatch')
    if completed.shape != (12,) or not completed.all() or not np.isfinite(velocities).all() or not np.isfinite(actions).all():
        raise ValueError('Incomplete or nonfinite vectors')
    if noise.shape != (2, 50, 32) or not np.isfinite(noise).all():
        raise ValueError('Saved initial noise shape/finite gate failed')
    if len(episode_ids) != 12 or len(set(episode_ids)) != 12:
        raise ValueError('Episode identity gate failed')
    sliced_v = velocities[:, :, :, :8, :8, :7].reshape(12, 2, 3, 8, 56)
    denominator = np.linalg.norm(sliced_v, axis=-1).sum(-1)
    if (denominator <= 1e-12).any():
        raise ValueError('Degenerate accel denominator: inconclusive')
    accel = 8 * np.linalg.norm(np.diff(sliced_v, axis=3), axis=-1).sum(-1) / denominator
    sliced_a = actions[:, :, :, :8, :]
    drift = np.square(sliced_a - sliced_a[:, :, :1]).mean(axis=(-1, -2))
    magnitude = np.linalg.norm(sliced_a.reshape(12, 2, 3, 56), axis=-1)

    def ranks(x):
        order = np.argsort(x, kind='stable')
        result = np.empty(len(x), dtype=np.float64)
        start = 0
        while start < len(x):
            end = start + 1
            while end < len(x) and x[order[end]] == x[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2
            start = end
        return result

    def spearman(x, y):
        if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
            return None
        return float(np.corrcoef(ranks(x), ranks(y))[0, 1])

    average_accel, average_drift, average_norm = (x.mean(1) for x in (accel, drift, magnitude))
    loci = {}
    for arm in (1, 2):
        rho_q = spearman(average_accel[:, arm], average_drift[:, arm])
        rho_fp = spearman(average_accel[:, 0], average_drift[:, arm])
        rho_norm = spearman(average_norm[:, arm], average_drift[:, arm])
        defined = all(x is not None for x in (rho_q, rho_fp, rho_norm))
        passed = defined and rho_q >= .5-1e-12 and rho_q-rho_fp >= .1-1e-12 and rho_q-rho_norm >= .1-1e-12
        loci[arms[arm]] = {
            'rho_q_accel': rho_q, 'rho_fp_accel': rho_fp, 'rho_q_action_norm': rho_norm,
            'delta_vs_fp_accel': None if not defined else rho_q-rho_fp,
            'delta_vs_q_norm': None if not defined else rho_q-rho_norm,
            'decision': 'no_binding_locus' if not defined else 'preliminary_go' if passed else 'mechanism_no_go',
            'episode_mean_drift': average_drift[:, arm].tolist(),
            'episode_mean_accel': average_accel[:, arm].tolist(),
            'episode_mean_action_norm': average_norm[:, arm].tolist(),
        }
    decisions = [item['decision'] for item in loci.values()]
    overall = ('preliminary_go' if decisions.count('preliminary_go') == 2 else
               ('inconclusive_partial_binding' if 'no_binding_locus' in decisions else 'scope_limited_preliminary_go') if 'preliminary_go' in decisions else
               'no_binding_locus' if decisions.count('no_binding_locus') == 2 else
               'inconclusive_partial_binding' if 'no_binding_locus' in decisions else 'mechanism_no_go')
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    if recovery is None:
        require(summary.get('status') == 'complete' and summary.get('completed_episodes') == 12, 'runtime completion')
        require('v100' in summary.get('gpu', {}).get('name', '').casefold(), 'V100 identity')
        require(summary.get('parameters', {}).get('noise_seeds') == [1701, 1702], 'fixed noise seeds')
        require(summary.get('input_manifest', {}).get('episode_ids') == episode_ids, 'raw/manifest episode binding')
        for key in ('action_exact_equal', 'velocity_finite', 'wrapper_restored'):
            require(summary.get('no_op', {}).get(key) is True, 'recording no-op gate: ' + key)
        require(summary.get('quantizer', {}).get('bypassed_modules_unchanged') is True, 'bypassed state integrity')
        require(summary.get('metrics', {}).get('overall_decision') == overall, 'GPU/CPU overall decision mismatch')
    else:
        require(episode_ids == expected_episode_ids, 'raw/asset manifest episode binding')
    for locus, item in ([] if recovery is not None else loci.items()):
        reference = summary.get('metrics', {}).get('loci', {}).get(locus, {})
        require(reference.get('status') == item['decision'], locus + ': decision mismatch')
        for cpu_key, gpu_key in [('rho_q_accel', 'q_accel_vs_drift'), ('rho_fp_accel', 'fp_accel_vs_drift'), ('rho_q_action_norm', 'q_action_norm_vs_drift')]:
            a, b = item[cpu_key], reference.get('rho', {}).get(gpu_key)
            require((a is None and b is None) or (a is not None and b is not None and abs(a-b) <= 1e-10), locus + ': ' + cpu_key + ' mismatch')
    report = {
        'schema': 'flow-geometry-independent-verification-v1', 'allocation': allocation,
        'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_job_dir': str(args.input), 'raw_vector_validation_pass': True,
        'runtime_record_checks_pass': not errors, 'errors': errors,
        'recovery': recovery,
        'gpu_cpu_metric_agreement_verified': recovery is None and not errors,
        'decision': overall, 'loci': loci,
        'accel_episode_noise_arm': accel.tolist(), 'drift_episode_noise_arm': drift.tolist(),
        'action_norm_episode_noise_arm': magnitude.tolist(),
        'aggregation': 'two noise seeds averaged inside each of 12 episodes before Spearman; exact ties use mean ranks',
        'scope': 'Independent numerical replay only; model identity, quantizer, identical input noise and recording no-op must also pass runtime gates. Offline FP fidelity diagnostic, not a detector or success test.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'decision': overall, 'loci': loci}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
