"""Independent frozen conditional-marginal screen; CPU allocation required."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import numpy as np

    raw_path = args.input / 'raw_marginal.npz'
    with np.load(raw_path, allow_pickle=False) as raw:
        if raw['schema'].item() != 'conditional-marginal-raw-v1':
            raise ValueError('Wrong raw schema')
        data = {key: raw[key].copy() for key in ('actions', 'noise', 'completed',
            'episode_ids', 'arm_names', 'batch_check_batched', 'batch_check_individual')}
    shapes = {'actions': (8, 2, 64, 50, 7), 'noise': (64, 50, 32),
              'completed': (8, 2), 'episode_ids': (8,), 'arm_names': (2,),
              'batch_check_batched': (2, 4, 50, 7), 'batch_check_individual': (2, 4, 50, 7)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f'Wrong {key} shape: {data[key].shape}')
    if data['arm_names'].tolist() != ['FP32', 'expert_W4']:
        raise ValueError('Wrong arm order')
    if not data['completed'].all() or len(set(data['episode_ids'].tolist())) != 8:
        raise ValueError('Incomplete or duplicate episodes')
    for key in ('actions', 'noise', 'batch_check_batched', 'batch_check_individual'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Nonfinite {key}')
        data[key] = data[key].astype(np.float64)

    noise_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in data['noise']]
    import torch
    expected_noise = np.concatenate([
        torch.randn((32, 50, 32), generator=torch.Generator(device='cpu').manual_seed(seed), dtype=torch.float32).numpy()
        for seed in (1901, 1902)], axis=0).astype(np.float64)
    batch_errors = np.max(np.abs(data['batch_check_batched'] - data['batch_check_individual']), axis=(1, 2, 3))
    raw_checks = {
        '64_unique_noises': len(set(noise_hashes)) == 64,
        'noise_matches_frozen_CPU_generators': bool(np.array_equal(data['noise'], expected_noise)),
        'FP_batch_gate': bool(batch_errors[0] <= 1e-5),
        'Q_batch_gate': bool(batch_errors[1] <= 1e-5),
        'batch_probe_matches_recorded_actions': bool(np.max(np.abs(data['batch_check_batched'] - data['actions'][0, :, :4])) <= 1e-5),
    }
    rng = np.random.default_rng(1903)
    random_directions = rng.standard_normal((9, 7))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
    directions = np.concatenate((np.eye(7), random_directions), axis=0)

    def swd(x, y):
        return float(np.abs(np.sort(x @ directions.T, axis=0) - np.sort(y @ directions.T, axis=0)).mean())

    def pair_norm(x, y):
        return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)

    def energy(x, y):
        value = float(2*pair_norm(x, y).mean() - pair_norm(x, x).mean() - pair_norm(y, y).mean())
        if value < -1e-10:
            raise ValueError('Energy V-statistic unexpectedly negative')
        return max(0.0, value)

    # Analytic checks of the distance definitions, executed here on cluster CPU.
    zeros = np.zeros((32, 7))
    translated = zeros.copy()
    translated[:, 0] = 0.3
    if abs(swd(zeros, translated) - np.abs(0.3*directions[:, 0]).mean()) > 1e-12:
        raise ValueError('SWD known-translation check failed')
    if energy(zeros, zeros) != 0 or abs(energy(zeros, translated) - .6) > 1e-12:
        raise ValueError('Energy analytic check failed')

    records = []
    for index, episode_id in enumerate(data['episode_ids']):
        fp, quant = data['actions'][index, :, :, 0, :]
        fa, fb, qa, qb = fp[:32], fp[32:], quant[:32], quant[32:]
        wff, eff, wqq = swd(fa, fb), energy(fa, fb), swd(qa, qb)
        wqf = (swd(qa, fb) + swd(qb, fa))/2
        eqf = (energy(qa, fb) + energy(qb, fa))/2
        dpair = float(np.square(fp-quant).mean())
        dnoise = float(np.square(fa[:, None, :] - fb[None, :, :]).mean())
        shift_vector = np.zeros(7)
        shift_vector[0] = np.sqrt(7*.25*dnoise)
        wcontrol, econtrol = swd(fa, fb+shift_vector), energy(fa, fb+shift_vector)
        reference = wff > 1e-4 and eff > 1e-8
        wr, er = (wqf/wff if wff > 0 else None), (eqf/eff if eff > 0 else None)
        wcr, ecr = (wcontrol/wff if wff > 0 else None), (econtrol/eff if eff > 0 else None)
        qqr = wqq/wff if wff > 0 else None
        mapping = dnoise > 1e-6 and dpair/dnoise >= .25
        projected_fp, projected_q = fp @ directions.T, quant @ directions.T
        fp_iqr = np.diff(np.quantile(projected_fp, [.25, .75], axis=0, method='linear'), axis=0)[0]
        q_iqr = np.diff(np.quantile(projected_q, [.25, .75], axis=0, method='linear'), axis=0)[0]
        valid = fp_iqr > 1e-6
        ratios = np.divide(q_iqr, fp_iqr, out=np.full(16, np.nan), where=valid)
        valid_count = int(valid.sum())
        spread_good_count = int((valid & (ratios >= .5) & (ratios <= 2)).sum())
        collapsed_count = int((valid & (ratios < .5)).sum())
        collapse = bool(collapsed_count >= 8 and qqr is not None and qqr < .5)
        sensitivity = bool(reference and wcr >= 1.5 and ecr >= 1.25)
        identifiable = bool(reference and valid_count >= 13 and sensitivity)
        in_band = bool(reference and wr <= 1.25 and er <= 1.25)
        shifted = bool(reference and wr >= 1.5 and er >= 1.25)
        positive = bool(mapping and identifiable and in_band and spread_good_count >= 13 and not collapse)
        negative = bool(mapping and identifiable and shifted)
        if not reference:
            label = 'reference_degenerate'
        elif valid_count < 13:
            label = 'spread_inconclusive'
        elif not sensitivity:
            label = 'sensitivity_inconclusive'
        elif not mapping:
            label = 'mapping_binding_absent'
        elif positive:
            label = 'bounded_dissociation'
        elif negative:
            label = 'shift_with_collapse' if collapse else 'marginal_shift'
        else:
            label = 'statistical_inconclusive'
        records.append({
            'episode_id': str(episode_id), 'label': label,
            'Dpair': dpair, 'Dnoise': dnoise, 'mapping_ratio': dpair/dnoise if dnoise > 0 else None,
            'W_FF': wff, 'E_FF': eff, 'W_QF': wqf, 'E_QF': eqf, 'W_QQ': wqq,
            'W_ratio': wr, 'E_ratio': er, 'QQ_ratio': qqr,
            'translation_vector': shift_vector.tolist(), 'translation_W_ratio': wcr, 'translation_E_ratio': ecr,
            'reference_identifiable': reference, 'mapping_binding': mapping, 'sensitivity_identifiable': sensitivity,
            'valid_projection_count': valid_count, 'spread_good_count': spread_good_count,
            'collapsed_projection_count': collapsed_count, 'collapse': collapse,
            'FP_projection_IQR': fp_iqr.tolist(), 'Q_projection_IQR': q_iqr.tolist(),
            'projection_IQR_ratios': [float(value) if np.isfinite(value) else None for value in ratios],
            'positive': positive, 'negative': negative,
        })
    positive_count = sum(row['positive'] for row in records)
    negative_count = sum(row['negative'] for row in records)
    collapse_count = sum(row['collapse'] for row in records)
    numerical = ('bounded_dissociation_preliminary_go' if positive_count >= 6 and collapse_count == 0 else
                 'mechanism_no_go' if negative_count >= 6 else 'statistical_inconclusive')

    engineering = json.loads((args.input / 'engineering.json').read_text())
    manifest_record = engineering['input_manifest']
    manifest_path = Path(manifest_record['path'])
    manifest = json.loads(manifest_path.read_text())
    identity_path = Path(manifest['base_identity_path'])
    identity = json.loads(identity_path.read_text())
    runtime = engineering['runtime_identity']
    top = Path('/tc1home/UG/yguo017/v100_newangles_ccds')
    selection = json.loads((top/'artifacts/64775/selection.json').read_text())
    expected_ids = [f"task{row['task_index']}:episode{row['episode_index']}:frame{row['frame_index']}" for row in manifest['samples']]
    expected_checkpoint = next(row['downloaded_sha256'] for row in identity['checkpoint']['files'] if row['rfilename'] == 'model.safetensors')
    source_checks = {}
    for name, record in runtime['source_identity'].items():
        if isinstance(record, dict) and record.get('path') and record.get('sha256'):
            source_checks[name] = sha(record['path']) == record['sha256']
    bypass = engineering['quantizer']['bypassed_state_digest_before']
    arm_checks = {}
    for arm in ('FP32', 'expert_W4'):
        record = engineering['arms'][arm]
        arm_checks[arm] = bool(record.get('restore_before_exact') is True and
            record.get('all_conditions_complete') is True and
            record.get('bypassed_state_digest_before') == bypass == record.get('bypassed_state_digest_after') and
            record.get('batch_check', {}).get('pass') is True)
    parameters = engineering['parameters']
    producer_checks = {
        'complete': engineering.get('status') == 'complete',
        'runtime_identity_saved_before_inference': engineering.get('runtime_identity_saved_before_inference') is True,
        'V100': 'v100' in engineering.get('gpu', {}).get('name', '').lower(),
        'producer_allocation_verified': engineering['allocation'].get('verified') is True,
        'producer_job_binding': str(engineering['allocation']['job_id']) == args.input.name,
        'source_hashes': len(source_checks) >= 3 and all(source_checks.values()),
        'lerobot_version': runtime['source_identity'].get('lerobot_version') == '0.4.4',
        'torch_rng_version': engineering['gpu'].get('torch') == torch.__version__,
        'FP32_offline_runtime': runtime.get('dtype') == 'float32' and runtime.get('load_vlm_weights') is False and runtime.get('compile_model') is False and runtime.get('rtc_config') is None,
        'checkpoint_identity': runtime['checkpoint']['sha256'] == expected_checkpoint,
        'manifest_hash': sha(manifest_path) == manifest_record['sha256'],
        'identity_hash': sha(identity_path) == manifest['base_identity_sha256'] == manifest_record['base_identity_sha256'],
        'frozen_selected_episodes': manifest['selection']['selected'] == selection['selected'],
        'raw_episode_identity': data['episode_ids'].tolist() == expected_ids == manifest_record['actual_episode_ids'],
        'sample_hashes': all(sha(manifest_path.parent/row['sample_path']) == row['sha256'] for row in manifest['samples']),
        'pinned_flow_helper': sha(args.input/'flow_screen.py') == engineering['flow_helper']['sha256'] == 'ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9',
        'expert_W4_recipe': engineering['quantizer']['bits'] == 4 and engineering['quantizer']['qmax'] == 7 and engineering['quantizer']['module_count'] == 112 and engineering['quantizer']['scope'] == 'expert transformer Linear modules only',
        'arms_restore_and_bypass': all(arm_checks.values()),
        'final_restore': engineering['weight_restore'].get('exact_elementwise') is True and engineering['weight_restore']['bypassed_state_digest_final'] == bypass,
        'frozen_shape_parameters': parameters['conditions'] == 8 and parameters['draws_per_condition_per_arm'] == 64 and parameters['micro_batch'] == 4 and parameters['num_steps'] == 10 and parameters['horizon'] == 50,
        'frozen_noise_parameters': engineering['noise']['seeds'] == [1901, 1902] and engineering['noise']['block_size'] == 32 and engineering['noise']['sha256'] == hashlib.sha256(expected_noise.astype(np.float32).tobytes()).hexdigest(),
    }
    report = {
        'schema': 'conditional-marginal-independent-verification-v1', 'allocation': allocation,
        'source_job_dir': str(args.input), 'raw_sha256': sha(raw_path),
        'engineering_sha256': sha(args.input / 'engineering.json'),
        'raw_checks': raw_checks, 'producer_checks': producer_checks,
        'producer_checks_pass': all(producer_checks.values()), 'source_checks': source_checks,
        'runtime_source_identity': runtime['source_identity'],
        'runner_sha256': sha(args.input/'marginal_screen.py'),
        'manifest_sha256': manifest_record['sha256'], 'checkpoint_sha256': expected_checkpoint,
        'batch_max_abs_errors': batch_errors.tolist(),
        'projection_directions': directions.tolist(),
        'projection_float64_sha256': hashlib.sha256(directions.tobytes()).hexdigest(),
        'numeric_decision': numerical,
        'decision': numerical if all(raw_checks.values()) and all(producer_checks.values()) else 'implementation_inconclusive',
        'positive_count': positive_count, 'negative_count': negative_count, 'collapse_count': collapse_count,
        'episodes': records,
        'scope': '8 independent conditions; 64 noise draws each. Heuristic bounded low-dimensional screen, not distribution equivalence, statistical significance, task success, or native low-bit performance.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if len(encoded.encode()) >= 64*1024:
        raise ValueError('Compact verification exceeds 64KiB')
    (args.output / 'verification.json').write_text(encoded, encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('decision', 'positive_count', 'negative_count', 'collapse_count')}))


if __name__ == '__main__':
    main()
