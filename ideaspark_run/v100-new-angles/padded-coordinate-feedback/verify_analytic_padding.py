"""Frozen analytic-padding raw replay; execute in a real CPU allocation only."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import numpy as np

    path = args.input / 'raw_analytic_padding.npz'
    with np.load(path, allow_pickle=False) as raw:
        if raw['schema'].item() != 'analytic-padding-raw-v1':
            raise ValueError('Wrong raw schema')
        data = {key: raw[key].copy() for key in (
            'predicted_velocity', 'used_velocity', 'x_inputs', 'actions',
            'noise', 'times', 'completed', 'episode_ids', 'arm_names')}
    shapes = {key: (8, 2, 4, 10, 50, 32) for key in ('predicted_velocity', 'used_velocity', 'x_inputs')}
    shapes.update(actions=(8, 2, 4, 50, 7), noise=(2, 50, 32), times=(10,),
                  completed=(8,), episode_ids=(8,), arm_names=(4,))
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f'{key}: wrong shape {data[key].shape}, expected {shape}')
    if not data['completed'].all() or len(set(data['episode_ids'].tolist())) != 8:
        raise ValueError('Incomplete or duplicated episodes')
    if data['arm_names'].tolist() != ['FPnative', 'Qnative', 'FPanalytic', 'Qanalytic']:
        raise ValueError('Wrong arm order')
    for key in ('predicted_velocity', 'used_velocity', 'x_inputs', 'actions', 'noise', 'times'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Nonfinite {key}')
        data[key] = data[key].astype(np.float64)

    predicted, used, x = (data[key] for key in ('predicted_velocity', 'used_velocity', 'x_inputs'))
    noise, times = data['noise'], data['times']
    checks = {
        'native_velocity_unchanged': bool(np.array_equal(predicted[:, :, :2], used[:, :, :2])),
        'analytic_physical_unchanged': bool(np.array_equal(predicted[:, :, 2:, :, :, :7], used[:, :, 2:, :, :, :7])),
        'analytic_padding_equals_initial_noise': bool(np.array_equal(
            used[:, :, 2:, :, :, 7:], np.broadcast_to(noise[None, :, None, None, :, 7:], (8, 2, 2, 10, 50, 25)))),
        'first_physical_FP_pair_equal': bool(np.array_equal(predicted[:, :, 0, 0, :, :7], predicted[:, :, 2, 0, :, :7])),
        'first_physical_Q_pair_equal': bool(np.array_equal(predicted[:, :, 1, 0, :, :7], predicted[:, :, 3, 0, :, :7])),
        'all_initial_states_equal_noise': bool(np.array_equal(
            x[:, :, :, 0], np.broadcast_to(noise[None, :, None], (8, 2, 4, 50, 32)))),
        'ten_official_times': bool(np.max(np.abs(times - (1 - np.arange(10)/10))) <= 1e-6),
    }
    expected_pad = noise[None, :, None, None, :, 7:] * times[None, None, None, :, None, None]
    pad_path_error = float(np.max(np.abs(x[:, :, 2:, :, :, 7:] - expected_pad)))
    recurrence_error = float(np.max(np.abs(x[:, :, :, 1:] - (x[:, :, :, :-1] - .1*used[:, :, :, :-1]))))
    endpoint_error = float(np.max(np.abs(data['actions'] - (x[:, :, :, -1, :, :7] - .1*used[:, :, :, -1, :, :7]))))
    checks.update(analytic_path=pad_path_error <= 1e-5,
                  full_state_recurrence=recurrence_error <= 1e-5,
                  physical_endpoint_recurrence=endpoint_error <= 1e-5)

    actions = data['actions'][:, :, :, :8]
    def mse(a, b):
        return np.square(actions[:, :, a] - actions[:, :, b]).mean(axis=(-1, -2)).mean(axis=1)
    e0, shift, e1, e2 = mse(1, 0), mse(2, 0), mse(3, 0), mse(3, 2)
    p0 = np.square(predicted[:, :, 1, 0, :, 7:] - predicted[:, :, 0, 0, :, 7:]).mean(axis=(-1, -2)).mean(axis=1)
    records = []
    for ep in range(8):
        binding = bool(e0[ep] > 1e-6 and shift[ep] <= .01*e0[ep] + 1e-12 and p0[ep] > 1e-10)
        g1 = float(1-e1[ep]/e0[ep]) if e0[ep] > 0 else None
        g2 = float(1-e2[ep]/e0[ep]) if e0[ep] > 0 else None
        records.append({
            'episode_id': str(data['episode_ids'][ep]),
            'E0_Qnative_FPnative': float(e0[ep]), 'S_FPanalytic_FPnative': float(shift[ep]),
            'E1_Qanalytic_FPnative': float(e1[ep]), 'E2_Qanalytic_FPanalytic': float(e2[ep]),
            'P0_first_step_padding_quantization_mse': float(p0[ep]),
            'G1': g1, 'G2': g2, 'binding': binding,
            'binding_reasons': {'nondegenerate_drift': bool(e0[ep] > 1e-6),
                                'small_FP_shift': bool(shift[ep] <= .01*e0[ep] + 1e-12),
                                'nondegenerate_padding_perturbation': bool(p0[ep] > 1e-10)},
            'joint_positive': bool(binding and g1 > 0 and g2 > 0),
        })
    binding_records = [row for row in records if row['binding']]
    median1 = float(np.median([row['G1'] for row in binding_records])) if binding_records else None
    median2 = float(np.median([row['G2'] for row in binding_records])) if binding_records else None
    positive_count = sum(row['joint_positive'] for row in records)
    numeric_decision = ('inconclusive_binding' if len(binding_records) < 6 else
                        'preliminary_go' if median1 >= .25-1e-12 and median2 >= .25-1e-12 and positive_count >= 6 else 'method_no_go')
    engineering = json.loads((args.input / 'engineering.json').read_text())
    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()
    manifest_path = Path(engineering['manifest']['path'])
    manifest = json.loads(manifest_path.read_text())
    base_path = Path(manifest['base_identity_path'])
    base = json.loads(base_path.read_text())
    runtime = engineering['runtime_identity']
    expected_ids = [f"task{row['task_index']}:episode{row['episode_index']}:frame{row['frame_index']}" for row in manifest['samples']]
    expected_checkpoint = next(row['downloaded_sha256'] for row in base['checkpoint']['files'] if row['rfilename'] == 'model.safetensors')
    source_checks = {}
    for name, record in runtime['source_identity'].items():
        if isinstance(record, dict) and record.get('path') and record.get('sha256'):
            source_checks[name] = sha(record['path']) == record['sha256']
    gate_names = ('complete8', 'no_op_action_exact', 'native_used_equals_predicted',
                  'analytic_physical_slice_exact', 'first_step_input_exact',
                  'first_step_fp_physical_exact', 'first_step_q_physical_exact',
                  'analytic_used_pad_noise_exact', 'analytic_x_pad_time_noise',
                  'time_grid', 'weight_restore_exact')
    producer_checks = {
        'status_complete': engineering.get('status') == 'complete',
        'all_recorded_gates_pass': all(engineering.get('engineering_gates', {}).get(name) is True for name in gate_names),
        'runtime_V100': 'v100' in engineering.get('gpu', {}).get('name', '').lower(),
        'producer_allocation_verified': engineering.get('allocation', {}).get('verified') is True,
        'producer_job_binding': str(engineering['allocation']['job_id']) == args.input.name,
        'FP32_offline_runtime': runtime.get('dtype') == 'float32' and runtime.get('load_vlm_weights') is False and runtime.get('compile_model') is False and runtime.get('rtc_config') is None,
        'lerobot_version': runtime['source_identity'].get('lerobot_version') == '0.4.4',
        'recorded_runtime_sources_match': len(source_checks) >= 3 and all(source_checks.values()),
        'manifest_hash': sha(manifest_path) == engineering['manifest']['sha256'],
        'base_identity_hash': sha(base_path) == manifest['base_identity_sha256'] == engineering['base_identity']['sha256'],
        'checkpoint_identity': runtime['checkpoint']['sha256'] == expected_checkpoint,
        'episode_manifest_binding': data['episode_ids'].tolist() == expected_ids == engineering['episode_ids'],
        'sample_hashes': all(sha(manifest_path.parent / row['sample_path']) == row['sha256'] for row in manifest['samples']),
        'frozen_flow_helper': sha(args.input / 'flow_screen.py') == engineering['flow_screen_sha256'] == 'ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9',
        'expert_W4_only': engineering['quantizer']['scope'] == 'expert_W4 only' and engineering['quantizer']['bits'] == 4 and engineering['quantizer']['qmax'] == 7 and engineering['quantizer']['module_count'] > 0,
        'no_op_and_restore': engineering['no_op']['action_exact_equal'] is True and engineering['weight_restore']['exact_elementwise'] is True,
    }
    report = {
        'schema': 'analytic-padding-independent-verification-v1', 'allocation': allocation,
        'source_job_dir': str(args.input), 'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'engineering_artifact_sha256': hashlib.sha256((args.input / 'engineering.json').read_bytes()).hexdigest(),
        'producer_engineering_status': engineering.get('status'),
        'producer_checks': producer_checks, 'producer_checks_pass': all(producer_checks.values()),
        'source_checks': source_checks, 'runtime_source_identity': runtime['source_identity'],
        'manifest_sha256': engineering['manifest']['sha256'], 'checkpoint_sha256': expected_checkpoint,
        'runner_sha256': sha(args.input / 'analytic_padding_screen.py'),
        'raw_engineering_checks': checks, 'raw_engineering_pass': all(checks.values()),
        'analytic_path_max_abs_error': pad_path_error, 'recurrence_max_abs_error': recurrence_error,
        'endpoint_max_abs_error': endpoint_error, 'numeric_decision': numeric_decision,
        'decision': numeric_decision if all(checks.values()) and all(producer_checks.values()) else 'implementation_inconclusive',
        'binding_count': len(binding_records), 'joint_positive_count': positive_count,
        'binding_median_G1': median1, 'binding_median_G2': median2, 'episodes': records,
        'scope': 'Two noises averaged inside each episode. Offline normalized FP fidelity only; not success, hardware speed, or novelty certification.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('decision', 'binding_count', 'joint_positive_count', 'binding_median_G1', 'binding_median_G2')}))


if __name__ == '__main__':
    main()
