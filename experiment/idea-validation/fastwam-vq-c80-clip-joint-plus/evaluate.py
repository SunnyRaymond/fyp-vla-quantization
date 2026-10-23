"""Run one persistent worker for the frozen C80 clip-only versus joint cohort."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = 'fastwam-vq-c80-clip-joint-plus-v1'
COHORT_PROTOCOL = 'fastwam-rotation-baselines-v1'
ARMS = ('clip_only', 'joint')
TIMING_FIELDS = ('denoising_latency_ms', 'full_query_wall_ms', 'peak_allocated_gpu_GB',
                 'baseline_allocated_gpu_GB', 'peak_reserved_gpu_GB')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def worker_arm(worker_id, arm):
    if type(worker_id) is not int or worker_id not in range(4):
        raise ValueError('worker_id must be 0,1,2,3')
    if arm not in ARMS:
        raise ValueError(f'arm must be one of {ARMS}')
    return arm


def worker_indices(worker_id, arm):
    worker_arm(worker_id, arm)
    return list(range(worker_id, 280, 4))


def worker_coverage(worker_id, arm, completed):
    """Keep the fixed shard separate from its durable completed prefix."""
    return {'indices': worker_indices(worker_id, arm), 'completed_indices': list(completed)}


def environment_seed(index):
    return 800_000_000 + index * 1000


def sampler_seed(index, replan=0):
    return 900_000_000 + index * 1000 + replan


def validate_protocol(protocol):
    inference = {
        'mode': 'idm', 'video_steps': 10, 'action_steps': 10, 'sigma_shift': 1.0,
        'action_horizon': 32, 'execute_per_replan': 10, 'action_ensembler': False,
        'settling_steps': 30, 'max_steps_normal': 400, 'max_steps_long': 700,
    }
    cohort = {
        'manifest_protocol': COHORT_PROTOCOL, 'variants_per_arm': 280,
        'total_new_episodes': 560, 'suites': 4, 'dimensions': 7,
        'variants_per_suite': 70, 'variants_per_dimension': 40,
        'variants_per_cell': 10, 'state_id': 0, 'env_seed_base': 800_000_000,
        'sampler_seed_base': 900_000_000, 'seed_stride': 1000,
        'paired_seeds_across_arms': True, 'sampling_changed': False,
    }
    configs = {'clip_only': {'rho': 1.5, 'epochs': 1, 'updates': 80,
                             'train_weights': False, 'train_clips': True},
               'joint': {'rho': 1.5, 'epochs': 1, 'updates': 80,
                         'train_weights': True, 'train_clips': True}}
    calibration = protocol.get('calibration', {})
    expected_cases = list(range(2000, 2080))
    expected_order = [2000 + 20 * suite + offset
                      for offset in range(20) for suite in range(4)]
    expected_root = '/scratch/users/ntu/yguo017/fastwam-vq-calibration-suites-20261013'
    expected_remote = '/scratch/users/ntu/yguo017/fastwam-vq-c80-clip-joint-plus-20261017'
    training = {
        'additional_updates': 80, 'epochs': 1, 'weight_learning_rate': 0.0005,
        'clip_learning_rate': 0.01, 'optimizer': 'Adam', 'fresh_optimizer_each_arm': True,
        'optimizer_reset_after_update1': True, 'gradient_norm_clip': 1.0, 'rho': 1.5,
        'normalization_c': 0.028303017887797623,
        'loss': 'first10 x6 motor MSE + rho * frozen normalization_c * clean future global NMSE excluding frame0',
        'checkpoint_selection': 'fixed final update80 only; no held-out checkpoint selection or early stopping',
    }
    workers = {
        'workers': 4, 'clip_only': [0, 1, 2, 3], 'joint': [0, 1, 2, 3],
        'indices_per_worker': 'index % 4 == worker_id',
        'variants_per_worker': 70, 'start_stagger_seconds': 20,
        'evaluation_arm_per_job': 'FW_EVAL_ARM',
        'result_directory': 'results/evaluations/{arm}/worker_{worker_id}',
    }
    expected = {
        'protocol': PROTOCOL, 'arms': list(ARMS), 'arm_configs': configs,
        'inference': inference, 'soft_deadline_seconds': 85500,
        'source_parity_max_abs_tolerance': 1e-6,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('C80 clip/joint protocol differs from the frozen worker/inference contract')
    if protocol.get('remote_root') != expected_remote:
        raise ValueError('C80 clip/joint protocol changed the frozen results root')
    if any(protocol.get('cohort', {}).get(key) != value for key, value in cohort.items()):
        raise ValueError('C80 clip/joint protocol changed the frozen paired cohort or seeds')
    if any(protocol.get('training', {}).get(key) != value for key, value in training.items()):
        raise ValueError('C80 clip/joint protocol changed the frozen training recipe')
    if any(protocol.get('worker_assignment', {}).get(key) != value for key, value in workers.items()):
        raise ValueError('C80 clip/joint protocol changed the four-worker parity partition')
    allocation = protocol.get('allocation', {})
    if (allocation.get('jobs') != 2 or allocation.get('gpus') != 4
            or allocation.get('walltime') != '24:00:00'
            or allocation.get('episodes_per_job') != 280):
        raise ValueError('C80 clip/joint protocol changed the two per-arm 4-GPU 24-hour jobs')
    if (calibration.get('root') != expected_root
            or calibration.get('plan_file') != expected_root + '/data/plan.json'
            or calibration.get('bank_directory') != expected_root + '/results/c8/banks/micro_scale_vq_a4'
            or calibration.get('source_job_id') != '25789685.pbs101'
            or calibration.get('case_ids') != expected_cases or calibration.get('cohort') != 'c80'
            or calibration.get('contexts') != 80 or calibration.get('per_suite') != 20
            or calibration.get('update_order') != expected_order):
        raise ValueError('C80 clip/joint protocol changed its frozen C80 calibration binding')
    clipping = protocol.get('clipping', {})
    if (clipping.get('initial_ratio') != 1.0 or clipping.get('minimum_ratio') != 0.5
            or clipping.get('maximum_ratio') != 1.0
            or protocol.get('source_c80_protocol', {}).get('protocol') !=
            'fastwam-vq-c80-activation-clipping-v2-original-heldout'):
        raise ValueError('C80 clip/joint protocol changed the clipping range or C80 source')
    if protocol.get('closed_loop_reference') != {
            'case_id': 2000, 'seed_index': 0, 'max_abs_tolerance': 1e-6}:
        raise ValueError('C80 clip/joint protocol changed the trained-action replay gate')
    if (protocol.get('sources', {}).get('clip_root') !=
            '/scratch/users/ntu/yguo017/fastwam-vq-activation-clipping-20261017'
            or protocol.get('sources', {}).get('baseline_arm') != 'rho1.5'
            or protocol.get('initialization') !=
            'both arms start from the exact existing C8 rho1.5 update8 baseline, ratios=1; use C80 calibration contexts for80 additional updates; joint is codebooks+log row gains+ratios, no channel scaling'):
        raise ValueError('C80 clip/joint protocol changed the C8 rho1.5 initialization')
    quantization = protocol.get('quantization', {})
    if (quantization.get('kv_cache') != 'bf16' or quantization.get('native_int4_kernel') is not False
            or quantization.get('transform_implementation') != 'batched_h128_v2'
            or quantization.get('weights') !=
            'fixed-index B32 VQ initialized from current C8 rho1.5; clip_only weights frozen, joint FP32 master books/log row gains optimized; deploy BF16'
            or quantization.get('activations') !=
            'learned per-Linear clipping ratio [0.5,1] with dynamic per-token A4 qmax7, decoded BF16'):
        raise ValueError('C80 clip/joint protocol changed the decoded-BF16/A4 execution path')
    references = protocol.get('references', {})
    expected_references = {
        'rotation_results': '/scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008/results',
        'scalar_results': '/scratch/users/ntu/yguo017/fastwam-smooth-scalar-plus-20261009/results',
        'vq_results': '/scratch/users/ntu/yguo017/fastwam-vq-plus-20261012/results',
        'expected_reference_slots': 2240,
        'c8_trained_arms': ['rho0.1', 'rho1.5'],
    }
    if any(references.get(key) != value for key, value in expected_references.items()):
        raise ValueError('C80 clip/joint protocol changed its 2240 frozen reference slots')
    if (references.get('c8_trained_results') !=
            '/scratch/users/ntu/yguo017/fastwam-vq-c8-action-latent-plus-20261015/results'
            or references.get('reference_episode_slots') != 2240
            or references.get('arms') != ['bf16', 'quarot_adapted_w4a4', 'spinquant_adapted_w4a4',
                'smooth05_hadamard_scalar_w4a4', 'row_scale_vq_a4', 'micro_scale_vq_a4', 'rho0.1', 'rho1.5']):
        raise ValueError('C80 clip/joint protocol changed its exact prior-reference arms')
    return protocol


def allocation_elapsed_seconds(start_epoch, now_epoch):
    if (isinstance(start_epoch, bool) or not isinstance(start_epoch, (int, float))
            or isinstance(now_epoch, bool) or not isinstance(now_epoch, (int, float))
            or not math.isfinite(start_epoch) or not math.isfinite(now_epoch)
            or start_epoch <= 0 or now_epoch < start_epoch):
        raise ValueError('Invalid allocation start/now epoch')
    return float(now_epoch) - float(start_epoch)


def allocation_start_epoch():
    raw = os.environ.get('FW_JOB_START_EPOCH')
    if raw is None:
        raise RuntimeError('PBS runner must export allocation-wide FW_JOB_START_EPOCH')
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError('FW_JOB_START_EPOCH must be a Unix epoch in seconds') from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError('FW_JOB_START_EPOCH must be a positive finite Unix epoch')
    return value


def load_manifest(path):
    doc = read_json(path)
    rows = doc.get('variants')
    if doc.get('protocol') != COHORT_PROTOCOL or not isinstance(rows, list) or len(rows) != 280:
        raise ValueError('Expected the original frozen 280-variant manifest')
    ids, cells = set(), collections.Counter()
    for index, row in enumerate(rows):
        if row.get('index') != index or type(row.get('index')) is not int:
            raise ValueError('Invalid global variant index')
        identity = str(row.get('variant_id', ''))
        if not identity or identity in ids or 'task_id' not in row:
            raise ValueError('Duplicate/missing frozen variant identity')
        ids.add(identity)
        cells[row['suite'], row['dimension']] += 1
    if (len(cells) != 28 or set(cells.values()) != {10}
            or len({suite for suite, _ in cells}) != 4 or len({dim for _, dim in cells}) != 7):
        raise ValueError('Expected 4 suites x 7 dimensions x 10 variants')
    return doc, rows


def validate_install_metadata(metadata, protocol, arm):
    if arm not in ARMS or not isinstance(metadata, dict):
        raise ValueError('Missing trained C80 install metadata')
    config = protocol['arm_configs'][arm]
    calibration = protocol['calibration']
    expected = {
        'protocol': PROTOCOL,
        'arm': arm,
        'source_bank': calibration['bank_directory'],
        'source_protocol': 'fastwam-vq-calibration-suites-v1',
        'source_cohort': 'c80',
        'source_case_ids': calibration['case_ids'],
        'calibration_ids': calibration['case_ids'],
        'initialization_protocol': 'fastwam-vq-c8-action-latent-plus-v1',
        'initialization_cohort': 'c8', 'initialization_arm': 'rho1.5',
        'initialization_updates': 8,
        'initialization_training_job': protocol['sources']['baseline_training_job'],
        'module_count': 614,
        'execution': 'decoded_bf16_vq',
        'activation_bits': 4,
        'kv_cache_dtype': 'bf16',
        'native_low_bit_kernel': False,
        'rho': config['rho'],
        'epochs': config['epochs'],
        'updates': config['updates'],
        'transform_implementation': 'batched_h128_v2',
        'smooth_alpha': 0.5,
        'hadamard_block': 128,
        'scale_block': 32,
        'fixed_indices': True,
        'fixed_scale_exponents': True,
        'deployed_books_dtype': 'bfloat16',
        'deployed_row_scales_dtype': 'bfloat16',
        'train_weights': config['train_weights'],
        'train_clips': config['train_clips'],
        'updates': 80,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Trained C80 install metadata differs from protocol for {arm}')
    if metadata.get('weights_unchanged') is not (not config['train_weights']):
        raise ValueError(f'Trained C80 weight-freeze receipt differs for {arm}')
    if (metadata.get('normalization_c') != protocol['training']['normalization_c']
            or metadata.get('rho') != 1.5
            or metadata.get('epochs') != 1
            or metadata.get('updates') != 80
            or metadata.get('calibration_ids') != calibration['case_ids']
            or metadata.get('native_int4_gemm_calls') != 0):
        raise ValueError(f'Trained C80 update/source binding differs for {arm}')
    ratio_summary = metadata.get('ratios')
    if (not isinstance(ratio_summary, dict) or ratio_summary.get('count') != 614
            or any(isinstance(ratio_summary.get(key), bool)
                   or not isinstance(ratio_summary.get(key), (int, float))
                   or not math.isfinite(ratio_summary[key]) for key in ('min', 'mean', 'max'))
            or ratio_summary['min'] < 0.5 or ratio_summary['max'] > 1.0
            or ratio_summary['min'] > ratio_summary['mean']
            or ratio_summary['mean'] > ratio_summary['max']):
        raise ValueError(f'Invalid trained C80 clipping-ratio summary for {arm}')
    return metadata


def validate_record(record, rows, worker_id, protocol, install_metadata, arm):
    arm, indices = worker_arm(worker_id, arm), worker_indices(worker_id, arm)
    validate_install_metadata(install_metadata, protocol, arm)
    index = record.get('index')
    if type(index) is not int or index not in indices:
        raise ValueError('Record is outside its frozen worker assignment')
    row = rows[index]
    expected = {'protocol': PROTOCOL, 'arm': arm, 'worker_id': worker_id,
                'complete': True, 'suite': row['suite'], 'dimension': row['dimension'],
                'task_id': int(row['task_id']), 'state_id': 0,
                'env_seed': environment_seed(index), 'sampler_seed': sampler_seed(index)}
    if (any(record.get(key) != value for key, value in expected.items())
            or str(record.get('variant_id')) != str(row['variant_id'])
            or type(record.get('success')) is not bool):
        raise ValueError(f'Frozen episode identity/seed differs: {arm}/{index}')
    evidence = record.get('vq_evidence', {})
    required = {**install_metadata, 'observed_module_count': 614,
                'native_int4_gemm_calls': 0, 'kv_packed_prefills': 0, 'kv_layer_reads': 0}
    if (any(evidence.get(key) != value for key, value in required.items())
            or evidence.get('quantized_linear_calls', 0) <= 0):
        raise ValueError(f'Trained C80 decoded-BF16/A4 execution evidence differs: {arm}/{index}')
    seconds = record.get('episode_seconds')
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Invalid episode wall time')
    replans = record.get('replans')
    metrics = record.get('metrics')
    if type(replans) is not int or replans <= 0 or not isinstance(metrics, list) or len(metrics) != replans:
        raise ValueError('Missing or inconsistent action-query metrics')
    if record.get('transform_implementation') != 'batched_h128_v2' or not record.get('pbs_job_id'):
        raise ValueError('Missing trained C80 transform/PBS identity')
    for replan, metric in enumerate(metrics):
        q = metric.get('query_index_in_arm')
        if (type(q) is not int or q < 0 or type(metric.get('first_query_in_arm')) is not bool
                or metric['first_query_in_arm'] != (q == 0)
                or metric.get('replan') != replan
                or metric.get('sampler_seed') != sampler_seed(index, replan)
                or metric.get('transform_implementation') != record['transform_implementation']
                or metric.get('pbs_job_id') != record['pbs_job_id']
                or type(metric.get('first_query_in_run')) is not bool):
            raise ValueError('Invalid resumed implementation/run/query marker')
        for key in TIMING_FIELDS:
            value = metric.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'Invalid {key}')
    return index


def recover_records(path, rows, worker_id, protocol, install_metadata, arm):
    path = Path(path)
    if not path.exists():
        return {}
    if install_metadata is None:
        raise ValueError('Cannot resume episodes without the matching setup receipt')
    payload = path.read_bytes()
    chunks = payload.splitlines(keepends=True)
    if chunks and not chunks[-1].endswith(b'\n'):
        try:
            json.loads(chunks[-1])
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = b''.join(chunks[:-1])
        else:
            payload += b'\n'
        temp = path.with_suffix('.recovered.tmp')
        temp.write_bytes(payload)
        temp.replace(path)
    found, queries, seen_runs = {}, [], set()
    for line in payload.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index = validate_record(record, rows, worker_id, protocol, install_metadata, arm)
        if index in found:
            raise ValueError('Duplicate completed episode')
        found[index] = record
        queries.extend(metric['query_index_in_arm'] for metric in record['metrics'])
        for metric in record['metrics']:
            run = metric['pbs_job_id']
            if metric['first_query_in_run'] != (run not in seen_runs):
                raise ValueError('Invalid first formal query marker within a resumed worker run')
            seen_runs.add(run)
    indices = list(found)
    expected_prefix = worker_indices(worker_id, arm)[:len(indices)]
    if indices != expected_prefix:
        raise ValueError('Saved episodes are not the frozen worker prefix')
    if queries != list(range(len(queries))):
        raise ValueError('Saved query sequence is not a consecutive prefix')
    return found


def append_episode(path, record):
    with Path(path).open('ab') as stream:
        stream.write((json.dumps(record, allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


def _setup_receipt(path, arm, protocol, reference_action):
    receipt = read_json(path)
    max_abs = receipt.get('max_abs_vs_reference')
    reference = protocol['closed_loop_reference']
    if (receipt.get('passed') is not True or receipt.get('protocol') != PROTOCOL
            or receipt.get('arm') != arm or receipt.get('case_id') != reference['case_id']
            or receipt.get('seed_index') != reference['seed_index']
            or receipt.get('sampler_seed') != reference_action['sampler_seed']
            or receipt.get('reference_actual_infer_parity') != reference_action['actual_infer_parity']
            or receipt.get('observed_modules') != 614
            or isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
            or not math.isfinite(max_abs) or max_abs < 0
            or max_abs > reference['max_abs_tolerance']):
        raise ValueError('C80 trained action replay did not pass')
    metadata = validate_install_metadata(receipt.get('install_metadata'), protocol, arm)
    return receipt, metadata


def run_worker(args):
    from vq_adapter import validate_training_artifacts

    p = validate_protocol(read_json(args.protocol))
    arm = args.arm or os.environ.get('FW_EVAL_ARM')
    if arm not in ARMS:
        raise ValueError('Pass --arm clip_only|joint or set FW_EVAL_ARM')
    arm, indices = worker_arm(args.worker_id, arm), worker_indices(args.worker_id, arm)
    _, rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / 'protocol.json'
    if frozen.exists() and read_json(frozen) != p:
        raise ValueError('Cannot resume with changed C80 clip/joint configuration')
    save(frozen, p)

    # Layout: results/evaluations/{arm}/worker_{id}; training is a sibling
    # at results/training/{arm}, not below the arm's evaluations directory.
    results_root = args.out.parent.parent.parent
    training_dir = results_root / 'training' / arm
    preflight = validate_training_artifacts(p, arm, training_dir)
    import runner
    runner.guard()
    from vq_adapter import TrainedRuntime
    setup_path = args.out / 'setup_receipt.json'
    existing_metadata = None
    if setup_path.is_file():
        _, existing_metadata = _setup_receipt(setup_path, arm, p, preflight['reference_action'])
    path = args.out / 'episodes.jsonl'
    completed = recover_records(path, rows, args.worker_id, p, existing_metadata, arm)
    if completed and existing_metadata is None:
        raise ValueError('Saved episodes exist without their setup/parity receipt')

    runtime, metadata, soft_stop = None, existing_metadata, False
    allocation_start = allocation_start_epoch()

    def receipt(status, **extra):
        return {'protocol': PROTOCOL, 'worker_id': args.worker_id, 'arm': arm,
                'status': status, 'complete': status == 'complete', 'episodes': len(completed),
                'completed': len(completed), 'expected': 70, 'soft_deadline_reached': soft_stop,
                **worker_coverage(args.worker_id, arm, completed),
                'arm_episodes': {arm: len(completed)},
                'install_metadata': metadata, 'setup_receipt_file': 'setup_receipt.json', **extra}

    try:
        save(args.out / 'progress.json', receipt('running', stage='model_loading'))
        if len(completed) != 70:
            runtime = TrainedRuntime(args.out, p, arm, training_dir, preflight)
            metadata = runtime.install_metadata
            if existing_metadata is not None and metadata != existing_metadata:
                raise ValueError('Resumed worker loaded a different trained deployment')
            setup = runtime.validate_reference()
            save(setup_path, setup)
            runtime.arm_calls[arm] = sum(len(row['metrics']) for row in completed.values())
            runtime.calls = runtime.arm_calls[arm]
            for index in indices:
                if index in completed:
                    continue
                if allocation_elapsed_seconds(allocation_start, time.time()) >= p['soft_deadline_seconds']:
                    soft_stop = True
                    break
                save(args.out / 'progress.json', receipt('running', stage='episode', index=index))
                record = runtime.episode(rows[index], index)
                record['worker_id'] = args.worker_id
                validate_record(record, rows, args.worker_id, p, metadata, arm)
                append_episode(path, record)
                completed[index] = record
                warm = [m for m in record['metrics'] if not m['first_query_in_run']]
                save(args.out / 'last_episode_timing.json', {
                    'pbs_job_id': record['pbs_job_id'], 'transform_implementation': record['transform_implementation'],
                    'index': index, 'completed_episodes': len(completed), 'episode_seconds': record['episode_seconds'],
                    'replans': record['replans'], 'subsequent_queries': len(warm),
                    'full_query_wall_ms_mean': sum(m['full_query_wall_ms'] for m in warm) / len(warm) if warm else None,
                    'denoising_latency_ms_mean': sum(m['denoising_latency_ms'] for m in warm) / len(warm) if warm else None,
                })
                save(args.out / 'progress.json', receipt('running', stage='episode_saved', index=index))
                print(f'SLOT_COMPLETE worker={args.worker_id} arm={arm} index={index} success={record["success"]}', flush=True)
        else:
            _, metadata = _setup_receipt(setup_path, arm, p, preflight['reference_action'])
        status = 'checkpointed' if soft_stop else 'complete'
        if not soft_stop and set(completed) != set(indices):
            raise RuntimeError('Worker completed coverage differs from its frozen shard')
        result = receipt(status)
        save(args.out / 'summary.json', result)
        save(args.out / 'progress.json', result)
        print(f'WORKER_{status.upper()} worker={args.worker_id} episodes={len(completed)}', flush=True)
    except Exception as exc:
        save(args.out / 'failure.json', {'protocol': PROTOCOL, 'worker_id': args.worker_id,
             'arm': arm, 'type': type(exc).__name__, 'message': str(exc), 'completed': len(completed)})
        raise
    finally:
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker-id', type=int, choices=range(4), required=True)
    parser.add_argument('--arm', choices=ARMS)
    run_worker(parser.parse_args())


if __name__ == '__main__':
    main()
