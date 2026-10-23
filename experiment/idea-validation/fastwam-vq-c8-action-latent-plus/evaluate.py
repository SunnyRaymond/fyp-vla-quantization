"""Run one persistent worker for a frozen C8-trained, paired LIBERO-Plus shard."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = 'fastwam-vq-c8-action-latent-plus-v1'
COHORT_PROTOCOL = 'fastwam-rotation-baselines-v1'
ARMS = ('rho0.1', 'rho1.5')
TIMING_FIELDS = ('denoising_latency_ms', 'full_query_wall_ms', 'peak_allocated_gpu_GB',
                 'baseline_allocated_gpu_GB', 'peak_reserved_gpu_GB')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def worker_arm(worker_id):
    if type(worker_id) is not int or worker_id not in range(4):
        raise ValueError('worker_id must be 0,1,2,3')
    return ARMS[worker_id // 2]


def worker_indices(worker_id):
    worker_arm(worker_id)
    return list(range(worker_id % 2, 280, 2))


def worker_coverage(worker_id, completed):
    """Keep the fixed shard separate from its durable completed prefix."""
    return {'indices': worker_indices(worker_id), 'completed_indices': list(completed)}


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
    configs = {'rho0.1': {'rho': 0.1, 'epochs': 2, 'updates': 16},
               'rho1.5': {'rho': 1.5, 'epochs': 1, 'updates': 8}}
    c8 = protocol.get('calibration', {})
    expected_cases = [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070]
    expected_root = '/scratch/users/ntu/yguo017/fastwam-vq-calibration-suites-20261013'
    expected_remote = '/scratch/users/ntu/yguo017/fastwam-vq-c8-action-latent-plus-20261015'
    training = {
        'learning_rate': 0.0005, 'gradient_norm_clip': 1.0, 'optimizer': 'Adam',
        'optimizer_reset_after_update1': True, 'reference_max_abs_tolerance': 0.002,
        'reference_rmse_tolerance': 0.001, 'save_reserve_seconds': 120,
    }
    workers = {
        'workers': 4, 'rho0.1': [0, 1], 'rho1.5': [2, 3],
        'indices_per_worker': 'index % 2 == worker_id % 2',
        'variants_per_worker': 140,
    }
    expected = {
        'protocol': PROTOCOL, 'arms': list(ARMS), 'arm_configs': configs,
        'inference': inference, 'soft_deadline_seconds': 85500,
        'source_parity_max_abs_tolerance': 1e-6,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('C8 action-latent protocol differs from the frozen worker/inference contract')
    if (protocol.get('remote_root') != expected_remote
            or protocol.get('reference_directory') != expected_remote + '/results/reference'):
        raise ValueError('C8 action-latent protocol changed the frozen results/reference root')
    if any(protocol.get('cohort', {}).get(key) != value for key, value in cohort.items()):
        raise ValueError('C8 action-latent protocol changed the frozen paired cohort or seeds')
    if any(protocol.get('training', {}).get(key) != value for key, value in training.items()):
        raise ValueError('C8 action-latent protocol changed the frozen training recipe')
    if any(protocol.get('worker_assignment', {}).get(key) != value for key, value in workers.items()):
        raise ValueError('C8 action-latent protocol changed the four-worker parity partition')
    if (c8.get('root') != expected_root
            or c8.get('plan_file') != expected_root + '/data/plan.json'
            or c8.get('bank_directory') != expected_root + '/results/c8/banks/micro_scale_vq_a4'
            or c8.get('source_job_id') != '25789685.pbs101'
            or c8.get('case_ids') != expected_cases or c8.get('cohort') != 'c8'
            or c8.get('domain') != 'original' or c8.get('contexts') != 8
            or c8.get('per_suite') != 2 or c8.get('suites') != 4 or c8.get('seed_index') != 0):
        raise ValueError('C8 action-latent protocol changed its frozen C8 source binding')
    quantization = protocol.get('quantization', {})
    if (quantization.get('kv_cache') != 'bf16' or quantization.get('native_int4_kernel') is not False
            or quantization.get('transform_implementation') != 'batched_h128_v2'):
        raise ValueError('C8 action-latent protocol changed the decoded-BF16/transform execution path')
    references = protocol.get('references', {})
    expected_references = {
        'rotation_results': '/scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008/results',
        'scalar_results': '/scratch/users/ntu/yguo017/fastwam-smooth-scalar-plus-20261009/results',
        'vq_results': '/scratch/users/ntu/yguo017/fastwam-vq-plus-20261012/results',
        'expected_reference_slots': 1680,
        'arms': ['bf16', 'quarot_adapted_w4a4', 'spinquant_adapted_w4a4',
                 'smooth05_hadamard_scalar_w4a4', 'row_scale_vq_a4', 'micro_scale_vq_a4'],
    }
    if any(references.get(key) != value for key, value in expected_references.items()):
        raise ValueError('C8 action-latent protocol changed its six frozen reference arms')
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
        raise ValueError('Missing trained C8 install metadata')
    config = protocol['arm_configs'][arm]
    calibration = protocol['calibration']
    expected = {
        'arm': arm,
        'source_bank': calibration['bank_directory'],
        'source_job_id': calibration['source_job_id'],
        'source_protocol': 'fastwam-vq-calibration-suites-v1',
        'source_cohort': 'c8',
        'source_case_ids': calibration['case_ids'],
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
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Trained C8 install metadata differs from protocol for {arm}')
    for key in ('normalization_c', 'latent_weight'):
        value = metadata.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'Invalid trained C8 {key}')
    if not math.isclose(metadata['latent_weight'], config['rho'] * metadata['normalization_c'],
                        rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('Trained C8 latent_weight differs from rho * normalization_c')
    return metadata


def validate_record(record, rows, worker_id, protocol, install_metadata):
    arm, indices = worker_arm(worker_id), worker_indices(worker_id)
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
        raise ValueError(f'Trained C8 decoded-BF16/A4 execution evidence differs: {arm}/{index}')
    seconds = record.get('episode_seconds')
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Invalid episode wall time')
    replans = record.get('replans')
    metrics = record.get('metrics')
    if type(replans) is not int or replans <= 0 or not isinstance(metrics, list) or len(metrics) != replans:
        raise ValueError('Missing or inconsistent action-query metrics')
    if record.get('transform_implementation') != 'batched_h128_v2' or not record.get('pbs_job_id'):
        raise ValueError('Missing trained C8 transform/PBS identity')
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


def recover_records(path, rows, worker_id, protocol, install_metadata):
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
        index = validate_record(record, rows, worker_id, protocol, install_metadata)
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
    expected_prefix = worker_indices(worker_id)[:len(indices)]
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


def _setup_receipt(path, arm, protocol):
    receipt = read_json(path)
    max_abs = receipt.get('max_abs_vs_reference')
    if (receipt.get('passed') is not True or receipt.get('arm') != arm
            or receipt.get('observed_modules') != 614
            or isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
            or not math.isfinite(max_abs)
            or max_abs > protocol['source_parity_max_abs_tolerance']):
        raise ValueError('Trained C8 action replay did not pass')
    metadata = validate_install_metadata(receipt.get('install_metadata'), protocol, arm)
    return receipt, metadata


def run_worker(args):
    import runner
    runner.guard()
    from vq_adapter import TrainedRuntime, validate_training_artifacts

    p = validate_protocol(read_json(args.protocol))
    arm, indices = worker_arm(args.worker_id), worker_indices(args.worker_id)
    _, rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / 'protocol.json'
    if frozen.exists() and read_json(frozen) != p:
        raise ValueError('Cannot resume with changed C8 action-latent configuration')
    save(frozen, p)

    results_root = args.out.parent
    training_dir = results_root / 'training' / arm
    reference_dir = Path(p['reference_directory'])
    preflight = validate_training_artifacts(p, arm, training_dir, reference_dir)
    setup_path = args.out / 'setup_receipt.json'
    existing_metadata = None
    if setup_path.is_file():
        _, existing_metadata = _setup_receipt(setup_path, arm, p)
    path = args.out / 'episodes.jsonl'
    completed = recover_records(path, rows, args.worker_id, p, existing_metadata)
    if completed and existing_metadata is None:
        raise ValueError('Saved episodes exist without their setup/parity receipt')

    runtime, metadata, soft_stop = None, existing_metadata, False
    allocation_start = allocation_start_epoch()

    def receipt(status, **extra):
        return {'protocol': PROTOCOL, 'worker_id': args.worker_id, 'arm': arm,
                'status': status, 'complete': status == 'complete', 'episodes': len(completed),
                'completed': len(completed), 'expected': 140, 'soft_deadline_reached': soft_stop,
                **worker_coverage(args.worker_id, completed),
                'arm_episodes': {arm: len(completed)},
                'install_metadata': metadata, 'setup_receipt_file': 'setup_receipt.json', **extra}

    try:
        save(args.out / 'progress.json', receipt('running', stage='model_loading'))
        if len(completed) != 140:
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
                validate_record(record, rows, args.worker_id, p, metadata)
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
            _, metadata = _setup_receipt(setup_path, arm, p)
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
    run_worker(parser.parse_args())


if __name__ == '__main__':
    main()
