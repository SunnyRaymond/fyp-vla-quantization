"""Run one persistent worker for the frozen C80 per-channel + clipping cohort."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = 'fastwam-vq-c80-per-channel-plus-v1'
COHORT_PROTOCOL = 'fastwam-rotation-baselines-v1'
ARMS = ('per_channel_clip',)
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
    if (protocol.get('protocol') != PROTOCOL or protocol.get('arm') != 'per_channel_clip'
            or protocol.get('arms') != list(ARMS)):
        raise ValueError('C80 per-channel protocol/arm identity differs')
    inference = {
        'mode': 'idm', 'video_steps': 10, 'action_steps': 10, 'sigma_shift': 1.0,
        'action_horizon': 32, 'execute_per_replan': 10, 'action_ensembler': False,
        'settling_steps': 30, 'max_steps_normal': 400, 'max_steps_long': 700,
    }
    if protocol.get('inference') != inference:
        raise ValueError('C80 per-channel inference contract changed')
    cohort = protocol.get('cohort', {})
    expected_cohort = {
        'manifest_protocol': COHORT_PROTOCOL, 'variants_per_arm': 280,
        'total_new_episodes': 280, 'suites': 4, 'dimensions': 7,
        'variants_per_suite': 70, 'variants_per_dimension': 40,
        'variants_per_cell': 10, 'state_id': 0, 'env_seed_base': 800_000_000,
        'sampler_seed_base': 900_000_000, 'seed_stride': 1000,
        'paired_seeds_across_arms': True, 'sampling_changed': False,
    }
    if any(cohort.get(key) != value for key, value in expected_cohort.items()):
        raise ValueError('C80 per-channel cohort or paired seeds changed')
    calibration = protocol.get('calibration', {})
    if (calibration.get('case_ids') != list(range(2000, 2080))
            or calibration.get('contexts') != 80 or calibration.get('cohort') != 'c80'
            or calibration.get('per_suite') != 20):
        raise ValueError('C80 per-channel calibration binding changed')
    expected_order = [2000 + 20 * suite + offset for offset in range(20) for suite in range(4)]
    if calibration.get('update_order') != expected_order:
        raise ValueError('C80 per-channel update order changed')
    training = protocol.get('training', {})
    if (training.get('additional_updates') != 80 or training.get('epochs') != 1
            or training.get('rho') != 1.5 or training.get('optimizer') != 'Adam'
            or training.get('checkpoint_selection') !=
            'fixed final update80 only; no held-out checkpoint selection or early stopping'):
        raise ValueError('C80 per-channel source training contract changed')
    clipping, scaling = protocol.get('clipping', {}), protocol.get('scaling', {})
    if (clipping.get('minimum_ratio') != 0.5 or clipping.get('maximum_ratio') != 1.0
            or scaling.get('group_size') != 1 or scaling.get('minimum_log2') != -1.0
            or scaling.get('maximum_log2') != 1.0):
        raise ValueError('C80 per-channel clipping/scaling bounds changed')
    workers = protocol.get('worker_assignment', {})
    if (workers.get('workers') != 4 or workers.get('per_channel_clip') != [0, 1, 2, 3]
            or workers.get('indices_per_worker') != 'index % 4 == worker_id'
            or workers.get('variants_per_worker') != 70):
        raise ValueError('C80 per-channel worker partition changed')
    allocation = protocol.get('allocation', {})
    if (allocation.get('jobs') != 1 or allocation.get('gpus') != 4
            or allocation.get('walltime') != '24:00:00'
            or allocation.get('episodes_per_job') != 280):
        raise ValueError('C80 per-channel allocation changed')
    reference = protocol.get('closed_loop_reference', {})
    if (reference.get('case_id') != 2000 or reference.get('seed_index') != 0
            or reference.get('max_abs_tolerance') != 1e-6
            or reference.get('source_metric_absolute_tolerance') != 1e-7
            or reference.get('source_metric_relative_tolerance') != 1e-5):
        raise ValueError('C80 per-channel setup/parity gate changed')
    references = protocol.get('references', {})
    if (references.get('reference_episode_slots') != 2800
            or len(references.get('arms', [])) != 10
            or references.get('c80_protocol') != 'fastwam-vq-c80-clip-joint-plus-v1'
            or not references.get('c80_clip_evaluation_job')
            or not references.get('c80_joint_evaluation_job')):
        raise ValueError('C80 per-channel frozen references changed')
    source = protocol.get('sources', {})
    if (source.get('training_protocol') != 'fastwam-vq-per-channel-scaling-clipping-v1'
            or not source.get('training_job') or not source.get('training_results')):
        raise ValueError('C80 per-channel source training binding is incomplete')
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


def expected_install_metadata(protocol):
    source, calibration = protocol['sources'], protocol['calibration']
    return {
        'protocol': PROTOCOL, 'arm': 'per_channel_clip',
        'source_protocol': source['training_protocol'],
        'source_training_job': source['training_job'],
        'source_bank': calibration['bank_directory'],
        'baseline_deployment': source['baseline_deployment'],
        'calibration_ids': calibration['case_ids'],
        'updates': 80, 'module_count': 614, 'execution': 'decoded_bf16_vq',
        'activation_bits': 4, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
        'native_int4_gemm_calls': 0,
        'transform_implementation': 'batched_h128_v2', 'smooth_alpha': 0.5,
        'hadamard_block': 128, 'scale_block': 32,
        'fixed_indices': True, 'fixed_scale_exponents': True,
        'train_weights': False, 'train_clips': True, 'train_channel_scales': True,
        'channel_scale_group_size': 1, 'scale_metadata_dtype': 'float32_log2',
        'cached_compensated_weight_evaluation': True,
    }


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def validate_install_metadata(metadata, protocol, arm):
    if arm not in ARMS or not isinstance(metadata, dict):
        raise ValueError('Missing C80 per-channel install metadata')
    expected = expected_install_metadata(protocol)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('C80 per-channel install metadata differs from the frozen source')
    source_expected = protocol.get('source_expected', {})
    if metadata.get('source_cached_query_parity') != source_expected.get('cached_query_parity'):
        raise ValueError('C80 per-channel cached-query parity differs from its source receipt')
    ratios = metadata.get('ratios')
    if (not isinstance(ratios, dict) or ratios.get('count') != 614
            or any(not _finite_number(ratios.get(key)) for key in ('min', 'mean', 'max'))
            or ratios['min'] < protocol['clipping']['minimum_ratio']
            or ratios['max'] > protocol['clipping']['maximum_ratio']
            or ratios['min'] > ratios['mean'] or ratios['mean'] > ratios['max']
            or type(ratios.get('below_one')) is not int or not 0 <= ratios['below_one'] <= 614):
        raise ValueError('Invalid C80 per-channel clipping-ratio summary')
    scaling = metadata.get('scaling')
    if not isinstance(scaling, dict):
        raise ValueError('Missing C80 per-channel scaling summary')
    expected_channels = source_expected.get('scale_channels', 1_806_863)
    expected_bytes = source_expected.get('raw_scale_metadata_bytes', expected_channels * 4)
    if (scaling.get('modules') != 614 or scaling.get('group_size') != 1
            or scaling.get('channels') != expected_channels
            or scaling.get('raw_scale_metadata_bytes') != expected_bytes
            or scaling.get('cached_compensated_weight_evaluation') is not True
            or any(not _finite_number(scaling.get(key))
                   for key in ('scale_min', 'scale_max', 'mean_abs_log2'))
            or scaling['scale_min'] < 0.5 or scaling['scale_max'] > 2.0
            or scaling['scale_min'] > scaling['scale_max']
            or scaling['mean_abs_log2'] < 0 or scaling['mean_abs_log2'] > 1):
        raise ValueError('Invalid C80 per-channel scaling summary')
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
    tolerance = protocol['closed_loop_reference']['max_abs_tolerance']
    max_abs = receipt.get('max_abs_vs_reference')
    parity_fields = ('uncached_cached_action_max_abs', 'uncached_cached_future_max_abs',
                     'actual_infer_action_max_abs')
    if (receipt.get('passed') is not True or receipt.get('protocol') != PROTOCOL
            or receipt.get('arm') != arm or receipt.get('case_id') != 2000
            or receipt.get('seed_index') != 0
            or receipt.get('sampler_seed') != reference_action['sampler_seed']
            or receipt.get('reference_actual_infer_parity') != reference_action['actual_infer_parity']
            or receipt.get('observed_modules') != 614
            or receipt.get('training_receipt_status') != 'complete'
            or receipt.get('startup_queries') != 3
            or receipt.get('source_endpoint_metrics_passed') is not True
            or not _finite_number(max_abs) or max_abs < 0 or max_abs > tolerance
            or any(not _finite_number(receipt.get(key))
                   or receipt[key] < 0 or receipt[key] > tolerance for key in parity_fields)):
        raise ValueError('C80 source endpoint or live action/future parity gate did not pass')
    metadata = validate_install_metadata(receipt.get('install_metadata'), protocol, arm)
    return receipt, metadata

def run_worker(args):
    from vq_adapter import validate_training_artifacts

    p = validate_protocol(read_json(args.protocol))
    arm = args.arm
    if arm not in ARMS:
        raise ValueError(f'--arm must be {ARMS[0]}')
    arm, indices = worker_arm(args.worker_id, arm), worker_indices(args.worker_id, arm)
    _, rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / 'protocol.json'
    if frozen.exists() and read_json(frozen) != p:
        raise ValueError('Cannot resume with a changed C80 per-channel protocol')
    save(frozen, p)

    # Reuse the source training directory directly; this worker never retrains.
    training_dir = Path(p['sources']['training_results'])
    preflight = validate_training_artifacts(p, arm, training_dir)
    import runner
    runner.guard()
    from vq_adapter import TrainedRuntime
    setup_path = args.out / 'setup_receipt.json'
    existing_metadata = None
    if setup_path.is_file():
        _, existing_metadata = _setup_receipt(
            setup_path, arm, p, preflight['reference_action'])
    path = args.out / 'episodes.jsonl'
    completed = recover_records(path, rows, args.worker_id, p, existing_metadata, arm)
    if completed and existing_metadata is None:
        raise ValueError('Saved episodes exist without their source setup/parity receipt')

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
            metadata = validate_install_metadata(runtime.install_metadata, p, arm)
            if existing_metadata is not None and metadata != existing_metadata:
                raise ValueError('Resumed worker loaded a different source-trained deployment')
            setup = runtime.validate_reference()
            save(setup_path, setup)
            _, metadata = _setup_receipt(setup_path, arm, p, preflight['reference_action'])
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
                    'pbs_job_id': record['pbs_job_id'],
                    'transform_implementation': record['transform_implementation'],
                    'index': index, 'completed_episodes': len(completed),
                    'episode_seconds': record['episode_seconds'],
                    'replans': record['replans'], 'subsequent_queries': len(warm),
                    'full_query_wall_ms_mean': sum(m['full_query_wall_ms'] for m in warm) / len(warm) if warm else None,
                    'denoising_latency_ms_mean': sum(m['denoising_latency_ms'] for m in warm) / len(warm) if warm else None,
                })
                save(args.out / 'progress.json', receipt('running', stage='episode_saved', index=index))
                print(f'SLOT_COMPLETE worker={args.worker_id} arm={arm} index={index} success={record.get("success")}', flush=True)
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
    parser.add_argument('--arm', choices=ARMS, required=True)
    run_worker(parser.parse_args())


if __name__ == '__main__':
    main()
