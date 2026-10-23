"""Run one persistent worker on its frozen 140-episode shard."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = 'fastwam-diffusion-baselines-v1'
COHORT_PROTOCOL = 'fastwam-rotation-baselines-v1'
ARMS = ('svdquant_w4a4', 'viditq_w4a4')
TIMING_FIELDS = ('denoising_latency_ms', 'full_query_wall_ms', 'peak_allocated_gpu_GB',
                 'baseline_allocated_gpu_GB', 'peak_reserved_gpu_GB')
WORKER_ARMS = ('svdquant_w4a4', 'svdquant_w4a4', 'viditq_w4a4', 'viditq_w4a4')
MAX_TARGET_BPW = 4.60
MAX_BPW_GAP = 0.05


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with temp.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(directory):
    if os.name == 'nt':
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def worker_arm(worker_id, arm=None):
    if type(worker_id) is not int or worker_id not in range(4):
        raise ValueError('worker_id must be 0, 1, 2, or 3')
    assigned = WORKER_ARMS[worker_id]
    if arm is not None and arm != assigned:
        raise ValueError(f'worker {worker_id} is assigned to {assigned}')
    return assigned


def worker_indices(worker_id, arm=None):
    worker_arm(worker_id, arm)
    return list(range(worker_id % 2, 280, 2))


def environment_seed(index):
    return 800_000_000 + index * 1000


def sampler_seed(index, replan=0):
    return 900_000_000 + index * 1000 + replan


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def bank_manifest_job_id(metadata):
    return metadata.get('bank_manifest_pbs_jobidstr', metadata.get('bank_manifest_pbs_jobid'))


def validate_protocol(protocol):
    if (protocol.get('protocol') != PROTOCOL or protocol.get('arms') != list(ARMS)
            or protocol.get('soft_deadline_seconds') != 85500):
        raise ValueError('Diffusion baseline protocol identity/deadline changed')
    cohort = protocol.get('cohort', {})
    expected_cohort = {
        'manifest_protocol': COHORT_PROTOCOL, 'variants_per_arm': 280,
        'total_new_episodes': 560, 'suites': 4, 'dimensions': 7,
        'variants_per_suite': 70, 'variants_per_dimension': 40,
        'variants_per_cell': 10, 'state_id': 0, 'env_seed_base': 800_000_000,
        'sampler_seed_base': 900_000_000, 'seed_stride': 1000,
        'paired_seeds_across_arms': True, 'sampling_changed': False,
    }
    if any(cohort.get(key) != value for key, value in expected_cohort.items()):
        raise ValueError('Frozen C80 cohort or paired seeds changed')
    calibration = protocol.get('calibration', {})
    if (calibration.get('case_ids') != list(range(2000, 2080))
            or calibration.get('contexts') != 80
            or calibration.get('optimizer_updates') != 0):
        raise ValueError('Frozen 80-context calibration contract changed')
    quant = protocol.get('quantization', {})
    expected_quant = {
        'target_linears': 614, 'weight_bits': 4, 'activation_bits': 4,
        'group_size': 128, 'kv_cache_dtype': 'bf16',
        'execution': 'decoded_bf16_reference', 'native_low_bit_kernel': False,
        'max_relative_target_bpw_gap': MAX_BPW_GAP, 'maximum_target_bpw': MAX_TARGET_BPW,
    }
    if any(quant.get(key) != value for key, value in expected_quant.items()):
        raise ValueError('Frozen W4A4/decoded-BF16 quantization contract changed')
    workers = protocol.get('worker_assignment', {})
    mapping = workers.get('mapping', {})
    expected_mapping = {
        '0': {'arm': ARMS[0], 'shard': 0}, '1': {'arm': ARMS[0], 'shard': 1},
        '2': {'arm': ARMS[1], 'shard': 0}, '3': {'arm': ARMS[1], 'shard': 1},
    }
    if (workers.get('workers') != 4 or mapping != expected_mapping
            or workers.get('indices_per_worker') != 'index % 2 == worker_id % 2'
            or workers.get('variants_per_worker') != 140):
        raise ValueError('Four-worker arm assignment or frozen shard changed')
    allocation = protocol.get('allocation', {})
    if (allocation.get('jobs') != 1 or allocation.get('gpus') != 4
            or allocation.get('walltime') != '24:00:00'
            or allocation.get('episodes_per_job') != 560):
        raise ValueError('Frozen four-GPU allocation changed')
    references = protocol.get('references', {})
    if (len(references.get('arms', [])) != 8
            or references.get('expected_reference_slots') != 2240
            or references.get('reference_episode_slots') != 2240):
        raise ValueError('Eight frozen reference arms/2240 slots changed')
    startup = protocol.get('startup_checks', {})
    if (startup.get('expected_modules') != 614
            or startup.get('bf16_teacher_max_abs_tolerance') != 1e-6
            or startup.get('cross_worker_action_max_abs_tolerance') != 1e-6):
        raise ValueError('Startup parity thresholds changed')
    return protocol


def load_manifest(path):
    doc = read_json(path)
    rows = doc.get('variants')
    if doc.get('protocol') != COHORT_PROTOCOL or not isinstance(rows, list) or len(rows) != 280:
        raise ValueError('Expected the original frozen 280-variant manifest')
    identities, cells = set(), collections.Counter()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get('index')) is not int
                or row['index'] != index or row.get('state_id', 0) != 0):
            raise ValueError(f'Invalid frozen manifest row {index}')
        identity = str(row.get('variant_id', ''))
        if not identity or identity in identities or 'task_id' not in row:
            raise ValueError('Duplicate/missing frozen variant identity')
        identities.add(identity)
        cells[row.get('suite'), row.get('dimension')] += 1
    if (len(cells) != 28 or set(cells.values()) != {10}
            or len({suite for suite, _ in cells}) != 4
            or len({dimension for _, dimension in cells}) != 7):
        raise ValueError('Expected frozen 4-suite x 7-dimension x 10-variant cohort')
    return doc, rows


def validate_install_metadata(metadata, protocol, arm):
    if arm not in ARMS or not isinstance(metadata, dict):
        raise ValueError('Missing diffusion install metadata')
    expected = {
        'protocol': PROTOCOL, 'arm': arm, 'weight_bits': 4, 'activation_bits': 4,
        'group_size': 128, 'execution': 'decoded_bf16_reference',
        'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
        'module_count': 614, 'calibration_contexts': 80, 'optimizer_updates': 0,
        'representation': ('svdquant_low_rank_residual' if arm == ARMS[0]
                           else 'viditq_channel_balance_rotation'),
        'transform_implementation': ('svdquant_smoothed_low_rank_v1' if arm == ARMS[0]
                                     else 'viditq_structured_hadamard_v1'),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Install metadata differs from the frozen {arm} method')
    bpw = metadata.get('encoded_target_bpw')
    if not _finite_number(bpw) or bpw <= 0 or bpw > MAX_TARGET_BPW:
        raise ValueError('encoded_target_bpw must be finite, positive, and <= 4.60')
    for key in ('encoded_target_bytes', 'target_weight_parameters'):
        if type(metadata.get(key)) is not int or metadata[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    rank = metadata.get('low_rank_global_rank')
    if (type(rank) is not int or rank < 0
            or (arm == ARMS[0] and rank <= 0)
            or (arm == ARMS[1] and rank != 0)):
        raise ValueError('Invalid method-specific low-rank rank')
    bank_job_id = bank_manifest_job_id(metadata)
    if not isinstance(bank_job_id, str) or not bank_job_id:
        raise ValueError('Missing calibration bank manifest PBS job id')
    if any(not isinstance(key, str) for key in metadata):
        raise ValueError('Install metadata keys must be strings')
    return metadata


def validate_setup_receipt(receipt, protocol, worker_id, arm, install_metadata=None):
    worker_arm(worker_id, arm)
    if not isinstance(receipt, dict):
        raise ValueError('Missing setup receipt')
    if (receipt.get('protocol') != PROTOCOL or receipt.get('worker_id') != worker_id
            or receipt.get('arm') != arm or receipt.get('passed') is not True
            or receipt.get('observed_modules') != 614
            or not isinstance(receipt.get('pbs_job_id'), str) or not receipt['pbs_job_id']):
        raise ValueError(f'Worker {worker_id} setup identity/gate failed')
    for key in ('bf16_teacher_max_abs', 'cross_worker_action_max_abs'):
        value = receipt.get(key)
        if not _finite_number(value) or value < 0 or value > 1e-6:
            raise ValueError(f'Worker {worker_id} startup parity {key} failed')
    selfcheck = receipt.get('component_selfcheck')
    if not isinstance(selfcheck, dict) or selfcheck.get('passed') is not True:
        raise ValueError(f'Worker {worker_id} component self-check failed')
    bank = receipt.get('bank_manifest')
    if (not isinstance(bank, dict) or bank.get('protocol') != PROTOCOL
            or bank.get('complete') is not True or bank.get('modules') != 614
            or bank.get('calibration_contexts') != 80):
        raise ValueError(f'Worker {worker_id} bank manifest summary is missing')
    installed = validate_install_metadata(receipt.get('install_metadata'), protocol, arm)
    if bank.get('pbs_job_id') != bank_manifest_job_id(installed):
        raise ValueError(f'Worker {worker_id} bank manifest job id differs from install metadata')
    if install_metadata is not None and installed != install_metadata:
        raise ValueError(f'Worker {worker_id} setup/install metadata disagree')
    return installed


def validate_record(record, rows, worker_id, protocol, install_metadata, arm):
    worker_arm(worker_id, arm)
    validate_install_metadata(install_metadata, protocol, arm)
    index = record.get('index')
    if type(index) is not int or index not in worker_indices(worker_id, arm):
        raise ValueError('Record is outside its frozen worker shard')
    row = rows[index]
    expected = {
        'protocol': PROTOCOL, 'arm': arm, 'worker_id': worker_id, 'complete': True,
        'suite': row['suite'], 'dimension': row['dimension'], 'task_id': int(row['task_id']),
        'state_id': 0, 'env_seed': environment_seed(index), 'sampler_seed': sampler_seed(index),
    }
    if (any(record.get(key) != value for key, value in expected.items())
            or str(record.get('variant_id')) != str(row['variant_id'])
            or type(record.get('success')) is not bool):
        raise ValueError(f'Frozen episode identity/seed differs: {arm}/{index}')
    implementation = install_metadata['transform_implementation']
    if record.get('transform_implementation') != implementation:
        raise ValueError('Episode transform differs from installed method')
    job_id = record.get('pbs_job_id')
    if not isinstance(job_id, str) or not job_id:
        raise ValueError('Episode is missing PBS job id')
    evidence = record.get('quantization_evidence')
    required = {**install_metadata, 'observed_module_count': 614,
                'native_int4_gemm_calls': 0, 'kv_packed_prefills': 0, 'kv_layer_reads': 0}
    if (not isinstance(evidence, dict)
            or any(evidence.get(key) != value for key, value in required.items())
            or type(evidence.get('quantized_linear_calls')) is not int
            or evidence['quantized_linear_calls'] <= 0):
        raise ValueError(f'Decoded-BF16/A4 execution evidence differs: {arm}/{index}')
    branch_call = 'low_rank_calls' if arm == ARMS[0] else 'rotation_calls'
    if type(evidence.get(branch_call)) is not int or evidence[branch_call] <= 0:
        raise ValueError(f'{arm} execution branch did not run')
    seconds = record.get('episode_seconds')
    if not _finite_number(seconds) or seconds < 0:
        raise ValueError('Invalid episode wall time')
    replans, metrics = record.get('replans'), record.get('metrics')
    if type(replans) is not int or replans <= 0 or not isinstance(metrics, list) or len(metrics) != replans:
        raise ValueError('Missing/inconsistent action-query metrics')
    for replan, metric in enumerate(metrics):
        query = metric.get('query_index_in_arm')
        if (type(query) is not int or query < 0
                or type(metric.get('first_query_in_arm')) is not bool
                or metric['first_query_in_arm'] != (query == 0)
                or metric.get('replan') != replan
                or metric.get('sampler_seed') != sampler_seed(index, replan)
                or metric.get('transform_implementation') != implementation
                or metric.get('pbs_job_id') != job_id
                or type(metric.get('first_query_in_run')) is not bool):
            raise ValueError('Invalid resumed implementation/run/query marker')
        for key in TIMING_FIELDS:
            value = metric.get(key)
            if not _finite_number(value) or value < 0:
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
        temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.recovered.tmp')
        with temp.open('wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        _fsync_dir(path.parent)
    found, queries, seen_jobs = {}, [], set()
    for line in payload.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index = validate_record(record, rows, worker_id, protocol, install_metadata, arm)
        if index in found:
            raise ValueError('Duplicate completed episode')
        found[index] = record
        for metric in record['metrics']:
            query = metric['query_index_in_arm']
            queries.append(query)
            job = metric['pbs_job_id']
            if metric['first_query_in_run'] != (job not in seen_jobs):
                raise ValueError('Invalid first formal query marker within a worker run')
            seen_jobs.add(job)
    indices = list(found)
    if indices != worker_indices(worker_id, arm)[:len(indices)]:
        raise ValueError('Saved episodes are not the frozen worker prefix')
    if queries != list(range(len(queries))):
        raise ValueError('Saved query sequence is not a consecutive worker prefix')
    return found


def append_episode(path, record):
    payload = (json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('Episode append made no progress')
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


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


def allocation_elapsed_seconds(start_epoch, now_epoch):
    if (not _finite_number(start_epoch) or not _finite_number(now_epoch)
            or start_epoch <= 0 or now_epoch < start_epoch):
        raise ValueError('Invalid allocation start/now epoch')
    return float(now_epoch) - float(start_epoch)


def run_worker(args):
    import runner
    runner.guard()
    protocol = validate_protocol(read_json(args.protocol))
    arm, indices = worker_arm(args.worker_id), worker_indices(args.worker_id)
    _, rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / 'protocol.json'
    if frozen.exists() and read_json(frozen) != protocol:
        raise ValueError('Cannot resume with a changed diffusion baseline protocol')
    save(frozen, protocol)

    setup_path, episode_path = args.out / 'setup_receipt.json', args.out / 'episodes.jsonl'
    old_setup = read_json(setup_path) if setup_path.is_file() else None
    old_metadata = (validate_setup_receipt(old_setup, protocol, args.worker_id, arm)
                    if old_setup is not None else None)
    completed = recover_records(episode_path, rows, args.worker_id, protocol, old_metadata, arm)
    if completed and old_metadata is None:
        raise ValueError('Saved episodes exist without their setup receipt')
    start_epoch = allocation_start_epoch()
    results_root = args.out.parent.parent
    runtime, metadata, soft_stop = None, old_metadata, False

    def receipt(status, **extra):
        return {
            'protocol': PROTOCOL, 'worker_id': args.worker_id, 'arm': arm,
            'status': status, 'complete': status == 'complete',
            'episodes': len(completed), 'completed': len(completed), 'expected': 140,
            'soft_deadline_reached': soft_stop, 'indices': indices,
            'completed_indices': list(completed), 'arm_episodes': {arm: len(completed)},
            'install_metadata': metadata, 'setup_receipt_file': 'setup_receipt.json', **extra,
        }

    try:
        if len(completed) != len(indices):
            from runtime import DiffusionRuntime
            save(args.out / 'progress.json', receipt('running', stage='model_loading'))
            runtime = DiffusionRuntime(args.out, protocol, args.worker_id)
            setup = runtime.prepare(results_root)
            metadata = validate_install_metadata(runtime.install_metadata, protocol, arm)
            validate_setup_receipt(setup, protocol, args.worker_id, arm, metadata)
            if old_metadata is not None and metadata != old_metadata:
                raise ValueError('Resumed worker loaded different diffusion install metadata')
            save(setup_path, setup)
            runtime.arm_calls[arm] = sum(len(row['metrics']) for row in completed.values())
            runtime.calls = runtime.arm_calls[arm]
            save(args.out / 'progress.json', receipt('running', stage='episode'))
            for index in indices:
                if index in completed:
                    continue
                if allocation_elapsed_seconds(start_epoch, time.time()) >= protocol['soft_deadline_seconds']:
                    soft_stop = True
                    break
                save(args.out / 'progress.json', receipt('running', stage='episode', index=index))
                record = runtime.episode(rows[index], index)
                if not isinstance(record, dict):
                    raise TypeError('runtime.episode must return an episode dictionary')
                record.setdefault('worker_id', args.worker_id)
                validate_record(record, rows, args.worker_id, protocol, metadata, arm)
                append_episode(episode_path, record)
                completed[index] = record
                save(args.out / 'progress.json', receipt('running', stage='episode_saved', index=index))
                print(f'SLOT_COMPLETE worker={args.worker_id} arm={arm} index={index} success={record["success"]}',
                      flush=True)
        else:
            if old_setup is None:
                raise ValueError('Completed worker is missing its setup receipt')
            metadata = validate_setup_receipt(old_setup, protocol, args.worker_id, arm)
        status = 'checkpointed' if soft_stop else 'complete'
        if not soft_stop and set(completed) != set(indices):
            raise RuntimeError('Worker completed coverage differs from its frozen 140-slot shard')
        result = receipt(status)
        save(args.out / 'summary.json', result)
        save(args.out / 'progress.json', result)
        print(f'WORKER_{status.upper()} worker={args.worker_id} episodes={len(completed)}', flush=True)
    except Exception as exc:
        save(args.out / 'failure.json', {
            'protocol': PROTOCOL, 'worker_id': args.worker_id, 'arm': arm,
            'type': type(exc).__name__, 'message': str(exc), 'completed': len(completed),
        })
        raise
    finally:
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--worker-id', type=int, choices=range(4), required=True)
    parser.add_argument('--out', type=Path, required=True)
    run_worker(parser.parse_args())


if __name__ == '__main__':
    main()
