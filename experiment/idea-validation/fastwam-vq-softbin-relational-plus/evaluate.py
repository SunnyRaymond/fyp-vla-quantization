"""Run one resumable worker for the frozen softbin/relational C80 cohort."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

from runtime import expected_install_metadata, validate_source_receipt

PROTOCOL = 'fastwam-vq-softbin-relational-plus-v1'
COHORT_PROTOCOL = 'fastwam-rotation-baselines-v1'
ARMS = ('softbin', 'relational')
WORKER_ARMS = ('softbin', 'softbin', 'relational', 'relational')
TIMING_FIELDS = ('denoising_latency_ms', 'full_query_wall_ms', 'peak_allocated_gpu_GB',
                 'baseline_allocated_gpu_GB', 'peak_reserved_gpu_GB')
_SOURCE_RECEIPTS = {}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(directory):
    if os.name == 'nt':
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def worker_coverage(worker_id, completed):
    return {'indices': worker_indices(worker_id), 'completed_indices': list(completed)}


def environment_seed(index):
    return 800_000_000 + index * 1000


def sampler_seed(index, replan=0):
    return 900_000_000 + index * 1000 + replan


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get('protocol') != PROTOCOL:
        raise ValueError('Softbin/relational protocol identity differs')
    if protocol.get('arms') != list(ARMS) or protocol.get('soft_deadline_seconds') != 85500:
        raise ValueError('Frozen arms or soft deadline changed')

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
        raise ValueError('Frozen 280-episode paired cohort changed')

    calibration = protocol.get('calibration', {})
    expected_ids = list(range(2000, 2080))
    expected_order = [2000 + 20 * suite + offset for offset in range(20) for suite in range(4)]
    if (calibration.get('case_ids') != expected_ids or calibration.get('contexts') != 80
            or calibration.get('cohort') != 'c80' or calibration.get('per_suite') != 20
            or calibration.get('update_order') != expected_order or calibration.get('seed_index') != 0):
        raise ValueError('Frozen C80 calibration ids or update order changed')

    training = protocol.get('training', {})
    if (training.get('additional_updates') != 80 or training.get('epochs') != 1
            or training.get('rho') != 1.5 or training.get('optimizer') != 'Adam'
            or training.get('fresh_optimizer_each_arm') is not True
            or training.get('checkpoint_selection') !=
            'fixed final update80 only; no held-out checkpoint selection or early stopping'):
        raise ValueError('Frozen final80 source-training contract changed')
    clipping, scaling = protocol.get('clipping', {}), protocol.get('scaling', {})
    if (clipping.get('minimum_ratio') != 0.5 or clipping.get('maximum_ratio') != 1.0
            or scaling.get('group_size') != 1 or scaling.get('minimum_log2') != -1.0
            or scaling.get('maximum_log2') != 1.0):
        raise ValueError('Frozen clipping or per-channel scaling bounds changed')

    quant = protocol.get('quantization', {})
    expected_quant = {
        'target_linears': 614, 'activation_bits': 4, 'nominal_weight_bits': 4,
        'execution': 'decoded_bf16_vq', 'kv_cache_dtype': 'bf16',
        'native_low_bit_kernel': False, 'transform_implementation': 'batched_h128_v2',
        'smooth_alpha': 0.5, 'hadamard_block': 128, 'scale_block': 32,
        'channel_scale_group_size': 1, 'scale_metadata_dtype': 'float32_log2',
        'fixed_indices': True, 'fixed_scale_exponents': True,
        'frozen_codebooks_and_row_gains': True, 'cache_compensated_weights_once': True,
    }
    if any(quant.get(key) != value for key, value in expected_quant.items()):
        raise ValueError('Frozen decoded-BF16/A4 execution contract changed')

    inference = {
        'mode': 'idm', 'video_steps': 10, 'action_steps': 10, 'sigma_shift': 1.0,
        'action_horizon': 32, 'execute_per_replan': 10, 'action_ensembler': False,
        'settling_steps': 30, 'max_steps_normal': 400, 'max_steps_long': 700,
    }
    if protocol.get('inference') != inference:
        raise ValueError('Frozen inference contract changed')

    workers = protocol.get('worker_assignment', {})
    expected_mapping = {
        '0': {'arm': 'softbin', 'shard': 0}, '1': {'arm': 'softbin', 'shard': 1},
        '2': {'arm': 'relational', 'shard': 0}, '3': {'arm': 'relational', 'shard': 1},
    }
    if (workers.get('workers') != 4 or workers.get('mapping') != expected_mapping
            or workers.get('indices_per_worker') != 'index % 2 == worker_id % 2'
            or workers.get('variants_per_worker') != 140):
        raise ValueError('Four-worker assignment or frozen shard changed')
    allocation = protocol.get('allocation', {})
    if (allocation.get('jobs') != 1 or allocation.get('gpus') != 4
            or allocation.get('walltime') != '24:00:00'
            or allocation.get('episodes_per_job') != 560
            or allocation.get('gpu_monitor_seconds') != 15):
        raise ValueError('Frozen PBS allocation or telemetry interval changed')

    startup = protocol.get('startup_checks', {})
    if (startup.get('case_id') != 2000 or startup.get('seed_index') != 0
            or startup.get('expected_modules') != 614
            or startup.get('startup_queries_per_worker') != 3
            or startup.get('saved_source_action_max_abs_tolerance') != 1e-6
            or startup.get('saved_source_future_max_abs_tolerance') != 1e-6
            or startup.get('uncached_cached_max_abs_tolerance') != 0.0
            or startup.get('actual_infer_action_max_abs_tolerance') != 1e-6):
        raise ValueError('Frozen source/cache startup gates changed')

    source_expected = protocol.get('source_expected', {})
    if (source_expected.get('updates') != 80 or source_expected.get('new_training_updates') != 0
            or source_expected.get('target_linears') != 614
            or source_expected.get('scale_channels') != 1_806_863
            or source_expected.get('scale_metadata_bytes') != 7_227_452
            or source_expected.get('cached_query_parity') !=
            {'action_max_abs': 0.0, 'future_max_abs': 0.0}):
        raise ValueError('Frozen source-receipt coverage/parity expectations changed')

    references = protocol.get('references', {})
    expected_arms = [
        'bf16', 'quarot_adapted_w4a4', 'spinquant_adapted_w4a4',
        'smooth05_hadamard_scalar_w4a4', 'row_scale_vq_a4', 'micro_scale_vq_a4',
        'rho0.1', 'rho1.5', 'clip_only', 'joint', 'per_channel_clip',
        'svdquant_w4a4', 'viditq_w4a4',
    ]
    if (references.get('arms') != expected_arms
            or references.get('reference_episode_slots') != 3640
            or references.get('expected_reference_slots') != 3640):
        raise ValueError('Frozen 13-arm/3640-slot reference contract changed')
    for key in ('rotation_results', 'scalar_results', 'vq_results', 'c8_trained_results',
                'c80_results', 'per_channel_results', 'diffusion_results'):
        if not isinstance(references.get(key), str) or not references[key]:
            raise ValueError(f'Missing frozen reference path: {key}')
    if (references.get('per_channel_protocol') != 'fastwam-vq-c80-per-channel-plus-v1'
            or references.get('diffusion_protocol') != 'fastwam-diffusion-baselines-v1'
            or references.get('per_channel_job') != '25863760.pbs101'
            or references.get('diffusion_job') != '25878591.pbs101'):
        raise ValueError('Frozen per-channel/diffusion reference identity changed')
    if set(protocol.get('arm_sources', {})) != set(ARMS):
        raise ValueError('Frozen source bindings must cover exactly both new arms')
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


def _source_receipt(protocol, arm):
    source = protocol['arm_sources'][arm]
    cache_key = (protocol['protocol'], arm, source['training_receipt'])
    if cache_key in _SOURCE_RECEIPTS:
        return _SOURCE_RECEIPTS[cache_key]
    receipt_path = Path(source['training_receipt'])
    if not receipt_path.is_file():
        # Local source receipts are small frozen fixtures for stdlib-only checks.
        receipt_path = Path(__file__).with_name(f'source_{arm}_training_receipt.json')
    receipt = validate_source_receipt(protocol, arm, read_json(receipt_path))
    _SOURCE_RECEIPTS[cache_key] = receipt
    return receipt


def validate_install_metadata(metadata, protocol, arm):
    if arm not in ARMS or not isinstance(metadata, dict):
        raise ValueError('Missing softbin/relational install metadata')
    expected = expected_install_metadata(protocol, arm)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Install metadata differs from the frozen {arm} deployment')
    if any(not isinstance(key, str) for key in metadata):
        raise ValueError('Install metadata keys must be strings')

    source = _source_receipt(protocol, arm)
    if (metadata.get('ratios') != source.get('clipping')
            or metadata.get('scaling') != source.get('scaling')
            or metadata.get('source_cached_query_parity') != source.get('cached_query_parity')):
        raise ValueError('Install clipping/scaling/source parity differs from frozen training receipt')
    ratios, scaling = metadata['ratios'], metadata['scaling']
    if (ratios.get('count') != 614
            or any(not _finite_number(ratios.get(key)) for key in ('min', 'mean', 'max'))
            or ratios['min'] < 0.5 or ratios['max'] > 1.0
            or ratios['min'] > ratios['mean'] or ratios['mean'] > ratios['max']
            or type(ratios.get('below_one')) is not int or not 0 <= ratios['below_one'] <= 614):
        raise ValueError('Invalid frozen clipping-ratio summary')
    if (scaling.get('modules') != 614 or scaling.get('group_size') != 1
            or scaling.get('channels') != 1_806_863
            or scaling.get('raw_scale_metadata_bytes') != 7_227_452
            or scaling.get('scale_metadata_dtype') != 'float32_log2'
            or scaling.get('cached_compensated_weight_evaluation') is not True
            or any(not _finite_number(scaling.get(key))
                   for key in ('scale_min', 'scale_median', 'scale_max', 'mean_abs_log2'))
            or scaling['scale_min'] < 0.5 or scaling['scale_max'] > 2.0
            or scaling['scale_min'] > scaling['scale_median']
            or scaling['scale_median'] > scaling['scale_max']
            or scaling['mean_abs_log2'] < 0 or scaling['mean_abs_log2'] > 1):
        raise ValueError('Invalid frozen per-channel scaling summary')
    if metadata.get('cached_compensated_weight_evaluation') is not True:
        raise ValueError('Compensated weights must be cached for evaluation')
    return metadata


def validate_setup_receipt(receipt, protocol, worker_id, arm, install_metadata=None):
    worker_arm(worker_id, arm)
    if not isinstance(receipt, dict):
        raise ValueError('Missing setup receipt')
    source = protocol['arm_sources'][arm]
    if (receipt.get('protocol') != PROTOCOL or receipt.get('worker_id') != worker_id
            or receipt.get('arm') != arm or receipt.get('passed') is not True
            or receipt.get('observed_modules') != 614
            or receipt.get('case_id') != 2000 or receipt.get('seed_index') != 0
            or receipt.get('sampler_seed') != 910_004_000
            or not isinstance(receipt.get('pbs_job_id'), str) or not receipt['pbs_job_id']
            or receipt.get('source_training_job') != source['training_job']
            or receipt.get('source_endpoint_job') != source['endpoint_job']
            or receipt.get('training_updates') != 80 or receipt.get('new_training_updates') != 0
            or receipt.get('startup_queries') != 3
            or receipt.get('cached_compensated_weights') is not True):
        raise ValueError(f'Worker {worker_id} setup identity/source gate failed')
    checks = protocol['startup_checks']
    for key, limit_key in (
        ('source_action_max_abs', 'saved_source_action_max_abs_tolerance'),
        ('source_future_max_abs', 'saved_source_future_max_abs_tolerance'),
        ('actual_infer_action_max_abs', 'actual_infer_action_max_abs_tolerance'),
    ):
        value = receipt.get(key)
        if not _finite_number(value) or value < 0 or value > checks[limit_key]:
            raise ValueError(f'Worker {worker_id} startup parity {key} failed')
    for key in ('uncached_cached_action_max_abs', 'uncached_cached_future_max_abs'):
        value = receipt.get(key)
        if not _finite_number(value) or value != checks['uncached_cached_max_abs_tolerance']:
            raise ValueError(f'Worker {worker_id} uncached/cached parity {key} failed')
    installed = validate_install_metadata(receipt.get('install_metadata'), protocol, arm)
    if install_metadata is not None and installed != install_metadata:
        raise ValueError(f'Worker {worker_id} setup/install metadata disagree')
    return installed


def validate_record(record, rows, worker_id, protocol, install_metadata, arm):
    worker_arm(worker_id, arm)
    validate_install_metadata(install_metadata, protocol, arm)
    index = record.get('index')
    indices = worker_indices(worker_id, arm)
    if type(index) is not int or index not in indices:
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
    job_id = record.get('pbs_job_id')
    if (record.get('transform_implementation') != implementation
            or not isinstance(job_id, str) or not job_id):
        raise ValueError('Episode transform or PBS identity differs from installed method')
    evidence = record.get('vq_evidence')
    if (not isinstance(evidence, dict)
            or any(evidence.get(key) != value for key, value in install_metadata.items())
            or evidence.get('observed_module_count') != 614
            or evidence.get('native_int4_gemm_calls') != 0
            or evidence.get('kv_packed_prefills') != 0 or evidence.get('kv_layer_reads') != 0
            or type(evidence.get('quantized_linear_calls')) is not int
            or evidence['quantized_linear_calls'] <= 0):
        raise ValueError(f'Decoded-BF16 VQ/A4 episode evidence differs: {arm}/{index}')
    seconds = record.get('episode_seconds')
    if not _finite_number(seconds) or seconds < 0:
        raise ValueError('Invalid episode wall time')
    replans, metrics = record.get('replans'), record.get('metrics')
    if (type(replans) is not int or replans <= 0 or not isinstance(metrics, list)
            or len(metrics) != replans):
        raise ValueError('Missing or inconsistent action-query metrics')
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
            if not _finite_number(metric.get(key)) or metric[key] < 0:
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
            queries.append(metric['query_index_in_arm'])
            job = metric['pbs_job_id']
            if metric['first_query_in_run'] != (job not in seen_jobs):
                raise ValueError('Invalid first formal query marker within worker run')
            seen_jobs.add(job)
    indices = list(found)
    if indices != worker_indices(worker_id, arm)[:len(indices)]:
        raise ValueError('Saved episodes are not the frozen worker prefix')
    if queries != list(range(len(queries))):
        raise ValueError('Saved query sequence is not a consecutive worker prefix')
    return found


def append_episode(path, record):
    with Path(path).open('ab') as stream:
        stream.write((json.dumps(record, allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


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
        raise ValueError('Cannot resume with a changed softbin/relational protocol')
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
            'soft_deadline_reached': soft_stop, **worker_coverage(args.worker_id, completed),
            'arm_episodes': {arm: len(completed)}, 'install_metadata': metadata,
            'setup_receipt_file': 'setup_receipt.json', **extra,
        }

    try:
        if len(completed) != len(indices):
            from runtime import SoftbinRuntime
            save(args.out / 'progress.json', receipt('running', stage='model_loading'))
            runtime = SoftbinRuntime(args.out, protocol, args.worker_id)
            setup = runtime.prepare(results_root)
            metadata = validate_install_metadata(runtime.install_metadata, protocol, arm)
            validate_setup_receipt(setup, protocol, args.worker_id, arm, metadata)
            if old_metadata is not None and metadata != old_metadata:
                raise ValueError('Resumed worker loaded different source-trained metadata')
            save(setup_path, setup)
            runtime.arm_calls[arm] = sum(len(record['metrics']) for record in completed.values())
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
                print(f'SLOT_COMPLETE worker={args.worker_id} arm={arm} index={index} '
                      f'success={record["success"]}', flush=True)
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
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker-id', type=int, choices=range(4), required=True)
    run_worker(parser.parse_args())


if __name__ == '__main__':
    main()
