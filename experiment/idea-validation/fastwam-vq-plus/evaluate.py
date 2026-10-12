"""One persistent worker for the frozen two-arm, 560-episode VQ evaluation."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = 'fastwam-vq-plus-v1'
COHORT_PROTOCOL = 'fastwam-rotation-baselines-v1'
ARMS = ('row_scale_vq_a4', 'micro_scale_vq_a4')
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


def validate_record(record, rows, worker_id, bank):
    arm, indices = worker_arm(worker_id), worker_indices(worker_id)
    index = record.get('index')
    if type(index) is not int or index not in indices:
        raise ValueError('Record is outside its frozen worker assignment')
    row = rows[index]
    expected = {'protocol': PROTOCOL, 'arm': arm, 'worker_id': worker_id,
                'complete': True, 'suite': row['suite'], 'dimension': row['dimension'],
                'task_id': int(row['task_id']), 'state_id': 0,
                'env_seed': 800_000_000 + index * 1000, 'sampler_seed': 900_000_000 + index * 1000}
    if (any(record.get(key) != value for key, value in expected.items())
            or str(record.get('variant_id')) != str(row['variant_id'])
            or type(record.get('success')) is not bool):
        raise ValueError(f'Frozen episode identity/seed differs: {arm}/{index}')
    evidence = record.get('vq_evidence', {})
    required = {'arm': arm, 'source_bank': bank, 'module_count': 614,
                'observed_module_count': 614, 'execution': 'decoded_bf16_vq',
                'activation_bits': 4, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
                'frozen_bank_reused': True, 'native_int4_gemm_calls': 0,
                'kv_packed_prefills': 0, 'kv_layer_reads': 0}
    if any(evidence.get(key) != value for key, value in required.items()) or evidence.get('quantized_linear_calls', 0) <= 0:
        raise ValueError(f'Frozen VQ/A4 execution evidence differs: {arm}/{index}')
    seconds = record.get('episode_seconds')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid episode wall time')
    metrics = record.get('metrics')
    if not isinstance(metrics, list) or not metrics:
        raise ValueError('Missing action-query metrics')
    for metric in metrics:
        q = metric.get('query_index_in_arm')
        if (type(q) is not int or q < 0 or type(metric.get('first_query_in_arm')) is not bool
                or metric['first_query_in_arm'] != (q == 0)):
            raise ValueError('Invalid first/query marker')
        for key in TIMING_FIELDS:
            value = metric.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {key}')
    return index


def recover_records(path, rows, worker_id, bank):
    if not path.exists():
        return {}
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
    found, queries = {}, []
    for line in payload.splitlines():
        record = json.loads(line)
        index = validate_record(record, rows, worker_id, bank)
        if index in found:
            raise ValueError('Duplicate completed episode')
        found[index] = record
        queries.extend(metric['query_index_in_arm'] for metric in record['metrics'])
    if queries != list(range(len(queries))):
        raise ValueError('Saved query sequence is not a consecutive prefix')
    return found


def append_episode(path, record):
    with path.open('ab') as stream:
        stream.write((json.dumps(record, allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


def run_worker(args):
    started = time.monotonic()
    import runner
    runner.guard()
    from vq_adapter import VQRuntime
    p = read_json(args.protocol)
    if p.get('protocol') != PROTOCOL or tuple(p.get('arms', [])) != ARMS:
        raise ValueError('Unexpected frozen VQ protocol')
    arm, indices = worker_arm(args.worker_id), worker_indices(args.worker_id)
    bank = p['sources'][arm]['bank']
    _, rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / 'protocol.json'
    if frozen.exists() and read_json(frozen) != p:
        raise ValueError('Cannot resume with changed VQ configuration')
    save(frozen, p)
    path = args.out / 'episodes.jsonl'
    completed = recover_records(path, rows, args.worker_id, bank)
    runtime, metadata, soft_stop = None, None, False

    def receipt(status, **extra):
        return {'protocol': PROTOCOL, 'worker_id': args.worker_id, 'arm': arm,
                'status': status, 'complete': status == 'complete', 'episodes': len(completed),
                'completed': len(completed), 'expected': 140, 'soft_deadline_reached': soft_stop,
                'install_metadata': metadata, 'setup_receipt_file': 'setup_receipt.json', **extra}

    try:
        save(args.out / 'progress.json', receipt('running', stage='model_loading'))
        if len(completed) != 140:
            runtime = VQRuntime(args.out, p, arm)
            metadata = runtime.install_metadata
            setup = runtime.validate_reference()
            save(args.out / 'setup_receipt.json', setup)
            runtime.arm_calls[arm] = sum(len(row['metrics']) for row in completed.values())
            runtime.calls = runtime.arm_calls[arm]
            for index in indices:
                if index in completed:
                    continue
                if time.monotonic() - started >= p.get('soft_deadline_seconds', 85500):
                    soft_stop = True
                    break
                save(args.out / 'progress.json', receipt('running', stage='episode', index=index))
                record = runtime.episode(rows[index], index)
                record['worker_id'] = args.worker_id
                validate_record(record, rows, args.worker_id, bank)
                append_episode(path, record)
                completed[index] = record
                save(args.out / 'progress.json', receipt('running', stage='episode_saved', index=index))
                print(f'SLOT_COMPLETE worker={args.worker_id} arm={arm} index={index} success={record["success"]}', flush=True)
        else:
            metadata = read_json(args.out / 'setup_receipt.json')['install_metadata']
        status = 'checkpointed' if soft_stop else 'complete'
        if not soft_stop and set(completed) != set(indices):
            raise RuntimeError('Worker completed coverage differs from its frozen shard')
        result = receipt(status, indices=sorted(completed), arm_episodes={arm: len(completed)})
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
