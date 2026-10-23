"""Aggregate the single C80 per-channel+clipping arm against frozen references."""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from evaluate import (
    ARMS, PROTOCOL, TIMING_FIELDS, load_manifest, read_json, recover_records,
    save, validate_install_metadata, validate_protocol, worker_indices,
)

REFERENCE_C80_ARMS = ('clip_only', 'joint')
MAX_SUMMARY_BYTES = 256 * 1024


def _stats(values, unit):
    values = sorted(float(value) for value in values)
    if not values:
        return {'n': 0, f'mean_{unit}': None, f'median_{unit}': None,
                f'p95_{unit}': None, f'max_{unit}': None}
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Summary values must be finite and non-negative')
    return {'n': len(values), f'mean_{unit}': statistics.mean(values),
            f'median_{unit}': statistics.median(values),
            f'p95_{unit}': values[max(0, math.ceil(.95 * len(values)) - 1)],
            f'max_{unit}': values[-1]}


def _rate(records, indices):
    wins = sum(records[i]['success'] for i in indices)
    return {'successes': wins, 'episodes': len(indices), 'success_rate': wins / len(indices)}


def _paired(candidate, reference, indices):
    wins = sum(candidate[i]['success'] and not reference[i]['success'] for i in indices)
    losses = sum(reference[i]['success'] and not candidate[i]['success'] for i in indices)
    return {'wins': wins, 'losses': losses, 'ties': len(indices) - wins - losses,
            'episodes': len(indices), 'success_rate_difference': (wins - losses) / len(indices)}


def summarize_timing_by_implementation(records):
    groups = collections.defaultdict(list)
    for record in records.values():
        groups[record['transform_implementation']].append(record)
    result = {}
    for implementation, episodes in sorted(groups.items()):
        samples = {kind: {field: [] for field in TIMING_FIELDS}
                   for kind in ('first_query_in_run', 'subsequent_queries')}
        for record in episodes:
            for metric in record['metrics']:
                kind = 'first_query_in_run' if metric['first_query_in_run'] else 'subsequent_queries'
                for field in TIMING_FIELDS:
                    samples[kind][field].append(metric[field])
        result[implementation] = {
            'episodes': len(episodes),
            'pbs_jobs': sorted({record['pbs_job_id'] for record in episodes}),
            'query_timing': {kind: {
                'queries': len(fields['full_query_wall_ms']),
                'continuous_denoising_gpu_interval_ms': _stats(fields['denoising_latency_ms'], 'ms'),
                'full_action_query_wall_ms': _stats(fields['full_query_wall_ms'], 'ms'),
                'memory': {f: _stats(fields[f], 'GB') for f in TIMING_FIELDS if f.endswith('_GB')},
            } for kind, fields in samples.items()},
            'episode_wall_including_environment_ms': _stats(
                [record['episode_seconds'] * 1000 for record in episodes], 'ms'),
        }
    return {'by_implementation': result}


def _validate_source_and_training(protocol, arm):
    from vq_adapter import validate_training_artifacts
    return validate_training_artifacts(
        protocol, arm, Path(protocol['sources']['training_results']))


def _validate_setup(receipt, protocol, reference_action, arm):
    tolerance = protocol['closed_loop_reference']['max_abs_tolerance']
    finite = lambda v: (not isinstance(v, bool) and isinstance(v, (int, float))
                        and math.isfinite(v))
    max_abs = receipt.get('max_abs_vs_reference')
    fields = ('uncached_cached_action_max_abs', 'uncached_cached_future_max_abs',
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
            or not finite(max_abs) or max_abs < 0 or max_abs > tolerance
            or any(not finite(receipt.get(key)) or receipt[key] < 0 or receipt[key] > tolerance
                   for key in fields)):
        raise ValueError('Source endpoint or live action/future setup gate failed')
    return validate_install_metadata(receipt.get('install_metadata'), protocol, arm)


def load_workers(results, protocol, rows, training, arm):
    records, setup, checkpointed, metadata = {}, {}, [], None
    for worker in range(4):
        out = results / 'evaluations' / arm / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        keys = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                'soft_deadline_reached', 'indices', 'completed_indices', 'arm_episodes',
                'install_metadata')
        if any(progress.get(k) != summary.get(k) for k in keys):
            raise ValueError(f'Worker {worker} progress and summary disagree')
        indices = worker_indices(worker, arm)
        if (summary.get('protocol') != PROTOCOL or summary.get('worker_id') != worker
                or summary.get('arm') != arm or summary.get('expected') != 70
                or summary.get('indices') != indices):
            raise ValueError(f'Worker {worker} differs from its frozen 70-slot shard')
        found = recover_records(out / 'episodes.jsonl', rows, worker, protocol,
                                summary.get('install_metadata'), arm)
        if (summary.get('episodes') != len(found) or summary.get('completed_indices') != list(found)
                or summary.get('arm_episodes') != {arm: len(found)}):
            raise ValueError(f'Worker {worker} receipt count differs from saved episodes')
        if set(found) == set(indices):
            if summary.get('status') != 'complete' or summary.get('complete') is not True:
                raise ValueError(f'Complete worker {worker} has invalid terminal status')
        elif (summary.get('status') == 'checkpointed' and summary.get('complete') is False
              and summary.get('soft_deadline_reached') is True):
            checkpointed.append(worker)
        else:
            raise ValueError(f'Incomplete worker {worker} did not reach the soft deadline')
        receipt = read_json(out / 'setup_receipt.json')
        current = _validate_setup(receipt, protocol, training['reference_action'], arm)
        installed = validate_install_metadata(summary.get('install_metadata'), protocol, arm)
        if current != installed or (metadata is not None and installed != metadata):
            raise ValueError(f'Worker {worker} install metadata differs from its source receipt')
        metadata = installed
        setup[str(worker)] = receipt
        for index, record in found.items():
            if (arm, index) in records:
                raise ValueError('Duplicate new episode slot across workers')
            records[arm, index] = record
    return records, {arm: metadata}, setup, checkpointed


def _c80_install_metadata(metadata, source_protocol, arm):
    cal, config = source_protocol['calibration'], source_protocol['arm_configs'][arm]
    expected = {
        'protocol': source_protocol['protocol'], 'arm': arm,
        'source_bank': cal['bank_directory'], 'source_protocol': 'fastwam-vq-calibration-suites-v1',
        'source_cohort': 'c80', 'source_case_ids': cal['case_ids'],
        'calibration_ids': cal['case_ids'],
        'initialization_protocol': 'fastwam-vq-c8-action-latent-plus-v1',
        'initialization_cohort': 'c8', 'initialization_arm': 'rho1.5',
        'initialization_updates': 8,
        'initialization_training_job': source_protocol['sources']['baseline_training_job'],
        'module_count': 614, 'execution': 'decoded_bf16_vq', 'activation_bits': 4,
        'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
        'rho': 1.5, 'epochs': 1, 'updates': 80,
        'normalization_c': source_protocol['training']['normalization_c'],
        'transform_implementation': 'batched_h128_v2', 'smooth_alpha': 0.5,
        'hadamard_block': 128, 'scale_block': 32, 'fixed_indices': True,
        'fixed_scale_exponents': True, 'deployed_books_dtype': 'bfloat16',
        'deployed_row_scales_dtype': 'bfloat16',
        'train_weights': config['train_weights'], 'train_clips': True,
        'weights_unchanged': not config['train_weights'], 'native_int4_gemm_calls': 0,
    }
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Existing C80 {arm} install metadata differs from its source protocol')
    ratios = metadata.get('ratios', {})
    if (ratios.get('count') != 614
            or any(isinstance(ratios.get(k), bool)
                   or not isinstance(ratios.get(k), (int, float))
                   or not math.isfinite(ratios[k]) for k in ('min', 'mean', 'max'))
            or ratios['min'] < .5 or ratios['max'] > 1
            or ratios['min'] > ratios['mean'] or ratios['mean'] > ratios['max']):
        raise ValueError(f'Existing C80 {arm} clipping summary is invalid')
    return metadata


def _read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed source episode at {path}:{line_number}') from exc


def _load_c80_results(results_root, rows, protocol):
    root = Path(results_root)
    old = read_json(root.parent / 'protocol.json')
    summary, marker = read_json(root / 'summary.json'), read_json(root / 'PIPELINE_COMPLETE')
    old_arms = list(REFERENCE_C80_ARMS)
    if (old.get('protocol') != protocol['references']['c80_protocol'] or old.get('arms') != old_arms
            or summary.get('protocol') != old['protocol'] or summary.get('status') != 'complete'
            or summary.get('complete') is not True or summary.get('manifest_variants') != 280
            or summary.get('episode_slots') != 560 or summary.get('reference_episode_slots') != 2240
            or summary.get('arm_episodes') != {arm: 280 for arm in old_arms}
            or marker.get('protocol') != old['protocol'] or marker.get('complete') is not True
            or marker.get('episode_slots') != 560
            or marker.get('arm_episodes') != {arm: 280 for arm in old_arms}
            or marker.get('reference_episode_slots') != 2240):
        raise ValueError('Existing C80 root summary or completion marker is invalid')
    found, arm_metadata = {}, {}
    for arm in old_arms:
        arm_root = root / 'evaluations' / arm
        arm_summary, arm_marker = read_json(arm_root / 'summary.json'), read_json(arm_root / 'PIPELINE_COMPLETE')
        if (arm_summary.get('protocol') != old['protocol'] or arm_summary.get('arm') != arm
                or arm_summary.get('status') != 'complete' or arm_summary.get('complete') is not True
                or arm_summary.get('episode_slots') != 280
                or arm_summary.get('reference_episode_slots') != 2240
                or arm_summary.get('arm_episodes') != {arm: 280}
                or arm_marker.get('protocol') != old['protocol'] or arm_marker.get('arm') != arm
                or arm_marker.get('complete') is not True or arm_marker.get('episode_slots') != 280
                or arm_marker.get('reference_episode_slots') != 2240):
            raise ValueError(f'Existing C80 {arm} summary or completion marker is invalid')
        for worker in range(4):
            indices = list(range(worker, 280, 4))
            out = arm_root / f'worker_{worker}'
            progress, receipt = read_json(out / 'progress.json'), read_json(out / 'summary.json')
            fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                      'indices', 'completed_indices', 'arm_episodes', 'install_metadata')
            if (any(progress.get(k) != receipt.get(k) for k in fields)
                    or receipt.get('protocol') != old['protocol'] or receipt.get('worker_id') != worker
                    or receipt.get('arm') != arm or receipt.get('status') != 'complete'
                    or receipt.get('complete') is not True or receipt.get('episodes') != 70
                    or receipt.get('expected') != 70 or receipt.get('indices') != indices
                    or receipt.get('completed_indices') != indices
                    or receipt.get('arm_episodes') != {arm: 70}):
                raise ValueError(f'Existing C80 {arm} worker {worker} is incomplete or misassigned')
            metadata = _c80_install_metadata(receipt.get('install_metadata'), old, arm)
            if arm in arm_metadata and arm_metadata[arm] != metadata:
                raise ValueError(f'Existing C80 {arm} workers used different install metadata')
            arm_metadata[arm] = metadata
            setup = read_json(out / 'setup_receipt.json')
            max_abs = setup.get('max_abs_vs_reference')
            if (setup.get('passed') is not True or setup.get('protocol') != old['protocol']
                    or setup.get('arm') != arm or setup.get('case_id') != 2000
                    or setup.get('seed_index') != 0 or setup.get('observed_modules') != 614
                    or isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
                    or not math.isfinite(max_abs) or max_abs < 0
                    or max_abs > old['closed_loop_reference']['max_abs_tolerance']
                    or setup.get('install_metadata') != metadata):
                raise ValueError(f'Existing C80 {arm} setup receipt is invalid')
            query_indices, seen_runs, worker_slots = [], set(), set()
            for line_number, record in _read_jsonl(out / 'episodes.jsonl'):
                index = record.get('index')
                row = rows[index] if type(index) is int and 0 <= index < 280 else None
                if (row is None or index not in indices or record.get('protocol') != old['protocol']
                        or record.get('worker_id') != worker or record.get('arm') != arm
                        or record.get('complete') is not True or type(record.get('success')) is not bool
                        or str(record.get('variant_id')) != str(row['variant_id'])
                        or record.get('suite') != row['suite'] or record.get('dimension') != row['dimension']
                        or record.get('task_id') != int(row['task_id']) or record.get('state_id') != 0
                        or record.get('env_seed') != 800_000_000 + index * 1000
                        or record.get('sampler_seed') != 900_000_000 + index * 1000
                        or record.get('transform_implementation') != 'batched_h128_v2'
                        or not record.get('pbs_job_id')):
                    raise ValueError(f'Existing C80 {arm} episode identity differs at line {line_number}')
                evidence = record.get('vq_evidence', {})
                if (any(evidence.get(k) != v for k, v in metadata.items())
                        or evidence.get('observed_module_count') != 614
                        or evidence.get('native_int4_gemm_calls') != 0
                        or evidence.get('kv_packed_prefills') != 0 or evidence.get('kv_layer_reads') != 0
                        or type(evidence.get('quantized_linear_calls')) is not int
                        or evidence['quantized_linear_calls'] <= 0):
                    raise ValueError(f'Existing C80 {arm} VQ evidence differs at {worker}/{index}')
                metrics, replans = record.get('metrics'), record.get('replans')
                seconds = record.get('episode_seconds')
                if (type(replans) is not int or replans <= 0 or not isinstance(metrics, list)
                        or len(metrics) != replans or isinstance(seconds, bool)
                        or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
                    raise ValueError(f'Existing C80 {arm} timing is invalid at {worker}/{index}')
                for replan, metric in enumerate(metrics):
                    query, job = metric.get('query_index_in_arm'), metric.get('pbs_job_id')
                    if (type(query) is not int or query < 0
                            or metric.get('first_query_in_arm') is not (query == 0)
                            or metric.get('replan') != replan
                            or metric.get('sampler_seed') != 900_000_000 + index * 1000 + replan
                            or metric.get('transform_implementation') != 'batched_h128_v2'
                            or job != record['pbs_job_id']
                            or metric.get('first_query_in_run') is not (job not in seen_runs)):
                        raise ValueError(f'Existing C80 {arm} query identity differs at {worker}/{index}')
                    seen_runs.add(job)
                    query_indices.append(query)
                    for key in TIMING_FIELDS:
                        value = metric.get(key)
                        if (isinstance(value, bool) or not isinstance(value, (int, float))
                                or not math.isfinite(value) or value < 0):
                            raise ValueError(f'Existing C80 {arm} timing field {key} is invalid')
                if index in worker_slots or (arm, index) in found:
                    raise ValueError('Duplicate existing C80 episode slot')
                worker_slots.add(index)
                found[arm, index] = record
            if worker_slots != set(indices) or query_indices != list(range(len(query_indices))):
                raise ValueError(f'Existing C80 {arm} worker {worker} is missing records or has a query gap')
    expected = {(arm, i) for arm in old_arms for i in range(280)}
    if set(found) != expected:
        raise ValueError(f'Expected exactly 560 existing C80 episodes, got {len(found)}')
    return found


def validate_reference_slots(references, reference_arms):
    expected = {(arm, i) for arm in reference_arms for i in range(280)}
    if len(references) != 2800 or set(references) != expected:
        raise ValueError(f'Expected exactly 2800 slots across ten frozen references, got {len(references)}')
    return references


def _load_references(protocol, manifest, rows):
    import reference_aggregate as scalar
    import reference_loaders as trained

    rotation_root = Path(protocol['references']['rotation_results'])
    old_manifest, _rows, shards = scalar.load_manifest(rotation_root / 'manifest.json')
    if manifest != old_manifest:
        raise ValueError('Manifest differs from the frozen original cohort')
    references = dict(scalar.load_reference_episodes(rotation_root, rows, shards))
    scalar_records, _metadata, _setup = scalar.load_scalar_episodes(
        Path(protocol['references']['scalar_results']), rows)
    references.update({(scalar.ARM, index): record for index, record in scalar_records.items()})
    references.update(trained._load_old_vq(Path(protocol['references']['vq_results']), rows))
    references.update(trained._load_c8_trained(
        Path(protocol['references']['c8_trained_results']), rows))
    references.update(_load_c80_results(Path(protocol['references']['c80_results']), rows, protocol))
    arms = protocol['references']['arms']
    return validate_reference_slots(references, arms), arms


def _compact_training_receipt(training, protocol):
    receipt = training.get('training_receipt', {})
    result = {key: receipt[key] for key in (
        'protocol', 'status', 'complete', 'updates', 'epochs', 'target_linears',
        'calibration_ids', 'normalization_c', 'rho', 'source_job_id') if key in receipt}
    result.update(source_training_job=protocol['sources']['training_job'],
                  training_directory=str(protocol['sources']['training_results']))
    return result


def _make_summary(protocol, rows, records, by_arm, metadata, setup, training, checkpointed, status):
    arm = ARMS[0]
    complete = status == 'complete'
    summary = {
        'protocol': PROTOCOL, 'arm': arm, 'status': status, 'complete': complete,
        'manifest_variants': 280, 'episode_slots': len(records),
        'reference_episode_slots': 2800, 'new_arms': [arm],
        'reference_arms': list(protocol['references']['arms']),
        'arm_episodes': {arm: len(records)}, 'checkpointed_workers': checkpointed,
        'install_metadata': metadata, 'training_receipt': _compact_training_receipt(training, protocol),
        'setup_receipts': setup,
        'source_receipts': {
            'training_protocol': protocol['sources']['training_protocol'],
            'training_job': protocol['sources']['training_job'],
            'training_directory': protocol['sources']['training_results'],
            'reference_endpoint': protocol['sources']['reference_endpoint'],
            'reference_action_parity': training['reference_action']['actual_infer_parity'],
            'c80_results': protocol['references']['c80_results'],
            'c80_protocol': protocol['references']['c80_protocol'],
        },
        'claim_scope': protocol['claim_scope'],
    }
    if complete:
        if set(records) != {(arm, i) for i in range(280)}:
            raise ValueError('Complete aggregate has missing per-channel episode slots')

        def bucket(indices):
            return {
                'arms': {name: _rate(values, indices) for name, values in by_arm.items()},
                'paired': {arm: {other: _paired(by_arm[arm], by_arm[other], indices)
                                 for other in by_arm if other != arm}},
            }

        suites = sorted({row['suite'] for row in rows})
        dimensions = sorted({row['dimension'] for row in rows})
        summary['overall'] = bucket(list(range(280)))
        summary['suites'] = {s: bucket([i for i, row in enumerate(rows) if row['suite'] == s])
                             for s in suites}
        summary['dimensions'] = {d: bucket([i for i, row in enumerate(rows) if row['dimension'] == d])
                                 for d in dimensions}
        summary['cells'] = {f'{s}::{d}': bucket([
            i for i, row in enumerate(rows) if row['suite'] == s and row['dimension'] == d])
            for s in suites for d in dimensions}
        summary['timing'] = {arm: summarize_timing_by_implementation({
            i: by_arm[arm][i] for i in range(280)})}
        summary['timing_boundary'] = (
            'Query timing separates each worker process first query from later queries; '
            'paired episodes can have different query counts.')
    return summary


def _write_report(directory, summary):
    lines = [
        '# C80 per-channel + clipping：LIBERO-Plus 闭环评测', '',
        f"状态：{summary['status']}；新增 episode：{summary['episode_slots']}/280；复用 frozen references：2800/2800。", '',
        '本轮直接复用源目录中完成的 C80 update 80 deployment，未重新训练。评测采用 decoded BF16 VQ、per-token A4 activations、BF16 KV cache。',
        '', 'Setup gate 检查源 endpoint 的 7 项 scalar metrics，并验证 uncached/cached source action/future parity 和实际 infer action parity。源结果没有保存最终 action vector，因此这不是 saved-action replay。',
    ]
    if summary['complete']:
        lines += ['', '| Arm | Successes / episodes | Success rate |', '|---|---:|---:|']
        for name, value in summary['overall']['arms'].items():
            lines.append(f"| {name} | {value['successes']}/{value['episodes']} | {value['success_rate']:.2%} |")
        lines += ['', 'Paired win/loss/tie 及 suite、dimension、suite×dimension 统计见 summary.json。',
                  'Timing 按 batched_h128_v2 单独报告；它是 decoded-BF16 实现证据，不代表 native low-bit kernel latency。']
    else:
        lines += ['', '当前结果为 allocation soft deadline 前缀；已完成 episode 保存在 worker JSONL，可从该前缀继续。']
    atomic_text(Path(directory) / 'RESULTS.zh.md', '\n'.join(lines) + '\n')


def atomic_text(path, content):
    import os
    import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(content, encoding='utf-8')
    temp.replace(path)


def save_compact_summary(path, value):
    import os
    import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if len(payload) > MAX_SUMMARY_BYTES:
        raise ValueError(f'Aggregate summary is {len(payload)} bytes; limit is {MAX_SUMMARY_BYTES}')
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_bytes(payload)
    temp.replace(path)
    return len(payload)


def run(args):
    import runner
    runner.guard()
    protocol = validate_protocol(read_json(args.protocol))
    arm = args.arm
    if arm not in ARMS:
        raise ValueError(f'--arm must be {ARMS[0]}')
    manifest, rows = load_manifest(args.manifest)
    training = _validate_source_and_training(protocol, arm)
    records, metadata, setup, checkpointed = load_workers(
        args.results, protocol, rows, training, arm)
    references, reference_arms = _load_references(protocol, manifest, rows)
    if tuple(reference_arms) != tuple(protocol['references']['arms']):
        raise ValueError('Reference arm order differs from the frozen protocol')
    root = args.results / 'evaluations' / arm
    complete = len(records) == 280 and not checkpointed
    by_arm = {name: {i: references[name, i] for i in range(280)} for name in reference_arms}
    by_arm[arm] = {i: record for (name, i), record in records.items() if name == arm}
    summary = _make_summary(protocol, rows, records, by_arm, metadata, setup, training,
                            checkpointed, 'complete' if complete else 'checkpointed')
    save_compact_summary(root / 'summary.json', summary)
    _write_report(root, summary)
    marker = root / 'PIPELINE_COMPLETE'
    checkpoint = root / 'PIPELINE_CHECKPOINTED'
    if complete:
        save(marker, {'protocol': PROTOCOL, 'arm': arm, 'complete': True,
                      'episode_slots': 280, 'reference_episode_slots': 2800})
        checkpoint.unlink(missing_ok=True)
    else:
        marker.unlink(missing_ok=True)
        save(checkpoint, {'protocol': PROTOCOL, 'arm': arm, 'complete': False,
                          'episode_slots': len(records), 'reference_episode_slots': 2800})
    print(f"C80_PER_CHANNEL_AGGREGATE status={summary['status']} slots={len(records)}/280 reference_slots=2800",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
