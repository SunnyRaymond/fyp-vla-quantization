"""Aggregate two diffusion arms against the frozen eight-arm LIBERO cohort."""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from evaluate import (
    ARMS, MAX_BPW_GAP, PROTOCOL, TIMING_FIELDS, bank_manifest_job_id, load_manifest, read_json,
    recover_records, save, validate_install_metadata, validate_protocol,
    validate_setup_receipt, worker_arm, worker_indices,
)

MAX_SUMMARY_BYTES = 256 * 1024
REFERENCE_SLOTS = 2240


def _stats(values, unit):
    values = sorted(float(value) for value in values)
    if not values:
        return {'n': 0, f'mean_{unit}': None, f'median_{unit}': None,
                f'p95_{unit}': None, f'max_{unit}': None}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Summary values must be finite and non-negative')
    return {
        'n': len(values), f'mean_{unit}': statistics.mean(values),
        f'median_{unit}': statistics.median(values),
        f'p95_{unit}': values[max(0, math.ceil(.95 * len(values)) - 1)],
        f'max_{unit}': values[-1],
    }


def _rate(records, indices):
    available = [records[index] for index in indices if index in records]
    successes = sum(record['success'] for record in available)
    count = len(available)
    return {'successes': successes, 'episodes': count,
            'success_rate': successes / count if count else None}


def _paired(candidate, reference, indices):
    paired = [(candidate[index]['success'], reference[index]['success']) for index in indices
              if index in candidate and index in reference]
    wins = sum(left and not right for left, right in paired)
    losses = sum(right and not left for left, right in paired)
    count = len(paired)
    return {
        'wins': wins, 'losses': losses, 'ties': count - wins - losses,
        'episodes': count,
        'success_rate_difference': (wins - losses) / count if count else None,
    }


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
                'memory': {field: _stats(fields[field], 'GB') for field in TIMING_FIELDS
                           if field.endswith('_GB')},
            } for kind, fields in samples.items()},
            'episode_wall_including_environment_ms': _stats(
                [record['episode_seconds'] * 1000 for record in episodes], 'ms'),
        }
    return {'by_implementation': result}


def load_workers(results_root, protocol, manifest, rows):
    found, by_arm, setups, metadata_by_arm, checkpointed = {}, {}, {}, {}, []
    worker_metadata = {}
    arm_implementations = {}
    bank_job_ids, bpw_by_arm, declared_bpw_gaps = set(), {}, []
    root = Path(results_root) / 'workers'
    for worker in range(4):
        arm = worker_arm(worker)
        out = root / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        receipt_fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes',
                          'completed', 'expected', 'soft_deadline_reached', 'indices',
                          'completed_indices', 'arm_episodes', 'install_metadata')
        if any(progress.get(key) != summary.get(key) for key in receipt_fields):
            raise ValueError(f'Worker {worker} progress and summary disagree')
        indices = worker_indices(worker, arm)
        if (summary.get('protocol') != PROTOCOL or summary.get('worker_id') != worker
                or summary.get('arm') != arm or summary.get('expected') != 140
                or summary.get('indices') != indices):
            raise ValueError(f'Worker {worker} differs from its frozen 140-slot shard')
        metadata = validate_install_metadata(summary.get('install_metadata'), protocol, arm)
        setup = read_json(out / 'setup_receipt.json')
        validate_setup_receipt(setup, protocol, worker, arm, metadata)
        if setup['pbs_job_id'] != summary.get('pbs_job_id', setup['pbs_job_id']):
            raise ValueError(f'Worker {worker} setup/PBS identity differs')
        if arm in metadata_by_arm and metadata_by_arm[arm] != metadata:
            raise ValueError(f'Workers for {arm} produced different install metadata')
        metadata_by_arm[arm] = metadata
        worker_metadata[str(worker)] = metadata
        bank_job_ids.add(bank_manifest_job_id(metadata))
        declared_bpw_gaps.append(setup['bank_manifest'].get('relative_bpw_gap'))
        bpw_by_arm[arm] = metadata['encoded_target_bpw']
        arm_implementations[arm] = metadata['transform_implementation']

        episode_path = out / 'episodes.jsonl'
        records = recover_records(episode_path, rows, worker, protocol, metadata, arm)
        if (summary.get('episodes') != len(records)
                or summary.get('completed') != len(records)
                or summary.get('completed_indices') != list(records)
                or summary.get('arm_episodes') != {arm: len(records)}):
            raise ValueError(f'Worker {worker} receipt count differs from saved episodes')
        if len(records) == len(indices):
            if summary.get('status') != 'complete' or summary.get('complete') is not True:
                raise ValueError(f'Full worker {worker} has invalid terminal status')
        elif (summary.get('status') == 'checkpointed' and summary.get('complete') is False
              and summary.get('soft_deadline_reached') is True):
            checkpointed.append(worker)
        else:
            raise ValueError(f'Incomplete worker {worker} did not reach the soft deadline')
        setups[str(worker)] = setup
        arm_records = by_arm.setdefault(arm, {})
        for index, record in records.items():
            if index in arm_records:
                raise ValueError('Duplicate new episode slot across workers')
            arm_records[index] = record
            found[arm, index] = record

    if len(bank_job_ids) != 1:
        raise ValueError('Workers do not share one calibration bank manifest PBS job id')
    bpw_gap = abs(bpw_by_arm[ARMS[0]] - bpw_by_arm[ARMS[1]]) / bpw_by_arm[ARMS[1]]
    if bpw_gap > MAX_BPW_GAP:
        raise ValueError(f'Encoded target bpw gap {bpw_gap:.6f} exceeds 0.05')
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or abs(value - bpw_gap) > 1e-6
           for value in declared_bpw_gaps):
        raise ValueError('Bank manifest relative bpw gap differs from installed target metadata')
    if metadata_by_arm[ARMS[0]]['target_weight_parameters'] != metadata_by_arm[ARMS[1]]['target_weight_parameters']:
        raise ValueError('Diffusion arms do not cover the same target weight parameters')
    for arm in ARMS:
        expected_indices = {index for worker in range(4) if worker_arm(worker) == arm
                            for index in worker_indices(worker)}
        if set(by_arm.get(arm, {})) - expected_indices:
            raise ValueError(f'{arm} has slots outside its frozen arm coverage')
    return found, by_arm, metadata_by_arm, setups, checkpointed, {
        'bank_manifest_pbs_jobidstr': next(iter(bank_job_ids)),
        'encoded_target_bpw_relative_gap': bpw_gap,
        'transform_implementations': arm_implementations,
    }


def validate_reference_slots(references, reference_arms):
    expected = {(arm, index) for arm in reference_arms for index in range(280)}
    if len(reference_arms) != 8 or len(references) != REFERENCE_SLOTS or set(references) != expected:
        raise ValueError(f'Expected exactly {REFERENCE_SLOTS} slots across eight frozen references')
    if any(type(record.get('success')) is not bool for record in references.values()):
        raise ValueError('Frozen reference record is missing boolean success')
    return references


def _load_references(protocol, manifest, rows):
    import reference_aggregate as scalar
    import reference_loaders as trained

    rotation_root = Path(protocol['references']['rotation_results'])
    old_manifest, _old_rows, shards = scalar.load_manifest(rotation_root / 'manifest.json')
    if manifest != old_manifest:
        raise ValueError('Manifest differs from the original frozen 280-variant cohort')
    references = dict(scalar.load_reference_episodes(rotation_root, rows, shards))
    scalar_records, _metadata, _setup = scalar.load_scalar_episodes(
        Path(protocol['references']['scalar_results']), rows)
    references.update({(scalar.ARM, index): record for index, record in scalar_records.items()})
    references.update(trained._load_old_vq(Path(protocol['references']['vq_results']), rows))
    references.update(trained._load_c8_trained(
        Path(protocol['references']['c8_trained_results']), rows))
    arms = protocol['references']['arms']
    return validate_reference_slots(references, arms), arms


def _compact_scalars(value, keywords):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        name = str(key).lower()
        if not any(word in name for word in keywords):
            continue
        if item is None or isinstance(item, (str, bool, int, float)):
            result[key] = item
        elif isinstance(item, dict) and len(item) <= 12 and all(
                child is None or isinstance(child, (str, bool, int, float))
                for child in item.values()):
            result[key] = item
    return result


def _make_summary(protocol, rows, records, by_arm, metadata, setups,
                  checkpointed_workers, status, cross_worker_summary):
    complete = status == 'complete'
    reference_arms = list(protocol['references']['arms'])
    arm_episodes = {arm: len(by_arm.get(arm, {})) for arm in ARMS}
    reference_slots = sum(len(by_arm.get(arm, {})) for arm in reference_arms)
    summary = {
        'protocol': PROTOCOL, 'status': status, 'complete': complete,
        'manifest_variants': 280, 'episode_slots': len(records),
        'reference_episode_slots': reference_slots,
        'new_arms': list(ARMS), 'reference_arms': reference_arms,
        'arm_episodes': arm_episodes, 'arm_counts': arm_episodes,
        'checkpointed_workers': checkpointed_workers,
        'cross_worker_checks': cross_worker_summary,
        'source_bank_budget_startup': {
            'bank_manifest_by_worker': {
                worker: _compact_scalars(setup.get('bank_manifest'),
                                         ('protocol', 'pbs_job', 'passed', 'module', 'context',
                                          'byte', 'count', 'bpw', 'rank'))
                for worker, setup in setups.items()},
            'startup_by_worker': {
                worker: {
                    'pbs_job_id': setup['pbs_job_id'], 'passed': setup['passed'],
                    'observed_modules': setup['observed_modules'],
                    'bf16_teacher_max_abs': setup['bf16_teacher_max_abs'],
                    'cross_worker_action_max_abs': setup['cross_worker_action_max_abs'],
                    'component_selfcheck': _compact_scalars(
                        setup['component_selfcheck'], ('passed', 'max_abs', 'count', 'modules')),
                } for worker, setup in setups.items()},
            'quantization_budget_by_arm': {
                arm: {key: metadata[arm][key] for key in (
                    'weight_bits', 'activation_bits', 'group_size', 'execution',
                    'kv_cache_dtype', 'native_low_bit_kernel', 'encoded_target_bpw',
                    'encoded_target_bytes', 'target_weight_parameters', 'low_rank_global_rank',
                    'representation', 'transform_implementation')}
                | {key: metadata[arm][key] for key in (
                    'encoded_bytes_by_kind', 'whole_model_parameter_bpw')
                    if key in metadata[arm]}
                for arm in ARMS},
        },
    }
    if complete and (len(records) != 560 or arm_episodes != {arm: 280 for arm in ARMS}
                     or reference_slots != REFERENCE_SLOTS):
        raise ValueError('Complete aggregate requires exactly 560 new and 2240 reference slots')

    all_names = list(ARMS) + reference_arms

    def bucket(indices):
        return {
            'arms': {name: _rate(by_arm.get(name, {}), indices) for name in all_names},
            'paired': {
                arm: {other: _paired(by_arm.get(arm, {}), by_arm.get(other, {}), indices)
                      for other in all_names if other != arm}
                for arm in ARMS},
        }

    suites = sorted({row['suite'] for row in rows})
    dimensions = sorted({row['dimension'] for row in rows})
    summary['overall'] = bucket(list(range(280)))
    summary['suites'] = {
        suite: bucket([index for index, row in enumerate(rows) if row['suite'] == suite])
        for suite in suites}
    summary['dimensions'] = {
        dimension: bucket([index for index, row in enumerate(rows) if row['dimension'] == dimension])
        for dimension in dimensions}
    summary['cells'] = {
        f'{suite}::{dimension}': bucket([
            index for index, row in enumerate(rows)
            if row['suite'] == suite and row['dimension'] == dimension])
        for suite in suites for dimension in dimensions}
    summary['timing'] = {
        arm: summarize_timing_by_implementation(by_arm.get(arm, {})) for arm in ARMS}
    summary['timing_boundary'] = (
        'Query timing separates each worker process first query from later queries; '
        'paired episodes can have different query counts.')
    return summary


def _write_report(directory, summary):
    lines = [
        '# Diffusion W4A4 Baselines：LIBERO-Plus 闭环评测', '',
        f"状态：{summary['status']}；新增 episode：{summary['episode_slots']}/560；"
        f"frozen references：{summary['reference_episode_slots']}/2240。", '',
        '本轮比较 SVDQuant adapted W4A4 与 ViDiT-Q adapted W4A4。权重和激活以 decoded BF16 形式模拟，KV cache 为 BF16；SVDQuant 低秩 residual 按预算保留 BF16 factors。',
        '报告中的 timing 按每个 arm 的 transform implementation 分开统计，包含完整 action query 和 GPU denoising interval；不代表 native low-bit kernel 的速度或显存。',
    ]
    if summary['complete']:
        lines += ['', '| Arm | Successes / episodes | Success rate |', '|---|---:|---:|']
        for arm, value in summary['overall']['arms'].items():
            if arm in ARMS:
                rate = 'n/a' if value['success_rate'] is None else f"{value['success_rate']:.2%}"
                lines.append(f"| {arm} | {value['successes']}/{value['episodes']} | {rate} |")
        lines += ['', '与 8 个 frozen reference arms 及两个新 arm 之间的 paired wins/losses/ties，'
                  '以及 suite、dimension、suite×dimension 明细见 summary.json。']
    else:
        lines += ['', '当前为 soft deadline 前缀结果；已完成 episode 可从各 worker 的连续前缀恢复。']
    atomic_text(Path(directory) / 'RESULTS.zh.md', '\n'.join(lines) + '\n')


def atomic_text(path, content):
    import os
    import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(directory):
    import os
    if os.name == 'nt':
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_compact_summary(path, value):
    import os
    import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
               + '\n').encode('utf-8')
    if len(payload) > MAX_SUMMARY_BYTES:
        raise ValueError(f'Aggregate summary is {len(payload)} bytes; limit is {MAX_SUMMARY_BYTES}')
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    _fsync_dir(path.parent)
    return len(payload)


def run(args):
    protocol = validate_protocol(read_json(args.protocol))
    manifest, rows = load_manifest(args.manifest)
    results = Path(args.results)
    references, reference_arms = _load_references(protocol, manifest, rows)
    references = validate_reference_slots(references, reference_arms)
    records, new_by_arm, metadata, setups, checkpointed, cross_worker = load_workers(
        results, protocol, manifest, rows)
    if list(reference_arms) != protocol['references']['arms']:
        raise ValueError('Reference loader arms differ from the frozen protocol')
    by_arm = {arm: {index: record for (name, index), record in references.items() if name == arm}
              for arm in reference_arms}
    by_arm.update(new_by_arm)
    new_complete = all(len(new_by_arm.get(arm, {})) == 280 for arm in ARMS)
    status = 'complete' if len(records) == 560 and new_complete else 'checkpointed'
    summary = _make_summary(protocol, rows, records, by_arm, metadata, setups,
                            checkpointed, status, cross_worker)
    save_compact_summary(results / 'summary.json', summary)
    _write_report(results, summary)
    marker = results / 'PIPELINE_COMPLETE'
    checkpoint = results / 'PIPELINE_CHECKPOINTED'
    marker_value = {
        'protocol': PROTOCOL, 'complete': summary['complete'],
        'episode_slots': summary['episode_slots'],
        'reference_episode_slots': summary['reference_episode_slots'],
        'arm_episodes': summary['arm_episodes'], 'arm_counts': summary['arm_counts'],
    }
    if summary['complete']:
        save(marker, marker_value)
        if checkpoint.exists():
            checkpoint.unlink()
    else:
        save(checkpoint, marker_value)
        if marker.exists():
            marker.unlink()
    print(f"DIFFUSION_BASELINES_AGGREGATE status={summary['status']} "
          f"slots={summary['episode_slots']}/560 reference_slots=2240", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    import runner
    runner.guard()
    run(args)


if __name__ == '__main__':
    main()
