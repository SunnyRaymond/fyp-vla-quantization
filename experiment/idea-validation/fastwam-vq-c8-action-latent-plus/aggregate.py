"""Validate trained workers and pair their closed-loop outcomes with existing references."""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from evaluate import (ARMS, PROTOCOL, TIMING_FIELDS, load_manifest, read_json, recover_records,
                      save, validate_install_metadata, validate_protocol,
                      worker_arm, worker_indices)

REFERENCE_ARMS = ('bf16', 'quarot_adapted_w4a4', 'spinquant_adapted_w4a4',
                  'smooth05_hadamard_scalar_w4a4', 'row_scale_vq_a4', 'micro_scale_vq_a4')
OLD_VQ_ARMS = ('row_scale_vq_a4', 'micro_scale_vq_a4')


def _stats(values, unit):
    values = sorted(float(value) for value in values)
    if not values:
        return {'n': 0, f'mean_{unit}': None, f'median_{unit}': None,
                f'p95_{unit}': None, f'max_{unit}': None}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Summary values must be finite and non-negative')
    p95 = max(0, math.ceil(0.95 * len(values)) - 1)
    return {'n': len(values), f'mean_{unit}': statistics.mean(values),
            f'median_{unit}': statistics.median(values), f'p95_{unit}': values[p95],
            f'max_{unit}': values[-1]}


def _rate(records, indices):
    successes = sum(records[index]['success'] for index in indices)
    return {'successes': successes, 'episodes': len(indices), 'success_rate': successes / len(indices)}


def _paired(candidate, reference, indices):
    wins = sum(candidate[index]['success'] and not reference[index]['success'] for index in indices)
    losses = sum(reference[index]['success'] and not candidate[index]['success'] for index in indices)
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
            'worker_process_first_queries': sum(len(fields['full_query_wall_ms'])
                                                for kind, fields in samples.items()
                                                if kind == 'first_query_in_run'),
            'query_timing': {
                kind: {
                    'queries': len(fields['full_query_wall_ms']),
                    'continuous_denoising_gpu_interval_ms': _stats(fields['denoising_latency_ms'], 'ms'),
                    'full_action_query_wall_ms': _stats(fields['full_query_wall_ms'], 'ms'),
                    'memory': {field: _stats(fields[field], 'GB')
                               for field in TIMING_FIELDS if field.endswith('_GB')},
                } for kind, fields in samples.items()},
            'episode_wall_including_environment_ms': _stats(
                [record['episode_seconds'] * 1000 for record in episodes], 'ms'),
        }
    return {'by_implementation': result}


def _validate_source_and_training(protocol, results):
    from vq_adapter import validate_training_artifacts

    reference_dir = Path(protocol['reference_directory'])
    training = {}
    for arm in ARMS:
        training[arm] = validate_training_artifacts(
            protocol, arm, results / 'training' / arm, reference_dir)
    values = [training[arm]['source']['normalization_c'] for arm in ARMS]
    if not math.isclose(values[0], values[1], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('The two rho arms do not share the same C8 normalization_c')
    return training


def load_workers(results, protocol, rows, training):
    records, metadata, setup, checkpointed = {}, {}, {}, []
    for worker in range(4):
        arm = worker_arm(worker)
        out = results / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                  'soft_deadline_reached', 'indices', 'completed_indices',
                  'arm_episodes', 'install_metadata')
        if any(progress.get(key) != summary.get(key) for key in fields):
            raise ValueError(f'Worker {worker} progress and summary disagree')
        if (summary.get('protocol') != PROTOCOL or summary.get('worker_id') != worker
                or summary.get('arm') != arm or summary.get('expected') != 140):
            raise ValueError(f'Worker {worker} receipt has a different frozen assignment')
        if summary.get('indices') != worker_indices(worker):
            raise ValueError(f'Worker {worker} receipt differs from its exact frozen parity shard')
        found = recover_records(out / 'episodes.jsonl', rows, worker, protocol,
                                summary.get('install_metadata'))
        if (summary.get('episodes') != len(found)
                or summary.get('completed_indices') != list(found)):
            raise ValueError(f'Worker {worker} receipt count differs from saved episodes')
        complete = set(found) == set(worker_indices(worker))
        if complete:
            if summary.get('status') != 'complete' or summary.get('complete') is not True:
                raise ValueError(f'Complete worker {worker} has an invalid terminal status')
        elif (summary.get('status') == 'checkpointed' and summary.get('complete') is False
              and summary.get('soft_deadline_reached') is True):
            checkpointed.append(worker)
        else:
            raise ValueError(f'Incomplete worker {worker} did not reach its allocation soft deadline')
        if summary.get('arm_episodes') != {arm: len(found)}:
            raise ValueError(f'Worker {worker} arm coverage receipt differs')

        receipt = read_json(out / 'setup_receipt.json')
        arm_training = training[arm]['training_receipt']
        replay_max_abs = receipt.get('max_abs_vs_reference')
        if (receipt.get('passed') is not True or receipt.get('protocol') != PROTOCOL
                or receipt.get('arm') != arm or receipt.get('observed_modules') != 614
                or receipt.get('training_receipt_status') != 'complete'
                or receipt.get('source_c8') != arm_training.get('source_c8')
                or isinstance(replay_max_abs, bool)
                or not isinstance(replay_max_abs, (int, float))
                or not math.isfinite(replay_max_abs)
                or replay_max_abs > protocol['source_parity_max_abs_tolerance']):
            raise ValueError(f'Worker {worker} trained C8 action replay did not pass')
        current = validate_install_metadata(summary.get('install_metadata'), protocol, arm)
        if current != receipt.get('install_metadata'):
            raise ValueError(f'Worker {worker} install metadata differs from source replay receipt')
        if arm in metadata and metadata[arm] != current:
            raise ValueError(f'Workers for {arm} installed different trained deployments')
        metadata[arm] = current
        setup[str(worker)] = receipt
        for index, record in found.items():
            if (arm, index) in records:
                raise ValueError('Duplicate new episode slot across workers')
            records[arm, index] = record
    return records, metadata, setup, checkpointed


def _read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed completed reference record at {path}:{line_number}') from exc


def _load_old_vq(results_root, rows):
    results_root = Path(results_root)
    old_path = results_root.parent / 'protocol.json'
    old_protocol = read_json(old_path)
    old_protocol_name = 'fastwam-vq-plus-v1'
    if (old_protocol.get('protocol') != old_protocol_name
            or tuple(old_protocol.get('arms', ())) != OLD_VQ_ARMS):
        raise ValueError('Existing VQ reference protocol differs')
    found, arm_metadata = {}, {}
    for worker in range(4):
        arm = OLD_VQ_ARMS[worker // 2]
        indices = list(range(worker % 2, 280, 2))
        out = results_root / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        if (progress.get('protocol') != old_protocol_name or progress.get('worker_id') != worker
                or progress.get('status') != 'complete' or progress.get('completed') != 140
                or summary.get('protocol') != old_protocol_name or summary.get('complete') is not True
                or summary.get('worker_id') != worker or summary.get('arm') != arm
                or summary.get('episodes') != 140 or summary.get('expected') != 140
                or summary.get('indices') != indices or summary.get('arm_episodes') != {arm: 140}):
            raise ValueError(f'Existing VQ worker {worker} is not complete')
        setup = read_json(out / 'setup_receipt.json')
        metadata = summary.get('install_metadata')
        bank = old_protocol['sources'][arm]['bank']
        if (setup.get('passed') is not True or setup.get('arm') != arm
                or setup.get('observed_modules') != 614
                or setup.get('max_abs_vs_frozen_action', math.inf) > old_protocol.get('source_parity_max_abs_tolerance', 1e-6)
                or setup.get('install_metadata') != metadata
                or metadata.get('arm') != arm or metadata.get('source_bank') != bank
                or metadata.get('module_count') != 614):
            raise ValueError(f'Existing VQ worker {worker} setup receipt differs')
        if arm in arm_metadata and arm_metadata[arm] != metadata:
            raise ValueError(f'Existing VQ workers for {arm} installed different banks')
        arm_metadata[arm] = metadata
        for line_number, record in _read_jsonl(out / 'episodes.jsonl'):
            index = record.get('index')
            row = rows[index] if type(index) is int and 0 <= index < len(rows) else None
            if (row is None or index not in indices or record.get('protocol') != old_protocol_name
                    or record.get('worker_id') != worker or record.get('arm') != arm
                    or record.get('complete') is not True or type(record.get('success')) is not bool
                    or str(record.get('variant_id')) != str(row['variant_id'])
                    or record.get('suite') != row['suite'] or record.get('dimension') != row['dimension']
                    or record.get('task_id') != int(row['task_id']) or record.get('state_id') != 0
                    or record.get('env_seed') != 800_000_000 + index * 1000
                    or record.get('sampler_seed') != 900_000_000 + index * 1000):
                raise ValueError(f'Existing VQ episode identity differs at worker {worker}, line {line_number}')
            evidence = record.get('vq_evidence', {})
            required = {'arm': arm, 'source_bank': bank, 'module_count': 614,
                        'observed_module_count': 614, 'execution': 'decoded_bf16_vq',
                        'activation_bits': 4, 'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
                        'frozen_bank_reused': True, 'native_int4_gemm_calls': 0,
                        'kv_packed_prefills': 0, 'kv_layer_reads': 0}
            if (any(evidence.get(key) != value for key, value in required.items())
                    or evidence.get('quantized_linear_calls', 0) <= 0):
                raise ValueError(f'Existing VQ episode compute evidence differs at {worker}/{index}')
            seconds = record.get('episode_seconds')
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise ValueError('Invalid existing VQ episode duration')
            metrics = record.get('metrics')
            if not isinstance(metrics, list) or not metrics:
                raise ValueError('Existing VQ episode has no action-query metrics')
            for metric in metrics:
                if (type(metric.get('query_index_in_arm')) is not int
                        or type(metric.get('first_query_in_arm')) is not bool
                        or metric['first_query_in_arm'] != (metric['query_index_in_arm'] == 0)):
                    raise ValueError('Existing VQ query marker differs')
                for field in TIMING_FIELDS:
                    value = metric.get(field)
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0):
                        raise ValueError(f'Invalid existing VQ {field}')
            slot = arm, index
            if slot in found:
                raise ValueError('Duplicate existing VQ episode slot')
            found[slot] = record
    expected = {(arm, index) for arm in OLD_VQ_ARMS for index in range(280)}
    if found.keys() != expected:
        raise ValueError(f'Expected 560 completed old VQ episodes, found {len(found)}')
    return found


def run(args):
    import runner
    runner.guard()
    import reference_aggregate as old
    from vq_adapter import validate_training_artifacts

    protocol = validate_protocol(read_json(args.protocol))
    manifest, rows = load_manifest(args.manifest)
    rotation_root = Path(protocol['references']['rotation_results'])
    old_manifest, _old_rows, shards = old.load_manifest(rotation_root / 'manifest.json')
    if manifest != old_manifest:
        raise ValueError('New evaluation differs from the exact original frozen manifest')

    training = _validate_source_and_training(protocol, args.results)
    records, metadata, setup, checkpointed = load_workers(args.results, protocol, rows, training)
    rotation = old.load_reference_episodes(rotation_root, rows, shards)
    scalar, scalar_metadata, scalar_setup = old.load_scalar_episodes(
        Path(protocol['references']['scalar_results']), rows)
    references = dict(rotation)
    scalar_arm = old.ARM
    references.update({(scalar_arm, index): record for index, record in scalar.items()})
    old_vq = _load_old_vq(Path(protocol['references']['vq_results']), rows)
    references.update(old_vq)
    if len(references) != 1680 or {arm for arm, _ in references} != set(REFERENCE_ARMS):
        raise ValueError('Expected the six existing paired baselines, 280 episodes each')

    counts = collections.Counter(arm for arm, _ in records)
    complete = len(records) == 560 and not checkpointed
    summary = {
        'protocol': PROTOCOL, 'status': 'complete' if complete else 'checkpointed',
        'complete': complete, 'manifest_variants': 280, 'episode_slots': len(records),
        'reference_episode_slots': len(references), 'new_arms': list(ARMS),
        'reference_arms': list(REFERENCE_ARMS),
        'arm_episodes': {arm: counts[arm] for arm in ARMS},
        'checkpointed_workers': checkpointed, 'install_metadata': metadata,
        'training_receipts': {arm: training[arm]['training_receipt'] for arm in ARMS},
        'setup_receipts': setup,
        'source_receipts': {
            'c8': {'protocol': training[ARMS[0]]['source']['source_summary']['protocol'],
                   'cohort': 'c8', 'source_job_id': protocol['calibration']['source_job_id'],
                   'case_ids': protocol['calibration']['case_ids'], 'target_linears': 614,
                   'status': 'complete',
                   'summary': training[ARMS[0]]['source']['source_summary']},
            'reference': {'protocol': protocol['protocol'], 'status': 'complete',
                          'contexts': 8, 'target_linears': 614,
                          'binding': training[ARMS[0]]['source']['binding'],
                          'receipt': training[ARMS[0]]['source']['reference_receipt']},
        },
        'normalization_c': training[ARMS[0]]['source']['normalization_c'],
        'claim_scope': 'Paired closed-loop success on the frozen 280-case LIBERO-Plus cohort; C8-trained fixed-index VQ with decoded BF16 weights, dynamic A4 activations, and BF16 KV cache; rho and preselected epoch differ; no native low-bit kernel claim',
    }
    if complete:
        expected = {(arm, index) for arm in ARMS for index in range(280)}
        if set(records) != expected:
            raise ValueError('Missing or unexpected complete new episode slots')
        by_arm = {arm: {index: record for (name, index), record in records.items() if name == arm}
                  for arm in ARMS}
        reference_arms = list(old.REFERENCE_ARMS) + [scalar_arm] + list(OLD_VQ_ARMS)
        by_arm.update({arm: {index: references[arm, index] for index in range(280)}
                       for arm in reference_arms})

        def bucket(indices):
            return {
                'arms': {arm: _rate(values, indices) for arm, values in by_arm.items()},
                'paired': {arm: {other: _paired(by_arm[arm], by_arm[other], indices)
                                 for other in by_arm if other != arm} for arm in ARMS},
            }

        summary['overall'] = bucket(list(range(280)))
        suite_names = sorted({row['suite'] for row in rows})
        dimensions = sorted({row['dimension'] for row in rows})
        summary['suites'] = {suite: bucket([i for i, row in enumerate(rows) if row['suite'] == suite])
                             for suite in suite_names}
        summary['dimensions'] = {dim: bucket([i for i, row in enumerate(rows) if row['dimension'] == dim])
                                 for dim in dimensions}
        summary['cells'] = {f'{suite}::{dim}': bucket([i for i, row in enumerate(rows)
                              if row['suite'] == suite and row['dimension'] == dim])
                            for suite in suite_names for dim in dimensions}
        summary['timing'] = {
            arm: summarize_timing_by_implementation(
                {index: by_arm[arm][index] for index in range(280)}) for arm in ARMS}
        summary['timing_boundary'] = (
            'Query timing groups the first formal query in each worker process separately from later queries; '
            'paired trajectories can have different query counts.')

    save(args.results / 'summary.json', summary)
    lines = ['# C8 action + latent：LIBERO-Plus 闭环评测', '',
             f"状态：`{summary['status']}`；新增episode：{len(records)}/560；已有基线：{len(references)}/1680。", '',
             '新训练从同一冻结 C8 Micro-scale VQ bank 初始化；固定 index/exponent，FP32 master books 与 log row gains 按 protocol 的 epoch/update 配置训练，部署为 decoded BF16 weights、dynamic A4 activations、BF16 KV cache。', '',
             '每个worker在正式episode前重放 C8 case 2000 / seed index 0；只有 trained action 与保存的 reference action 最大绝对误差不超过 1e-6、且覆盖全部614个目标 Linear 时才开始评测。', '']
    if complete:
        lines += ['| 配置 | 成功 / episodes | Success rate |', '|---|---:|---:|']
        for arm, value in summary['overall']['arms'].items():
            lines.append(f"| {arm} | {value['successes']}/{value['episodes']} | {value['success_rate']:.2%} |")
        lines += ['', '新增 rho 两组之间及与六个既有 baseline 的配对 win/loss/tie、suite、dimension 和 suite×dimension 统计见 `summary.json`。',
                  'Latency 单独按 `batched_h128_v2` 实现记录，并区分各worker process的首个正式query与后续query；不作为native low-bit kernel性能证据。']
    else:
        lines += ['', '所有已完成的episode已保存在各worker JSONL中；只接受到 allocation soft deadline 的前缀，重新提交时可从该前缀继续。']
    (args.results / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'C8_ACTION_LATENT_AGGREGATE status={summary["status"]} slots={len(records)} reference_slots=1680', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
