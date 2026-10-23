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
MAX_SUMMARY_BYTES = 256 * 1024


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


def _validate_source_and_training(protocol, results, arms=ARMS):
    from vq_adapter import validate_training_artifacts

    training = {}
    for arm in arms:
        training[arm] = validate_training_artifacts(protocol, arm, results / 'training' / arm)
    values = [training[arm]['training_receipt']['normalization_c'] for arm in arms]
    if any(not math.isclose(value, protocol['training']['normalization_c'],
                            rel_tol=1e-12, abs_tol=1e-12) for value in values):
        raise ValueError('The C80 arms do not use the frozen C8 normalization_c')
    return training


def load_workers(results, protocol, rows, training, arm):
    records, metadata, setup, checkpointed = {}, {}, {}, []
    for worker in range(4):
        worker_arm(worker, arm)
        out = results / 'evaluations' / arm / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                  'soft_deadline_reached', 'indices', 'completed_indices',
                  'arm_episodes', 'install_metadata')
        if any(progress.get(key) != summary.get(key) for key in fields):
            raise ValueError(f'Worker {worker} progress and summary disagree')
        if (summary.get('protocol') != PROTOCOL or summary.get('worker_id') != worker
                or summary.get('arm') != arm or summary.get('expected') != 70):
            raise ValueError(f'Worker {worker} receipt has a different frozen assignment')
        if summary.get('indices') != worker_indices(worker, arm):
            raise ValueError(f'Worker {worker} receipt differs from its exact frozen parity shard')
        found = recover_records(out / 'episodes.jsonl', rows, worker, protocol,
                                summary.get('install_metadata'), arm)
        if (summary.get('episodes') != len(found)
                or summary.get('completed_indices') != list(found)):
            raise ValueError(f'Worker {worker} receipt count differs from saved episodes')
        complete = set(found) == set(worker_indices(worker, arm))
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
        replay_max_abs = receipt.get('max_abs_vs_reference')
        reference = training[arm]['reference_action']
        if (receipt.get('passed') is not True or receipt.get('protocol') != PROTOCOL
                or receipt.get('arm') != arm
                or receipt.get('case_id') != protocol['closed_loop_reference']['case_id']
                or receipt.get('seed_index') != protocol['closed_loop_reference']['seed_index']
                or receipt.get('sampler_seed') != reference['sampler_seed']
                or receipt.get('reference_actual_infer_parity') != reference['actual_infer_parity']
                or receipt.get('observed_modules') != 614
                or receipt.get('training_receipt_status') != 'complete'
                or isinstance(replay_max_abs, bool)
                or not isinstance(replay_max_abs, (int, float))
                or not math.isfinite(replay_max_abs)
                or replay_max_abs > protocol['closed_loop_reference']['max_abs_tolerance']):
            raise ValueError(f'Worker {worker} trained C80 action replay did not pass')
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
        query_indices, seen_runs = [], set()
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


def _load_c8_trained(results_root, rows):
    """Reuse the completed C8 rho episodes without importing its evaluate module."""
    results_root = Path(results_root)
    old_protocol = read_json(results_root.parent / 'protocol.json')
    old_name = 'fastwam-vq-c8-action-latent-plus-v1'
    old_arms = ('rho0.1', 'rho1.5')
    old_cal = old_protocol.get('calibration', {})
    if (old_protocol.get('protocol') != old_name or old_protocol.get('arms') != list(old_arms)
            or old_cal.get('cohort') != 'c8' or len(old_cal.get('case_ids', [])) != 8):
        raise ValueError('C8 trained reference protocol differs')
    root_summary = read_json(results_root / 'summary.json')
    if (root_summary.get('protocol') != old_name or root_summary.get('status') != 'complete'
            or root_summary.get('complete') is not True or root_summary.get('episode_slots') != 560
            or root_summary.get('manifest_variants') != 280
            or root_summary.get('new_arms') != list(old_arms)
            or root_summary.get('arm_episodes') != {'rho0.1': 280, 'rho1.5': 280}):
        raise ValueError('C8 trained result summary is not complete for both frozen arms')
    training_receipts = root_summary.get('training_receipts', {})
    configs = old_protocol['arm_configs']
    for arm in old_arms:
        receipt = training_receipts.get(arm, {})
        config = configs[arm]
        if (receipt.get('protocol') != old_name or receipt.get('arm') != arm
                or receipt.get('status') != 'complete' or receipt.get('complete') is not True
                or receipt.get('rho') != config['rho'] or receipt.get('epochs') != config['epochs']
                or receipt.get('updates') != config['updates'] or receipt.get('target_linears') != 614
                or receipt.get('case_ids') != old_cal['case_ids']):
            raise ValueError(f'C8 trained receipt differs for {arm}')

    found, arm_metadata = {}, {}
    for worker in range(4):
        arm = old_arms[worker // 2]
        indices = list(range(worker % 2, 280, 2))
        out = results_root / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                  'indices', 'completed_indices', 'arm_episodes', 'install_metadata')
        if (any(progress.get(key) != summary.get(key) for key in fields)
                or summary.get('protocol') != old_name or summary.get('worker_id') != worker
                or summary.get('arm') != arm or summary.get('status') != 'complete'
                or summary.get('complete') is not True or summary.get('episodes') != 140
                or summary.get('expected') != 140 or summary.get('indices') != indices
                or summary.get('completed_indices') != indices
                or summary.get('arm_episodes') != {arm: 140}):
            raise ValueError(f'C8 trained worker {worker} is not complete on its exact shard')
        metadata = summary['install_metadata']
        config = configs[arm]
        expected = {
            'arm': arm, 'source_bank': old_cal['bank_directory'],
            'source_job_id': old_cal['source_job_id'],
            'source_protocol': 'fastwam-vq-calibration-suites-v1', 'source_cohort': 'c8',
            'source_case_ids': old_cal['case_ids'], 'module_count': 614,
            'execution': 'decoded_bf16_vq', 'activation_bits': 4,
            'kv_cache_dtype': 'bf16', 'native_low_bit_kernel': False,
            'rho': config['rho'], 'epochs': config['epochs'], 'updates': config['updates'],
            'transform_implementation': 'batched_h128_v2', 'smooth_alpha': 0.5,
            'hadamard_block': 128, 'scale_block': 32, 'fixed_indices': True,
            'fixed_scale_exponents': True, 'deployed_books_dtype': 'bfloat16',
            'deployed_row_scales_dtype': 'bfloat16',
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f'C8 trained install metadata differs for {arm}')
        norm, latent_weight = metadata.get('normalization_c'), metadata.get('latent_weight')
        if (isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(norm)
                or isinstance(latent_weight, bool) or not isinstance(latent_weight, (int, float))
                or not math.isfinite(latent_weight)
                or not math.isclose(latent_weight, config['rho'] * norm, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError(f'C8 trained normalization differs for {arm}')
        if arm in arm_metadata and arm_metadata[arm] != metadata:
            raise ValueError(f'C8 workers for {arm} installed different deployments')
        arm_metadata[arm] = metadata

        setup = read_json(out / 'setup_receipt.json')
        max_abs = setup.get('max_abs_vs_reference')
        if (setup.get('passed') is not True or setup.get('protocol') != old_name
                or setup.get('arm') != arm or setup.get('observed_modules') != 614
                or setup.get('training_receipt_status') != 'complete'
                or setup.get('install_metadata') != metadata
                or isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
                or not math.isfinite(max_abs)
                or max_abs > old_protocol.get('source_parity_max_abs_tolerance', 1e-6)):
            raise ValueError(f'C8 trained worker {worker} action replay did not pass')

        query_indices, seen_runs = [], set()
        for line_number, record in _read_jsonl(out / 'episodes.jsonl'):
            index = record.get('index')
            row = rows[index] if type(index) is int and 0 <= index < len(rows) else None
            if (row is None or index not in indices or record.get('protocol') != old_name
                    or record.get('worker_id') != worker or record.get('arm') != arm
                    or record.get('complete') is not True or type(record.get('success')) is not bool
                    or str(record.get('variant_id')) != str(row['variant_id'])
                    or record.get('suite') != row['suite'] or record.get('dimension') != row['dimension']
                    or int(record.get('task_id', -1)) != int(row['task_id'])
                    or record.get('state_id') != 0
                    or record.get('env_seed') != 800_000_000 + index * 1000
                    or record.get('sampler_seed') != 900_000_000 + index * 1000
                    or record.get('transform_implementation') != 'batched_h128_v2'
                    or not record.get('pbs_job_id')):
                raise ValueError(f'C8 trained episode identity differs at {worker}/{line_number}')
            evidence = record.get('vq_evidence', {})
            if (any(evidence.get(key) != value for key, value in metadata.items())
                    or evidence.get('observed_module_count') != 614
                    or evidence.get('native_int4_gemm_calls') != 0
                    or evidence.get('kv_packed_prefills') != 0 or evidence.get('kv_layer_reads') != 0
                    or not isinstance(evidence.get('quantized_linear_calls'), int)
                    or evidence.get('quantized_linear_calls', 0) <= 0):
                raise ValueError(f'C8 trained VQ/A4 evidence differs at {worker}/{index}')
            metrics = record.get('metrics')
            replans = record.get('replans')
            seconds = record.get('episode_seconds')
            if (type(replans) is not int or replans <= 0 or not isinstance(metrics, list)
                    or len(metrics) != replans
                    or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise ValueError(f'C8 trained episode has no query timing at {worker}/{index}')
            for replan, metric in enumerate(metrics):
                query_index = metric.get('query_index_in_arm')
                run_id = metric.get('pbs_job_id')
                if (type(query_index) is not int or query_index < 0
                        or metric.get('first_query_in_arm') is not (query_index == 0)
                        or metric.get('replan') != replan
                        or metric.get('sampler_seed') != 900_000_000 + index * 1000 + replan
                        or metric.get('transform_implementation') != 'batched_h128_v2'
                        or run_id != record['pbs_job_id']
                        or metric.get('first_query_in_run') is not (run_id not in seen_runs)):
                    raise ValueError(f'C8 trained query timing identity differs at {worker}/{index}')
                seen_runs.add(run_id)
                query_indices.append(query_index)
                for field in TIMING_FIELDS:
                    value = metric.get(field)
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0):
                        raise ValueError(f'Invalid C8 trained {field} at {worker}/{index}')
            slot = arm, index
            if slot in found:
                raise ValueError('Duplicate C8 trained reference episode slot')
            found[slot] = record
        if query_indices != list(range(len(query_indices))):
            raise ValueError(f'C8 trained worker {worker} query sequence is not a prefix')
    expected_slots = {(arm, index) for arm in old_arms for index in range(280)}
    if found.keys() != expected_slots:
        raise ValueError(f'Expected 560 completed C8 trained episodes, found {len(found)}')
    return found


def run(args):
    import runner
    runner.guard()
    import reference_aggregate as old
    from vq_adapter import validate_training_artifacts

    protocol = validate_protocol(read_json(args.protocol))
    if args.arm not in ARMS:
        raise ValueError(f'--arm must be one of {ARMS}')
    manifest, rows = load_manifest(args.manifest)
    rotation_root = Path(protocol['references']['rotation_results'])
    old_manifest, _old_rows, shards = old.load_manifest(rotation_root / 'manifest.json')
    if manifest != old_manifest:
        raise ValueError('New evaluation differs from the exact original frozen manifest')

    arm = args.arm
    training = _validate_source_and_training(protocol, args.results, (arm,))
    records, metadata, setup, checkpointed = load_workers(
        args.results, protocol, rows, training, arm)
    rotation = old.load_reference_episodes(rotation_root, rows, shards)
    scalar, scalar_metadata, scalar_setup = old.load_scalar_episodes(
        Path(protocol['references']['scalar_results']), rows)
    references = dict(rotation)
    scalar_arm = old.ARM
    references.update({(scalar_arm, index): record for index, record in scalar.items()})
    old_vq = _load_old_vq(Path(protocol['references']['vq_results']), rows)
    references.update(old_vq)
    c8_trained = _load_c8_trained(Path(protocol['references']['c8_trained_results']), rows)
    references.update(c8_trained)
    if (len(references) != 2240
            or {arm for arm, _ in references} != set(REFERENCE_ARMS) | {'rho0.1', 'rho1.5'}):
        raise ValueError('Expected the six existing paired baselines, 280 episodes each')

    if len(references) != 2240:
        raise ValueError(f'Expected exactly 2240 reused reference slots, got {len(references)}')
    reference_arms = (list(old.REFERENCE_ARMS) + [scalar_arm] + list(OLD_VQ_ARMS)
                      + ['rho0.1', 'rho1.5'])
    expected_reference_slots = {(name, index) for name in reference_arms for index in range(280)}
    if set(references) != expected_reference_slots:
        raise ValueError('Reused references do not contain exactly the eight frozen arms x 280 slots')
    reference_by_arm = {name: {index: references[name, index] for index in range(280)}
                        for name in reference_arms}
    per_arm_root = args.results / 'evaluations' / arm
    complete = len(records) == 280 and not checkpointed
    expected_current = {(arm, index) for index in range(280)}
    if complete and set(records) != expected_current:
        raise ValueError(f'{arm} is missing or has unexpected complete episode slots')
    by_arm = {name: dict(values) for name, values in reference_by_arm.items()}
    by_arm[arm] = {index: records[arm, index] for index in range(280)}
    summary = _make_summary(
        protocol, rows, records, by_arm, metadata, setup, training, checkpointed,
        candidates=(arm,), status='complete' if complete else 'checkpointed',
        reference_slots=len(references), arm=arm)
    save_compact_summary(per_arm_root / 'summary.json', summary)
    _write_report(per_arm_root, summary, references=len(references), new_slots=len(records),
                  candidates=(arm,), complete=complete)
    if complete:
        save(per_arm_root / 'PIPELINE_COMPLETE', {
            'protocol': PROTOCOL, 'arm': arm, 'complete': True,
            'episode_slots': 280, 'reference_episode_slots': 2240,
        })
    else:
        (per_arm_root / 'PIPELINE_COMPLETE').unlink(missing_ok=True)

    _maybe_write_combined(args.results, protocol, rows, references, reference_by_arm,
                          arm, records, metadata, setup, training, checkpointed)
    print(f'C80_CLIP_JOINT_AGGREGATE arm={arm} status={summary["status"]} '
          f'slots={len(records)}/280 reference_slots=2240', flush=True)


def _make_summary(protocol, rows, records, by_arm, metadata, setup, training,
                  checkpointed, candidates, status, reference_slots, arm=None):
    candidate_counts = {name: sum(key[0] == name for key in records) for name in candidates}
    expected_slots = 280 * len(candidates)
    complete = status == 'complete'
    summary = {
        'protocol': PROTOCOL, 'status': status, 'complete': complete,
        'manifest_variants': 280, 'episode_slots': len(records),
        'reference_episode_slots': reference_slots, 'new_arms': list(candidates),
        'reference_arms': [name for name in by_arm if name not in candidates],
        'arm_episodes': candidate_counts, 'checkpointed_workers': checkpointed,
        'install_metadata': metadata,
        'training_receipts': {name: _compact_training_receipt(name, training[name])
                              for name in training},
        'setup_receipts': setup,
        'source_receipts': {
            'c80_training_receipt_paths': {
                name: str(Path(training[name]['training_dir']) / 'training_receipt.json')
                for name in training},
            'c80_reference_action': {
                name: training[name]['reference_action']['actual_infer_parity'] for name in training},
            'c8_reused_arms': ['rho0.1', 'rho1.5'],
            'c8_training_summary': _compact_c8_summary(protocol),
        },
        'normalization_c': protocol['training']['normalization_c'],
        'claim_scope': protocol['claim_scope'],
    }
    if arm is not None:
        summary['arm'] = arm
    if complete:
        expected = {(name, index) for name in candidates for index in range(280)}
        if len(records) != expected_slots or set(records) != expected:
            raise ValueError('Complete aggregate has missing or unexpected new episode slots')

        def bucket(indices):
            return {
                'arms': {name: _rate(values, indices) for name, values in by_arm.items()},
                'paired': {name: {other: _paired(by_arm[name], by_arm[other], indices)
                                  for other in by_arm if other != name} for name in candidates},
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
            name: summarize_timing_by_implementation(
                {index: by_arm[name][index] for index in range(280)}) for name in candidates}
        summary['timing_boundary'] = (
            'Query timing groups the first formal query in each worker process separately from later queries; '
            'paired trajectories can have different query counts.')
    return summary


def _write_report(directory, summary, references, new_slots, candidates, complete):
    directory = Path(directory)
    lines = ['# C80 clip-only 与 joint：LIBERO-Plus 闭环评测', '',
             f"状态：`{summary['status']}`；新增episode：{new_slots}/{280 * len(candidates)}；"
             f'复用baseline：{references}/2240。', '',
             '两个arm均从同一 C8 rho1.5 update8 bank开始，使用冻结C80的80个context完成80次更新；clip_only只训练614个clipping ratio，joint训练FP32 codebooks、log row gains与clipping ratios，不包含channel scaling。评测执行decoded BF16 weights、dynamic A4 activations、BF16 KV cache。', '',
             '每个worker在正式episode前重放case 2000 / seed index 0；保存的 reference action 和当前 runtime 都必须覆盖614个target Linear，且最大绝对误差不超过1e-6。', '']
    if complete:
        lines += ['| 配置 | 成功 / episodes | Success rate |', '|---|---:|---:|']
        for name, value in summary['overall']['arms'].items():
            lines.append(f"| {name} | {value['successes']}/{value['episodes']} | {value['success_rate']:.2%} |")
        lines += ['', '配对 win/loss/tie，以及suite、dimension和suite×dimension统计见 `summary.json`。',
                  'Timing按 `batched_h128_v2` implementation报告，并区分每个worker process的首个正式query与后续query；该实验没有native low-bit kernel调用。']
    else:
        lines += ['', '所有已完成的episode已保存在各worker JSONL中；只接受到 allocation soft deadline 的前缀，重新提交时可从该前缀继续。']
    atomic_text(directory / 'RESULTS.zh.md', '\n'.join(lines) + '\n')


def _maybe_write_combined(results, protocol, rows, references, reference_by_arm,
                          current_arm, current_records, current_metadata, current_setup,
                          current_training, current_checkpointed):
    """Write the shared 560-slot result only after both arm artifacts are complete."""
    from vq_adapter import validate_training_artifacts

    results = Path(results)
    candidate_training = dict(current_training)
    candidate_records = dict(current_records)
    candidate_metadata = dict(current_metadata)
    candidate_setup = dict(current_setup)
    checkpointed = list(current_checkpointed)
    for other in ARMS:
        if other == current_arm:
            continue
        summary_path = results / 'evaluations' / other / 'summary.json'
        marker_path = results / 'evaluations' / other / 'PIPELINE_COMPLETE'
        if not summary_path.is_file() or not marker_path.is_file():
            _invalidate_combined(results)
            return
        other_summary = read_json(summary_path)
        if (other_summary.get('protocol') != PROTOCOL or other_summary.get('arm') != other
                or other_summary.get('status') != 'complete'
                or other_summary.get('complete') is not True
                or other_summary.get('episode_slots') != 280
                or other_summary.get('reference_episode_slots') != 2240
                or other_summary.get('arm_episodes') != {other: 280}):
            _invalidate_combined(results)
            return
        candidate_training[other] = validate_training_artifacts(
            protocol, other, results / 'training' / other)
        found, meta, setup, partial = load_workers(
            results, protocol, rows, candidate_training, other)
        if partial or len(found) != 280:
            _invalidate_combined(results)
            return
        candidate_records.update(found)
        candidate_metadata.update(meta)
        candidate_setup.update(setup)
    if len(candidate_records) != 560:
        raise ValueError('Both per-arm summaries exist but combined coverage is not exactly 560')
    by_arm = dict(reference_by_arm)
    for name in ARMS:
        by_arm[name] = {index: candidate_records[name, index] for index in range(280)}
    summary = _make_summary(
        protocol, rows, candidate_records, by_arm, candidate_metadata, candidate_setup,
        candidate_training, checkpointed, candidates=ARMS, status='complete',
        reference_slots=len(references))
    summary.pop('arm', None)
    _write_report(results, summary, references=len(references), new_slots=560,
                  candidates=ARMS, complete=True)
    save_compact_summary(results / 'summary.json', summary)
    save(results / 'PIPELINE_COMPLETE', {
        'protocol': PROTOCOL, 'complete': True, 'episode_slots': 560,
        'arm_episodes': {name: 280 for name in ARMS}, 'reference_episode_slots': 2240,
    })


def _invalidate_combined(results):
    (Path(results) / 'PIPELINE_COMPLETE').unlink(missing_ok=True)
    (Path(results) / 'summary.json').unlink(missing_ok=True)


def atomic_text(path, content):
    import os
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(content, encoding='utf-8')
    temp.replace(path)


def _compact_training_receipt(arm, training_entry):
    receipt = training_entry['training_receipt']
    fields = ('schema_version', 'protocol', 'arm', 'status', 'complete', 'updates',
              'epochs', 'target_linears', 'calibration_ids', 'update_order',
              'normalization_c', 'rho', 'weights_unchanged', 'history_source')
    compact = {key: receipt[key] for key in fields if key in receipt}
    compact['receipt_path'] = str(Path(training_entry['training_dir']) / 'training_receipt.json')
    if isinstance(receipt.get('source'), dict):
        compact['source'] = receipt['source']
    backward = receipt.get('backward_check')
    if isinstance(backward, dict):
        compact['backward_check'] = {
            key: backward[key] for key in ('passed', 'implementation', 'max_vectors_per_chunk')
            if key in backward}
    return compact


def _compact_c8_summary(protocol):
    results = Path(protocol['references']['c8_trained_results'])
    summary = read_json(results / 'summary.json')
    old_protocol = read_json(results.parent / 'protocol.json')
    expected_arms = list(protocol['references']['c8_trained_arms'])
    if (summary.get('protocol') != 'fastwam-vq-c8-action-latent-plus-v1'
            or summary.get('status') != 'complete' or summary.get('complete') is not True
            or summary.get('episode_slots') != 560
            or summary.get('arm_episodes') != {arm: 280 for arm in expected_arms}
            or old_protocol.get('protocol') != summary.get('protocol')
            or old_protocol.get('arms') != expected_arms):
        raise ValueError('C8 source summary is not complete for both frozen arms')
    calibration = old_protocol.get('calibration', {})
    configs = old_protocol.get('arm_configs', {})
    return {
        'protocol': summary['protocol'], 'status': summary['status'],
        'complete': summary['complete'], 'episode_slots': summary['episode_slots'],
        'manifest_variants': summary.get('manifest_variants'),
        'new_arms': expected_arms, 'arm_episodes': summary['arm_episodes'],
        'source_job_id': protocol['references']['c8_trained_job_id'],
        'results_path': str(results), 'summary_path': str(results / 'summary.json'),
        'calibration': {
            'cohort': calibration.get('cohort'), 'source_job_id': calibration.get('source_job_id'),
            'case_ids': calibration.get('case_ids'), 'bank_directory': calibration.get('bank_directory'),
            'target_linears': 614,
        },
        'arm_configs': {arm: {key: configs[arm].get(key) for key in ('rho', 'epochs', 'updates')}
                        for arm in expected_arms},
        'normalization_c': summary.get('normalization_c'),
    }


def save_compact_summary(path, value):
    """Atomically persist the controller-readable summary under its 256 KiB cap."""
    import os
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False) + '\n').encode('utf-8')
    if len(payload) > MAX_SUMMARY_BYTES:
        raise ValueError(f'Aggregate summary is {len(payload)} bytes; limit is {MAX_SUMMARY_BYTES}')
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_bytes(payload)
    temp.replace(path)
    return len(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
