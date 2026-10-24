"""Aggregate frozen QClip episodes against existing LIBERO-Plus references."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from evaluate import (
    ARMS, PROTOCOL, TIMING_FIELDS, load_manifest, read_json, recover_records, save,
    validate_install_metadata, validate_protocol, validate_setup_receipt,
    worker_arm, worker_indices,
)

REFERENCE_SLOTS = 4760
ORIGINAL_REFERENCE_ARMS = (
    'bf16', 'quarot_adapted_w4a4', 'spinquant_adapted_w4a4',
    'smooth05_hadamard_scalar_w4a4', 'row_scale_vq_a4', 'micro_scale_vq_a4',
    'rho0.1', 'rho1.5', 'clip_only', 'joint', 'per_channel_clip',
    'svdquant_w4a4', 'viditq_w4a4',
)
REFERENCE_ARMS = ORIGINAL_REFERENCE_ARMS + ('H_W16A4', 'C80_R128_W16A4', 'az', 'reader_kl')
MAX_SUMMARY_BYTES = 256 * 1024


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
    return {'wins': wins, 'losses': losses, 'ties': count - wins - losses,
            'episodes': count,
            'success_rate_difference': (wins - losses) / count if count else None}


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


def validate_reference_slots(references, reference_arms):
    if (list(reference_arms) != list(REFERENCE_ARMS)
            or len(references) != REFERENCE_SLOTS
            or set(references) != {(arm, index) for arm in REFERENCE_ARMS for index in range(280)}):
        raise ValueError(f'Expected exactly {REFERENCE_SLOTS} slots across seventeen frozen references')
    if any(type(record.get('success')) is not bool for record in references.values()):
        raise ValueError('Frozen reference record is missing boolean success')
    return references


def _read_jsonl(path):
    """Read a frozen JSONL without repairing or otherwise changing it."""
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'JSONL row is not an object: {path}:{line_number}')
            yield line_number, value


def _merge_references(target, incoming):
    overlap = target.keys() & incoming.keys()
    if overlap:
        raise ValueError(f'Reference loaders produced duplicate slots: {sorted(overlap)[:3]}')
    target.update(incoming)


def _load_per_channel_reference(results_root, rows, manifest, protocol):
    import reference_channel_eval as channel

    root = Path(results_root)
    old = read_json(root.parent / 'protocol.json')
    channel.validate_protocol(old)
    arm = 'per_channel_clip'
    if (old.get('protocol') != protocol['references']['per_channel_protocol']
            or old.get('arm') != arm or old.get('arms') != [arm]):
        raise ValueError('Existing per-channel reference protocol differs')
    old_manifest, old_rows = channel.load_manifest(root / 'manifest.json')
    if old_manifest != manifest or old_rows != rows:
        raise ValueError('Per-channel reference manifest differs from the original frozen cohort')

    eval_root = root / 'evaluations' / arm
    summary, marker = read_json(eval_root / 'summary.json'), read_json(eval_root / 'PIPELINE_COMPLETE')
    if (summary.get('protocol') != old['protocol'] or summary.get('arm') != arm
            or summary.get('status') != 'complete' or summary.get('complete') is not True
            or summary.get('manifest_variants') != 280 or summary.get('episode_slots') != 280
            or summary.get('reference_episode_slots') != 2800
            or summary.get('arm_episodes') != {arm: 280}
            or marker.get('protocol') != old['protocol'] or marker.get('arm') != arm
            or marker.get('complete') is not True or marker.get('episode_slots') != 280
            or marker.get('reference_episode_slots') != 2800):
        raise ValueError('Existing per-channel root summary or completion marker is invalid')

    endpoint = channel.read_json(old['sources']['reference_endpoint'])
    reference_action = {
        'sampler_seed': endpoint.get('sampler_seed'),
        'actual_infer_parity': old['source_expected']['cached_query_parity'],
    }
    found, arm_metadata = {}, None
    for worker in range(4):
        indices = list(range(worker, 280, 4))
        out = eval_root / f'worker_{worker}'
        progress, receipt = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                  'indices', 'completed_indices', 'arm_episodes', 'install_metadata')
        if (any(progress.get(key) != receipt.get(key) for key in fields)
                or receipt.get('protocol') != old['protocol'] or receipt.get('worker_id') != worker
                or receipt.get('arm') != arm or receipt.get('status') != 'complete'
                or receipt.get('complete') is not True or receipt.get('episodes') != 70
                or receipt.get('expected') != 70 or receipt.get('indices') != indices
                or receipt.get('completed_indices') != indices
                or receipt.get('arm_episodes') != {arm: 70}):
            raise ValueError(f'Per-channel reference worker {worker} is incomplete or misassigned')
        metadata = channel.validate_install_metadata(receipt.get('install_metadata'), old, arm)
        if arm_metadata is not None and arm_metadata != metadata:
            raise ValueError('Per-channel reference workers used different install metadata')
        arm_metadata = metadata
        setup, setup_metadata = channel._setup_receipt(
            out / 'setup_receipt.json', arm, old, reference_action)
        if setup_metadata != metadata:
            raise ValueError(f'Per-channel reference worker {worker} setup metadata differs')

        worker_slots, query_indices, seen_jobs = set(), [], set()
        for line_number, record in _read_jsonl(out / 'episodes.jsonl'):
            index = record.get('index')
            if type(index) is not int or not 0 <= index < 280:
                raise ValueError(f'Per-channel reference has invalid index at {worker}:{line_number}')
            channel.validate_record(record, rows, worker, old, metadata, arm)
            if index in worker_slots or (arm, index) in found:
                raise ValueError('Duplicate per-channel reference slot')
            worker_slots.add(index)
            found[arm, index] = record
            for metric in record['metrics']:
                query, job = metric['query_index_in_arm'], metric['pbs_job_id']
                query_indices.append(query)
                if metric['first_query_in_run'] != (job not in seen_jobs):
                    raise ValueError('Per-channel reference first-query marker is invalid')
                seen_jobs.add(job)
        if worker_slots != set(indices) or query_indices != list(range(len(query_indices))):
            raise ValueError(f'Per-channel reference worker {worker} has missing episodes or query gaps')
    if set(found) != {(arm, index) for index in range(280)}:
        raise ValueError('Expected exactly 280 completed per-channel reference episodes')
    return found


def _load_diffusion_reference(results_root, rows, manifest, protocol):
    import reference_diffusion_eval as diffusion

    root = Path(results_root)
    old = read_json(root.parent / 'protocol.json')
    diffusion.validate_protocol(old)
    if (old.get('protocol') != protocol['references']['diffusion_protocol']
            or old.get('arms') != ['svdquant_w4a4', 'viditq_w4a4']):
        raise ValueError('Existing diffusion reference protocol differs')
    old_manifest, old_rows = diffusion.load_manifest(root / 'manifest.json')
    if old_manifest != manifest or old_rows != rows:
        raise ValueError('Diffusion reference manifest differs from the original frozen cohort')

    old_arms = tuple(old['arms'])
    summary, marker = read_json(root / 'summary.json'), read_json(root / 'PIPELINE_COMPLETE')
    if (summary.get('protocol') != old['protocol'] or summary.get('status') != 'complete'
            or summary.get('complete') is not True or summary.get('manifest_variants') != 280
            or summary.get('episode_slots') != 560
            or summary.get('reference_episode_slots') != 2240
            or summary.get('arm_episodes') != {arm: 280 for arm in old_arms}
            or marker.get('protocol') != old['protocol'] or marker.get('complete') is not True
            or marker.get('episode_slots') != 560
            or marker.get('reference_episode_slots') != 2240):
        raise ValueError('Existing diffusion root summary or completion marker is invalid')

    found = {}
    for worker in range(4):
        arm = diffusion.worker_arm(worker)
        indices = diffusion.worker_indices(worker, arm)
        out = root / 'workers' / f'worker_{worker}'
        progress, receipt = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                  'indices', 'completed_indices', 'arm_episodes', 'install_metadata')
        if (any(progress.get(key) != receipt.get(key) for key in fields)
                or receipt.get('protocol') != old['protocol'] or receipt.get('worker_id') != worker
                or receipt.get('arm') != arm or receipt.get('status') != 'complete'
                or receipt.get('complete') is not True or receipt.get('episodes') != 140
                or receipt.get('expected') != 140 or receipt.get('indices') != indices
                or receipt.get('completed_indices') != indices
                or receipt.get('arm_episodes') != {arm: 140}):
            raise ValueError(f'Diffusion reference worker {worker} is incomplete or misassigned')
        metadata = diffusion.validate_install_metadata(receipt.get('install_metadata'), old, arm)
        setup = read_json(out / 'setup_receipt.json')
        diffusion.validate_setup_receipt(setup, old, worker, arm, metadata)
        if setup.get('pbs_job_id') != receipt.get('pbs_job_id', setup['pbs_job_id']):
            raise ValueError(f'Diffusion reference worker {worker} setup/PBS identity differs')

        worker_slots, query_indices, seen_jobs = set(), [], set()
        for line_number, record in _read_jsonl(out / 'episodes.jsonl'):
            index = record.get('index')
            if type(index) is not int or not 0 <= index < 280:
                raise ValueError(f'Diffusion reference has invalid index at {worker}:{line_number}')
            diffusion.validate_record(record, rows, worker, old, metadata, arm)
            if index in worker_slots or (arm, index) in found:
                raise ValueError('Duplicate diffusion reference slot')
            worker_slots.add(index)
            found[arm, index] = record
            for metric in record['metrics']:
                query, job = metric['query_index_in_arm'], metric['pbs_job_id']
                query_indices.append(query)
                if metric['first_query_in_run'] != (job not in seen_jobs):
                    raise ValueError('Diffusion reference first-query marker is invalid')
                seen_jobs.add(job)
        if worker_slots != set(indices) or query_indices != list(range(len(query_indices))):
            raise ValueError(f'Diffusion reference worker {worker} has missing episodes or query gaps')
    expected = {(arm, index) for arm in old_arms for index in range(280)}
    if set(found) != expected:
        raise ValueError(f'Expected exactly 560 completed diffusion reference episodes, got {len(found)}')
    return found


def _load_references(protocol, manifest, rows):
    import reference_aggregate as rotation
    import reference_loaders as trained
    import reference_c80_loaders as c80

    refs = protocol['references']
    rotation_root = Path(refs['rotation_results'])
    old_manifest, _old_rows, shards = rotation.load_manifest(rotation_root / 'manifest.json')
    if old_manifest != manifest:
        raise ValueError('Manifest differs from the original frozen 280-variant cohort')

    references = dict(rotation.load_reference_episodes(rotation_root, rows, shards))
    scalar_records, _metadata, _setup = rotation.load_scalar_episodes(Path(refs['scalar_results']), rows)
    _merge_references(references, {(rotation.ARM, index): record
                                  for index, record in scalar_records.items()})
    _merge_references(references, trained._load_old_vq(Path(refs['vq_results']), rows))
    _merge_references(references, trained._load_c8_trained(Path(refs['c8_trained_results']), rows))
    _merge_references(references, c80._load_c80_results(Path(refs['c80_results']), rows, protocol))
    _merge_references(references, _load_per_channel_reference(
        Path(refs['per_channel_results']), rows, manifest, protocol))
    _merge_references(references, _load_diffusion_reference(
        Path(refs['diffusion_results']), rows, manifest, protocol))
    _merge_references(references, _load_w16_references(rows, protocol))
    _merge_references(references, _load_q0_pair_references(rows, manifest, protocol))
    return validate_reference_slots(references, refs['arms']), list(refs['arms'])


def _load_w16_references(rows, protocol):
    """Read and validate the completed W16A4 Plus episodes with its frozen validator."""
    refs = protocol['references']
    source_artifacts = Path(refs['w16a4_artifacts'])
    artifact_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(artifact_root))
    try:
        import reference_w16_eval as old
    finally:
        try:
            sys.path.remove(str(artifact_root))
        except ValueError:
            pass

    old_protocol = read_json(refs['w16a4_protocol_file'])
    old.validate_protocol(old_protocol)
    _manifest, old_rows = old.load_manifest(protocol['cohort']['reference_manifest'])
    if old_rows != rows:
        raise ValueError('Completed W16A4 reference uses a different frozen manifest')

    result_root = Path(refs['w16a4_results'])
    summary = read_json(result_root / 'summary.json')
    marker = read_json(result_root / 'PIPELINE_COMPLETE')
    phase = read_json(source_artifacts / 'RUN_PHASE.json')
    if ((source_artifacts / 'exit_code.txt').read_text().strip() != '0'
            or not (source_artifacts / 'PIPELINE_COMPLETE').is_file()
            or phase.get('protocol') != old.PROTOCOL or phase.get('phase') != 'complete'
            or summary.get('protocol') != old.PROTOCOL or summary.get('complete') is not True
            or summary.get('status') != 'complete' or summary.get('episode_slots') != 560
            or summary.get('reference_episode_slots') != 3640
            or summary.get('arm_counts') != {'H_W16A4': 280, 'C80_R128_W16A4': 280}
            or marker.get('protocol') != old.PROTOCOL or marker.get('complete') is not True
            or marker.get('episode_slots') != 560 or marker.get('reference_episode_slots') != 3640):
        raise ValueError('Completed W16A4 reference job or aggregate is incomplete')

    found = {}
    for worker in range(4):
        arm = old.worker_arm(worker)
        out = result_root / 'workers' / f'worker_{worker}'
        progress = read_json(out / 'progress.json')
        receipt = read_json(out / 'summary.json')
        if (progress != receipt or receipt.get('protocol') != old.PROTOCOL
                or receipt.get('worker_id') != worker or receipt.get('arm') != arm
                or receipt.get('status') != 'complete' or receipt.get('complete') is not True
                or receipt.get('expected') != 140 or receipt.get('episodes') != 140
                or receipt.get('indices') != old.worker_indices(worker)
                or receipt.get('completed_indices') != old.worker_indices(worker)):
            raise ValueError(f'Completed W16A4 reference worker {worker} is incomplete')
        metadata = old.validate_install_metadata(receipt.get('install_metadata'), old_protocol, arm)
        setup = read_json(out / 'setup_receipt.json')
        old.validate_setup_receipt(setup, old_protocol, worker, arm, metadata)
        path = out / 'episodes.jsonl'
        payload = path.read_bytes()
        if payload and not payload.endswith(b'\n'):
            raise ValueError(f'W16A4 reference worker {worker} has an incomplete final record')
        records, queries, seen_jobs = {}, [], set()
        for line_number, line in enumerate(payload.splitlines(), 1):
            record = json.loads(line)
            index = old.validate_record(record, rows, worker, old_protocol, metadata, arm)
            if index in records:
                raise ValueError(f'Duplicate W16A4 reference slot at worker {worker}:{line_number}')
            records[index] = record
            for metric in record['metrics']:
                queries.append(metric['query_index_in_arm'])
                job = metric['pbs_job_id']
                if metric['first_query_in_run'] != (job not in seen_jobs):
                    raise ValueError(f'Invalid W16A4 first-query marker at worker {worker}:{line_number}')
                seen_jobs.add(job)
        indices = old.worker_indices(worker)
        if list(records) != indices or queries != list(range(len(queries))):
            raise ValueError(f'W16A4 reference worker {worker} has missing or reordered episodes')
        for index, record in records.items():
            key = (arm, index)
            if key in found:
                raise ValueError('Duplicate W16A4 reference episode slot')
            found[key] = record
    return found


def _load_q0_pair_references(rows, manifest, protocol):
    """Read completed Q0 paired episodes with their validator; never repair source JSONL."""
    refs = protocol['references']
    artifact_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(artifact_root))
    try:
        import reference_pair_eval as old
    finally:
        try:
            sys.path.remove(str(artifact_root))
        except ValueError:
            pass

    old_protocol = read_json(refs['pair_protocol_file'])
    old.validate_protocol(old_protocol)
    if old_protocol.get('protocol') != refs['pair_protocol']:
        raise ValueError('Completed Q0 paired reference protocol differs')
    root = Path(refs['pair_results'])
    old_manifest, old_rows = old.load_manifest(root / 'manifest.json')
    if old_manifest != manifest or old_rows != rows:
        raise ValueError('Q0 paired reference uses a different frozen manifest')

    artifact = Path(refs['pair_artifacts'])
    summary = read_json(root / 'summary.json')
    marker = read_json(root / 'PIPELINE_COMPLETE')
    phase = read_json(artifact / 'RUN_PHASE.json')
    expected_counts = {'az': 280, 'reader_kl': 280}
    if ((artifact / 'exit_code.txt').read_text(encoding='utf-8').strip() != '0'
            or not (artifact / 'PIPELINE_COMPLETE').is_file()
            or phase.get('protocol') != old.PROTOCOL or phase.get('phase') != 'complete'
            or phase.get('pbs_jobid') != refs['pair_job']
            or summary.get('protocol') != old.PROTOCOL or summary.get('status') != 'complete'
            or summary.get('complete') is not True or summary.get('episode_slots') != 560
            or summary.get('reference_episode_slots') != 4200
            or summary.get('arm_counts') != expected_counts
            or marker.get('protocol') != old.PROTOCOL or marker.get('complete') is not True
            or marker.get('episode_slots') != 560
            or marker.get('reference_episode_slots') != 4200
            or marker.get('arm_counts') != expected_counts):
        raise ValueError('Completed Q0 paired job or aggregate is incomplete')

    found = {}
    for worker in range(4):
        arm = old.worker_arm(worker)
        indices = old.worker_indices(worker, arm)
        out = root / 'workers' / f'worker_{worker}'
        progress, receipt = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                  'indices', 'completed_indices', 'arm_episodes', 'install_metadata')
        if (any(progress.get(key) != receipt.get(key) for key in fields)
                or receipt.get('protocol') != old.PROTOCOL or receipt.get('worker_id') != worker
                or receipt.get('arm') != arm or receipt.get('status') != 'complete'
                or receipt.get('complete') is not True or receipt.get('expected') != 140
                or receipt.get('episodes') != 140 or receipt.get('indices') != indices
                or receipt.get('completed_indices') != indices
                or receipt.get('arm_episodes') != {arm: 140}):
            raise ValueError(f'Completed Q0 paired worker {worker} is incomplete')
        metadata = old.validate_install_metadata(receipt.get('install_metadata'), old_protocol, arm)
        setup = read_json(out / 'setup_receipt.json')
        old.validate_setup_receipt(setup, old_protocol, worker, arm, metadata)
        if (setup.get('pbs_job_id') != refs['pair_job']
                or setup.get('pbs_job_id') != receipt.get('pbs_job_id', setup.get('pbs_job_id'))):
            raise ValueError(f'Q0 paired worker {worker} setup/PBS identity differs')

        path = out / 'episodes.jsonl'
        payload = path.read_bytes()
        if payload and not payload.endswith(b'\n'):
            raise ValueError(f'Q0 paired worker {worker} has an incomplete final record')
        records, queries, seen_jobs = {}, [], set()
        for line_number, line in enumerate(payload.splitlines(), 1):
            record = json.loads(line)
            index = old.validate_record(record, rows, worker, old_protocol, metadata, arm)
            if index in records:
                raise ValueError(f'Duplicate Q0 paired slot at worker {worker}:{line_number}')
            records[index] = record
            for metric in record['metrics']:
                query, job = metric['query_index_in_arm'], metric['pbs_job_id']
                queries.append(query)
                if metric['first_query_in_run'] != (job not in seen_jobs):
                    raise ValueError(f'Invalid Q0 paired first-query marker at worker {worker}:{line_number}')
                seen_jobs.add(job)
        if list(records) != indices or queries != list(range(len(queries))):
            raise ValueError(f'Q0 paired worker {worker} has missing or reordered episodes')
        for index, record in records.items():
            key = (arm, index)
            if key in found:
                raise ValueError('Duplicate Q0 paired reference episode slot')
            found[key] = record
    expected = {(arm, index) for arm in ('az', 'reader_kl') for index in range(280)}
    if set(found) != expected:
        raise ValueError(f'Expected exactly 560 completed Q0 paired episodes, got {len(found)}')
    return found


def load_workers(results_root, protocol, manifest, rows):
    found, by_arm, setups, metadata_by_arm, checkpointed = {}, {}, {}, {}, []
    root = Path(results_root) / 'workers'
    for worker in range(4):
        arm = worker_arm(worker)
        out = root / f'worker_{worker}'
        progress, summary = read_json(out / 'progress.json'), read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'completed',
                  'expected', 'soft_deadline_reached', 'indices', 'completed_indices',
                  'arm_episodes', 'install_metadata')
        if any(progress.get(key) != summary.get(key) for key in fields):
            raise ValueError(f'Worker {worker} progress and summary disagree')
        indices = worker_indices(worker, arm)
        if (summary.get('protocol') != PROTOCOL or summary.get('worker_id') != worker
                or summary.get('arm') != arm or summary.get('expected') != 70
                or summary.get('indices') != indices):
            raise ValueError(f'Worker {worker} differs from its frozen 70-slot shard')
        metadata = validate_install_metadata(summary.get('install_metadata'), protocol, arm)
        setup = read_json(out / 'setup_receipt.json')
        validate_setup_receipt(setup, protocol, worker, arm, metadata)
        if setup['pbs_job_id'] != summary.get('pbs_job_id', setup['pbs_job_id']):
            raise ValueError(f'Worker {worker} setup/PBS identity differs')
        if arm in metadata_by_arm and metadata_by_arm[arm] != metadata:
            raise ValueError(f'Workers for {arm} produced different install metadata')
        metadata_by_arm[arm] = metadata

        records = recover_records(out / 'episodes.jsonl', rows, worker, protocol, metadata, arm)
        if (summary.get('episodes') != len(records) or summary.get('completed') != len(records)
                or summary.get('completed_indices') != list(records)
                or summary.get('arm_episodes') != {arm: len(records)}):
            raise ValueError(f'Worker {worker} receipt count differs from saved episodes')
        if len(records) == len(indices):
            if (summary.get('status') != 'complete' or summary.get('complete') is not True
                    or summary.get('soft_deadline_reached') is not False):
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

    expected_indices = {index for worker in range(4) for index in worker_indices(worker)}
    if set(by_arm.get('qclip', {})) - expected_indices:
        raise ValueError('qclip has slots outside its frozen 280-slot coverage')
    return found, by_arm, metadata_by_arm, setups, checkpointed


def _setup_summary(setup):
    keys = ('worker_id', 'arm', 'pbs_job_id', 'source_job', 'source_endpoint',
            'passed', 'observed_modules', 'source_metrics_passed', 'source_metric_errors',
            'q0_anchor_passed', 'q0_action_max_abs', 'q0_future_max_abs',
            'actual_infer_action_max_abs', 'startup_queries', 'new_training_updates')
    return {key: setup.get(key) for key in keys}


def _metadata_summary(metadata):
    return metadata


def _make_summary(protocol, rows, records, by_arm, metadata, setups, checkpointed_workers, status):
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
        'setup_by_worker': {worker: _setup_summary(value) for worker, value in setups.items()},
        'install_metadata_by_arm': {arm: _metadata_summary(value)
                                    for arm, value in metadata.items()},
        'source_bindings': {
            arm: {key: protocol['arm_sources'][arm][key] for key in (
                'source_protocol', 'source_job', 'source_protocol_file', 'source_summary',
                'deployment', 'endpoint', 'source_checkpoint', 'endpoint_arm', 'updates',
                'rotation_updates', 'vq_updates', 'clipping_updates', 'training_exposures')}
            for arm in ARMS},
    }
    if complete and (len(records) != 280 or arm_episodes != {arm: 280 for arm in ARMS}
                     or reference_slots != REFERENCE_SLOTS):
        raise ValueError('Complete aggregate requires exactly 280 new and 4760 reference slots')

    all_arms = list(ARMS) + reference_arms

    def bucket(indices):
        return {
            'arms': {name: _rate(by_arm.get(name, {}), indices) for name in all_arms},
            'paired': {
                arm: {other: _paired(by_arm.get(arm, {}), by_arm.get(other, {}), indices)
                      for other in all_arms if other != arm}
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
        'Action-query timings are grouped by transform implementation and process-first versus '
        'subsequent queries; execution uses decoded BF16 dense MXVQ4 weights, dynamic A4 activations, and '
        'BF16 KV cache, so these values '
        'do not measure a native low-bit kernel. Fifteen-second GPU utilization/VRAM samples are '
        'collected separately in the PBS job log.')
    return summary


def _write_report(directory, summary):
    lines = [
        '# Final QClip decoded MXVQ4 evaluation — LIBERO-Plus', '',
        f"状态：{summary['status']}；新增 episodes：{summary['episode_slots']}/280；"
        f"冻结 references：{summary['reference_episode_slots']}/4760（原15个参考arms及Q0 paired的az/reader_kl）。", '',
        'QClip 使用 final80 checkpoint 的 614 个 learned clipping ratios；R0、VQ weights 和其他参数保持冻结，'
        '不进行新训练。执行使用 decoded BF16 dense MXVQ4 weights、dynamic A4 activations 和 BF16 KV cache。', '',
        'Timing 按 transform implementation 汇总，并区分每个 worker process 的 first query 与 '
        'subsequent queries。这些数据不代表 native low-bit kernel latency；GPU utilization 和 VRAM '
        'telemetry 由 PBS job 单独记录。',
    ]
    if summary['complete']:
        lines += ['', '| Arm | Successes / episodes | Success rate |', '|---|---:|---:|']
        for arm, value in summary['overall']['arms'].items():
            if arm in ARMS:
                rate = 'n/a' if value['success_rate'] is None else f"{value['success_rate']:.2%}"
                lines.append(f"| {arm} | {value['successes']}/{value['episodes']} | {rate} |")
        lines += ['', 'Paired wins/losses/ties 比较 qclip 与 17 个冻结 references；'
                  'suite、dimension 和 cell 结果见 summary.json。']
    else:
        lines += ['', '这是 soft deadline checkpoint；已保存的 episode 前缀可继续恢复。']
    atomic_text(Path(directory) / 'RESULTS.zh.md', '\n'.join(lines) + '\n')


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def save_compact_summary(path, value):
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
    return len(payload)


def run(args):
    protocol = validate_protocol(read_json(args.protocol))
    manifest, rows = load_manifest(args.manifest)
    results = Path(args.results)
    references, reference_arms = _load_references(protocol, manifest, rows)
    references = validate_reference_slots(references, reference_arms)
    records, new_by_arm, metadata, setups, checkpointed = load_workers(
        results, protocol, manifest, rows)
    if reference_arms != protocol['references']['arms']:
        raise ValueError('Reference loader arms differ from the frozen protocol')
    by_arm = {arm: {index: record for (name, index), record in references.items() if name == arm}
              for arm in reference_arms}
    by_arm.update(new_by_arm)
    new_complete = all(len(new_by_arm.get(arm, {})) == 280 for arm in ARMS)
    status = 'complete' if len(records) == 280 and new_complete else 'checkpointed'
    summary = _make_summary(protocol, rows, records, by_arm, metadata, setups,
                            checkpointed, status)
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
    print(f"R128_QCLIP_AGGREGATE status={summary['status']} "
          f"slots={summary['episode_slots']}/280 reference_slots=4760", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
