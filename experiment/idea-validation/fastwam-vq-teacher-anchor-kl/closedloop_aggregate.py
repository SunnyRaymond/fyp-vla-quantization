"""Aggregate the frozen anchor-KL episodes against the reused per-channel MSE arm."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import time

import closedloop as worker

REFERENCE_ARM = 'per_channel_clip'


def _same_manifest(path, manifest, rows):
    other, other_rows = worker.load_manifest(path)
    if other != manifest or other_rows != rows:
        raise ValueError(f'Frozen reference manifest differs: {path}')


def load_reference(protocol, source_protocol, manifest, rows):
    refs = source_protocol['references']
    results = Path(refs['per_channel_results'])
    old_path = results.parent / 'protocol.json'
    old = worker.read_json(old_path)
    if (old.get('protocol') != refs.get('per_channel_protocol')
            or old.get('arm') != REFERENCE_ARM or old.get('arms') != [REFERENCE_ARM]):
        raise ValueError('MSE reference protocol is not the frozen per-channel arm')
    _same_manifest(results / 'manifest.json', manifest, rows)

    eval_root = results / 'evaluations' / REFERENCE_ARM
    summary = worker.read_json(eval_root / 'summary.json')
    marker = worker.read_json(eval_root / 'PIPELINE_COMPLETE')
    if (summary.get('protocol') != old['protocol'] or summary.get('arm') != REFERENCE_ARM
            or summary.get('status') != 'complete' or summary.get('complete') is not True
            or summary.get('manifest_variants') != 280 or summary.get('episode_slots') != 280
            or summary.get('reference_episode_slots') != 2800
            or summary.get('arm_episodes') != {REFERENCE_ARM: 280}
            or marker.get('protocol') != old['protocol'] or marker.get('arm') != REFERENCE_ARM
            or marker.get('complete') is not True or marker.get('episode_slots') != 280
            or marker.get('reference_episode_slots') != 2800):
        raise ValueError('The frozen MSE reference summary or completion marker is invalid')

    found = {}
    for worker_id in range(4):
        indices = list(range(worker_id, 280, 4))
        out = eval_root / f'worker_{worker_id}'
        progress, receipt = worker.read_json(out / 'progress.json'), worker.read_json(out / 'summary.json')
        fields = ('protocol', 'worker_id', 'arm', 'status', 'complete', 'episodes', 'expected',
                  'indices', 'completed_indices', 'arm_episodes')
        if (any(progress.get(key) != receipt.get(key) for key in fields)
                or receipt.get('protocol') != old['protocol'] or receipt.get('worker_id') != worker_id
                or receipt.get('arm') != REFERENCE_ARM or receipt.get('status') != 'complete'
                or receipt.get('complete') is not True or receipt.get('episodes') != 70
                or receipt.get('expected') != 70 or receipt.get('indices') != indices
                or receipt.get('completed_indices') != indices
                or receipt.get('arm_episodes') != {REFERENCE_ARM: 70}):
            raise ValueError(f'MSE reference worker {worker_id} is incomplete or misassigned')
        for line_no, record in _read_jsonl(out / 'episodes.jsonl'):
            index = record.get('index')
            if (type(index) is not int or index not in indices or index in found
                    or record.get('protocol') != old['protocol']
                    or record.get('arm') != REFERENCE_ARM
                    or record.get('worker_id') != worker_id or record.get('complete') is not True
                    or type(record.get('success')) is not bool):
                raise ValueError(f'Invalid MSE reference episode {worker_id}:{line_no}')
            row = rows[index]
            expected = {
                'suite': row['suite'], 'dimension': row['dimension'],
                'task_id': int(row['task_id']), 'state_id': 0,
                'env_seed': worker.environment_seed(index),
                'sampler_seed': worker.sampler_seed(index),
            }
            if (any(record.get(key) != value for key, value in expected.items())
                    or str(record.get('variant_id')) != str(row['variant_id'])):
                raise ValueError(f'MSE reference episode does not match paired slot {index}')
            found[index] = record
    if set(found) != set(range(280)):
        raise ValueError('Expected all 280 paired MSE reference episodes')
    successes = sum(item['success'] for item in found.values())
    if successes != protocol['closedloop']['expected_reference_success']:
        raise ValueError(f'MSE reference successes differ: {successes}/280')
    return found, {'protocol': old['protocol'], 'arm': REFERENCE_ARM,
                   'successes': successes, 'episodes': len(found), 'source': str(eval_root)}


def _read_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'Expected an object at {path}:{line_no}')
                yield line_no, value


def load_candidates(root, protocol, rows):
    found, setups, checkpointed = {}, {}, []
    for worker_id in range(2):
        out = Path(root) / f'closedloop_{worker_id}'
        if (out / 'failure.json').exists():
            raise ValueError(f'Candidate worker {worker_id} has a failure record')
        summary = worker.read_json(out / 'summary.json')
        setup = worker.read_json(out / 'setup_receipt.json')
        worker._validate_setup(setup, worker_id, protocol)
        records = worker.recover_episodes(out, rows, worker_id, protocol)
        indices = worker.worker_indices(worker_id)
        if (summary.get('protocol') != worker.PROTOCOL or summary.get('worker_id') != worker_id
                or summary.get('unit') != f'closedloop_{worker_id}'
                or summary.get('arm') != worker.ARM or summary.get('expected') != 140
                or summary.get('pbs_jobid') != setup.get('pbs_job_id')
                or summary.get('episodes') != len(records)
                or summary.get('indices') != indices
                or summary.get('episodes') != len(records)
                or summary.get('completed_indices') != list(records)):
            raise ValueError(f'Candidate worker {worker_id} summary differs from saved episodes')
        if len(records) == 140:
            if summary.get('status') != 'complete' or summary.get('complete') is not True:
                raise ValueError(f'Full candidate worker {worker_id} lacks complete status')
        elif (summary.get('status') == 'checkpointed' and summary.get('complete') is False
              and summary.get('soft_deadline_reached') is True):
            checkpointed.append(worker_id)
        else:
            raise ValueError(f'Candidate worker {worker_id} has an invalid partial status')
        if worker_id in setups and setups[worker_id] != setup:
            raise ValueError('Duplicate candidate setup receipt')
        setups[worker_id] = setup
        for index, record in records.items():
            if index in found:
                raise ValueError('Duplicate candidate episode slot')
            found[index] = record
    return found, setups, checkpointed


def _rate(records, indices):
    paired = [records[index] for index in indices if index in records]
    count = len(paired)
    successes = sum(record['success'] for record in paired)
    return {'successes': successes, 'episodes': count,
            'success_rate': successes / count if count else None}


def _pair(candidate, reference, indices):
    pairs = [(candidate[index]['success'], reference[index]['success']) for index in indices
             if index in candidate and index in reference]
    wins = sum(left and not right for left, right in pairs)
    losses = sum(right and not left for left, right in pairs)
    count = len(pairs)
    return {'wins': wins, 'losses': losses, 'ties': count - wins - losses,
            'episodes': count,
            'success_rate_difference': (wins - losses) / count if count else None}


def summarize(protocol, rows, candidates, references, setups, checkpointed, pbs_jobid=None):
    complete = len(candidates) == 280
    if complete and (set(candidates) != set(range(280)) or len(references) != 280):
        raise ValueError('A complete result requires all 280 paired slots')
    candidates_by_arm = {worker.ARM: candidates, REFERENCE_ARM: references}

    def bucket(indices):
        return {
            'arms': {arm: _rate(records, indices) for arm, records in candidates_by_arm.items()},
            'paired': {worker.ARM: {REFERENCE_ARM: _pair(candidates, references, indices)}},
        }

    suites = sorted({row['suite'] for row in rows})
    dimensions = sorted({row['dimension'] for row in rows})
    return {
        'protocol': worker.PROTOCOL, 'unit': 'closedloop_aggregate',
        'pbs_jobid': pbs_jobid, 'episodes': len(candidates),
        'status': 'complete' if complete else 'checkpointed', 'complete': complete,
        'manifest_variants': 280, 'episode_slots': len(candidates),
        'reference_episode_slots': len(references), 'arm_episodes': {worker.ARM: len(candidates)},
        'reference_arm': REFERENCE_ARM,
        'reference_successes': sum(record['success'] for record in references.values()),
        'expected_reference_successes': protocol['closedloop']['expected_reference_success'],
        'setup_by_worker': {str(index): {
            'pbs_job_id': setup['pbs_job_id'], 'source_action_max_abs': setup['source_action_max_abs'],
            'source_future_max_abs': setup['source_future_max_abs'],
            'uncached_cached_action_max_abs': setup['uncached_cached_action_max_abs'],
            'uncached_cached_future_max_abs': setup['uncached_cached_future_max_abs'],
            'actual_infer_action_max_abs': setup['actual_infer_action_max_abs'],
            'install_metadata': setup['install_metadata'],
        } for index, setup in setups.items()},
        'checkpointed_workers': checkpointed,
        'overall': bucket(list(range(280))),
        'suites': {suite: bucket([i for i, row in enumerate(rows) if row['suite'] == suite])
                   for suite in suites},
        'dimensions': {dimension: bucket([i for i, row in enumerate(rows)
                                         if row['dimension'] == dimension])
                       for dimension in dimensions},
        'cells': {f'{suite}::{dimension}': bucket([
            i for i, row in enumerate(rows)
            if row['suite'] == suite and row['dimension'] == dimension])
            for suite in suites for dimension in dimensions},
        'interpretation': ('Descriptive paired result on the frozen 280-variant cohort; variants '
                           'are perturbations of the same benchmark tasks, not independent task replicates. '
                           'Execution is decoded BF16 VQ with dynamic A4 and BF16 KV; no native kernel claim.'),
    }


def run(args):
    import sampler
    sampler._require_compute_allocation()
    protocol = worker.read_json(args.protocol)
    source_protocol = worker.read_json(Path(protocol['closedloop']['source_root']) / 'protocol.json')
    worker.validate_protocol(protocol, source_protocol)
    manifest_path = args.manifest or Path(protocol['closedloop']['manifest'])
    manifest, rows = worker.load_manifest(manifest_path)
    _same_manifest(Path(protocol['closedloop']['source_root']) / 'results/manifest.json', manifest, rows)
    if Path(manifest_path).as_posix() != protocol['closedloop']['manifest']:
        if os.name != 'nt' or Path(manifest_path).resolve() != Path(protocol['closedloop']['manifest']).resolve():
            raise ValueError('Manifest argument differs from the frozen source manifest path')
    results_root = Path(args.results or (Path(protocol['remote_root']) / 'results'))
    root = results_root
    references, reference_meta = load_reference(protocol, source_protocol, manifest, rows)
    candidates, setups, checkpointed = load_candidates(root, protocol, rows)
    summary = summarize(protocol, rows, candidates, references, setups, checkpointed,
                        os.environ['PBS_JOBID'])
    summary['reference_binding'] = reference_meta
    aggregate_root = results_root / 'closedloop_aggregate'
    aggregate_root.mkdir(parents=True, exist_ok=True)
    worker.save(aggregate_root / 'summary.json', summary)
    lines = [
        '# Teacher-anchor KL paired LIBERO-Plus evaluation', '',
        f"Status: {summary['status']}; new anchor-KL episodes: {summary['episode_slots']}/280; "
        f"reused MSE reference episodes: {summary['reference_episode_slots']}/280 "
        f"({summary['reference_successes']}/280 successes).", '',
        '| Scope | Anchor-KL | MSE reference | Wins | Losses | Ties | Paired n |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for scope, item in [('Overall', summary['overall']), *
                        [(f'Suite: {key}', value) for key, value in summary['suites'].items()], *
                        [(f'Dimension: {key}', value) for key, value in summary['dimensions'].items()], *
                        [(f'Cell: {key}', value) for key, value in summary['cells'].items()]]:
        left, right = item['arms'][worker.ARM], item['arms'][REFERENCE_ARM]
        pair = item['paired'][worker.ARM][REFERENCE_ARM]
        lines.append(f"| {scope} | {left['successes']}/{left['episodes']} | "
                     f"{right['successes']}/{right['episodes']} | {pair['wins']} | "
                     f"{pair['losses']} | {pair['ties']} | {pair['episodes']} |")
    lines += ['', summary['interpretation'], '']
    report = aggregate_root / 'RESULTS.md'
    temp = report.with_name(report.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(report)
    marker = aggregate_root / 'PIPELINE_COMPLETE'
    checkpoint = aggregate_root / 'PIPELINE_CHECKPOINTED'
    marker_value = {'protocol': worker.PROTOCOL, 'unit': 'closedloop_aggregate',
                    'complete': summary['complete'], 'episode_slots': summary['episode_slots'],
                    'reference_episode_slots': summary['reference_episode_slots'],
                    'arm_episodes': summary['arm_episodes']}
    if summary['complete']:
        worker.save(marker, marker_value)
        if checkpoint.exists():
            checkpoint.unlink()
    else:
        worker.save(checkpoint, marker_value)
        if marker.exists():
            marker.unlink()
    print(f"ANCHOR_KL_AGGREGATE status={summary['status']} "
          f"episodes={summary['episode_slots']}/280 reference=280", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        help='defaults to the protocol-bound frozen LIBERO-Plus manifest')
    parser.add_argument('--results', type=Path)
    result = run(parser.parse_args())
    if result['complete'] is not True:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
