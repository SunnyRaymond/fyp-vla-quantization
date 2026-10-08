#!/usr/bin/env python3
"""Aggregate complete Fast-WAM LIBERO-Plus paired slots and measured performance."""
import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ('bf16', 'w4a8', 'w4a4', 'w4a4kv4')
PROTOCOL = 'fastwam-optional-idm-plus-pilot-v1'
ROOT = Path('/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005')
GIB = 1024 ** 3


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def real_quant_evidence(evidence, arm):
    if (not isinstance(evidence, dict) or evidence.get('real_quant') is not True
            or evidence.get('weight_state') != 'w4' or evidence.get('bf16_target_weight_absent') is not True
            or not isinstance(evidence.get('packed_weight_tensor_bytes'), (int, float))
            or evidence['packed_weight_tensor_bytes'] <= 0):
        return False
    if arm == 'w4a8':
        return (evidence.get('activation_bits') == 8 and evidence.get('activation_storage') == 'int8'
                and isinstance(evidence.get('integer_gemm_calls'), (int, float))
                and evidence['integer_gemm_calls'] > 0)
    if arm in ('w4a4', 'w4a4kv4'):
        valid = (evidence.get('activation_bits') == 4 and evidence.get('activation_storage') == 'packed_int4'
                 and evidence.get('native_int4_tensorcore') is True
                 and isinstance(evidence.get('native_int4_gemm_calls'), (int, float))
                 and evidence['native_int4_gemm_calls'] > 0)
        if arm == 'w4a4kv4':
            valid = valid and all(isinstance(evidence.get(k), (int, float)) and evidence[k] > 0
                                  for k in ('kv_packed_prefills', 'kv_layer_reads'))
        return valid
    return False


def verify_real_quant(artifact_dir, mode, expected_arms=None):
    runtime = read_json(artifact_dir / 'runtime.json')
    summary = read_json(artifact_dir / 'summary.json')
    selfcheck = read_json(artifact_dir / 'self_check.json')
    if selfcheck.get('passed') is not True:
        raise ValueError(f'{artifact_dir}: real-quant self-check/runtime metadata failed')
    q = summary.get('quantization_runtime')
    if expected_arms is None:
        if (summary.get('real_quant') is not True or not real_quant_evidence(q, 'w4a4kv4')
                or q.get('arm') != 'w4a4kv4'
                or not isinstance(q.get('integer_gemm_calls'), (int, float)) or q['integer_gemm_calls'] <= 0):
            raise ValueError(f'{artifact_dir}: missing explicit W4A8/native-INT4 runtime evidence')
    else:
        actual_arms = [arm for arm in ARMS if arm in set(expected_arms)]
        if not actual_arms:
            raise ValueError(f'{artifact_dir}: assigned retry contains no recognized arms')
        last_arm = actual_arms[-1]
        if last_arm == 'bf16':
            initial_q = runtime.get('quantization') if isinstance(runtime, dict) else None
            if (not isinstance(q, dict) or q.get('arm') != 'bf16' or q.get('weight_state') != 'bf16'
                    or not isinstance(initial_q, dict) or initial_q.get('arm') not in (None, 'bf16')
                    or initial_q.get('weight_state') != 'bf16'):
                raise ValueError(f'{artifact_dir}: BF16-only retry runtime evidence is inconsistent')
        elif (summary.get('real_quant') is not True or not real_quant_evidence(q, last_arm)
              or q.get('arm') != last_arm
              or not isinstance(q.get('integer_gemm_calls'), (int, float)) or q['integer_gemm_calls'] <= 0):
            raise ValueError(f'{artifact_dir}: missing real-quant runtime evidence for assigned arm {last_arm}')
    return runtime, summary


def manifest_rows(path):
    data = read_json(path)
    rows = data['variants']
    if len(rows) != 1400:
        raise ValueError(f'Expected 1400 frozen variants, found {len(rows)}')
    by_id, by_index = {}, {}
    for index, row in enumerate(rows):
        variant_id = str(row['variant_id'])
        if variant_id in by_id:
            raise ValueError(f'Duplicate manifest variant_id: {variant_id}')
        by_id[variant_id] = (index, row)
        by_index[index] = (variant_id, row)
    counts = Counter((r['suite'], r['dimension']) for r in rows)
    if len({r['suite'] for r in rows}) != 4 or len({r['dimension'] for r in rows}) != 7:
        raise ValueError('Frozen cohort must contain 4 base suites x 7 dimensions')
    if len(counts) != 28 or set(counts.values()) != {50}:
        raise ValueError(f'Expected 50 variants in each of 28 cells: {counts}')
    return data, rows, by_id, by_index


def source_list(doc):
    entries = doc.get('shards', doc.get('sources'))
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not isinstance(entries, list) or not entries:
        raise ValueError('shard_sources must contain a nonempty shards or sources list')
    return entries


def exit_status(source, artifact_dir):
    value = source.get('exit_status', source.get('exit_code'))
    if value is None and (artifact_dir / 'exit_code.txt').is_file():
        value = (artifact_dir / 'exit_code.txt').read_text(encoding='utf-8').strip()
    if value is None:
        raise ValueError(f'{artifact_dir}: source exit status is missing')
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f'{artifact_dir}: invalid exit status {value!r}')


def load_episode_lines(path, failed):
    if not path.is_file():
        return []
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if failed and i == len(lines) - 1 and not line.endswith('\n'):
                continue
            raise ValueError(f'Malformed episode JSONL at {path}:{i + 1}')
    return out


def parse_slots(value, label, rows, by_index):
    if not isinstance(value, list):
        raise ValueError(f'{label}: assigned_slots must be a list')
    if not value:
        raise ValueError(f'{label}: assigned_slots must not be empty')
    slots = set()
    for slot in value:
        if not isinstance(slot, dict) or set(slot) != {'index', 'arm', 'variant_id'}:
            raise ValueError(f'{label}: each assigned slot must contain index, arm, and variant_id')
        index, arm, variant_id = slot['index'], slot['arm'], slot['variant_id']
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError(f'{label}: invalid assigned slot index {index!r}')
        if arm not in ARMS:
            raise ValueError(f'{label}: unknown arm {arm!r}')
        if not isinstance(variant_id, str):
            raise ValueError(f'{label}: assigned variant_id must be a string')
        expected_id, _ = by_index[index]
        if variant_id != expected_id:
            raise ValueError(f'{label}: manifest identity mismatch at index {index}')
        key = (index, arm, variant_id)
        if key in slots:
            raise ValueError(f'{label}: duplicate assigned slot {key}')
        slots.add(key)
    return slots


def ordered_slot_records(keys):
    return [{'index': index, 'arm': arm, 'variant_id': variant_id}
            for index, arm, variant_id in sorted(keys, key=lambda x: (x[0], ARMS.index(x[1])))]


def load_shards(sources_path, doc, rows, by_id, by_index, allow_missing=False):
    base = sources_path.parent
    collected = {}
    provenance = []
    caveats = []
    for source in source_list(doc):
        if not isinstance(source, dict) or not source.get('artifact_dir', source.get('artifacts')):
            raise ValueError(f'Invalid shard source entry: {source!r}')
        artifact_dir = resolve(base, source.get('artifact_dir', source.get('artifacts')))
        status = exit_status(source, artifact_dir)
        index = source.get('array_index')
        assigned = parse_slots(source['assigned_slots'], artifact_dir, rows, by_index) if 'assigned_slots' in source else None
        if assigned is not None:
            if ('start' in source) != ('stop' in source):
                raise ValueError(f'{artifact_dir}: assigned source must provide both start and stop or neither')
            start = int(source['start']) if 'start' in source else None
            stop = int(source['stop']) if 'stop' in source else None
            if start is not None and not 0 <= start < stop <= len(rows):
                raise ValueError(f'{artifact_dir}: invalid assigned-source interval [{start},{stop})')
        else:
            size = int(source.get('shard_size', doc.get('shard_size', 4)))
            if index is None:
                if 'start' not in source or 'stop' not in source:
                    raise ValueError(f'{artifact_dir}: array_index or explicit start/stop is required')
                start, stop = int(source['start']), int(source['stop'])
            else:
                index = int(index)
                start = int(source.get('start', index * size))
                stop = int(source.get('stop', min(start + size, len(rows))))
            if size <= 0 or not 0 <= start < stop <= len(rows):
                raise ValueError(f'{artifact_dir}: invalid shard interval [{start},{stop})')
            if index is not None and (start != index * size or stop != min(start + size, len(rows))):
                raise ValueError(f'{artifact_dir}: interval does not match its array index and shard size')
        progress_path = artifact_dir / 'progress.json'
        summary_path = artifact_dir / 'summary.json'
        progress = read_json(progress_path) if progress_path.is_file() else None
        summary = read_json(summary_path) if summary_path.is_file() else None
        if status == 0:
            if not (artifact_dir / 'PIPELINE_COMPLETE').is_file():
                raise ValueError(f'{artifact_dir}: exit 0 without PIPELINE_COMPLETE')
            if not isinstance(progress, dict) or progress.get('status') != 'complete':
                raise ValueError(f'{artifact_dir}: successful shard progress is not complete')
            if not isinstance(summary, dict) or summary.get('mode') != 'shard' or summary.get('complete') is not True:
                raise ValueError(f'{artifact_dir}: successful shard summary is missing/incomplete')
            if assigned is None:
                if summary.get('episodes') != (stop - start) * len(ARMS):
                    raise ValueError(f'{artifact_dir}: summary episode count is inconsistent')
                if set(summary.get('indices', [])) != set(range(start, stop)):
                    raise ValueError(f'{artifact_dir}: summary indices do not match the assigned interval')
            else:
                if (type(summary.get('episodes')) is not int or summary['episodes'] != len(assigned)
                        or type(progress.get('completed')) is not int or progress['completed'] != len(assigned)
                        or type(progress.get('expected')) is not int or progress['expected'] != len(assigned)):
                    raise ValueError(f'{artifact_dir}: retry progress/summary cardinality does not match assigned_slots')
                summary_slots = parse_slots(summary.get('assigned_slots'), f'{artifact_dir} summary', rows, by_index)
                if summary_slots != assigned:
                    raise ValueError(f'{artifact_dir}: summary assigned_slots do not match the source assignment')
                expected_indices = {slot[0] for slot in assigned}
                indices = summary.get('indices')
                if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                        or len(indices) != len(set(indices)) or set(indices) != expected_indices):
                    raise ValueError(f'{artifact_dir}: summary indices do not match assigned_slots')
            expected_arms = [arm for arm in ARMS if any(slot[1] == arm for slot in assigned)] if assigned is not None else None
            verify_real_quant(artifact_dir, 'shard', expected_arms=expected_arms)
        elif progress is not None and progress.get('status') not in ('running', 'complete', 'failed'):
            raise ValueError(f'{artifact_dir}: invalid progress status for failed source')
        records = load_episode_lines(artifact_dir / 'episodes.jsonl', status != 0)
        source_slot_keys = set()
        for record in records:
            if record.get('protocol') != PROTOCOL or record.get('complete') is not True:
                raise ValueError(f'{artifact_dir}: invalid/incomplete episode record')
            arm = record.get('arm')
            if arm not in ARMS:
                raise ValueError(f'{artifact_dir}: unknown arm {arm!r}')
            try:
                row_index = int(record['index'])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f'{artifact_dir}: episode index missing/invalid')
            variant_id = str(record.get('variant_id'))
            if row_index not in by_index or (assigned is None and not start <= row_index < stop):
                raise ValueError(f'{artifact_dir}: episode index {row_index} is outside its shard')
            expected_id, expected_row = by_index[row_index]
            if variant_id != expected_id or by_id.get(variant_id, (None,))[0] != row_index:
                raise ValueError(f'{artifact_dir}: manifest identity mismatch at index {row_index}')
            if record.get('suite') != expected_row['suite'] or record.get('dimension') != expected_row['dimension']:
                raise ValueError(f'{artifact_dir}: suite/dimension mismatch for {variant_id}')
            if record.get('task_id') != int(expected_row['task_id']):
                raise ValueError(f'{artifact_dir}: task_id mismatch for {variant_id}')
            if not isinstance(record.get('success'), bool):
                raise ValueError(f'{artifact_dir}: success outcome is not boolean for {variant_id}')
            if arm != 'bf16' and not real_quant_evidence(record.get('real_quant_evidence'), arm):
                raise ValueError(f'{artifact_dir}: real-quant episode evidence missing for {arm}/{variant_id}')
            if assigned is not None and (row_index, arm, variant_id) not in assigned:
                raise ValueError(f'{artifact_dir}: episode slot is outside source assigned_slots: {arm}/{variant_id}')
            key = (arm, variant_id)
            if key in source_slot_keys or key in collected:
                raise ValueError(f'Duplicate episode slot: arm={arm}, variant_id={variant_id}')
            source_slot_keys.add(key)
            collected[key] = record
        if status != 0:
            caveats.append({
                'pbs_jobid': source.get('pbs_jobid', source.get('job_id')),
                'array_index': index, 'exit_status': status,
                'artifact_dir': str(artifact_dir), 'accepted_complete_slots': len(records),
                'note': 'nonzero-exit shard; only individually complete, manifest-matched episode slots were retained',
            })
        provenance.append({'pbs_jobid': source.get('pbs_jobid', source.get('job_id')),
                           'array_index': index, 'start': start, 'stop': stop, 'exit_status': status,
                           'artifact_dir': str(artifact_dir), 'episode_records': len(records),
                           'assigned_slots': ordered_slot_records(assigned) if assigned is not None else None})
    expected = {(arm, str(row['variant_id'])) for arm in ARMS for row in rows}
    missing = expected - set(collected)
    extra = set(collected) - expected
    if extra or (missing and not allow_missing):
        raise ValueError(f'Final cohort incomplete: missing={len(missing)}, unexpected={len(extra)}')
    return collected, provenance, caveats, missing


def inventory(args):
    sources_path = args.sources.resolve()
    sources_doc = read_json(sources_path)
    _, rows, by_id, by_index = manifest_rows(args.manifest)
    episodes, provenance, caveats, missing = load_shards(
        sources_path, sources_doc, rows, by_id, by_index, allow_missing=True)
    accepted = {(by_id[variant_id][0], arm, variant_id) for arm, variant_id in episodes}
    missing_keys = {(by_id[variant_id][0], arm, variant_id) for arm, variant_id in missing}
    missing_slots = ordered_slot_records(missing_keys)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'inventory.json', {
        'protocol': PROTOCOL, 'expected_slots': len(rows) * len(ARMS),
        'accepted_slots': len(accepted), 'missing_slots': len(missing_slots),
        'complete': not missing_slots, 'slots': ordered_slot_records(accepted),
        'shard_sources': provenance, 'failed_job_caveats': caveats,
    })
    write_json(output / 'missing_slots.json', {'protocol': PROTOCOL, 'slots': missing_slots})
    print(f'SLOT_INVENTORY expected={len(rows) * len(ARMS)} accepted={len(accepted)} '
          f'missing={len(missing_slots)} output={output}', flush=True)


def perf_path(doc, sources_path, cli_path):
    value = cli_path or doc.get('performance_artifact_dir', doc.get('performance'))
    if isinstance(value, dict):
        value = value.get('artifact_dir', value.get('artifacts'))
    if not value:
        raise ValueError('Performance artifact directory is required in shard_sources.json or --performance')
    return resolve(sources_path.parent, value)


def expected_contexts(rows):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row['suite'], row['dimension']), (index, row))
    return {index: str(row['variant_id']) for index, row in groups.values()}


def measured_storage(perf_dir, summary):
    if summary.get('mode') != 'performance' or summary.get('complete') is not True or summary.get('episodes') != 0:
        raise ValueError('Performance artifact is not a completed timing-only run')
    reported = summary.get('storage')
    if not isinstance(reported, dict) or not {'bf16', 'w4a8'}.issubset(reported):
        raise ValueError('Performance summary lacks BF16/W4 packed-storage exports')
    if not (perf_dir / 'storage.json').is_file():
        raise ValueError('Performance storage.json is missing')
    manifest = read_json(perf_dir / 'storage.json')
    totals = {}
    for arm in ('bf16', 'w4a8'):
        path = perf_dir / f'storage_{arm}'
        if not path.is_dir():
            raise ValueError(f'Missing packed storage artifact: {path}')
        size = 0
        for file in path.rglob('*'):
            if not file.is_file() or file.is_symlink():
                continue
            lowered = '/'.join(file.relative_to(path).parts).lower()
            if 't5' in lowered or 'vae' in lowered:
                continue
            size += file.stat().st_size
        if size <= 0:
            raise ValueError(f'Packed storage artifact is empty: {path}')
        totals[arm] = size
    if reported.get('bf16') is None or reported.get('w4a8') is None:
        raise ValueError('Runner storage manifest has null BF16 or W4 measurement')
    return totals, manifest


def load_performance(perf_dir, rows):
    runtime, summary = verify_real_quant(perf_dir, 'performance')
    contexts = expected_contexts(rows)
    by_arm = {}
    for arm in ARMS:
        path = perf_dir / f'timing_{arm}.json'
        if not path.is_file():
            raise ValueError(f'Missing timing results for {arm}: {path}')
        records = read_json(path)
        if len(records) != 84:
            raise ValueError(f'{arm}: expected 28 contexts x 3 repeats, found {len(records)} records')
        seen = set()
        per_context = defaultdict(list)
        all_latency, all_peak = [], []
        for r in records:
            if r.get('arm') != arm:
                raise ValueError(f'{path}: wrong arm label')
            if arm != 'bf16' and not real_quant_evidence(r.get('real_quant_evidence'), arm):
                raise ValueError(f'{path}: real-quant timing evidence failed for {arm} at index {r.get("index")} repeat {r.get("repeat")}')
            index, repeat = int(r['index']), int(r['repeat'])
            if index not in contexts or str(r.get('variant_id')) != contexts[index]:
                raise ValueError(f'{path}: timing context identity mismatch at index {index}')
            key = (index, repeat)
            if key in seen or repeat not in (0, 1, 2):
                raise ValueError(f'{path}: duplicate or invalid repeat key {key}')
            seen.add(key)
            latency = float(r['denoising_latency_ms'])
            peak = float(r['peak_allocated_gpu_GB'])
            if not math.isfinite(latency) or latency <= 0 or not math.isfinite(peak) or peak <= 0:
                raise ValueError(f'{path}: nonpositive/nonfinite latency or allocated peak')
            per_context[index].append(latency)
            all_latency.append(latency)
            all_peak.append(peak)
        if set(seen) != {(i, k) for i in contexts for k in range(3)}:
            raise ValueError(f'{path}: timing records do not cover every context/repeat')
        by_arm[arm] = {
            'latency_by_context_ms': {i: statistics.median(vals) for i, vals in per_context.items()},
            'latency_samples_ms': all_latency, 'peak_samples_GB': all_peak,
            'context_count': len(contexts), 'query_count': len(all_latency),
        }
    timing_summary = summary.get('timing', {})
    if set(timing_summary) != set(ARMS):
        raise ValueError('Performance summary does not cover all four arms')
    for arm in ARMS:
        if timing_summary[arm].get('paired_reference_contexts') != 28 or timing_summary[arm].get('queries') != 84:
            raise ValueError(f'Performance summary has wrong context/query count for {arm}')
    return by_arm, summary, runtime


def paired_counts(left, right, ids):
    wins = losses = ties = 0
    for variant_id in ids:
        a = left[variant_id]['success']
        b = right[variant_id]['success']
        if a and not b:
            wins += 1
        elif b and not a:
            losses += 1
        else:
            ties += 1
    return {'wins': wins, 'losses': losses, 'ties': ties, 'n': len(ids)}


def aggregate(args):
    sources_path = args.sources.resolve()
    sources_doc = read_json(sources_path)
    _, rows, by_id, by_index = manifest_rows(args.manifest)
    episodes, provenance, caveats, _ = load_shards(sources_path, sources_doc, rows, by_id, by_index)
    performance_dir = perf_path(sources_doc, sources_path, args.performance)
    timing, perf_summary, perf_runtime = load_performance(performance_dir, rows)
    storage_bytes, storage_manifest = measured_storage(performance_dir, perf_summary)

    arm_variant = {arm: {str(row['variant_id']): episodes[(arm, str(row['variant_id']))] for row in rows} for arm in ARMS}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['suite'], row['dimension'])].append(str(row['variant_id']))
    cells, csv_rows = [], []
    dimension_values = defaultdict(list)
    cell_pairings = {}
    for (suite, dimension), ids in sorted(grouped.items()):
        cell_arms = {}
        for arm in ARMS:
            successes = sum(arm_variant[arm][vid]['success'] for vid in ids)
            cell_arms[arm] = {'successes': successes, 'episodes': len(ids), 'success_rate': successes / len(ids)}
            dimension_values[(arm, dimension)].append(successes / len(ids))
        pairings = {arm: paired_counts(arm_variant[arm], arm_variant['bf16'], ids)
                    for arm in ARMS if arm != 'bf16'}
        pairings['w4a4kv4_vs_w4a4'] = paired_counts(arm_variant['w4a4kv4'], arm_variant['w4a4'], ids)
        cell_pairings[(suite, dimension)] = pairings
        cells.append({'suite': suite, 'dimension': dimension, 'n': len(ids), 'arms': cell_arms, 'paired': pairings})
        for arm in ARMS:
            latencies = timing[arm]['latency_by_context_ms'].values()
            paired_speedups = [timing['bf16']['latency_by_context_ms'][i] / timing[arm]['latency_by_context_ms'][i]
                               for i in timing[arm]['latency_by_context_ms']]
            compare = pairings.get(arm)
            kv = pairings['w4a4kv4_vs_w4a4'] if arm == 'w4a4kv4' else None
            csv_rows.append({
                'suite': suite, 'dimension': dimension, 'arm': arm,
                'successes': cell_arms[arm]['successes'], 'episodes': len(ids),
                'success_rate': cell_arms[arm]['success_rate'],
                'paired_wins_vs_bf16': '' if compare is None else compare['wins'],
                'paired_losses_vs_bf16': '' if compare is None else compare['losses'],
                'paired_ties_vs_bf16': '' if compare is None else compare['ties'],
                'paired_wins_kv4_vs_a4': '' if kv is None else kv['wins'],
                'paired_losses_kv4_vs_a4': '' if kv is None else kv['losses'],
                'Lat_median_ms': statistics.median(timing[arm]['latency_samples_ms']),
                'Spd_paired_median': statistics.median(paired_speedups),
                'Peak_max_allocated_GB': max(timing[arm]['peak_samples_GB']),
                'Storage_GiB': storage_bytes['bf16' if arm == 'bf16' else 'w4a8'] / GIB,
            })

    dimensions = []
    arm_totals = {}
    overall_pairings = {}
    all_ids = [str(r['variant_id']) for r in rows]
    for arm in ARMS:
        dim_scores = {dimension: statistics.mean(values) for (which, dimension), values in dimension_values.items() if which == arm}
        arm_totals[arm] = {
            'successes': sum(x['success'] for x in arm_variant[arm].values()),
            'episodes': len(rows), 'success_rate': sum(x['success'] for x in arm_variant[arm].values()) / len(rows),
            'balanced_7_dimension_mean': statistics.mean(dim_scores.values()),
            'dimension_means': dim_scores,
            'Lat_median_ms': statistics.median(timing[arm]['latency_samples_ms']),
            'Spd_paired_median': statistics.median(
                timing['bf16']['latency_by_context_ms'][i] / timing[arm]['latency_by_context_ms'][i]
                for i in timing[arm]['latency_by_context_ms']),
            'Peak_max_allocated_GB': max(timing[arm]['peak_samples_GB']),
            'Storage_bytes': storage_bytes['bf16' if arm == 'bf16' else 'w4a8'],
            'Storage_GiB': storage_bytes['bf16' if arm == 'bf16' else 'w4a8'] / GIB,
        }
    for dimension in sorted({r['dimension'] for r in rows}):
        dimensions.append({'dimension': dimension,
                           'arms': {arm: arm_totals[arm]['dimension_means'][dimension] for arm in ARMS}})
    for arm in ARMS:
        if arm != 'bf16':
            overall_pairings[f'{arm}_vs_bf16'] = paired_counts(arm_variant[arm], arm_variant['bf16'], all_ids)
    overall_pairings['w4a4kv4_vs_w4a4'] = paired_counts(arm_variant['w4a4kv4'], arm_variant['w4a4'], all_ids)

    result = {
        'protocol': PROTOCOL, 'complete': True, 'episode_slots': len(episodes),
        'manifest_variants': len(rows), 'cells': cells, 'dimension_balanced_means': dimensions,
        'arms': arm_totals, 'paired': overall_pairings,
        'latency_method': 'median denoising latency; speedup is median of per-context BF16/arm latency ratios across the same 28 fixed observations',
        'storage_method': 'measured packed-artifact file bytes, including scales/metadata/kept core parameters; T5/VAE paths excluded; W4A8 measured artifact represents shared packed core weights for all W4 arms',
        'performance_artifact_dir': str(performance_dir), 'storage_manifest': storage_manifest,
        'shard_sources': provenance, 'failed_job_caveats': caveats,
        'real_quant': {'performance_runtime': perf_runtime.get('quantization'),
                       'performance_summary_runtime': perf_summary.get('quantization_runtime')},
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'aggregate_result.json', result)
    fields = list(csv_rows[0])
    with (output / 'results.csv.tmp').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(csv_rows)
    (output / 'results.csv.tmp').replace(output / 'results.csv')
    write_report(output / 'RESULTS.zh.md', result)
    print(f'AGGREGATION_COMPLETE slots={len(episodes)} cells={len(cells)} output={output}', flush=True)


def write_report(path, result):
    lines = ['# Fast-WAM LIBERO-Plus pilot 结果', '',
             f"- 有效 episode slots：{result['episode_slots']} / 5600。",
             '- Lat.：28 个固定 observation 的 denoising latency 中位数；Spd.：按同一 observation 配对计算 BF16 / 当前 arm，再取 28 个 ratio 的中位数。',
             '- Peak：性能测量 queries 的最大 allocated GPU memory。Storage：实际 packed artifact 文件大小（GiB，含 scales/metadata/kept core params，排除 T5/VAE）；三个 W4 arms 共用实测 W4A8 core-weight artifact 大小。', '']
    lines += ['## 四个 arm', '', '| Arm | Success | Balanced 7-dimension mean | Lat. (ms) | Spd. | Peak (GB) | Storage (GiB) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, value in result['arms'].items():
        lines.append(f"| {arm} | {value['successes']}/{value['episodes']} ({value['success_rate']:.4f}) | {value['balanced_7_dimension_mean']:.4f} | {value['Lat_median_ms']:.3f} | {value['Spd_paired_median']:.4f}x | {value['Peak_max_allocated_GB']:.3f} | {value['Storage_GiB']:.6f} |")
    lines += ['', '## 28 个 suite × dimension cells', '', '| Base suite | Dimension | n | BF16 | W4A8 | W4A4 | W4A4KV4 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for cell in result['cells']:
        values = [cell['arms'][arm]['successes'] for arm in ARMS]
        lines.append(f"| {cell['suite']} | {cell['dimension']} | {cell['n']} | {values[0]}/50 | {values[1]}/50 | {values[2]}/50 | {values[3]}/50 |")
    lines += ['', '## 七个 dimension 的 base-suite 均衡均值', '', '| Dimension | BF16 | W4A8 | W4A4 | W4A4KV4 |',
              '|---|---:|---:|---:|---:|']
    for dim in result['dimension_balanced_means']:
        lines.append('| {} | {} | {} | {} | {} |'.format(dim['dimension'], *(f"{dim['arms'][a]:.4f}" for a in ARMS)))
    lines += ['', '## Paired success outcomes', '', '| Comparison | Wins | Losses | Ties | n |', '|---|---:|---:|---:|---:|']
    for label, value in result['paired'].items():
        lines.append(f"| {label} | {value['wins']} | {value['losses']} | {value['ties']} | {value['n']} |")
    if result['failed_job_caveats']:
        lines += ['', '## Provenance caveats', '']
        for item in result['failed_job_caveats']:
            lines.append(f"- {item.get('pbs_jobid') or item['artifact_dir']}: exit={item['exit_status']}, accepted complete slots={item['accepted_complete_slots']}; {item['note']}.")
    lines += ['', '未填写或未测量的值不会以 0 代替；本报告只有在所有 5600 个唯一 slot、28 个配对 timing contexts 和真实 storage artifacts 均通过校验后才会生成。', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=ROOT / 'shard_sources.json')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'manifest.json')
    parser.add_argument('--performance', help='performance job artifact directory; otherwise read from shard_sources.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'aggregate')
    parser.add_argument('--inventory-only', action='store_true', help='validate shards and write accepted/missing slot inventory without final reports')
    args = parser.parse_args()
    inventory(args) if args.inventory_only else aggregate(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'AGGREGATION_FAILED: {exc}', file=sys.stderr, flush=True)
        raise
