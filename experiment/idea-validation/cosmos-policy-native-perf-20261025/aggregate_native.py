"""Aggregate completed, matched native performance summaries in a CPU allocation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path

import perf_native as perf
from perf_native import (ARMS, compatible_gpu_key, _gate_limits, _require_gate,
                         read_json, validate_variant)

SCOPES = ('sampler', 'dit_replay')
LABELS = ('BF16', 'SpinQuant', 'QuaRot', 'SVDQuant', 'ViDiT-Q', 'Ours')
LEGACY_SAMPLES = 720
LEGACY_BLOCKS = 3
LEGACY_REPETITIONS = 3
LEGACY_DESIGN = '80q_x3blocks_x3repetitions'
LEGACY_COHORT_QUERIES = 80
ACTIVE_COHORT_QUERIES = 40
LEGACY_SEEDS = (0, 1)
ACTIVE_SEEDS = (0,)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def base_binding(binding):
    result = {key: value for key, value in binding.items() if key != 'dit_replay_dispatch'}
    if result.get('adapter_epilogue') == 'fused_native_scaled_add':
        result['adapter_epilogue'] = 'fused_cuda_fp32_group_scaled_add'
    return result


def _summary_devices(summary, expected_samples):
    provenance = summary.get('sample_hardware_provenance')
    if provenance is None:
        hardware = summary.get('hardware', {})
        uuid = hardware.get('gpu_uuid')
        require(bool(uuid), 'Summary has no measured GPU UUID provenance')
        return [{'gpu_uuid': str(uuid),
                 'hostnames': [str(hardware['hostname'])] if hardware.get('hostname') else [],
                 'pbs_job_ids': [str(hardware['pbs_job_id'])] if hardware.get('pbs_job_id') else []}]
    require(provenance.get('sample_count') == expected_samples,
            f'Hardware provenance does not cover the selected {expected_samples} performance samples')
    devices = provenance.get('devices')
    require(isinstance(devices, list) and devices,
            'Summary has no sample-level GPU UUID provenance')
    result = []
    for row in devices:
        uuid = row.get('gpu_uuid') if isinstance(row, dict) else None
        require(bool(uuid), 'Sample hardware provenance is missing a GPU UUID')
        result.append({'gpu_uuid': str(uuid),
                       'hostnames': sorted({str(value) for value in row.get('hostnames', [])}),
                       'pbs_job_ids': sorted({str(value) for value in row.get('pbs_job_ids', [])})})
    return result


def _matches_measurement_shape(summary, scope, samples, blocks, repetitions, design, cohort_queries):
    if (not isinstance(summary, dict) or summary.get('status') != 'complete'
            or summary.get('complete') is not True):
        return False
    if design == LEGACY_DESIGN:
        if 'performance_design' in summary:
            return False
    elif summary.get('performance_design') != perf.PERFORMANCE_DESIGN:
        return False
    coverage = summary.get('coverage', {})
    timing_key = 'denoising_wall_seconds' if scope == 'sampler' else 'dit_replay_seconds'
    timing = summary.get(timing_key)
    peak = summary.get('peak_allocated_gb_decimal')
    provenance = summary.get('sample_hardware_provenance')
    return (isinstance(coverage, dict) and isinstance(timing, dict) and isinstance(peak, dict)
            and all(coverage.get(key) == value for key, value in
                {'queries': cohort_queries, 'samples': samples, 'expected_samples': samples,
                 'blocks': blocks, 'repetitions': repetitions}.items())
            and timing.get('count') == samples and peak.get('count') == samples
            and (provenance is None or isinstance(provenance, dict)
                 and provenance.get('sample_count') == samples))


def _select_summary(root: Path, scope: str, variant: str, arm: str):
    legacy_path = root / 'results/native_performance' / scope / variant / arm / 'summary.json'
    active_path = perf.performance_root(root) / scope / variant / arm / 'summary.json'
    incomplete_legacy = False
    if legacy_path.is_file():
        legacy = read_json(legacy_path)
        if _matches_measurement_shape(legacy, scope, LEGACY_SAMPLES, LEGACY_BLOCKS,
                                      LEGACY_REPETITIONS, LEGACY_DESIGN, LEGACY_COHORT_QUERIES):
            return (legacy_path, legacy, LEGACY_SAMPLES, LEGACY_BLOCKS, LEGACY_REPETITIONS,
                    LEGACY_DESIGN, LEGACY_COHORT_QUERIES, LEGACY_SEEDS, None)
        incomplete_legacy = True
    if active_path.is_file():
        active = read_json(active_path)
        require(_matches_measurement_shape(active, scope, perf.PERFORMANCE_SAMPLES,
                                           perf.PERFORMANCE_BLOCKS, perf.PERFORMANCE_REPETITIONS,
                                           perf.PERFORMANCE_DESIGN, ACTIVE_COHORT_QUERIES),
                f'{scope}/{arm}: active 40-context summary is incomplete or malformed')
        unused_legacy = str(legacy_path.relative_to(root)) if incomplete_legacy else None
        return (active_path, active, perf.PERFORMANCE_SAMPLES, perf.PERFORMANCE_BLOCKS,
                perf.PERFORMANCE_REPETITIONS, perf.PERFORMANCE_DESIGN,
                ACTIVE_COHORT_QUERIES, ACTIVE_SEEDS, unused_legacy)
    if incomplete_legacy:
        raise ValueError(f'{scope}/{arm}: incomplete legacy 720 summary and no complete 40-context summary')
    active = read_json(active_path)
    require(_matches_measurement_shape(active, scope, perf.PERFORMANCE_SAMPLES,
                                       perf.PERFORMANCE_BLOCKS, perf.PERFORMANCE_REPETITIONS,
                                       perf.PERFORMANCE_DESIGN, ACTIVE_COHORT_QUERIES),
            f'{scope}/{arm}: active 40-context summary is incomplete or malformed')
    return (active_path, active, perf.PERFORMANCE_SAMPLES, perf.PERFORMANCE_BLOCKS,
            perf.PERFORMANCE_REPETITIONS, perf.PERFORMANCE_DESIGN,
            ACTIVE_COHORT_QUERIES, ACTIVE_SEEDS, None)


def _merge_device_provenance(existing, rows):
    for row in rows:
        uuid = row['gpu_uuid']
        device = existing.setdefault(uuid, {'gpu_uuid': uuid, 'hostnames': set(), 'pbs_job_ids': set()})
        device['hostnames'].update(row['hostnames'])
        device['pbs_job_ids'].update(row['pbs_job_ids'])


def _baseline_report_only(protocol: dict, arm: str) -> bool:
    policy = protocol.get('baseline_fidelity_policy')
    if policy is None:
        return False
    require(policy == 'report_only', f'Unsupported baseline fidelity policy: {policy!r}')
    helper = getattr(perf, 'baseline_fidelity_is_report_only', None)
    require(callable(helper), 'perf_native baseline fidelity policy helper is missing')
    return bool(helper(protocol, arm))


def _fidelity_phase_evidence(root: Path, protocol: dict, arm: str, variant: str,
                             scope: str, phase: str, binding: dict) -> dict:
    path = (root / 'results/native_gate' / scope / variant / arm / f'{phase}.json')
    frozen_thresholds = _gate_limits(scope, arm, variant, full_panel=(phase == 'full80'))
    expected_queries = 80 if phase == 'full80' else 4
    evidence = {'phase': phase, 'path': str(path.relative_to(root)),
                'status': 'missing', 'completed_queries': None,
                'expected_queries': expected_queries, 'recorded_query_count': None,
                'query_status_counts': {}, 'observed_thresholds': None,
                'frozen_thresholds': frozen_thresholds}
    if not path.is_file():
        return evidence
    gate = read_json(path)
    require(isinstance(gate, dict), f'{phase} fidelity evidence is not an object: {path}')
    require((gate.get('protocol'), gate.get('arm'), gate.get('variant'),
             gate.get('scope'), gate.get('phase')) ==
            (protocol['source_protocol'], arm, variant, scope, phase),
            f'{phase} fidelity evidence identity differs: {path}')
    observed_thresholds = gate.get('thresholds')
    if observed_thresholds is not None:
        require(observed_thresholds == frozen_thresholds,
                f'{phase} fidelity evidence thresholds differ from the frozen values: {path}')
    gate_binding = gate.get('backend_binding')
    if gate_binding:
        require(base_binding(gate_binding) == binding,
                f'{phase} fidelity evidence belongs to a different measured backend: {path}')
    queries = gate.get('queries')
    query_status_counts = {}
    if isinstance(queries, list):
        for query in queries:
            status = query.get('status', 'unknown') if isinstance(query, dict) else 'malformed'
            query_status_counts[status] = query_status_counts.get(status, 0) + 1
    evidence.update({'status': gate.get('status', 'unknown'),
                     'completed_queries': gate.get('completed_queries'),
                     'expected_queries': gate.get('expected_queries', expected_queries),
                     'recorded_query_count': len(queries) if isinstance(queries, list) else None,
                     'query_status_counts': query_status_counts,
                     'observed_thresholds': observed_thresholds})
    return evidence


def _fidelity_evidence(root: Path, protocol: dict, arm: str, variant: str,
                       scope: str, binding: dict) -> dict:
    return {'policy': 'report_only',
            'assessment': 'raw fidelity evidence; not a performance eligibility gate',
            'phases': {phase: _fidelity_phase_evidence(
                root, protocol, arm, variant, scope, phase, binding)
                for phase in ('representative4', 'full80')}}


def collect(root: Path) -> dict:
    protocol = read_json(root / 'native_protocol.json')
    require(protocol.get('baseline_fidelity_policy') in (None, 'report_only'),
            'Unsupported baseline fidelity policy')
    require(protocol['arms'] == list(ARMS), 'Six frozen arms are required')
    require(protocol.get('selection_status') == 'ready_for_formal_panel', 'Native backend selection is not finalized')
    expected_hw, payload, bindings, rows = None, {}, {}, []
    measured_devices = {}
    for scope in SCOPES:
        for arm, label in zip(ARMS, LABELS):
            variant = protocol.get('selected_variants', {}).get(
                arm, 'optimized' if arm == 'ours_vqwam' else 'current')
            validate_variant(arm, variant)
            (summary_path, summary, sample_count, blocks, repetitions, measurement_design,
             cohort_queries, seed_indices, unused_legacy_path) = \
                _select_summary(root, scope, variant, arm)
            relative = summary_path.relative_to(root)
            prefix = f'{scope}/{arm}'
            require(summary.get('format') == protocol['protocol']
                    and summary.get('protocol') == protocol['source_protocol'], prefix + ': protocol differs')
            require(summary.get('status') == 'complete' and summary.get('complete') is True,
                    prefix + ': measurements incomplete')
            require((summary.get('scope'), summary.get('arm'), summary.get('variant')) ==
                    (scope, arm, variant), prefix + ': arm/scope/variant differs')
            coverage = summary['coverage']
            require(all(coverage.get(key) == value for key, value in
                        {'queries': cohort_queries, 'samples': sample_count,
                         'expected_samples': sample_count,
                         'blocks': blocks, 'repetitions': repetitions}.items()),
                    prefix + ': selected performance measurement design differs')
            require(summary.get('training_updates') == 0 and summary.get('closed_loop_episodes') == 0,
                    prefix + ': unexpected training or closed loop')
            hw = compatible_gpu_key(summary['hardware'])
            require(hw['gpu_name'] == protocol['hardware']['gpu']
                    and hw['capability'] == protocol['hardware']['capability']
                    and hw['torch_version'] == protocol['hardware']['torch']
                    and hw['cuda_version'] == protocol['hardware']['cuda_runtime'],
                    prefix + ': GPU SKU or software differs')
            expected_hw = hw if expected_hw is None else expected_hw
            require(hw == expected_hw, prefix + ': GPU compatibility differs between arms')
            _merge_device_provenance(measured_devices, _summary_devices(summary, sample_count))
            binding = base_binding(summary['backend_binding'])
            if arm in bindings:
                require(binding == bindings[arm], prefix + ': backend differs between scopes')
            bindings[arm] = binding
            if arm not in ('bf16', 'ours_vqwam'):
                require(summary['install_metadata'].get('native_int4_kernel_used') is True
                        and binding.get('native_int4_kernel_used') is True
                        and binding.get('extension_build', {}).get('pbs_job_id'),
                        prefix + ': native extension evidence missing')
            report_only = _baseline_report_only(protocol, arm)
            fidelity = None
            if arm != 'bf16':
                if report_only:
                    fidelity = _fidelity_evidence(root, protocol, arm, variant, scope, binding)
                else:
                    _require_gate(root, arm, variant, scope)
                    gate = read_json(root / 'results/native_gate' / scope / variant / arm / 'full80.json')
                    limits = _gate_limits(scope, arm, variant, full_panel=True)
                    require(base_binding(gate['backend_binding']) == binding,
                            prefix + ': performance differs from validated backend')
                    require(gate.get('thresholds') == limits, prefix + ': frozen gate thresholds differ')
                    queries = gate.get('queries', [])
                    require(len(queries) == 80 and {(q['case_id'], q['seed_index']) for q in queries}
                            == {(case, seed) for case in range(3000, 3040) for seed in (0, 1)},
                            prefix + ': fidelity panel incomplete')
                    require(all(q['status'] == 'passed' and all(
                        math.isfinite(q['metrics'].get(key, float('nan'))) and q['metrics'][key] <= limit
                        for key, limit in limits.items()) for q in queries), prefix + ': fidelity gate failed')
            timing_key = 'denoising_wall_seconds' if scope == 'sampler' else 'dit_replay_seconds'
            timing, peak = summary[timing_key], summary['peak_allocated_gb_decimal']
            require(timing.get('count') == sample_count and peak.get('count') == sample_count,
                    prefix + ': timing/memory count differs')
            require(all(positive(timing[key]) for key in ('mean', 'median', 'p95', 'max'))
                    and positive(peak['max']), prefix + ': invalid measured values')
            storage = summary['storage']
            count = storage['packed_storage_bytes']
            require(isinstance(count, int) and count > 0 and math.isclose(
                count / 2**30, storage['packed_storage_gib_binary'], rel_tol=1e-12),
                prefix + ': invalid tensor storage units')
            require(arm not in payload or payload[arm] == count,
                    prefix + ': packed payload differs between scopes')
            payload[arm] = count
            if scope == 'dit_replay':
                require(summary.get('graph_capture_count') == 5 and summary.get('graph_pool_contract'),
                        prefix + ': five-forward graph scope missing')
                require(isinstance(summary.get('non_dit_offload', {}).get('modules'), list),
                        prefix + ': non-DiT offload receipt missing')
            row = {'scope': scope, 'arm': arm, 'label': label, 'variant': variant,
                         'measurement_design': measurement_design,
                         'cohort': {'queries': cohort_queries,
                                    'case_ids': list(range(3000, 3040)),
                                    'seed_indices': list(seed_indices)},
                         'samples': sample_count,
                         'latency_mean_ms': timing['mean'] * 1000,
                         'latency_median_ms': timing['median'] * 1000,
                         'latency_p95_ms': timing['p95'] * 1000,
                         'peak_allocated_gb_decimal': peak['max'],
                         'packed_storage_gib_binary': count / 2**30,
                         'packed_storage_bytes': count,
                         'summary': str(relative), 'pbs_job_id': summary['hardware'].get('pbs_job_id')}
            if unused_legacy_path is not None:
                row['unused_incomplete_legacy_summary'] = unused_legacy_path
            if fidelity is not None:
                row['fidelity_evidence'] = fidelity
            rows.append(row)
    for scope in SCOPES:
        reference = next(row for row in rows if row['scope'] == scope and row['arm'] == 'bf16')
        for row in rows:
            if row['scope'] == scope:
                row['speedup_mean'] = reference['latency_mean_ms'] / row['latency_mean_ms']
                row['speedup_median'] = reference['latency_median_ms'] / row['latency_median_ms']
    hardware_report = dict(expected_hw or {})
    device_rows = [{'gpu_uuid': uuid, 'hostnames': sorted(device['hostnames']),
                    'pbs_job_ids': sorted(device['pbs_job_ids'])}
                   for uuid, device in sorted(measured_devices.items())]
    hardware_report.update(measured_device_count=len(device_rows),
                           measured_gpu_uuids=[row['gpu_uuid'] for row in device_rows],
                           device_provenance=device_rows)
    fidelity_policy = protocol.get('baseline_fidelity_policy', 'strict')
    measurement_designs = sorted({row['measurement_design'] for row in rows})
    return {'format': 'cosmos_native_performance_mixed_aggregate_v1', 'status': 'complete',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'aggregate_pbs_job_id': os.environ.get('PBS_JOBID'), 'hardware': hardware_report,
            'cohort_note': 'Rows use either the existing 80-query/two-seed cohort or the new 40-query/seed-0 cohort; compare rows with the recorded cohort and N.',
            'measurement_designs': measurement_designs,
            'measurement_design_by_scope_arm': {f"{row['scope']}/{row['arm']}": {
                'design': row['measurement_design'], 'samples': row['samples']}
                for row in rows},
            'rows': rows,
            'baseline_fidelity_policy': fidelity_policy,
            'completion_scope': 'performance',
            'fidelity_full80_design_unchanged': True,
            'new_training_updates': 0, 'new_closed_loop_episodes': 0,
            'boundary': 'Native execution fidelity and performance only; existing closed-loop scores are unchanged checkpoint/reference evidence.'}


def render(report: dict) -> str:
    lines = ['# Cosmos Policy native performance (mixed measurement designs)', '',
             (f"兼容硬件：`{report['hardware']['gpu_name']}` / SM{report['hardware']['capability'][0]}{report['hardware']['capability'][1]}，"
              f"Torch `{report['hardware']['torch_version']}` / CUDA `{report['hardware']['cuda_version']}`；"
              f"实测设备 {report['hardware']['measured_device_count']} 台，UUID："
              f"{', '.join(f'`{uuid}`' for uuid in report['hardware']['measured_gpu_uuids'])}。"
              "每个输入仍执行 5 NFE。各行 cohort 可能是已有 40 cases × seeds 0/1 的 80-query legacy 设计，也可能是新 40 cases × seed 0 的 40-query 设计；请按表格中的 Cohort、Seeds、Design 和 N 比较，不应把它们当作同一输入 cohort。Fidelity full80 是独立 gate，仍按原 80-query panel 判定。"), '',
             'Lat. 为 mean；Spd. 为同口径 BF16 mean / method mean；Peak 为实测最大 allocated GB（10^9）；Storage 为 whole-DiT unique tensor payload GiB（2^30），包含 padding/scales/transforms，排除 workspace。', '']
    descriptions = {'sampler': '真实五次 NFE sampler wall time；Peak 包含 resident tokenizer/VAE 和 endpoint 临时内存。',
                    'dit_replay': '冻结 BF16 网络输入的五次 DiT CUDA Graph replay，CUDA event 计时；非 DiT 模块已 offload。'}
    for scope in SCOPES:
        lines += [f'## {scope}', '', descriptions[scope], '',
                  '| Method | Cohort q | Seeds | Design | N | Lat. ms | Spd. | p50 ms | p95 ms | Peak GB | Storage GiB |',
                  '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in report['rows']:
            if row['scope'] == scope:
                seeds = ','.join(str(value) for value in row['cohort']['seed_indices'])
                lines.append(f"| {row['label']} | {row['cohort']['queries']} | {seeds} | {row['measurement_design']} | {row['samples']} | {row['latency_mean_ms']:.3f} | {row['speedup_mean']:.3f}× | {row['latency_median_ms']:.3f} | {row['latency_p95_ms']:.3f} | {row['peak_allocated_gb_decimal']:.4f} | {row['packed_storage_gib_binary']:.4f} |")
        lines.append('')
    if report.get('baseline_fidelity_policy') == 'report_only':
        lines += ['## Fidelity evidence (report-only)', '',
                  '按用户要求，四个 native baseline 的 fidelity 误差不阻断性能聚合。下表的 status、query 数和 thresholds 是已有 gate receipt 的原始证据；它们不构成本报告的性能资格判定。性能行的 cohort 和 N 均单独标出；独立 fidelity full80 仍需 80 个 gate queries，性能 timing samples 不能证明 fidelity 通过或 full80 已完成。', '',
                  '| Scope | Method | Phase | Raw status | Completed / expected | Recorded queries | Frozen thresholds | Receipt |',
                  '|---|---|---|---|---:|---:|---|---|']
        for row in report['rows']:
            evidence = row.get('fidelity_evidence')
            if evidence is None:
                continue
            for phase in ('representative4', 'full80'):
                item = evidence['phases'][phase]
                completed = item.get('completed_queries')
                expected = item.get('expected_queries')
                completed_text = f"{completed if completed is not None else 'unknown'} / {expected}"
                thresholds = json.dumps(item['frozen_thresholds'], ensure_ascii=False,
                                        sort_keys=True, separators=(',', ':'))
                lines.append(f"| {row['scope']} | {row['label']} | {phase} | {item['status']} | {completed_text} | {item.get('recorded_query_count') if item.get('recorded_query_count') is not None else 'unknown'} | `{thresholds}` | `{item['path']}` |")
        lines.append('')
    lines += ['本轮 new training updates = 0；new closed-loop episodes = 0。Native backend 的闭环表现尚未重测，旧闭环结果只属于此前 reference/checkpoint 执行路径。', '']
    return '\n'.join(lines)


def render_html(report: dict) -> str:
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8">',
             '<title>Cosmos Policy native performance (mixed designs)</title>',
             '<style>body{font:15px system-ui,sans-serif;margin:2rem;color:#17202a}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}th,td{border:1px solid #ccd1d1;padding:.45rem;text-align:left}th{background:#f2f4f4}code{font-size:.9em}</style>',
             '<h1>Cosmos Policy native performance (mixed measurement designs)</h1>',
             f"<p>Hardware: {html.escape(str(report['hardware']['gpu_name']))}, SM{report['hardware']['capability'][0]}{report['hardware']['capability'][1]}; Torch {html.escape(str(report['hardware']['torch_version']))}; CUDA {html.escape(str(report['hardware']['cuda_version']))}. Each input uses five NFE. Rows may use the existing 40-case × seeds 0/1 80-query cohort or the new 40-case × seed 0 40-query cohort; cohort, design, and N are shown per row and are not treated as the same input cohort. The independent fidelity full80 gate remains an 80-query panel.</p>",
             '<table><thead><tr><th>Scope</th><th>Method</th><th>Cohort q</th><th>Seeds</th><th>Design</th><th>N</th><th>Mean ms</th><th>Speedup</th><th>p50 ms</th><th>p95 ms</th><th>Peak GB</th><th>Storage GiB</th></tr></thead><tbody>']
    for row in report['rows']:
        parts.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            row['scope'], row['label'], str(row['cohort']['queries']),
            ','.join(str(value) for value in row['cohort']['seed_indices']),
            row['measurement_design'], str(row['samples']),
            f"{row['latency_mean_ms']:.3f}",
            f"{row['speedup_mean']:.3f}×", f"{row['latency_median_ms']:.3f}",
            f"{row['latency_p95_ms']:.3f}", f"{row['peak_allocated_gb_decimal']:.4f}",
            f"{row['packed_storage_gib_binary']:.4f}")) + '</tr>')
    parts.append('</tbody></table>')
    if report.get('baseline_fidelity_policy') == 'report_only':
        note = ('按用户要求，四个 native baseline 的 fidelity 误差不阻断性能聚合。下表保留已有 gate receipt 的原始状态、query 数和 thresholds，仅作为 fidelity evidence；它们不构成本报告的性能资格判定。性能行的 cohort 和 N 均单独标出；独立 fidelity full80 仍为 80 个 gate queries，不能据此声称 fidelity 通过或 full80 已完成。')
        parts += [f'<h2>Fidelity evidence (report-only)</h2><p>{html.escape(note)}</p>',
                  '<table><thead><tr><th>Scope</th><th>Method</th><th>Phase</th><th>Raw status</th><th>Completed / expected</th><th>Recorded queries</th><th>Frozen thresholds</th><th>Receipt</th></tr></thead><tbody>']
        for row in report['rows']:
            evidence = row.get('fidelity_evidence')
            if evidence is None:
                continue
            for phase in ('representative4', 'full80'):
                item = evidence['phases'][phase]
                completed = item.get('completed_queries')
                values = (row['scope'], row['label'], phase, item['status'],
                          f"{completed if completed is not None else 'unknown'} / {item['expected_queries']}",
                          item.get('recorded_query_count') if item.get('recorded_query_count') is not None else 'unknown',
                          json.dumps(item['frozen_thresholds'], ensure_ascii=False, sort_keys=True),
                          item['path'])
                parts.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>'
                                              for value in values) + '</tr>')
        parts.append('</tbody></table>')
    parts.append('<p>New training updates: 0. New closed-loop episodes: 0. Native backend closed-loop performance has not been remeasured; existing closed-loop scores remain reference/checkpoint evidence.</p></html>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    from prepare_sources import guard
    guard()
    report = collect(args.root)
    (args.root / 'native_performance_summary_mixed.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (args.root / 'RESULTS_mixed.zh.md').write_text(render(report), encoding='utf-8')
    (args.root / 'RESULTS_mixed.html').write_text(render_html(report), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'rows': len(report['rows']),
                      'measured_device_count': report['hardware']['measured_device_count'],
                      'measured_gpu_uuids': report['hardware']['measured_gpu_uuids']}))


if __name__ == '__main__':
    main()
