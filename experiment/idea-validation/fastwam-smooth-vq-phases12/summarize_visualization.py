"""Small measured tables for the visualization report; PBS CPU allocation only."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path


def main():
    import run_validation
    run_validation.guard()
    import numpy as np
    from plot_visualization import CASES, CONFIGS, WEIGHT_CONFIGS, aggregate_energy, jsonl, read_json, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    collector = read_json(args.data / 'summary.json')
    plots = read_json(args.out / 'summary.json')
    assert collector['status'] == plots['status'] == 'complete'
    assert collector['bf16_queries_completed'] == 3
    assert collector['observed_module_coverage_per_query'] == [614, 614, 614]
    target = args.out / 'report_metrics.json'
    if target.exists():
        raise FileExistsError(target)
    activations = jsonl(args.data / 'activations.jsonl')
    weights = jsonl(args.data / 'weights.jsonl')
    selected_ids = {row['module_index'] for row in collector['selected_modules']}
    selected = [row for row in weights if row['module_index'] in selected_ids]
    grouped = aggregate_energy(activations, ('case_id', 'stream', 'config'))
    activation_tables = []
    for (case_id, stream, config), bucket in sorted(grouped.items()):
        elements = sum(row['elements'] for row in activations
                       if (row['case_id'], row['stream'], row['config']) == (case_id, stream, config))
        activation_tables.append({'case_id': case_id, 'stream': stream, 'config': config,
            'rrmse': bucket['rrmse'], 'zero_fraction': bucket['zero_count'] / elements,
            'signal_energy': bucket['signal_squared'], 'error_energy': bucket['error_squared'],
            'calls': bucket['calls'], 'elements': elements})
    assert len(activation_tables) == 36

    activation_peaks = []
    peak_fields = ('case_id', 'config', 'module', 'module_index', 'stage', 'step', 'call_index',
                   'activation_rrmse', 'error_absmax', 'error_peak_position', 'absmax', 'rms', 'outlier_ratio')
    for config in CONFIGS:
        rows = [row for row in activations if row['config'] == config and row['step'] >= 0]
        for metric in ('activation_rrmse', 'error_absmax'):
            for rank, row in enumerate(sorted(rows, key=lambda item: item[metric], reverse=True)[:3], 1):
                activation_peaks.append({'ranking_metric': metric, 'rank': rank,
                                         **{key: row[key] for key in peak_fields}})
    example_module = plots['counts']['activation_example_module_index']
    examples = [{key: row[key] for key in (*peak_fields, 'zero_fraction', 'input_shape', 'snapshot_file')}
                for row in activations if row['module_index'] == example_module and 'snapshot_file' in row
                and row['step'] == {1: 4, 13: 9, 21: 9}[row['case_id']]]
    assert len(examples) == 12

    selected_weights = []
    for row in selected:
        record = {key: row[key] for key in ('config', 'module', 'module_index', 'stream', 'shape', 'weight_rms',
            'scalar_rrmse', 'vq_rrmse', 'ratio', 'scalar_peak_error', 'vq_peak_error',
            'scalar_peak_position', 'vq_peak_position')}
        path = args.data / 'snapshots' / f"weight_{row['module_index']:04d}_{row['config']}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            for method in ('scalar', 'vq'):
                energy = np.asarray(arrays[f'{method}_column_energy'], dtype=np.float64)
                total = float(energy.sum())
                order = np.argsort(energy)[::-1]
                count = max(1, math.ceil(len(energy) * .01))
                record[f'{method}_column_top1pct_count'] = count
                record[f'{method}_column_top1pct_share'] = float(energy[order[:count]].sum() / total) if total else None
                record[f'{method}_column_top16_share'] = float(energy[order[:16]].sum() / total) if total else None
                record[f'{method}_column_top4_indices'] = [int(index) for index in order[:4]]
        selected_weights.append(record)

    energy_names = ('reference_output_energy', 'weight_error_energy', 'activation_error_energy',
                    'cross_error_energy', 'total_error_energy')
    local = {}
    for row in activations:
        if row['config'] not in WEIGHT_CONFIGS:
            continue
        for method in ('scalar', 'vq'):
            key = row['config'], row['module_index'], method
            bucket = local.setdefault(key, {'calls': 0, **dict.fromkeys(energy_names, 0.0)})
            for name in energy_names:
                bucket[name] += row['local_output'][method][name]
            bucket['calls'] += 1
    for bucket in local.values():
        assert bucket['reference_output_energy'] > 0
        for name in energy_names[1:]:
            bucket[name.replace('_error_energy', '_rrmse')] = math.sqrt(bucket[name] / bucket['reference_output_energy'])
    assert len(local) == 2456
    local_by_stream = []
    selected_local = []
    top_local = []
    for config in WEIGHT_CONFIGS:
        module_rows = [row for row in weights if row['config'] == config]
        for stream in ('video', 'action', 'proprio'):
            indices = [row['module_index'] for row in module_rows if row['stream'] == stream]
            ratios = [local[config, index, 'vq']['total_rrmse'] / local[config, index, 'scalar']['total_rrmse']
                      for index in indices]
            local_by_stream.append({'config': config, 'stream': stream, 'layers': len(indices),
                'vq_lower_count': sum(ratio < 1 for ratio in ratios),
                'vq_higher_count': sum(ratio > 1 for ratio in ratios),
                'median_vq_over_scalar_total_rrmse': float(np.median(ratios))})
        for row in module_rows:
            if row['module_index'] in selected_ids:
                for method in ('scalar', 'vq'):
                    selected_local.append({'config': config, 'module': row['module'], 'module_index': row['module_index'],
                        'stream': row['stream'], 'method': method, **local[config, row['module_index'], method]})
        for method in ('scalar', 'vq'):
            ordered = sorted(module_rows, key=lambda row: local[config, row['module_index'], method]['total_rrmse'], reverse=True)
            for rank, row in enumerate(ordered[:5], 1):
                top_local.append({'rank': rank, 'config': config, 'method': method,
                    'module': row['module'], 'module_index': row['module_index'], 'stream': row['stream'],
                    'paired_scalar_total_rrmse': local[config, row['module_index'], 'scalar']['total_rrmse'],
                    'paired_vq_total_rrmse': local[config, row['module_index'], 'vq']['total_rrmse'],
                    'paired_scalar_weight_rrmse': local[config, row['module_index'], 'scalar']['weight_rrmse'],
                    'paired_vq_weight_rrmse': local[config, row['module_index'], 'vq']['weight_rrmse'],
                    **local[config, row['module_index'], method]})

    action_map = {(row['case_id'], row['seed_index'], row['arm']): np.asarray(row['action'], dtype=np.float64)
                  for row in jsonl(args.full / 'actions.jsonl')}
    action_tables = []
    for case_id in CASES:
        reference = action_map[case_id, 0, 'bf16']
        for config in WEIGHT_CONFIGS:
            for method in ('scalar', 'vq'):
                delta = action_map[case_id, 0, f'{config}_{method}_a4'] - reference
                assert delta.shape == (32, 7)
                primary = np.abs(delta[:10, :6])
                all_motor = np.abs(delta[:, :6])
                action_tables.append({'case_id': case_id, 'seed_index': 0, 'config': config, 'method': method,
                    'motor_rmse_prefix10': float(np.sqrt(np.mean(delta[:10, :6] ** 2))),
                    'gripper_rmse_prefix10': float(np.sqrt(np.mean(delta[:10, 6] ** 2))),
                    'motor_rmse_full32': float(np.sqrt(np.mean(delta[:, :6] ** 2))),
                    'primary_motor_peak_abs': float(primary.max()),
                    'primary_motor_peak_position': list(map(int, np.unravel_index(np.argmax(primary), primary.shape))),
                    'full32_motor_peak_abs': float(all_motor.max()),
                    'full32_motor_peak_position': list(map(int, np.unravel_index(np.argmax(all_motor), all_motor.shape)))})
    result = {'status': 'complete', 'pbs_jobid': os.environ['PBS_JOBID'],
        'sources': plots['source_pbsids'],
        'scope': 'Cases 1/13/21, seed 0, BF16 teacher trajectory; exploratory; no restoration/refit/episodes',
        'activation_table_scope': 'All callbacks including conditioning; energy aggregation within each case/stream/config',
        'activation_by_case_stream_config': activation_tables,
        'activation_top3_per_config': activation_peaks, 'activation_example_metrics': examples,
        'selected_weight_metrics': selected_weights,
        'weight_column_metric_scope': 'Weight-error Frobenius energy summed over output rows in transformed input coordinates',
        'local_output_comparison_by_stream': local_by_stream, 'selected_local_output': selected_local,
        'local_output_top5': top_local,
        'local_output_scope': 'Same deterministic <=16 BF16 rows per call, FP32 projection, conditioning included; not final action sensitivity',
        'saved_action_metrics': action_tables,
        'action_position_space': 'zero-based predicted action timestep, motor coordinate 0..5; not denoising step'}
    write_json(target, result)
    assert target.stat().st_size <= 256 * 1024, 'Report table exceeds small artifact limit'
    print(f'VISUALIZATION_REPORT_TABLES_COMPLETE {target.stat().st_size} bytes', flush=True)


if __name__ == '__main__':
    main()
