"""Frozen two-arm numerical experiment, resumable between <=2h PBS allocations."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def tensor_save(torch, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def aggregate(records):
    grouped = {}
    metrics = ('motor_rmse_first10_vs_bf16', 'gripper_rmse_first10_vs_bf16', 'motor_rmse_32_vs_bf16')
    for record in records:
        grouped.setdefault(record['case_id'], []).append(record)
    values = {str(case): {metric: statistics.mean(row[metric] for row in rows) for metric in metrics}
              for case, rows in grouped.items()}
    return {'case_count': len(grouped), 'context_count': len(records), 'per_case': values,
            **{metric: statistics.mean(row[metric] for row in values.values()) for metric in metrics}}


def run(args):
    import run_experiment as original
    original.guard()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import codebook
    import micro_codebook as micro
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    if p['protocol'] != 'fastwam-vq-microscale-v1' or p['walltime'] != '02:00:00':
        raise ValueError('Unexpected frozen protocol/walltime')
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != p:
        raise RuntimeError('Cannot resume with a changed experimental protocol')
    save(protocol_path, p)
    artifact = Path(os.environ['ARTIFACTS'])
    job = os.environ['PBS_JOBID']
    started = time.monotonic()
    elapsed_at_start = time.time() - float(os.environ['FW_JOB_START_EPOCH'])
    deadline = started + p['soft_deadline_seconds'] - elapsed_at_start
    source_root = Path(p['source_root'])
    source_dir = source_root / 'results/full'
    old_inputs = Path(p['roots']['old_inputs'])
    source_summary = json.loads((source_dir / 'summary.json').read_text())
    if source_summary['status'] != 'complete' or source_summary['pbs_jobid'] != p['source_full_job']:
        raise RuntimeError('Source campaign identity/completion differs')
    old_plan = json.loads((old_inputs / 'plan.json').read_text())
    fresh = {row['case_id']: row for row in json.loads((source_root / 'plan.json').read_text())['inputs']}
    cached = {}
    for line in (source_dir / 'actions.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        if row['arm'] not in ('bf16', 'bf16w_a4', p['control_arm'], 'scalar_w4a4'):
            continue
        key = (row['arm'], row['case_id'], row['seed_index'])
        if key in cached:
            raise RuntimeError(f'Duplicate source action {key}')
        cached[key] = row

    def manifest(case):
        return source.validate_plan(old_plan, case) if case < 1000 else fresh[case]

    def cached_action(arm, case, seed):
        row = cached[arm, case, seed]
        if row['sampler_seed'] != manifest(case)['sampler_seeds'][seed]:
            raise RuntimeError(f'Paired sampler identity differs: {arm} {case} {seed}')
        action = torch.tensor(row['action'], dtype=torch.float32)
        if action.shape != (32, 7) or not bool(torch.isfinite(action).all()):
            raise RuntimeError('Invalid cached action')
        return action

    for case in p['selection_cases'] + p['test_cases']:
        for seed in p['evaluation_seed_indices']:
            for arm in ('bf16', p['control_arm'], 'scalar_w4a4'):
                cached_action(arm, case, seed)
    for case in p['calibration_cases']:
        cached_action('bf16w_a4', case, 0)

    query_dir = args.out / 'query_records'
    bank_dir = args.out / 'banks' / p['new_arm']
    query_dir.mkdir(exist_ok=True)
    bank_dir.mkdir(parents=True, exist_ok=True)
    stage, records, query_records, runtime, banks = 'self_check', [], [], None, None
    moment_completed = []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    captures, data = None, {}
    recent_fit_seconds = []

    def progress(**extra):
        row = {'status': 'running', 'phase': 'full', 'job_id': job, 'stage': stage,
               'fitted_linears': len(records), 'target_linears': 614, 'new_queries': len(query_records),
               'moment_completed_contexts': len(moment_completed),
               'elapsed_job_seconds': time.monotonic() - started + elapsed_at_start, **extra}
        save(args.out / 'progress.json', row)
        print('MICROSCALE_PROGRESS ' + json.dumps(row), flush=True)

    def checkpoint():
        summary = {'status': 'checkpointed', 'phase': 'full', 'protocol': p['protocol'], 'job_id': job,
                   'stage': stage, 'target_linears': 614, 'fitted_linears': len(records),
                   'new_queries': len(query_records), 'moment_completed_contexts': len(moment_completed),
                   'walltime_limit': p['walltime'], 'soft_deadline_seconds': p['soft_deadline_seconds'],
                   'source_full_job': p['source_full_job'], 'requires_resume': True}
        save(args.out / 'checkpoint.json', summary)
        save(artifact / 'summary.json', summary)
        print('MICROSCALE_CHECKPOINT ' + json.dumps(summary), flush=True)

    def enough_time(reserve=300):
        return deadline - time.monotonic() > reserve

    def datum(case):
        if case not in data:
            row = manifest(case)
            observation, metadata = source.load_case(old_inputs if case < 1000 else source_root, row)
            data[case] = (runtime.datum(observation), metadata['description'])
        return data[case]

    def infer(case, seed, arm):
        value, description = datum(case)
        runtime.description, runtime.current_arm = description, arm
        banks.seen.clear()
        action, _ = runtime.infer(value, manifest(case)['sampler_seeds'][seed], measure=False)
        if len(banks.seen) != 614:
            raise RuntimeError(f'Only {len(banks.seen)}/614 targets called')
        return action.detach().float().cpu()

    def diagnostics(name, weight, dense, old_dense, moment):
        nonlocal captures
        if captures is None:
            captures = [torch.load(source_dir / 'sensitivity' / f'case_{case}.pt',
                                   map_location='cpu', weights_only=True, mmap=True)
                        for case in p['calibration_cases']]
        xs, ys, factors = [], [], []
        for capture in captures:
            calls = capture['layers'].get(name, [])
            if not calls:
                raise RuntimeError(f'Missing frozen teacher calls for {name}')
            for call in calls:
                inputs = call['inputs'].to('cuda')
                xs.append(original.qinput(torch, banks.core, inputs, banks.transforms[name]).float())
                ys.append(call['teacher_y'].to('cuda').float())
                factors.append(torch.full((inputs.shape[0],), call['total_rows'] / inputs.shape[0], device='cuda'))
        x, y, factor = torch.cat(xs), torch.cat(ys), torch.cat(factors)
        bias = banks.modules[name].bias
        bias = None if bias is None else bias.detach().float()
        result = {}
        denominator = factor.sum() * y.shape[1]
        for label, candidate in (('control', old_dense), ('micro', dense)):
            delta = candidate.float() - weight.float()
            result[label + '_weight_mse'] = float(delta.square().mean())
            result[label + '_moment_weighted_weight_mse'] = float((delta.square() * moment[None]).mean())
            error = torch.nn.functional.linear(x, candidate.float(), bias) - y
            result[label + '_local_output_mse'] = float((error.square().sum(1) * factor).sum() / denominator)
        return result

    try:
        micro.self_check()
        stage = 'model_loading'
        progress()
        runtime = pilot.Runtime(artifact)
        runtime.arm('bf16')
        if 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Frozen comparison requires the existing A100 hardware class')
        runtime_record = json.loads((artifact / 'runtime.json').read_text())
        runtime_record.update(protocol=p['protocol'], real_quant=False, native_low_bit_kernel=False,
                              execution='B32 exponent-scaled BF16 additive VQ; decoded BF16 Linear')
        save(artifact / 'runtime.json', runtime_record)
        banks = original.FrozenBanks(runtime, args.out, p)

        stage = 'prepare_original_q4_input_moments'
        moment_path = args.out / 'moments.pt'
        if moment_path.exists():
            moment_cache = torch.load(moment_path, map_location='cpu', weights_only=True)
            if moment_cache['source_full_job'] != p['source_full_job']:
                raise RuntimeError('Moment source differs')
            moment_completed = moment_cache['completed_cases']
            if moment_completed != p['calibration_cases'][:len(moment_completed)]:
                raise RuntimeError('Moment calibration order differs')
            banks.moments = {name: value.to('cuda') for name, value in moment_cache['sums'].items()}
            banks.moment_rows = dict(moment_cache['rows'])
            replay = list(moment_cache['replay'])
        else:
            replay = []
        if len(moment_completed) < len(p['calibration_cases']):
            banks.activate('bf16w_a4')
            banks.collect_moment = True
            for case in p['calibration_cases'][len(moment_completed):]:
                if not enough_time():
                    checkpoint()
                    return
                progress(case_id=case)
                action = infer(case, 0, 'bf16w_a4')
                error = float((action - cached_action('bf16w_a4', case, 0)).abs().max())
                if error > p['evaluation']['calibration_replay_max_abs_tolerance']:
                    raise RuntimeError(f'Original A4 calibration replay differs: case={case} max_abs={error}')
                replay.append({'case_id': case, 'max_abs_vs_cached_original': error})
                moment_completed.append(case)
                tensor_save(torch, moment_path, {'source_full_job': p['source_full_job'],
                            'completed_cases': list(moment_completed), 'replay': replay,
                            'sums': {name: value.detach().cpu() for name, value in banks.moments.items()},
                            'rows': banks.moment_rows})
            banks.collect_moment = False
        if len(banks.moments) != 614 or len(banks.moment_rows) != 614:
            raise RuntimeError('Original input moment coverage is incomplete')
        moments = {name: value.detach().cpu() / banks.moment_rows[name] for name, value in banks.moments.items()}
        banks.moments.clear()
        save(args.out / 'moment_preparation.json', {'status': 'complete', 'source_full_job': p['source_full_job'],
             'case_ids': moment_completed, 'contexts': 8, 'target_linears': len(moments), 'replay': replay,
             'teacher_or_sensitivity_recaptured': False})

        stage = 'fit_micro_banks'
        settings = p['vq']
        with torch.no_grad():
            for index, (name, module) in enumerate(banks.modules.items()):
                path = bank_dir / f'{index:04d}.pt'
                if path.exists():
                    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                    if saved['module'] != name or tuple(saved['encoded']['shape']) != tuple(module.weight.shape):
                        raise RuntimeError(f'Resumed bank identity differs: {index} {name}')
                    if saved['encoded']['format'] != micro.FORMAT:
                        raise RuntimeError('Resumed bank format differs')
                    records.append(saved['fit_record'])
                    del saved
                    continue
                reserve = max(300, 2 * max(recent_fit_seconds[-3:], default=0))
                if not enough_time(reserve):
                    checkpoint()
                    return
                progress(module=name, module_index=index)
                fit_started = time.monotonic()
                weight = banks.dense_teacher_weight(name)
                old_bank = torch.load(source_dir / 'banks' / p['control_arm'] / f'{index:04d}.pt',
                                      map_location='cuda', weights_only=True)
                if old_bank['module'] != name or tuple(old_bank['encoded']['shape']) != tuple(weight.shape):
                    raise RuntimeError(f'Control bank identity differs: {name}')
                for key, value in banks.transforms[name].items():
                    prior = old_bank['transform'][key]
                    if (not torch.equal(value, prior) if torch.is_tensor(value) else value != prior):
                        raise RuntimeError(f'Frozen transform differs: {name} {key}')
                moment = moments[name].to('cuda')
                encoded = micro.fit_vq(weight, second_moment=moment, seed=settings['seed'],
                                       sample_limit=settings['samples'], iterations=settings['iterations'])
                if not torch.equal(encoded['row_scales'], old_bank['encoded']['row_scales']):
                    raise RuntimeError(f'Existing row gain differs: {name}')
                dense, old_dense = micro.decode_vq(encoded), codebook.decode_vq(old_bank['encoded'])
                stats = diagnostics(name, weight, dense, old_dense, moment)
                exponent = encoded['scale_exponents'].to(torch.int32) - 127
                transform = original.cpu_encoding(torch, banks.transforms[name])
                transform_bytes = sum(value.numel() * value.element_size() for value in transform.values() if torch.is_tensor(value))
                record = {'module': name, 'module_index': index, 'weights': weight.numel(),
                          'encoded_bytes': micro.encoded_nbytes(encoded),
                          'control_encoded_bytes': codebook.encoded_nbytes(old_bank['encoded']),
                          'transform_bytes': transform_bytes, 'exponent_bytes': exponent.numel(),
                          'nonzero_exponents': int((exponent != 0).sum()),
                          'exponent_min': int(exponent.min()), 'exponent_max': int(exponent.max()),
                          'clipped_exponents': encoded['fit_stats']['clipped_exponents'], **stats}
                record['fit_wall_seconds'] = time.monotonic() - fit_started
                tensor_save(torch, path, {'module': name, 'transform': transform,
                                         'encoded': original.cpu_encoding(torch, encoded), 'fit_record': record})
                records.append(record)
                recent_fit_seconds.append(record['fit_wall_seconds'])
                progress(last_fit_seconds=record['fit_wall_seconds'])
                del weight, encoded, dense, old_dense, old_bank, moment
        if len(records) != 614:
            raise RuntimeError('Full bank fitting coverage differs')
        captures = None
        with (args.out / 'module_fit.jsonl').open('w', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        freeze_path = args.out / 'bank_freeze.json'
        if not freeze_path.exists():
            save(freeze_path, {'fitted_linears': 614, 'protocol': p['protocol'], 'job_id': job,
                              'settings': settings, 'method_selection': 'none; predetermined B32'})
        elif json.loads(freeze_path.read_text())['fitted_linears'] != 614:
            raise RuntimeError('Invalid bank freeze')

        stage = 'evaluate_micro'
        for split in ('selection', 'test'):
            for case in p[split + '_cases']:
                for seed in p['evaluation_seed_indices']:
                    path = query_dir / f'{split}_{case}_{seed}.json'
                    if path.exists():
                        row = json.loads(path.read_text())
                        if (row['case_id'], row['seed_index'], row['sampler_seed'], row['split'], row['arm']) != (
                                case, seed, manifest(case)['sampler_seeds'][seed], split, p['new_arm']):
                            raise RuntimeError('Resumed query identity differs')
                        query_records.append(row)
        if len(query_records) < 52:
            if not enough_time():
                checkpoint()
                return
            with torch.no_grad():
                for index, (name, module) in enumerate(banks.modules.items()):
                    bank = torch.load(bank_dir / f'{index:04d}.pt', map_location='cuda', weights_only=True)
                    if bank['module'] != name:
                        raise RuntimeError('Evaluation bank identity differs')
                    module.weight.data = micro.decode_vq(bank['encoded'])
                    del bank
            banks.transformed = True
            done = {(row['split'], row['case_id'], row['seed_index']) for row in query_records}
            for split in ('selection', 'test'):
                for case in p[split + '_cases']:
                    for seed in p['evaluation_seed_indices']:
                        if (split, case, seed) in done:
                            continue
                        if not enough_time():
                            checkpoint()
                            return
                        progress(split=split, case_id=case, seed_index=seed)
                        action = infer(case, seed, p['new_arm'])
                        row = {'split': split, 'case_id': case, 'seed_index': seed,
                               'sampler_seed': manifest(case)['sampler_seeds'][seed], 'arm': p['new_arm'],
                               'domain': manifest(case)['domain'], 'dimension': manifest(case)['dimension'],
                               'observed_module_count': len(banks.seen), 'bank_frozen_before_query': True,
                               'episodes': 0, 'predicted_actions_executed': 0, 'action': action.tolist(),
                               **source.action_metrics(torch, action, cached_action('bf16', case, seed))}
                        save(query_dir / f'{split}_{case}_{seed}.json', row)
                        query_records.append(row)
        if len(query_records) != 52:
            raise RuntimeError('New-arm query coverage differs')
        primary = 'motor_rmse_first10_vs_bf16'
        results = {}
        for split in ('selection', 'test'):
            new = aggregate([row for row in query_records if row['split'] == split])
            baseline_rows = []
            scalar_rows = []
            for case in p[split + '_cases']:
                for seed in p['evaluation_seed_indices']:
                    reference = cached_action('bf16', case, seed)
                    baseline_rows.append({'case_id': case, **source.action_metrics(torch, cached_action(p['control_arm'], case, seed), reference)})
                    scalar_rows.append({'case_id': case, **source.action_metrics(torch, cached_action('scalar_w4a4', case, seed), reference)})
            control = aggregate(baseline_rows)
            paired = {case: new['per_case'][case][primary] - control['per_case'][case][primary]
                      for case in new['per_case']}
            results[split] = dict(new, control=control, historical_scalar=aggregate(scalar_rows),
                                 paired_case_differences_micro_minus_control=paired,
                                 improved_cases=sum(value < 0 for value in paired.values()),
                                 relative_motor_improvement=1 - new[primary] / control[primary])
        if (results['selection']['case_count'], results['selection']['context_count'],
                results['test']['case_count'], results['test']['context_count']) != (4, 8, 22, 44):
            raise RuntimeError('Final paired evaluation denominator differs')
        weight_count = sum(row['weights'] for row in records)
        new_bytes = sum(row['encoded_bytes'] + row['transform_bytes'] for row in records)
        old_bytes = sum(row['control_encoded_bytes'] + row['transform_bytes'] for row in records)
        diagnostics_summary = {}
        for key in ('weight_mse', 'moment_weighted_weight_mse', 'local_output_mse'):
            ratios = [row['micro_' + key] / max(row['control_' + key], 1e-12) for row in records]
            diagnostics_summary[key] = {'mean_per_module_ratio_micro_over_control': statistics.mean(ratios),
                'median_per_module_ratio': statistics.median(ratios),
                'improved_modules': sum(row['micro_' + key] < row['control_' + key] for row in records)}
        gate = p['evaluation']['engineering_gate']
        summary = {'status': 'complete', 'phase': 'full', 'protocol': p['protocol'], 'job_id': job,
                   'source_full_job': p['source_full_job'], 'target_linears': 614, 'fitted_linears': len(records),
                   'new_queries': len(query_records), 'moment_preparation_queries': len(moment_completed),
                   'self_check_passed': True, 'bank_frozen_before_evaluation': True, 'test_method_selection': False,
                   'walltime_limit': p['walltime'], 'soft_deadline_seconds': p['soft_deadline_seconds'],
                   'selection': results['selection'], 'test': results['test'],
                   'storage': {'target_weight_count': weight_count, 'micro_payload_bytes_with_transforms': new_bytes,
                               'control_payload_bytes_with_transforms': old_bytes,
                               'micro_effective_bpw': new_bytes * 8 / weight_count,
                               'control_effective_bpw': old_bytes * 8 / weight_count,
                               'exponent_bytes': sum(row['exponent_bytes'] for row in records)},
                   'exponents': {'min': min(row['exponent_min'] for row in records),
                                 'max': max(row['exponent_max'] for row in records),
                                 'nonzero_count': sum(row['nonzero_exponents'] for row in records),
                                 'clipped_count': sum(row['clipped_exponents'] for row in records)},
                   'local_diagnostics': diagnostics_summary,
                   'engineering_gate_passed': results['test']['relative_motor_improvement'] >= gate['relative_test_motor_improvement']
                       and results['test']['improved_cases'] >= gate['minimum_improved_cases'],
                   'claim_scope': p['claim_scope'], 'reuse_boundary': p['evaluation']['reuse_boundary'],
                   'episodes': 0, 'predicted_actions_executed': 0}
        with (args.out / 'actions.jsonl').open('w', encoding='utf-8') as actions, (args.out / 'queries.jsonl').open('w', encoding='utf-8') as queries:
            for row in query_records:
                actions.write(json.dumps(row, allow_nan=False) + '\n')
                queries.write(json.dumps({key: value for key, value in row.items() if key != 'action'}, allow_nan=False) + '\n')
        save(args.out / 'summary.json', summary)
        save(artifact / 'summary.json', summary)
        report = ['# Microscaling + additive VQ：配对数值验证', '',
                  f"Microscaled VQ test motor RMSE：{results['test'][primary]:.8f}。",
                  f"Row-scale VQ 对照：{results['test']['control'][primary]:.8f}。",
                  f"相对改善：{results['test']['relative_motor_improvement']:.3%}；改善 cases：{results['test']['improved_cases']}/22。",
                  f"工程筛选门槛通过：{summary['engineering_gate_passed']}。", '',
                  '全部614个Linear；selection 4 cases/8 contexts，test 22 cases/44 contexts。',
                  'BF16 additive books，d4/M2/K256；A4、Smooth0.5、H128固定。B32 exponent scaling，row gain及exponents固定。',
                  '已有test结果已被看过，本轮为配对机制诊断。没有native kernel、INT4 books或closed-loop成功率结论。']
        (args.out / 'RESULTS.zh.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        progress(stage='complete')
        print('MICROSCALE_COMPLETE', flush=True)
    except BaseException as exc:
        failure = {'status': 'failed', 'phase': 'full', 'job_id': job, 'stage': stage,
                   'fitted_linears': len(records), 'new_queries': len(query_records),
                   'exception': repr(exc), 'traceback': traceback.format_exc()}
        save(artifact / 'failure.json', failure)
        save(args.out / ('failure-' + job + '.json'), failure)
        raise
    finally:
        if banks is not None:
            banks.close()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
