"""Paired fixed-observation numerical validation, exclusively inside PBS."""
from __future__ import annotations
import argparse
import faulthandler
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time


def guard():
    host = socket.gethostname().split('.')[0]
    nodefile = os.environ.get('PBS_NODEFILE')
    if not os.environ.get('PBS_JOBID') or not nodefile or 'login' in host.lower():
        raise RuntimeError('Approved PBS compute allocation required')
    if host not in {n.split('.')[0] for n in Path(nodefile).read_text().split()}:
        raise RuntimeError('Host is absent from PBS_NODEFILE')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def case_mean(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row['case_id'], []).append(row['motor_rmse_first10_vs_bf16'])
    values = [sum(v) / len(v) for v in grouped.values()]
    result = {'case_count': len(values), 'context_count': len(records),
            'mean_motor_rmse_first10': sum(values) / len(values),
            'per_case_motor_rmse_first10': {str(case): sum(v) / len(v) for case, v in grouped.items()},
            'median_case_motor_rmse_first10': statistics.median(values),
            'max_case_motor_rmse_first10': max(values),
            'mean_gripper_rmse_first10': sum(r['gripper_rmse_first10_vs_bf16'] for r in records) / len(records)}
    field = 'motor_rmse_first10_vs_same_transform_bf16'
    if all(field in r for r in records):
        result['mean_motor_rmse_first10_vs_same_transform_bf16'] = sum(r[field] for r in records) / len(records)
    return result


class Banks:
    def __init__(self, runtime, out, protocol):
        import torch
        import smooth_vq as core
        self.torch, self.core = torch, core
        self.modules, self.out, self.protocol = runtime.q.modules, out, protocol
        self.original = {name: mod.weight.detach().cpu().clone() for name, mod in self.modules.items()}
        self.forwards = {name: mod.forward for name, mod in self.modules.items()}
        self.act_max, self.moments, self.moment_rows = {}, {}, {}
        self.transforms, self.bits = {}, None
        self.collect_original = self.collect_moment = False
        self.reset_stats()
        for name, mod in self.modules.items():
            def forward(x, _name=name, _mod=mod):
                if self.collect_original:
                    maximum = x.detach().float().reshape(-1, x.shape[-1]).abs().amax(dim=0)
                    self.act_max[_name] = torch.maximum(self.act_max[_name], maximum) if _name in self.act_max else maximum
                if not self.transforms:
                    self.stats['calls'] += 1
                    self.seen.add(_name)
                    return self.forwards[_name](x)
                z = core.transform_input(x, self.transforms[_name])
                qz = z.to(torch.bfloat16) if self.bits is None else core.quant_activation(z, self.bits)[0]
                if self.collect_moment:
                    rows = qz.float().reshape(-1, qz.shape[-1])
                    sums = rows.square().sum(dim=0)
                    self.moments[_name] = self.moments[_name] + sums if _name in self.moments else sums
                    self.moment_rows[_name] = self.moment_rows.get(_name, 0) + rows.shape[0]
                self.stats['calls'] += 1
                self.seen.add(_name)
                if self.bits is not None:
                    self.stats['elements'] += z.numel()
                    self.stats['zeros'] += int((qz == 0).sum())
                    self.stats['error_squared'] += float((qz.float() - z).square().sum())
                    self.stats['signal_squared'] += float(z.square().sum())
                return torch.nn.functional.linear(qz, _mod.weight, _mod.bias)
            mod.forward = forward

    def reset_stats(self):
        self.stats = {'calls': 0, 'elements': 0, 'zeros': 0, 'error_squared': 0.0, 'signal_squared': 0.0}
        self.seen = set()

    def stats_summary(self):
        return {**self.stats, 'observed_module_count': len(self.seen),
                'zero_fraction_element_weighted': self.stats['zeros'] / max(self.stats['elements'], 1),
                'activation_relative_rmse_element_weighted': (self.stats['error_squared'] / max(self.stats['signal_squared'], 1e-30)) ** 0.5}

    def original_mode(self):
        self.transforms, self.bits = {}, None
        for name, mod in self.modules.items():
            mod.weight.data = self.original[name].to(device='cuda')
        self.reset_stats()

    def activate(self, config, kind='bf16', metric=None, reload_bank=False):
        torch, core = self.torch, self.core
        directory = self.out / 'banks' / f"{config['name']}_{kind}"
        self.transforms, self.bits = {}, None
        payload_bytes = transform_bytes = serialized_bytes = weight_count = 0
        records = []
        started = time.perf_counter()
        with torch.no_grad():
            for index, (name, mod) in enumerate(self.modules.items()):
                if index % 25 == 0:
                    progress = {'status': 'running', 'stage': 'bank_reload' if reload_bank else 'bank_build',
                                'config': config['name'], 'kind': kind, 'modules_completed': index,
                                'module_count': len(self.modules), 'current_module': name}
                    save(self.out / 'bank_progress.json', progress)
                    print('BANK_PROGRESS ' + json.dumps(progress), flush=True)
                bank_path = directory / f'{index:04d}.pt'
                original = self.original[name].to(device='cuda')
                if reload_bank:
                    bank = torch.load(bank_path, map_location='cuda', weights_only=True)
                    if bank['module'] != name:
                        raise RuntimeError('Stored module identity differs')
                    transform = bank['transform']
                    encoded = bank['encoded']
                    decoded = core.decode_scalar_quantized(encoded['packed'], encoded['scales'], encoded['shape'], encoded['group']) if kind == 'scalar' else core.decode_vq_quantized(encoded['indices'], encoded['codebooks'], encoded['shape'])
                    mod.weight.data = decoded
                else:
                    transform = core.make_transform(self.act_max.get(name, torch.ones(original.shape[-1], device='cuda')), original,
                                                    config['alpha'], config['hadamard'], self.protocol['hadamard_block'], self.protocol['transform_seed'])
                    transformed = core.transform_weight(original, transform)
                    if kind == 'bf16':
                        mod.weight.data = transformed.to(torch.bfloat16)
                    else:
                        if kind == 'scalar':
                            encoded = core.scalar_quantize(transformed, group=128)
                        else:
                            settings = self.protocol['vq']
                            encoded = core.fit_vq(transformed, None if metric is None else metric.get(name),
                                                  seed=settings['seed'], sample_limit=settings['samples'], iterations=settings['iterations'])
                        mod.weight.data = encoded['decoded']
                        rmse = float((transformed - encoded['decoded'].float()).square().mean().sqrt())
                        packed = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in encoded.items() if k != 'decoded'}
                        stored_transform = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in transform.items()}
                        stored_transform['signs'] = stored_transform['signs'].to(torch.int8)
                        directory.mkdir(parents=True, exist_ok=True)
                        torch.save({'module': name, 'transform': stored_transform, 'encoded': packed}, bank_path)
                        payload_bytes += encoded['storage_bytes']
                        transform_bytes += stored_transform['scale'].numel() * 4 + stored_transform['signs'].numel()
                        serialized_bytes += bank_path.stat().st_size
                        append(self.out / 'module_bank_stats.jsonl', {'config': config['name'], 'kind': kind, 'module': name,
                                                                    'weight_rmse': rmse, 'storage_bytes': encoded['storage_bytes'],
                                                                    'fit_stats': encoded.get('fit_stats'), 'bank_file': str(bank_path.relative_to(self.out))})
                self.transforms[name] = transform
                weight_count += original.numel()
                record = {'module': name, 'scale_min': float(transform['scale'].min()), 'scale_max': float(transform['scale'].max()),
                          'scale_lower_bound_fraction': float((transform['scale'] <= 1 / 16).float().mean()),
                          'scale_upper_bound_fraction': float((transform['scale'] >= 16).float().mean())}
                if not reload_bank:
                    record['transformed_weight_peak_ratio'] = float(transformed.abs().max() / original.float().abs().max().clamp_min(1e-30))
                records.append(record)
                del original
                if not reload_bank:
                    del transformed
                if kind != 'bf16':
                    del encoded
        self.reset_stats()
        if reload_bank:
            return json.loads((directory / 'receipt.json').read_text())
        receipt = {'config': config['name'], 'kind': kind, 'target_module_count': len(self.modules), 'weight_count': weight_count,
                   'encoded_weight_payload_bytes': payload_bytes, 'transform_tensor_payload_bytes': transform_bytes,
                   'effective_bpw_with_transform': 8 * (payload_bytes + transform_bytes) / max(weight_count, 1) if kind != 'bf16' else 16,
                   'serialized_module_file_bytes': serialized_bytes,
                   'bank_build_wall_seconds_diagnostic_only': time.perf_counter() - started,
                   'transform_scale_min': min(r['scale_min'] for r in records),
                   'transform_scale_max': max(r['scale_max'] for r in records),
                   'max_transformed_weight_peak_ratio': max(r['transformed_weight_peak_ratio'] for r in records),
                   'execution': 'decoded BF16 reference Linear; BF16 weights retained on GPU; no native low-bit kernel'}
        append(self.out / 'transform_stats.jsonl', {'config': config['name'], 'kind': kind, 'modules': records})
        if kind != 'bf16':
            save(directory / 'receipt.json', receipt)
        return receipt

    def close(self):
        for name, mod in self.modules.items():
            mod.forward = self.forwards[name]


def run(args):
    guard()
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    print('VALIDATION_START ' + json.dumps({'phase': args.phase, 'pbs_jobid': os.environ['PBS_JOBID']}), flush=True)
    import torch
    sys.path.append(str(args.source_root))
    import run_diagnostic as source
    import runner as pilot
    protocol = json.loads(args.protocol.read_text())
    if protocol['protocol'] != 'fastwam-smooth-vq-phases12-v1':
        raise ValueError('Unexpected protocol')
    split = {key: list(protocol[key]) for key in ('calibration_cases', 'selection_cases', 'test_cases')}
    if any(set(split[a]) & set(split[b]) for a, b in [('calibration_cases', 'selection_cases'), ('calibration_cases', 'test_cases'), ('selection_cases', 'test_cases')]):
        raise ValueError('Calibration, selection and test must be disjoint')
    configs = protocol['phase1_configs']
    phase2_names = protocol['phase2_config_names']
    if args.phase == 'preflight':
        split = {key: protocol['preflight'][key] for key in split}
        configs = [c for c in configs if c['name'] in protocol['preflight']['phase1_config_names']]
        phase2_names = protocol['preflight']['phase2_config_names']
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Refusing to overwrite previous validation artifacts')
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / 'protocol.json', protocol)
    plan = json.loads((args.source_root / 'plan.json').read_text())
    runtime = banks = None
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    stage = 'model_loading'
    queries = []
    refs, inputs = {}, {}
    try:
        runtime = pilot.Runtime(args.out)
        runtime.arm('bf16')
        runtime_record = json.loads((args.out / 'runtime.json').read_text())
        runtime_record.update(protocol=protocol['protocol'], real_quant=False, execution='decoded BF16 reference; compressed banks stored separately', native_low_bit_kernel=False)
        save(args.out / 'runtime.json', runtime_record)
        banks = Banks(runtime, args.out, protocol)

        def datum(case):
            if case not in inputs:
                row = source.validate_plan(plan, case)
                observation, metadata = source.load_case(args.source_root, row)
                inputs[case] = (runtime.datum(observation), metadata['description'], row)
            return inputs[case]

        def predict(case, seed_index, label, group, control=None, reference=False):
            value, description, row = datum(case)
            seed = row['sampler_seeds'][seed_index]
            runtime.description, runtime.current_arm = description, label
            save(args.out / 'progress.json', {'status': 'running', 'stage': stage,
                                              'queries_completed': len(queries), 'query_in_progress':
                                              {'case_id': case, 'sampler_seed': seed, 'arm': label, 'split': group}})
            torch.cuda.synchronize()
            started = time.perf_counter()
            action, _ = runtime.infer(value, seed, measure=False)
            torch.cuda.synchronize()
            action_cpu = action.detach().float().cpu()
            key = (case, seed_index)
            if reference:
                refs[key] = action_cpu
            metrics = source.action_metrics(torch, action_cpu, refs[key]) if key in refs else {}
            if control is not None:
                metrics.update(source.action_metrics(torch, action_cpu, control, compared_to='same_transform_bf16'))
            record = {'case_id': case, 'seed_index': seed_index, 'sampler_seed': seed, 'domain': row['domain'],
                      'split': group, 'arm': label, **metrics,
                      'wall_seconds_diagnostic_only': time.perf_counter() - started, 'episodes': 0, 'post_query_env_steps': 0}
            queries.append(record)
            append(args.out / 'queries.jsonl', record)
            append(args.out / 'actions.jsonl', {**record, 'action': action_cpu.tolist()})
            save(args.out / 'progress.json', {'status': 'running', 'stage': stage, 'queries_completed': len(queries), 'last_query': record})
            return action_cpu, record

        def evaluate(cases, label, group, controls=None, reference=False):
            records, actions = [], {}
            for case in cases:
                for seed_index in protocol['evaluation_seed_indices']:
                    key = (case, seed_index)
                    action, record = predict(case, seed_index, label, group, None if controls is None else controls[key], reference)
                    actions[key] = action
                    records.append(record)
            return records, actions

        stage = 'bf16_calibration'
        banks.collect_original = True
        for case in split['calibration_cases']:
            predict(case, 0, 'bf16_calibration', 'calibration', reference=True)
        banks.collect_original = False
        stage = 'bf16_selection_reference'
        evaluate(split['selection_cases'], 'bf16', 'selection', reference=True)
        calibrated_count = len(banks.act_max)
        metrics_by_config, phase1_scores, selection, receipts = {}, {}, {}, {}
        for config in configs:
            name = config['name']
            stage = f'phase1_{name}'
            receipt = banks.activate(config)
            _, controls = evaluate(split['selection_cases'], f'{name}_bf16_control', 'selection')
            banks.bits, banks.collect_moment = 4, True
            banks.moments, banks.moment_rows = {}, {}
            for case in split['calibration_cases']:
                predict(case, 0, f'{name}_bf16w_a4', 'calibration')
            banks.collect_moment = False
            metrics_by_config[name] = {mod: total.detach().cpu() / banks.moment_rows[mod] for mod, total in banks.moments.items()}
            banks.reset_stats()
            records, _ = evaluate(split['selection_cases'], f'{name}_bf16w_a4', 'selection', controls)
            result = {**case_mean(records), 'activation': banks.stats_summary(), 'transform': receipt}
            selection[f'{name}_bf16w_a4'] = result
            phase1_scores[name] = result['mean_motor_rmse_first10']
        phase2_scores = {'scalar': {}, 'vq': {}}
        config_map = {c['name']: c for c in configs}
        for name in phase2_names:
            for kind in ('scalar', 'vq'):
                stage = f'phase2_{name}_{kind}'
                receipt = banks.activate(config_map[name], kind, metrics_by_config[name])
                banks.bits = 4
                records, _ = evaluate(split['selection_cases'], f'{name}_{kind}_a4', 'selection')
                result = {**case_mean(records), 'activation': banks.stats_summary(), 'storage': receipt}
                selection[f'{name}_{kind}_a4'] = result
                phase2_scores[kind][name] = result['mean_motor_rmse_first10']
                receipts[f'{name}_{kind}'] = receipt
        winners = {'phase1': min(phase1_scores, key=phase1_scores.get),
                   'scalar': min(phase2_scores['scalar'], key=phase2_scores['scalar'].get),
                   'vq': min(phase2_scores['vq'], key=phase2_scores['vq'].get)}
        save(args.out / 'frozen_winners.json', {'protocol': protocol['protocol'], 'winners': winners,
             'selection_scores': {'phase1': phase1_scores, **phase2_scores}, 'test_queries_before_freeze': 0,
             'preflight_only': args.phase == 'preflight'})
        test = {}
        if split['test_cases']:
            stage = 'bf16_test_reference_after_freeze'
            banks.original_mode()
            evaluate(split['test_cases'], 'bf16', 'test', reference=True)
            phase1_controls = {}
            for name in dict.fromkeys(['identity', winners['phase1']]):
                stage = f'test_phase1_{name}'
                banks.activate(config_map[name])
                _, controls = evaluate(split['test_cases'], f'{name}_bf16_control', 'test')
                phase1_controls[name] = controls
                banks.bits = 4
                banks.reset_stats()
                records, _ = evaluate(split['test_cases'], f'{name}_bf16w_a4', 'test', controls)
                test[f'{name}_bf16w_a4'] = {**case_mean(records), 'activation': banks.stats_summary()}
            for kind in ('scalar', 'vq'):
                for name in dict.fromkeys(['identity', winners['scalar'], winners['vq']]):
                    stage = f'test_phase2_{name}_{kind}'
                    receipt = banks.activate(config_map[name], kind, reload_bank=True)
                    banks.bits = 4
                    records, _ = evaluate(split['test_cases'], f'{name}_{kind}_a4', 'test')
                    test[f'{name}_{kind}_a4'] = {**case_mean(records), 'activation': banks.stats_summary(), 'storage': receipt}
        summary = {'protocol': protocol['protocol'], 'phase': args.phase, 'status': 'complete', 'pbs_jobid': os.environ['PBS_JOBID'],
                   'split': split, 'calibration_context_count': len(split['calibration_cases']),
                   'selection_context_count': 2 * len(split['selection_cases']), 'test_context_count': 2 * len(split['test_cases']),
                   'query_count': len(queries), 'calibrated_module_count': calibrated_count, 'target_module_count': len(banks.modules),
                   'winners': winners, 'selection': selection, 'test': test,
                   'numerical_controls': {group: {arm: case_mean([r for r in queries if r['split'] == group and r['arm'] == arm])
                      for arm in sorted({r['arm'] for r in queries if r['split'] == group and r['arm'].endswith('_bf16_control')})}
                      for group in ('selection', 'test')},
                   'execution': 'decoded BF16 reference Linear; compressed weight bank storage verified by tensor bytes and saved file sizes',
                   'vq_metric': protocol['vq_metric'], 'episodes': 0, 'native_vq_latency_claim': False,
                   'max_gpu_memory_allocated_bytes': torch.cuda.max_memory_allocated(), 'inform_final_selection': args.phase == 'full'}
        save(args.out / 'summary.json', summary)
        save(args.out / 'progress.json', {'status': 'complete', 'queries_completed': len(queries)})
        print('VALIDATION_COMPLETE ' + json.dumps({'phase': args.phase, 'queries': len(queries), 'winners': winners}), flush=True)
    except BaseException as exc:
        save(args.out / 'failure.json', {'stage': stage, 'queries_completed': len(queries), 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        if banks is not None:
            banks.close()
        if runtime is not None:
            runtime.q.close()
            if runtime.env is not None:
                runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'full'), required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
