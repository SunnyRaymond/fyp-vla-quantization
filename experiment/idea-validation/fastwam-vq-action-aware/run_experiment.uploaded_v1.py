"""Frozen-input, same-format action-aware VQ campaign; numerical work is PBS-only."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import socket
import statistics
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
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def means(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row['case_id'], []).append(row['motor_rmse_first10_vs_bf16'])
    values = [statistics.mean(v) for v in grouped.values()]
    return {'case_count': len(values), 'context_count': len(records),
            'mean_motor_rmse_first10': statistics.mean(values),
            'per_case_motor_rmse_first10': {str(k): statistics.mean(v) for k, v in grouped.items()}}


def cpu_encoding(torch, encoding):
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in encoding.items()}


def qinput(torch, core, x, transform):
    z = core.transform_input(x, transform).float()
    scale = z.abs().amax(dim=-1, keepdim=True) / 7
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(z / safe).clamp(-7, 7) * scale).to(torch.bfloat16)


class FrozenBanks:
    def __init__(self, runtime, out, protocol):
        import torch
        import smooth_vq as core
        self.torch, self.core = torch, core
        self.modules, self.out = runtime.q.modules, out
        if len(self.modules) != 614:
            raise RuntimeError(f'Expected all 614 target Linears, found {len(self.modules)}')
        self.original = {n: m.weight.detach().cpu().clone() for n, m in self.modules.items()}
        self.forwards = {n: m.forward for n, m in self.modules.items()}
        self.old_root = Path(protocol['roots']['old_banks']) / 'results/full/banks'
        self.transforms = {}
        self.moments, self.moment_rows = {}, {}
        self.transformed = self.collect_moment = False
        self.seen = set()
        self.weight_count = sum(w.numel() for w in self.original.values())
        for index, (name, mod) in enumerate(self.modules.items()):
            bank = torch.load(self.old_path('scalar', index), map_location='cpu', weights_only=True)
            if bank['module'] != name or tuple(bank['encoded']['shape']) != tuple(mod.weight.shape):
                raise RuntimeError(f'Old saved bank identity/shape differs at {name}')
            self.transforms[name] = {k: v.to('cuda') if torch.is_tensor(v) else v
                                     for k, v in bank['transform'].items()}
            del bank

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                if not self.transformed:
                    return self.forwards[_name](x)
                qz = qinput(torch, core, x, self.transforms[_name])
                if self.collect_moment:
                    rows = qz.detach().float().reshape(-1, qz.shape[-1])
                    sums = rows.square().sum(0)
                    self.moments[_name] = self.moments.get(_name, 0) + sums
                    self.moment_rows[_name] = self.moment_rows.get(_name, 0) + rows.shape[0]
                return torch.nn.functional.linear(qz, _mod.weight, _mod.bias)

            mod.forward = forward
            if index % 100 == 0:
                print(f'TRANSFORMS_LOADED {index}/614', flush=True)

    def old_path(self, kind, index):
        return self.old_root / f'smooth05_hadamard_{kind}' / f'{index:04d}.pt'

    def dense_teacher_weight(self, name):
        return self.core.transform_weight(self.original[name].to('cuda'), self.transforms[name])

    def activate(self, arm, label=None):
        torch, core = self.torch, self.core
        self.transformed = arm != 'bf16'
        self.seen.clear()
        with torch.no_grad():
            for index, (name, mod) in enumerate(self.modules.items()):
                if arm == 'bf16':
                    dense = self.original[name].to('cuda')
                elif arm == 'bf16w_a4':
                    dense = self.dense_teacher_weight(name).to(torch.bfloat16)
                elif arm in ('scalar_w4a4', 'original_vq_a4'):
                    kind = 'scalar' if arm == 'scalar_w4a4' else 'vq'
                    bank = torch.load(self.old_path(kind, index), map_location='cuda', weights_only=True)
                    if bank['module'] != name:
                        raise RuntimeError(f'Old bank identity differs at {name}')
                    encoded = bank['encoded']
                    dense = (core.decode_scalar_quantized(encoded['packed'], encoded['scales'], encoded['shape'], encoded['group'])
                             if kind == 'scalar' else core.decode_vq_quantized(encoded['indices'], encoded['codebooks'], encoded['shape']))
                    del bank, encoded
                else:
                    import codebook
                    path = self.out / 'banks' / (label or arm) / f'{index:04d}.pt'
                    if path.exists():
                        bank = torch.load(path, map_location='cuda', weights_only=True)
                        if bank['module'] != name:
                            raise RuntimeError(f'New bank identity differs at {name}')
                        dense = codebook.decode_vq(bank['encoded'])
                        del bank
                    elif self.out.name == 'preflight':
                        dense = self.dense_teacher_weight(name).to(torch.bfloat16)
                    else:
                        raise FileNotFoundError(f'Full arm missing bank: {path}')
                mod.weight.data = dense
        torch.cuda.empty_cache()

    def close(self):
        for name, mod in self.modules.items():
            mod.forward = self.forwards[name]


class LayerData:
    """Sampled teacher calls, retaining context identity for coupled action projections."""
    def __init__(self, banks, name, captures):
        torch = banks.torch
        xs, ys, adjoints, factors, contexts = [], [], [], [], []
        for context, capture in enumerate(captures):
            calls = capture['layers'].get(name, [])
            if not calls:
                raise RuntimeError(f'No calibration teacher calls for {name}')
            for call in calls:
                rows = call['inputs'].to('cuda')
                xs.append(qinput(torch, banks.core, rows, banks.transforms[name]).float())
                ys.append(call['teacher_y'].to('cuda'))
                adjoints.append(call['adjoints'].to('cuda'))
                count = rows.shape[0]
                factors.append(torch.full((count,), call['total_rows'] / count, device='cuda'))
                contexts.append(torch.full((count,), context, dtype=torch.long, device='cuda'))
        self.torch = torch
        self.x, self.y = torch.cat(xs), torch.cat(ys)
        self.adjoint = torch.cat(adjoints, dim=1)
        self.factors, self.contexts = torch.cat(factors), torch.cat(contexts)
        self.context_count, self.probes = len(captures), self.adjoint.shape[0]
        self.bias = banks.modules[name].bias
        self.denominator = self.factors.sum() * self.y.shape[1]

    def error(self, dense):
        bias = None if self.bias is None else self.bias.detach().float()
        return self.torch.nn.functional.linear(self.x, dense.float(), bias) - self.y

    def local(self, dense):
        return (self.error(dense).square().sum(1) * self.factors).sum() / self.denominator

    def project(self, dense):
        row_values = (self.adjoint * self.error(dense).unsqueeze(0)).sum(-1).T * self.factors[:, None]
        return row_values.new_zeros((self.context_count, self.probes)).index_add(0, self.contexts, row_values)

    def block_metric(self, local_scale, action_scale, module_count, action_lambda):
        torch = self.torch
        rows, width = self.x.shape
        padded = torch.nn.functional.pad(self.x, (0, (-width) % 4)).reshape(rows, -1, 4)
        local = torch.einsum('rgd,rge,r->gde', padded, padded, self.factors) / self.denominator
        local = local / local_scale / module_count
        if not action_lambda:
            return local
        out_width = self.y.shape[1]
        groups = padded.shape[1]
        task = self.x.new_zeros((out_width, groups, 4, 4))
        for context in range(self.context_count):
            mask = self.contexts == context
            factor = self.adjoint[:, mask] * self.factors[mask][None, :, None]
            # Sum repeated weight uses before forming their quadratic metric.
            gradient = torch.einsum('pro,ri->poi', factor, self.x[mask])
            gradient = torch.nn.functional.pad(gradient, (0, (-width) % 4)).reshape(self.probes, out_width, groups, 4)
            task.add_(torch.einsum('pogd,poge->ogde', gradient, gradient))
            del factor, gradient
        task *= 0.5 * action_lambda / (self.context_count * self.probes * action_scale)
        task += local.unsqueeze(0)
        return task


def run(args):
    guard()
    import torch
    import codebook
    import runner as pilot
    import run_diagnostic as source
    import sensitivity
    import test_codebook
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol['protocol'] != 'fastwam-vq-action-aware-v1':
        raise ValueError('Unexpected protocol')
    if args.out.exists():
        raise FileExistsError(f'Refusing previous result path: {args.out}')
    args.out.mkdir(parents=True)
    save(args.out / 'protocol.json', protocol)
    root, old_inputs = Path(protocol['remote_root']), Path(protocol['roots']['old_inputs'])
    plan = json.loads((root / 'plan.json').read_text())
    old_plan = json.loads((old_inputs / 'plan.json').read_text())
    fresh_rows = {r['case_id']: r for r in plan['inputs']}
    split = {key: list(protocol[key]) for key in ('calibration_cases', 'selection_cases', 'test_cases')}
    seeds = protocol['evaluation_seed_indices']
    probes, steps = protocol['action_sensitivity']['probes'], protocol['vq']['tune_steps']
    if args.phase == 'preflight':
        split = {key: list(protocol['preflight'][key]) for key in split}
        seeds = protocol['preflight']['evaluation_seed_indices']
        probes, steps = protocol['preflight']['probes'], protocol['preflight']['tune_steps']
    if any(set(split[a]) & set(split[b]) for a, b in [('calibration_cases', 'selection_cases'), ('calibration_cases', 'test_cases'), ('selection_cases', 'test_cases')]):
        raise ValueError('Calibration, selection and test overlap')
    runtime = banks = None
    stage = 'self_checks'
    queries, refs, data, captures, receipts = [], {}, {}, [], {}
    frozen = False
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()

    def progress(**extra):
        value = {'status': 'running', 'stage': stage, 'queries_completed': len(queries), **extra}
        save(args.out / 'progress.json', value)
        print('PROGRESS ' + json.dumps(value), flush=True)

    def datum(case):
        if case not in data:
            input_root = old_inputs if case < 1000 else root
            row = source.validate_plan(old_plan, case) if case < 1000 else fresh_rows[case]
            observation, metadata = source.load_case(input_root, row)
            data[case] = (runtime.datum(observation), metadata['description'], row)
        return data[case]

    def predict(case, seed_index, arm, group, action_lambda=None, projected=None):
        if group == 'test' and not frozen:
            raise RuntimeError('Test query requested before lambda freeze')
        value, description, row = datum(case)
        runtime.description, runtime.current_arm = description, arm
        banks.seen.clear()
        progress(case_id=case, seed_index=seed_index, arm=arm, split=group)
        query_started = time.perf_counter()
        action, _ = runtime.infer(value, row['sampler_seeds'][seed_index], measure=False)
        action = action.detach().float().cpu()
        if arm == 'bf16':
            refs[case, seed_index] = action
        if (case, seed_index) not in refs:
            raise RuntimeError('Missing paired BF16 action reference')
        metrics = source.action_metrics(torch, action, refs[case, seed_index])
        record = {'case_id': case, 'seed_index': seed_index, 'sampler_seed': row['sampler_seeds'][seed_index],
                  'domain': row['domain'], 'dimension': row['dimension'], 'split': group, 'arm': arm,
                  **metrics, 'episodes0': 0, 'episodes': 0, 'predicted_actions_executed': 0,
                  'lambda_frozen_before_test': frozen, 'observed_module_count': len(banks.seen),
                  'wall_seconds_diagnostic_only': time.perf_counter() - query_started}
        if action_lambda is not None:
            record['action_lambda'] = action_lambda
        if projected is not None:
            record['projected_action_rmse_sampled_proxy'] = float(projected.square().mean().sqrt())
        if len(banks.seen) != 614:
            raise RuntimeError(f'Query target coverage is only {len(banks.seen)}/614')
        queries.append(record)
        append(args.out / 'queries.jsonl', record)
        append(args.out / 'actions.jsonl', {**record, 'action': action.tolist()})
        return record

    def evaluate(cases, arm, group, action_lambda=None, projection=None):
        records = []
        for case in cases:
            for seed_index in ([0] if group == 'calibration' else seeds):
                predicted = None if projection is None else projection[split['calibration_cases'].index(case)]
                records.append(predict(case, seed_index, arm, group, action_lambda, predicted))
        return means(records)

    def store_bank(name, index, label, encoding, stats):
        directory = args.out / 'banks' / label
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{index:04d}.pt'
        if path.exists():
            raise FileExistsError(f'Bank already exists: {path}')
        transform = cpu_encoding(torch, banks.transforms[name])
        torch.save({'module': name, 'transform': transform, 'encoded': cpu_encoding(torch, encoding)}, path)
        transform_bytes = sum(v.numel() * v.element_size() for v in transform.values() if torch.is_tensor(v))
        row = {'arm': label, 'module': name, 'module_index': index, 'encoded_bytes': codebook.encoded_nbytes(encoding),
               'transform_bytes': transform_bytes, 'serialized_bytes': path.stat().st_size, 'fit': stats}
        append(args.out / 'module_fit.jsonl', row)
        receipt = receipts.setdefault(label, {'arm': label, 'module_count': 0, 'encoded_bytes': 0, 'serialized_bytes': 0,
                                              'transform_tensor_payload_bytes': 0, 'transforms': protocol['transform']})
        receipt['module_count'] += 1
        receipt['encoded_bytes'] += row['encoded_bytes'] + transform_bytes
        receipt['serialized_bytes'] += row['serialized_bytes']
        receipt['transform_tensor_payload_bytes'] += transform_bytes
        receipt['effective_bpw_with_transform'] = receipt['encoded_bytes'] * 8 / banks.weight_count

    def load_encoding(label, index):
        return torch.load(args.out / 'banks' / label / f'{index:04d}.pt', map_location='cuda', weights_only=True)['encoded']

    try:
        if args.phase == 'preflight':
            test_codebook.main()
        stage = 'model_loading'
        progress()
        runtime = pilot.Runtime(args.out)
        runtime.arm('bf16')
        banks = FrozenBanks(runtime, args.out, protocol)
        names = tuple(banks.modules) if args.phase == 'full' else tuple(protocol['preflight']['numerical_modules'])
        indices = {name: index for index, name in enumerate(banks.modules)}
        runtime_record = json.loads((args.out / 'runtime.json').read_text())
        runtime_record.update(protocol=protocol['protocol'], real_quant=False, native_low_bit_kernel=False,
                              execution='saved VQ encoding; decoded BF16 reference Linear on GPU')
        save(args.out / 'runtime.json', runtime_record)

        stage = 'teacher_action_sensitivity'
        banks.activate('bf16')
        for case in split['calibration_cases']:
            value, description, row = datum(case)
            runtime.description, runtime.current_arm = description, 'bf16'
            progress(case_id=case, probes=probes, sensitivity_scope=614)
            capture = sensitivity.capture_action_projections(runtime, value, row['sampler_seeds'][0], None,
                        probes, protocol['action_sensitivity']['probe_seed'],
                        rows_per_call=protocol['calibration_reservoir']['rows_per_call'],
                        artifact_path=args.out / 'sensitivity' / f'case_{case}.pt')
            sensitivity.validate_preflight(capture)
            if capture['summary']['target_module_count'] != 614 or capture['summary']['uncalled_target_modules']:
                raise RuntimeError('Full sensitivity target call coverage is incomplete')
            refs[case, 0] = capture['baseline_action']
            save(args.out / 'sensitivity' / f'case_{case}.json', capture['summary'])
            captures.append(capture)

        stage = 'original_candidate_q4_input_moments'
        banks.activate('bf16w_a4')
        banks.collect_moment = True
        evaluate(split['calibration_cases'], 'bf16w_a4', 'calibration')
        banks.collect_moment = False
        if len(banks.moments) != 614:
            raise RuntimeError('Original candidate input moments do not cover 614 Linears')
        moments = {n: x.detach().cpu() / banks.moment_rows[n] for n, x in banks.moments.items()}
        banks.moments.clear()
        banks.activate('bf16')
        stage = 'bf16_selection_reference'
        selection = {'bf16': evaluate(split['selection_cases'], 'bf16', 'selection')}

        stage = 'row_scale_vq_build'
        base_projections, base_local = {}, {}
        global_base = torch.zeros((len(captures), probes), device='cuda')
        settings = protocol['vq']
        for count, name in enumerate(names):
            progress(module=name, modules_completed=count, fitted_module_count=len(names))
            index = indices[name]
            weight = banks.dense_teacher_weight(name)
            layer = LayerData(banks, name, captures)
            fit_started = time.perf_counter()
            encoding = codebook.fit_vq(weight, second_moment=moments[name].to('cuda'), seed=settings['seed'],
                                       sample_limit=settings['samples'], iterations=settings['iterations'])
            with torch.no_grad():
                dense = codebook.decode_vq(encoding).float()
                projection = layer.project(dense)
                local = max(float(layer.local(dense)), 1e-12)
            base_local[name], base_projections[name] = local, projection.detach()
            global_base += projection
            store_bank(name, index, 'row_scale_vq_a4', encoding,
                       {'local_output_mse': local, 'fit_wall_seconds_diagnostic': time.perf_counter() - fit_started})
            del layer, encoding, weight, dense
        task_scale = max(float(0.5 * global_base.square().mean()), 1e-12)
        projections = {'row_scale_vq_a4': global_base.detach().clone()}
        save(args.out / 'objective_normalization.json', {'local_initial_mse': base_local, 'global_initial_projected_half_mse': task_scale,
             'local_aggregation': 'mean of per-module baseline-normalized losses',
             'task_aggregation': 'sum all captured calls and fitted modules within each context before squaring',
             'sampling_boundary': 'uniform row projection is an approximate sampled proxy; squared estimator is not claimed unbiased'})

        labels = [('row_output_vq_a4', 0.0)] + [(f'row_action_vq_a4_lambda_{value:g}', value)
                                              for value in protocol['action_sensitivity']['lambda_candidates']]
        for label, action_lambda in labels:
            stage = f'fit_{label}'
            total_projection = global_base.detach().clone()
            for count, name in enumerate(names):
                progress(module=name, modules_completed=count, action_lambda=action_lambda)
                index = indices[name]
                layer = LayerData(banks, name, captures)
                encoding = load_encoding('row_scale_vq_a4', index)
                weight = banks.dense_teacher_weight(name)
                offset = (total_projection - base_projections[name]).detach()
                local_scale = base_local[name]

                def objective(dense):
                    local = layer.local(dense) / local_scale / len(names)
                    if not action_lambda:
                        return local
                    task = 0.5 * (offset + layer.project(dense)).square().mean() / task_scale
                    return local + action_lambda * task

                fit_started = time.perf_counter()
                with torch.no_grad():
                    initial_objective = float(objective(codebook.decode_vq(encoding).float()))
                    metric = layer.block_metric(local_scale, task_scale, len(names), action_lambda)
                    reassigned, assignment_stats = codebook.reassign_indices(encoding, weight, block_metric=metric, iterations=2)
                    reassigned_objective = float(objective(codebook.decode_vq(reassigned).float()))
                    accepted = reassigned_objective < initial_objective
                    if accepted:
                        encoding = reassigned
                del metric, reassigned
                tuned, tune_stats = codebook.tune_fixed_indices(encoding, objective, max_steps=steps,
                          patience=min(settings['tune_patience'], steps), learning_rate=settings['tune_learning_rate'],
                          tune_scales=settings['tune_scales'])
                with torch.no_grad():
                    dense = codebook.decode_vq(tuned).float()
                    current_projection = layer.project(dense)
                    total_projection = offset + current_projection
                    final_local = float(layer.local(dense))
                store_bank(name, index, label, tuned, {'initial_objective': initial_objective,
                      'reassigned_objective': reassigned_objective, 'reassignment_accepted': accepted,
                      'assignment': assignment_stats, 'tuning': tune_stats, 'local_output_mse': final_local,
                      'global_projected_half_mse_after_module': float(0.5 * total_projection.square().mean()),
                      'fit_wall_seconds_diagnostic': time.perf_counter() - fit_started})
                del layer, encoding, tuned, weight, dense
            projections[label] = total_projection.detach().clone()

        stage = 'selection_and_calibration_validation'
        for arm in protocol['arms'][1:-1]:
            banks.activate(arm)
            selection[arm] = evaluate(split['selection_cases'], arm, 'selection')
            if arm in projections:
                evaluate(split['calibration_cases'], arm, 'calibration', projection=projections[arm])
        lambda_scores = {}
        for label, value in labels[1:]:
            banks.activate('row_action_vq_a4', label)
            selection[label] = evaluate(split['selection_cases'], label, 'selection', action_lambda=value)
            lambda_scores[str(value)] = selection[label]['mean_motor_rmse_first10']
            evaluate(split['calibration_cases'], label, 'calibration', action_lambda=value, projection=projections[label])
        winner = min(protocol['action_sensitivity']['lambda_candidates'], key=lambda x: lambda_scores[str(x)])
        winner_label = f'row_action_vq_a4_lambda_{winner:g}'
        save(args.out / 'frozen_winners.json', {'protocol': protocol['protocol'], 'selected_action_lambda': winner,
             'selection_scores': lambda_scores, 'test_queries_before_freeze': 0, 'frozen_query_count': len(queries),
             'preflight_only': args.phase == 'preflight'})
        frozen = True

        for arm, kind in [('scalar_w4a4', 'scalar'), ('original_vq_a4', 'vq')]:
            receipt = json.loads((banks.old_root / f'smooth05_hadamard_{kind}/receipt.json').read_text())
            receipts[arm] = {'arm': arm, 'module_count': receipt['target_module_count'],
                   'encoded_bytes': receipt['encoded_weight_payload_bytes'] + receipt['transform_tensor_payload_bytes'],
                   'serialized_bytes': receipt['serialized_module_file_bytes'],
                   'effective_bpw_with_transform': receipt['effective_bpw_with_transform'], 'transforms': protocol['transform'],
                   'source': str(banks.old_root / f'smooth05_hadamard_{kind}')}
        for arm in ['bf16', 'bf16w_a4']:
            transform_bytes = 0 if arm == 'bf16' else sum(v.numel() * v.element_size()
                               for t in banks.transforms.values() for v in t.values() if torch.is_tensor(v))
            receipts[arm] = {'arm': arm, 'module_count': 614, 'encoded_bytes': 2 * banks.weight_count + transform_bytes,
                            'effective_bpw_with_transform': 16 + 8 * transform_bytes / banks.weight_count,
                            'transforms': 'identity' if arm == 'bf16' else protocol['transform'],
                            'storage_boundary': 'dense target weights and transform tensors; non-target model tensors excluded'}
        receipts['row_action_vq_a4'] = {**receipts[winner_label], 'arm': 'row_action_vq_a4', 'action_lambda': winner}
        save(args.out / 'receipts.json', receipts)

        test = {}
        if split['test_cases']:
            stage = 'held_out_test_after_freeze'
            for arm in protocol['arms']:
                banks.activate(arm, winner_label if arm == 'row_action_vq_a4' else None)
                test[arm] = evaluate(split['test_cases'], arm, 'test', winner if arm == 'row_action_vq_a4' else None)
        summary = {'status': 'complete', 'protocol': protocol['protocol'], 'phase': args.phase,
               'pbs_jobid': os.environ['PBS_JOBID'], 'target_module_count': 614, 'calibrated_module_count': len(names),
               'sensitivity_target_module_count': 614, 'numerically_fitted_modules': list(names), 'split': split,
               'query_count': len(queries), 'selected_action_lambda': winner, 'selection_scores': lambda_scores,
               'test_queries_before_freeze': 0, 'selection': selection, 'test': test, 'receipts': receipts,
               'evaluation_episodes': 0, 'predicted_actions_executed': 0, 'restoration_experiments': False,
               'native_latency_claim': False, 'wall_seconds_diagnostic_only': time.perf_counter() - started,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'execution': 'decoded BF16 reference; stored books/indices/scales; no packed VQ compute kernel',
               'claim_scope': protocol['claim_scope'], 'preflight_only': args.phase == 'preflight'}
        save(args.out / 'summary.json', summary)
        save(args.out / 'progress.json', {'status': 'complete', 'query_count': len(queries)})
        print('EXPERIMENT_COMPLETE ' + json.dumps({'phase': args.phase, 'queries': len(queries), 'selected_action_lambda': winner}), flush=True)
    except BaseException as exc:
        save(args.out / 'failure.json', {'stage': stage, 'queries_completed': len(queries), 'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        torch.backends.cuda.matmul.allow_tf32 = tf32
        if banks is not None:
            banks.close()
        if runtime is not None:
            runtime.q.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'full'), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
