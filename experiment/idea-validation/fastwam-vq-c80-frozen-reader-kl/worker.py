"""Matched C80 A+Z/reader-KL, 80 updates, final student and frozen-reader endpoints."""
import argparse
import gc
import math
import os
from pathlib import Path
import signal
import time
from anchor_selfcheck import require_allocation
from reuse_reference import read, save, queries, load_reference
from reader_loss import base_terms, base_objective, reader_terms
from reader_runtime import reader_session, compact_counts


class Deadline:
    def __init__(self):
        self.end = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ['FW_SOFT_DEADLINE_SECONDS'])
        self.stopped = False
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_):
        self.stopped = True

    def check(self, seconds=120):
        if self.stopped or time.time() + seconds + 120 >= self.end:
            raise TimeoutError('Committed updates and queries can be resumed without changing the recipe')


def run(p, arm):
    require_allocation()
    import numpy as np
    import torch
    import sampler
    import runner
    import calibration_source
    import train_c8_reference as original
    from train_banks import TrainableBanks
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, optimizer_to_device, to_cpu
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(p['remote_root'])
    out = root / 'results' / arm
    out.mkdir(parents=True, exist_ok=True)
    cpu_prep = read(root / 'results/prepare/summary.json')
    check = read(Path(os.environ['ARTIFACTS']) / 'selfcheck.json')
    if (cpu_prep['protocol'] != p['protocol'] or not cpu_prep['passed']
            or cpu_prep['heldout_used'] or not check['passed']
            or check['pbs_jobid'] != os.environ['PBS_JOBID']):
        raise ValueError('Frozen CPU preparation and same-allocation checks required')
    cache, saved_rows, rows = load_reference(torch, p)
    ids, stats = p['calibration']['case_ids'], cpu_prep['stats']
    order = p['calibration']['update_order']
    total_updates = p['training']['updates']
    deadline = Deadline()
    state = {'protocol': p['protocol'], 'unit': arm, 'pbs_jobid': os.environ['PBS_JOBID'],
             'complete': False, 'phase': 'loading', 'updates': 0, 'new_model_queries': 0,
             'new_bf16_reader_queries': 0, 'parameter_reverse_passes': 0,
             'baseline_new_model_queries': 0, 'new_teacher_queries': 0, 'new_closedloop_episodes': 0}
    counter_path = out / 'query_counters.json'
    if counter_path.is_file():
        counters = read(counter_path)
        if counters['protocol'] != p['protocol'] or counters['arm'] != arm:
            raise ValueError('Query counters differ')
        state.update(counters['counters'])
    runtime = None

    def count(key):
        state[key] += 1
        save(counter_path, {'protocol': p['protocol'], 'arm': arm, 'counters': {k: state[k] for k in ('new_model_queries', 'new_bf16_reader_queries', 'parameter_reverse_passes')}})

    def progress(phase, **values):
        state.update(phase=phase, **values)
        save(out / 'progress.json', state)
        print('FROZEN_READER_PROGRESS', state, flush=True)

    def parity(actual, expected):
        return original._parity(torch, actual.detach().float().cpu(), expected.float().cpu())

    try:
        progress('loading')
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        masters = {name: module.weight.detach().cpu().clone() for name, module in runtime.q.modules.items()}
        banks = TrainableBanks(runtime, p['calibration']['bank_directory'])
        parameters = banks.parameters()
        if len(parameters) != 1228 or any(value.requires_grad for value in runtime.model.parameters()):
            raise ValueError('Only the original 614 books/log-row-gain pairs may train')
        data = {}

        def datum(case):
            if case not in data:
                obs, meta = calibration_source.load_observation(np, Path(p['calibration']['plan_file']).parent, rows[case])
                if case in saved_rows and meta != saved_rows[case]['metadata']:
                    raise ValueError('Original C8 observation metadata differs')
                data[case] = runtime.datum(obs), meta
            value, meta = data[case]
            runtime.description, runtime.current_arm = meta['description'], arm
            return value

        def teacher(split, case, seed):
            if split == 'calibration':
                return cache['teacher'][str(case)]
            target = torch.load(Path(p['teacher_heldout_directory']) / f'{case}_{seed}.pt',
                                map_location='cpu', weights_only=True)
            if target['plan_row'] != rows[case] or target['sampler_seed'] != rows[case]['sampler_seeds'][seed]:
                raise ValueError('Heldout teacher binding differs')
            return target

        def student(case, seed, gradient):
            value = datum(case)
            banks.seen.clear()
            count('new_model_queries')
            with torch.set_grad_enabled(gradient):
                action, counts, future = sampler.differentiable_action(
                    runtime, value, rows[case]['sampler_seeds'][seed],
                    checkpointed=gradient, return_future=True)
            if len(banks.seen) != 614 or counts != {'video_steps': 10, 'action_steps': 10}:
                raise ValueError('Original student coverage or sampler steps differ')
            if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all()):
                raise FloatingPointError('Nonfinite student output')
            return action, future, counts

        def read_action(reader, case, future, gradient=False, seed=0):
            count('new_bf16_reader_queries')
            return reader(runtime, datum(case), rows[case]['sampler_seeds'][seed],
                          future, checkpointed=gradient)

        # Only KL needs initial reader calibration; neither arm uses heldout here.
        prep_path = root / 'results/reader_preparation/summary.json'
        if arm == 'az':
            prep = {'protocol': p['protocol'], 'frozen_protocol': p, 'complete': True,
                    'heldout_used': False, 'stats': stats, 'betas': {'az': 0.0}}
        elif prep_path.is_file():
            prep = read(prep_path)
            if prep['frozen_protocol'] != p or not prep['complete'] or prep['heldout_used']:
                raise ValueError('Frozen reader preparation differs')
        else:
            initial_rows = []
            for case in ids:
                deadline.check(300)
                saved_path = root / 'results/reader_preparation/cases' / f'{case}.json'
                if saved_path.is_file():
                    record = read(saved_path)
                    if record['case_id'] != case or record['protocol'] != p['protocol']:
                        raise ValueError('Initial reader loss binding differs')
                    initial_rows.append(record)
                    continue
                progress('initial_reader_calibration', case_id=case, initial_cases=len(initial_rows))
                if str(case) in cache['parent']:
                    future_cpu = cache['parent'][str(case)]['future']
                else:
                    initial_a, initial_z, _ = student(case, 0, False)
                    future_cpu = initial_z.detach().cpu()
                    del initial_a, initial_z
                z = cache['teacher'][str(case)]['future'].float()[:, :, 1:]
                nmse = float((future_cpu.float()[:, :, 1:] - z).square().sum() / z.square().sum().clamp_min(1e-12))
                with reader_session(torch, runtime, banks, masters) as reader:
                    if case == ids[0]:
                        reference, _ = read_action(reader, case, cache['teacher'][str(case)]['future'], False)
                        reader_parity = parity(reference, cache['teacher'][str(case)]['action'])
                        if reader_parity['max_abs'] != 0:
                            raise ValueError('Frozen-reader teacher replay must be exact')
                        del reference
                    action, counts = read_action(reader, case, future_cpu, False)
                    values = reader_terms(torch, action, cache['teacher'][str(case)]['action'], stats, p['reader'])
                    record = {'protocol': p['protocol'], 'case_id': case, 'latent_nmse': nmse,
                              **{key: float(value) for key, value in values.items()},
                              'counts': compact_counts(counts), 'pbs_jobid': os.environ['PBS_JOBID']}
                    if case == ids[0]:
                        record['teacher_parity'] = reader_parity
                    del action, values, counts
                del future_cpu, z
                save(saved_path, record)
                initial_rows.append(record)
            means = {key: sum(row[key] for row in initial_rows) / len(ids)
                     for key in ('reader_kl', 'reader_mse', 'latent_nmse')}
            contribution = p['reader']['coefficient_fraction'] * p['training']['rho'] * p['training']['normalization_c'] * means['latent_nmse']
            if any(not math.isfinite(value) or value <= 0 for value in means.values()):
                raise ValueError('Initial C80 reader losses must be nondegenerate')
            prep = {'protocol': p['protocol'], 'frozen_protocol': p, 'complete': True, 'heldout_used': False,
                    'calibration_case_ids': ids, 'stats': stats, 'initial_rows': initial_rows,
                    'initial_means': means, 'initial_weighted_contribution': contribution,
                    'betas': {'reader_kl': contribution / means['reader_kl']},
                    'pbs_jobid': os.environ['PBS_JOBID']}
            save(prep_path, prep)
        beta = prep['betas'][arm]
        optimizer = torch.optim.Adam(parameters, lr=p['training']['learning_rate'])
        checkpoint = out / 'checkpoint.pt'
        updates, history, initial_parity = 0, [], None
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['protocol'] != p or saved['preparation'] != prep or saved['arm'] != arm:
                raise ValueError('Resume recipe differs')
            updates, history, initial_parity = saved['updates'], saved['history'], saved['initial_parity']
            if not 0 <= updates <= total_updates or [row['case_id'] for row in history] != order[:updates]:
                raise ValueError('Committed update order differs')
            banks.restore(saved['module_masters'])
            optimizer.load_state_dict(saved['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda')
            restore_rng(torch, saved['rng'])
            for key in ('new_model_queries', 'new_bf16_reader_queries', 'parameter_reverse_passes'):
                state[key] = max(state[key], saved['counters'][key])
        while updates < total_updates:
            deadline.check(max([row['seconds'] for row in history[-2:]], default=900) * 1.2)
            case = order[updates]
            started = time.monotonic()
            progress('training', updates=updates, case_id=case, beta=beta)
            optimizer.zero_grad(set_to_none=True)
            action, future, counts = student(case, 0, False)
            action_cpu, future_cpu = action.detach().cpu(), future.detach().cpu()
            if updates == 0:
                initial_parity = {key: parity(actual, cache['parent'][str(case)][key])
                                  for key, actual in [('action', action), ('future', future)]}
                if any(value['max_abs'] != 0 for value in initial_parity.values()):
                    raise ValueError('Original untrained C8 bank initialization differs')
            a_leaf = action.detach().requires_grad_(True)
            z_leaf = future.detach().requires_grad_(True)
            values = base_terms(torch, a_leaf, z_leaf, teacher('calibration', case, 0))
            base = base_objective(values, p)
            ga, gz = torch.autograd.grad(base, (a_leaf, z_leaf))
            base_values = {key: float(value.detach()) for key, value in values.items()}
            base_value = float(base.detach())
            del action, future, a_leaf, z_leaf, values, base
            extra = {}
            if arm == 'reader_kl':
                with reader_session(torch, runtime, banks, masters) as reader:
                    leaf = future_cpu.to(device=next(runtime.model.parameters()).device).detach().requires_grad_(True)
                    plain = None
                    if updates == 0:
                        plain, _ = read_action(reader, case, leaf.detach(), False)
                        plain = plain.detach().cpu()
                    reader_action, reader_counts = read_action(reader, case, leaf, True)
                    if plain is not None and parity(reader_action, plain)['max_abs'] != 0:
                        raise ValueError('Reader grad/no-grad parity differs')
                    rvalues = reader_terms(torch, reader_action, cache['teacher'][str(case)]['action'], stats, p['reader'])
                    rg, = torch.autograd.grad(beta * rvalues['reader_kl'], leaf)
                    if not bool(torch.isfinite(rg).all()) or bool((rg[:, :, :1] != 0).any()) or not bool((rg[:, :, 1:] != 0).any()):
                        raise ValueError('Reader cotangent must be finite, nonzero and exclude frame0')
                    gz = gz + rg.detach()
                    extra = {key: float(value.detach()) for key, value in rvalues.items()}
                    extra['weighted_reader_future_gradient_norm'] = float(rg.float().norm())
                    extra['reader_counts'] = compact_counts(reader_counts)
                    del leaf, reader_action, reader_counts, rvalues, rg, plain
            replay_a, replay_z, counts = student(case, 0, True)
            replay_parity = {'action': parity(replay_a, action_cpu), 'future': parity(replay_z, future_cpu)}
            if any(value['max_abs'] != 0 for value in replay_parity.values()):
                raise ValueError('Staged student replay must match exactly before VJP')
            torch.autograd.backward((replay_a, replay_z), (ga.to(replay_a), gz.to(replay_z)))
            count('parameter_reverse_passes')
            coverage = banks.gradient_report()
            if coverage['connected'] != 614 or coverage['finite'] != 614 or coverage['nonzero'] < 1:
                raise ValueError('Original bank-gradient coverage failed')
            if any(value.grad is not None for value in runtime.model.parameters()):
                raise ValueError('Frozen model accumulated parameter gradients')
            norm = torch.nn.utils.clip_grad_norm_(parameters, p['training']['gradient_norm_clip'], error_if_nonfinite=True)
            optimizer.step()
            if updates == 0:
                optimizer.state.clear()
            banks.refresh()
            del replay_a, replay_z, ga, gz
            torch.cuda.synchronize()
            updates += 1
            actual_step = None
            if updates == 1:
                after_a, after_z, _ = student(case, 0, False)
                after_values = base_terms(torch, after_a, after_z, teacher('calibration', case, 0))
                actual_step = {key: float(value.detach()) for key, value in after_values.items()}
                actual_step['first10motor_rmse'] = math.sqrt(actual_step['action_mse'])
                actual_step['output_changed'] = parity(after_a, action_cpu)['max_abs'] > 0 or parity(after_z, future_cpu)['max_abs'] > 0
                if not actual_step['output_changed']:
                    raise ValueError('First optimizer step produced no deployed output change')
                del after_a, after_z, after_values
            history.append({'update': updates, 'case_id': case, 'seed_index': 0, **base_values, **extra,
                            'baseline_objective': base_value, 'loss': base_value + (beta * extra['reader_kl'] if beta else 0.0),
                            'beta': beta, 'coverage': coverage, 'gradient_norm': float(norm),
                            'staged_replay_parity': replay_parity, 'actual_first_step': actual_step,
                            'seconds': time.monotonic() - started, 'pbs_jobid': os.environ['PBS_JOBID'], **counts})
            atomic_torch_save(torch, checkpoint, {'protocol': p, 'arm': arm, 'preparation': prep, 'updates': updates,
                              'history': history, 'initial_parity': initial_parity, 'module_masters': banks.snapshot(),
                              'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
                              'counters': {key: state[key] for key in ('new_model_queries', 'new_bf16_reader_queries', 'parameter_reverse_passes')}})
            del action_cpu, future_cpu
            gc.collect()
            torch.cuda.empty_cache()
            progress('training_committed', updates=updates, last_update=history[-1])
        for value in parameters:
            value.requires_grad_(False)
            value.grad = None
        optimizer.state.clear()
        atomic_torch_save(torch, out / 'deployment.pt', {'protocol': p, 'arm': arm, 'updates': total_updates,
                          'target_linears': 614, 'module_masters': banks.snapshot(), 'preparation': prep,
                          'reference_binding': cache['binding'], 'initial_parity': initial_parity})
        endpoint_count, actual_parity = 0, None
        for split in ('calibration', 'heldout'):
            for case, seed in queries(p, split):
                deadline.check()
                expected = {'protocol': p['protocol'], 'arm': arm, 'split': split, 'updates': total_updates,
                            'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                            'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed]}
                tensor_path = out / 'tensors' / split / f'{case}_{seed}.pt'
                if tensor_path.is_file():
                    saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
                    if any(saved.get(key) != value for key, value in expected.items()):
                        raise ValueError('Committed evaluation binding differs')
                    action, future, counts = saved['action'], saved['future'], saved['counts']
                else:
                    action, future, counts = student(case, seed, False)
                    action, future = action.detach().cpu(), future.detach().cpu()
                    tensor_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_torch_save(torch, tensor_path, {**expected, 'action': action, 'future': future, 'counts': counts})
                if split == 'calibration' and case == ids[0]:
                    actual, _ = runtime.infer(datum(case), rows[case]['sampler_seeds'][seed], measure=False)
                    count('new_model_queries')
                    actual_parity = parity(actual, action)
                    if actual_parity['max_abs'] > 1e-6:
                        raise ValueError('Final native inference/sampler parity failed')
                    del actual
                target = teacher(split, case, seed)
                values = base_terms(torch, action, future, target)
                delta = action.float() - target['action'].float()
                rmse = lambda value: float(value.square().mean().sqrt())
                record = {**expected, 'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts,
                          'first10motor_rmse': rmse(delta[:10, :6]), 'full32motor_rmse': rmse(delta[:, :6]),
                          'late22motor_rmse': rmse(delta[10:, :6]), 'first10gripper_rmse': rmse(delta[:10, 6]),
                          'future_latent_nmse': float(values['latent_nmse']), 'first10motor_mse': float(values['action_mse']),
                          'mixed_loss': float(base_objective(values, p))}
                record_path = out / split / 'queries' / f'{case}_{seed}.json'
                if record_path.is_file():
                    old_record = read(record_path)
                    for key in ('reader_kl', 'reader_mse', 'reader_first10motor_rmse', 'reader_gap_first10motor_rmse', 'reader_error_cross_term', 'reader_counts'):
                        if key in old_record:
                            record[key] = old_record[key]
                save(record_path, record)
                endpoint_count += 1
                progress('evaluation', updates=total_updates, split=split, queries=endpoint_count, last_case=case, last_seed=seed)
                del action, future, target, values
        reader_endpoint_count = 0
        reader_parity_path = out / 'reader_teacher_parity.json'
        with reader_session(torch, runtime, banks, masters) as reader:
            if not reader_parity_path.is_file():
                reference, _ = read_action(reader, ids[0], cache['teacher'][str(ids[0])]['future'])
                reader_parity = parity(reference, cache['teacher'][str(ids[0])]['action'])
                if reader_parity['max_abs'] != 0:
                    raise ValueError('Final frozen-reader teacher replay differs')
                save(reader_parity_path, {'protocol': p['protocol'], 'arm': arm, 'parity': reader_parity})
                del reference
            for split in ('calibration', 'heldout'):
                for case, seed in queries(p, split):
                    deadline.check()
                    record_path = out / split / 'queries' / f'{case}_{seed}.json'
                    record = read(record_path)
                    if 'reader_kl' not in record:
                        tensor_path = out / 'tensors' / split / f'{case}_{seed}.pt'
                        saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
                        target = teacher(split, case, seed)
                        reader_action, counts = read_action(reader, case, saved['future'], seed=seed)
                        reader_action = reader_action.detach().cpu()
                        rvalues = reader_terms(torch, reader_action, target['action'], stats, p['reader'])
                        effect = reader_action[:10, :6].float() - target['action'][:10, :6].float()
                        gap = saved['action'][:10, :6].float() - reader_action[:10, :6].float()
                        record.update({key: float(value) for key, value in rvalues.items()})
                        record.update(reader_first10motor_rmse=float(effect.square().mean().sqrt()),
                                      reader_gap_first10motor_rmse=float(gap.square().mean().sqrt()),
                                      reader_error_cross_term=float(2 * (effect * gap).mean()),
                                      reader_counts=compact_counts(counts))
                        reconstructed = float(effect.square().mean() + gap.square().mean() + 2 * (effect * gap).mean())
                        if not math.isclose(reconstructed, record['first10motor_mse'], rel_tol=1e-5, abs_tol=1e-8):
                            raise ValueError('Student/reader error decomposition failed')
                        reader_tensor = out / 'reader_tensors' / split / f'{case}_{seed}.pt'
                        reader_tensor.parent.mkdir(parents=True, exist_ok=True)
                        atomic_torch_save(torch, reader_tensor, {'protocol': p['protocol'], 'arm': arm,
                                          'case_id': case, 'seed_index': seed, 'action': reader_action})
                        save(record_path, record)
                        del saved, target, reader_action, rvalues, effect, gap
                    reader_endpoint_count += 1
                    progress('reader_evaluation', updates=total_updates, split=split,
                             queries=endpoint_count, reader_queries=reader_endpoint_count,
                             last_case=case, last_seed=seed)
        receipt = {'protocol': p['protocol'], 'arm': arm, 'complete': True, 'updates': total_updates, 'history': history,
                   'update_order': order, 'initial_parity': initial_parity, 'actual_infer_parity': actual_parity,
                   'optimizer_reset_after_update1': True, 'activation_clipping': False, 'per_channel_scaling': False,
                   'beta': beta, 'pbs_jobid': os.environ['PBS_JOBID']}
        save(out / 'training_receipt.json', receipt)
        summary = {**state, 'complete': True, 'phase': 'complete', 'updates': total_updates,
                   'evaluation_queries': endpoint_count, 'reader_evaluation_queries': reader_endpoint_count,
                   'training_receipt': receipt, 'preparation': prep}
        save(out / 'summary.json', summary)
        progress('complete', complete=True, updates=total_updates, queries=endpoint_count,
                 reader_queries=reader_endpoint_count)
        print('FROZEN_READER_WORKER_COMPLETE', flush=True)
        return 0
    except TimeoutError as error:
        progress('checkpointed', reason=str(error))
        return 75
    except Exception as error:
        progress('failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--unit', choices=('reader_kl', 'az'), required=True)
    args = parser.parse_args()
    raise SystemExit(run(read(args.protocol), args.unit))
