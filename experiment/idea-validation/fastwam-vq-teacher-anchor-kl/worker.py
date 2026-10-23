"""Fresh80 additive latent KL and exact replay of the original MSE/NMSE control."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80
from anchor_loss import loss_terms, objectives
from anchor_selfcheck import require_allocation


def read(path):
    return json.loads(Path(path).read_text())


def validate_sources(p, prep, stats, artifact):
    if p['protocol'] != 'fastwam-vq-teacher-anchor-kl-v1':
        raise RuntimeError('Unexpected protocol')
    source = read(Path(p['source_per_channel_root']) / 'protocol.json')
    if read(artifact / 'source_protocol.json') != source:
        raise RuntimeError('Runtime snapshot differs from original per-channel source')
    for key in ('calibration', 'clipping', 'scaling', 'source_clipping_protocol'):
        if p[key] != source[key]:
            raise RuntimeError('Loss-only invariant changed: ' + key)
    for key in ('additional_updates', 'epochs', 'weight_learning_rate', 'clip_learning_rate',
                'optimizer', 'fresh_optimizer_each_arm', 'optimizer_reset_after_update1',
                'gradient_norm_clip', 'rho', 'normalization_c'):
        if p['training'][key] != source['training'][key]:
            raise RuntimeError('Training setting changed: ' + key)
    for key in ('case_ids', 'seed_indices', 'aggregation'):
        if p['evaluation'][key] != source['evaluation'][key]:
            raise RuntimeError('Original heldout changed: ' + key)
    for value in (prep, stats):
        if (not value['complete'] or value['heldout_used']
                or value['calibration_case_ids'] != p['calibration']['case_ids']):
            raise RuntimeError('Complete calibration-only preparation required')
    if not prep['passed']:
        raise RuntimeError('KL definition checks did not pass')
    for name, keys in (('anchor_selfcheck', ('passed',)),
                       ('channel_selfcheck', ('passed', 'independent_reference_passed', 'old_surrogate_rejected'))):
        check = read(artifact / (name + '.json'))
        if check['pbs_jobid'] != os.environ['PBS_JOBID'] or not all(check.get(k) is True for k in keys):
            raise RuntimeError('Same-allocation selfcheck required: ' + name)


def run(p, unit):
    require_allocation()
    import numpy as np
    import torch
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from channel_banks import ChannelBanks
    from metrics_reference import metrics
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, optimizer_to_device, to_cpu

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    root = Path(p['remote_root']); artifact = Path(os.environ['ARTIFACTS'])
    out = root / 'results' / unit; out.mkdir(parents=True, exist_ok=True)
    prep = read(root / 'results/prepare/summary.json')
    stats = read(Path(p['source_dual_root']) / 'results/fit_stats/summary.json')
    validate_sources(p, prep, stats, artifact)
    data_root, rows, _, base_deployment, reference_action = ref.load_inputs(p['source_clipping_protocol'], torch)
    teacher_dirs = {'calibration': Path(p['control_root']) / 'results/teacher/calibration',
                    'heldout': Path(p['clipping_root']) / 'results/teacher'}
    for split, directory in teacher_dirs.items():
        receipt = read(directory / 'receipt.json')
        ids = p['calibration' if split == 'calibration' else 'evaluation']['case_ids']
        if (not receipt['complete'] or receipt['queries'] != 80 or receipt['case_ids'] != ids
                or receipt['seed_indices'] != ([0] if split == 'calibration' else [0, 1])):
            raise RuntimeError('Original shared teacher coverage differs')
    state = {'protocol': p['protocol'], 'unit': unit, 'pbs_jobid': os.environ['PBS_JOBID'],
             'complete': False, 'phase': 'loading', 'updates': 0, 'new_teacher_queries': 0,
             'new_model_queries': 0}

    def progress(**values):
        state.update(values); ref.save_json(out / 'progress.json', state)
        print('ANCHOR_PROGRESS ' + json.dumps(state), flush=True)

    runtime = None; deadline = ref.Deadline(); progress()
    try:
        runtime = runner.Runtime(out); runtime.model.eval(); runtime.arm('bf16'); data = {}
        banks = ChannelBanks(runtime, p['calibration']['bank_directory'], base_deployment['module_masters'], p['clipping'], p['scaling'])
        if any(bool((v != 0).any()) for v in banks.shadows.values()) or any(float(v) != 1 for v in banks.ratios.values()):
            raise RuntimeError('Fresh D=1/r=1 initialization required')
        groups = banks.optimizer_groups(p['training'], p['scaling'])
        parameters = [v for group in groups for v in group['params']]
        if (len(banks.ratios) != 614 or sum(v.numel() for v in banks.shadows.values()) != 1_806_863
                or {id(v) for v in parameters} != {id(v) for v in list(banks.ratios.values()) + list(banks.shadows.values())}
                or any(v.requires_grad for v in runtime.model.parameters())
                or any(v.requires_grad for v in list(banks.books.values()) + list(banks.logs.values()))):
            raise RuntimeError('Original trainable/frozen parameter coverage differs')

        def datum(case):
            if case not in data:
                obs, meta = calibration_source.load_observation(np, data_root, rows[case])
                data[case] = runtime.datum(obs), meta
            return data[case]

        def query(case, seed):
            deadline.check(); value, meta = datum(case)
            result = ref.query(runtime, sampler, torch, value, meta['description'], rows[case]['sampler_seeds'][seed], banks)
            state['new_model_queries'] += 1
            return result

        def teacher(split, case, seed):
            target = torch.load(teacher_dirs[split] / f'{case}_{seed}.pt', map_location='cpu', weights_only=True)
            if target['plan_row'] != rows[case] or target['sampler_seed'] != rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Teacher input/seed binding differs')
            return target

        def terms(action, future, target):
            return loss_terms(torch, action, future, target, stats, p['anchor'], prep['temperature'])

        first = p['calibration']['case_ids'][0]
        initial = query(first, 0)
        initial_action_reference, initial_future_reference = initial[0], initial[1]
        parity = float((initial[0].float() - torch.tensor(reference_action['action'], dtype=torch.float32)).abs().max())
        if parity != 0:
            raise RuntimeError('Original baseline action replay differs')
        post_initial_rng = capture_rng(torch)
        progress(initial_parity_max_abs=parity)
        coefficient_path = root / 'results/anchor_kl/loss_scaling.json'
        if unit == 'anchor_kl':
            if not coefficient_path.is_file():
                initial_rows = []
                for case in p['calibration']['case_ids']:
                    path = out / 'initial' / f'{case}_0.json'
                    if path.is_file():
                        record = read(path)
                        if record['protocol'] != p['protocol'] or record['plan_row'] != rows[case]:
                            raise RuntimeError('Initial calibration cache binding differs')
                    else:
                        action, future, counts = initial if case == first else query(case, 0)
                        target = teacher('calibration', case, 0)
                        values = terms(action, future, target)
                        measured = metrics(torch, latent_reference, action, future, target, p['training'])
                        old = read(Path(p['control_root']) / 'results/calibration/baseline/queries' / f'{case}_0.json')
                        if old['plan_row'] != rows[case] or any(measured[m] != old[m] for m in c80.METRICS):
                            raise RuntimeError('Full C80 original initialization replay differs')
                        record = {'protocol': p['protocol'], 'case_id': case, 'seed_index': 0,
                                  'plan_row': rows[case], 'pbs_jobid': os.environ['PBS_JOBID'],
                                  **{k: float(v) for k, v in values.items()}}
                        ref.save_json(path, record)
                        del action, future, target, values
                    initial_rows.append(record)
                    progress(phase='initial_calibration', queries=len(initial_rows), last_case=case)
                means = {key: sum(row[key] for row in initial_rows) / 80 for key in ('action_mse', 'latent_nmse', 'latent_anchor_kl')}
                if any(not math.isfinite(v) or v <= 0 for v in means.values()):
                    raise RuntimeError('Nondegenerate complete-C80 initial means required')
                latent_weight = p['training']['rho'] * p['training']['normalization_c']
                coefficient = {'protocol': p['protocol'], 'complete': True, 'case_ids': p['calibration']['case_ids'],
                               'seed_indices': [0], 'heldout_used': False, 'initial_means': means,
                               'beta': p['anchor']['coefficient_fraction'] * latent_weight * means['latent_nmse'] / means['latent_anchor_kl'],
                               'original_latent_weight': latent_weight, 'temperature': prep['temperature'],
                               'rule': p['anchor']['coefficient_rule'], 'pbs_jobid': os.environ['PBS_JOBID']}
                ref.save_json(coefficient_path, coefficient)
            coefficient = read(coefficient_path)
            if (coefficient['protocol'] != p['protocol'] or not coefficient['complete'] or coefficient['heldout_used']
                    or coefficient['case_ids'] != p['calibration']['case_ids'] or coefficient['temperature'] != prep['temperature']):
                raise RuntimeError('Frozen additive coefficient differs')
            beta = coefficient['beta']
            if not math.isfinite(beta) or beta <= 0:
                raise RuntimeError('Positive finite beta required')
            restore_rng(torch, post_initial_rng)
            del initial
            optimizer = torch.optim.Adam(groups); checkpoint = out / 'checkpoint.pt'
            updates, history = 0, []
            if checkpoint.is_file():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved['protocol'] != p or saved['unit'] != unit or saved['loss_scaling'] != coefficient:
                    raise RuntimeError('Resume contract differs')
                updates, history = saved['updates'], saved['history']
                if [h['case_id'] for h in history] != p['calibration']['update_order'][:updates]:
                    raise RuntimeError('Committed update order differs')
                banks.restore_clips(saved['ratios']); banks.restore_scales(saved['shadow_log2_scales'])
                optimizer.load_state_dict(saved['optimizer']); optimizer_to_device(torch, optimizer, 'cuda')
                restore_rng(torch, saved['rng'])
            while updates < 80:
                deadline.check(max([h['seconds'] for h in history[-3:]], default=300) * 1.15)
                case = p['calibration']['update_order'][updates]; value, meta = datum(case)
                runtime.description = meta['description']; banks.seen.clear()
                target = teacher('calibration', case, 0)
                optimizer.zero_grad(set_to_none=True); started = time.monotonic()
                action, counts, future = sampler.differentiable_action(runtime, value, rows[case]['sampler_seeds'][0], checkpointed=True, return_future=True)
                state['new_model_queries'] += 1
                values = terms(action, future, target)
                base_loss, loss = objectives(values, p, beta)
                if len(banks.seen) != 614 or not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError('Training loss/Linear coverage failed')
                extra = {}
                if updates == 0:
                    gradient_action_error = float((action.detach().float().cpu() - initial_action_reference.float()).abs().max())
                    gradient_future_error = float((future.detach().float().cpu() - initial_future_reference.float()).abs().max())
                    if gradient_action_error != 0 or gradient_future_error != 0:
                        raise RuntimeError('First checkpointed gradient forward differs from initialization')
                    extra.update(gradient_forward_action_max_abs=gradient_action_error,
                                 gradient_forward_future_max_abs=gradient_future_error)
                    cotangent = torch.autograd.grad(values['latent_anchor_kl'], future, retain_graph=True)[0]
                    if (not bool(torch.isfinite(cotangent).all()) or not bool((cotangent[:, :, 1:] != 0).any())
                            or bool((cotangent[:, :, :1] != 0).any())):
                        raise RuntimeError('New KL future gradient is invalid')
                    extra['anchor_future_gradient_norm'] = float(cotangent.float().norm())
                    del cotangent
                loss.backward(); coverage = banks.coverage()
                if any(v['connected'] != 614 or v['finite'] != 614 or v['nonzero'] < 1 for v in coverage.values()):
                    raise RuntimeError('Ratio/channel gradients disconnected or nonfinite')
                norm = torch.nn.utils.clip_grad_norm_(parameters, p['training']['gradient_norm_clip'], error_if_nonfinite=True)
                optimizer.step()
                if updates == 0 and p['training']['optimizer_reset_after_update1']:
                    optimizer.state.clear()
                banks.project(); torch.cuda.synchronize(); updates += 1
                history.append({'update': updates, 'case_id': case, 'seed_index': 0,
                                **{k: float(v.detach()) for k, v in values.items()},
                                'baseline_objective': float(base_loss.detach()), 'loss': float(loss.detach()),
                                'beta': beta, 'gradient_norm': float(norm), 'coverage': coverage,
                                'seconds': time.monotonic() - started, 'pbs_jobid': os.environ['PBS_JOBID'],
                                **extra, **counts})
                del action, future, target, values, base_loss, loss
                atomic_torch_save(torch, checkpoint, {'protocol': p, 'unit': unit, 'loss_scaling': coefficient,
                                  'updates': updates, 'history': history, 'ratios': banks.clip_snapshot(),
                                  'shadow_log2_scales': banks.scale_snapshot(), 'optimizer': to_cpu(optimizer.state_dict()),
                                  'rng': capture_rng(torch)})
                progress(phase='training', updates=updates, last_update=history[-1])
        else:
            del initial
            old_receipt = read(Path(p['source_per_channel_root']) / 'results/train_eval/training_receipt.json')
            saved = torch.load(Path(p['source_per_channel_root']) / 'results/train_eval/deployment.pt', map_location='cpu', weights_only=True)
            if (not old_receipt['complete'] or old_receipt['updates'] != 80 or saved['updates'] != 80
                    or old_receipt['pbs_jobid'] != p['source_per_channel_job']
                    or old_receipt['update_order'] != p['calibration']['update_order']):
                raise RuntimeError('Original exact80 MSE/NMSE control differs')
            banks.restore_clips(saved['ratios']); banks.restore_scales(saved['log2_scales'])
            updates, history, coefficient, beta = 80, [], None, 0.

        progress(phase='deployment_parity', updates=80)
        uncached = query(first, 0); rounding = banks.cache_compensated_weights(); cached = query(first, 0)
        folded = {'action_max_abs': float((uncached[0].float() - cached[0].float()).abs().max()),
                  'future_max_abs': float((uncached[1].float() - cached[1].float()).abs().max())}
        if any(v != 0 for v in folded.values()):
            raise RuntimeError('Uncached/cached action or future differs')
        runtime.current_arm = unit; runtime.description = datum(first)[1]['description']
        actual, _ = runtime.infer(datum(first)[0], rows[first]['sampler_seeds'][0], measure=False)
        state['new_model_queries'] += 1
        actual_parity = float((actual.float().cpu() - cached[0].float()).abs().max())
        if actual_parity != 0:
            raise RuntimeError('Native inference action differs')
        del uncached, actual
        frozen = banks.verify_frozen(base_deployment['module_masters'])
        if unit == 'anchor_kl':
            atomic_torch_save(torch, out / 'deployment.pt', {'protocol': p, 'unit': unit, 'updates': 80,
                              'baseline_checkpoint': str(Path(p['baseline_root']) / 'results/training/rho1.5/deployment.pt'),
                              'ratios': banks.clip_snapshot(), 'log2_scales': banks.scale_snapshot(),
                              'scale_metadata_dtype': 'float32_log2'})
            atomic_torch_save(torch, out / 'reference_endpoint.pt', {'protocol': p['protocol'], 'unit': unit,
                              'updates': 80, 'pbs_jobid': os.environ['PBS_JOBID'], 'case_id': first, 'seed_index': 0, 'plan_row': rows[first],
                              'sampler_seed': rows[first]['sampler_seeds'][0], 'action': cached[0].float().cpu(),
                              'future': cached[1].float().cpu(), 'counts': cached[2]})
        training = {'protocol': p['protocol'], 'complete': True, 'unit': unit, 'updates': 80,
                    'history': history, 'update_order': p['calibration']['update_order'], 'loss_scaling': coefficient,
                    'reused_training': unit == 'mse_nmse', 'source_job_id': p['source_per_channel_job'] if unit == 'mse_nmse' else None,
                    'initial_parity_max_abs': parity, 'cached_query_parity': folded,
                    'actual_infer_parity_max_abs': actual_parity, 'clipping': banks.clip_summary(),
                    'scaling': banks.scaling_summary(), 'pbs_jobid': os.environ['PBS_JOBID'], **frozen}
        ref.save_json(out / 'training_receipt.json', training)
        ref.save_json(out / 'weight_rounding.json', rounding)
        cache_first = cached; endpoint_count = 0
        for split in ('calibration', 'heldout'):
            records = []
            for case, seed in c80.queries(p, split):
                deadline.check()
                path = root / 'results' / split / unit / 'queries' / f'{case}_{seed}.json'
                tensor_path = out / 'tensors' / split / f'{case}_{seed}.pt'; tensor_path.parent.mkdir(parents=True, exist_ok=True)
                expected = {'protocol': p['protocol'], 'arm': unit, 'split': split, 'updates': 80,
                            'plan_row': rows[case], 'case_id': case, 'suite': rows[case]['suite'],
                            'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed]}
                target = teacher(split, case, seed)
                if path.is_file() and not tensor_path.is_file():
                    raise RuntimeError('Endpoint record lacks its reusable tensor')
                if tensor_path.is_file():
                    saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Saved endpoint binding differs')
                    action, future, counts = saved['action'], saved['future'], saved['counts']
                elif split == 'calibration' and case == first and seed == 0:
                    action, future, counts = cache_first
                else:
                    action, future, counts = query(case, seed)
                values = terms(action, future, target)
                measured = metrics(torch, latent_reference, action, future, target, p['training'])
                if unit == 'mse_nmse':
                    old = read(Path(p['source_per_channel_root']) / 'results' / split / 'per_channel_clip/queries' / f'{case}_{seed}.json')
                    if old['plan_row'] != rows[case] or any(old[m] != measured[m] for m in c80.METRICS):
                        raise RuntimeError('Original MSE/NMSE control endpoint replay differs')
                commands = runtime.command(action.detach().float().cpu().clone()).tolist()
                teacher_commands = runtime.command(target['action'].detach().float().cpu().clone()).tolist()
                signs = [int(row[6]) for row in commands[:10]]
                teacher_signs = [int(row[6]) for row in teacher_commands[:10]]
                record = {**expected, 'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts, **measured,
                          **{k: float(v) for k, v in values.items()},
                          'baseline_objective': float(objectives(values, p, 0.)[0]),
                          'gripper_command_errors': sum(a != b for a, b in zip(signs, teacher_signs)),
                          'gripper_commands': signs, 'teacher_gripper_commands': teacher_signs,
                          'gripper_events': [[i, signs[i - 1], signs[i]] for i in range(1, 10) if signs[i] != signs[i - 1]],
                          'teacher_gripper_events': [[i, teacher_signs[i - 1], teacher_signs[i]] for i in range(1, 10) if teacher_signs[i] != teacher_signs[i - 1]]}
                if not tensor_path.is_file():
                    atomic_torch_save(torch, tensor_path, {**expected, 'action': action.float().cpu(),
                                      'future': future.float().cpu(), 'counts': counts})
                ref.save_json(path, record); records.append(record); endpoint_count += 1
                progress(phase='evaluation', split=split, queries=len(records), last_case=case, last_seed=seed)
                del action, future, target, values
            summary = {'protocol': p['protocol'], 'complete': True, 'arm': unit, 'split': split,
                       'updates': 80, 'pbs_jobid': os.environ['PBS_JOBID'], **c80.aggregate_records(records, p, split)}
            ref.save_json(root / 'results' / split / unit / 'summary.json', summary)
        summary = {**state, 'complete': True, 'phase': 'complete', 'updates': 80,
                   'evaluation_queries': endpoint_count, 'training_receipt': training, 'new_teacher_queries': 0}
        ref.save_json(out / 'summary.json', summary); progress(phase='complete', complete=True, updates=80)
        print('ANCHOR_WORKER_COMPLETE', flush=True)
        return 0
    except TimeoutError as error:
        progress(phase='checkpointed', reason=str(error)); return 75
    except Exception as error:
        progress(phase='failed', error=type(error).__name__ + ': ' + str(error)); raise
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True)
    parser.add_argument('--unit', choices=('mse_nmse', 'anchor_kl'), required=True)
    args = parser.parse_args(); return run(read(args.protocol), args.unit)


if __name__ == '__main__':
    raise SystemExit(main())
