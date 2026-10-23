"""Two fixed-budget KL stages: C8 VQ books/gains, then unchanged C80 D/r."""
import argparse
import json
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80
import relational_reference as prior


def read(path):
    return json.loads(Path(path).read_text())


def parity(torch, actual, expected):
    error = actual.float().cpu() - expected.float().cpu()
    return {'max_abs': float(error.abs().max()), 'rmse': float(error.square().mean().sqrt())}


def fixed_encodings(banks):
    return {name: {key: (value[key].data_ptr(), value[key]._version)
                   for key in ('indices', 'scale_exponents')}
            for name, value in banks.encodings.items()}


def verify_encodings(banks, identities):
    for name, keys in identities.items():
        for key, identity in keys.items():
            value = banks.encodings[name][key]
            if (value.data_ptr(), value._version) != identity:
                raise RuntimeError('Fixed VQ indices/exponents changed')
    return {'passed': True, 'indices_unchanged': True, 'exponents_unchanged': True,
            'transforms_trainable': False, 'bias_trainable': False}


def run(p, unit):
    import numpy as np
    import torch
    import runner
    import sampler
    import calibration_source
    import latent_reference
    from train_banks import TrainableBanks
    from channel_banks import ChannelBanks
    from metrics_reference import metrics
    from softbin_loss import losses as softbin_losses
    from relational_loss import relational_losses
    from checkpointing import (atomic_torch_save, capture_rng, restore_rng,
                               optimizer_to_device, to_cpu)

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    root, source = Path(p['remote_root']), Path(p['source_relational_root'])
    old = read(source / 'protocol.json')
    for key in ('calibration', 'training', 'clipping', 'scaling', 'softbin', 'source_clipping_protocol'):
        if p[key] != old[key]:
            raise RuntimeError('Frozen activation protocol differs: ' + key)
    for key in ('case_ids', 'seed_indices', 'aggregation'):
        if p['evaluation'][key] != old['evaluation'][key]:
            raise RuntimeError('Original heldout cohort differs')
    prep = read(source / 'results/prepare/summary.json')
    stats = read(Path(p['source_dual_root']) / 'results/fit_stats/summary.json')
    scale = read(Path(p['source_dual_root']) / 'results/prepare/loss_scaling.json')
    if (not prep['complete'] or not prep['passed'] or prep['heldout_used']
            or not stats['complete'] or stats['heldout_used'] or not scale['complete'] or scale['heldout_used']):
        raise RuntimeError('Frozen teacher-only definitions must be complete')
    if stats['config'] != p['softbin'] or scale['case_ids'] != p['calibration']['case_ids']:
        raise RuntimeError('Frozen soft-bin definition or coefficient cohort differs')
    config, coefficients = prep['relational_config'], prep['coefficients']
    for value in (prep, stats):
        if value['calibration_case_ids'] != p['calibration']['case_ids']:
            raise RuntimeError('Teacher definition cohort differs')
    artifact = Path(os.environ['ARTIFACTS'])
    for name in ('channel_selfcheck', 'softbin_selfcheck'):
        check = read(artifact / (name + '.json'))
        if not check['passed'] or check['pbs_jobid'] != os.environ['PBS_JOBID']:
            raise RuntimeError('Current allocation must pass unchanged quantizer selfchecks')

    plan_path = Path(p['calibration']['plan_file'])
    plan = read(plan_path)
    if (plan['status'] != 'complete' or plan['calibration_cohorts']['c80'] != p['calibration']['case_ids']
            or plan['calibration_cohorts']['c8'] != p['vq_training']['case_ids']):
        raise RuntimeError('Frozen original C8/C80 plan differs')
    rows = {int(row['case_id']): row for row in plan['inputs']}
    teacher_dirs = {'calibration': Path(p['control_root']) / 'results/teacher/calibration',
                    'heldout': Path(p['clipping_root']) / 'results/teacher'}
    for split, directory in teacher_dirs.items():
        receipt = read(directory / 'receipt.json')
        expected = p['calibration' if split == 'calibration' else 'evaluation']['case_ids']
        if (not receipt['complete'] or receipt['queries'] != 80 or receipt['case_ids'] != expected
                or receipt['seed_indices'] != ([0] if split == 'calibration' else [0, 1])):
            raise RuntimeError('Shared teacher coverage differs')
    vq_stage = unit == 'vq_kl'
    order = p['vq_training']['case_ids'] if vq_stage else p['calibration']['update_order']
    total = 8 if vq_stage else 80
    if len(order) != total:
        raise RuntimeError('Exact frozen update count required')
    out = root / 'results' / unit
    out.mkdir(parents=True, exist_ok=True)
    state = {'protocol': p['protocol'], 'unit': unit, 'pbs_jobid': os.environ['PBS_JOBID'],
             'complete': False, 'phase': 'loading', 'updates': 0, 'new_teacher_queries': 0,
             'new_model_queries': 0, 'vq_updates': 8, 'activation_updates': 0 if vq_stage else 80}

    def progress(**values):
        state.update(values)
        ref.save_json(out / 'progress.json', state)
        print('VQ_KL_PROGRESS ' + json.dumps(state), flush=True)

    def teacher(split, case, seed):
        saved = torch.load(teacher_dirs[split] / f'{case}_{seed}.pt', map_location='cpu', weights_only=True)
        if saved['plan_row'] != rows[case] or saved['sampler_seed'] != rows[case]['sampler_seeds'][seed]:
            raise RuntimeError('Shared teacher paired input differs')
        return saved

    def values_for(action, future, target):
        return prior._terms(torch, softbin_losses, relational_losses, action, future, target, stats, p, config)

    deadline, runtime = ref.Deadline(), None
    progress()
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        vq_receipt, vq_deployment = None, None
        if vq_stage:
            banks = TrainableBanks(runtime, p['calibration']['bank_directory'])
            initial_masters = banks.snapshot()
            parameters = banks.parameters()
            groups = [{'params': parameters, 'lr': p['vq_training']['learning_rate']}]
        else:
            vq_receipt = read(root / 'results/vq_kl/training_receipt.json')
            vq_deployment = torch.load(root / 'results/vq_kl/deployment.pt', map_location='cpu', weights_only=True)
            if (not vq_receipt['complete'] or vq_receipt['vq_updates'] != 8
                    or vq_deployment['protocol'] != p or vq_deployment['updates'] != 8
                    or len(vq_deployment['module_masters']) != 614):
                raise RuntimeError('Completed KL-trained VQ final8 is required')
            banks = ChannelBanks(runtime, p['calibration']['bank_directory'],
                                 vq_deployment['module_masters'], p['clipping'], p['scaling'])
            groups = banks.optimizer_groups(p['training'], p['scaling'])
            parameters = [value for group in groups for value in group['params']]
            if (len(banks.ratios) != 614 or sum(v.numel() for v in banks.shadows.values()) != 1_806_863
                    or any(v.requires_grad for v in list(banks.books.values()) + list(banks.logs.values()))):
                raise RuntimeError('Activation trainable parameter coverage differs')
        if any(v.requires_grad for v in runtime.model.parameters()):
            raise RuntimeError('Non-target model parameters must stay frozen')
        identities = fixed_encodings(banks)
        data = {}

        def datum(case):
            if case not in data:
                observation, meta = calibration_source.load_observation(np, plan_path.parent, rows[case])
                data[case] = runtime.datum(observation), meta
            return data[case]

        def query(case, seed):
            deadline.check()
            value, meta = datum(case)
            result = ref.query(runtime, sampler, torch, value, meta['description'], rows[case]['sampler_seeds'][seed], banks)
            state['new_model_queries'] += 1
            return result

        first = p['calibration']['case_ids'][0]
        initial = query(first, 0)
        if vq_stage:
            original = torch.load(Path(p['baseline_root']) / 'results/reference/reference_cache.pt',
                                  map_location='cpu', weights_only=True)
            witness = original['parent'][str(first)]
            initial_parity = {'action': parity(torch, initial[0], witness['action']),
                              'future': parity(torch, initial[1], witness['future'])}
            if any(v['max_abs'] > .002 or v['rmse'] > .001 for v in initial_parity.values()):
                raise RuntimeError('Raw C8 VQ initialization parity failed')
            del original
        else:
            witness = torch.load(root / 'results/vq_kl/reference_tensor.pt', map_location='cpu', weights_only=True)
            initial_parity = {'action': parity(torch, initial[0], witness['action']),
                              'future': parity(torch, initial[1], witness['future'])}
            if any(v['max_abs'] != 0 for v in initial_parity.values()):
                raise RuntimeError('D1/r1 must exactly preserve KL VQ final8')
        del initial, witness
        optimizer = torch.optim.Adam(groups)
        checkpoint = out / 'checkpoint.pt'
        updates, history = 0, []
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['protocol'] != p or saved['unit'] != unit:
                raise RuntimeError('Checkpoint belongs to another frozen protocol')
            updates, history = saved['updates'], saved['history']
            if [h['case_id'] for h in history] != order[:updates] or not 0 <= updates <= total:
                raise RuntimeError('Checkpoint update order differs')
            if vq_stage:
                banks.restore(saved['module_masters'])
            else:
                banks.restore_clips(saved['ratios'])
                banks.restore_scales(saved['log2_scales'])
            optimizer.load_state_dict(saved['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda')
            restore_rng(torch, saved['rng'])
            del saved
        while updates < total:
            deadline.check(max([h['seconds'] for h in history[-3:]], default=600) * 1.2)
            case = order[updates]
            value, meta = datum(case)
            runtime.description = meta['description']
            banks.seen.clear()
            target = teacher('calibration', case, 0)
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            action, counts, future = sampler.differentiable_action(
                runtime, value, rows[case]['sampler_seeds'][0], checkpointed=True, return_future=True)
            state['new_model_queries'] += 1
            values = values_for(action, future, target)
            soft, loss = prior._objectives(values, p, scale, coefficients)
            if (len(banks.seen) != 614 or not future.requires_grad or not loss.requires_grad
                    or not bool(torch.isfinite(loss))):
                raise RuntimeError('KL loss or connected target coverage failed')
            loss.backward()
            coverage = banks.gradient_report() if vq_stage else banks.coverage()
            reports = [coverage] if vq_stage else list(coverage.values())
            if any(r['connected'] != 614 or r['finite'] != 614 or r['nonzero'] < 1 for r in reports):
                raise RuntimeError('VQ/activation gradients are disconnected or nonfinite')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if updates == 0:
                optimizer.state.clear()
            if vq_stage:
                banks.refresh()
            else:
                banks.project()
            torch.cuda.synchronize()
            updates += 1
            history.append({'update': updates, 'case_id': case, 'seed_index': 0,
                            **{k: float(v.detach()) for k, v in values.items()},
                            'loss': float(loss.detach()), 'relational_objective': float(loss.detach()),
                            'softbin_objective': float(soft.detach()), 'gradient_norm': float(norm),
                            'coverage': coverage, 'seconds': time.monotonic() - started, **counts})
            payload = {'protocol': p, 'unit': unit, 'updates': updates, 'history': history,
                       'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch)}
            if vq_stage:
                payload['module_masters'] = banks.snapshot()
            else:
                payload.update(ratios=banks.clip_snapshot(), log2_scales=banks.scale_snapshot())
            atomic_torch_save(torch, checkpoint, payload)
            del action, future, values, loss, soft, target, payload
            progress(phase='training', updates=updates, last_update=history[-1])

        final = query(first, 0)
        frozen = verify_encodings(banks, identities)
        deployment_parity = {}
        if vq_stage:
            masters = banks.snapshot()
            changed = {key: sum(not torch.equal(masters[name][key], initial_masters[name][key]) for name in masters)
                       for key in ('books', 'log_scales')}
            if not all(changed.values()):
                raise RuntimeError('KL VQ books and row gains must actually update')
            frozen.update(mse_aware_masters_applied=False, initialization='original C8 bank', changed_modules=changed)
            atomic_torch_save(torch, out / 'deployment.pt', {'protocol': p, 'unit': unit, 'updates': 8,
                'module_masters': masters, 'loss_scaling': scale, 'relational_config': config, 'coefficients': coefficients})
            atomic_torch_save(torch, out / 'reference_tensor.pt', {'action': final[0], 'future': final[1],
                'case_id': first, 'sampler_seed': rows[first]['sampler_seeds'][0], 'protocol': p})
            del masters, initial_masters
        else:
            rounding = banks.cache_compensated_weights()
            cached = query(first, 0)
            deployment_parity = {'action_max_abs': parity(torch, cached[0], final[0])['max_abs'],
                                 'future_max_abs': parity(torch, cached[1], final[1])['max_abs']}
            if any(v != 0 for v in deployment_parity.values()):
                raise RuntimeError('Uncached/cached activation output parity failed')
            final = cached
            frozen.update(banks.verify_frozen(vq_deployment['module_masters']))
            atomic_torch_save(torch, out / 'deployment.pt', {'protocol': p, 'unit': unit, 'updates': 80,
                'vq_checkpoint': str(root / 'results/vq_kl/deployment.pt'),
                'ratios': banks.clip_snapshot(), 'log2_scales': banks.scale_snapshot()})
            ref.save_json(out / 'weight_rounding.json', rounding)
        runtime.current_arm, runtime.description = unit, datum(first)[1]['description']
        actual, _ = runtime.infer(datum(first)[0], rows[first]['sampler_seeds'][0], measure=False)
        state['new_model_queries'] += 1
        actual_error = parity(torch, actual, final[0])['max_abs']
        if actual_error != 0:
            raise RuntimeError('Actual deployment inference action parity failed')
        deployment_parity.update(actual_infer_max_abs=actual_error)
        training = {'complete': True, 'vq_updates': 8, 'activation_updates': 0 if vq_stage else 80,
            'vq_update_order': order if vq_stage else vq_receipt['vq_update_order'],
            'vq_history': history if vq_stage else vq_receipt['vq_history'],
            'activation_update_order': [] if vq_stage else order, 'activation_history': [] if vq_stage else history,
            'gradient_coverage': {'vq': {'passed': True, 'last': history[-1]['coverage']} if vq_stage else vq_receipt['gradient_coverage']['vq'],
                                  'activation': {'passed': True, 'skipped': True} if vq_stage else {'passed': True, 'last': history[-1]['coverage']}},
            'frozen_checks': {'vq': frozen if vq_stage else vq_receipt['frozen_checks']['vq'],
                              'activation': {'passed': True, 'skipped': True} if vq_stage else frozen},
            'parity': {'passed': True, 'initial': initial_parity, 'deployment': deployment_parity},
            'optimizer_reset_after_update1': True, 'loss': p['objective']['both_training_stages'],
            'loss_scaling': scale, 'relational_config': config, 'coefficients': coefficients,
            'pbs_jobid': os.environ['PBS_JOBID']}
        ref.save_json(out / 'training_receipt.json', training)
        extra = ('action_kl', 'latent_kl', 'action_temporal_kl', 'latent_spatiotemporal_kl',
                 'softbin_objective', 'relational_objective')
        endpoint_count = 0
        for split in ('calibration', 'heldout'):
            records = []
            for case, seed in c80.queries(p, split):
                deadline.check()
                record_path = root / 'results' / split / unit / 'queries' / f'{case}_{seed}.json'
                tensor_path = out / 'tensors' / split / f'{case}_{seed}.pt'
                tensor_path.parent.mkdir(parents=True, exist_ok=True)
                expected = {'protocol': p['protocol'], 'arm': unit, 'split': split, 'updates': total,
                    'vq_updates': 8, 'activation_updates': 0 if vq_stage else 80, 'plan_row': rows[case],
                    'case_id': case, 'suite': rows[case]['suite'], 'seed_index': seed,
                    'sampler_seed': rows[case]['sampler_seeds'][seed]}
                target = teacher(split, case, seed)
                if record_path.is_file() and not tensor_path.is_file():
                    raise RuntimeError('Saved endpoint record lacks full tensor cache')
                if tensor_path.is_file():
                    saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Saved endpoint tensor binding differs')
                    if not torch.equal(saved['teacher_action'], target['action'].float().cpu()) or not torch.equal(saved['teacher_future'], target['future'].float().cpu()):
                        raise RuntimeError('Cached teacher differs')
                    action, future, counts = saved['action'], saved['future'], saved['counts']
                elif split == 'calibration' and case == first and seed == 0:
                    action, future, counts = final
                else:
                    action, future, counts = query(case, seed)
                values = values_for(action, future, target)
                measured = metrics(torch, latent_reference, action, future, target, p['training'])
                objectives = prior._objectives(values, p, scale, coefficients)
                record = {**expected, 'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts, **measured,
                          **{k: float(v) for k, v in values.items()},
                          'softbin_objective': float(objectives[0]), 'relational_objective': float(objectives[1])}
                if record_path.is_file():
                    old_record = read(record_path)
                    if any(old_record.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Saved endpoint record binding differs')
                    prior._record_values(old_record, values, measured, objectives)
                    record = old_record
                else:
                    atomic_torch_save(torch, tensor_path, {**expected, 'counts': counts,
                        'action': action.float().cpu(), 'future': future.float().cpu(),
                        'teacher_action': target['action'].float().cpu(), 'teacher_future': target['future'].float().cpu()})
                    ref.save_json(record_path, record)
                records.append(record)
                endpoint_count += 1
                progress(phase='evaluation', split=split, queries=len(records), last_case=case, last_seed=seed)
                del action, future, values, target
            summary = {'protocol': p['protocol'], 'complete': True, 'arm': unit, 'split': split,
                       'updates': total, 'pbs_jobid': os.environ['PBS_JOBID'], **c80.aggregate_records(records, p, split)}
            prior._extra_aggregate(summary, records, extra)
            ref.save_json(root / 'results' / split / unit / 'summary.json', summary)
        ref.save_json(out / 'summary.json', {**state, 'complete': True, 'phase': 'complete', 'updates': total,
            'evaluation_queries': endpoint_count, 'training_receipt': training, 'new_teacher_queries': 0,
            'loss_scaling': scale, 'relational_config': config, 'coefficients': coefficients})
        progress(phase='complete', complete=True, updates=total)
        print('VQ_ACTIVATION_KL_COMPLETE', flush=True)
        return 0
    except TimeoutError as error:
        progress(phase='checkpointed', reason=str(error))
        return 75
    except Exception as error:
        progress(phase='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--unit', required=True, choices=('vq_kl', 'vq_kl_activation_kl'))
    args = parser.parse_args()
    c80.guard()
    raise SystemExit(run(read(args.protocol), args.unit))
