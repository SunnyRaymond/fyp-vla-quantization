"""Frozen soft-bin control replay and relational-KL C80 training/evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import clipping_reference as ref
import c80_reference as c80


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _positive(mapping, key):
    value = float(mapping[key])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f'{key} must be finite and positive')
    return value


def _frozen_inputs(p, source_dual, prep, stats, scale, artifacts):
    if p.get('protocol') != 'fastwam-vq-relational-kl-v1':
        raise RuntimeError('Unexpected relational protocol')
    for key in ('calibration', 'clipping', 'scaling', 'softbin', 'source_clipping_protocol'):
        if p[key] != source_dual[key]:
            raise RuntimeError(f'Frozen dual-KL input changed: {key}')
    for key in ('case_ids', 'seed_indices', 'aggregation'):
        if p['evaluation'][key] != source_dual['evaluation'][key]:
            raise RuntimeError(f'Frozen evaluation cohort changed: {key}')
    for key in ('additional_updates', 'epochs', 'weight_learning_rate', 'clip_learning_rate',
                'optimizer', 'fresh_optimizer_each_arm', 'optimizer_reset_after_update1',
                'gradient_norm_clip', 'rho', 'normalization_c'):
        if p['training'][key] != source_dual['training'][key]:
            raise RuntimeError(f'Frozen training setting changed: {key}')
    if p['source_per_channel_root'] != source_dual['source_per_channel_root']:
        raise RuntimeError('Frozen per-channel source root differs')
    artifact_protocol = _json(Path(artifacts) / 'source_protocol.json')
    per_channel_protocol = _json(Path(p['source_per_channel_root']) / 'protocol.json')
    dual_source_protocol = _json(Path(source_dual['source_per_channel_root']) / 'protocol.json')
    if artifact_protocol != per_channel_protocol or artifact_protocol != dual_source_protocol:
        raise RuntimeError('Runtime per-channel source protocol differs')
    if prep.get('passed') is not True or prep.get('complete') is not True or prep.get('heldout_used') is not False:
        raise RuntimeError('Completed calibration-only relational preparation is required')
    if prep.get('calibration_case_ids') != p['calibration']['case_ids']:
        raise RuntimeError('Relational preparation cohort differs')
    if not stats.get('complete') or stats.get('heldout_used') is not False:
        raise RuntimeError('Frozen calibration-only soft-bin statistics are required')
    if stats.get('calibration_case_ids') != p['calibration']['case_ids'] or stats.get('config') != p['softbin']:
        raise RuntimeError('Frozen soft-bin statistics/configuration differ')
    if not scale.get('complete') or scale.get('heldout_used') is not False:
        raise RuntimeError('Frozen calibration-only soft-bin scaling is required')
    if scale.get('case_ids') != p['calibration']['case_ids']:
        raise RuntimeError('Frozen soft-bin scaling cohort differs')
    config, coefficients = prep.get('relational_config'), prep.get('coefficients')
    if not isinstance(config, dict) or not isinstance(coefficients, dict):
        raise RuntimeError('Relational temperatures and coefficients are required')
    for key in ('action_temperature', 'latent_temperature'):
        _positive(config, key)
    for key in ('action_beta', 'latent_beta'):
        _positive(coefficients, key)
    for key in ('action_kl_coefficient', 'latent_kl_coefficient'):
        _positive(scale, key)
    return config, coefficients


def _objectives(values, p, scale, coefficients):
    alpha_a = float(scale['action_kl_coefficient'])
    alpha_z = float(scale['latent_kl_coefficient'])
    latent_weight = p['training']['rho'] * p['training']['normalization_c']
    softbin = alpha_a * values['action_kl'] + latent_weight * alpha_z * values['latent_kl']
    relational = alpha_a * (values['action_kl'] + coefficients['action_beta'] * values['action_temporal_kl'])
    relational += latent_weight * alpha_z * (values['latent_kl'] + coefficients['latent_beta'] * values['latent_spatiotemporal_kl'])
    return softbin, relational


def _terms(torch, softbin_losses, relational_losses, action, future, teacher, stats, p, config):
    target_action = teacher['action'].to(action.device)
    target_future = teacher['future'].to(future.device)
    values = softbin_losses(torch, action, future, target_action, target_future, stats, p['softbin'])
    values.update(relational_losses(torch, action, future, target_action, target_future, stats, config))
    return values


def _record_values(record, values, measured, objectives):
    for key, value in measured.items():
        if record.get(key) != value:
            raise RuntimeError(f'Saved endpoint metric differs: {key}')
    for key, value in values.items():
        if record.get(key) != float(value):
            raise RuntimeError(f'Saved endpoint loss differs: {key}')
    for key, value in zip(('softbin_objective', 'relational_objective'), objectives):
        if record.get(key) != float(value):
            raise RuntimeError(f'Saved endpoint objective differs: {key}')


def _extra_aggregate(summary, records, keys):
    for key in keys:
        for suite in ref.SUITES:
            suite_records = [r for r in records if r['suite'] == suite]
            cases = sorted({r['case_id'] for r in suite_records})
            summary['by_suite'][suite][key] = sum(
                sum(r[key] for r in suite_records if r['case_id'] == case)
                / sum(r['case_id'] == case for r in suite_records)
                for case in cases
            ) / len(cases)
        summary['overall'][key] = sum(summary['by_suite'][suite][key] for suite in ref.SUITES) / 4
        if not math.isfinite(summary['overall'][key]) or summary['overall'][key] < 0:
            raise RuntimeError(f'Invalid aggregated metric: {key}')


def run(p, unit):
    import numpy as np
    import torch
    import sampler
    import runner
    import calibration_source
    import latent_reference
    from channel_banks import ChannelBanks
    from softbin_loss import losses as softbin_losses
    from relational_loss import relational_losses
    from metrics_reference import metrics
    from checkpointing import (atomic_torch_save, capture_rng, restore_rng,
                               optimizer_to_device, to_cpu)

    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    root = Path(p['remote_root'])
    source_root = Path(p['source_dual_root'])
    artifact = Path(os.environ['ARTIFACTS'])
    source_dual = _json(source_root / 'protocol.json')
    prep = _json(root / 'results/prepare/summary.json')
    stats = _json(source_root / 'results/fit_stats/summary.json')
    scale = _json(source_root / 'results/prepare/loss_scaling.json')
    relational_config, coefficients = _frozen_inputs(p, source_dual, prep, stats, scale, artifact)

    channel_check = _json(artifact / 'channel_selfcheck.json')
    softbin_check = _json(artifact / 'softbin_selfcheck.json')
    if not all(channel_check.get(k) is True for k in
               ('passed', 'independent_reference_passed', 'old_surrogate_rejected')) or not softbin_check.get('passed'):
        raise RuntimeError('GPU channel and soft-bin selfchecks must pass')
    if any(v.get('pbs_jobid') != os.environ['PBS_JOBID'] for v in (channel_check, softbin_check)):
        raise RuntimeError('Selfchecks must belong to this PBS allocation')

    source_receipt_path = source_root / 'results/dual_kl/training_receipt.json'
    source_deployment_path = source_root / 'results/dual_kl/deployment.pt'
    source_receipt = _json(source_receipt_path)
    source_deployment = torch.load(source_deployment_path, map_location='cpu', weights_only=True)
    update_order = p['calibration']['update_order']
    source_history = source_receipt.get('history', [])
    source_folded = source_receipt.get('cached_query_parity', {})
    if (source_receipt.get('complete') is not True or source_receipt.get('unit') != 'dual_kl'
            or source_receipt.get('updates') != 80 or source_receipt.get('pbs_jobid') != p['source_dual_job']
            or source_receipt.get('update_order') != update_order or len(source_history) != 80
            or [h.get('update') for h in source_history] != list(range(1, 81))
            or [h.get('case_id') for h in source_history] != update_order
            or source_receipt.get('initial_parity_max_abs') != 0
            or source_folded.get('action_max_abs') != 0 or source_folded.get('future_max_abs') != 0
            or source_receipt.get('actual_infer_parity_max_abs') != 0
            or source_deployment.get('protocol') != source_dual
            or source_deployment.get('unit') != 'dual_kl' or source_deployment.get('updates') != 80):
        raise RuntimeError('Completed exact80 dual-KL deployment is required')

    data_root, rows, _, base_deployment, reference_action = ref.load_inputs(p['source_clipping_protocol'], torch)
    teacher_dirs = {
        'calibration': Path(p['control_root']) / 'results/teacher/calibration',
        'heldout': Path(p['clipping_root']) / 'results/teacher',
    }
    for split, directory in teacher_dirs.items():
        receipt = _json(directory / 'receipt.json')
        expected = p['calibration' if split == 'calibration' else 'evaluation']['case_ids']
        seeds = [0] if split == 'calibration' else [0, 1]
        if (receipt.get('complete') is not True or receipt.get('queries') != 80
                or receipt.get('case_ids') != expected or receipt.get('seed_indices') != seeds):
            raise RuntimeError(f'Frozen shared teacher coverage differs: {split}')

    out = root / 'results' / unit
    out.mkdir(parents=True, exist_ok=True)
    state = {'protocol': p['protocol'], 'unit': unit, 'pbs_jobid': os.environ['PBS_JOBID'],
             'complete': False, 'phase': 'loading', 'updates': 0,
             'new_teacher_queries': 0, 'new_model_queries': 0}

    def progress(**values):
        state.update(values)
        ref.save_json(out / 'progress.json', state)
        print('RELATIONAL_KL_PROGRESS ' + json.dumps(state), flush=True)

    runtime = None
    deadline = ref.Deadline()
    progress()
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        data = {}
        banks = ChannelBanks(runtime, p['calibration']['bank_directory'],
                             base_deployment['module_masters'], p['clipping'], p['scaling'])
        if any(bool((v != 0).any()) for v in banks.shadows.values()) or any(float(v) != 1 for v in banks.ratios.values()):
            raise RuntimeError('Relational training must start at D=1/r=1')
        groups = banks.optimizer_groups(p['training'], p['scaling'])
        parameters = [v for group in groups for v in group['params']]
        if {id(v) for v in parameters} != {id(v) for v in list(banks.ratios.values()) + list(banks.shadows.values())}:
            raise RuntimeError('Only the 614 clipping ratios and 1,806,863 log2D values may train')
        if (len(banks.ratios) != 614 or sum(v.numel() for v in banks.shadows.values()) != 1_806_863
                or any(v.requires_grad for v in runtime.model.parameters())
                or any(v.requires_grad for v in list(banks.books.values()) + list(banks.logs.values()))):
            raise RuntimeError('Trainable/frozen parameter coverage differs')

        def datum(case):
            if case not in data:
                observation, meta = calibration_source.load_observation(np, data_root, rows[case])
                data[case] = runtime.datum(observation), meta
            return data[case]

        def query(case, seed):
            deadline.check()
            value, meta = datum(case)
            result = ref.query(runtime, sampler, torch, value, meta['description'],
                               rows[case]['sampler_seeds'][seed], banks)
            state['new_model_queries'] += 1
            return result

        def teacher(split, case, seed):
            saved = torch.load(teacher_dirs[split] / f'{case}_{seed}.pt',
                               map_location='cpu', weights_only=True)
            if saved.get('plan_row') != rows[case] or saved.get('sampler_seed') != rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Shared teacher binding differs')
            return saved

        def values_for(action, future, target):
            return _terms(torch, softbin_losses, relational_losses, action, future,
                          target, stats, p, relational_config)

        first = p['calibration']['case_ids'][0]
        initial_parity = None
        if unit == 'relational':
            initial = query(first, 0)
            initial_parity = float((initial[0].float() - torch.tensor(reference_action['action'], dtype=torch.float32)).abs().max())
            if initial_parity != 0:
                raise RuntimeError('Original C8 rho1.5 baseline forward parity failed')
            del initial
            progress(initial_parity_max_abs=initial_parity)

        training = None
        checkpoint_path = out / 'checkpoint.pt'
        if unit == 'relational':
            optimizer = torch.optim.Adam(groups)
            updates, history = 0, []
            if checkpoint_path.is_file():
                saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if (saved.get('protocol') != p or saved.get('unit') != unit
                        or saved.get('loss_scaling') != scale or saved.get('relational_config') != relational_config
                        or saved.get('coefficients') != coefficients):
                    raise RuntimeError('Training checkpoint contract differs')
                updates, history = saved['updates'], saved['history']
                if [h['case_id'] for h in history] != update_order[:updates]:
                    raise RuntimeError('Training checkpoint update order differs')
                banks.restore_clips(saved['ratios'])
                banks.restore_scales(saved['shadow_log2_scales'])
                optimizer.load_state_dict(saved['optimizer'])
                optimizer_to_device(torch, optimizer, 'cuda')
                restore_rng(torch, saved['rng'])
            while updates < 80:
                deadline.check(max([h['seconds'] for h in history[-3:]], default=300) * 1.15)
                case = update_order[updates]
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
                softbin_objective, relational_objective = _objectives(values, p, scale, coefficients)
                loss = relational_objective
                if len(banks.seen) != 614 or not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError('Relational training loss or Linear coverage failed')
                loss.backward()
                coverage = banks.coverage()
                if any(v['connected'] != 614 or v['finite'] != 614 or v['nonzero'] < 1 for v in coverage.values()):
                    raise RuntimeError('Ratio/channel gradients are disconnected or nonfinite')
                norm = torch.nn.utils.clip_grad_norm_(parameters, p['training']['gradient_norm_clip'], error_if_nonfinite=True)
                optimizer.step()
                if updates == 0 and p['training']['optimizer_reset_after_update1']:
                    optimizer.state.clear()
                banks.project()
                torch.cuda.synchronize()
                updates += 1
                history.append({'update': updates, 'case_id': case, 'seed_index': 0,
                                **{k: float(v.detach()) for k, v in values.items()},
                                'softbin_objective': float(softbin_objective.detach()),
                                'relational_objective': float(relational_objective.detach()),
                                'loss': float(loss.detach()), 'gradient_norm': float(norm),
                                'coverage': coverage, 'seconds': time.monotonic() - started,
                                'pbs_jobid': os.environ['PBS_JOBID'], **counts})
                del action, future, values, softbin_objective, relational_objective, loss
                atomic_torch_save(torch, checkpoint_path, {
                    'protocol': p, 'unit': unit, 'loss_scaling': scale,
                    'relational_config': relational_config, 'coefficients': coefficients,
                    'updates': updates, 'history': history,
                    'ratios': banks.clip_snapshot(), 'shadow_log2_scales': banks.scale_snapshot(),
                    'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch),
                })
                progress(phase='training', updates=updates, last_update=history[-1])

        if unit == 'softbin':
            banks.restore_clips(source_deployment['ratios'])
            banks.restore_scales(source_deployment['log2_scales'])
        deadline.check(120)
        uncached = query(first, 0)
        rounding = banks.cache_compensated_weights()
        cached = query(first, 0)
        folded = {
            'action_max_abs': float((uncached[0].float() - cached[0].float()).abs().max()),
            'future_max_abs': float((uncached[1].float() - cached[1].float()).abs().max()),
        }
        if any(value != 0 for value in folded.values()):
            raise RuntimeError('Uncached and cached deployment action/future differ')
        runtime.current_arm = unit
        runtime.description = datum(first)[1]['description']
        actual, _ = runtime.infer(datum(first)[0], rows[first]['sampler_seeds'][0], measure=False)
        state['new_model_queries'] += 1
        actual_parity = float((actual.float().cpu() - cached[0].float()).abs().max())
        if actual_parity != 0:
            raise RuntimeError('Actual inference action differs from cached action')
        del actual, uncached

        if unit == 'relational':
            frozen = banks.verify_frozen(base_deployment['module_masters'])
            atomic_torch_save(torch, out / 'deployment.pt', {
                'protocol': p, 'unit': unit, 'updates': 80,
                'baseline_checkpoint': str(Path(p['baseline_root']) / 'results/training/rho1.5/deployment.pt'),
                'ratios': banks.clip_snapshot(), 'log2_scales': banks.scale_snapshot(),
                'scale_metadata_dtype': 'float32_log2',
            })
            training = {
                'protocol': p['protocol'], 'complete': True, 'unit': unit, 'updates': 80,
                'history': history, 'update_order': update_order, 'loss_scaling': scale,
                'relational_config': relational_config, 'coefficients': coefficients,
                'initial_parity_max_abs': initial_parity, 'cached_query_parity': folded,
                'actual_infer_parity_max_abs': actual_parity,
                'clipping': banks.clip_summary(), 'scaling': banks.scaling_summary(),
                'gradient_coupling_complete': True, 'input_absmax_chain': True,
                'softbin_selfcheck': softbin_check, 'channel_selfcheck': channel_check,
                'pbs_jobid': os.environ['PBS_JOBID'], **frozen,
            }
            ref.save_json(out / 'training_receipt.json', training)
            ref.save_json(out / 'weight_rounding.json', rounding)
            state['updates'] = 80
        else:
            banks.verify_frozen(base_deployment['module_masters'])
            training = {
                'protocol': p['protocol'], 'complete': True, 'unit': unit, 'updates': 80,
                'reused_training': True, 'source_unit': 'dual_kl',
                'source_job_id': source_receipt['pbs_jobid'],
                'source_receipt': str(source_receipt_path), 'source_receipt_complete': True,
                'source_receipt_updates': source_receipt['updates'],
                'source_deployment': str(source_deployment_path),
                'source_deployment_unit': source_deployment['unit'],
                'source_deployment_updates': source_deployment['updates'],
                'cached_query_parity': folded, 'actual_infer_parity_max_abs': actual_parity,
            }
            ref.save_json(out / 'training_receipt.json', training)
            state['updates'] = 80

        cache_first = cached
        tensor_root = out / 'tensors'
        tensor_root.mkdir(parents=True, exist_ok=True)
        extra_keys = ('action_kl', 'latent_kl', 'action_temporal_kl',
                      'latent_spatiotemporal_kl', 'softbin_objective', 'relational_objective')
        endpoint_count = 0
        for split in ('calibration', 'heldout'):
            records = []
            for case, seed in c80.queries(p, split):
                deadline.check()
                record_path = root / 'results' / split / unit / 'queries' / f'{case}_{seed}.json'
                tensor_path = tensor_root / split / f'{case}_{seed}.pt'
                tensor_path.parent.mkdir(parents=True, exist_ok=True)
                expected = {
                    'protocol': p['protocol'], 'arm': unit, 'split': split, 'updates': 80,
                    'plan_row': rows[case], 'case_id': case, 'suite': rows[case]['suite'],
                    'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed],
                }
                record_exists, tensor_exists = record_path.is_file(), tensor_path.is_file()
                if record_exists and not tensor_exists:
                    raise RuntimeError('Endpoint record has no full action/future tensor; refusing stale cache')
                target = teacher(split, case, seed)
                if tensor_exists:
                    saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Saved tensor endpoint binding differs')
                    action, future, counts = saved['action'], saved['future'], saved['counts']
                    if (tuple(action.shape) != (32, 7) or tuple(future.shape) != (1, 48, 3, 14, 28)
                            or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())
                            or not torch.equal(saved['teacher_action'].float(), target['action'].float())
                            or not torch.equal(saved['teacher_future'].float(), target['future'].float())):
                        raise RuntimeError('Saved endpoint tensor or teacher binding is invalid')
                elif case == first and seed == 0 and split == 'calibration':
                    action, future, counts = cache_first
                    cache_first = None
                else:
                    action, future, counts = query(case, seed)
                values = values_for(action, future, target)
                measured = metrics(torch, latent_reference, action, future, target, p['training'])
                objectives = _objectives(values, p, scale, coefficients)
                old = None
                if unit == 'softbin':
                    old_path = source_root / 'results' / split / 'dual_kl' / 'queries' / f'{case}_{seed}.json'
                    old = _json(old_path)
                    old_expected = {
                        'protocol': source_dual['protocol'], 'arm': 'dual_kl', 'split': split,
                        'updates': 80, 'plan_row': rows[case], 'case_id': case,
                        'suite': rows[case]['suite'], 'seed_index': seed,
                        'sampler_seed': rows[case]['sampler_seeds'][seed],
                    }
                    if any(old.get(k) != v for k, v in old_expected.items()):
                        raise RuntimeError('Source dual-KL endpoint binding differs')
                    if any(measured[k] != old[k] for k in c80.METRICS):
                        raise RuntimeError('Softbin RMSE/NMSE replay is not exact')
                    if float(values['action_kl']) != old['action_kl'] or float(values['latent_kl']) != old['latent_kl']:
                        raise RuntimeError('Softbin action/latent KL replay is not exact')
                    if 'action' not in old or not torch.equal(
                            action.float().cpu(), torch.tensor(old['action'], dtype=torch.float32)):
                        raise RuntimeError('Softbin deployment action replay is not exact')
                if record_exists:
                    record = _json(record_path)
                    if any(record.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Saved endpoint record binding differs')
                    _record_values(record, values, measured, objectives)
                else:
                    record = {
                        **expected, 'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts,
                        **measured, **{k: float(v) for k, v in values.items()},
                        'softbin_objective': float(objectives[0]),
                        'relational_objective': float(objectives[1]),
                    }
                    if not tensor_exists:
                        atomic_torch_save(torch, tensor_path, {
                            **expected, 'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts,
                            'action': action.float().cpu(), 'future': future.float().cpu(),
                            'teacher_action': target['action'].float().cpu(),
                            'teacher_future': target['future'].float().cpu(),
                        })
                    ref.save_json(record_path, record)
                records.append(record)
                endpoint_count += 1
                progress(phase='evaluation', split=split, queries=len(records),
                         last_case=case, last_seed=seed)
                del action, future, values, target
            summary = {
                'protocol': p['protocol'], 'complete': True, 'arm': unit,
                'split': split, 'updates': 80, 'pbs_jobid': os.environ['PBS_JOBID'],
                **c80.aggregate_records(records, p, split),
            }
            _extra_aggregate(summary, records, extra_keys)
            ref.save_json(root / 'results' / split / unit / 'summary.json', summary)

        summary = {
            **state, 'complete': True, 'phase': 'complete', 'updates': 80,
            'evaluation_queries': endpoint_count, 'training_receipt': training,
            'new_teacher_queries': 0, 'source_dual_job': p['source_dual_job'],
            'source_stats': str(source_root / 'results/fit_stats/summary.json'),
            'source_loss_scaling': str(source_root / 'results/prepare/loss_scaling.json'),
            'relational_config': relational_config, 'coefficients': coefficients,
        }
        ref.save_json(out / 'summary.json', summary)
        progress(phase='complete', complete=True, updates=80)
        print('RELATIONAL_KL_WORKER_COMPLETE', flush=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=('softbin', 'relational'), required=True)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    c80.guard()
    return run(_json(args.protocol), args.unit)


if __name__ == '__main__':
    raise SystemExit(main())
