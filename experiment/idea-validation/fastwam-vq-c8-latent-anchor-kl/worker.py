"""Original C8 eight-update weight training with one added latent anchor KL."""
import argparse
import math
import os
from pathlib import Path
import signal
import time
from anchor_loss import loss_terms, objectives
from anchor_selfcheck import require_allocation
from prepare import read, save, queries, load_reference


class Deadline:
    def __init__(self):
        self.end = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ['FW_SOFT_DEADLINE_SECONDS'])
        self.interrupted = False
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_):
        self.interrupted = True

    def check(self, estimate=120):
        if self.interrupted or time.time() + estimate + 120 >= self.end:
            raise TimeoutError('Soft deadline; committed updates and queries are reusable')


def run(p):
    require_allocation()
    import numpy as np
    import torch
    import sampler
    import runner
    import calibration_source
    import latent_reference
    import train_c8_reference as original
    from train_banks import TrainableBanks
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, optimizer_to_device, to_cpu
    sampler._require_compute_allocation()
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(p['remote_root']); out = root / 'results/anchor_kl'
    out.mkdir(parents=True, exist_ok=True)
    prep = read(root / 'results/prepare/summary.json')
    check = read(Path(os.environ['ARTIFACTS']) / 'anchor_selfcheck.json')
    if (prep['protocol'] != p['protocol'] or not prep['complete'] or not prep['passed']
            or prep['heldout_used'] or prep['calibration_case_ids'] != p['calibration']['case_ids']
            or not prep['coefficient_frozen'] or not check['passed'] or check['pbs_jobid'] != os.environ['PBS_JOBID']
            or read(Path(os.environ['ARTIFACTS']) / 'baseline_protocol.json') != p['baseline_protocol']):
        raise RuntimeError('Frozen C8 preparation/runtime contract differs')
    cache, saved_rows, rows = load_reference(torch, p)
    beta, temperature, stats = prep['beta'], prep['temperature'], prep['stats']
    if any(not math.isfinite(v) or v <= 0 for v in (beta, temperature)):
        raise RuntimeError('Positive frozen C8 coefficient and temperature required')
    ids = p['calibration']['case_ids']
    state = {'protocol': p['protocol'], 'unit': 'anchor_kl', 'pbs_jobid': os.environ['PBS_JOBID'],
             'complete': False, 'phase': 'loading', 'updates': 0, 'new_model_queries': 0,
             'baseline_new_model_queries': 0, 'new_teacher_queries': 0, 'new_closedloop_episodes': 0}

    def progress(**values):
        state.update(values); save(out / 'progress.json', state)
        print('C8_ANCHOR_PROGRESS', state, flush=True)

    runtime = None; deadline = Deadline(); progress()
    try:
        runtime = runner.Runtime(out); runtime.model.eval(); runtime.arm('bf16')
        banks = TrainableBanks(runtime, p['calibration']['bank_directory'])
        parameters = banks.parameters()
        if len(parameters) != 1228 or any(v.requires_grad for v in runtime.model.parameters()):
            raise RuntimeError('Only the original 614 codebooks and row gains may be trained')
        optimizer = torch.optim.Adam(parameters, lr=p['training']['learning_rate'])
        checkpoint = out / 'checkpoint.pt'; updates, history, initial_parity = 0, [], None
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['protocol'] != p or saved['preparation'] != prep:
                raise RuntimeError('Resume contract differs')
            updates, history, initial_parity = saved['updates'], saved['history'], saved['initial_parity']
            if not 0 <= updates <= 8 or [r['case_id'] for r in history] != ids[:updates]:
                raise RuntimeError('Committed C8 update cursor differs')
            banks.restore(saved['module_masters']); optimizer.load_state_dict(saved['optimizer'])
            optimizer_to_device(torch, optimizer, 'cuda'); restore_rng(torch, saved['rng'])
        data = {}

        def datum(case):
            if case not in data:
                obs, meta = calibration_source.load_observation(np, Path(p['calibration']['plan_file']).parent, rows[case])
                if case in saved_rows and meta != saved_rows[case]['metadata']:
                    raise RuntimeError('C8 observation metadata differs from the original reference')
                data[case] = runtime.datum(obs), meta
            return data[case]

        def teacher(split, case, seed):
            if split == 'calibration':
                return cache['teacher'][str(case)]
            target = torch.load(Path(p['teacher_heldout_directory']) / f'{case}_{seed}.pt', map_location='cpu', weights_only=True)
            if target['plan_row'] != rows[case] or target['sampler_seed'] != rows[case]['sampler_seeds'][seed]:
                raise RuntimeError('Heldout teacher binding differs')
            return target

        while updates < 8:
            deadline.check(max([h['seconds'] for h in history[-3:]], default=1800) * 1.2)
            case = ids[updates]; value, meta = datum(case)
            runtime.description, runtime.current_arm = meta['description'], 'anchor_kl'
            banks.seen.clear(); optimizer.zero_grad(set_to_none=True); started = time.monotonic()
            action, counts, future = sampler.differentiable_action(
                runtime, value, rows[case]['sampler_seeds'][0], checkpointed=True, return_future=True)
            state['new_model_queries'] += 1
            values = loss_terms(torch, action, future, teacher('calibration', case, 0), stats, p['anchor'], temperature)
            base, total = objectives(values, p, beta)
            if len(banks.seen) != 614 or not total.requires_grad or not bool(torch.isfinite(total)):
                raise RuntimeError('Original full-model gradient path or finite loss failed')
            extra = {}
            if updates == 0:
                parent = cache['parent'][str(case)]
                initial_parity = {name: original._parity(torch, actual, parent[name])
                                  for name, actual in [('action', action), ('future', future)]}
                if any(v['max_abs'] > .002 or v['rmse'] > .001 for v in initial_parity.values()):
                    raise RuntimeError('Original untrained C8 initialization parity failed')
                gradient = torch.autograd.grad(values['latent_anchor_kl'], future, retain_graph=True)[0]
                if (not bool(torch.isfinite(gradient).all()) or not bool((gradient[:, :, 1:] != 0).any())
                        or bool((gradient[:, :, :1] != 0).any())):
                    raise RuntimeError('Added KL must connect only to generated future frames')
                extra['kl_future_gradient_norm'] = float(gradient.float().norm()); del gradient
            total.backward(); coverage = banks.gradient_report()
            if coverage['connected'] != 614 or coverage['finite'] != 614 or coverage['nonzero'] < 1:
                raise RuntimeError('Original codebook/row gain gradient coverage failed')
            norm = torch.nn.utils.clip_grad_norm_(parameters, p['training']['gradient_norm_clip'], error_if_nonfinite=True)
            optimizer.step()
            if updates == 0:
                optimizer.state.clear()
            banks.refresh(); torch.cuda.synchronize(); updates += 1
            history.append({'update': updates, 'case_id': case, 'seed_index': 0,
                            **{k: float(v.detach()) for k, v in values.items()},
                            'baseline_objective': float(base.detach()), 'loss': float(total.detach()),
                            'gradient_norm': float(norm), 'coverage': coverage, 'beta': beta,
                            'seconds': time.monotonic() - started, 'pbs_jobid': os.environ['PBS_JOBID'], **extra, **counts})
            del action, future, values, base, total
            atomic_torch_save(torch, checkpoint, {'protocol': p, 'preparation': prep, 'updates': updates,
                              'history': history, 'initial_parity': initial_parity, 'module_masters': banks.snapshot(),
                              'optimizer': to_cpu(optimizer.state_dict()), 'rng': capture_rng(torch)})
            progress(phase='training', updates=updates, last_update=history[-1])

        for parameter in parameters:
            parameter.requires_grad_(False); parameter.grad = None
        optimizer.state.clear()
        atomic_torch_save(torch, out / 'deployment.pt', {'protocol': p, 'arm': 'anchor_kl', 'updates': 8,
                          'target_linears': 614, 'module_masters': banks.snapshot(), 'preparation': prep,
                          'reference_binding': cache['binding'], 'initial_parity': initial_parity})
        endpoint_count, actual_parity = 0, None
        for split in ('calibration', 'heldout'):
            for case, seed in queries(p, split):
                deadline.check(); tensor_path = out / 'tensors' / split / f'{case}_{seed}.pt'
                expected = {'protocol': p['protocol'], 'arm': 'anchor_kl', 'split': split, 'updates': 8,
                            'case_id': case, 'suite': rows[case]['suite'], 'plan_row': rows[case],
                            'seed_index': seed, 'sampler_seed': rows[case]['sampler_seeds'][seed]}
                if tensor_path.is_file():
                    saved = torch.load(tensor_path, map_location='cpu', weights_only=True)
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Reusable endpoint binding differs')
                    action, future, counts = saved['action'], saved['future'], saved['counts']
                else:
                    value, meta = datum(case); runtime.description = meta['description']; banks.seen.clear()
                    with torch.no_grad():
                        action, counts, future = sampler.differentiable_action(
                            runtime, value, rows[case]['sampler_seeds'][seed], checkpointed=False, return_future=True)
                    state['new_model_queries'] += 1
                    if len(banks.seen) != 614 or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all()):
                        raise RuntimeError('Endpoint output/Linear coverage failed')
                    action, future = action.detach().cpu(), future.detach().cpu()
                    tensor_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_torch_save(torch, tensor_path, {**expected, 'action': action, 'future': future, 'counts': counts})
                if split == 'calibration' and case == ids[0]:
                    value, meta = datum(case); runtime.description = meta['description']
                    actual, _ = runtime.infer(value, rows[case]['sampler_seeds'][seed], measure=False)
                    state['new_model_queries'] += 1
                    actual_parity = original._parity(torch, actual, action)
                    if actual_parity['max_abs'] > 1e-6:
                        raise RuntimeError('Native inference versus sampler parity failed')
                    del actual
                target = teacher(split, case, seed)
                values = loss_terms(torch, action, future, target, stats, p['anchor'], temperature)
                delta = action.float() - target['action'].float()
                rmse = lambda x: float(x.square().mean().sqrt())
                latent = float(latent_reference.future_nmse_loss(torch, future, target['future']))
                base, total = objectives(values, p, beta)
                record = {**expected, 'pbs_jobid': os.environ['PBS_JOBID'], 'counts': counts,
                          'first10motor_rmse': rmse(delta[:10, :6]), 'full32motor_rmse': rmse(delta[:, :6]),
                          'late22motor_rmse': rmse(delta[10:, :6]), 'first10gripper_rmse': rmse(delta[:10, 6]),
                          'future_latent_nmse': latent, 'first10motor_mse': float(values['action_mse']),
                          **{k: float(v) for k, v in values.items()}, 'baseline_objective': float(base),
                          'mixed_loss': float(base), 'total_loss': float(total), 'composite_objective': float(total)}
                save(out / split / 'queries' / f'{case}_{seed}.json', record)
                endpoint_count += 1
                progress(phase='evaluation', updates=8, split=split, queries=endpoint_count, last_case=case, last_seed=seed)
                del action, future, target, values, base, total
        training = {'protocol': p['protocol'], 'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
                    'updates': 8, 'update_order': ids, 'history': history, 'initial_parity': initial_parity,
                    'actual_infer_parity': actual_parity, 'optimizer_reset_after_update1': True,
                    'activation_clipping': False, 'per_channel_scaling': False, 'beta': beta, 'temperature': temperature}
        save(out / 'training_receipt.json', training)
        summary = {**state, 'complete': True, 'phase': 'complete', 'updates': 8,
                   'evaluation_queries': endpoint_count, 'training_receipt': training}
        save(out / 'summary.json', summary); progress(phase='complete', updates=8, complete=True)
        print('C8_ANCHOR_WORKER_COMPLETE', flush=True)
        return 0
    except TimeoutError as error:
        progress(phase='checkpointed', reason=str(error)); return 75
    except Exception as error:
        progress(phase='failed', error=type(error).__name__ + ': ' + str(error)); raise
    finally:
        if runtime is not None and runtime.env is not None:
            runtime.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True)
    parser.add_argument('--unit', choices=('anchor_kl',), default='anchor_kl')
    args = parser.parse_args(); raise SystemExit(run(read(args.protocol)))
