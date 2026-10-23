"""Sum one frozen 40-pair shard of an isolated control objective; no updates."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path

from control_common import Deadline, guard, read, save
from control_losses import action_response_kl, latent_control_kl
from parameter_gate import (dynamic_student, frozen_snapshot, paired_vjp,
                            resident_weights, parked_student_weights)


def tensor_save(torch, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def add_gradients(total, incoming):
    if len(total) != len(incoming):
        raise ValueError('Gradient parameter count differs')
    return [old if new is None else new.detach().clone() if old is None
            else old + new.detach() for old, new in zip(total, incoming)]


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True,
                        choices=tuple(f'c80_grad_{kind}_{shard}'
                                      for kind in ('response', 'latent') for shard in (0, 1)))
    args = parser.parse_args()
    import numpy as np
    import torch
    from aggregate import completed
    from calibration_worker import _load_case_inputs, _pair_binding
    from checkpointing import capture_rng, restore_rng
    from diagnostic_runtime import SwitchingRuntime, validate_binding
    from frozen_reader import BF16Reader

    p = read(Path(__file__).with_name('protocol.json'))
    config = read(Path(__file__).with_name('c80_gate_protocol.json'))
    if (config['parent_protocol'] != p['protocol']
            or config['case_ids'] != list(range(2000, 2080))
            or config['fractions'] != [1.0, 0.1]):
        raise ValueError('Frozen full-C80 protocol differs')
    root, unit = Path(p['remote_root']), args.unit
    kind, shard = unit.split('_')[2], int(unit[-1])
    objective, arm = config['objective_keys'][kind], config['checkpoint_arm']
    gate = completed(root, 'parameter_gate')
    if not (gate.get('original_parameters_restored')
            and gate.get('frozen_reader_and_model_unchanged')):
        raise ValueError('Real single-pair parameter implementation gate required')
    completed(root, 'aggregate_calibration')
    completed(root, f'calibration_{shard}')
    pair_summary = completed(root, 'calibration_pairs')
    catalog = pair_summary['records']
    if (sorted(case['case_id'] for case in catalog) != config['case_ids']
            or any(case['training_use'] is not False for case in catalog)):
        raise ValueError('Original diagnostic-only all80 pair catalog required')
    cases = sorted((case for case in catalog if case['shard'] == shard),
                   key=lambda case: case['case_id'])
    if len(cases) != 40:
        raise ValueError('Exactly40 cases per original shard required')
    out = root / 'results' / unit
    binding = {'protocol': config['protocol'], 'parent_protocol': p['protocol'],
               'unit': unit, 'objective': objective, 'shard': shard,
               'case_ids': [case['case_id'] for case in cases], 'config': config}
    state = {'protocol': p['protocol'], 'gate_protocol': config['protocol'],
             'unit': unit, 'pbs_jobid': os.environ['PBS_JOBID'], 'objective': objective,
             'shard': shard, 'expected_cases': 40, 'completed_cases': 0,
             'complete': False, 'training_updates': 0, 'heldout_inputs_used': False,
             'sensor_evaluation_inputs_used': False,
             'new_student_full_queries_attempted': 0,
             'new_student_full_queries_completed': 0, 'new_bf16_reader_passes': 0,
             'new_teacher_full_queries': 0, 'native_parity_all_zero': False,
             'reader_native_parity': [], 'reader_gpu_loads': 0}
    save(out / 'summary.json', state)
    switching, parameters, originals, total, keys = None, [], [], [], []
    done, loss_sum = [], 0.0
    deadline = Deadline()
    frozen_before = None

    def checkpoint(complete=False):
        payload = {**binding, 'pbs_jobid': os.environ['PBS_JOBID'],
                   'complete': complete, 'parameter_keys': keys,
                   'originals': [value.detach().cpu().clone() for value in originals],
                   'gradient_sum': [None if value is None else value.detach().cpu()
                                    for value in total],
                   'loss_sum': loss_sum, 'case_metrics': done}
        tensor_save(torch, out / 'gradient_state.pt', payload)
        if complete:
            tensor_save(torch, out / 'gradient.pt', payload)

    try:
        switching = SwitchingRuntime(p, out)
        runtime, banks = switching.runtime, switching.banks
        dynamic_student(switching, arm)
        switching.original_weights = {
            name: value.detach().cpu() for name, value in switching.original_weights.items()}
        torch.cuda.empty_cache()
        groups = banks.optimizer_groups(p['source_closed_protocol']['training'],
                                        p['source_closed_protocol']['scaling'])
        if [group['lr'] for group in groups] != [config['ratio_learning_rate'],
                                                 config['channel_learning_rate']]:
            raise ValueError('Original learning rates differ')
        keys = ([f'ratio::{name}' for name in banks.ratios]
                + [f'channel::{name}' for name in banks.shadows])
        if len(keys) != 1228 or len(set(keys)) != len(keys):
            raise ValueError('614 ratio and614 channel parameter tensors required')
        parameters = [value for group in groups for value in group['params']]
        originals = [value.detach().clone() for value in parameters]
        total = [None] * len(parameters)
        device = next(runtime.model.parameters()).device
        frozen_before = frozen_snapshot(switching, parameters, True)
        stats = read(Path(p['source_relational_protocol']['source_dual_root'])
                     / 'results/fit_stats/summary.json')
        partial_path = out / 'gradient_state.pt'
        if partial_path.is_file():
            partial = torch.load(partial_path, map_location='cpu', weights_only=True)
            if (any(partial.get(key) != value for key, value in binding.items())
                    or partial.get('parameter_keys') != keys
                    or len(partial.get('originals', [])) != len(originals)
                    or not all(torch.equal(old, value.cpu()) for old, value
                               in zip(partial['originals'], originals))
                    or len(partial.get('gradient_sum', [])) != len(total)):
                raise ValueError('Partial C80 gradient binding or original parameters differ')
            done = partial['case_metrics']
            expected_prefix = binding['case_ids'][:len(done)]
            if [row['case_id'] for row in done] != expected_prefix:
                raise ValueError('Partial gradients are not an original case-order prefix')
            loss_sum = float(partial['loss_sum'])
            total = [None if value is None else value.to(device=device)
                     for value in partial['gradient_sum']]
            for value, parameter in zip(total, parameters):
                if value is not None and (value.shape != parameter.shape
                                          or not bool(torch.isfinite(value).all())):
                    raise ValueError('Invalid resumed gradient tensor')
            state['reused_case_gradients'] = len(done)
            state['resumed_from_pbs_jobid'] = partial['pbs_jobid']
        anchored = False

        @contextmanager
        def use_reader():
            state['reader_gpu_loads'] += 1
            with parked_student_weights(torch, switching):
                with resident_weights(torch, switching.original_weights, device) as weights:
                    active = BF16Reader(runtime, weights)
                    try:
                        yield active
                    finally:
                        active.original_weights.clear()

        for case in cases[len(done):]:
            if deadline.near(600):
                state.update(phase='deadline_checkpointed', completed_cases=len(done))
                checkpoint()
                save(out / 'summary.json', state)
                return 0
            state['case_id'] = case['case_id']
            datums, identities, _ = _load_case_inputs(
                torch, np, switching, root, case,
                root / 'data/calibration_pairs' / str(case['case_id']) / 'metadata.json')
            runtime.description = case['description']
            native = {}
            for source_arm in (arm, 'bf16'):
                native[source_arm] = {}
                for variant in ('clean', 'noisy'):
                    source_unit = f'calibration_{shard}'
                    value = torch.load(root / 'results' / source_unit / 'records'
                                       / f"{case['case_id']}_{source_arm}_{variant}.pt",
                                       map_location='cpu', weights_only=True)
                    validate_binding(value, {
                        **_pair_binding(p['protocol'], source_unit, case, source_arm, variant),
                        'input_identity': identities[variant]})
                    if value['counts'] != {'video_steps': 10, 'action_steps': 10}:
                        raise ValueError('Native source sampler counts differ')
                    native[source_arm][variant] = value
            for variant in ('clean', 'noisy'):
                if not torch.equal(native[arm][variant]['action_initial_noise'],
                                   native['bf16'][variant]['action_initial_noise']):
                    raise ValueError('Native student/teacher initial action noise differs')

            def read_future(reader, variant, future, gradient):
                state['new_bf16_reader_passes'] += 1
                save(out / 'progress.json', {**state, 'stage': 'frozen_reader',
                                            'variant': variant, 'gradient': gradient})
                return reader(runtime, datums[variant], case['sampler_seed'], future,
                              checkpointed=gradient)[0]

            cotangents = {variant: (None, None) for variant in ('clean', 'noisy')}
            if kind == 'response':
                leaves = {v: native[arm][v]['action'].to(device).detach().requires_grad_(True)
                          for v in ('clean', 'noisy')}
                loss = action_response_kl(
                    torch, leaves['clean'], leaves['noisy'],
                    native['bf16']['clean']['action'].to(device),
                    native['bf16']['noisy']['action'].to(device), stats)
                vectors = torch.autograd.grad(loss, [leaves['clean'], leaves['noisy']])
                cotangents = {v: (vector.detach(), None)
                              for v, vector in zip(('clean', 'noisy'), vectors)}
                pair_loss = float(loss.detach())
                del leaves, vectors, loss
            else:
                pair_loss = 0.0
                with use_reader() as reader:
                    for variant in ('clean', 'noisy'):
                        if not anchored:
                            reference = read_future(
                                reader, variant, native['bf16'][variant]['future'], False)
                            if not torch.equal(reference.detach().float().cpu(),
                                               native['bf16'][variant]['action']):
                                raise RuntimeError('Staged reader/BF16 native parity differs')
                            state['reader_native_parity'].append({
                                'case_id': case['case_id'], 'variant': variant, 'action': 0.0})
                            del reference
                        leaf = native[arm][variant]['future'].to(
                            device=device, dtype=getattr(runtime.model, 'torch_dtype', torch.bfloat16)
                        ).detach().requires_grad_(True)
                        plain = read_future(reader, variant, leaf.detach(), False).detach()
                        action = read_future(reader, variant, leaf, True)
                        if not torch.equal(action.detach(), plain):
                            raise RuntimeError('Reader gradient/no-grad action parity differs')
                        loss = 0.5 * latent_control_kl(
                            torch, action, native['bf16'][variant]['action'].to(device), stats)
                        vector, = torch.autograd.grad(loss, leaf)
                        if not bool(torch.isfinite(vector).all()) or vector[:, :, 0].abs().max() != 0:
                            raise RuntimeError('Reader gradient or observed-frame clamp differs')
                        cotangents[variant] = (None, vector.detach())
                        pair_loss += float(loss.detach())
                        del leaf, plain, action, loss, vector
                anchored = True

            def forward(variant):
                if deadline.near(300):
                    raise TimeoutError('Global gradient-check deadline reached')
                if banks.cached or not all(value.requires_grad for value in parameters):
                    raise RuntimeError('Dynamic D/r graph must remain enabled')
                before_rng = capture_rng(torch)
                banks.seen.clear()
                state['new_student_full_queries_attempted'] += 1
                save(out / 'progress.json', {**state, 'stage': 'student_vjp', 'variant': variant})
                try:
                    with torch.enable_grad():
                        action, counts, future = switching.sampler.differentiable_action(
                            runtime, datums[variant], case['sampler_seed'],
                            checkpointed=True, return_future=True)
                    if len(banks.seen) != 614 or counts != {'video_steps': 10, 'action_steps': 10}:
                        raise RuntimeError('Student gradient forward coverage differs')
                    if (not torch.equal(action.detach().float().cpu(), native[arm][variant]['action'])
                            or not torch.equal(future.detach().float().cpu(), native[arm][variant]['future'])):
                        raise RuntimeError('Original-parameter native/dynamic parity failed')
                    state['new_student_full_queries_completed'] += 1
                    return action, future
                finally:
                    restore_rng(torch, before_rng)

            grads = paired_vjp(torch, parameters, forward, cotangents)
            if not any(g is not None for g in grads) or not all(
                    bool(torch.isfinite(g).all()) for g in grads if g is not None):
                raise RuntimeError('Disconnected or non-finite C80 surrogate gradient')
            if not all(any(g is not None for g in part)
                       for part in (grads[:614], grads[614:])):
                raise RuntimeError('Both ratio and channel groups must be connected')
            if not all(torch.equal(value.detach(), old) for value, old in zip(parameters, originals)):
                raise RuntimeError('Frozen original D/r changed during gradient extraction')
            total = add_gradients(total, grads)
            loss_sum += pair_loss
            done.append({'case_id': case['case_id'], 'loss': pair_loss})
            state.update(completed_cases=len(done), native_parity_all_zero=True)
            if len(done) % 5 == 0 or len(done) == len(cases):
                checkpoint()
            save(out / 'summary.json', state)
            del grads, cotangents, datums, native

        if any(not bool(torch.isfinite(g).all()) for g in total if g is not None):
            raise RuntimeError('Gradient sum is non-finite')
        baseline = torch.load(p['source_closed_protocol']['sources']['baseline_deployment'],
                              map_location='cpu', weights_only=True)
        state['frozen_vq'] = banks.verify_frozen(baseline['module_masters'])
        if frozen_snapshot(switching, parameters, True) != frozen_before:
            raise RuntimeError('Frozen reader/model/banks changed')
        state.update(complete=True, phase='complete', completed_cases=40,
                     case_ids=binding['case_ids'], native_parity_all_zero=True,
                     original_parameters_restored=True, frozen_reader_and_model_unchanged=True,
                     baseline_loss=loss_sum / 40, saved_trained_checkpoints=0,
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        checkpoint(True)
        save(out / 'summary.json', state)
        return 0
    except Exception as exc:
        state.update(phase='failed', failure=f'{type(exc).__name__}: {exc}',
                     completed_cases=len(done))
        if originals:
            checkpoint()
        save(out / 'summary.json', state)
        raise
    finally:
        if originals:
            with torch.no_grad():
                for value, old in zip(parameters, originals):
                    value.copy_(old)
            if not all(torch.equal(value.detach(), old) for value, old in zip(parameters, originals)):
                raise RuntimeError('Original parameter restoration failed')
        if switching is not None:
            switching.close()


if __name__ == '__main__':
    raise SystemExit(main())
