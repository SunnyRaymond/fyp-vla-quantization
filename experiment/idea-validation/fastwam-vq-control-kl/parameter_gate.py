"""Calibration-only D/r gradient and reversible finite-step implementation probe."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys

from control_common import Deadline, guard, read, save
from control_losses import action_response_kl, latent_control_kl

VARIANTS = ('clean', 'noisy')


def paired_vjp(torch, parameters, forward, cotangents):
    """Apply the chain rule without keeping two student/reader graphs resident."""
    total = [None] * len(parameters)
    for variant in VARIANTS:
        action, future = forward(variant)
        values, vectors = [], []
        for value, vector in zip((action, future), cotangents[variant]):
            if vector is not None:
                values.append(value)
                vectors.append(vector.to(device=value.device, dtype=value.dtype))
        grads = torch.autograd.grad(values, parameters, grad_outputs=vectors,
                                    allow_unused=True)
        total = [old if grad is None else grad.detach().clone() if old is None
                 else old + grad.detach() for old, grad in zip(total, grads)]
        del action, future, values, vectors, grads
    return total


@contextmanager
def projected_step(torch, parameters, originals, delta, fraction, project):
    try:
        with torch.no_grad():
            for parameter, original, step in zip(parameters, originals, delta):
                parameter.copy_(original + fraction * step)
            project()
        yield [parameter.detach() - original
               for parameter, original in zip(parameters, originals)]
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)


@contextmanager
def resident_weights(torch, masters, device):
    if any(value.device.type != 'cpu' or value.requires_grad for value in masters.values()):
        raise ValueError('Immutable CPU reader masters required')
    weights = {}
    try:
        for name, value in masters.items():
            weights[name] = value.to(device=device, copy=True)
        yield weights
    finally:
        unchanged = all(torch.equal(value.detach().cpu(), masters[name])
                        for name, value in weights.items())
        weights.clear()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
        if not unchanged:
            raise RuntimeError('Resident BF16 reader weights changed')


@contextmanager
def parked_student_weights(torch, switching):
    """Reader forwards bypass these weights; call only after student backward."""
    modules = switching.runtime.q.modules
    first = next(switching.runtime.model.parameters())
    parked = {name: module.weight.device for name, module in modules.items()
              if module.weight is not first}
    model_device = getattr(switching.runtime.model, 'device', first.device)
    succeeded = False
    try:
        for name in parked:
            module = modules[name]
            if module.weight.requires_grad:
                raise RuntimeError('Student dense weights must remain frozen')
            module.weight.data = switching.decoded_base_weights[name]
        if getattr(switching.runtime.model, 'device', first.device) != model_device:
            raise RuntimeError('Parking student weights changed the sampler device')
        if first.device.type == 'cuda':
            torch.cuda.empty_cache()
        yield
        succeeded = True
    finally:
        # On failure the process exits with unchanged CPU values; avoid another
        # GPU allocation while a failed autograd traceback may retain its graph.
        if succeeded:
            for name, device in parked.items():
                module = modules[name]
                base = switching.decoded_base_weights[name]
                module.weight.data = base.to(device=device)
                if not torch.equal(module.weight.detach().cpu(), base):
                    raise RuntimeError('Student dense-weight roundtrip changed values')


def dynamic_student(switching, arm):
    switching.activate(arm)
    for name, module in switching.runtime.q.modules.items():
        module.forward = switching.student_forwards[name]
        module.weight.data = switching.decoded_base_weights[name].to(module.weight.device)
    switching.banks.cached = False
    for parameter in (*switching.banks.ratios.values(), *switching.banks.shadows.values()):
        parameter.requires_grad_(True)


def frozen_snapshot(switching, parameters, relocatable_student_weights=False):
    from reader_worker import _parameter_snapshot
    snapshot = _parameter_snapshot(switching.runtime.model,
                                   switching.original_weights, switching.banks)
    # D/r copy_ rollback changes version counters even when values are exact.
    mutable = {parameter.data_ptr() for parameter in parameters}
    snapshot['bank_tensors'] = {key: sig for key, sig in snapshot['bank_tensors'].items()
                               if sig[0] not in mutable}
    if relocatable_student_weights:
        target_ids = {id(module.weight) for module in switching.runtime.q.modules.values()}
        for name, parameter in switching.runtime.model.named_parameters():
            if id(parameter) in target_ids:
                sig = snapshot['parameters'][name]
                snapshot['parameters'][name] = (id(parameter), *sig[1:])
    return snapshot


def main():
    guard()
    import numpy as np
    import torch
    from aggregate import completed
    from calibration_worker import _load_case_inputs, _pair_binding
    from diagnostic_runtime import SwitchingRuntime, validate_binding
    from frozen_reader import BF16Reader
    from checkpointing import capture_rng, restore_rng

    p = read(Path(__file__).with_name('protocol.json'))
    config = read(Path(__file__).with_name('parameter_gate_protocol.json'))
    if config['parent_protocol'] != p['protocol'] or config['case_id'] != 2000:
        raise ValueError('Frozen parameter-gate binding differs')
    root = Path(p['remote_root'])
    completed(root, 'aggregate_calibration')
    completed(root, 'reader_0')
    pair_summary = completed(root, 'calibration_pairs')
    case = next(row for row in pair_summary['records']
                if row['case_id'] == config['case_id'])
    if case['training_use'] is not False or case['sampler_seed'] != case['row']['sampler_seeds'][0]:
        raise ValueError('Original diagnostic-only C80 pair and seed required')
    unit, arm = 'parameter_gate', config['checkpoint_arm']
    out = root / 'results' / unit
    deadline = Deadline()
    state = {'protocol': p['protocol'], 'probe_protocol': config, 'unit': unit,
             'pbs_jobid': os.environ['PBS_JOBID'], 'complete': False,
             'new_student_full_queries': 0, 'new_bf16_reader_passes': 0,
             'new_teacher_full_queries': 0, 'training_updates': 0,
             'heldout_inputs_used': False, 'sensor_evaluation_inputs_used': False,
             'components': [], 'native_parity': [], 'reader_native_parity': [],
             'reader_gpu_loads': 0,
             'memory_strategy': 'CPU BF16 masters during student work; CPU student dense weights during reader work'}
    save(out / 'summary.json', state)
    spec = importlib.util.spec_from_file_location('control_direction_helpers',
                                                 config['direction_helper'])
    directions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(directions)
    switching, parameters, originals = None, [], []
    try:
        switching = SwitchingRuntime(p, out)
        runtime, banks = switching.runtime, switching.banks
        datums, identities, _ = _load_case_inputs(
            torch, np, switching, root, case,
            root / 'data/calibration_pairs' / str(case['case_id']) / 'metadata.json')
        native = {}
        source_unit = 'calibration_' + str(case['shard'])
        for source_arm in (arm, 'bf16'):
            native[source_arm] = {}
            for variant in VARIANTS:
                path = root / 'results' / source_unit / 'records' / (
                    f"{case['case_id']}_{source_arm}_{variant}.pt")
                value = torch.load(path, map_location='cpu', weights_only=True)
                binding = {**_pair_binding(p['protocol'], source_unit, case, source_arm, variant),
                           'input_identity': identities[variant]}
                validate_binding(value, binding)
                if value['counts'] != {'video_steps': 10, 'action_steps': 10}:
                    raise ValueError('Native C80 sampler counts differ')
                native[source_arm][variant] = value
        for variant in VARIANTS:
            if not torch.equal(native[arm][variant]['action_initial_noise'],
                               native['bf16'][variant]['action_initial_noise']):
                raise ValueError('Native student/teacher initial action noise differs')
        dynamic_student(switching, arm)
        # Student backward does not use BF16 reader weights; keep that copy off GPU.
        switching.original_weights = {
            name: value.detach().cpu() for name, value in switching.original_weights.items()}
        torch.cuda.empty_cache()
        groups = banks.optimizer_groups(p['source_closed_protocol']['training'],
                                        p['source_closed_protocol']['scaling'])
        for group, name in zip(groups, ('ratio', 'channel')):
            group['name'] = name
            if group['lr'] != config[name + '_learning_rate']:
                raise ValueError('Original D/r learning rates differ')
        parameters = [value for group in groups for value in group['params']]
        originals = [value.detach().clone() for value in parameters]
        frozen_before = frozen_snapshot(switching, parameters, True)
        device = next(runtime.model.parameters()).device
        stats = read(Path(p['source_relational_protocol']['source_dual_root'])
                     / 'results/fit_stats/summary.json')
        runtime.description = case['description']
        state['teacher_response_rms'] = float((
            native['bf16']['noisy']['action'][:10, :6]
            - native['bf16']['clean']['action'][:10, :6]
        ).square().mean().sqrt())

        def forward(variant, gradient=False):
            if deadline.near(300):
                raise TimeoutError('Parameter probe reached its global soft deadline')
            if banks.cached:
                raise RuntimeError('Student VJP must keep dynamic quantization active')
            before_rng = capture_rng(torch)
            banks.seen.clear()
            state['new_student_full_queries'] += 1
            save(out / 'progress.json', {**state, 'stage': 'student_forward',
                                        'variant': variant, 'gradient': gradient})
            try:
                with torch.set_grad_enabled(gradient):
                    action, counts, future = switching.sampler.differentiable_action(
                        runtime, datums[variant], case['sampler_seed'],
                        checkpointed=gradient, return_future=True)
                if len(banks.seen) != 614 or counts != {'video_steps': 10, 'action_steps': 10}:
                    raise RuntimeError('Student dynamic-forward coverage/counts differ')
                if gradient:
                    errors = {name: float((value.detach().float().cpu()
                                           - native[arm][variant][name]).abs().max())
                              for name, value in (('action', action), ('future', future))}
                    if any(errors.values()):
                        raise RuntimeError('Same-parameter native/dynamic parity failed: ' + str(errors))
                    state['native_parity'].append({'objective': state['objective'],
                                                   'variant': variant, **errors})
                return action, future
            finally:
                restore_rng(torch, before_rng)

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

        def read_future(reader, variant, future, gradient=False):
            state['new_bf16_reader_passes'] += 1
            save(out / 'progress.json', {**state, 'stage': 'frozen_reader',
                                        'variant': variant, 'gradient': gradient})
            return reader(runtime, datums[variant], case['sampler_seed'], future,
                          checkpointed=gradient)[0]

        def response(actions):
            return action_response_kl(torch, actions['clean'], actions['noisy'],
                                      native['bf16']['clean']['action'].to(device),
                                      native['bf16']['noisy']['action'].to(device), stats)

        def reader_loss(variant, action):
            return latent_control_kl(torch, action,
                                     native['bf16'][variant]['action'].to(device), stats)

        for objective in config['objectives']:
            state['objective'] = objective
            if deadline.near(600):
                raise TimeoutError('Insufficient time for another reversible component probe')
            cotangents = {variant: (None, None) for variant in VARIANTS}
            if objective == 'action_response_kl':
                leaves = {v: native[arm][v]['action'].to(device).detach().requires_grad_(True)
                          for v in VARIANTS}
                loss = response(leaves)
                vectors = torch.autograd.grad(loss, [leaves[v] for v in VARIANTS])
                cotangents = {v: (vector.detach(), None)
                              for v, vector in zip(VARIANTS, vectors)}
                baseline = float(loss.detach())
            elif objective == 'latent_control_kl':
                baseline = 0.0
                with use_reader() as reader:
                    for variant in VARIANTS:
                        reference = read_future(
                            reader, variant, native['bf16'][variant]['future'], False)
                        if not torch.equal(reference.detach().float().cpu(),
                                           native['bf16'][variant]['action']):
                            raise RuntimeError('CPU-staged reader/BF16 native parity differs')
                        state['reader_native_parity'].append({'variant': variant, 'action': 0.0})
                        leaf = native[arm][variant]['future'].to(
                            device=device, dtype=getattr(runtime.model, 'torch_dtype', torch.bfloat16)
                        ).detach().requires_grad_(True)
                        plain = read_future(reader, variant, leaf.detach(), False).detach()
                        action = read_future(reader, variant, leaf, True)
                        if not torch.equal(action.detach(), plain):
                            raise RuntimeError('Reader gradient/no-grad action parity differs')
                        loss = 0.5 * reader_loss(variant, action)
                        vector, = torch.autograd.grad(loss, leaf)
                        if not torch.isfinite(vector).all() or vector[:, :, 0].abs().max() != 0:
                            raise RuntimeError('Reader future gradient/frame0 clamp differs')
                        cotangents[variant] = (None, vector.detach())
                        baseline += float(loss.detach())
                        del reference, leaf, plain, action, loss, vector
            else:
                raise ValueError('Unfrozen probe objective')
            grads = paired_vjp(torch, parameters, lambda v: forward(v, True), cotangents)
            summary, offset = {}, 0
            for group in groups:
                part = grads[offset:offset + len(group['params'])]
                live = [g for g in part if g is not None]
                if not live or not all(bool(torch.isfinite(g).all()) for g in live):
                    raise RuntimeError('Missing or non-finite D/r surrogate gradient')
                summary[group['name']] = {
                    'parameter_tensors': len(part), 'connected_tensors': len(live),
                    'nonzero_tensors': sum(bool(g.count_nonzero()) for g in live),
                    'norm': float(sum(g.float().square().sum() for g in live).sqrt())}
                offset += len(part)
            norm = sum(g.float().square().sum() for g in grads if g is not None).sqrt()
            factor = min(1.0, config['gradient_norm_clip'] / (float(norm) + 1e-6))
            delta = directions.adam_delta(
                torch, [None if g is None else g * factor for g in grads], groups)
            component = {'objective': objective, 'baseline_loss': baseline,
                         'surrogate_gradient': summary, 'gradient_clip_factor': factor,
                         'complete': False, 'candidates': [],
                         'scope': 'single fixed C80 pair; zero gradients can reflect weak signal'}
            for fraction in config['fractions']:
                with projected_step(torch, parameters, originals, delta, fraction, banks.project) as actual:
                    predicted = sum(float((g * step).sum()) for g, step in zip(grads, actual)
                                    if g is not None)
                    outputs = {v: forward(v, False) for v in VARIANTS}
                    if objective == 'action_response_kl':
                        value = float(response({v: outputs[v][0] for v in VARIANTS}).detach())
                    else:
                        with use_reader() as reader:
                            value = sum(0.5 * float(reader_loss(
                                v, read_future(reader, v, outputs[v][1], False)).detach())
                                        for v in VARIANTS)
                    component['candidates'].append({
                        'fraction': fraction, 'surrogate_g_dot_delta': predicted,
                        'actual_loss': value, 'actual_loss_delta': value - baseline,
                        'actual_descent': value < baseline,
                        'projected_step_l2': float(sum(x.float().square().sum()
                                                       for x in actual).sqrt())})
                if not all(torch.equal(value.detach(), original)
                           for value, original in zip(parameters, originals)):
                    raise RuntimeError('D/r candidate did not roll back exactly')
                save(out / (objective + '.json'), component)
            component['complete'] = True
            save(out / (objective + '.json'), component)
            state['components'].append(component)
            save(out / 'summary.json', state)
        baseline_checkpoint = torch.load(
            p['source_closed_protocol']['sources']['baseline_deployment'],
            map_location='cpu', weights_only=True)
        state['frozen_vq'] = banks.verify_frozen(baseline_checkpoint['module_masters'])
        if frozen_snapshot(switching, parameters, True) != frozen_before:
            raise RuntimeError('Frozen reader/model/bank state changed during probe')
        state.update(complete=True, phase='complete', original_parameters_restored=True,
                     frozen_reader_and_model_unchanged=True, saved_trained_checkpoints=0,
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     full_c80_optimization_proven=False,
                     query_count_scope='logical sampler calls; checkpoint recomputation is additional compute')
        save(out / 'summary.json', state)
        return 0
    except Exception as exc:
        state.update(phase='failed', failure=f'{type(exc).__name__}: {exc}')
        save(out / 'summary.json', state)
        raise
    finally:
        if originals:
            with torch.no_grad():
                for value, original in zip(parameters, originals):
                    value.copy_(original)
            if not all(torch.equal(value.detach(), original)
                       for value, original in zip(parameters, originals)):
                raise RuntimeError('Final D/r rollback failed')
        if switching is not None:
            switching.close()


if __name__ == '__main__':
    sys.exit(main())
