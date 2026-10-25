"""C80 R128 + Microscaling VQ A/Z/frozen-joint-reader calibration."""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import random
from types import SimpleNamespace

from common import (read_json, save_json, load_tensors, save_tensors, cpu_tree,
    prepare_runtime, get_context, endpoint, has_time, torch_module, restore_bf16,
    check_tensor_parity, target_manifest)
from training_state import (RotationState, VQState, get_transforms, prepare_vq, deploy,
                            rotation_orthogonality)
from reader import LoopCapture, FrozenSuffixReader


def loss_terms(result, target):
    from endpoint import endpoint_losses
    torch = torch_module()
    reference = SimpleNamespace(action=target['action'].to(result.action.device),
        future_image_slots={k: v.to(result.action.device) for k, v in target['future_image_slots'].items()})
    return endpoint_losses(result, reference, action_width=16 * 7)


def balanced_batches(contexts):
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    pools = {suite: sorted(case for case, context in contexts.items()
                          if context['metadata']['suite'] == suite) for suite in suites}
    if any(len(pool) != 20 for pool in pools.values()):
        raise RuntimeError('C80 must contain exactly20 contexts in each frozen suite')
    generator = random.Random(20261024)
    for pool in pools.values():
        generator.shuffle(pool)
    cursors = {suite: 0 for suite in suites}
    batches = []
    for batch in range(8):
        result = []
        for i, suite in enumerate(suites):
            number = 3 if (i - batch) % 4 < 2 else 2
            start = cursors[suite]
            result += pools[suite][start:start + number]
            cursors[suite] += number
        generator.shuffle(result)
        batches.append(result)
    if len(set(case for batch in batches for case in batch)) != 80:
        raise RuntimeError('C80 schedule duplicated or omitted a context')
    return batches


def target(root, case):
    return load_tensors(Path(root) / 'results/teacher/targets' / f'{case}.pt')


def student_capture(runtime, context, state, gradient):
    parameters = state.parameters()
    original = [v.requires_grad for v in parameters]
    if not gradient:
        for value in parameters:
            value.requires_grad_(False)
    try:
        with LoopCapture(runtime.protocol['training']['reader']['cut_evaluation'],
                         retain_graph=gradient) as capture:
            result = endpoint(runtime, context, grad_enabled=True,
                              checkpoint_denoiser=gradient)
        return result, capture.require()
    finally:
        for value, requires in zip(parameters, original):
            value.requires_grad_(requires)


def vq_initial_metrics(runtime, p, root, out, state, contexts, statistics, deadline):
    path = Path(out) / 'state/vq_initial_metrics.json'
    metrics = read_json(path) if path.exists() else {}
    for case, context in contexts.items():
        if str(case) in metrics:
            continue
        if not has_time(deadline, 240):
            return None
        restore_bf16(runtime)
        state.install()
        result, capture = student_capture(runtime, context, state, False)
        a, z = loss_terms(result, target(root, case))
        noisy = cpu_tree(capture.live_x)
        measured = {'action_mse': float(a.detach()), 'latent_nmse': float(z.detach())}
        del result, capture, a, z
        restore_bf16(runtime)
        reader = FrozenSuffixReader(runtime, context)
        identity = reader.identity()
        if not identity['passed']:
            raise RuntimeError('Frozen joint-reader teacher identity failed')
        _, diagnostic, _ = reader.cotangent(noisy, statistics)
        measured.update(diagnostic)
        measured['identity'] = identity
        metrics[str(case)] = measured
        save_json(path, metrics)
        print(f'OURS_INITIAL_READER case={case} completed={len(metrics)}/80 kl={measured["reader_kl"]:.7g}', flush=True)
        del reader, noisy
        gc.collect()
    n = len(metrics)
    mean = lambda key: sum(v[key] for v in metrics.values()) / n
    action, latent, kl = mean('action_mse'), mean('latent_nmse'), mean('reader_kl')
    connected = sum(v['gradient_norm'] > 0 for v in metrics.values())
    if n != 80 or connected == 0 or kl <= 1e-12:
        raise RuntimeError('C80 reader auxiliary is absent or numerically uninformative')
    coefficient = p['training']['rho'] * action / max(latent, 1e-12)
    beta = p['training']['reader']['beta_fraction'] * coefficient * latent / kl
    return {'lambda': coefficient, 'beta': beta, 'contexts': n,
            'initial_action_mse': action, 'initial_latent_nmse': latent,
            'initial_reader_kl': kl, 'connected_reader_cases': connected,
            'frozen_before_update1': True}


def rotation_initial_metrics(runtime, p, root, out, rotation, contexts, deadline):
    path = Path(out) / 'state/rotation_initial_metrics.json'
    metrics = read_json(path) if path.exists() else {}
    rotation.prewarm()
    rotation.install()
    for case, context in contexts.items():
        if str(case) in metrics:
            continue
        if not has_time(deadline, 120):
            return None
        result = endpoint(runtime, context)
        action, latent = loss_terms(result, target(root, case))
        metrics[str(case)] = {'action_mse': float(action), 'latent_nmse': float(latent)}
        if len(metrics) % 4 == 0:
            save_json(path, metrics)
        del result
    save_json(path, metrics)
    a = sum(v['action_mse'] for v in metrics.values()) / len(metrics)
    z = sum(v['latent_nmse'] for v in metrics.values()) / len(metrics)
    return p['training']['rho'] * a / max(z, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    for key in ('protocol', 'root', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--stage', default='ours')
    parser.add_argument('--soft-deadline-seconds', type=int, default=6600)
    args = parser.parse_args()
    p, root, out = read_json(args.protocol), Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'state').mkdir(exist_ok=True)
    teacher = read_json(root / 'results/teacher/summary.json')
    if teacher.get('status') != 'complete' or teacher.get('completed_cases') != p['calibration']['case_ids']:
        raise RuntimeError('Complete frozen C80 teacher preparation is required')
    reader_math = read_json(root / 'results/reader_math/reader_math.json')
    if (reader_math.get('status') != 'passed' or reader_math.get('nfe') != 5
            or reader_math.get('cut') != p['training']['reader']['cut_evaluation']
            or not reader_math.get('checks')
            or not all(row.get('passed') for row in reader_math['checks'].values())):
        raise RuntimeError('Original-solver CPU reader math checks must pass before training')
    torch = torch_module()
    runtime = prepare_runtime(p, root, out)
    runtime.controller._backup_weights()
    contexts = {case: get_context(runtime, case) for case in p['calibration']['case_ids']}
    batches = balanced_batches(contexts)
    save_json(out / 'schedule.json', {'batches': batches, 'contexts': 80, 'batch_size': 10,
                                    'seed': 20261024, 'heldout_used': False})
    path = out / 'checkpoint.pt'
    saved = load_tensors(path) if path.exists() else {'rotation_completed': 0, 'vq_completed': 0,
        'rotation_lambda': None, 'coefficients': None, 'history': []}
    transforms = get_transforms(runtime, root, out)
    rotation = RotationState(runtime, transforms, saved.get('rotation_raw'))
    rotation_optimizer = torch.optim.Adam([rotation.raw], lr=p['training']['rotation_lr'])
    if saved.get('rotation_optimizer'):
        rotation_optimizer.load_state_dict(saved['rotation_optimizer'])
    state, optimizer = None, None

    def checkpoint(phase, complete=False, extra=None):
        payload = dict(saved)
        payload.update(rotation_raw=cpu_tree(rotation.raw), rotation_optimizer=cpu_tree(rotation_optimizer.state_dict()))
        if state is not None:
            payload['vq_state'] = state.snapshot()
            payload['vq_optimizer'] = cpu_tree(optimizer.state_dict()) if optimizer is not None else None
        save_tensors(path, payload)
        saved.update(payload)
        summary = {'protocol': p['protocol'], 'arm': 'ours_vqwam',
            'status': 'complete' if complete else 'checkpointed', 'phase': phase,
            'pbs_job_id': os.environ['PBS_JOBID'], 'rotation_completed': saved['rotation_completed'],
            'rotation_expected': 8, 'completed_updates': saved['vq_completed'], 'expected_updates': 8,
            'coefficients': saved.get('coefficients'), 'calibration_contexts': 80,
            'history': saved['history'], 'reader': p['training']['reader'],
            'reader_math_job': reader_math['pbs_job_id'],
            'no_AZ_control_or_ablation': True, 'target_module_count': len(runtime.target_linears)}
        if extra:
            summary.update(extra)
        save_json(out / 'summary.json', summary)
        save_json(out / 'progress.json', summary)

    if saved['rotation_lambda'] is None:
        saved['rotation_lambda'] = rotation_initial_metrics(
            runtime, p, root, out, rotation, contexts, args.soft_deadline_seconds)
        if saved['rotation_lambda'] is None:
            checkpoint('rotation_normalization')
            return
    while saved['rotation_completed'] < 8:
        if not has_time(args.soft_deadline_seconds, 600):
            checkpoint('rotation')
            return
        import time
        started = time.time()
        update = saved['rotation_completed']
        rotation_optimizer.zero_grad(set_to_none=True)
        rotation.install()
        action_values, latent_values = [], []
        for case in batches[update]:
            rotation.prewarm()
            result = endpoint(runtime, contexts[case], grad_enabled=True, checkpoint_denoiser=True)
            action, latent = loss_terms(result, target(root, case))
            objective = action + saved['rotation_lambda'] * latent
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError('Non-finite rotation objective')
            (objective / 10).backward()
            action_values.append(float(action.detach())); latent_values.append(float(latent.detach()))
            del result, action, latent, objective
        if rotation.raw.grad is None or not bool(torch.isfinite(rotation.raw.grad).all()):
            raise RuntimeError('Rotation gradient is absent/non-finite')
        norm = float(torch.nn.utils.clip_grad_norm_([rotation.raw], p['training']['gradient_clip']))
        rotation_optimizer.step()
        rotation.prewarm()
        saved['rotation_completed'] += 1
        saved['history'].append({'phase': 'rotation', 'update': saved['rotation_completed'],
            'action_mse': sum(action_values) / 10, 'latent_nmse': sum(latent_values) / 10,
            'gradient_norm': norm, 'seconds': time.time() - started})
        checkpoint('rotation')
        print(f'OURS_ROTATION_UPDATE {saved["rotation_completed"]}/8', flush=True)
    matrix = rotation.freeze()
    orthogonality = rotation_orthogonality(matrix)
    with torch.no_grad(), torch.autocast(device_type=matrix.device.type, enabled=False):
        legacy_orthogonality = float((matrix.t() @ matrix - torch.eye(
            128, device=matrix.device, dtype=matrix.dtype)).abs().max())
    save_json(out / 'rotation_gate.json', {'protocol': p['protocol'],
        'pbs_job_id': os.environ['PBS_JOBID'], 'rotation_completed': saved['rotation_completed'],
        'matrix_dtype': str(matrix.dtype), 'gram_dtype': 'torch.float64',
        'legacy_fp32_gram_max_abs': legacy_orthogonality,
        'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'orthogonality_max_abs': orthogonality, 'threshold': 5e-5,
        'passed': orthogonality <= 5e-5})
    print(f'OURS_ROTATION_GATE fp64_gram_max_abs={orthogonality:.9g}', flush=True)
    if not orthogonality <= 5e-5:
        raise RuntimeError('Learned Cosmos R128 orthogonality gate failed')
    save_tensors(out / 'state/frozen_rotation.pt', {'raw': cpu_tree(rotation.raw),
                 'matrix': cpu_tree(matrix), 'orthogonality_max_abs': orthogonality})
    checkpoint('vq_fit')
    restore_bf16(runtime)
    if not prepare_vq(runtime, p, root, out, transforms, matrix, contexts, args.soft_deadline_seconds):
        checkpoint('vq_fit')
        return
    restore_bf16(runtime)
    state = VQState(runtime, out / 'initial_banks', transforms, matrix)
    if saved.get('vq_state'):
        state.restore(saved['vq_state'])
    optimizer = torch.optim.Adam(state.parameters(), lr=p['training']['vq_lr'])
    if saved.get('vq_optimizer'):
        optimizer.load_state_dict(saved['vq_optimizer'])
    statistics = load_tensors(root / 'results/teacher/action_statistics.pt')
    if saved['coefficients'] is None:
        saved['coefficients'] = vq_initial_metrics(runtime, p, root, out, state, contexts,
                                                  statistics, args.soft_deadline_seconds)
        if saved['coefficients'] is None:
            checkpoint('reader_normalization')
            return
        checkpoint('reader_normalization')
    coefficients = saved['coefficients']
    while saved['vq_completed'] < 8:
        prior = [row['seconds'] for row in saved['history'] if row['phase'] == 'vq_reader']
        estimate = max(300, int(max(prior[-2:], default=600) * 1.15))
        if not has_time(args.soft_deadline_seconds, estimate):
            checkpoint('vq_reader')
            return
        import time
        started = time.time()
        update = saved['vq_completed']
        optimizer.zero_grad(set_to_none=True)
        metrics, replay_max = [], 0.
        for case in batches[update]:
            restore_bf16(runtime)
            state.install()
            initial, capture = student_capture(runtime, contexts[case], state, False)
            noisy = cpu_tree(capture.live_x)
            initial_action = cpu_tree(initial.action)
            initial_future = cpu_tree(initial.future_image_slots)
            del initial, capture
            restore_bf16(runtime)
            reader = FrozenSuffixReader(runtime, contexts[case])
            identity = reader.identity()
            if not identity['passed']:
                raise RuntimeError('Frozen teacher suffix self-replay failed during training')
            gradient, diagnostic, _ = reader.cotangent(noisy, statistics)
            del reader
            gc.collect()
            state.install()
            result, capture = student_capture(runtime, contexts[case], state, True)
            checks = [check_tensor_parity(result.action, initial_action, atol=1e-6),
                      check_tensor_parity(capture.live_x, noisy, atol=1e-6)]
            checks += [check_tensor_parity(result.future_image_slots[key], initial_future[key], atol=1e-6)
                       for key in initial_future]
            if not all(row['passed'] for row in checks):
                save_json(out / 'staged_replay_failure.json', {'case': case, 'checks': checks})
                raise RuntimeError('No-grad/grad student replay changed the reader intervention point')
            replay_max = max(replay_max, max(row['max_abs'] for row in checks))
            action, latent = loss_terms(result, target(root, case))
            cotangent = gradient.to(device=capture.live_x.device, dtype=capture.live_x.dtype)
            surrogate = (capture.live_x * cotangent).sum()
            objective = action + coefficients['lambda'] * latent + coefficients['beta'] * surrogate
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError('Non-finite VQ reader objective')
            (objective / 10).backward()
            metrics.append({'action_mse': float(action.detach()), 'latent_nmse': float(latent.detach()),
                            **diagnostic})
            print(f'OURS_READER_CASE update={update + 1} case={case} KL={diagnostic["reader_kl"]:.7g}', flush=True)
            del result, capture, noisy, gradient, cotangent, surrogate, action, latent, objective
        parameters = state.parameters()
        connected = sum(value.grad is not None for value in parameters)
        if connected != len(parameters) or not all(bool(torch.isfinite(value.grad).all()) for value in parameters):
            raise RuntimeError('Full VQ codebook/row-gain gradient connectivity failed')
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, p['training']['gradient_clip']))
        optimizer.step()
        state.refresh()
        if update == 0 and p['training']['optimizer_reset_after_update1']:
            optimizer.state.clear()
        saved['vq_completed'] += 1
        measured = {key: sum(row[key] for row in metrics) / len(metrics)
                    for key in ('action_mse', 'latent_nmse', 'reader_kl', 'gradient_norm')}
        saved['history'].append({'phase': 'vq_reader', 'update': saved['vq_completed'],
            **measured, 'parameter_gradient_norm': gradient_norm, 'connected_parameters': connected,
            'staged_replay_max_abs': replay_max, 'seconds': time.time() - started})
        checkpoint('vq_reader')
        print(f'OURS_VQ_READER_UPDATE {saved["vq_completed"]}/8', flush=True)
    state.install()
    manifest = deploy(runtime, p, state, out)
    anchor = endpoint(runtime, contexts[p['calibration']['case_ids'][0]])
    save_tensors(out / 'final_anchor.pt', {'case_id': p['calibration']['case_ids'][0],
                 'action': cpu_tree(anchor.action), 'future_image_slots': cpu_tree(anchor.future_image_slots),
                 'generated_latent': cpu_tree(anchor.generated_latent)})
    checkpoint('complete', complete=True, extra={'deployment_manifest': str(out / 'deployment/manifest.json'),
        'effective_bpw': manifest['effective_bpw'], 'whole_dit_effective_bpw': manifest['whole_dit_effective_bpw'],
        'rotation_orthogonality_max_abs': orthogonality,
        'reader_gate': 'passed; frozen teacher identity, future-only finite cotangent, exact staged replay'})


if __name__ == '__main__':
    main()
