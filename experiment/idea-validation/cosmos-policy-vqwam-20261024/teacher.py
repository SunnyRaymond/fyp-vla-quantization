"""C80 Cosmos teacher packets/statistics from existing original-LIBERO captures."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from common import (read_json, save_json, load_tensors, save_tensors, cpu_tree,
                    prepare_runtime, get_context, endpoint, has_time, torch_module,
                    check_tensor_parity, target_manifest)


def main():
    parser = argparse.ArgumentParser()
    for key in ('protocol', 'root', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--soft-deadline-seconds', type=int, default=6600)
    args = parser.parse_args()
    p, root, out = read_json(args.protocol), Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch = torch_module()
    runtime = prepare_runtime(p, root, out)
    modules = runtime.target_linears
    cases = list(p['calibration']['case_ids'])
    partial = out / 'statistics_partial.pt'
    saved = load_tensors(partial) if partial.exists() else {'statistics': {}, 'completed_cases': []}
    statistics = saved['statistics']
    completed = set(saved['completed_cases'])
    device = next(runtime.model.parameters()).device

    def to_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {k: to_device(v) for k, v in value.items()}
        return value

    statistics = to_device(statistics)
    active_step = [0]
    runtime.statistics_step = lambda step: active_step.__setitem__(0, int(step))

    def accumulate(bucket, rows):
        maximum = rows.abs().amax(0)
        square = rows.square().sum(0)
        if 'absmax' not in bucket:
            bucket.update(absmax=maximum, sum_sq=square, count=rows.shape[0])
        else:
            bucket['absmax'] = torch.maximum(bucket['absmax'], maximum)
            bucket['sum_sq'].add_(square)
            bucket['count'] += rows.shape[0]

    def make_hook(name, width):
        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            rows = inputs[0].detach().float().reshape(-1, width)
            bucket = statistics.setdefault(name, {})
            by_step = bucket.setdefault('by_timestep', {})
            step_bucket = by_step.setdefault(str(active_step[0]), {})
            accumulate(bucket, rows)
            accumulate(step_bucket, rows)
        return hook

    parity_path = out / 'endpoint_parity.json'
    if not parity_path.exists() or read_json(parity_path).get('status') != 'passed':
        case = cases[0]
        official = runtime.c8_query(case, capture=True)
        context = get_context(runtime, case)
        result = endpoint(runtime, context)
        captured = official.get('captured', {})
        generated = official.get('generated_latent', captured.get('generated_latent'))
        if generated is None:
            raise RuntimeError('Official query did not retain the generated latent')
        action = torch.as_tensor(official['actions'], device=result.action.device).reshape_as(result.action)
        checks = {'action': check_tensor_parity(result.action, action, 2e-3, 2e-3),
                  'latent': check_tensor_parity(result.generated_latent, generated, 2e-3, 2e-3)}
        grad_result = endpoint(runtime, context, grad_enabled=True, checkpoint_denoiser=False)
        checks['grad_mode_action'] = check_tensor_parity(grad_result.action, result.action, 2e-3, 2e-3)
        checks['grad_mode_latent'] = check_tensor_parity(grad_result.generated_latent,
                                                     result.generated_latent, 2e-3, 2e-3)
        passed = all(v['passed'] for v in checks.values())
        save_json(parity_path, {'status': 'passed' if passed else 'failed', 'case_id': case,
                               'pbs_job_id': os.environ['PBS_JOBID'], 'checks': checks})
        if not passed:
            raise RuntimeError('Native/differentiable Cosmos teacher parity failed')
        del result, grad_result, official

    hooks = [module.register_forward_pre_hook(make_hook(name, module.in_features))
             for name, module in modules.items()]
    try:
        for case in cases:
            if case in completed:
                continue
            if not has_time(args.soft_deadline_seconds, 180):
                break
            context = get_context(runtime, case)
            result = endpoint(runtime, context, collect_packets=True)
            if result.denoiser_calls != int(p['inference']['denoising_steps']):
                raise RuntimeError('Actual teacher denoiser calls differ from frozen five-NFE recipe')
            target = {'case_id': case, 'seed': context['seed'], 'action': cpu_tree(result.action),
                      'future_image_slots': cpu_tree(result.future_image_slots),
                      'generated_latent': cpu_tree(result.generated_latent),
                      'protocol': p['protocol']}
            save_tensors(out / 'targets' / f'{case}.pt', target)
            save_tensors(out / 'packets' / f'{case}.pt', result.packets)
            completed.add(case)
            if len(completed) % 4 == 0 or len(completed) == len(cases):
                save_tensors(partial, {'statistics': cpu_tree(statistics),
                                      'completed_cases': sorted(completed)})
            save_json(out / 'progress.json', {'phase': 'teacher', 'completed_cases': len(completed),
                                             'expected_cases': len(cases), 'last_case': case})
            print(f'TEACHER_CASE case={case} completed={len(completed)}/{len(cases)}', flush=True)
            del result
    finally:
        for hook in hooks:
            hook.remove()
        del runtime.statistics_step
    save_tensors(partial, {'statistics': cpu_tree(statistics), 'completed_cases': sorted(completed)})
    complete = completed == set(cases)
    if complete:
        save_tensors(out / 'statistics.pt', cpu_tree(statistics))
        if set(statistics) != set(modules):
            raise RuntimeError('Teacher activation statistics do not cover all target modules')
        actions = torch.cat([load_tensors(out / 'targets' / f'{case}.pt')['action'].float()
                             for case in cases], 0)
        lo = torch.as_tensor(runtime.dataset_stats['actions_min']).float()
        hi = torch.as_tensor(runtime.dataset_stats['actions_max']).float()
        normalized = (2 * (actions - lo) / (hi - lo).clamp_min(1e-12) - 1)[..., :6]
        flat = normalized.reshape(-1, 6)
        save_tensors(out / 'action_statistics.pt', {'mean': flat.mean(0),
                     'scale': flat.std(0, correction=0).clamp_min(p['training']['reader']['std_floor']),
                     'actions_min': lo, 'actions_max': hi, 'contexts': len(cases)})
    save_json(out / 'summary.json', {'protocol': p['protocol'], 'stage': 'teacher',
        'status': 'complete' if complete else 'checkpointed', 'pbs_job_id': os.environ['PBS_JOBID'],
        'completed_cases': sorted(completed), 'expected_cases': cases,
        'target_modules': target_manifest(runtime), 'shared_inputs_reused': True,
        'teacher_model': p['model'], 'teacher_targets_from_fastwam': False,
        'endpoint_parity': read_json(parity_path), 'statistics_modules': len(statistics)})


if __name__ == '__main__':
    main()
