"""Frozen H/R diagnostics, C8 stage boundaries, then matched-budget Joint8."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import time

PROTOCOL = 'fastwam-vq-r128-joint-c8-v1'
CASES = [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070]
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'mixed_loss',
           'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_protocol(p):
    if (p['protocol'] != PROTOCOL or p['calibration']['case_ids'] != CASES
            or p['evaluation']['case_ids'] != list(range(3000, 3040))
            or p['evaluation']['seed_indices'] != [0, 1] or p['target_linears'] != 614
            or p['training']['joint_updates'] != 8 or p['training']['rotation_updates'] != 0
            or p['training']['rotation_learning_rate'] != .0001
            or p['training']['vq_learning_rate'] != .0005
            or p['training']['rho'] != 1.5
            or p['training']['normalization_c'] != .028303017887797623
            or not math.isclose(p['training']['latent_weight'], .04245452683169643, rel_tol=1e-14)
            or p['quantization']['D'] != 1 or p['quantization']['clipping_ratio'] != 1
            or p['quantization']['native_kernel']):
        raise RuntimeError('Frozen joint protocol differs')
    if (p['comparison_arms'] != ['H_W16A4', 'R8_W16A4', 'H_VQ8', 'R8_VQ8', 'Joint8']
            or p['training']['update_order'] != CASES):
        raise RuntimeError('Frozen comparison arms or update order differs')


def run(p):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import numpy as np
    import runner
    import calibration_source as source
    import latent_reference
    from rotation_bank import RotationBanks, VQBanks
    from joint_bank import JointBanks
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, to_cpu, optimizer_to_device
    from selfcheck import run as selfcheck
    from aggregate import write_report

    validate_protocol(p)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(p['training']['random_seed'])
    root, parent = Path(p['remote_root']), Path(p['parent_root'])
    previous, baseline = Path(p['existing_rotation_root']), Path(p['baseline_root'])
    out = root / 'results'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists() and read(out / 'summary.json').get('complete'):
        raise RuntimeError('Already complete; refuse duplicate work')
    job, started = os.environ['PBS_JOBID'], time.monotonic()
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['allocation']['soft_deadline_seconds']
    stopped = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.append(signum))
    state_path = out / 'state.pt'
    payload = torch.load(state_path, map_location='cpu', weights_only=True) if state_path.exists() else None
    if payload and payload['protocol'] != p:
        raise RuntimeError('Resume protocol changed')
    state = payload['state'] if payload else {'stage': 'no_vq_diagnostic', 'joint_updates': 0,
        'history': [], 'completed_queries': {}, 'new_sampler_queries': 0, 'reused_queries': 0, 'lineage': []}
    state['lineage'].append(job)
    bank, optimizer = None, None
    joint_bank_active = False

    def progress(**extra):
        state.update(extra)
        save(out / 'progress.json', {**state, 'protocol': PROTOCOL, 'pbs_jobid': job,
                                     'wall_seconds_this_job': time.monotonic() - started})
        print('JOINT_PROGRESS ' + json.dumps({k: state.get(k) for k in
              ('stage', 'joint_updates', 'completed_queries', 'last_update', 'last_query')}), flush=True)

    def persist():
        value = {'protocol': p, 'state': state, 'rng': capture_rng(torch)}
        if joint_bank_active:
            value.update(module_masters=bank.snapshot(), rotations=bank.rotations.snapshot(),
                         optimizer=to_cpu(optimizer.state_dict()) if optimizer is not None else None)
        elif payload and 'module_masters' in payload:
            # A resumed job may stop while replaying already committed diagnostics.
            value.update({k: payload[k] for k in ('module_masters', 'rotations', 'optimizer')})
            value['rng'] = payload['rng']
        atomic_torch_save(torch, state_path, value)

    def enough():
        if stopped or time.time() + 300 >= deadline:
            persist(); progress(status='checkpointed')
            raise TimeoutError('Committed queries/updates retained for guarded resume')

    progress(status='running'); persist()
    checks = selfcheck(); save(out / 'selfcheck.json', checks)
    plan = read(p['calibration']['plan_file'])
    rows = {r['case_id']: r for r in plan['inputs']}
    if (plan.get('status') != 'complete' or plan['calibration_cohorts']['c8'] != CASES
            or Counter(rows[c]['suite'] for c in CASES) != Counter({s: 2 for s in p['suites']})
            or Counter(rows[c]['suite'] for c in p['evaluation']['case_ids']) != Counter({s: 10 for s in p['suites']})):
        raise RuntimeError('Frozen plan membership differs')
    refroot = parent / 'results/reference'
    receipt = read(refroot / 'reference_receipt.json')
    cache = torch.load(refroot / 'reference_cache.pt', map_location='cpu', weights_only=True)
    if (not receipt.get('complete') or not receipt.get('passed')
            or cache['normalization']['normalization_c'] != p['training']['normalization_c']
            or set(cache['teacher']) != {str(c) for c in CASES}
            or any(r['plan_row'] != rows[r['case_id']] for r in cache['rows'])):
        raise RuntimeError('C8 teacher binding differs')
    prior_protocol, prior_summary = read(previous / 'protocol.json'), read(previous / 'results/summary.json')
    if (not prior_summary.get('complete') or prior_summary.get('rotation_updates') != 8
            or prior_summary.get('vq_updates') != 8
            or prior_summary.get('pbs_jobid') != p['existing_rotation_job']
            or prior_protocol['calibration'] != p['calibration']
            or prior_protocol['training']['latent_weight'] != p['training']['latent_weight']):
        raise RuntimeError('Existing R8/VQ8 binding differs')
    parent_receipt = read(parent / 'results/training/rho1.5/training_receipt.json')
    base_protocol = read(baseline / 'protocol.json')
    base_validation = read(baseline / 'results/baseline/validation.json')
    teacher_receipt = read(baseline / 'results/teacher/receipt.json')
    if (not parent_receipt.get('complete') or parent_receipt.get('updates') != 8
            or parent_receipt.get('latent_weight') != p['training']['latent_weight']
            or parent_receipt.get('binding', {}).get('learning_rate') != .0005
            or base_protocol.get('parent_root') != str(parent) or base_protocol.get('parent_updates') != 8
            or not base_validation.get('complete') or not base_validation.get('recomputed_from_query_records')
            or not teacher_receipt.get('complete') or teacher_receipt.get('case_ids') != p['evaluation']['case_ids']
            or teacher_receipt.get('seed_indices') != [0, 1]):
        raise RuntimeError('H+VQ8/teacher provenance differs')
    saved_rotation = torch.load(previous / 'results/rotation.pt', map_location='cpu', weights_only=True)
    if saved_rotation.get('protocol') != prior_protocol or saved_rotation.get('updates') != 8:
        raise RuntimeError('Saved R8 parameters differ')
    runtime = runner.Runtime(out); runtime.model.eval(); runtime.arm('bf16')
    observations = {}

    def datum(case):
        if case not in observations:
            observation, meta = source.load_observation(np, Path(p['calibration']['plan_file']).parent, rows[case])
            observations[case] = (runtime.datum(observation), meta['description'])
        return observations[case]

    def teacher(case, seed):
        if case in CASES:
            return cache['teacher'][str(case)]
        saved = torch.load(baseline / f'results/teacher/{case}_{seed}.pt', map_location='cpu', weights_only=True)
        if (saved.get('plan_row') != rows[case] or saved.get('seed_index') != seed
                or saved.get('sampler_seed') != rows[case]['sampler_seeds'][seed]
                or saved.get('protocol') != base_protocol['protocol']):
            raise RuntimeError('Heldout teacher identity differs')
        return saved

    def query(case, seed, active, grad=False):
        value, description = datum(case)
        runtime.description, runtime.current_arm = description, 'joint_diagnostic'
        active.seen.clear()
        with torch.set_grad_enabled(grad):
            for stream in active.rotations.raw:
                active.rotations.matrix(stream)
            action, counts, future = sampler.differentiable_action(runtime, value,
                rows[case]['sampler_seeds'][seed], checkpointed=grad, return_future=True)
        if len(active.seen) != 614:
            raise RuntimeError('Sampler did not reach all 614 Linears')
        return action, future, counts

    def loss(action, future, target):
        a = (action[:10, :6].float() - target['action'][:10, :6].to('cuda').float()).square().mean()
        z = latent_reference.future_nmse_loss(torch, future, target['future'].to('cuda'))
        return a + p['training']['latent_weight'] * z, a, z

    def metrics(action, future, target):
        total, a, z = loss(action, future, target)
        d = action.float() - target['action'].to('cuda').float()
        return dict(first10motor_rmse=float(a.sqrt()), first10motor_mse=float(a),
            future_latent_nmse=float(z), mixed_loss=float(total),
            full32motor_rmse=float(d[:, :6].square().mean().sqrt()),
            late22motor_rmse=float(d[10:, :6].square().mean().sqrt()),
            first10gripper_rmse=float(d[:10, 6].square().mean().sqrt()))

    def endpoint(arm, split, case, seed, active=None, reuse=None, source_path=None):
        path = out / split / arm / 'queries' / f'{case}_{seed}.json'
        binding = dict(protocol=PROTOCOL, arm=arm, split=split, case_id=case, seed_index=seed,
                       suite=rows[case]['suite'], sampler_seed=rows[case]['sampler_seeds'][seed])
        if path.exists():
            record = read(path)
            if any(record.get(k) != v for k, v in binding.items()):
                raise RuntimeError('Saved endpoint identity differs')
        else:
            enough(); tick = time.monotonic()
            if reuse is None:
                action, future, counts = query(case, seed, active)
                value = metrics(action, future, teacher(case, seed))
                del action, future
                state['new_sampler_queries'] += 1
            else:
                value = dict(reuse); counts = None
                if 'first10motor_mse' not in value:
                    value['first10motor_mse'] = value['first10motor_rmse'] ** 2
                if 'mixed_loss' not in value:
                    value['mixed_loss'] = value['first10motor_mse'] + p['training']['latent_weight'] * value['future_latent_nmse']
                state['reused_queries'] += 1
            record = {**{k: value[k] for k in METRICS}, **binding, 'counts': counts,
                'source_path': source_path, 'reused': reuse is not None, 'pbs_jobid': job,
                'seconds': time.monotonic() - tick}
            if any(not math.isfinite(record[k]) or record[k] < 0 for k in METRICS):
                raise RuntimeError('Nonfinite endpoint metric')
            save(path, record)
        key = split + '/' + arm
        # Counts come from saved records as well, so resume does not double-count them.
        state['completed_queries'][key] = len(list(path.parent.glob('*.json')))
        progress(last_query={'arm': arm, 'split': split, 'case_id': case, 'seed_index': seed})
        return record

    def evaluate(arm, active, split):
        selected, seeds = (CASES, [0]) if split == 'calibration' else (p['evaluation']['case_ids'], [0, 1])
        return [endpoint(arm, split, c, s, active) for c in selected for s in seeds]

    heldout, calibration = {}, {}
    rb = RotationBanks(runtime, p['calibration']['bank_directory'])
    # These must run before any decoded VQ bank replaces the original BF16 weights.
    rb.rotations.freeze()
    calibration['H_W16A4'] = evaluate('H_W16A4', rb, 'calibration')
    heldout['H_W16A4'] = evaluate('H_W16A4', rb, 'heldout')
    rb.rotations.restore(saved_rotation['raw'])
    calibration['R8_W16A4'] = []
    for case in CASES:
        path = previous / f'results/moments/{case}.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('protocol') != prior_protocol or saved.get('case_id') != case:
            raise RuntimeError('Existing R8-W16A4 C8 metrics differ')
        calibration['R8_W16A4'].append(endpoint('R8_W16A4', 'calibration', case, 0,
                                                reuse=saved['metrics'], source_path=str(path)))
    heldout['R8_W16A4'] = evaluate('R8_W16A4', rb, 'heldout')
    rb.close(); progress(stage='calibration_boundaries'); persist()

    for arm in ('H_VQ8', 'R8_VQ8'):
        heldout[arm], calibration[arm] = [], []
        for case in p['evaluation']['case_ids']:
            for seed in (0, 1):
                path = (baseline / f'results/baseline/queries/{case}_{seed}.json' if arm == 'H_VQ8'
                        else previous / f'results/heldout/queries/{case}_{seed}.json')
                saved = read(path)
                if (saved.get('case_id') != case or saved.get('seed_index') != seed
                        or saved.get('sampler_seed') != rows[case]['sampler_seeds'][seed]
                        or saved.get('suite') != rows[case]['suite']):
                    raise RuntimeError('Reused heldout query identity differs')
                if arm == 'H_VQ8':
                    valid = saved.get('protocol') == base_protocol['protocol'] and saved.get('arm') == 'baseline' and saved.get('updates') == 0
                else:
                    valid = (saved.get('protocol') == prior_protocol['protocol'] and saved.get('arm') == 'R8+VQ8'
                             and saved.get('rotation_updates') == 8 and saved.get('vq_updates') == 8)
                if not valid:
                    raise RuntimeError('Reused heldout checkpoint identity differs')
                heldout[arm].append(endpoint(arm, 'heldout', case, seed, reuse=saved, source_path=str(path)))
    for case in CASES:
        value = cache['parent'][str(case)]
        calibration.setdefault('H_VQ0', []).append(endpoint('H_VQ0', 'calibration', case, 0,
            reuse=metrics(value['action'].to('cuda'), value['future'].to('cuda'), teacher(case, 0)),
            source_path=str(refroot / 'reference_cache.pt')))
        path = previous / f'results/calibration/queries/{case}_0.json'
        saved = read(path)
        if (saved.get('protocol') != prior_protocol['protocol'] or saved.get('case_id') != case
                or saved.get('sampler_seed') != rows[case]['sampler_seeds'][0]):
            raise RuntimeError('Reused R8+VQ8 C8 query differs')
        calibration['R8_VQ8'].append(endpoint('R8_VQ8', 'calibration', case, 0,
                                              reuse=saved, source_path=str(path)))
    bank = JointBanks(runtime, p['calibration']['bank_directory'])
    bank.rotations.freeze()
    deployment = torch.load(parent / 'results/training/rho1.5/deployment.pt', map_location='cpu', weights_only=True)
    if deployment.get('updates') != 8:
        raise RuntimeError('H+VQ8 deployment update count differs')
    bank.restore(deployment['module_masters']); del deployment
    calibration['H_VQ8'] = evaluate('H_VQ8', bank, 'calibration'); bank.close(); del bank; bank = None
    bank = VQBanks(runtime, previous / 'results/banks', rb.rotations, rb.transforms, prior_protocol['protocol'])
    calibration['R8_VQ0'] = evaluate('R8_VQ0', bank, 'calibration')
    for name, mod in bank.modules.items():
        mod.forward = bank.forwards[name]
    del bank, rb; bank = None
    progress(stage='joint_training'); persist()

    bank = JointBanks(runtime, p['calibration']['bank_directory'])
    joint_bank_active = True
    vq, rotations = bank.vq_parameters(), list(bank.rotations.parameters())
    optimizer = torch.optim.Adam([{'params': vq, 'lr': .0005}, {'params': rotations, 'lr': .0001}])
    if payload and 'module_masters' in payload:
        bank.restore(payload['module_masters']); bank.rotations.restore(payload['rotations'])
        if payload.get('optimizer') is not None:
            optimizer.load_state_dict(payload['optimizer']); optimizer_to_device(torch, optimizer, 'cuda')
        restore_rng(torch, payload['rng'])
    elif state['joint_updates']:
        raise RuntimeError('Joint update cursor has no checkpoint parameters')
    state['joint_initialized'] = True
    if not state.get('initial_parity'):
        action, future, counts = query(CASES[0], 0, bank)
        initial = metrics(action, future, teacher(CASES[0], 0))
        expected = calibration['H_VQ0'][0]
        if any(abs(initial[k] - expected[k]) > 2e-6 for k in METRICS):
            raise RuntimeError('Joint0 differs from the exact original H+VQ0')
        state['initial_parity'] = {k: abs(initial[k] - expected[k]) for k in METRICS}
        del action, future
    persist()
    while state['joint_updates'] < 8:
        enough(); index = state['joint_updates']; case = CASES[index]; tick = time.monotonic()
        bank.rotations.refresh(); optimizer.zero_grad(set_to_none=True)
        action, future, counts = query(case, 0, bank, True)
        total, a, z = loss(action, future, teacher(case, 0))
        before = metrics(action.detach(), future.detach(), teacher(case, 0))
        if not future.requires_grad or not torch.isfinite(total):
            raise RuntimeError('Joint loss is disconnected or nonfinite')
        total.backward(); coverage = bank.gradient_report()
        rgrad = {s: bool(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0)
                 for s, v in bank.rotations.raw.items()}
        if coverage['connected'] != 614 or coverage['finite'] != 614 or not rgrad or not all(rgrad.values()):
            raise RuntimeError('Joint gradient coverage failed')
        vnorm = torch.nn.utils.clip_grad_norm_(vq, 1, error_if_nonfinite=True)
        rnorm = torch.nn.utils.clip_grad_norm_(rotations, 1, error_if_nonfinite=True)
        optimizer.step()
        if index == 0:
            for value in vq:
                optimizer.state.pop(value, None)
        bank.refresh(); bank.rotations.refresh(); state['joint_updates'] += 1
        del action, future, total, a, z
        action, future, counts = query(case, 0, bank)
        after = metrics(action, future, teacher(case, 0)); del action, future
        row = dict(update=state['joint_updates'], case_id=case, before=before, after=after,
            vq_gradient_norm=float(vnorm), rotation_gradient_norm=float(rnorm), rotation_gradients=rgrad,
            coverage=coverage, seconds=time.monotonic() - tick, **counts)
        state['history'].append(row)
        save(out / 'training.json', {'protocol': PROTOCOL, 'history': state['history']})
        persist(); progress(last_update=row)
    progress(stage='joint_evaluation')
    for value in bank.parameters():
        value.requires_grad_(False)
    bank.rotations.refresh()
    orthogonality = bank.rotations.orthogonality()
    if max(orthogonality.values()) > 5e-5:
        raise RuntimeError('Joint orthogonality failed')
    atomic_torch_save(torch, out / 'deployment.pt', {'protocol': p, 'updates': 8,
        'module_masters': bank.snapshot(), 'rotations': bank.rotations.snapshot(), 'target_linears': 614})
    action, future, counts = query(CASES[0], 0, bank)
    value, _ = datum(CASES[0]); actual, _ = runtime.infer(value, rows[CASES[0]]['sampler_seeds'][0], measure=False)
    parity = float((action.float().cpu() - actual.float().cpu()).abs().max())
    if parity > 1e-6:
        raise RuntimeError('Final sampler/formal-infer parity failed')
    del action, future, actual
    calibration['Joint8'] = evaluate('Joint8', bank, 'calibration')
    heldout['Joint8'] = evaluate('Joint8', bank, 'heldout')
    movement = {s: float((bank.rotations.matrix(s) - torch.eye(128, device='cuda')).norm())
                for s in bank.rotations.raw}
    provenance = dict(source_rotation_job=p['existing_rotation_job'],
        baseline_validation=base_validation, parent_training_receipt=str(parent / 'results/training/rho1.5/training_receipt.json'),
        orthogonality=orthogonality, residual_frobenius_from_identity=movement,
        initial_parity=state['initial_parity'], final_formal_infer_max_abs=parity,
        new_teacher_queries=0, training_history=state['history'], lineage=state['lineage'],
        timing={'joint_updates_seconds': sum(r['seconds'] for r in state['history']),
                'wall_seconds_this_job': time.monotonic() - started},
        same_vq_updates=True, equal_walltime=False, fixed_final_checkpoint=True)
    progress(stage='aggregate'); write_report(out, p, heldout, calibration, provenance)
    summary = dict(protocol=PROTOCOL, status='complete', complete=True, pbs_jobid=job,
        target_linears=614, joint_updates=8, existing_R_updates=8,
        diagnostic_heldout_queries_per_arm=80, joint_heldout_queries=80,
        calibration_arms=7, calibration_queries_per_arm=8, new_teacher_queries=0,
        all_parameters_frozen=all(not v.requires_grad for v in bank.parameters())
            and all(not v.requires_grad for v in runtime.model.parameters()),
        orthogonality=orthogonality, final_formal_infer_max_abs=parity,
        comparison_file=str(out / 'comparison.json'), lineage=state['lineage'])
    save(out / 'summary.json', summary); progress(stage='complete', complete=True); persist()
    print('JOINT_COMPLETE ' + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', type=Path, required=True)
    p = read(parser.parse_args().protocol)
    try:
        run(p)
    except TimeoutError:
        raise SystemExit(75)
    except Exception as exc:
        save(Path(p['remote_root']) / 'results/failure.json',
             {'protocol': PROTOCOL, 'pbs_jobid': os.environ.get('PBS_JOBID'), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
