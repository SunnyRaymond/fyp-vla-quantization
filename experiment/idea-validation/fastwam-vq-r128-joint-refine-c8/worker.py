"""Reuse completed diagnostics; compare a frozen-R control and smaller-LR Joint8."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

PROTOCOL = 'fastwam-vq-r128-joint-refine-c8-v1'
CASES = [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070]
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'mixed_loss',
           'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse')
REUSED = ('H_W16A4', 'R8_W16A4', 'H_VQ8', 'R8_VQ8', 'Joint8_R1e4', 'H_VQ0', 'R8_VQ0')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def validate_protocol(p):
    if (p['protocol'] != PROTOCOL or p['calibration']['case_ids'] != CASES
            or p['evaluation']['case_ids'] != list(range(3000, 3040))
            or p['evaluation']['seed_indices'] != [0, 1] or p['target_linears'] != 614
            or p['training']['joint_updates'] != 8 or p['training']['control_updates'] != 8
            or p['training']['rotation_learning_rate'] != .00001
            or p['training']['vq_learning_rate'] != .0005 or p['training']['update_order'] != CASES
            or p['training']['rho'] != 1.5 or p['quantization']['D'] != 1
            or p['quantization']['clipping_ratio'] != 1 or p['quantization']['native_kernel']
            or p['training']['latent_weight'] != .04245452683169643):
        raise RuntimeError('Frozen refinement scope differs')


def run(p):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import numpy as np
    import runner
    import calibration_source as source
    import latent_reference
    from joint_bank import JointBanks
    from checkpointing import atomic_torch_save, capture_rng, restore_rng, to_cpu, optimizer_to_device
    from selfcheck import run as selfcheck
    from aggregate import write_report

    validate_protocol(p)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(p['training']['random_seed'])
    root, prior = Path(p['remote_root']), Path(p['stage_source_root'])
    out = root / 'results'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists() and read(out / 'summary.json').get('complete'):
        raise RuntimeError('Already complete; refuse duplicate work')
    job, started = os.environ['PBS_JOBID'], time.monotonic()
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['allocation']['soft_deadline_seconds']
    stopped = []; signal.signal(signal.SIGTERM, lambda signum, frame: stopped.append(signum))
    state_path = out / 'state.pt'
    payload = torch.load(state_path, map_location='cpu', weights_only=True) if state_path.exists() else None
    if payload and payload['protocol'] != p:
        raise RuntimeError('Resume protocol differs')
    state = payload['state'] if payload else dict(stage='reuse_completed_diagnostics', joint_updates=0,
        control_updates=0, history=[], control_history=[], completed_queries={}, arm_updates={},
        frozen={}, parities={}, movements={}, orthogonality={}, lineage=[])
    state['lineage'].append(job)
    snapshots = payload['arms'] if payload else {}
    bank = optimizer = active_arm = None

    def progress(**extra):
        state.update(extra)
        save(out / 'progress.json', {**state, 'protocol': PROTOCOL, 'pbs_jobid': job,
                                    'wall_seconds_this_job': time.monotonic() - started})
        print('REFINE_PROGRESS ' + json.dumps({k: state.get(k) for k in
              ('stage', 'joint_updates', 'control_updates', 'completed_queries', 'last_update', 'last_query')}), flush=True)

    def persist():
        if bank is not None:
            snapshots[active_arm] = dict(module_masters=bank.snapshot(), rotations=bank.rotations.snapshot(),
                optimizer=to_cpu(optimizer.state_dict()) if optimizer is not None else None, rng=capture_rng(torch))
        atomic_torch_save(torch, state_path, dict(protocol=p, state=state, arms=snapshots))

    def enough():
        if stopped or time.time() + 300 >= deadline:
            persist(); progress(status='checkpointed'); raise TimeoutError('Saved units retained for resume')

    progress(status='running'); persist()
    save(out / 'selfcheck.json', selfcheck())
    previous_p = read(prior / 'protocol.json')
    previous_summary = read(prior / 'results/summary.json')
    original = read(prior / 'results/comparison.json')
    if (previous_p['protocol'] != p['stage_source_protocol'] or not previous_summary.get('complete')
            or previous_summary.get('pbs_jobid') != p['stage_source_job']
            or not original.get('complete') or previous_p['calibration'] != p['calibration']
            or previous_p['training']['latent_weight'] != p['training']['latent_weight']
            or previous_p['training']['rotation_learning_rate'] != .0001):
        raise RuntimeError('Completed source diagnostic binding differs')
    hcal = original['calibration']['H_VQ8']['overall']['mixed_loss']
    jcal = original['calibration']['Joint8']['overall']['mixed_loss']
    if jcal < hcal:
        raise RuntimeError('Calibration-only refinement condition did not pass')
    save(out / 'refinement_gate.json', dict(p['refinement_gate'], observed_H_VQ8=hcal,
                                          observed_Joint8_R1e4=jcal, triggered=True))
    plan = read(p['calibration']['plan_file']); rows = {r['case_id']: r for r in plan['inputs']}
    parent, baseline = Path(p['parent_root']), Path(p['baseline_root'])
    cache = torch.load(parent / 'results/reference/reference_cache.pt', map_location='cpu', weights_only=True)
    if (plan['calibration_cohorts']['c8'] != CASES or cache['normalization']['normalization_c'] != p['training']['normalization_c']
            or any(r['plan_row'] != rows[r['case_id']] for r in cache['rows'])):
        raise RuntimeError('Frozen input/teacher binding differs')
    base_protocol = read(baseline / 'protocol.json')
    runtime = runner.Runtime(out); runtime.model.eval(); runtime.arm('bf16')
    data = {}

    def datum(case):
        if case not in data:
            observation, meta = source.load_observation(np, Path(p['calibration']['plan_file']).parent, rows[case])
            data[case] = runtime.datum(observation), meta['description']
        return data[case]

    def teacher(case, seed):
        if case in CASES:
            return cache['teacher'][str(case)]
        saved = torch.load(baseline / f'results/teacher/{case}_{seed}.pt', map_location='cpu', weights_only=True)
        if (saved.get('plan_row') != rows[case] or saved.get('seed_index') != seed
                or saved.get('sampler_seed') != rows[case]['sampler_seeds'][seed]
                or saved.get('protocol') != base_protocol['protocol']):
            raise RuntimeError('Heldout teacher binding differs')
        return saved

    def query(case, seed, grad=False):
        value, description = datum(case); runtime.description, runtime.current_arm = description, active_arm
        bank.seen.clear()
        with torch.set_grad_enabled(grad):
            for stream in bank.rotations.raw:
                bank.rotations.matrix(stream)
            action, counts, future = sampler.differentiable_action(runtime, value,
                rows[case]['sampler_seeds'][seed], checkpointed=grad, return_future=True)
        if len(bank.seen) != 614:
            raise RuntimeError('Incomplete 614-Linear sampler coverage')
        return action, future, counts

    def loss(action, future, target):
        a = (action[:10, :6].float() - target['action'][:10, :6].to('cuda').float()).square().mean()
        z = latent_reference.future_nmse_loss(torch, future, target['future'].to('cuda'))
        return a + p['training']['latent_weight'] * z, a, z

    def metrics(action, future, target):
        total, a, z = loss(action, future, target)
        d = action.float() - target['action'].to('cuda').float()
        return dict(first10motor_rmse=float(a.sqrt()), first10motor_mse=float(a), future_latent_nmse=float(z),
            mixed_loss=float(total), full32motor_rmse=float(d[:, :6].square().mean().sqrt()),
            late22motor_rmse=float(d[10:, :6].square().mean().sqrt()),
            first10gripper_rmse=float(d[:10, 6].square().mean().sqrt()))

    def endpoint(arm, split, case, seed, reused=None, source_path=None):
        path = out / split / arm / 'queries' / f'{case}_{seed}.json'
        binding = dict(protocol=PROTOCOL, arm=arm, split=split, case_id=case, seed_index=seed,
                       suite=rows[case]['suite'], sampler_seed=rows[case]['sampler_seeds'][seed])
        if path.exists():
            record = read(path)
            if any(record.get(k) != v for k, v in binding.items()):
                raise RuntimeError('Saved query binding differs')
        else:
            enough(); tick = time.monotonic()
            if reused is None:
                action, future, counts = query(case, seed)
                values = metrics(action, future, teacher(case, seed)); del action, future
            else:
                values, counts = reused, None
            record = {**{k: values[k] for k in METRICS}, **binding, 'reused': reused is not None,
                'source_path': source_path, 'counts': counts, 'seconds': time.monotonic() - tick, 'pbs_jobid': job}
            if any(not math.isfinite(record[k]) or record[k] < 0 for k in METRICS):
                raise RuntimeError('Nonfinite endpoint')
            save(path, record)
        state['completed_queries'][split + '/' + arm] = len(list(path.parent.glob('*.json')))
        progress(last_query={'arm': arm, 'split': split, 'case_id': case, 'seed_index': seed})
        return record

    heldout, calibration = {}, {}
    for arm in REUSED:
        source_arm = 'Joint8' if arm == 'Joint8_R1e4' else arm
        for split, selected, seeds in (('calibration', CASES, [0]),
                ('heldout', p['evaluation']['case_ids'], [0, 1])):
            if split == 'heldout' and arm in ('H_VQ0', 'R8_VQ0'):
                continue
            records = []
            for case in selected:
                for seed in seeds:
                    path = prior / 'results' / split / source_arm / 'queries' / f'{case}_{seed}.json'
                    saved = read(path)
                    expected = dict(protocol=p['stage_source_protocol'], arm=source_arm, split=split,
                        case_id=case, seed_index=seed, suite=rows[case]['suite'], sampler_seed=rows[case]['sampler_seeds'][seed])
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Reused diagnostic query differs')
                    records.append(endpoint(arm, split, case, seed, saved, str(path)))
            (calibration if split == 'calibration' else heldout)[arm] = records
    progress(stage='H_currentpath_training_evaluation'); persist()
    for active_arm in ('H_currentpath', 'Joint8'):
        joint = active_arm == 'Joint8'
        bank = JointBanks(runtime, p['calibration']['bank_directory'])
        if not joint:
            bank.rotations.freeze()
        vq = bank.vq_parameters()
        rotations = [v for v in bank.rotations.parameters() if v.requires_grad]
        groups = [{'params': vq, 'lr': p['training']['vq_learning_rate']}]
        if joint:
            groups.append({'params': rotations, 'lr': p['training']['rotation_learning_rate']})
        optimizer = torch.optim.Adam(groups)
        if active_arm in snapshots:
            saved = snapshots[active_arm]
            bank.restore(saved['module_masters']); bank.rotations.restore(saved['rotations'])
            if saved.get('optimizer') is not None:
                optimizer.load_state_dict(saved['optimizer']); optimizer_to_device(torch, optimizer, 'cuda')
            restore_rng(torch, saved['rng'])
        updates = state['arm_updates'].get(active_arm, 0)
        if updates and active_arm not in snapshots:
            raise RuntimeError('Training cursor has no parameter checkpoint')
        history_key = 'history' if joint else 'control_history'
        if not updates:
            action, future, counts = query(CASES[0], 0)
            initial = metrics(action, future, teacher(CASES[0], 0)); del action, future
            if any(abs(initial[k] - calibration['H_VQ0'][0][k]) > 2e-6 for k in METRICS):
                raise RuntimeError('Learner0 differs from exact H+VQ0')
        persist()
        while updates < 8:
            enough(); case = CASES[updates]; tick = time.monotonic()
            bank.rotations.refresh(); optimizer.zero_grad(set_to_none=True)
            action, future, counts = query(case, 0, True)
            total, a, z = loss(action, future, teacher(case, 0))
            before = metrics(action.detach(), future.detach(), teacher(case, 0))
            if not future.requires_grad or not torch.isfinite(total):
                raise RuntimeError('Disconnected/nonfinite training loss')
            total.backward(); coverage = bank.gradient_report()
            rgrad = {s: bool(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0)
                     for s, v in bank.rotations.raw.items() if v.requires_grad}
            if (coverage['connected'] != 614 or coverage['finite'] != 614
                    or joint and (not rgrad or not all(rgrad.values()))):
                raise RuntimeError('Gradient coverage failed')
            vnorm = torch.nn.utils.clip_grad_norm_(vq, 1, error_if_nonfinite=True)
            rnorm = torch.nn.utils.clip_grad_norm_(rotations, 1, error_if_nonfinite=True) if joint else 0.
            optimizer.step()
            if updates == 0:
                for parameter in vq:
                    optimizer.state.pop(parameter, None)
            bank.refresh(); bank.rotations.refresh(); updates += 1
            state['arm_updates'][active_arm] = updates
            state['joint_updates' if joint else 'control_updates'] = updates
            del action, future, total, a, z
            action, future, counts = query(case, 0)
            after = metrics(action, future, teacher(case, 0)); del action, future
            row = dict(update=updates, case_id=case, before=before, after=after, coverage=coverage,
                rotation_gradients=rgrad, vq_gradient_norm=float(vnorm), rotation_gradient_norm=float(rnorm),
                seconds=time.monotonic() - tick, **counts)
            state[history_key].append(row)
            save(out / 'training.json', dict(protocol=PROTOCOL, history=state['history'], control_history=state['control_history']))
            persist(); progress(last_update=row)
        for value in bank.parameters():
            value.requires_grad_(False)
        bank.rotations.refresh()
        orth = bank.rotations.orthogonality()
        movement = {s: float((bank.rotations.matrix(s) - torch.eye(128, device='cuda')).norm()) for s in bank.rotations.raw}
        if max(orth.values()) > 5e-5 or not joint and any(value != 0 for value in movement.values()):
            raise RuntimeError('Orthogonality/frozen-control rotation failed')
        state['orthogonality'][active_arm], state['movements'][active_arm] = orth, movement
        state['frozen'][active_arm] = all(not value.requires_grad for value in bank.parameters())
        action, future, counts = query(CASES[0], 0)
        value, _ = datum(CASES[0]); actual, _ = runtime.infer(value, rows[CASES[0]]['sampler_seeds'][0], measure=False)
        parity = float((action.float().cpu() - actual.float().cpu()).abs().max())
        if parity > 1e-6:
            raise RuntimeError('Formal-infer parity failed')
        state['parities'][active_arm] = parity; del action, future, actual
        for split, selected, seeds in (('calibration', CASES, [0]),
                ('heldout', p['evaluation']['case_ids'], [0, 1])):
            records = [endpoint(active_arm, split, c, s) for c in selected for s in seeds]
            (calibration if split == 'calibration' else heldout)[active_arm] = records
        atomic_torch_save(torch, out / (active_arm + '_deployment.pt'), dict(protocol=p,
            arm=active_arm, updates=8, module_masters=bank.snapshot(), rotations=bank.rotations.snapshot()))
        optimizer = None; persist(); bank.close(); del bank; bank = None
        progress(stage='Joint8_training_evaluation' if not joint else 'aggregate')
        persist()
    provenance = dict(source_job=p['stage_source_job'], reused_diagnostic_protocol=p['stage_source_protocol'],
        refinement_gate=p['refinement_gate'], original_incumbent='H_VQ8',
        matched_implementation_control='H_currentpath', parities=state['parities'], orthogonality=state['orthogonality'],
        residual_frobenius_from_identity=state['movements'], training_history=state['history'],
        control_training_history=state['control_history'], new_teacher_queries=0,
        timing=dict(joint_updates_seconds=sum(r['seconds'] for r in state['history']),
            control_updates_seconds=sum(r['seconds'] for r in state['control_history']),
            wall_seconds_this_job=time.monotonic() - started), lineage=state['lineage'],
        equal_walltime=False, fixed_final_checkpoint=True)
    write_report(out, p, heldout, calibration, provenance)
    summary = dict(protocol=PROTOCOL, status='complete', complete=True, pbs_jobid=job, target_linears=614,
        existing_R_updates=8, joint_updates=8, control_updates=8, diagnostic_heldout_queries_per_arm=80,
        joint_heldout_queries=80, calibration_arms=9, calibration_queries_per_arm=8, new_teacher_queries=0,
        all_parameters_frozen=all(state['frozen'].values()) and all(not v.requires_grad for v in runtime.model.parameters()),
        final_formal_infer_max_abs=max(state['parities'].values()), comparison_file=str(out / 'comparison.json'),
        orthogonality=state['orthogonality'], lineage=state['lineage'])
    save(out / 'summary.json', summary); progress(stage='complete', complete=True); persist()
    print('REFINE_COMPLETE ' + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', type=Path, required=True)
    p = read(parser.parse_args().protocol)
    try:
        run(p)
    except TimeoutError:
        raise SystemExit(75)
    except Exception as exc:
        save(Path(p['remote_root']) / 'results/failure.json',
             dict(protocol=PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
