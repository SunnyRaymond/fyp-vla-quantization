"""Fixed orthogonal paths from the existing R8 checkpoint; no optimization."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import time

PROTOCOL = 'fastwam-r128-improvements-v1'
C8 = [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070]
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse',
           'mixed_loss', 'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_protocol(p):
    specs = p['lanes']['path']['arms']
    if (p['protocol'] != PROTOCOL or p['target_linears'] != 614
            or p['calibration']['case_ids'] != C8 or p['calibration']['seed_indices'] != [0]
            or p['evaluation']['case_ids'] != list(range(3000, 3040))
            or p['evaluation']['seed_indices'] != [0, 1]
            or [(x['name'], x['t'], x['reused']) for x in specs] != [
                ('H_W16A4', 0, True), ('Rt025_W16A4', .25, False),
                ('Rt050_W16A4', .5, False), ('R8_W16A4', 1, True)]
            or p['training']['rho'] != 1.5
            or p['training']['latent_weight'] != .04245452683169643
            or p['training']['vq_updates'] != 0 or p['quantization']['D'] != 1
            or p['quantization']['clipping_ratio'] != 1 or p['quantization']['native_kernel']):
        raise RuntimeError('Frozen path protocol differs')


def validate_record(record, binding):
    if any(record.get(k) != v for k, v in binding.items()):
        raise RuntimeError('Existing query record binding differs')
    for name in METRICS:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError('Invalid query metric: ' + name)
    return record


def run(p):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import numpy as np
    import runner
    import calibration_source as source
    import latent_reference
    from checkpointing import atomic_torch_save
    from rotation_bank import RotationBanks
    from rotation_selfcheck import run as selfcheck

    validate_protocol(p)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(p['training']['random_seed'])
    root = Path(p['remote_root'])
    out = root / 'results/path'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists() and read(out / 'summary.json').get('complete'):
        raise RuntimeError('Path already complete; refusing duplicate computation')
    job, started = os.environ['PBS_JOBID'], time.monotonic()
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + p['allocation']['soft_deadline_seconds']
    stopped = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.append(signum))
    state_path = out / 'state.pt'
    payload = torch.load(state_path, map_location='cpu', weights_only=True) if state_path.exists() else None
    if payload and payload['protocol'] != p:
        raise RuntimeError('Path resume protocol changed')
    state = payload['state'] if payload else dict(completed_queries={}, arms={}, lineage=[],
        new_student_queries=0, reused_queries=0)
    state['lineage'].append(job)

    def progress(**extra):
        state.update(extra)
        save(out / 'progress.json', dict(state, protocol=PROTOCOL, pbs_jobid=job,
             lane='path', wall_seconds_this_job=time.monotonic() - started))
        atomic_torch_save(torch, state_path, dict(protocol=p, state=state))
        print('PATH_PROGRESS ' + json.dumps({k: state.get(k) for k in
              ('stage', 'completed_queries', 'last_query')}), flush=True)

    def enough():
        if stopped or time.time() + 300 >= deadline:
            progress(status='checkpointed')
            raise TimeoutError('Committed path query records retained for resume')

    progress(stage='source_binding', status='running')
    checks = selfcheck()
    save(out / 'selfcheck.json', checks)
    plan = read(p['calibration']['plan_file'])
    rows = {r['case_id']: r for r in plan['inputs']}
    if (plan.get('status') != 'complete' or plan['calibration_cohorts']['c8'] != C8
            or Counter(rows[c]['suite'] for c in C8) != Counter({s: 2 for s in p['suites']})):
        raise RuntimeError('Frozen plan membership differs')
    prior = Path(p['existing_rotation_root'])
    old_protocol, old_summary = read(prior / 'protocol.json'), read(prior / 'results/summary.json')
    if (old_summary.get('complete') is not True or old_summary.get('pbs_jobid') != p['existing_rotation_job']
            or old_summary.get('rotation_updates') != 8
            or old_protocol['calibration']['case_ids'] != C8
            or old_protocol['training']['latent_weight'] != p['training']['latent_weight']):
        raise RuntimeError('Existing R8 source differs')
    rotation = torch.load(prior / 'results/rotation.pt', map_location='cpu', weights_only=True)
    if rotation.get('protocol') != old_protocol or rotation.get('updates') != 8:
        raise RuntimeError('Saved raw Cayley parameters differ')
    diag = Path(p['diagnostic_root'])
    diag_summary = read(diag / 'results/summary.json')
    if (diag_summary.get('complete') is not True or diag_summary.get('pbs_jobid') != p['diagnostic_job']
            or diag_summary.get('protocol') != p['diagnostic_protocol']):
        raise RuntimeError('Completed H/R endpoint source differs')
    parent = Path(p['parent_root'])
    receipt = read(parent / 'results/reference/reference_receipt.json')
    cache = torch.load(parent / 'results/reference/reference_cache.pt', map_location='cpu', weights_only=True)
    if (not receipt.get('passed') or not receipt.get('complete')
            or set(cache['teacher']) != {str(c) for c in C8}
            or cache['normalization']['normalization_c'] != p['training']['normalization_c']
            or any(row['plan_row'] != rows[row['case_id']] for row in cache['rows'])):
        raise RuntimeError('C8 teacher binding differs')
    baseline = Path(p['baseline_root'])
    baseline_protocol = read(baseline / 'protocol.json')
    tr = read(baseline / 'results/teacher/receipt.json')
    if (not tr.get('complete') or tr.get('case_ids') != p['evaluation']['case_ids']
            or tr.get('seed_indices') != [0, 1]):
        raise RuntimeError('Heldout teacher coverage differs')
    runtime = runner.Runtime(out)
    runtime.model.eval()
    runtime.arm('bf16')
    bank = RotationBanks(runtime, p['calibration']['bank_directory'])
    bank.rotations.freeze()
    observations = {}

    def datum(case):
        if case not in observations:
            observation, meta = source.load_observation(np, Path(p['calibration']['plan_file']).parent, rows[case])
            observations[case] = runtime.datum(observation), meta['description']
        return observations[case]

    def teacher(case, seed):
        if case in C8:
            return cache['teacher'][str(case)]
        saved = torch.load(baseline / f'results/teacher/{case}_{seed}.pt', map_location='cpu', weights_only=True)
        if (saved.get('plan_row') != rows[case] or saved.get('seed_index') != seed
                or saved.get('sampler_seed') != rows[case]['sampler_seeds'][seed]
                or saved.get('protocol') != baseline_protocol['protocol']):
            raise RuntimeError('Heldout teacher identity differs')
        return saved

    def query(case, seed):
        value, description = datum(case)
        runtime.description, runtime.current_arm = description, 'R128_path'
        bank.seen.clear()
        with torch.no_grad():
            for stream in bank.rotations.raw:
                bank.rotations.matrix(stream)
            action, counts, future = sampler.differentiable_action(runtime, value,
                rows[case]['sampler_seeds'][seed], checkpointed=False, return_future=True)
        if len(bank.seen) != 614:
            raise RuntimeError('Path query did not reach 614 target Linears')
        state['new_student_queries'] += 1
        return action, future, counts

    def metrics(action, future, target):
        d = action.float() - target['action'].to('cuda').float()
        a = d[:10, :6].square().mean()
        z = latent_reference.future_nmse_loss(torch, future, target['future'].to('cuda'))
        return dict(first10motor_rmse=float(a.sqrt()), first10motor_mse=float(a),
            future_latent_nmse=float(z), mixed_loss=float(a + p['training']['latent_weight'] * z),
            full32motor_rmse=float(d[:, :6].square().mean().sqrt()),
            late22motor_rmse=float(d[10:, :6].square().mean().sqrt()),
            first10gripper_rmse=float(d[:10, 6].square().mean().sqrt()))

    for spec in p['lanes']['path']['arms']:
        arm, t = spec['name'], spec['t']
        bank.rotations.restore({s: a * t for s, a in rotation['raw'].items()})
        # Scaling raw A scales its skew part and keeps U(t) exactly on the Cayley path.
        progress(stage=arm)
        if arm not in state['arms']:
            enough()
            action, future, _ = query(C8[0], 0)
            value, _ = datum(C8[0])
            with torch.no_grad():
                formal, _ = runtime.infer(value, rows[C8[0]]['sampler_seeds'][0], measure=False)
            state['new_student_queries'] += 1
            parity = float((formal.float().cpu() - action.float().cpu()).abs().max())
            if parity > 1e-6:
                raise RuntimeError('Formal inference differs from the diagnostic sampler')
            if spec['reused']:
                expected = read(diag / f'results/calibration/{arm}/queries/{C8[0]}_0.json')
                actual = metrics(action, future, teacher(C8[0], 0))
                source_delta = max(abs(actual[m] - expected[m]) for m in METRICS)
                if source_delta > 2e-6:
                    raise RuntimeError('Existing H/R endpoint replay changed')
            else:
                source_delta = None
            orth = bank.rotations.orthogonality()
            if max(orth.values()) > 5e-5:
                raise RuntimeError('Orthogonal path check failed')
            frozen = not any(v.requires_grad for v in runtime.model.parameters()) and not any(
                v.requires_grad for v in bank.rotations.parameters())
            if not frozen:
                raise RuntimeError('Frozen path has trainable parameters')
            state['arms'][arm] = dict(updates=0, training_exposures=0, source_rotation_updates=8,
                t=t, reused=spec['reused'], all_parameters_frozen=frozen,
                formal_infer_max_abs=parity, source_replay_metric_max_abs=source_delta,
                orthogonality=orth, movement={s: float((bank.rotations.matrix(s)
                - torch.eye(128, device='cuda')).norm()) for s in bank.rotations.raw})
            del action, future, formal
            progress()
        for split, cases, seeds in [('calibration', C8, [0]),
                ('heldout', p['evaluation']['case_ids'], [0, 1])]:
            count = 0
            for case in cases:
                for seed in seeds:
                    binding = dict(protocol=PROTOCOL, lane='path', arm=arm, split=split,
                        case_id=case, seed_index=seed, suite=rows[case]['suite'],
                        sampler_seed=rows[case]['sampler_seeds'][seed], t=t)
                    path = out / split / arm / 'queries' / f'{case}_{seed}.json'
                    if path.exists():
                        validate_record(read(path), binding)
                    else:
                        enough()
                        if spec['reused']:
                            source_path = diag / f'results/{split}/{arm}/queries/{case}_{seed}.json'
                            old = read(source_path)
                            if any(old.get(k) != binding[k] for k in
                                   ('arm', 'split', 'case_id', 'seed_index', 'suite', 'sampler_seed')):
                                raise RuntimeError('Reused query identity differs')
                            if old.get('protocol') != p['diagnostic_protocol']:
                                raise RuntimeError('Reused query protocol differs')
                            record = dict(binding, **{m: old[m] for m in METRICS}, source_path=str(source_path))
                            state['reused_queries'] += 1
                        else:
                            action, future, counts = query(case, seed)
                            record = dict(binding, **metrics(action, future, teacher(case, seed)), counts=counts)
                            del action, future
                        validate_record(record, binding)
                        save(path, record)
                    count += 1
                    state['completed_queries'][split + '/' + arm] = count
                    progress(last_query=dict(arm=arm, split=split, case_id=case, seed_index=seed))
            state['arms'][arm][split + '_queries'] = count
    bank.close()
    save(out / 'training.json', dict(protocol=PROTOCOL, lane='path', updates=0, history=[]))
    from aggregate import write_lane_report
    write_lane_report(out, p, 'path')
    progress(stage='complete', status='complete')
    summary = dict(protocol=PROTOCOL, status='complete', complete=True, pbs_jobid=job,
        lane='path', target_linears=614, arms=state['arms'], new_teacher_queries=0,
        lineage=state['lineage'], new_student_queries=state['new_student_queries'],
        reused_queries=state['reused_queries'], wall_seconds_this_job=time.monotonic() - started)
    save(out / 'summary.json', summary)
    print('PATH_COMPLETE ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--lane', choices=['path'], default='path')
    args = parser.parse_args()
    run(read(args.protocol))
