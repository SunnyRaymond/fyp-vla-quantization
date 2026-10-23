"""Frozen original-C80 BF16 teacher sampling; no optimizer or student model."""
import argparse
import json
import os
from pathlib import Path
import time

from common import guard, load_protocol, packet_valid, save, seeds


def atomic_tensor(torch, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def prepare(p):
    import torch
    root = Path(p['remote_root'])
    plan = json.loads(Path(p['calibration']['plan_file']).read_text())
    frozen = json.loads((root / 'source_plan.json').read_text())
    if plan != frozen or plan['status'] != 'complete' or plan['calibration_cohorts']['c80'] != p['calibration']['case_ids']:
        raise RuntimeError('Frozen original C80 input plan changed')
    rows = {r['case_id']: r for r in plan['inputs']}
    manifest = []
    for case in p['calibration']['case_ids']:
        row = rows[case]
        source = Path(p['cached_teacher_root']) / f'{case}_0.pt'
        d = torch.load(source, map_location='cpu', weights_only=True)
        if d['plan_row'] != row or d['sampler_seed'] != row['sampler_seeds'][0]:
            raise RuntimeError('Cached BF16 teacher input or seed differs')
        a = d['action'].detach().float().cpu()
        if tuple(a.shape) != (32, 7) or not bool(torch.isfinite(a).all()):
            raise RuntimeError('Invalid cached action')
        value = dict(protocol=p['protocol'], case_id=case, suite=row['suite'], seed_index=0,
                     sampler_seed=seeds(row)[0], plan_row=row, action=a,
                     counts={'video_steps': 10, 'action_steps': 10}, source_kind='reused_original_teacher',
                     collection_jobid=os.environ['PBS_JOBID'], source_path=str(source))
        output = root / 'results/teacher' / f'{case}_0.pt'
        if output.exists():
            old = torch.load(output, map_location='cpu', weights_only=True)
            if not packet_valid(old, p, row, 0) or not torch.equal(old['action'], a):
                raise RuntimeError('Existing reused teacher packet differs')
        else:
            atomic_tensor(torch, output, value)
        manifest.append(dict(case_id=case, suite=row['suite'], plan_row=row, sampler_seeds=seeds(row)))
    save(root / 'results/prepare/manifest.json', {'protocol': p['protocol'], 'cases': manifest})
    save(root / 'results/prepare/summary.json', dict(protocol=p['protocol'], complete=True,
         pbs_jobid=os.environ['PBS_JOBID'], cases=80, reused_teacher_queries=80, new_teacher_queries=0,
         training_updates=0, optimizer_steps=0, heldout_used=False, source_plan_equal=True))


def collect(p):
    import numpy as np
    import torch
    import calibration_source
    import runner
    import sampler
    root = Path(p['remote_root'])
    manifest = json.loads((root / 'results/prepare/manifest.json').read_text())['cases']
    if [r['case_id'] for r in manifest] != p['calibration']['case_ids']:
        raise RuntimeError('Original C80 changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261021)
    out = root / 'results/collect'
    out.mkdir(parents=True, exist_ok=True)
    state = dict(protocol=p['protocol'], complete=False, pbs_jobid=os.environ['PBS_JOBID'],
                 phase='loading', completed_packets=80, new_teacher_queries=0, reused_teacher_queries=80,
                 parity_queries=0, training_updates=0, optimizer_steps=0, heldout_used=False)
    start = time.time()
    deadline = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ.get('FW_SOFT_DEADLINE_SECONDS', '6840'))
    runtime = None
    cached_input = {}
    replay = {'protocol': p['protocol'], 'complete': False, 'cached_replays': [], 'same_seed_repeats': []}

    def progress(**kw):
        state.update(kw)
        state['elapsed_seconds'] = time.time() - start
        save(out / 'progress.json', state)
        print('TEACHER_PROGRESS ' + json.dumps(state), flush=True)

    def query(row, seed):
        if time.time() + 180 >= deadline:
            raise TimeoutError('Soft deadline; completed teacher packets are reusable')
        if cached_input.get('case_id') != row['case_id']:
            obs, meta = calibration_source.load_observation(np, Path(p['calibration']['plan_file']).parent, row)
            cached_input.update(case_id=row['case_id'], datum=runtime.datum(obs), description=meta['description'])
        runtime.description = cached_input['description']
        with torch.no_grad():
            a, counts, z = sampler.differentiable_action(runtime, cached_input['datum'], seed,
                                                       checkpointed=False, return_future=True)
        torch.cuda.synchronize()
        if counts != {'video_steps': 10, 'action_steps': 10}:
            raise RuntimeError('Original 10+10 sampler changed')
        return a.detach().float().cpu(), z.detach().float().cpu(), counts

    try:
        progress()
        runtime = runner.Runtime(out)
        runtime.arm('bf16')
        runtime.model.eval()
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)
        parity_cases = set(p['replay_cases'])
        for entry in manifest:
            row, case = entry['plan_row'], entry['case_id']
            if case in parity_cases:
                a, z, _ = query(row, seeds(row)[0])
                old = torch.load(Path(p['cached_teacher_root']) / f'{case}_0.pt', map_location='cpu', weights_only=True)
                item = dict(case_id=case, action_max_abs=float((a - old['action'].float()).abs().max()),
                            future_max_abs=float((z - old['future'].float()).abs().max()))
                state['parity_queries'] += 1
                replay['cached_replays'].append(item)
                if item['action_max_abs'] != 0 or item['future_max_abs'] != 0:
                    raise RuntimeError('Original BF16 teacher replay is not exact: ' + str(item))
            for k in range(1, 16):
                output = root / 'results/teacher' / f'{case}_{k}.pt'
                if output.exists():
                    d = torch.load(output, map_location='cpu', weights_only=True)
                    if not packet_valid(d, p, row, k) or tuple(d['action'].shape) != (32, 7) or not bool(torch.isfinite(d['action']).all()):
                        raise RuntimeError('Existing packet binding differs')
                else:
                    a, _, counts = query(row, seeds(row)[k])
                    state['new_teacher_queries'] += 1
                    atomic_tensor(torch, output, dict(protocol=p['protocol'], case_id=case, suite=row['suite'],
                         seed_index=k, sampler_seed=seeds(row)[k], plan_row=row, action=a, counts=counts,
                         source_kind='new_bf16_teacher', collection_jobid=os.environ['PBS_JOBID']))
                state['completed_packets'] += 1
                progress(phase='sampling', case_id=case, seed_index=k)
            if case in parity_cases:
                a, _, _ = query(row, seeds(row)[15])
                old = torch.load(root / 'results/teacher' / f'{case}_15.pt', map_location='cpu', weights_only=True)
                item = dict(case_id=case, seed_index=15, action_max_abs=float((a - old['action'].float()).abs().max()))
                state['parity_queries'] += 1
                replay['same_seed_repeats'].append(item)
                if item['action_max_abs'] != 0:
                    raise RuntimeError('Same-seed teacher is not deterministic')
        expected = {(c, k) for c in p['calibration']['case_ids'] for k in range(16)}
        observed = set()
        for entry in manifest:
            for k in range(16):
                d = torch.load(root / 'results/teacher' / f"{entry['case_id']}_{k}.pt", map_location='cpu', weights_only=True)
                if not packet_valid(d, p, entry['plan_row'], k):
                    raise RuntimeError('Final packet binding differs')
                observed.add((d['case_id'], k))
        if observed != expected or len(replay['cached_replays']) != 8 or len(replay['same_seed_repeats']) != 8:
            raise RuntimeError('Complete original C80 x16 plus replay coverage required')
        if runtime.weight_converted or runtime.current_arm != 'bf16':
            raise RuntimeError('Teacher must remain original BF16')
        replay['complete'] = True
        save(root / 'results/replay.json', replay)
        progress(phase='complete', complete=True, cases=80, seeds_per_case=16, total_teacher_packets=1280,
                 model_parameters_require_grad=False, original_bf16=True, stochastic_scope='joint video+action sampler noise')
        save(out / 'summary.json', state)
    except BaseException as exc:
        progress(phase='failed', error=repr(exc))
        save(root / 'results/replay.json', replay)
        raise


if __name__ == '__main__':
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=('prepare', 'collect'), required=True)
    args = parser.parse_args()
    globals()[args.unit](load_protocol())
