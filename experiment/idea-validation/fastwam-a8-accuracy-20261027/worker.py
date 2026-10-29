"""Phase 1a fixed-input check on frozen Q0: A16 / A8 / H-only INT8-row weights with A8.

Q0_A4 is not rerun: analysis reuses the Q-Clip job's 80 Q0 heldout endpoints.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

PROTOCOL = 'fastwam-a8-accuracy-20261027-v2'
ARMS = ('Q0_A16', 'Q0_A8', 'Q0_H_INT8_A8')
METRICS = ('first10motor_rmse', 'first10motor_mse', 'future_latent_nmse', 'mixed_az_loss',
           'full32motor_rmse', 'late22motor_rmse', 'first10gripper_rmse', 'first10gripper_disagreement')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def aggregate(records, cases, seeds, suites):
    keys = [(r['case_id'], r['seed_index']) for r in records]
    if Counter(keys) != Counter((c, s) for c in cases for s in seeds):
        raise ValueError('Complete paired endpoint coverage required')
    result = {}
    for suite in suites:
        rows = [r for r in records if r['suite'] == suite]
        ids = sorted({r['case_id'] for r in rows})
        if len(ids) != len(cases) // 4:
            raise ValueError('Equal four-suite context coverage required')
        result[suite] = {m: sum(sum(r[m] for r in rows if r['case_id'] == c) / len(seeds)
                                for c in ids) / len(ids) for m in METRICS}
        result[suite].update(contexts=len(ids), queries=len(rows))
    return dict(queries=len(records), contexts=len(cases), by_suite=result,
                overall={m: sum(result[s][m] for s in suites) / 4 for m in METRICS})


def run(cfg, index):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import torch.nn.functional as F
    import batched_transform
    import q0_worker as parent
    from rotation_bank import stream_for, transformed

    arms = cfg['workers'][str(index)]
    root = Path(cfg['remote_root'])
    out = root / 'results'
    unit = out / f'worker_{index}'
    (unit / 'failure.json').unlink(missing_ok=True)  # stale record from an earlier attempt
    started = time.monotonic()

    class Work(parent.Work):
        def progress(self, phase, **extra):
            # Parent __init__ calls this before any workload; redirect all outputs here.
            self.root, self.out, self.unit = root, out, unit
            unit.mkdir(parents=True, exist_ok=True)
            value = dict(protocol=PROTOCOL, worker=index, pbs_jobid=self.job, phase=phase,
                         seconds=time.monotonic() - started, **extra)
            save(unit / 'progress.json', value)
            print('A8ACC_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

        def count(self, key):
            self.counters[key] += 1

        def enough(self):
            if self.stopped or time.time() + 300 >= self.deadline:
                raise TimeoutError('Committed endpoint records retained for resume')

    parent_p = read(Path(__file__).with_name('parent_protocol.json'))
    work = Work(parent_p, 'reader_kl')
    work.deadline = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ['FW_SOFT_DEADLINE_SECONDS'])
    work.counters = dict(student_queries=0, formal_infer_queries=0, reader_queries=0,
                         parameter_reverse_passes=0, new_teacher_queries=0)
    ev = cfg['evaluation']
    receipt = read(work.heldout_teacher_root / 'receipt.json')
    if not receipt.get('complete') or receipt['case_ids'] != ev['case_ids'] or receipt['seed_indices'] != [0, 1]:
        raise ValueError('Cached heldout teachers differ')
    source_summary = read(cfg['source_summary'])
    if (not source_summary.get('complete') or source_summary.get('pbs_jobid') != cfg['source_job']
            or source_summary.get('arm') != cfg['source_arm'] or source_summary.get('updates') != 8):
        raise ValueError('Final Q0 checkpoint is not complete')

    work.progress('loading_q0')
    work.load_model()
    bank = work.bank
    source = torch.load(cfg['source_checkpoint'], map_location='cpu', weights_only=True)
    if (source['protocol'] != parent_p or source['arm'] != cfg['source_arm'] or source['updates'] != 8
            or source['target_linears'] != 614
            or any(not torch.equal(source['rotations'][s], v) for s, v in work.initial_rotation.items())):
        raise ValueError('Q0 deployment binding differs')
    bank.restore(source['module_masters'])
    for v in bank.parameters():
        v.requires_grad_(False)
        v.grad = None
    q0_forwards = {n: m.forward for n, m in bank.modules.items()}

    def symmetric(x, qmax):
        x = x.float()
        scale = x.abs().amax(-1, keepdim=True) / qmax
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(x / safe).clamp(-qmax, qmax) * scale).to(torch.bfloat16)

    def right_blocks(value, matrix):
        full = value.shape[-1] // 128 * 128
        if not full:
            return value.float()
        head = value[..., :full].float().reshape(*value.shape[:-1], full // 128, 128)
        head = (head @ matrix).reshape(*value.shape[:-1], full)
        return torch.cat((head, value[..., full:].float()), -1) if full < value.shape[-1] else head

    def no_u(name, weight):
        # x_R = x_H U, so x_R W_R^T = x_H (W_R U^T)^T; only full 128-wide blocks carry U.
        if bank.modules[name].in_features < 128:
            return weight.float()
        return right_blocks(weight, bank.rotations.matrix(stream_for(name)).t())

    def bf16_weight(name):
        return transformed(work.masters[name].to('cuda'), bank.transforms[name], bank.rotations,
                           stream_for(name), weight=True).to(torch.bfloat16)

    @torch.no_grad()
    def set_weights(kind):
        bank.refresh()  # exact Q0 decode from the restored books/gains
        if kind == 'q0':
            return
        for name, mod in bank.modules.items():
            if kind == 'q0_int8row':
                mod.weight.data = symmetric(mod.weight.data, 127)
            elif kind == 'q0_noU_int8row':
                mod.weight.data = symmetric(no_u(name, mod.weight.data), 127)
            else:
                raise ValueError('Unknown weight kind: ' + kind)

    def install(arm):
        if arm == 'Q0_A4':  # unchanged Q0 forward; only used for the source anchor (Q0_A4 results are reused)
            for name, mod in bank.modules.items():
                mod.forward = q0_forwards[name]
            return
        spec = cfg['arms'][arm]
        act, rotated = spec['activation'], spec['coordinates'] == 'R0'
        for name, mod in bank.modules.items():

            def forward(x, _name=name, _mod=mod):
                bank.seen.add(_name)
                t = bank.transforms[_name]
                z = (transformed(x, t, bank.rotations, stream_for(_name)) if rotated
                     else batched_transform.transform_input(x, t))
                qx = z.to(torch.bfloat16) if act == 'bf16' else symmetric(z, 127)
                return F.linear(qx, _mod.weight, _mod.bias)
            mod.forward = forward

    @torch.no_grad()
    def transform_check():
        # Catch a wrong transpose/pairing before spending queries: both rewrites must reproduce
        # the reference Linear on a random input up to BF16 rounding.
        name = next(n for n, m in bank.modules.items() if stream_for(n) == 'video' and m.in_features >= 256)
        mod, t = bank.modules[name], bank.transforms[name]
        generator = torch.Generator(device='cuda').manual_seed(20261027)
        x = torch.randn(8, mod.in_features, device='cuda', generator=generator).to(torch.bfloat16)
        rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
        z_r = transformed(x, t, bank.rotations, stream_for(name)).to(torch.bfloat16)
        z_h = batched_transform.transform_input(x, t).to(torch.bfloat16)
        reference = F.linear(x, work.masters[name].to('cuda'))
        q0_r = F.linear(z_r, mod.weight)
        values = dict(module=name,
                      bf16_pairing_vs_original=rel(F.linear(z_r, bf16_weight(name)), reference),
                      noU_vs_R0_q0=rel(F.linear(z_h, no_u(name, mod.weight).to(torch.bfloat16)), q0_r),
                      q0_vs_original=rel(q0_r, reference))
        limit = cfg['gates']['transform_check_max_relative']
        if values['bf16_pairing_vs_original'] > limit or values['noU_vs_R0_q0'] > limit:
            raise ValueError('Weight/activation coordinate rewrite check failed: ' + json.dumps(values))
        return values

    def metrics(action, future, target):
        values = work.metrics(action, future, target)
        values['mixed_az_loss'] = values.pop('mixed_loss')
        actual = work.runtime.command(action.detach().clone())
        teacher = work.runtime.command(target['action'].detach().clone())
        values['first10gripper_disagreement'] = float((actual[:10, 6] != teacher[:10, 6]).mean())
        return values

    # Source binding: the unchanged Q0 forward must replay the saved case-2000 anchor.
    install('Q0_A4')
    anchor = torch.load(cfg['source_anchor'], map_location='cpu', weights_only=True)
    a, z, _ = work.query(2000)
    parity = max(float((a.float().cpu() - anchor['action'].float()).abs().max()),
                 float((z.float().cpu() - anchor['future'].float()).abs().max()))
    if parity > cfg['gates']['anchor_max_abs']:
        raise ValueError(f'Q0 source anchor parity failed: {parity}')
    del a, z
    check = transform_check()
    save(unit / 'startup.json', dict(protocol=PROTOCOL, worker=index, pbs_jobid=work.job, passed=True,
         anchor_max_abs=parity, transform_check=check, arms=arms,
         gpu=torch.cuda.get_device_name(0)))

    @torch.no_grad()
    def weight_diagnostic():
        # Weight-level only (no queries): INT8-row rounding error vs the VQ error itself,
        # all normalized by the original transformed weight energy.
        bank.refresh()
        totals = Counter()
        ratios = []
        for name, mod in bank.modules.items():
            reference = bf16_weight(name).float()
            q0 = mod.weight.data.float()
            q0_h = no_u(name, mod.weight.data)
            energy = float(reference.square().sum())
            vq = float((q0 - reference).square().sum())
            int8_r0 = float((symmetric(q0, 127).float() - q0).square().sum())
            int8_h = float((symmetric(q0_h, 127).float() - q0_h).square().sum())
            totals.update(energy=energy, vq=vq, int8_r0=int8_r0, int8_h=int8_h)
            ratios.append(int8_h / vq if vq > 0 else 0.)
            del reference, q0, q0_h
        ratios.sort()
        return dict(layers=len(ratios), vq_nmse=totals['vq'] / totals['energy'],
                    int8_row_r0_nmse=totals['int8_r0'] / totals['energy'],
                    int8_row_h_nmse=totals['int8_h'] / totals['energy'],
                    int8_h_over_vq_median=ratios[len(ratios) // 2], int8_h_over_vq_max=ratios[-1])

    save(out / 'weight_diagnostic.json', dict(protocol=PROTOCOL, pbs_jobid=work.job, **weight_diagnostic()))
    work.progress('weight_diagnostic_complete')

    for arm in arms:
        set_weights(cfg['arms'][arm]['weights'])
        install(arm)
        # Eager anchor for the engine evaluator's startup bitwise check.
        (out / 'anchors').mkdir(parents=True, exist_ok=True)
        a, z, _ = work.query(2000)
        torch.save(dict(action=a.detach().cpu(), future=z.detach().cpu(), arm=arm, case_id=2000, seed_index=0,
                        sampler_seed=work.rows[2000]['sampler_seeds'][0], pbs_jobid=work.job),
                   out / 'anchors' / f'{arm}.pt')
        del a, z
        records = []
        for case in ev['case_ids']:
            for seed in ev['seed_indices']:
                path = out / 'endpoints' / arm / f'{case}_{seed}.json'
                binding = dict(protocol=PROTOCOL, arm=arm, split='heldout', case_id=case, seed_index=seed,
                               sampler_seed=work.rows[case]['sampler_seeds'][seed],
                               suite=work.rows[case]['suite'])
                if path.exists():
                    row = read(path)
                    if any(row.get(k) != v for k, v in binding.items()):
                        raise ValueError('Reusable endpoint identity differs')
                else:
                    action, future, _ = work.query(case, seed)
                    row = dict(binding, pbs_jobid=work.job, **metrics(action, future, work.teacher(case, seed)))
                    if any(not math.isfinite(row[k]) or row[k] < 0 for k in METRICS):
                        raise ValueError('Nonfinite endpoint metric')
                    save(path, row)
                    del action, future
                records.append(row)
                work.progress('evaluation', arm=arm, queries=len(records), case_id=case, seed_index=seed)
        save(out / 'arms' / f'{arm}.json', dict(protocol=PROTOCOL, arm=arm, worker=index, pbs_jobid=work.job,
             complete=True, spec=cfg['arms'][arm],
             heldout=aggregate(records, ev['case_ids'], ev['seed_indices'], cfg['suites'])))

    save(unit / 'summary.json', dict(protocol=PROTOCOL, worker=index, pbs_jobid=work.job, status='complete',
         complete=True, arms=arms, anchor_max_abs=parity, transform_check=check, counters=work.counters,
         target_linears=len(bank.modules), new_teacher_queries=0, training_updates=0,
         seconds=time.monotonic() - started))
    work.progress('complete', arms=arms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--worker', type=int, choices=(0,), default=0)
    args = parser.parse_args()
    cfg = read(args.protocol)
    if cfg['protocol'] != PROTOCOL or tuple(cfg['arms']) != ARMS:
        raise ValueError('Protocol identity differs')
    try:
        run(cfg, args.worker)
    except BaseException as error:
        save(Path(cfg['remote_root']) / f'results/worker_{args.worker}/failure.json',
             dict(protocol=PROTOCOL, worker=args.worker, pbs_jobid=os.environ.get('PBS_JOBID'),
                  error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
