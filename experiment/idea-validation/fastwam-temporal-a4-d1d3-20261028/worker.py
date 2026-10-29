"""D1 (cross-step activation statistics) and D3 (cross-step INT4 activation quantizers, held-out 40x2).

BF16 weights in the frozen Smooth+signs+H128 coordinates of the historical H_W16A4 arm; only the activation
operand changes between arms. Model, plan, cached teachers and metrics come from the loss-pair runtime chain.
0 training, 0 new teacher queries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import time

import sampler  # owns the PBS allocation guard; nothing model-scale is imported before it

sampler._require_compute_allocation()
import torch
import torch.nn.functional as F

import lp_worker as lp
from rotation_bank import RotationBanks, stream_for, transformed
import temporal_quant as tq

PROTOCOL = 'fastwam-temporal-a4-d1d3-20261028-v1'
HERE = Path(__file__).resolve().parent
EXTRA = ('first10gripper_disagreement',)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def aggregate(records, cases, seeds, suites, metrics):
    if Counter((r['case_id'], r['seed_index']) for r in records) != Counter((c, s) for c in cases for s in seeds):
        raise ValueError('Complete paired endpoint coverage required')
    by_suite = {}
    for suite in suites:
        rows = [r for r in records if r['suite'] == suite]
        ids = sorted({r['case_id'] for r in rows})
        by_suite[suite] = {m: sum(sum(r[m] for r in rows if r['case_id'] == c) / len(seeds) for c in ids) / len(ids)
                           for m in metrics}
        by_suite[suite].update(contexts=len(ids), queries=len(rows))
    return dict(queries=len(records), by_suite=by_suite,
                overall={m: sum(by_suite[s][m] for s in suites) / len(suites) for m in metrics})


class Bank:
    """All 614 target Linears call the installed quantizer for their operand; weights are H-coordinate BF16."""

    def __init__(self, modules, rotations, transforms):
        self.modules, self.rotations, self.transforms = modules, rotations, transforms
        self.seen, self.quantizer = set(), None
        for name, mod in modules.items():
            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                return F.linear(self.quantizer.operand(_name, x), _mod.weight, _mod.bias)
            mod.forward = forward

    def z(self, name, x):
        return transformed(x, self.transforms[name], self.rotations, stream_for(name)).float()

    def install(self, spec, prefill, recorder=None):
        self.quantizer = tq.CrossStepQuantizer(spec, transform=self.z, stream=stream_for, prefill=prefill,
                                               recorder=recorder)


class Work(lp.Work):
    def __init__(self, cfg, index):
        self.cfg, self.index = cfg, index
        self.results = Path(cfg['remote_root']) / 'results'
        self.wdir = self.results / f'worker_{index}'
        self.wdir.mkdir(parents=True, exist_ok=True)
        parent = read(HERE / 'lp_protocol.json')
        parent['remote_root'] = cfg['remote_root']
        super().__init__(parent, 'reader_kl')
        self.deadline = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ['FW_SOFT_DEADLINE_SECONDS'])
        self.seed_now = 0

    def progress(self, phase, **extra):
        value = dict(protocol=PROTOCOL, worker=self.index, pbs_jobid=self.job, phase=phase,
                     seconds=time.monotonic() - self.started, **self.counters, **extra)
        save(self.wdir / 'progress.json', value)
        print('FWTQ_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

    def count(self, key):  # two workers share the parent unit directory; keep counters in memory
        self.counters[key] += 1

    def query(self, case, seed=0, grad=False):
        self.bank.quantizer.reset(self.cfg['dither_seed'] + 10 * case + seed)
        return super().query(case, seed, grad)

    def load(self):
        rt = self.runner.Runtime(self.unit)
        rt.model.eval()
        rt.arm('bf16')
        self.runtime = rt
        rb = RotationBanks(rt, self.p['calibration']['bank_directory'])
        transforms, rotations = rb.transforms, rb.rotations
        rb.close()
        if any(float(v.abs().max()) != 0 for v in rotations.raw.values()):
            raise ValueError('Fresh rotations must be zero (U = I)')
        rotations.freeze()
        modules = rt.q.modules
        if len(modules) != 614:
            raise ValueError('Expected 614 target Linears')
        self.check_name = next(n for n, m in modules.items() if stream_for(n) == 'video' and m.in_features >= 256)
        self.check_master = modules[self.check_name].weight.detach().clone()
        with torch.no_grad():
            for name, mod in modules.items():
                mod.weight.data = transformed(mod.weight, transforms[name], rotations, stream_for(name),
                                              weight=True).to(torch.bfloat16)
        self.bank = Bank(modules, rotations, transforms)
        self.bank.install(self.cfg['arms']['W16A16'], self.cfg['prefill'])
        self.helpers.Work.reset_rng(self)

    @torch.no_grad()
    def checks(self):
        mod = self.bank.modules[self.check_name]
        g = torch.Generator(device='cuda').manual_seed(20261028)
        x = torch.randn(8, mod.in_features, device='cuda', generator=g).to(torch.bfloat16)
        reference = F.linear(x, self.check_master).float()
        actual = F.linear(self.bank.z(self.check_name, x).to(torch.bfloat16), mod.weight).float()
        rel = float((actual - reference).norm() / reference.norm())
        # cross-step identity on the GPU for a drifting random sequence
        st, ws, worst = tq.State(), [], 0.
        z = torch.randn(1, 294, 3072, device='cuda', generator=g)
        for s in range(10):
            z = .95 * z + .31 * torch.randn(z.shape, device='cuda', generator=g)
            xhat, w = tq.step(z, 4, None, True, -1.0, st)
            expect = w + (-1.0 * ws[-1] if ws else 0)
            worst = max(worst, float((xhat - z - expect).abs().max()))
            st.xhat, st.w = xhat, w
            ws.append(w)
        values = dict(module=self.check_name, pairing_relative=rel, cross_step_identity_max_abs=worst,
                      gpu=torch.cuda.get_device_name(0))
        if rel > self.cfg['gates']['transform_check_max_relative'] or worst > 1e-4:
            raise ValueError('Startup check failed: ' + json.dumps(values))
        return values

    def d1(self):
        cfg = self.cfg['d1']
        out = self.results / 'd1' / 'raw.json'
        if out.exists():
            return read(out)
        tg = cfg['token_groups']

        def groups(name, tokens):
            stream = stream_for(name)
            if stream == 'video' and tokens == tg['video_tokens']:
                return [('all', slice(None)), ('frame0', slice(0, tg['frame_tokens'])),
                        ('future', slice(tg['frame_tokens'], tokens))]
            if stream == 'action' and tokens == tg['action_tokens']:
                return [('all', slice(None)), ('first10', slice(0, tg['action_executed'])),
                        ('rest', slice(tg['action_executed'], tokens))]
            return [('all', slice(None))]

        recorder = tq.StepRecorder(groups, stream_for)
        self.bank.install(self.cfg['arms']['W16A16'], self.cfg['prefill'], recorder=recorder)
        sums = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        counts = {}
        for number, case in enumerate(cfg['case_ids']):
            action, future, _ = self.query(case, cfg['seed_index'])
            del action, future
            for key, call, groups_values in recorder.flush():
                k = f'{key[0]}|{key[1][-2] if len(key[1]) > 1 else 1}'
                for label, values in groups_values.items():
                    sums[k][call][label].update(values)
            for k, v in self.bank.quantizer.call_counts().items():
                if k in counts and counts[k] != v:
                    raise ValueError(f'D1 call pattern changed between queries: {k} {counts[k]} {v}')
                counts[k] = v
            self.progress('d1', contexts=number + 1)
        meta = {f'{key[0]}|{key[1][-2] if len(key[1]) > 1 else 1}': value for key, value in recorder.meta.items()}
        widths = {n: [m.in_features, m.out_features] for n, m in self.bank.modules.items()}
        value = dict(protocol=PROTOCOL, pbs_jobid=self.job, contexts=len(cfg['case_ids']),
                     keys={k: dict(meta=meta[k], shape=widths[meta[k]['name']], calls=counts[k],
                                   steps={str(c): {lab: dict(v) for lab, v in by_label.items()}
                                          for c, by_label in sorted(by_call.items())})
                           for k, by_call in sums.items()})
        save(out, value)
        return value

    def reference_gate(self, row):
        ref = self.cfg['reference_h_w16a4']
        hist = read(Path(ref['queries']) / f"{row['case_id']}_{row['seed_index']}.json")
        if hist['sampler_seed'] != row['sampler_seed'] or hist['suite'] != row['suite']:
            raise ValueError('Historical H_W16A4 endpoint binding differs')
        diff = {m: abs(row[m] - hist[m]) for m in ('first10motor_rmse', 'future_latent_nmse', 'full32motor_rmse',
                                                   'first10gripper_rmse')}
        rel = max(diff[m] / max(abs(hist[m]), 1e-12) for m in diff)
        value = dict(case_id=row['case_id'], seed_index=row['seed_index'], abs_diff=diff, max_relative=rel,
                     bitwise_metrics=all(d == 0 for d in diff.values()))
        save(self.results / 'gates' / 'h_w16a4_first_endpoint.json', dict(protocol=PROTOCOL, pbs_jobid=self.job,
                                                                            **value))
        if rel > self.cfg['gates']['reference_first_endpoint_max_relative']:
            raise ValueError('A4 arm does not reproduce historical H_W16A4: ' + json.dumps(value))
        return value

    def endpoint(self, arm, case, seed):
        path = self.results / 'endpoints' / arm / f'{case}_{seed}.json'
        binding = dict(protocol=PROTOCOL, arm=arm, split='heldout', case_id=case, seed_index=seed,
                       sampler_seed=self.rows[case]['sampler_seeds'][seed], suite=self.rows[case]['suite'])
        if path.exists():
            row = read(path)
            if any(row.get(k) != v for k, v in binding.items()):
                raise ValueError('Reusable endpoint identity differs')
            return row
        tick = time.monotonic()
        action, future, _ = self.query(case, seed)
        target = self.teacher(case, seed)
        actual = self.runtime.command(action.detach().clone())
        teacher = self.runtime.command(target['action'].detach().clone())
        row = dict(binding, pbs_jobid=self.job, **self.metrics(action, future, target),
                   first10gripper_disagreement=float((actual[:10, 6] != teacher[:10, 6]).mean()),
                   static_reuse=sum(v['static'] for v in self.bank.quantizer.call_counts().values()),
                   seconds=time.monotonic() - tick)
        if any(not math.isfinite(row[m]) or row[m] < 0 for m in tuple(self.helpers.METRICS) + EXTRA):
            raise ValueError('Nonfinite endpoint metric')
        save(path, row)
        return row

    def evaluate(self, arm):
        out = self.results / 'arms' / f'{arm}.json'
        if out.exists():
            return read(out)
        spec, ev = self.cfg['arms'][arm], self.cfg['evaluation']
        self.bank.install(spec, self.cfg['prefill'])
        records, gate = [], None
        for case in ev['case_ids']:
            for seed in ev['seed_indices']:
                row = self.endpoint(arm, case, seed)
                if arm == 'A4' and gate is None:
                    gate = self.reference_gate(row)
                records.append(row)
                self.progress('heldout', arm=arm, queries=len(records))
        metrics = tuple(self.helpers.METRICS) + EXTRA
        result = dict(protocol=PROTOCOL, arm=arm, spec=spec, worker=self.index, pbs_jobid=self.job, complete=True,
                      seconds_per_query=sum(r.get('seconds', 0.) for r in records) / len(records),
                      static_reuse_per_query=sorted({r.get('static_reuse') for r in records}, key=str),
                      **aggregate(records, ev['case_ids'], ev['seed_indices'], self.cfg['suites'], metrics))
        if arm == 'A4':
            ref = self.cfg['reference_h_w16a4']
            worst = {m: 0. for m in ('first10motor_rmse', 'future_latent_nmse')}
            for r in records:
                hist = read(Path(ref['queries']) / f"{r['case_id']}_{r['seed_index']}.json")
                for m in worst:
                    worst[m] = max(worst[m], abs(r[m] - hist[m]))
            result['h_w16a4_reference'] = dict(first_endpoint=gate, max_abs_diff_all_endpoints=worst,
                                               historical_overall=ref['overall'])
        save(out, result)
        return result

    def run(self):
        self.load()
        check = self.checks()
        save(self.wdir / 'startup.json', dict(protocol=PROTOCOL, worker=self.index, pbs_jobid=self.job, **check))
        self.progress('checks_passed')
        if str(self.cfg['d1']['worker']) == str(self.index):
            self.d1()
            self.progress('d1_complete')
        arms = self.cfg['workers'][str(self.index)]
        for arm in arms:
            self.evaluate(arm)
        save(self.wdir / 'summary.json', dict(protocol=PROTOCOL, worker=self.index, pbs_jobid=self.job,
                                              status='complete', complete=True, arms=arms, startup=check,
                                              counters=self.counters, new_teacher_queries=0, training_updates=0,
                                              seconds=time.monotonic() - self.started))
        self.progress('complete', arms=arms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    cfg = read(args.protocol)
    if cfg['protocol'] != PROTOCOL or str(args.worker) not in cfg['workers']:
        raise ValueError('Protocol identity or worker index differs')
    wdir = Path(cfg['remote_root']) / 'results' / f'worker_{args.worker}'
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / 'failure.json').unlink(missing_ok=True)
    try:
        Work(cfg, args.worker).run()
    except BaseException as error:
        save(wdir / 'failure.json', dict(protocol=PROTOCOL, worker=args.worker, pbs_jobid=os.environ.get('PBS_JOBID'),
                                         error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
