"""Scalar W4A8 baselines on the frozen Fast-WAM runtime: {plain, QuaRot R128, SpinQuant R128} x {RTN, GPTQ}, held-out 40x2.

Reuses the loss-pair runtime chain (as 1b does) for the model, the C80 / held-out plan and the cached BF16
teachers, and the rotation-baselines R128 module and learned SpinQuant checkpoint. 0 training, 0 new teachers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import sampler  # owns the PBS allocation guard; nothing model-scale is imported before it

sampler._require_compute_allocation()
import torch
import torch.nn.functional as F

import lp_worker as lp
import rotation as rot
import scalar_quant as sq
from rotation_bank import stream_for

PROTOCOL = 'fastwam-w4a8-scalar-baselines-20261028-v1'
HERE = Path(__file__).resolve().parent
EXTRA = ('first10gripper_disagreement',)
COORDINATES = ('plain', 'quarot', 'spinquant')


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


class NoRotation:
    """The sampler refreshes bank.rotations; these baselines rotate inside their own forward."""
    raw = {}

    def refresh(self):
        pass


class Coordinates:
    """x -> xR and W -> WR with the rotation-baselines R128 (fixed or learned); plain is the identity."""
    def __init__(self, kind, state):
        self.kind, self.state = kind, state

    def x(self, value, name):
        return value.float() if self.state is None else self.state.transform(value.float(), stream_for(name))

    def weight(self, weight, name):
        return self.x(weight, name)

    def matrix(self, width, name):
        return self.x(torch.eye(width, device='cuda'), name)


class Bank:
    """Hessian collection under the BF16 model, then per-arm installed W4A8 forwards."""
    def __init__(self, runtime, masters):
        self.modules, self.masters = runtime.q.modules, masters
        self.rotations, self.seen = NoRotation(), set()
        self.sums, self.rows = {}, {}

    def collect(self, names):
        self.sums = {n: torch.zeros(self.modules[n].in_features, self.modules[n].in_features, device='cuda')
                     for n in names}
        self.rows = dict.fromkeys(names, 0)
        for name, mod in self.modules.items():
            mod.weight.data = self.masters[name].to('cuda')

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                if _name in self.sums:
                    rows = x.detach().float().reshape(-1, _mod.in_features)
                    self.sums[_name].addmm_(rows.t(), rows)
                    self.rows[_name] += rows.shape[0]
                return F.linear(x, _mod.weight, _mod.bias)
            mod.forward = forward

    def install(self, coords, weights, activation_bits):
        for name, mod in self.modules.items():
            mod.weight.data = weights(name)

            def forward(x, _name=name, _mod=mod):
                self.seen.add(_name)
                z = coords.x(x, _name)
                return F.linear(sq.a8(z) if activation_bits == 8 else z.to(torch.bfloat16), _mod.weight, _mod.bias)
            mod.forward = forward


class Work(lp.Work):
    """Loss-pair worker bound to this root; queries, teachers and metrics are inherited unchanged."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.results = Path(cfg['remote_root']) / 'results'
        parent = read(HERE / 'lp_protocol.json')
        parent['remote_root'] = cfg['remote_root']
        super().__init__(parent, 'reader_kl')
        self.deadline = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ['FW_SOFT_DEADLINE_SECONDS'])

    def progress(self, phase, **extra):
        value = dict(protocol=PROTOCOL, pbs_jobid=self.job, phase=phase,
                     seconds=time.monotonic() - self.started, **self.counters, **extra)
        save(self.results / 'progress.json', value)
        print('FWW4A8_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

    def load(self):
        self.runtime = self.runner.Runtime(self.unit)
        self.runtime.model.eval()
        self.runtime.arm('bf16')
        self.masters = {n: m.weight.detach().cpu().clone() for n, m in self.runtime.q.modules.items()}
        if len(self.masters) != 614:
            raise ValueError('Expected 614 target Linears')
        self.bank = Bank(self.runtime, self.masters)
        src = self.cfg['rotation_source']
        root = Path(src['root'])
        receipt = read(root / src['receipt'])
        saved = torch.load(root / src['learned'], map_location='cpu', weights_only=True)
        if (receipt.get('selected_step') != src['selected_step'] or saved.get('mode') != 'learned'
                or saved.get('seed') != src['seed']):
            raise ValueError('SpinQuant rotation checkpoint binding differs')
        modules = self.runtime.q.modules
        fixed = rot.build_rotation_state(modules, mode='fixed', seed=src['seed'])
        learned = rot.build_rotation_state(modules, mode='learned', seed=src['seed'])
        learned.load_state_dict(saved['state_dict'])
        for state in (fixed, learned):
            for value in state.parameters():
                value.requires_grad_(False)
            state.refresh()
        self.coords = dict(plain=Coordinates('plain', None), quarot=Coordinates('quarot', fixed),
                           spinquant=Coordinates('spinquant', learned))
        self.helpers.Work.reset_rng(self)

    @torch.no_grad()
    def checks(self):
        g = self.cfg['gates']
        widths = {}
        for name, mod in self.bank.modules.items():
            widths.setdefault((stream_for(name), mod.in_features), name)
        ortho = {}
        for kind in ('quarot', 'spinquant'):
            for (stream, width), name in widths.items():
                r = self.coords[kind].matrix(width, name)
                ortho[f'{kind}/{stream}/{width}'] = float((r.t() @ r - torch.eye(width, device='cuda')).abs().max())
                del r
        change = {s: float((self.coords['spinquant'].state.matrix(s) - self.coords['quarot'].state.matrix(s)).abs().max())
                  for s in ('video', 'action')}
        name = next(n for n, m in self.bank.modules.items() if stream_for(n) == 'video' and m.in_features >= 3072)
        x = torch.randn(16, self.bank.modules[name].in_features, device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(20261028))
        w = self.masters[name].to('cuda').float()
        reference = x @ w.t()
        pairing = {}
        for kind in ('quarot', 'spinquant'):
            c = self.coords[kind]
            actual = c.x(x, name) @ c.weight(w, name).t()
            pairing[kind] = float((actual - reference).norm() / reference.norm())
        values = dict(orthogonality=ortho, spinquant_change=change, pairing_module=name, pairing_relative=pairing)
        if (max(ortho.values()) > g['orthogonality_max_abs'] or min(change.values()) < g['spinquant_min_change']
                or max(pairing.values()) > g['transform_check_max_relative']):
            raise ValueError('Rotation gate failed: ' + json.dumps(values))
        return values

    def qpath(self, kind, index):
        return self.results / 'gptq' / kind / f'{index:04d}.pt'

    def hessian_passes(self):
        """Group the layers still lacking GPTQ codes into passes whose FP32 Hessians fit the measured free memory."""
        names = list(self.bank.modules)
        todo = [n for i, n in enumerate(names) if not all(self.qpath(k, i).exists() for k in COORDINATES)]
        if not todo:
            return [], {}
        self.bank.collect([])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        action, future, _ = self.query(lp.CASES[0])
        del action, future
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory
        peak = torch.cuda.max_memory_reserved()
        budget = total - peak - self.cfg['gates']['hessian_headroom_gb'] * 2 ** 30
        passes, current, used = [], [], 0
        for name in sorted(todo, key=lambda n: -self.bank.modules[n].in_features):
            size = 4 * self.bank.modules[name].in_features ** 2
            if current and used + size > budget:
                passes.append(current)
                current, used = [], 0
            current.append(name)
            used += size
        passes.append(current)
        return passes, dict(total_gb=total / 2 ** 30, query_peak_gb=peak / 2 ** 30, budget_gb=budget / 2 ** 30,
                            layers=len(todo), passes=len(passes))

    @torch.no_grad()
    def quantize_layer(self, name, index, hessian):
        w = self.masters[name].to('cuda').float()
        stats = dict(module=name, module_index=index, shape=list(w.shape))
        for kind in COORDINATES:
            c = self.coords[kind]
            if kind == 'plain':
                h = hessian
            else:
                r = c.matrix(w.shape[1], name)
                h = r.t() @ hessian @ r
                del r
            wc = c.weight(w, name)
            codes, scales = sq.gptq(wc, h, damp=self.cfg['weights']['gptq']['damp'])
            rc, rs = sq.rtn(wc)
            self.qpath(kind, index).parent.mkdir(parents=True, exist_ok=True)
            stats[kind] = dict(gptq_proxy_nmse=sq.proxy_nmse(wc, sq.dequant(codes, scales), h),
                               rtn_proxy_nmse=sq.proxy_nmse(wc, sq.dequant(rc, rs), h))
            self.atomic_torch_save(torch, self.qpath(kind, index), dict(
                protocol=PROTOCOL, module=name, module_index=index, coordinates=kind, shape=list(w.shape),
                packed=sq.pack(codes).cpu(), scales=scales.cpu(), stats=stats[kind]))
            del h, wc, codes, scales, rc, rs
        save(self.results / 'gptq_stats' / f'{index:04d}.json', stats)
        return stats

    def calibrate(self):
        names = list(self.bank.modules)
        index_of = {n: i for i, n in enumerate(names)}
        passes, plan = self.hessian_passes()
        save(self.results / 'hessian_plan.json', dict(protocol=PROTOCOL, pbs_jobid=self.job, **plan))
        for number, group in enumerate(passes):
            self.bank.collect(group)
            for count, case in enumerate(lp.CASES):
                action, future, _ = self.query(case)
                del action, future
                self.progress('hessian', pass_index=number, passes=len(passes), contexts=count + 1, layers=len(group))
            for done, name in enumerate(group):
                self.enough()
                h = self.bank.sums.pop(name) / self.bank.rows[name]
                self.quantize_layer(name, index_of[name], h)
                del h
                self.progress('gptq', pass_index=number, passes=len(passes), layers_done=done + 1, layers=len(group))
            self.bank.sums.clear()
            torch.cuda.empty_cache()
        stats = [read(self.results / 'gptq_stats' / f'{i:04d}.json') for i in range(len(names))]
        weights = sum(s['shape'][0] * s['shape'][1] for s in stats)
        summary = {kind: {f'{m}_weighted': sum(s[kind][m] * s['shape'][0] * s['shape'][1] for s in stats) / weights
                          for m in ('gptq_proxy_nmse', 'rtn_proxy_nmse')} for kind in COORDINATES}
        summary['bpw'] = sum(sq.bpw(*s['shape']) * s['shape'][0] * s['shape'][1] for s in stats) / weights
        summary['target_weights'] = weights
        save(self.results / 'gptq_summary.json', dict(protocol=PROTOCOL, pbs_jobid=self.job, **summary))
        return summary

    def weights_for(self, spec):
        kind, mode = spec['coordinates'], spec['weights']
        c, names = self.coords[kind], list(self.bank.modules)

        def get(name):
            w = self.masters[name].to('cuda')
            if mode == 'bf16':
                return c.weight(w, name).to(torch.bfloat16)
            if mode == 'rtn':
                codes, scales = sq.rtn(c.weight(w, name))
            else:
                saved = torch.load(self.qpath(kind, names.index(name)), map_location='cuda', weights_only=True)
                if saved['protocol'] != PROTOCOL or saved['module'] != name or saved['coordinates'] != kind:
                    raise RuntimeError('Saved GPTQ layer identity differs: ' + name)
                codes, scales = sq.unpack(saved['packed'], w.shape[1]), saved['scales']
            return sq.dequant(codes, scales).to(torch.bfloat16)
        return get

    def endpoint(self, label, case, seed):
        path = self.results / 'endpoints' / label / f'{case}_{seed}.json'
        binding = dict(protocol=PROTOCOL, arm=label, split='heldout', case_id=case, seed_index=seed,
                       sampler_seed=self.rows[case]['sampler_seeds'][seed], suite=self.rows[case]['suite'])
        if path.exists():
            row = read(path)
            if any(row.get(k) != v for k, v in binding.items()):
                raise ValueError('Reusable endpoint identity differs')
            return row
        action, future, _ = self.query(case, seed)
        target = self.teacher(case, seed)
        actual = self.runtime.command(action.detach().clone())
        teacher = self.runtime.command(target['action'].detach().clone())
        row = dict(binding, pbs_jobid=self.job, **self.metrics(action, future, target),
                   first10gripper_disagreement=float((actual[:10, 6] != teacher[:10, 6]).mean()))
        save(path, row)
        return row

    def evaluate(self, label):
        out = self.results / 'heldout' / f'{label}.json'
        if out.exists():
            return read(out)
        spec, ev = self.cfg['arms'][label], self.cfg['evaluation']
        self.bank.install(self.coords[spec['coordinates']], self.weights_for(spec), spec['activation_bits'])
        records = []
        for case in ev['case_ids']:
            for seed in ev['seed_indices']:
                records.append(self.endpoint(label, case, seed))
                self.progress('heldout', arm=label, queries=len(records))
        result = dict(protocol=PROTOCOL, arm=label, spec=spec, pbs_jobid=self.job,
                      **aggregate(records, ev['case_ids'], ev['seed_indices'], self.cfg['suites'],
                                  tuple(self.helpers.METRICS) + EXTRA))
        save(out, result)
        return result

    def run(self):
        self.load()
        check = self.checks()
        save(self.results / 'checks.json', dict(protocol=PROTOCOL, pbs_jobid=self.job, **check))
        self.progress('checks_passed')
        calibration = self.calibrate()
        heldout = {arm: self.evaluate(arm)['overall'] for arm in self.cfg['arms']}
        summary = dict(protocol=PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, checks=check,
                       gptq=calibration, heldout=heldout, references=self.cfg['references'],
                       counters=self.counters, new_teacher_queries=0, training_updates=0,
                       seconds=time.monotonic() - self.started)
        save(self.results / 'summary.json', summary)
        (self.results / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')
        self.progress('complete')


def report(s):
    ref, b = s['references']['q0_a4']['overall'], s['references']['1b']
    lines = ['# 标量 W4A8 基线：held-out 40×2', '',
             f"作业 `{s['pbs_jobid']}`；权重 INT4 g128 BF16 scale（{s['gptq']['bpw']:.3f} bpw），A8 per-token。", '',
             '| 臂 | first10 motor RMSE | future NMSE | first10 gripper RMSE | gripper 分歧 |',
             '|---|---:|---:|---:|---:|']
    for arm, o in s['heldout'].items():
        lines.append(f"| {arm} | {o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | "
                     f"{o['first10gripper_rmse']:.4f} | {o['first10gripper_disagreement']:.4f} |")
    lines.append(f"| 1b（VQ 4.02 bpw，A8，A+Z+K） | {b['first10motor_rmse']:.4f} | {b['future_latent_nmse']:.4f} | "
                 f"{b['first10gripper_rmse']:.4f} | — |")
    lines.append(f"| Q0（A4） | {ref['first10motor_rmse']:.4f} | {ref['future_latent_nmse']:.4f} | "
                 f"{ref['first10gripper_rmse']:.4f} | {ref['first10gripper_disagreement']:.4f} |")
    lines += ['', '## 权重误差代理（tr(ΔW H ΔWᵀ)/tr(W H Wᵀ)，按权重数加权）', '', '| 坐标 | RTN | GPTQ |', '|---|---:|---:|']
    for kind in COORDINATES:
        g = s['gptq'][kind]
        lines.append(f"| {kind} | {g['rtn_proxy_nmse_weighted']:.5f} | {g['gptq_proxy_nmse_weighted']:.5f} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    cfg = read(args.protocol)
    if cfg['protocol'] != PROTOCOL:
        raise ValueError('Protocol identity differs')
    root = Path(cfg['remote_root']) / 'results'
    (root / 'failure.json').unlink(missing_ok=True)
    try:
        Work(cfg).run()
    except BaseException as error:
        save(root / 'failure.json', dict(protocol=PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'),
                                         error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
