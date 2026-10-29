"""WAM-guided Hessian plugin on Fast-WAM-IDM (BF16 model, A16).

S1  Gauss-Newton probes of L = A + lambda Z + beta K on C80: query (video loop) and a reader pass (prefill + action
    loop from the query's own future, same weights) give per-call token weights u and row weights d.
S2  Token/step-weighted Hessians H~ = sum (u/ubar + gamma) x x^T / sum w, saved in the A16-baseline file format.
S3  QTIP W2 host (the A16 baselines' methods/qtip, same per-layer seeds) on H~; the baselines' QTIP_W2 is the H_U arm.
S4  Held-out 40x2 for QTIP_U_W2 and QTIP_G_W2, plus the C1 swap (quantized reader on the BF16 teacher future).
Phase 2 (rollout-coherent): the same probes, but per layer the sketch S = R^T G of the probe-loss gradient of the
    shared weight, G summed over every call (all denoising steps, prefill, reader), so the Hessian S^T S keeps the
    cross-token and cross-step terms the token weights drop; variants C (A + lambda Z + beta K) and CA (A only), each
    QTIP W2 with the baseline seeds; gate, held-out, closed loop. Outputs under results/coherent/ and
    results/{quantized,heldout,loops}/QTIP_C_W2, QTIP_CA_W2.
Model, plan, cached teachers and metrics come from the loss-pair runtime chain. 0 training, 0 new teacher queries.
One process per GPU and step (FW_STAGE check / sketch / gate / quant / eval / final, shard FW_SHARD of FW_SHARDS);
run.pbs runs the steps in order with the 280 closed loops before final.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import sampler  # owns the PBS allocation guard; nothing model-scale is imported before it

sampler._require_compute_allocation()
import torch
import torch.nn.functional as F

import lp_worker as lp
import probe as pr
from rotation_bank import stream_for
import softbin_loss

PROTOCOL = 'fastwam-guided-hessian-20261028-v1'
HERE = Path(__file__).resolve().parent
SWAP = ('first10motor_rmse_swap', 'first10motor_mse_swap', 'full32motor_rmse_swap')
EXTRA = ('first10gripper_disagreement',) + SWAP
GUIDED = 'QTIP_G_W2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    """Atomic JSON write; the temporary name is per host and process because gdev shards share result dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{socket.gethostname()}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def max_abs(a, b):
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())


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


def mcnemar(wins, losses):
    """Two-sided exact McNemar p on the discordant pairs."""
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


class NoRotation:
    """The sampler refreshes bank.rotations; this bank runs in original coordinates."""
    raw = {}

    def refresh(self):
        pass


@contextmanager
def passthrough(*_args):
    yield


class Replay:
    """Weighted x x^T for one group of layers during plain C80 queries; checks inputs against the S1 signatures."""

    def __init__(self, group, modules, ubar, gamma, check):
        self.H = {i: torch.zeros(modules[i].in_features, modules[i].in_features, device='cuda') for i in group}
        self.wsum = {i: torch.zeros((), device='cuda', dtype=torch.float64) for i in group}
        self.ubar, self.gamma, self.check = ubar, gamma, check
        self.counts, self.u, self.sig, self.pending, self.worst = Counter(), None, None, [], 0.0

    def start(self, u, sig):
        self.counts.clear()
        self.u, self.sig, self.pending = u, sig, []

    def observe(self, idx, x):
        n = self.counts[idx]
        self.counts[idx] += 1
        if self.check:
            self.pending.append((idx, n, pr.signature(x)))
        if idx in self.H:
            rows, w = pr.weighted_rows(x, self.u.get(idx, {}).get(n), self.ubar[idx], self.gamma)
            self.H[idx].addmm_(rows.t(), rows)
            self.wsum[idx] += w.double()

    def finish(self, tol):
        if not self.check:
            return
        values = torch.stack([s for _, _, s in self.pending]).cpu().tolist()
        for (idx, n, _), got in zip(self.pending, values):
            want = self.sig[idx][n]
            scale = max(abs(want[1]), 1e-30)
            self.worst = max(self.worst, abs(got[0] - want[0]) / scale, abs(got[1] - want[1]) / scale)
        if self.worst > tol:
            raise RuntimeError(f'Replay inputs differ from the S1 query inputs (relative {self.worst:.3g})')


class Bank:
    """614 target Linears: plain BF16 F.linear, probe mode (Probe nodes + call bookkeeping) or weighted replay."""

    def __init__(self, modules):
        self.modules, self.names = modules, list(modules)
        self.by_index = [modules[n] for n in self.names]
        self.rotations, self.seen = NoRotation(), set()
        self.anchor = torch.zeros((), device='cuda', requires_grad=True)
        self.mode, self.counting, self.phase, self.sink, self.replay = 'plain', False, None, None, None
        self.calls, self.xn, self.rmap, self.R, self.Q = {}, defaultdict(list), {}, None, None
        for idx, name in enumerate(self.names):
            self.modules[name].forward = self._forward(idx, name, self.modules[name])

    def _forward(self, idx, name, mod):
        def run(x):
            self.seen.add(name)
            y = F.linear(x, mod.weight, mod.bias)
            if self.mode == 'probe':
                tag = None
                if self.counting:
                    calls = self.calls[self.phase][idx]
                    xn = x.detach().float().reshape(-1, mod.in_features).square().sum(1)
                    tag = (self, idx, self.phase, len(calls), xn)
                    calls.append(pr.signature(x))
                    if self.phase == 'q':
                        self.xn[idx].append(xn)
                return pr.Probe.apply(y, self.anchor, tag, x)
            if self.mode == 'replay':
                self.replay.observe(idx, x)
            return y
        return run

    def begin_probe(self):
        self.mode, self.counting, self.phase, self.sink = 'probe', True, 'q', None
        self.calls = {'q': defaultdict(list), 'r': defaultdict(list)}
        self.xn, self.rmap = defaultdict(list), {}

    def finish_forward(self, tol):
        """Freeze call counting (checkpoint recomputation re-enters the wrappers) and map reader calls to query calls."""
        self.counting = False
        self.qsig = {i: torch.stack(v).cpu().tolist() for i, v in self.calls['q'].items()}
        rsig = {i: torch.stack(v).cpu().tolist() for i, v in self.calls['r'].items()}
        worst = 0.0
        for i, r in rsig.items():
            self.rmap[i], w = pr.match_suffix(self.qsig.get(i, []), r, tol)
            worst = max(worst, w)
        return dict(query_calls=sum(len(v) for v in self.qsig.values()),
                    reader_calls=sum(len(v) for v in rsig.values()), signature_worst_relative=worst)

    def record(self, tag, g, x):
        if self.sink is None:
            return
        _, idx, phase, n, xn = tag
        s = self.sink
        if 'S' in s:  # coherent pass: R^T g x^T and g x^T Q summed over every call of this layer (reader calls included)
            part = pr.sketch(g, x, self.R[idx])
            s['S'][idx] = s['S'][idx] + part if idx in s['S'] else part
            part = pr.row_sketch(g, x, self.Q[idx])
            s['T'][idx] = s['T'][idx] + part if idx in s['T'] else part
        if phase == 'r':
            n = self.rmap[idx][n]
        u, d = pr.token_and_row(g, xn)
        s['u'][(idx, n)] = s['u'][(idx, n)] + u if (idx, n) in s['u'] else u
        s['d'][idx] = s['d'][idx] + d if idx in s['d'] else d

    def end_probe(self):
        self.mode, self.counting, self.phase, self.sink = 'plain', False, None, None
        self.calls, self.rmap, self.R, self.Q = {}, {}, None, None


def owned(shard, shards, cost, items=range(614)):
    """Greedy largest-first split of items (layer indices or cases) across shards by cost; returns this shard's."""
    loads, owner = [0.0] * shards, {}
    for i in sorted(items, key=lambda i: (-cost(i), i)):
        k = min(range(shards), key=lambda s: (loads[s], s))
        owner[i] = k
        loads[k] += cost(i)
    return [i for i in items if owner[i] == shard]


class Work(lp.Work):
    def __init__(self, cfg, job):
        self.cfg, self.job_name = cfg, job
        self.host = read(HERE / 'host.json')
        self.base = Path(cfg['remote_root']) / 'results'
        self.results = self.base
        self.baseline = Path(cfg['baseline']['root']) / 'results'
        parent = read(HERE / 'lp_protocol.json')
        parent['remote_root'] = cfg['remote_root']
        super().__init__(parent, 'reader_kl')
        self.deadline = float(os.environ['FW_JOB_START_EPOCH']) + float(os.environ['FW_SOFT_DEADLINE_SECONDS'])

    def progress(self, phase, **extra):
        value = dict(protocol=PROTOCOL, pbs_jobid=self.job, job=self.job_name, phase=phase,
                     seconds=time.monotonic() - self.started, **self.counters, **extra)
        save(self.base / 'progress' / f'{self.job_name}.json', value)
        print('FWGH_PROGRESS ' + json.dumps(value, allow_nan=False), flush=True)

    def count(self, key):
        self.counters[key] += 1

    def tsave(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.atomic_torch_save(torch, path, value)

    def barrier(self, label, paths):
        """Wait for files the other GPU's process of the same step writes; run.pbs stops both if either fails."""
        missing, waited = [p for p in paths if not p.exists()], 0
        if missing:
            self.progress('barrier', step=label, missing=len(missing))
        while missing:
            self.enough()
            time.sleep(self.host['barrier_poll_seconds'])
            waited += self.host['barrier_poll_seconds']
            missing = [p for p in missing if not p.exists()]
        return waited

    def load(self):
        import frozen_reader as fr
        unit = self.unit / self.job_name  # Runtime writes its config/receipt here; shards must not share it
        unit.mkdir(parents=True, exist_ok=True)
        rt = self.runner.Runtime(unit)
        rt.model.eval()
        rt.arm('bf16')
        for value in rt.model.parameters():
            value.requires_grad_(False)
        self.runtime = rt
        modules = rt.q.modules
        if len(modules) != 614:
            raise ValueError('Expected 614 target Linears')
        self.bank = Bank(modules)
        # The reader keeps the bank's Linear wrappers: the installed weights are what it reads with.
        fr._bf16_forward_context = passthrough
        fr._checkpoint_context_fn = lambda *_args: (lambda: (nullcontext(), nullcontext()))
        self.reader = fr.BF16Reader(rt, {n: m.weight.detach() for n, m in modules.items()})
        self.helpers.Work.reset_rng(self)

    def diagnostic_cases(self):
        per, chosen = self.cfg['probes']['diagnostic_cases_per_suite'], []
        for suite in self.cfg['suites']:
            chosen += [c for c in lp.CASES if self.rows[c]['suite'] == suite][:per]
        return chosen

    # ---------------- S1: probes ----------------
    def stats_context(self, case, separate):
        b, cfg, loss = self.bank, self.cfg['probes'], self.cfg['loss']
        target = self.teacher(case, 0)
        b.begin_probe()
        action_q, future, _ = self.query(case, 0, grad=True)
        del action_q
        b.phase = 'r'
        value, description = self.datum(case)
        self.runtime.description = description
        leaf = future.detach().float().requires_grad_(True)
        action, _ = self.reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], leaf, checkpointed=True)
        self.count('reader_queries')
        calls = b.finish_forward(self.cfg['gates']['signature_tolerance'])
        parity = dict(future_max_abs=max_abs(future, target['future']), action_max_abs=max_abs(action, target['action']))
        if max(parity.values()) > self.cfg['gates']['teacher_parity_max_abs']:
            raise RuntimeError(f'BF16 query/reader differ from the cached teacher: {parity}')

        stats = self.p['reader_source']['action_stats']
        curv = pr.softbin_kl_curvature(softbin_loss, target['action'][:10, :6], stats['mean'], stats['scale'],
                                       self.p['reader_source']['config']).to('cuda')
        znorm = float(target['future'][:, :, 1:].float().norm())
        scales = pr.head_scales(loss['lambda'], loss['beta'])
        gen = torch.Generator(device='cuda').manual_seed(cfg['seed'] + case)
        heads = ('A', 'Az', 'K', 'Z') if separate else ('A', 'V')
        sinks = {h: dict(u={}, d={}) for h in heads}
        probes = cfg['probes_per_context']
        for p in range(probes):
            last = p == probes - 1
            ga = torch.zeros_like(action)
            ga[:10, :6] = pr.rademacher((10, 6), gen, 'cuda') / math.sqrt(60)
            gk = torch.zeros_like(action)
            gk[:10, :6] = pr.rademacher((10, 6), gen, 'cuda') * (curv / 2).sqrt()
            vz = torch.zeros(future.shape, device='cuda')
            vz[:, :, 1:] = pr.rademacher(vz[:, :, 1:].shape, gen, 'cuda') / znorm
            b.sink = sinks['A']
            c1 = torch.autograd.grad(action, [leaf, b.anchor], ga, retain_graph=True, allow_unused=True)[0]
            b.sink = None
            c2, = torch.autograd.grad(action, [leaf], gk, retain_graph=not last)
            self.counters['parameter_reverse_passes'] += 2
            parts = ([('Az', c1), ('K', scales['K'] * c2), ('Z', scales['Z'] * vz)] if separate
                     else [('V', c1 + scales['K'] * c2 + scales['Z'] * vz)])
            for k, (head, c) in enumerate(parts):
                b.sink = sinks[head]
                torch.autograd.grad(future, [b.anchor], c.to(future.dtype),
                                    retain_graph=not (last and k == len(parts) - 1), allow_unused=True)
                b.sink = None
                self.counters['parameter_reverse_passes'] += 1
            del c1, c2, parts

        u, d, masses = defaultdict(dict), {}, {}
        for head, s in sinks.items():
            per_layer = defaultdict(float)
            for (idx, n), value in s['u'].items():
                u[idx][n] = u[idx][n] + value if n in u[idx] else value
                per_layer[(idx, n)] += float((value * b.xn[idx][n]).sum()) / probes
            for idx, value in s['d'].items():
                d[idx] = d[idx] + value if idx in d else value
            masses[head] = per_layer
        rows = {idx: int(sum(t.numel() for t in v)) for idx, v in b.xn.items()}
        record = dict(case=case, separate=separate, probes=probes, rows=rows,
                      u={idx: {n: v.div(probes).cpu() for n, v in calls.items()} for idx, calls in u.items()},
                      d={idx: v.div(probes).cpu() for idx, v in d.items()},
                      sig={idx: dict(enumerate(v)) for idx, v in b.qsig.items()},
                      masses={h: {f'{i}|{n}': v for (i, n), v in m.items()} for h, m in masses.items()})
        self.tsave(self.results / 'stats' / f'{case}.pt', record)
        summary = dict(protocol=PROTOCOL, case_id=case, separate=separate, parity=parity, calls=calls,
                       znorm=znorm, kl_curvature_mean=float(curv.mean()), groups=self.group_masses(masses),
                       pbs_jobid=self.job)
        save(self.results / 'stats' / f'{case}.json', summary)
        b.end_probe()
        del action, future, leaf, sinks, u, d
        torch.cuda.empty_cache()
        return summary

    def group_masses(self, masses):
        """Per head: isotropic-weight-noise loss mass split into video steps / prefill / action steps / proprio."""
        out = {}
        for head, per in masses.items():
            g = dict(video_steps=[0.0] * 10, video_prefill=0.0, action_steps=[0.0] * 10, other=0.0)
            for key, value in per.items():
                idx, n = (key if isinstance(key, tuple) else tuple(int(v) for v in key.split('|')))
                stream = stream_for(self.bank.names[idx])
                if stream == 'video' and n < 10:
                    g['video_steps'][n] += value
                elif stream == 'video':
                    g['video_prefill'] += value
                elif stream == 'action' and n < 10:
                    g['action_steps'][n] += value
                else:
                    g['other'] += value
            out[head] = g
        return out

    def stats(self, shard=0, shards=1):
        diag = set(self.diagnostic_cases())
        # A diagnostic context back-propagates the video loop three times instead of once: count it double.
        mine = owned(shard, shards, lambda c: 2.0 if c in diag else 1.0, lp.CASES)
        for count, case in enumerate(mine):
            if not (self.results / 'stats' / f'{case}.json').exists():
                self.enough()
                self.stats_context(case, case in diag)
            self.progress('stats', contexts=count + 1, of=len(mine))

    # ---------------- S2: weighted Hessians ----------------
    def passes(self, todo):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        action, future, _ = self.query(lp.CASES[0])
        del action, future
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory
        budget = total - torch.cuda.max_memory_reserved() - self.cfg['gates']['hessian_headroom_gb'] * 2 ** 30
        groups, current, used = [], [], 0
        for i in sorted(todo, key=lambda i: -self.bank.by_index[i].in_features):
            size = 4 * self.bank.by_index[i].in_features ** 2
            if current and used + size > budget:
                groups.append(current)
                current, used = [], 0
            current.append(i)
            used += size
        groups.append(current)
        return groups, dict(budget_gb=budget / 2 ** 30, passes=len(groups), layers=len(todo))

    def hessians(self, shard=0, shards=1):
        out = self.results / 'hessians'
        missing = [c for c in lp.CASES if not (self.results / 'stats' / f'{c}.pt').exists()]
        if missing:
            raise RuntimeError(f'S1 statistics missing for {len(missing)} contexts, e.g. {missing[:4]}')
        mine = owned(shard, shards, lambda i: self.bank.by_index[i].in_features ** 2)
        todo = [i for i in mine if not (out / f'{i:04d}.pt').exists()]
        if todo:
            gamma = self.cfg['hessian']['gamma']
            records = {case: torch.load(self.results / 'stats' / f'{case}.pt', map_location='cpu', weights_only=True)
                       for case in lp.CASES}
            usum, rows, dsum = defaultdict(float), defaultdict(int), {}
            for r in records.values():
                for idx, calls in r['u'].items():
                    usum[idx] += sum(float(v.sum()) for v in calls.values())
                for idx, n in r['rows'].items():
                    rows[idx] += n
                for idx, v in r['d'].items():
                    dsum[idx] = dsum[idx] + v if idx in dsum else v.clone()
            ubar = {i: usum[i] / rows[i] if usum[i] > 0 else 1.0 for i in range(614)}
            groups, plan = self.passes(todo)
            save(self.results / 'hessian_plan' / f'{self.job_name}.json',
                 dict(protocol=PROTOCOL, pbs_jobid=self.job, shard=shard, shards=shards, **plan))
            for number, group in enumerate(groups):
                replay = Replay(group, self.bank.by_index, ubar, gamma, check=number == 0)
                self.bank.mode, self.bank.replay = 'replay', replay
                with torch.no_grad():
                    for count, case in enumerate(lp.CASES):
                        replay.start(records[case]['u'], records[case]['sig'])
                        action, future, _ = self.query(case)
                        del action, future
                        replay.finish(self.cfg['gates']['signature_tolerance'])
                        self.progress('hessian', pass_index=number, passes=len(groups), contexts=count + 1)
                self.bank.mode, self.bank.replay = 'plain', None
                for i in group:
                    self.tsave(out / f'{i:04d}.pt', dict(
                        module=self.bank.names[i], rows=rows[i], hessian=(replay.H[i] / replay.wsum[i]).float().cpu(),
                        weight_sum=float(replay.wsum[i]), ubar=ubar[i], gamma=gamma,
                        row_weight=dsum[i].float() if i in dsum else None, replay_signature_worst=replay.worst))
                del replay
                torch.cuda.empty_cache()
        return dict(layers=614, gamma=self.cfg['hessian']['gamma'])

    # ---------------- S3: QTIP host ----------------
    def qtip(self):
        method = HERE / 'methods' / self.host['host']['method']
        if str(method) not in sys.path:
            sys.path.insert(0, str(method))
        return importlib.import_module('quant')

    def method(self, name):
        """The A16 baselines' quantizer methods/<name>/quant.py under its own module name, so several hosts share a
        process (each loads its vendored code relative to its own file)."""
        key = f'fwgh_quant_{name}'
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, HERE / 'methods' / name / 'quant.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        return sys.modules[key]

    def arm_root(self, arm):
        hosts = self.host['hosts']['arms']
        if arm == GUIDED or arm in self.host['coherent']['arms'] or arm in hosts:
            return self.base / 'quantized' / arm
        if arm == 'QTIP_U_W2':
            return self.baseline / 'quantized' / self.host['host']['baseline_arm']
        if arm in {c['baseline_arm'] for c in hosts.values()}:
            return self.baseline / 'quantized' / arm
        raise ValueError(arm)

    def quantize(self, shard=0, shards=1):
        """QTIP W2 on H~ for this shard's layers; proxies of ours and of the baseline QTIP_W2 weight under H_U and H~."""
        impl, prep, bits = self.qtip(), self.baseline / 'prep', self.host['host']['bits']
        if any(not (self.results / 'hessians' / f'{i:04d}.pt').exists() for i in range(614)):
            raise RuntimeError('S2 guided Hessians are incomplete')
        base = self.arm_root('QTIP_U_W2')
        if not read(base / 'COMPLETE.json').get('complete'):
            raise RuntimeError('Baseline QTIP_W2 is incomplete')
        rows, root = read(prep / 'modules.json'), self.arm_root(GUIDED)
        # Balance the GPUs on what each layer actually cost the baseline's QTIP W2 run (same code, shapes, A100).
        spent = [read(base / 'stats' / f'{i:04d}.json')['seconds'] for i in range(614)]
        mine = owned(shard, shards, spent.__getitem__)
        torch.use_deterministic_algorithms(False)  # as quant_worker: ported ops must not be refused
        try:
            for done, i in enumerate(mine):
                if (root / 'stats' / f'{i:04d}.json').exists():
                    continue
                self.enough()
                name = rows[i]['module']
                saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                         for k, path in (('master', prep / 'masters'), ('U', prep / 'hessians'),
                                         ('G', self.results / 'hessians'), ('baseline', base / 'weights'))}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                w = saved['master']['weight'].to('cuda', torch.float32)
                h = {k: saved[k]['hessian'].to('cuda', torch.float32) for k in ('U', 'G')}
                begin = time.monotonic()
                result = impl.quantize_layer(w, h['G'], bits, seed=self.host['host']['seed_base'] + i)
                seconds = time.monotonic() - begin
                dq = result['dequant'].to('cuda', torch.float32)
                if tuple(dq.shape) != tuple(w.shape) or not bool(torch.isfinite(dq).all()):
                    raise RuntimeError(f'QTIP produced an invalid weight at {name}')

                def proxy(q):
                    d = q - w
                    return {k: float(((d @ v) * d).sum() / ((w @ v) * w).sum().clamp_min(1e-30)) for k, v in h.items()}
                ours, theirs = proxy(dq), proxy(saved['baseline']['weight'].to('cuda', torch.float32))
                self.tsave(root / 'weights' / f'{i:04d}.pt', dict(module=name, weight=dq.to(torch.bfloat16).cpu()))
                save(root / 'stats' / f'{i:04d}.json', dict(
                    index=i, module=name, shape=list(w.shape), bits_total=result['bits_total'],
                    bpw=result['bits_total'] / w.numel(), seconds=seconds, config=result['config'], hessian='G',
                    proxy_U=ours['U'], proxy_G=ours['G'], baseline_proxy_U=theirs['U'], baseline_proxy_G=theirs['G']))
                del saved, w, h, result, dq
                torch.cuda.empty_cache()
                if done % 25 == 0:
                    self.progress('quant', layers=done + 1, of=len(mine))
        finally:
            torch.use_deterministic_algorithms(True)

    def quant_summary(self):
        """quantized/QTIP_G_W2/COMPLETE.json (also what the baselines' analysis.py reads) and both arms' proxies."""
        stats = [read(self.arm_root(GUIDED) / 'stats' / f'{i:04d}.json') for i in range(614)]
        n = sum(s['shape'][0] * s['shape'][1] for s in stats)

        def weighted(key):
            return sum(s[key] * s['shape'][0] * s['shape'][1] for s in stats) / n
        base = read(self.arm_root('QTIP_U_W2') / 'COMPLETE.json')
        guided = dict(arm=GUIDED, method=self.host['host']['method'], bits=self.host['host']['bits'], hessian='G',
                      bpw=sum(s['bits_total'] for s in stats) / n, proxy_nmse_weighted=weighted('proxy_U'),
                      proxy_U_weighted=weighted('proxy_U'), proxy_G_weighted=weighted('proxy_G'),
                      quantize_seconds=sum(s['seconds'] for s in stats), config_example=stats[0]['config'])
        save(self.arm_root(GUIDED) / 'COMPLETE.json', dict(protocol=PROTOCOL, complete=True, pbs_jobid=self.job,
                                                           target_weights=n, **guided))
        uniform = dict(arm='QTIP_U_W2', source=str(self.arm_root('QTIP_U_W2')), hessian='U', bpw=base['bpw'],
                       proxy_U_weighted=weighted('baseline_proxy_U'), proxy_G_weighted=weighted('baseline_proxy_G'),
                       baseline_reported_proxy=base['proxy_nmse_weighted'])
        return {'QTIP_U_W2': uniform, GUIDED: {k: v for k, v in guided.items() if k != 'config_example'}}

    # ---------------- S4: held-out + C1 swap ----------------
    def install(self, arm):
        root = self.arm_root(arm)
        with torch.no_grad():
            for i, name in enumerate(self.bank.names):
                saved = torch.load(root / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                if saved['module'] != name:
                    raise RuntimeError(f'Stored weight identity differs at {i}')
                self.bank.modules[name].weight.data.copy_(saved['weight'].to('cuda'))

    def endpoint(self, arm, case, seed):
        path = self.results / 'endpoints' / arm / f'{case}_{seed}.json'
        sampler_seed = self.rows[case]['sampler_seeds'][seed]
        binding = dict(protocol=PROTOCOL, arm=arm, split='heldout', case_id=case, seed_index=seed,
                       sampler_seed=sampler_seed, suite=self.rows[case]['suite'])
        if path.exists():
            row = read(path)
            if any(row.get(k) != v for k, v in binding.items()):
                raise ValueError('Reusable endpoint identity differs')
            return row
        action, future, _ = self.query(case, seed)
        target = self.teacher(case, seed)
        actual = self.runtime.command(action.detach().clone())
        teacher = self.runtime.command(target['action'].detach().clone())
        value, description = self.datum(case)
        self.runtime.description = description
        with torch.no_grad():
            swapped, _ = self.reader(self.runtime, value, sampler_seed, target['future'].to('cuda'), checkpointed=False)
        self.count('reader_queries')
        delta = swapped.float().cpu() - target['action'].float()
        row = dict(binding, pbs_jobid=self.job, **self.metrics(action, future, target),
                   first10gripper_disagreement=float((actual[:10, 6] != teacher[:10, 6]).mean()),
                   first10motor_rmse_swap=float(delta[:10, :6].square().mean().sqrt()),
                   first10motor_mse_swap=float(delta[:10, :6].square().mean()),
                   full32motor_rmse_swap=float(delta[:, :6].square().mean().sqrt()))
        save(path, row)
        return row

    def evaluate(self, arm):
        out = self.results / 'heldout' / f'{arm}.json'
        if out.exists():
            return read(out)
        ev = self.cfg['evaluation']
        self.install(arm)
        records = []
        for case in ev['case_ids']:
            for seed in ev['seed_indices']:
                records.append(self.endpoint(arm, case, seed))
                self.progress('heldout', arm=arm, queries=len(records))
        result = dict(protocol=PROTOCOL, arm=arm, pbs_jobid=self.job,
                      **aggregate(records, ev['case_ids'], ev['seed_indices'], self.cfg['suites'],
                                  tuple(self.helpers.METRICS) + EXTRA))
        save(out, result)
        return result

    def diagnostics(self):
        rows = [read(self.results / 'stats' / f'{case}.json') for case in lp.CASES]

        def share(groups):
            total = sum(groups['video_steps']) + groups['video_prefill'] + sum(groups['action_steps']) + groups['other']
            total = max(total, 1e-30)
            return dict(video_steps=sum(groups['video_steps']) / total, video_prefill=groups['video_prefill'] / total,
                        action_steps=sum(groups['action_steps']) / total, other=groups['other'] / total,
                        video_step_profile=[v / total for v in groups['video_steps']],
                        action_step_profile=[v / total for v in groups['action_steps']])

        def combine(row):
            g = row['groups']
            heads = ('A', 'V') if 'V' in g else ('A', 'Az', 'K', 'Z')
            merged = dict(video_steps=[sum(g[h]['video_steps'][s] for h in heads) for s in range(10)],
                          video_prefill=sum(g[h]['video_prefill'] for h in heads),
                          action_steps=[sum(g[h]['action_steps'][s] for h in heads) for s in range(10)],
                          other=sum(g[h]['other'] for h in heads))
            return share(merged)

        def mean(values):
            keys = values[0].keys()
            return {k: ([sum(v[k][s] for v in values) / len(values) for s in range(len(values[0][k]))]
                        if isinstance(values[0][k], list) else sum(v[k] for v in values) / len(values)) for k in keys}

        combined = mean([combine(r) for r in rows])
        diag = [r for r in rows if r['separate']]
        heads = {}
        for r in diag:
            g = r['groups']
            tot = {h: sum(g[h]['video_steps']) + g[h]['video_prefill'] + sum(g[h]['action_steps']) + g[h]['other']
                   for h in ('A', 'Az', 'K', 'Z')}
            whole = max(sum(tot.values()), 1e-30)
            heads.setdefault('cases', []).append(r['case_id'])
            for h, v in tot.items():
                heads.setdefault(h, []).append(v / whole)
            heads.setdefault('A_via_future', []).append(tot['Az'] / max(tot['Az'] + tot['A'], 1e-30))
        head_share = {k: (sum(v) / len(v) if k != 'cases' else v) for k, v in heads.items()}
        parity = max(max(r['parity'].values()) for r in rows)
        worst_sig = max(r['calls']['signature_worst_relative'] for r in rows)
        return dict(combined_share=combined, diagnostic_head_share=head_share, teacher_parity_max_abs=parity,
                    reader_signature_worst_relative=worst_sig)

    def check(self):
        """Start of the job, before hours of probes, on GPU 0 (run.pbs checks the closed-loop glue on GPU 1):
        QTIP run here on the baseline H_U against the baseline's QTIP_W2 weight; one probe context and a three-layer
        replay with the signature check; held-out endpoints (BF16: swap = 0; baseline QTIP_W2 installed).
        Writes only results/check/."""
        self.results, out = self.base / 'check', {}

        def timed(label, fn):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            value = fn()
            torch.cuda.synchronize()
            out[label] = dict(seconds=time.monotonic() - begin, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            self.progress('check', step=label, step_seconds=out[label]['seconds'], peak_gb=out[label]['peak_gb'])
            return value

        impl, prep, base = self.qtip(), self.baseline / 'prep', self.arm_root('QTIP_U_W2')
        row = min((r for r in read(prep / 'modules.json') if r['shape'][0] % 128 == 0 and r['shape'][1] % 128 == 0),
                  key=lambda r: (r['shape'][0] * r['shape'][1], r['index']))
        i, name = row['index'], row['module']
        saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                 for k, path in (('master', prep / 'masters'), ('U', prep / 'hessians'), ('baseline', base / 'weights'))}
        torch.use_deterministic_algorithms(False)
        result = timed('qtip_one_layer', lambda: impl.quantize_layer(
            saved['master']['weight'].to('cuda', torch.float32), saved['U']['hessian'].to('cuda', torch.float32),
            self.host['host']['bits'], seed=self.host['host']['seed_base'] + i))
        torch.use_deterministic_algorithms(True)
        mine = result['dequant'].to(torch.bfloat16).cpu()
        out['qtip_one_layer'].update(module=name, shape=row['shape'], viterbi_update=result['config'].get('viterbi_update'),
                                     baseline_weight_equal=bool(torch.equal(mine, saved['baseline']['weight'])),
                                     baseline_weight_max_abs=max_abs(mine, saved['baseline']['weight']))
        del result, mine
        torch.cuda.empty_cache()

        timed('load', self.load)
        if not torch.equal(saved['master']['weight'], self.bank.modules[name].weight.detach().cpu()):
            raise RuntimeError(f'Baseline master differs from the runtime weight: {name}')
        diag = self.diagnostic_cases()
        case = next(c for c in lp.CASES if c not in diag)
        s = timed('probe_context', lambda: self.stats_context(case, False))
        out['probe_context'].update(case=case, parity=s['parity'], calls=s['calls'])
        record = torch.load(self.results / 'stats' / f'{case}.pt', map_location='cpu', weights_only=True)
        streams = [stream_for(n) for n in self.bank.names]
        group = [streams.index('video'), streams.index('action'), len(streams) - 1 - streams[::-1].index('video')]
        replay = Replay(group, self.bank.by_index, defaultdict(lambda: 1.0), self.cfg['hessian']['gamma'], check=True)
        self.bank.mode, self.bank.replay = 'replay', replay

        def run_replay():
            with torch.no_grad():
                replay.start(record['u'], record['sig'])
                self.query(case)
                replay.finish(self.cfg['gates']['signature_tolerance'])
        timed('replay', run_replay)
        self.bank.mode, self.bank.replay = 'plain', None
        out['replay'].update(group=[self.bank.names[k] for k in group], signature_worst=replay.worst,
                             hessian_finite=all(bool(torch.isfinite(replay.H[k]).all()) for k in group))
        del replay, record
        torch.cuda.empty_cache()

        held, keys = self.cfg['evaluation']['case_ids'][0], ('first10motor_rmse', 'future_latent_nmse',
                                                             'first10motor_rmse_swap', 'first10gripper_disagreement')
        r = timed('endpoint_bf16', lambda: self.endpoint('BF16_CHECK', held, 0))
        out['endpoint_bf16'].update({k: r[k] for k in keys})
        timed('install_qtip_u_w2', lambda: self.install('QTIP_U_W2'))
        r = timed('endpoint_qtip_u_w2', lambda: self.endpoint('QTIP_U_W2_CHECK', held, 0))
        out['endpoint_qtip_u_w2'].update({k: r[k] for k in keys})
        save(self.results / 'summary.json', dict(protocol=PROTOCOL, pbs_jobid=self.job, complete=True, **out))

    def final(self):
        """Diagnostics, QTIP receipts and the report from the files the earlier steps wrote; no model."""
        diagnostics = self.diagnostics()
        save(self.results / 'diagnostics.json', dict(protocol=PROTOCOL, pbs_jobid=self.job, **diagnostics))
        quant = self.quant_summary()
        heldout = {arm: read(self.results / 'heldout' / f'{arm}.json')['overall'] for arm in self.host['arms']}
        summary = dict(protocol=PROTOCOL, pbs_jobid=self.job, status='complete', complete=True,
                       diagnostics=diagnostics, quant=quant, heldout=heldout,
                       new_teacher_queries=0, training_updates=0)
        save(self.results / 'summary.json', summary)
        (self.results / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')

    # ---------------- Phase 2: rollout-coherent Gauss-Newton (sketched weight gradients) ----------------
    def coherent_context(self, case, separate, root, check=()):
        """Probes on one C80 context; per probe and layer the sketch S = R^T G of the probe-loss weight gradient, with G
        summed over every call of the shared weight (10 video steps, prefill, 10 action steps). Variants: C = A + lambda Z
        + beta K, CA = A alone (its direct path and its path through the imagined future).

        Passes per probe: reader v_A (head A: prefill/action/proprio calls that produce the executed action; gives c1
        at z), reader v_K (c2 at z only: K reads with the frozen BF16 reader), query from z with c1 (head Az) and with
        sqrt(beta) c2 + sqrt(lambda) v_Z (head R), or on the diagnostic contexts each alone (heads K, Z).
        check: layer indices whose autograd weight gradient is compared with the sketch (start-of-job check only)."""
        b, loss, coh = self.bank, self.cfg['loss'], self.host['coherent']
        weights = [b.by_index[i].weight for i in check]
        for w in weights:
            w.requires_grad_(True)
        target = self.teacher(case, 0)
        b.begin_probe()
        action_q, future, _ = self.query(case, 0, grad=True)
        del action_q
        b.phase = 'r'
        value, description = self.datum(case)
        self.runtime.description = description
        leaf = future.detach().float().requires_grad_(True)
        action, _ = self.reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], leaf, checkpointed=True)
        self.count('reader_queries')
        calls = b.finish_forward(self.cfg['gates']['signature_tolerance'])
        parity = dict(future_max_abs=max_abs(future, target['future']), action_max_abs=max_abs(action, target['action']))
        if max(parity.values()) > self.cfg['gates']['teacher_parity_max_abs']:
            raise RuntimeError(f'BF16 query/reader differ from the cached teacher: {parity}')

        stats = self.p['reader_source']['action_stats']
        curv = pr.softbin_kl_curvature(softbin_loss, target['action'][:10, :6], stats['mean'], stats['scale'],
                                       self.p['reader_source']['config']).to('cuda')
        znorm = float(target['future'][:, :, 1:].float().norm())
        scales = pr.head_scales(loss['lambda'], loss['beta'])
        gen = torch.Generator(device='cuda').manual_seed(coh['seed'] + case)
        heads = ('A', 'Az', 'K', 'Z') if separate else ('A', 'Az', 'R')
        acc = {h: dict(u={}, d={}) for h in heads}
        probes, rows, cols = coh['probes_per_context'], coh['sketch_rows'], coh['row_sketch_cols']
        streams = [stream_for(n) for n in b.names]
        stacks = {v: [[] for _ in range(614)] for v in ('C', 'CA')}
        rowsq = {v: [torch.zeros(m.out_features, device='cuda') for m in b.by_index] for v in ('C', 'CA')}
        mass = {k: defaultdict(float) for k in heads + ('C', 'CA')}
        sketch_vs_autograd = {}
        for p in range(probes):
            last = p == probes - 1
            b.R = [pr.sketch_matrix(m.out_features, rows, gen, 'cuda') for m in b.by_index]
            b.Q = [pr.sketch_matrix(m.in_features, cols, gen, 'cuda') for m in b.by_index]
            ga = torch.zeros_like(action)
            ga[:10, :6] = pr.rademacher((10, 6), gen, 'cuda') / math.sqrt(60)
            gk = torch.zeros_like(action)
            gk[:10, :6] = pr.rademacher((10, 6), gen, 'cuda') * (curv / 2).sqrt()
            vz = torch.zeros(future.shape, device='cuda')
            vz[:, :, 1:] = pr.rademacher(vz[:, :, 1:].shape, gen, 'cuda') / znorm
            S, T, wg = {h: {} for h in heads}, {h: {} for h in heads}, {}
            b.sink = dict(acc['A'], S=S['A'], T=T['A'])
            got = torch.autograd.grad(action, [leaf, b.anchor, *weights], ga, retain_graph=True, allow_unused=True)
            c1, wg['A'] = got[0], got[2:]
            b.sink = None
            c2, = torch.autograd.grad(action, [leaf], gk, retain_graph=not last)
            self.counters['parameter_reverse_passes'] += 2
            parts = ([('Az', c1), ('K', scales['K'] * c2), ('Z', scales['Z'] * vz)] if separate
                     else [('Az', c1), ('R', scales['K'] * c2 + scales['Z'] * vz)])
            for k, (head, c) in enumerate(parts):
                b.sink = dict(acc[head], S=S[head], T=T[head])
                got = torch.autograd.grad(future, [b.anchor, *weights], c.to(future.dtype),
                                          retain_graph=not (last and k == len(parts) - 1), allow_unused=True)
                wg[head] = got[1:]
                b.sink = None
                self.counters['parameter_reverse_passes'] += 1
            del c1, c2, parts

            def variants(parts):
                ca = parts['A'] + parts['Az']
                return ca, ca + sum((parts[h] for h in heads if h not in ('A', 'Az')), torch.zeros_like(ca))
            for i, m in enumerate(b.by_index):
                zero, tzero = torch.zeros(rows, m.in_features, device='cuda'), torch.zeros(m.out_features, cols, device='cuda')
                part = {h: S[h].get(i, zero) for h in heads}
                ca, full = variants(part)
                for h, t in list(part.items()) + [('CA', ca), ('C', full)]:
                    mass[h][streams[i]] += float(t.square().sum()) / probes
                stacks['CA'][i].append(ca.to(torch.bfloat16).cpu())
                stacks['C'][i].append(full.to(torch.bfloat16).cpu())
                tca, tfull = variants({h: T[h].get(i, tzero) for h in heads})
                rowsq['CA'][i] += tca.square().sum(1) / probes
                rowsq['C'][i] += tfull.square().sum(1) / probes
            for j, i in enumerate(check):
                for h in heads:
                    grad = wg[h][j]
                    m = b.by_index[i]
                    pairs = (('S', S[h].get(i), torch.zeros(rows, m.in_features, device='cuda') if grad is None else
                              b.R[i].float().t() @ grad.float() / math.sqrt(rows)),
                             ('T', T[h].get(i), torch.zeros(m.out_features, cols, device='cuda') if grad is None else
                              grad.float() @ b.Q[i].float() / math.sqrt(cols)))
                    for kind, have, want in pairs:
                        have = torch.zeros_like(want) if have is None else have
                        scale = max(float(want.norm()), float(have.norm()), 1e-30)
                        key = f'{b.names[i]}|{h}|{kind}'
                        sketch_vs_autograd[key] = max(sketch_vs_autograd.get(key, 0.0),
                                                      float((want - have).norm()) / scale)
            del S, T, wg
        for w in weights:
            w.requires_grad_(False)

        diag = {}
        for head, s in acc.items():
            per = defaultdict(float)
            for (idx, n), value in s['u'].items():
                per[(idx, n)] += float((value * b.xn[idx][n]).sum()) / probes
            diag[head] = per
        groups = self.group_masses(diag)

        def diag_rows(names):  # the per-token row weight sum_t g_tc^2 |x_t|^2 of the variant (phase-1 definition)
            return {i: sum(acc[h]['d'][i] for h in names if i in acc[h]['d']).float().div(probes).cpu()
                    if any(i in acc[h]['d'] for h in names) else torch.zeros(b.by_index[i].out_features)
                    for i in range(614)}
        payload = {v: {i: torch.cat(stacks[v][i]) for i in range(614)} for v in stacks}
        payload.update({f'rows_{v}': {i: rowsq[v][i].cpu() for i in range(614)} for v in rowsq})
        payload.update(diag_rows_CA=diag_rows(('A', 'Az')), diag_rows_C=diag_rows(heads))
        self.tsave(root / 'sketches' / f'{case}.pt', payload)
        summary = dict(protocol=PROTOCOL, case_id=case, separate=separate, probes=probes, sketch_rows=rows,
                       parity=parity, calls=calls, znorm=znorm, pbs_jobid=self.job,
                       coherent={k: sum(v.values()) for k, v in mass.items()},
                       coherent_by_stream={k: dict(v) for k, v in mass.items()},
                       diag={h: sum(g['video_steps']) + g['video_prefill'] + sum(g['action_steps']) + g['other']
                             for h, g in groups.items()},
                       diag_groups=groups, sketch_vs_autograd=sketch_vs_autograd)
        save(root / 'stats' / f'{case}.json', summary)
        b.end_probe()
        del action, future, leaf, stacks, acc
        torch.cuda.empty_cache()
        return summary

    def sketches(self, shard=0, shards=1):
        root, diag = self.base / 'coherent', set(self.diagnostic_cases())
        # A diagnostic context back-propagates the video loop three times per probe instead of twice.
        mine = owned(shard, shards, lambda c: 1.5 if c in diag else 1.0, lp.CASES)
        for count, case in enumerate(mine):
            if not (root / 'stats' / f'{case}.json').exists():
                self.enough()
                self.coherent_context(case, case in diag, root)
            self.progress('sketch', contexts=count + 1, of=len(mine))

    def gate(self):
        """Pre-registered: the coherent model must attribute at least gate.min of the executed-action error to the
        imagined future (the swap measured 44-50% in MSE; the per-token model 6.9%); otherwise no quantization."""
        root, g = self.base / 'coherent', self.host['coherent']['gate']
        rows = [read(root / 'stats' / f'{c}.json') for c in lp.CASES]
        a, ca = sum(r['coherent']['A'] for r in rows), sum(r['coherent']['CA'] for r in rows)
        az = sum(r['coherent']['Az'] for r in rows)
        da, daz = sum(r['diag']['A'] for r in rows), sum(r['diag']['Az'] for r in rows)
        value = dict(protocol=PROTOCOL, pbs_jobid=self.job, contexts=len(rows), threshold=g['min_a_via_future'],
                     coherent_a_via_future=1 - a / ca, coherent_az_share=az / (a + az),
                     diag_a_via_future=daz / (da + daz))
        value['passed'] = value['coherent_a_via_future'] >= g['min_a_via_future']
        save(root / 'gate.json', value)
        print('GATE ' + json.dumps(value), flush=True)

    def quantize_coherent(self, shard=0, shards=1):
        """QTIP W2 on H_C and H_CA for this shard's layers (same seeds as the baseline); per layer the proxies of all four
        QTIP W2 weights (baseline H_U, H~, H_C, H_CA) under all four Hessians."""
        impl, prep, coh = self.qtip(), self.baseline / 'prep', self.host['coherent']
        bits, arms = self.host['host']['bits'], coh['arms']
        files = [self.base / 'coherent' / 'sketches' / f'{c}.pt' for c in lp.CASES]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise RuntimeError(f'Sketches missing for {len(missing)} contexts, e.g. {missing[:4]}')
        base = self.arm_root('QTIP_U_W2')
        rows = read(prep / 'modules.json')
        spent = [read(base / 'stats' / f'{i:04d}.json')['seconds'] for i in range(614)]
        mine = owned(shard, shards, spent.__getitem__)
        todo = [i for i in mine if any(not (self.arm_root(a) / 'stats' / f'{i:04d}.json').exists() for a in arms)]
        sketches = [torch.load(f, map_location='cpu', mmap=True, weights_only=True) for f in files]
        sources = (('master', prep / 'masters'), ('U', prep / 'hessians'), ('G', self.results / 'hessians'),
                   ('QTIP_U_W2', base / 'weights'), (GUIDED, self.arm_root(GUIDED) / 'weights'))
        torch.use_deterministic_algorithms(False)
        try:
            for done, i in enumerate(todo):
                self.enough()
                name = rows[i]['module']
                saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True) for k, path in sources}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                w = saved['master']['weight'].to('cuda', torch.float32)
                h = {k: saved[k]['hessian'].to('cuda', torch.float32) for k in ('U', 'G')}
                for v in arms.values():
                    stack = torch.cat([s[v][i] for s in sketches]).to('cuda')
                    h[v] = pr.coherent_hessian(stack, h['U'], coh['gamma_u'])
                    del stack
                q = {k: saved[k]['weight'].to('cuda', torch.float32) for k in ('QTIP_U_W2', GUIDED)}
                half = len(sketches) // 2
                halves = {v: (sum(s[f'rows_{v}'][i] for s in sketches[:half]).to('cuda'),
                              sum(s[f'rows_{v}'][i] for s in sketches[half:]).to('cuda')) for v in arms.values()}
                out = {}
                for arm, v in arms.items():
                    rows_info, scale = {}, None
                    if coh['row_scale']:
                        # Coherent row weights E|G_c|^2 of this variant; the scale folds into QTIP's fp32 SV (0 bits).
                        d1, d2 = halves[v]
                        dd = sum(s[f'diag_rows_{v}'][i] for s in sketches).to('cuda')
                        lim = coh['row_scale_limit']
                        scale = pr.row_scale(d1 + d2, w, lim)
                        rows_info = dict(
                            row_ratio_in_sample=pr.row_ratio(d1 + d2, scale, w),
                            row_ratio_split=0.5 * (pr.row_ratio(d2, pr.row_scale(d1, w, lim), w)
                                                   + pr.row_ratio(d1, pr.row_scale(d2, w, lim), w)),
                            row_weight_half_corr=float(torch.corrcoef(torch.stack((d1, d2)).double())[0, 1]),
                            row_weight_corr_diag=float(torch.corrcoef(torch.stack((d1 + d2, dd)).double())[0, 1]),
                            row_scale_range=[float(scale.min()), float(scale.max())])
                    begin = time.monotonic()
                    target = w if scale is None else scale[:, None] * w
                    result = impl.quantize_layer(target, h[v], bits, seed=self.host['host']['seed_base'] + i)
                    dq = result['dequant'].to('cuda', torch.float32)
                    if scale is not None:
                        dq = dq / scale[:, None]
                    if tuple(dq.shape) != tuple(w.shape) or not bool(torch.isfinite(dq).all()):
                        raise RuntimeError(f'QTIP produced an invalid weight at {name} ({arm})')
                    q[arm], out[arm] = dq, (result, time.monotonic() - begin, rows_info)

                def proxy(x, hv, rw=None):  # rw: output-row weights (the coherent E|G_c|^2), else rows count equally
                    d = x - w
                    num, den = ((d @ hv) * d).sum(1), ((w @ hv) * w).sum(1)
                    if rw is not None:
                        num, den = num * rw, den * rw
                    return float(num.sum() / den.sum().clamp_min(1e-30))
                matrix = {a: {k: proxy(x, hv) for k, hv in h.items()} for a, x in q.items()}
                for a, x in q.items():  # the coherent model's full quadratic: input-side H_v and row weights of v
                    matrix[a].update({f'{v}_rows': proxy(x, h[v], sum(halves[v])) for v in arms.values()})
                for arm, v in arms.items():
                    result, seconds, rows_info = out[arm]
                    self.tsave(self.arm_root(arm) / 'weights' / f'{i:04d}.pt',
                               dict(module=name, weight=q[arm].to(torch.bfloat16).cpu()))
                    save(self.arm_root(arm) / 'stats' / f'{i:04d}.json', dict(
                        index=i, module=name, shape=list(w.shape), bits_total=result['bits_total'],
                        bpw=result['bits_total'] / w.numel(), seconds=seconds, config=result['config'], hessian=v,
                        row_scale=bool(coh['row_scale']), proxy_U=matrix[arm]['U'], proxies=matrix, **rows_info))
                del saved, w, h, q, out
                torch.cuda.empty_cache()
                if done % 25 == 0:
                    self.progress('quant', layers=done + 1, of=len(todo))
        finally:
            torch.use_deterministic_algorithms(True)

    def coherent_quant_summary(self):
        """quantized/<arm>/COMPLETE.json for the coherent arms and the parameter-weighted proxy matrix."""
        arms = self.host['coherent']['arms']
        stats = {a: [read(self.arm_root(a) / 'stats' / f'{i:04d}.json') for i in range(614)] for a in arms}
        first = next(iter(stats.values()))
        n = sum(s['shape'][0] * s['shape'][1] for s in first)
        matrix = {a: {k: sum(s['proxies'][a][k] * s['shape'][0] * s['shape'][1] for s in first) / n
                      for k in first[0]['proxies'][a]} for a in first[0]['proxies']}
        out = {}
        for arm, v in arms.items():
            s = stats[arm]
            row = dict(arm=arm, method=self.host['host']['method'], bits=self.host['host']['bits'], hessian=v,
                       bpw=sum(x['bits_total'] for x in s) / n, proxy_nmse_weighted=matrix[arm]['U'],
                       quantize_seconds=sum(x['seconds'] for x in s), row_scale=s[0]['row_scale'])
            if row['row_scale']:  # parameter-weighted, as the proxies
                row.update({k: sum(x[k] * x['shape'][0] * x['shape'][1] for x in s) / n
                            for k in ('row_ratio_in_sample', 'row_ratio_split', 'row_weight_half_corr',
                                      'row_weight_corr_diag')})
            save(self.arm_root(arm) / 'COMPLETE.json', dict(protocol=PROTOCOL, complete=True, pbs_jobid=self.job,
                                                            target_weights=n, config_example=s[0]['config'], **row))
            out[arm] = row
        return dict(arms=out, proxy_matrix=matrix)

    def coherent_diagnostics(self):
        root = self.base / 'coherent'
        rows = [read(root / 'stats' / f'{c}.json') for c in lp.CASES]
        diag = [r for r in rows if r['separate']]

        def total(rs, src, keys):
            return {k: sum(r[src][k] for r in rs) for k in keys}
        heads = ('A', 'Az', 'K', 'Z')
        coh, per = total(diag, 'coherent', heads), total(diag, 'diag', heads)
        streams = ('video', 'action', 'other')
        by_stream = {v: {s: sum(r['coherent_by_stream'][v].get(s, 0.0) for r in rows) for s in streams}
                     for v in ('C', 'CA')}
        worst = max(read(root / 'check' / 'summary.json')['sketch_context']['sketch_vs_autograd'].values())
        return dict(head_share_coherent={h: coh[h] / sum(coh.values()) for h in heads},
                    head_share_diag={h: per[h] / sum(per.values()) for h in heads},
                    stream_share={v: {s: x / max(sum(d.values()), 1e-30) for s, x in d.items()} for v, d in by_stream.items()},
                    teacher_parity_max_abs=max(max(r['parity'].values()) for r in rows),
                    reader_signature_worst_relative=max(r['calls']['signature_worst_relative'] for r in rows),
                    sketch_vs_autograd_worst=worst, diagnostic_cases=[r['case_id'] for r in diag])

    def loops(self):
        """Closed-loop totals and paired counts against the baseline QTIP_W2 and the phase-1 QTIP_G_W2."""
        def episodes(root):
            return {int(f.stem): read(f)['success'] for f in (root / 'episodes').glob('*.json')}
        roots = {'QTIP_U_W2': self.baseline / 'loops' / self.host['host']['baseline_arm'],
                 GUIDED: self.base / 'loops' / GUIDED}
        roots.update({a: self.base / 'loops' / a for a in self.host['coherent']['arms']})
        found = {a: episodes(r) for a, r in roots.items() if (r / 'DONE').exists()}
        out = {}
        for arm, e in found.items():
            row = dict(successes=sum(e.values()), episodes=len(e))
            for ref in ('QTIP_U_W2', GUIDED):
                if ref in found and ref != arm:
                    r = found[ref]
                    wins = sum(e[i] and not r[i] for i in e if i in r)
                    losses = sum(r[i] and not e[i] for i in e if i in r)
                    row[f'vs_{ref}'] = dict(wins=wins, losses=losses, p=mcnemar(wins, losses))
            out[arm] = row
        return out

    def final_coherent(self):
        root = self.base / 'coherent'
        gate = read(root / 'gate.json')
        summary = dict(protocol=PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, gate=gate,
                       diagnostics=self.coherent_diagnostics(), new_teacher_queries=0, training_updates=0)
        if gate['passed']:
            arms = ['QTIP_U_W2', GUIDED] + list(self.host['coherent']['arms'])
            summary.update(quant=self.coherent_quant_summary(),
                           heldout={a: read(self.results / 'heldout' / f'{a}.json')['overall'] for a in arms},
                           loops=self.loops())
        save(root / 'summary.json', summary)
        (root / 'RESULTS.zh.md').write_text(coherent_report(summary, self.host), encoding='utf-8')

    def check_coherent(self):
        """Start of the job on GPU 0 (run.pbs checks the closed-loop glue on GPU 1): one diagnostic context with the
        sketch compared against autograd weight gradients of a video and an action layer, the sketch file read back
        (mmap) into H_C for the smallest layer, QTIP W2 on it. Writes only results/coherent/check/."""
        root, out = self.base / 'coherent' / 'check', {}

        def timed(label, fn):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            value = fn()
            torch.cuda.synchronize()
            out[label] = dict(seconds=time.monotonic() - begin, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            self.progress('check', step=label, step_seconds=out[label]['seconds'], peak_gb=out[label]['peak_gb'])
            return value

        timed('load', self.load)
        names, streams = self.bank.names, [stream_for(n) for n in self.bank.names]

        def smallest(stream):  # a per-token Linear inside a block when there is one
            pick = ([i for i, n in enumerate(names) if streams[i] == stream and '.blocks.' in n]
                    or [i for i in range(len(names)) if streams[i] == stream])
            return min(pick, key=lambda i: (self.bank.by_index[i].weight.numel(), i))
        check = [smallest('video'), smallest('action')]
        case = self.diagnostic_cases()[0]
        s = timed('sketch_context', lambda: self.coherent_context(case, True, root, check=check))
        worst = max(s['sketch_vs_autograd'].values())
        out['sketch_context'].update(case=case, parity=s['parity'], calls=s['calls'], coherent=s['coherent'],
                                     diag=s['diag'], sketch_vs_autograd=s['sketch_vs_autograd'], layers=[names[i] for i in check])
        if worst > self.host['coherent']['check_tolerance']:
            raise RuntimeError(f'Sketch differs from the autograd weight gradient (relative {worst:.3g})')

        impl, prep = self.qtip(), self.baseline / 'prep'
        row = min((r for r in read(prep / 'modules.json') if r['shape'][0] % 128 == 0 and r['shape'][1] % 128 == 0),
                  key=lambda r: (r['shape'][0] * r['shape'][1], r['index']))
        i = row['index']
        sk = torch.load(root / 'sketches' / f'{case}.pt', map_location='cpu', mmap=True, weights_only=True)
        p = self.host['coherent']
        shapes_ok = all(tuple(sk[v][k].shape) == (p['probes_per_context'] * p['sketch_rows'], self.bank.by_index[k].in_features)
                        and tuple(sk[f'rows_{v}'][k].shape) == (self.bank.by_index[k].out_features,)
                        and tuple(sk[f'diag_rows_{v}'][k].shape) == (self.bank.by_index[k].out_features,)
                        for v in ('C', 'CA') for k in range(614))
        saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                 for k, path in (('master', prep / 'masters'), ('U', prep / 'hessians'))}
        w = saved['master']['weight'].to('cuda', torch.float32)
        h = pr.coherent_hessian(sk['C'][i].to('cuda'), saved['U']['hessian'].to('cuda', torch.float32), p['gamma_u'])
        scale = pr.row_scale(sk['rows_C'][i].to('cuda'), w, p['row_scale_limit'])
        torch.use_deterministic_algorithms(False)
        result = timed('qtip_one_layer', lambda: impl.quantize_layer(
            scale[:, None] * w, h, self.host['host']['bits'], seed=self.host['host']['seed_base'] + i))
        torch.use_deterministic_algorithms(True)
        dq = result['dequant'].to('cuda') / scale[:, None]
        out['qtip_one_layer'].update(module=row['module'], shape=row['shape'], sketch_shapes_ok=shapes_ok,
                                     hessian_finite=bool(torch.isfinite(h).all()),
                                     row_scale_range=[float(scale.min()), float(scale.max())],
                                     row_ratio=pr.row_ratio(sk['rows_C'][i].to('cuda'), scale, w),
                                     dequant_finite=bool(torch.isfinite(dq).all()))
        if not (shapes_ok and out['qtip_one_layer']['hessian_finite'] and out['qtip_one_layer']['dequant_finite']):
            raise RuntimeError(f"Coherent Hessian/QTIP check failed: {out['qtip_one_layer']}")
        save(root / 'summary.json', dict(protocol=PROTOCOL, pbs_jobid=self.job, complete=True, **out))

    # ---------------- Phase 3: the phase-1 H~ on more VQ hosts (GPTVQ, QuIP#) ----------------
    def quantize_hosts(self, shard=0, shards=1):
        """Each host of host.json 'hosts' on H~ (results/hessians) with the baseline's seed per layer; proxies of ours and
        of the baseline arm's weight under H_U and H~. Layers are split on what the baselines spent on them."""
        prep, arms = self.baseline / 'prep', self.host['hosts']['arms']
        if any(not (self.results / 'hessians' / f'{i:04d}.pt').exists() for i in range(614)):
            raise RuntimeError('Phase-1 guided Hessians are incomplete')
        for c in arms.values():
            if not read(self.arm_root(c['baseline_arm']) / 'COMPLETE.json').get('complete'):
                raise RuntimeError(f"Baseline {c['baseline_arm']} is incomplete")
        rows = read(prep / 'modules.json')
        spent = [sum(read(self.arm_root(c['baseline_arm']) / 'stats' / f'{i:04d}.json')['seconds'] for c in arms.values())
                 for i in range(614)]
        mine = owned(shard, shards, spent.__getitem__)
        todo = [i for i in mine if any(not (self.arm_root(a) / 'stats' / f'{i:04d}.json').exists() for a in arms)]
        torch.use_deterministic_algorithms(False)  # as the baselines' quant_worker
        try:
            for done, i in enumerate(todo):
                self.enough()
                name = rows[i]['module']
                saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                         for k, path in (('master', prep / 'masters'), ('U', prep / 'hessians'),
                                         ('G', self.results / 'hessians'))}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                w = saved['master']['weight'].to('cuda', torch.float32)
                h = {k: saved[k]['hessian'].to('cuda', torch.float32) for k in ('U', 'G')}

                def proxy(q):
                    d = q - w
                    return {k: float(((d @ v) * d).sum() / ((w @ v) * w).sum().clamp_min(1e-30)) for k, v in h.items()}
                for arm, c in arms.items():
                    out = self.arm_root(arm)
                    if (out / 'stats' / f'{i:04d}.json').exists():
                        continue
                    base = torch.load(self.arm_root(c['baseline_arm']) / 'weights' / f'{i:04d}.pt', map_location='cpu',
                                      weights_only=True)
                    if base['module'] != name:
                        raise RuntimeError(f"Baseline weight identity differs at {i}: {name}")
                    begin = time.monotonic()
                    result = self.method(c['method']).quantize_layer(w, h['G'], c['bits'],
                                                                     seed=self.host['host']['seed_base'] + i)
                    seconds = time.monotonic() - begin
                    dq = result['dequant'].to('cuda', torch.float32)
                    if tuple(dq.shape) != tuple(w.shape) or not bool(torch.isfinite(dq).all()):
                        raise RuntimeError(f'{arm} produced an invalid weight at {name}')
                    ours, theirs = proxy(dq), proxy(base['weight'].to('cuda', torch.float32))
                    self.tsave(out / 'weights' / f'{i:04d}.pt', dict(module=name, weight=dq.to(torch.bfloat16).cpu()))
                    save(out / 'stats' / f'{i:04d}.json', dict(
                        index=i, module=name, shape=list(w.shape), bits_total=result['bits_total'],
                        bpw=result['bits_total'] / w.numel(), seconds=seconds, config=result['config'], hessian='G',
                        proxy_U=ours['U'], proxy_G=ours['G'], baseline_proxy_U=theirs['U'], baseline_proxy_G=theirs['G']))
                    del base, result, dq
                del saved, w, h
                torch.cuda.empty_cache()
                if done % 25 == 0:
                    self.progress('quant', layers=done + 1, of=len(todo))
        finally:
            torch.use_deterministic_algorithms(True)

    def hosts_summary(self):
        """quantized/<arm>/COMPLETE.json of the host arms (what the baselines' analysis.py reads) and both arms' proxies."""
        out = {}
        for arm, c in self.host['hosts']['arms'].items():
            stats = [read(self.arm_root(arm) / 'stats' / f'{i:04d}.json') for i in range(614)]
            n = sum(s['shape'][0] * s['shape'][1] for s in stats)

            def weighted(key):
                return sum(s[key] * s['shape'][0] * s['shape'][1] for s in stats) / n
            row = dict(arm=arm, method=c['method'], bits=c['bits'], hessian='G', bpw=sum(s['bits_total'] for s in stats) / n,
                       proxy_nmse_weighted=weighted('proxy_U'), proxy_U_weighted=weighted('proxy_U'),
                       proxy_G_weighted=weighted('proxy_G'), quantize_seconds=sum(s['seconds'] for s in stats))
            save(self.arm_root(arm) / 'COMPLETE.json', dict(protocol=PROTOCOL, complete=True, pbs_jobid=self.job,
                                                            target_weights=n, config_example=stats[0]['config'], **row))
            base = read(self.arm_root(c['baseline_arm']) / 'COMPLETE.json')
            out[arm] = dict(guided=row, baseline=dict(
                arm=c['baseline_arm'], bpw=base['bpw'], proxy_U_weighted=weighted('baseline_proxy_U'),
                proxy_G_weighted=weighted('baseline_proxy_G'), baseline_reported_proxy=base['proxy_nmse_weighted']))
        return out

    def check_hosts(self):
        """Start of the job on GPU 0 (run.pbs checks the closed-loop glue on GPU 1): per host the smallest layer on the
        baseline H_U against the baseline's stored weight (same code and seed), then on H~; the model loaded and one
        held-out endpoint of a baseline host arm installed from the baseline root. Writes only results/hosts/check/."""
        self.results, out = self.base / 'hosts' / 'check', {}

        def timed(label, fn):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            value = fn()
            torch.cuda.synchronize()
            out[label] = dict(seconds=time.monotonic() - begin, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            self.progress('check', step=label, step_seconds=out[label]['seconds'], peak_gb=out[label]['peak_gb'])
            return value

        prep = self.baseline / 'prep'
        row = min((r for r in read(prep / 'modules.json') if r['shape'][0] % 512 == 0 and r['shape'][1] % 256 == 0),
                  key=lambda r: (r['shape'][0] * r['shape'][1], r['index']))
        i, name = row['index'], row['module']
        saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                 for k, path in (('master', prep / 'masters'), ('U', prep / 'hessians'),
                                 ('G', self.base / 'hessians'))}
        w = saved['master']['weight'].to('cuda', torch.float32)
        torch.use_deterministic_algorithms(False)
        for arm, c in self.host['hosts']['arms'].items():
            base = torch.load(self.arm_root(c['baseline_arm']) / 'weights' / f'{i:04d}.pt', map_location='cpu',
                              weights_only=True)['weight']
            impl, seed = self.method(c['method']), self.host['host']['seed_base'] + i
            u = timed(f'{arm}_on_H_U', lambda: impl.quantize_layer(w, saved['U']['hessian'].to('cuda', torch.float32),
                                                                    c['bits'], seed=seed))
            mine = u['dequant'].to(torch.bfloat16).cpu()
            rel = float((mine.float() - base.float()).norm() / base.float().norm().clamp_min(1e-30))
            out[f'{arm}_on_H_U'].update(module=name, shape=row['shape'], baseline_weight_equal=bool(torch.equal(mine, base)),
                                        baseline_weight_relative=rel)
            if rel > self.host['hosts']['reproduction_tolerance']:
                raise RuntimeError(f'{arm}: the H_U run differs from the baseline weight (relative {rel:.3g})')
            g = timed(f'{arm}_on_H_G', lambda: impl.quantize_layer(w, saved['G']['hessian'].to('cuda', torch.float32),
                                                                    c['bits'], seed=seed))
            out[f'{arm}_on_H_G'].update(dequant_finite=bool(torch.isfinite(g['dequant']).all()),
                                        bpw=g['bits_total'] / w.numel())
            if not out[f'{arm}_on_H_G']['dequant_finite']:
                raise RuntimeError(f'{arm}: non-finite weight on H~')
            del u, g, mine, base
            torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(True)
        del w, saved
        timed('load', self.load)
        arm = next(iter(self.host['hosts']['arms'].values()))['baseline_arm']
        timed('install', lambda: self.install(arm))
        r = timed('endpoint', lambda: self.endpoint(f'{arm}_CHECK', self.cfg['evaluation']['case_ids'][0], 0))
        out['endpoint'].update(arm=arm, **{k: r[k] for k in ('first10motor_rmse', 'future_latent_nmse',
                                                              'first10motor_rmse_swap', 'first10gripper_disagreement')})
        save(self.results / 'summary.json', dict(protocol=PROTOCOL, pbs_jobid=self.job, complete=True, **out))

    def final_hosts(self):
        """Per host (QTIP from phases 0/1, GPTVQ and QuIP# from this phase): bpw, proxies, held-out and closed loop of the
        baseline (H_U) and the guided (H~) arm, paired episode counts with McNemar p."""
        def episodes(root):
            return {int(f.stem): read(f)['success'] for f in (root / 'episodes').glob('*.json')} if (root / 'DONE').exists() else None

        p1 = read(self.base / 'summary.json')['quant']  # phase 1: QTIP_U_W2 and QTIP_G_W2 proxies
        quant = self.hosts_summary()
        table = {'qtip': dict(baseline='QTIP_U_W2', guided=GUIDED, proxies={
            'QTIP_U_W2': (p1['QTIP_U_W2']['proxy_U_weighted'], p1['QTIP_U_W2']['proxy_G_weighted'], p1['QTIP_U_W2']['bpw']),
            GUIDED: (p1[GUIDED]['proxy_U_weighted'], p1[GUIDED]['proxy_G_weighted'], p1[GUIDED]['bpw'])},
            loops=(self.baseline / 'loops' / self.host['host']['baseline_arm'], self.base / 'loops' / GUIDED))}
        for arm, c in self.host['hosts']['arms'].items():
            q = quant[arm]
            table[c['method']] = dict(baseline=c['baseline_arm'], guided=arm, proxies={
                c['baseline_arm']: (q['baseline']['proxy_U_weighted'], q['baseline']['proxy_G_weighted'], q['baseline']['bpw']),
                arm: (q['guided']['proxy_U_weighted'], q['guided']['proxy_G_weighted'], q['guided']['bpw'])},
                loops=(self.baseline / 'loops' / c['baseline_arm'], self.base / 'loops' / arm))
        hosts = {}
        for method, t in table.items():
            row = dict(baseline=t['baseline'], guided=t['guided'])
            for role in ('baseline', 'guided'):
                a = t[role]
                pu, pg, bpw = t['proxies'][a]
                held = self.results / 'heldout' / f'{a}.json'
                row[role] = dict(arm=a, bpw=bpw, proxy_U=pu, proxy_G=pg,
                                 heldout=read(held)['overall'] if held.exists() else None)
            e = [episodes(r) for r in t['loops']]
            if all(x is not None for x in e):
                b, g = e
                wins = sum(g[k] and not b[k] for k in g if k in b)
                losses = sum(b[k] and not g[k] for k in g if k in b)
                row['loop'] = dict(baseline=sum(b.values()), guided=sum(g.values()), episodes=len(g),
                                   wins=wins, losses=losses, p=mcnemar(wins, losses))
            hosts[method] = row
        summary = dict(protocol=PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, hosts=hosts,
                       hessian='phase-1 H~ (token weights of the A + lambda Z + beta K probes, gamma 0.1), unchanged',
                       new_teacher_queries=0, training_updates=0)
        save(self.base / 'hosts' / 'summary.json', summary)
        (self.base / 'hosts' / 'RESULTS.zh.md').write_text(hosts_report(summary), encoding='utf-8')

    def stage(self, name, shard, shards):
        """Phase 3 steps (phase 1, job 26043905: stats/hessians/quantize/check/final; phase 2, job 26046240:
        check_coherent/sketches/gate/quantize_coherent/final_coherent; both above)."""
        evals = self.host['hosts']['eval_arms']
        if name == 'eval':
            self.load()
        if name == 'check':
            self.check_hosts()
        elif name == 'quant':
            self.quantize_hosts(shard, shards)
        elif name == 'complete':
            self.hosts_summary()
        elif name == 'eval':
            for arm in evals[shard::shards]:
                self.evaluate(arm)
        elif name == 'final':
            self.final_hosts()
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


def report(s):
    d, lines = s['diagnostics'], []
    c, h = d['combined_share'], d['diagnostic_head_share']
    lines += ['# WAM-guided Hessian：S1 灵敏度 + QTIP W2 宿主（held-out 40×2）', '',
              f"作业 `{s['pbs_jobid']}`。BF16 模型，A16；L = A + λZ + βK 的 Gauss-Newton 探针，C80 × 1 探针。", '',
              '## 灵敏度分布（各向同性权重噪声下的 loss 质量，C80 平均）', '',
              '| video 去噪步 | prefill | action 步 | 其他（proprio 等） |', '|---:|---:|---:|---:|',
              f"| {c['video_steps']:.3f} | {c['video_prefill']:.3f} | {c['action_steps']:.3f} | {c['other']:.3f} |", '',
              'video 去噪步逐步占比：' + ', '.join(f'{v:.3f}' for v in c['video_step_profile']), '',
              'action 步逐步占比：' + ', '.join(f'{v:.3f}' for v in c['action_step_profile']), '',
              f"分头诊断（{len(h.get('cases', []))} 个 context，各头单独反传）：A 直接 {h['A']:.3f}、"
              f"A 经 future {h['Az']:.3f}、βK {h['K']:.3f}、λZ {h['Z']:.3f}；A 头里经 future 的比例 {h['A_via_future']:.3f}。",
              '', f"BF16 query/reader 对 teacher 最大差 {d['teacher_parity_max_abs']:.3g}；"
              f"reader 调用签名最大相对差 {d['reader_signature_worst_relative']:.3g}。", '',
              '## QTIP W2 宿主：H_U（baseline 的 QTIP_W2 权重）vs H̃', '',
              '| 臂 | bpw | proxy(H_U) | proxy(H̃) | first10 motor | future NMSE | gripper RMSE | 换回 BF16 future 后 motor |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, o in s['heldout'].items():
        g = s['quant'][arm]
        lines.append(f"| {arm} | {g['bpw']:.3f} | {g['proxy_U_weighted']:.5f} | {g['proxy_G_weighted']:.5f} | "
                     f"{o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | {o['first10gripper_rmse']:.4f} | "
                     f"{o['first10motor_rmse_swap']:.4f} |")
    lines += ['', f"QTIP_U_W2 的 proxy(H_U) 在本作业重算；baseline 自己记录的是 "
              f"{s['quant']['QTIP_U_W2']['baseline_reported_proxy']:.5f}。",
              '', '换回 BF16 future：同一量化模型的 action 通路读 BF16 teacher 的 future（C1 在 weight-only 下的重测）。',
              '', '闭环（LIBERO-Plus 280）见 `loops/QTIP_G_W2/RESULTS.zh.md`；QTIP_W2 的闭环在 A16 baseline 目录。']
    return '\n'.join(lines) + '\n'


def coherent_report(s, host):
    g, d = s['gate'], s['diagnostics']
    names = {'QTIP_U_W2': 'H_U（baseline QTIP_W2）', GUIDED: 'H̃（逐 token 对角）'}
    names.update({a: f'H_{v}（相干，{"A+λZ+βK" if v == "C" else "仅 A"}）' for a, v in host['coherent']['arms'].items()})
    hc, hd = d['head_share_coherent'], d['head_share_diag']
    lines = ['# 相干 Gauss-Newton 插件：QTIP W2 宿主', '',
             f"作业 `{s['pbs_jobid']}`。C80 × {host['coherent']['probes_per_context']} 探针，草图行数 "
             f"{host['coherent']['sketch_rows']}；H = S^T S/均值对角 + {host['coherent']['gamma_u']}·H_U/均值对角。", '',
             '## 门槛（预注册）', '',
             f"A 头里经 future 的比例：相干 **{g['coherent_a_via_future']:.3f}**，逐 token 对角 {g['diag_a_via_future']:.3f}"
             f"（同一批探针，{g['contexts']} 个 context）；门槛 ≥ {g['threshold']}，"
             f"{'通过' if g['passed'] else '未通过，不量化'}。换回 BF16 future 的实测：first10 motor MSE 降 44–50%。", '',
             '## 分头占比（分头诊断的 context）', '',
             '| | A 直接 | A 经 future | βK | λZ |', '|---|---:|---:|---:|---:|',
             f"| 相干 | {hc['A']:.3f} | {hc['Az']:.3f} | {hc['K']:.3f} | {hc['Z']:.3f} |",
             f"| 逐 token 对角 | {hd['A']:.3f} | {hd['Az']:.3f} | {hd['K']:.3f} | {hd['Z']:.3f} |", '',
             '相干 loss 质量按层所属流：' + '；'.join(
                 f"{v} " + '、'.join(f'{k} {x:.3f}' for k, x in sh.items()) for v, sh in d['stream_share'].items()), '',
             f"草图 vs autograd 权重梯度最大相对差 {d['sketch_vs_autograd_worst']:.3g}；BF16 对 teacher 最大差 "
             f"{d['teacher_parity_max_abs']:.3g}；reader 签名最大相对差 {d['reader_signature_worst_relative']:.3g}。"]
    if g['passed']:
        m, keys = s['quant']['proxy_matrix'], ('U', 'G', 'C', 'CA', 'C_rows', 'CA_rows')
        lines += ['', '## 各 Hessian 下的 proxy（参数加权）', '',
                  '| 权重 \\ Hessian | H_U | H̃ | H_C | H_CA | H_C + 行权重 | H_CA + 行权重 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        lines += [f"| {names[a]} | " + ' | '.join(f'{m[a][k]:.5f}' for k in keys) + ' |' for a in m]
        for a, r in s['quant']['arms'].items():
            if r.get('row_scale'):
                lines.append(f"\n{names[a]} 行缩放（折进 SV，0 bit）：预估 guided loss 比 {r['row_ratio_in_sample']:.3f}，"
                             f"两半交叉 {r['row_ratio_split']:.3f}；相干行权重两半相关 {r['row_weight_half_corr']:.3f}，"
                             f"与逐 token 行权重相关 {r['row_weight_corr_diag']:.3f}。")
        lines += ['', '## held-out 40×2', '',
                  '| 臂 | first10 motor | future NMSE | late22 motor | gripper RMSE | 换回 BF16 future 后 motor | mixed loss |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for a, o in s['heldout'].items():
            lines.append(f"| {names[a]} | {o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | "
                         f"{o['late22motor_rmse']:.4f} | {o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} | "
                         f"{o['mixed_loss']:.5f} |")
        lines += ['', '## LIBERO-Plus 280 闭环', '', '| 臂 | 成功 | 对 QTIP_W2 胜/负 (p) | 对 QTIP_G_W2 胜/负 (p) |',
                  '|---|---:|---:|---:|']
        for a, o in s['loops'].items():
            cells = [f"{o[k]['wins']}/{o[k]['losses']} ({o[k]['p']:.2f})" if k in o else '—'
                     for k in ('vs_QTIP_U_W2', f'vs_{GUIDED}')]
            lines.append(f"| {names[a]} | {o['successes']}/{o['episodes']} | {cells[0]} | {cells[1]} |")
    return '\n'.join(lines) + '\n'


def hosts_report(s):
    names = dict(qtip='QTIP（trellis）', gptvq='GPTVQ（4 维 VQ 码本）', quipsharp='QuIP#（E8 格）')
    lines = ['# 插件的舍入阶段迁移到更多 VQ 宿主（W2，A16）', '',
             f"作业 `{s['pbs_jobid']}`。所有宿主吃同一份 H̃（第一阶段：A + λZ + βK 探针的逐 token 权重，γ = 0.1），"
             '宿主代码和每层种子与 A16 baseline 相同，只换 Hessian。', '',
             '## 闭环（LIBERO-Plus 280；BF16 216）', '',
             '| 宿主 | bpw | H_U（baseline） | H̃（插件） | 差 | 配对 胜/负 | McNemar p |', '|---|---:|---:|---:|---:|---:|---:|']
    for m, r in s['hosts'].items():
        o = r.get('loop')
        cells = (f"{o['baseline']} | {o['guided']} | {o['guided'] - o['baseline']:+d} | {o['wins']}/{o['losses']} | {o['p']:.2f}"
                 if o else '— | — | — | — | —')
        lines.append(f"| {names.get(m, m)} | {r['guided']['bpw']:.3f} | {cells} |")
    lines += ['', '## held-out 40×2 与 proxy', '',
              '| 宿主 | 臂 | proxy(H_U) | proxy(H̃) | first10 motor | future NMSE | late22 motor | gripper RMSE | 换回 BF16 future 后 motor | mixed loss |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, r in s['hosts'].items():
        for role in ('baseline', 'guided'):
            x, o = r[role], r[role]['heldout']
            held = (f"{o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | {o['late22motor_rmse']:.4f} | "
                    f"{o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} | {o['mixed_loss']:.5f}"
                    if o else '— | — | — | — | — | —')
            lines.append(f"| {names.get(m, m)} | {x['arm']} | {x['proxy_U']:.5f} | {x['proxy_G']:.5f} | {held} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    cfg = read(args.protocol)
    if cfg['protocol'] != PROTOCOL:
        raise ValueError('Protocol identity differs')
    stage, shard, shards = os.environ['FW_STAGE'], int(os.environ['FW_SHARD']), int(os.environ['FW_SHARDS'])
    job = f"{os.environ['FW_JOB']}_{shard}"  # progress, failure and Runtime files are per GPU process
    failure = Path(cfg['remote_root']) / 'results' / 'failures' / f'{job}.json'
    failure.unlink(missing_ok=True)
    try:
        Work(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        save(failure, dict(protocol=PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                           error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
