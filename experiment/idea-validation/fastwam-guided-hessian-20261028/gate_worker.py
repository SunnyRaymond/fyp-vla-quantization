"""Gate 1 (gate.json): Gauss-Newton probes of A + lambda Z + beta K linearized at each host's own W2 model.

Phase 1 linearized at BF16, where the z -> a path is nearly silent (6.9% of the A head through the future, beta K 1.2%
of the mass), while at W2 the BF16-future swap removes 44-61% of the held-out first10 motor MSE. Here QTIP / GPTVQ /
QuIP# W2 (the A16 baselines' H_U weights) are installed before the probes. Steps (FW_STAGE, shard FW_SHARD of
FW_SHARDS, one process per GPU):
  probe    per host: sigma_l^2 of the baseline weight; K pass (the frozen BF16 reader reads the student's own future,
           c_K = J^T g_K, BF16 masters on the GPU only here); the phase-1 probe pass with A through the model's own W2
           reader and c_K added at the future. Diagnostic contexts back-propagate the heads separately.
  select   per host: damage sigma_l^2 * mass_l, the top layers; head shares at the operating point and at BF16.
  hessian  per host: H~_op of the selected layers (phase-1 form, inputs from the W2 trajectory).
  quant    the host requantizes the selected layers on H~_op with the baseline seeds.
  eval     held-out 40x2 + swap of GATE_<host>_G (selected layers from the phase-1 H~ arm) and GATE_<host>_O (H~_op).
  final    results/gate/{summary.json, RESULTS.zh.md, DECISION.json}.
The model, held-out panel and probe pieces are worker.Work's (phases 1-4), unchanged. 0 training, 0 teacher queries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import time

import worker as wk  # owns the PBS allocation guard and the runtime imports
import torch

import frozen_reader as fr
import lp_worker as lp
import probe as pr
import softbin_loss

HERE = Path(__file__).resolve().parent
GATE = json.loads((HERE / 'gate.json').read_text(encoding='utf-8'))
HOSTS = GATE['hosts']
HEADS = ('A', 'Az', 'K', 'Z')
VARIANTS = ('G', 'O')
# worker.Work.load routes the reader through the installed weights; the K pass needs the frozen BF16 reader back.
BF16_CONTEXTS = (fr._bf16_forward_context, fr._checkpoint_context_fn)


def gate_arm(host, variant):
    return f'GATE_{host}_{variant}'


def full_arm(host):
    return host + GATE['full']['arm_suffix']


def layer_damage(sigma, mass):
    """Predicted loss increase per layer under isotropic rounding noise of variance sigma_l^2 per weight:
    E[(g . dW x)^2] = sigma^2 |g|^2 |x|^2 summed over tokens and calls, i.e. sigma_l^2 * mass_l."""
    return [s * m for s, m in zip(sigma, mass)]


def top_layers(damage, k):
    order = sorted(range(len(damage)), key=lambda i: (-damage[i], i))
    chosen = sorted(order[:k])
    total = sum(damage)
    return chosen, (sum(damage[i] for i in chosen) / total if total > 0 else 0.0)


def head_shares(per_context):
    """per_context: [{head: mass}] -> context-mean share of each head and of the A head that goes through the future
    (the phase-1 definition: Az / (A + Az))."""
    out = defaultdict(float)
    for heads in per_context:
        whole = max(sum(heads.values()), 1e-30)
        for h, v in heads.items():
            out[h] += v / whole / len(per_context)
        out['A_via_future'] += heads['Az'] / max(heads['Az'] + heads['A'], 1e-30) / len(per_context)
    return dict(out)


def group_total(groups):
    """Whole loss mass of one context's group_masses (all heads)."""
    return sum(sum(g['video_steps']) + g['video_prefill'] + sum(g['action_steps']) + g['other'] for g in groups.values())


def ess(values):
    s, q = sum(values), sum(v * v for v in values)
    return s * s / q if q > 0 else 0.0


def future_share(o):
    """Share of the first10 motor MSE the BF16-future swap removes."""
    return 1 - o['first10motor_mse_swap'] / o['first10motor_mse']


def decide(heldout, min_hosts):
    """Per host: does an arm lower the held-out mixed loss and the future share of the first10 motor MSE against the
    baseline U (both went up with the phase-1 H~ on every host)? The decision uses the O arm."""
    hosts = {}
    for host, arms in heldout.items():
        u, row = arms['U'], {}
        for v in VARIANTS:
            o = arms[v]
            row[v] = dict(mixed_down=o['mixed_loss'] < u['mixed_loss'], future_share_down=future_share(o) < future_share(u))
            row[v]['c1_direction'] = row[v]['mixed_down'] and row[v]['future_share_down']
        hosts[host] = row
    passed = sum(r['O']['c1_direction'] for r in hosts.values())
    return dict(hosts=hosts, passed=passed, min_hosts=min_hosts, go=passed >= min_hosts)


class GateWork(wk.Work):
    def gate(self, host=None):
        return self.base / 'gate' / host if host else self.base / 'gate'

    def selection(self, host):
        return wk.read(self.gate() / 'selection.json')['hosts'][host]

    def arm_root(self, arm):
        if arm in {full_arm(h) for h in HOSTS}:
            return self.base / 'quantized' / arm
        return super().arm_root(arm)

    def install(self, arm):
        """GATE_<host>_<G|O>: the host's baseline W2, then the selected layers from the phase-1 H~ arm or from H~_op."""
        if not arm.startswith('GATE_'):
            return super().install(arm)
        host, variant = arm[len('GATE_'):].rsplit('_', 1)
        c = HOSTS[host]
        super().install(c['baseline_arm'])
        source = self.arm_root(c['bf16_point_arm']) if variant == 'G' else self.gate(host)
        with torch.no_grad():
            for i in self.selection(host)['layers']:
                saved = torch.load(source / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                name = self.bank.names[i]
                if saved['module'] != name:
                    raise RuntimeError(f'Stored weight identity differs at {i}')
                self.bank.modules[name].weight.data.copy_(saved['weight'].to('cuda'))

    # ---------------- probe ----------------
    def capture_masters(self):
        """The 614 BF16 weights on the CPU, before any install; one checked against the baselines' prep master."""
        self.masters = {n: m.weight.detach().cpu() for n, m in self.bank.modules.items()}
        first = self.bank.names[0]
        saved = torch.load(self.baseline / 'prep' / 'masters' / '0000.pt', map_location='cpu', weights_only=True)
        if saved['module'] != first or not torch.equal(saved['weight'], self.masters[first]):
            raise RuntimeError('Runtime BF16 weight differs from the baseline prep master')

    def sigma(self, host):
        path = self.gate(host) / 'sigma.json'
        if path.exists():
            return
        values = []
        with torch.no_grad():
            for name in self.bank.names:
                d = self.bank.modules[name].weight.detach().float() - self.masters[name].to('cuda', torch.float32)
                values.append(float(d.square().mean()))
        wk.save(path, dict(protocol=wk.PROTOCOL, host=host, arm=HOSTS[host]['baseline_arm'], sigma2=values,
                           definition='mean squared error of the installed W2 weight against the BF16 master'))

    @contextmanager
    def bf16_reader(self):
        """The loss-pair chain's frozen BF16 reader: target Linears read the masters (same biases) inside the reader;
        the installed W2 weights stay in the modules."""
        patched = (fr._bf16_forward_context, fr._checkpoint_context_fn)
        fr._bf16_forward_context, fr._checkpoint_context_fn = BF16_CONTEXTS
        weights, reader = {n: t.to('cuda') for n, t in self.masters.items()}, None
        try:
            reader = fr.BF16Reader(self.runtime, weights)
            yield reader
        finally:
            fr._bf16_forward_context, fr._checkpoint_context_fn = patched
            if reader is not None:
                reader.original_weights.clear()
            weights.clear()
            gc.collect()
            torch.cuda.empty_cache()

    def k_pass(self, host, cases):
        """c_K = J_R(z_hat)^T g_K per probe for the frozen BF16 reader R at the student's own future z_hat, before the
        probe pass (no query graph alive, so the masters fit next to the installed weights)."""
        out, probes = self.gate(host) / 'kpass', self.cfg['probes']['probes_per_context']
        todo = [c for c in cases if not (out / f'{c}.pt').exists()]
        if not todo:
            return
        stats = self.p['reader_source']['action_stats']
        with self.bf16_reader() as reader:
            for count, case in enumerate(todo):
                self.enough()
                target = self.teacher(case, 0)
                with torch.no_grad():
                    action, future, _ = self.query(case, 0)
                value, description = self.datum(case)
                self.runtime.description = description
                leaf = future.detach().float().clone().requires_grad_(True)  # a regular tensor even from inference mode
                read_action, _ = reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], leaf, checkpointed=True)
                self.count('reader_queries')
                curv = pr.softbin_kl_curvature(softbin_loss, target['action'][:10, :6], stats['mean'], stats['scale'],
                                               self.p['reader_source']['config']).to('cuda')
                gen = torch.Generator(device='cuda').manual_seed(GATE['probes']['k_seed'] + case)
                cots = []
                for p in range(probes):
                    gk = torch.zeros_like(read_action)
                    gk[:10, :6] = pr.rademacher((10, 6), gen, 'cuda') * (curv / 2).sqrt()
                    c, = torch.autograd.grad(read_action, [leaf], gk, retain_graph=p < probes - 1)
                    cots.append(c.detach().float().cpu())
                    self.counters['parameter_reverse_passes'] += 1
                t = target['action'].float().cpu()
                own, bf16 = action.float().cpu() - t, read_action.detach().float().cpu() - t
                self.tsave(out / f'{case}.pt', dict(
                    case=case, host=host, action=action.detach().cpu(), future=future.detach().cpu(),
                    c_k=torch.stack(cots), student_first10motor_mse=float(own[:10, :6].square().mean()),
                    bf16_reader_first10motor_mse=float(bf16[:10, :6].square().mean())))
                del action, future, leaf, read_action, cots, target, curv
                self.progress('kpass', host=host, contexts=count + 1, of=len(todo))

    def stats_op(self, host, case, separate):
        """worker.Work.stats_context at the operating point: parity against the K pass's query, c_K from the K pass."""
        b, cfg, loss = self.bank, self.cfg['probes'], self.cfg['loss']
        root = self.gate(host)
        k = torch.load(root / 'kpass' / f'{case}.pt', map_location='cpu', weights_only=True)
        target = self.teacher(case, 0)
        b.begin_probe()
        action_q, future, _ = self.query(case, 0, grad=True)
        b.phase = 'r'
        value, description = self.datum(case)
        self.runtime.description = description
        leaf = future.detach().float().requires_grad_(True)
        action, _ = self.reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], leaf, checkpointed=True)
        self.count('reader_queries')
        calls = b.finish_forward(self.cfg['gates']['signature_tolerance'])
        parity = dict(future_vs_kpass=wk.max_abs(future, k['future']), query_action_vs_kpass=wk.max_abs(action_q, k['action']),
                      reader_vs_query_action=wk.max_abs(action, action_q))
        del action_q
        if max(parity.values()) > self.cfg['gates']['teacher_parity_max_abs']:
            raise RuntimeError(f'W2 query / K pass / reader differ: {parity}')

        znorm = float(target['future'][:, :, 1:].float().norm())
        scales = pr.head_scales(loss['lambda'], loss['beta'])
        gen = torch.Generator(device='cuda').manual_seed(cfg['seed'] + case)
        heads = HEADS if separate else ('A', 'V')
        sinks = {h: dict(u={}, d={}) for h in heads}
        probes = cfg['probes_per_context']
        for p in range(probes):
            last = p == probes - 1
            ga = torch.zeros_like(action)
            ga[:10, :6] = pr.rademacher((10, 6), gen, 'cuda') / math.sqrt(60)
            vz = torch.zeros(future.shape, device='cuda')
            vz[:, :, 1:] = pr.rademacher(vz[:, :, 1:].shape, gen, 'cuda') / znorm
            b.sink = sinks['A']
            c1 = torch.autograd.grad(action, [leaf, b.anchor], ga, retain_graph=not last, allow_unused=True)[0]
            b.sink = None
            c2 = k['c_k'][p].to('cuda', torch.float32)
            self.counters['parameter_reverse_passes'] += 1
            parts = ([('Az', c1), ('K', scales['K'] * c2), ('Z', scales['Z'] * vz)] if separate
                     else [('V', c1 + scales['K'] * c2 + scales['Z'] * vz)])
            for j, (head, c) in enumerate(parts):
                b.sink = sinks[head]
                torch.autograd.grad(future, [b.anchor], c.to(future.dtype),
                                    retain_graph=not (last and j == len(parts) - 1), allow_unused=True)
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
        layer_mass = {h: [0.0] * len(b.names) for h in masses}
        for h, per in masses.items():
            for (idx, _), value in per.items():
                layer_mass[h][idx] += value
        rows = {idx: int(sum(t.numel() for t in v)) for idx, v in b.xn.items()}
        record = dict(case=case, separate=separate, probes=probes, rows=rows, linearization=host,
                      u={idx: {n: v.div(probes).cpu() for n, v in calls_.items()} for idx, calls_ in u.items()},
                      d={idx: v.div(probes).cpu() for idx, v in d.items()},
                      sig={idx: dict(enumerate(v)) for idx, v in b.qsig.items()})
        self.tsave(root / 'stats' / f'{case}.pt', record)
        summary = dict(protocol=wk.PROTOCOL, host=host, case_id=case, separate=separate, parity=parity, calls=calls,
                       znorm=znorm, groups=self.group_masses(masses), layer_mass=layer_mass,
                       student_first10motor_mse=k['student_first10motor_mse'],
                       bf16_reader_first10motor_mse=k['bf16_reader_first10motor_mse'], pbs_jobid=self.job)
        wk.save(root / 'stats' / f'{case}.json', summary)
        b.end_probe()
        del action, future, leaf, sinks, u, d, k
        torch.cuda.empty_cache()
        return summary

    def probe(self, shard, shards):
        self.load()
        self.capture_masters()
        diag = set(self.diagnostic_cases())
        # A diagnostic context back-propagates the video loop three times instead of once: count it double.
        mine = wk.owned(shard, shards, lambda c: 2.0 if c in diag else 1.0, lp.CASES)
        for host, c in HOSTS.items():
            root = self.gate(host)
            if all((root / 'stats' / f'{case}.json').exists() for case in mine) and (shard or (root / 'sigma.json').exists()):
                continue
            self.install(c['baseline_arm'])
            if shard == 0:
                self.sigma(host)
            self.k_pass(host, mine)
            for count, case in enumerate(mine):
                if not (root / 'stats' / f'{case}.json').exists():
                    self.enough()
                    self.stats_op(host, case, case in diag)
                self.progress('probe', host=host, contexts=count + 1, of=len(mine))

    # ---------------- select ----------------
    def bf16_point(self, cases):
        """Phase-1 per-layer masses of the diagnostic contexts (BF16 linearization point), per head."""
        out = []
        for case in cases:
            record = torch.load(self.base / 'stats' / f'{case}.pt', map_location='cpu', weights_only=True)
            per = {h: [0.0] * 614 for h in HEADS}
            for h in HEADS:
                for key, value in record['masses'][h].items():
                    per[h][int(key.split('|')[0])] += value
            out.append(per)
            del record
        return out

    def select(self):
        diag = self.diagnostic_cases()
        bf16 = self.bf16_point(diag)
        phase1 = [wk.read(self.base / 'stats' / f'{c}.json') for c in lp.CASES]
        out = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, selection=GATE['selection'], hosts={})
        for host in HOSTS:
            root = self.gate(host)
            rows = [wk.read(root / 'stats' / f'{c}.json') for c in lp.CASES]
            sigma = wk.read(root / 'sigma.json')['sigma2']
            mass = [sum(sum(r['layer_mass'][h][i] for h in r['layer_mass']) for r in rows) / len(rows)
                    for i in range(len(sigma))]
            damage = layer_damage(sigma, mass)
            layers, coverage = top_layers(damage, GATE['selection']['layers'])
            drows = [r for r in rows if r['separate']]
            unit = head_shares([{h: sum(r['layer_mass'][h]) for h in HEADS} for r in drows])
            weighted = head_shares([{h: sum(s * m for s, m in zip(sigma, r['layer_mass'][h])) for h in HEADS} for r in drows])
            weighted_bf16 = head_shares([{h: sum(s * m for s, m in zip(sigma, r[h])) for h in HEADS} for r in bf16])
            context_damage = [sum(s * sum(r['layer_mass'][h][i] for h in r['layer_mass']) for i, s in enumerate(sigma))
                              for r in rows]
            context_unit = [sum(sum(v) for v in r['layer_mass'].values()) for r in rows]
            groups = self.mean_groups([r['groups'] for r in rows])
            out['hosts'][host] = dict(
                layers=layers, modules=[self.module_name(i) for i in layers], coverage=coverage,
                damage={str(i): damage[i] for i in layers}, damage_total=sum(damage),
                head_share_unit_noise=unit, head_share_sigma_weighted=weighted, head_share_sigma_weighted_bf16_point=weighted_bf16,
                context_ess=ess(context_damage), context_ess_unit_noise=ess(context_unit), contexts=len(rows), groups=groups,
                student_first10motor_mse=sum(r['student_first10motor_mse'] for r in rows) / len(rows),
                bf16_reader_first10motor_mse=sum(r['bf16_reader_first10motor_mse'] for r in rows) / len(rows),
                parity_max=max(max(r['parity'].values()) for r in rows),
                signature_worst=max(r['calls']['signature_worst_relative'] for r in rows))
        out['bf16_point'] = dict(head_share_unit_noise=head_shares([{h: sum(r[h]) for h in HEADS} for r in bf16]),
                                 context_ess_unit_noise=ess([group_total(r['groups']) for r in phase1]))
        wk.save(self.gate() / 'selection.json', out)

    def module_name(self, i):
        if not hasattr(self, '_modules'):
            self._modules = wk.read(self.baseline / 'prep' / 'modules.json')
        return self._modules[i]['module']

    @staticmethod
    def mean_groups(groups):
        """Context-mean share of the loss mass (all heads) in video steps / prefill / action steps."""
        acc = dict(video_steps=[0.0] * 10, video_prefill=0.0, action_steps=[0.0] * 10, other=0.0)
        for g in groups:
            heads = [h for h in g]
            vs = [sum(g[h]['video_steps'][s] for h in heads) for s in range(10)]
            ac = [sum(g[h]['action_steps'][s] for h in heads) for s in range(10)]
            pre, other = sum(g[h]['video_prefill'] for h in heads), sum(g[h]['other'] for h in heads)
            total = max(sum(vs) + sum(ac) + pre + other, 1e-30)
            for s in range(10):
                acc['video_steps'][s] += vs[s] / total / len(groups)
                acc['action_steps'][s] += ac[s] / total / len(groups)
            acc['video_prefill'] += pre / total / len(groups)
            acc['other'] += other / total / len(groups)
        return acc

    # ---------------- hessian ----------------
    def hessians_op(self, host, todo):
        """worker.Work.hessians for the given layers from the operating-point statistics of this host."""
        root, gamma = self.gate(host), self.cfg['hessian']['gamma']
        keep, usum, rows, records = set(todo), defaultdict(float), defaultdict(int), {}
        for case in lp.CASES:
            r = torch.load(root / 'stats' / f'{case}.pt', map_location='cpu', weights_only=True)
            for idx, calls in r['u'].items():
                usum[idx] += sum(float(v.sum()) for v in calls.values())
            for idx, n in r['rows'].items():
                rows[idx] += n
            records[case] = dict(u={i: r['u'][i] for i in keep if i in r['u']}, sig=r['sig'])
            del r
        ubar = {i: usum[i] / rows[i] if usum[i] > 0 else 1.0 for i in todo}
        groups, plan = self.passes(todo)
        wk.save(root / 'hessian_plan' / f'{self.job_name}.json',
                dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, host=host, todo=todo, **plan))
        for number, group in enumerate(groups):
            replay = wk.Replay(group, self.bank.by_index, ubar, gamma, check=number == 0)
            self.bank.mode, self.bank.replay = 'replay', replay
            with torch.no_grad():
                for count, case in enumerate(lp.CASES):
                    self.enough()
                    replay.start(records[case]['u'], records[case]['sig'])
                    action, future, _ = self.query(case)
                    del action, future
                    replay.finish(self.cfg['gates']['signature_tolerance'])
                    self.progress('hessian', host=host, pass_index=number, passes=len(groups), contexts=count + 1)
            self.bank.mode, self.bank.replay = 'plain', None
            for i in group:
                self.tsave(root / 'hessians' / f'{i:04d}.pt', dict(
                    module=self.bank.names[i], rows=rows[i], hessian=(replay.H[i] / replay.wsum[i]).float().cpu(),
                    weight_sum=float(replay.wsum[i]), ubar=ubar[i], gamma=gamma, linearization=host,
                    replay_signature_worst=replay.worst))
            del replay
            torch.cuda.empty_cache()

    def ucheck(self, host):
        """The baseline W2's first held-out endpoint again, against the stored one the U row of the gate reads."""
        path = self.gate(host) / 'ucheck.json'
        if path.exists():
            return
        arm, case = HOSTS[host]['baseline_arm'], self.cfg['evaluation']['case_ids'][0]
        stored = wk.read(self.base / 'endpoints' / arm / f'{case}_0.json')
        row = self.endpoint(gate_arm(host, 'UCHECK'), case, 0)
        keys = ('first10motor_mse', 'first10motor_mse_swap', 'future_latent_nmse', 'mixed_loss')
        wk.save(path, dict(protocol=wk.PROTOCOL, host=host, arm=arm, case_id=case, pbs_jobid=self.job,
                           relative={k: abs(row[k] - stored[k]) / max(abs(stored[k]), 1e-30) for k in keys}))

    def hessian(self, shard, shards):
        self.load()
        for host, c in HOSTS.items():
            mine = wk.owned(shard, shards, lambda i: self.bank.by_index[i].in_features ** 2, self.selection(host)['layers'])
            todo = [i for i in mine if not (self.gate(host) / 'hessians' / f'{i:04d}.pt').exists()]
            if todo or (shard == 0 and not (self.gate(host) / 'ucheck.json').exists()):
                self.install(c['baseline_arm'])
            if shard == 0:
                self.ucheck(host)
            if todo:
                self.hessians_op(host, todo)

    # ---------------- quant ----------------
    def requant(self, host, i, device='cuda'):
        """Layer i on H~_op by the host with the baseline seed (a weight the gate already made is reused: same H~_op,
        seed and code); the proxies of U, G and O under H_U, H~ and H~_op. Returns (module, weight, record)."""
        prep, c, root = self.baseline / 'prep', HOSTS[host], self.gate(host)
        sources = (('master', prep / 'masters'), ('H_U', prep / 'hessians'), ('H_G', self.base / 'hessians'),
                   ('H_O', root / 'hessians'), ('U', self.arm_root(c['baseline_arm']) / 'weights'),
                   ('G', self.arm_root(c['bf16_point_arm']) / 'weights'))
        saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True) for k, path in sources}
        name = saved['master']['module']
        if any(v['module'] != name for v in saved.values()):
            raise RuntimeError(f'Layer identity differs at {i}: {name}')
        w = saved['master']['weight'].to(device, torch.float32)
        h = {k: saved[k]['hessian'].to(device, torch.float32) for k in ('H_U', 'H_G', 'H_O')}
        made = root / 'wstats' / f'{i:04d}.json'
        if made.exists():
            meta = wk.read(made)
            dq = torch.load(root / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)['weight'].to(device, torch.float32)
            bits_total, seconds, config = meta['bits_total'], meta['seconds'], meta['config']
        else:
            begin = time.monotonic()
            result = self.method(c['method']).quantize_layer(w, h['H_O'], c['bits'], seed=self.host['host']['seed_base'] + i)
            seconds = time.monotonic() - begin
            dq, bits_total, config = result['dequant'].to(device, torch.float32), result['bits_total'], result['config']
        if tuple(dq.shape) != tuple(w.shape) or not bool(torch.isfinite(dq).all()):
            raise RuntimeError(f'{host} produced an invalid weight at {name}')
        q = {'U': saved['U']['weight'].to(device, torch.float32), 'G': saved['G']['weight'].to(device, torch.float32), 'O': dq}

        def proxy(x, hv):
            e = x - w
            return float(((e @ hv) * e).sum() / ((w @ hv) * w).sum().clamp_min(1e-30))
        record = dict(host=host, index=i, module=name, shape=list(w.shape), bits_total=bits_total,
                      bpw=bits_total / w.numel(), seconds=seconds, config=config, hessian='H_O',
                      proxies={a: {k: proxy(x, v) for k, v in h.items()} for a, x in q.items()})
        return name, dq.to(torch.bfloat16).cpu(), record

    def quant(self, shard, shards, device='cuda'):
        """The selected layers of every host on H~_op -> results/gate/<host>/{weights, wstats}/."""
        items = [(h, i) for h in HOSTS for i in self.selection(h)['layers']]
        cost = {(h, i): wk.read(self.arm_root(HOSTS[h]['baseline_arm']) / 'stats' / f'{i:04d}.json')['seconds']
                for h, i in items}
        mine = wk.owned(shard, shards, cost.__getitem__, items)
        torch.use_deterministic_algorithms(False)  # as the baselines' quant_worker
        try:
            for done, (host, i) in enumerate(mine):
                root = self.gate(host)
                if (root / 'wstats' / f'{i:04d}.json').exists():
                    continue
                self.enough()
                name, dq, record = self.requant(host, i, device)
                self.tsave(root / 'weights' / f'{i:04d}.pt', dict(module=name, weight=dq))
                wk.save(root / 'wstats' / f'{i:04d}.json', record)
                if device == 'cuda':
                    torch.cuda.empty_cache()
                self.progress('quant', layers=done + 1, of=len(mine))
        finally:
            torch.use_deterministic_algorithms(True)

    # ---------------- full run (gate.json full) ----------------
    def full_hessian(self, shard, shards):
        self.load()
        for host, c in HOSTS.items():
            mine = wk.owned(shard, shards, lambda i: self.bank.by_index[i].in_features ** 2)
            todo = [i for i in mine if not (self.gate(host) / 'hessians' / f'{i:04d}.pt').exists()]
            if todo:
                self.install(c['baseline_arm'])
                self.hessians_op(host, todo)

    def full_quant(self, shard, shards, device='cuda'):
        """Every (host, layer) on H~_op -> results/quantized/<host>_O_W2/{weights, stats}/; split on baseline seconds."""
        layers = len(wk.read(self.baseline / 'prep' / 'modules.json'))
        items = [(h, i) for h in HOSTS for i in range(layers)]
        cost = {(h, i): wk.read(self.arm_root(HOSTS[h]['baseline_arm']) / 'stats' / f'{i:04d}.json')['seconds']
                for h, i in items}
        mine = wk.owned(shard, shards, cost.__getitem__, items)
        torch.use_deterministic_algorithms(False)
        try:
            for done, (host, i) in enumerate(mine):
                out = self.arm_root(full_arm(host))
                if (out / 'stats' / f'{i:04d}.json').exists():
                    continue
                self.enough()
                name, dq, record = self.requant(host, i, device)
                p = record['proxies']['O']
                self.tsave(out / 'weights' / f'{i:04d}.pt', dict(module=name, weight=dq))
                wk.save(out / 'stats' / f'{i:04d}.json', dict(record, proxy_U=p['H_U'], proxy_G=p['H_G'], proxy_O=p['H_O']))
                if device == 'cuda':
                    torch.cuda.empty_cache()
                if done % 25 == 0:
                    self.progress('quant', layers=done + 1, of=len(mine))
        finally:
            torch.use_deterministic_algorithms(True)

    def full_complete(self):
        """quantized/<host>_O_W2/COMPLETE.json (what the baselines' analysis.py reads) with the parameter-weighted proxies
        of U, G and O under H_U, H~ and H~_op."""
        layers, out = len(wk.read(self.baseline / 'prep' / 'modules.json')), {}
        for host, c in HOSTS.items():
            arm = full_arm(host)
            stats = [wk.read(self.arm_root(arm) / 'stats' / f'{i:04d}.json') for i in range(layers)]
            n = sum(s['shape'][0] * s['shape'][1] for s in stats)

            def weighted(f):
                return sum(f(s) * s['shape'][0] * s['shape'][1] for s in stats) / n
            row = dict(arm=arm, method=c['method'], bits=c['bits'], hessian='H_O', bpw=sum(s['bits_total'] for s in stats) / n,
                       proxy_nmse_weighted=weighted(lambda s: s['proxy_U']), quantize_seconds=sum(s['seconds'] for s in stats),
                       proxies={a: {k: weighted(lambda s: s['proxies'][a][k]) for k in ('H_U', 'H_G', 'H_O')}
                                for a in ('U', 'G', 'O')})
            wk.save(self.arm_root(arm) / 'COMPLETE.json', dict(protocol=wk.PROTOCOL, complete=True, pbs_jobid=self.job,
                                                              target_weights=n, config_example=stats[0]['config'], **row))
            out[host] = row
        return out

    def full_final(self):
        """Per host the baseline (H_U), the phase-1 H~ arm and the O arm: proxies, held-out, closed loop, O's paired
        counts against both with McNemar p; pooled over the hosts."""
        def episodes(root):
            return {int(f.stem): wk.read(f)['success'] for f in (root / 'episodes').glob('*.json')} if (root / 'DONE').exists() else None
        quant, hosts, pooled = self.full_complete(), {}, {'U': [0, 0], 'G': [0, 0]}
        for host, c in HOSTS.items():
            arms = dict(U=c['baseline_arm'], G=c['bf16_point_arm'], O=full_arm(host))
            roots = dict(U=self.baseline / 'loops' / c['baseline_loop'], G=self.base / 'loops' / c['bf16_point_arm'],
                         O=self.base / 'loops' / full_arm(host))
            held = {k: wk.read(self.base / 'heldout' / f'{a}.json')['overall'] for k, a in arms.items()
                    if (self.base / 'heldout' / f'{a}.json').exists()}
            row = dict(arms=arms, heldout=held, quant=quant[host], loop=None)
            e = {k: episodes(r) for k, r in roots.items()}
            if all(v is not None for v in e.values()):
                row['loop'] = {k: sum(v.values()) for k, v in e.items()}
                row['loop']['episodes'] = len(e['O'])
                for ref in ('U', 'G'):
                    wins = sum(e['O'][k] and not e[ref][k] for k in e['O'] if k in e[ref])
                    losses = sum(e[ref][k] and not e['O'][k] for k in e['O'] if k in e[ref])
                    row['loop'][f'vs_{ref}'] = dict(wins=wins, losses=losses, p=wk.mcnemar(wins, losses))
                    pooled[ref][0] += wins
                    pooled[ref][1] += losses
            hosts[host] = row
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, hosts=hosts,
                       pooled={f'vs_{k}': dict(wins=w, losses=l, p=wk.mcnemar(w, l)) for k, (w, l) in pooled.items()},
                       gate=wk.read(self.gate() / 'DECISION.json'), new_teacher_queries=0, training_updates=0,
                       hessian='H~_op: phase-1 form (gamma 0.1), probes and inputs at the host\'s baseline W2 model')
        wk.save(self.base / 'full' / 'summary.json', summary)
        (self.base / 'full' / 'RESULTS.zh.md').write_text(full_report(summary), encoding='utf-8')
        return summary

    # ---------------- final ----------------
    def final(self):
        sel = wk.read(self.gate() / 'selection.json')
        heldout, proxies = {}, {}
        for host, c in HOSTS.items():
            arms = dict(U=c['baseline_arm'], G=gate_arm(host, 'G'), O=gate_arm(host, 'O'), G_full=c['bf16_point_arm'])
            heldout[host] = {k: wk.read(self.base / 'heldout' / f'{a}.json')['overall'] for k, a in arms.items()}
            stats = [wk.read(self.gate(host) / 'wstats' / f'{i:04d}.json') for i in sel['hosts'][host]['layers']]
            n = sum(s['shape'][0] * s['shape'][1] for s in stats)
            proxies[host] = {a: {k: sum(s['proxies'][a][k] * s['shape'][0] * s['shape'][1] for s in stats) / n
                                 for k in ('H_U', 'H_G', 'H_O')} for a in ('U', 'G', 'O')}
            proxies[host]['bpw_selected'] = sum(s['bits_total'] for s in stats) / n
        decision = decide({h: {k: v for k, v in a.items() if k != 'G_full'} for h, a in heldout.items()},
                          GATE['decision']['min_hosts'])
        ucheck = {h: max(wk.read(self.gate(h) / 'ucheck.json')['relative'].values()) for h in HOSTS}
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, gate=GATE,
                       selection=sel, heldout=heldout, proxies=proxies, decision=decision, ucheck_relative_max=ucheck,
                       new_teacher_queries=0, training_updates=0)
        wk.save(self.gate() / 'summary.json', summary)
        wk.save(self.gate() / 'DECISION.json', dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, **decision))
        (self.gate() / 'RESULTS.zh.md').write_text(gate_report(summary), encoding='utf-8')
        return summary

    def stage(self, name, shard, shards):
        if name == 'probe':
            self.probe(shard, shards)
        elif name == 'select':
            self.select()
        elif name == 'hessian':
            self.hessian(shard, shards)
        elif name == 'quant':
            self.quant(shard, shards)
        elif name == 'eval':
            self.load()
            for arm in [gate_arm(h, v) for h in HOSTS for v in VARIANTS][shard::shards]:
                self.evaluate(arm)
        elif name == 'final':
            self.final()
        elif name == 'full_hessian':
            self.full_hessian(shard, shards)
        elif name == 'full_quant':
            self.full_quant(shard, shards)
        elif name == 'full_complete':
            self.full_complete()
        elif name == 'full_eval':
            self.load()
            for arm in [full_arm(h) for h in HOSTS][shard::shards]:
                self.evaluate(arm)
        elif name == 'full_final':
            self.full_final()
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


def gate_report(s):
    sel, held, prox, dec = s['selection'], s['heldout'], s['proxies'], s['decision']
    b = sel['bf16_point']['head_share_unit_noise']
    lines = ['# 门 1：在宿主自己的 W2 模型上取 GN 曲率', '',
             f"作业 `{s['pbs_jobid']}`。A + λZ + βK 的 GN 探针在三个宿主的 baseline W2 模型上线性化（K 用冻结的 BF16 reader "
             "读 student 自己的 future）；每个宿主挑 16 层，用工作点 H̃_op 重量化，与第一阶段 H̃（BF16 点）的同一批层对照。", '',
             '## 线性化点：BF16 vs 各宿主 W2（8 个诊断 context，分头）', '',
             '| 线性化点 | A 直接 | A 经 future | βK | λZ | A 头里经 future 的比例 | σ² 加权后经 future 的比例 |',
             '|---|---:|---:|---:|---:|---:|---:|',
             f"| BF16（第一阶段） | {b['A']:.3f} | {b['Az']:.3f} | {b['K']:.3f} | {b['Z']:.3f} | {b['A_via_future']:.3f} | — |"]
    for host, r in sel['hosts'].items():
        u, w = r['head_share_unit_noise'], r['head_share_sigma_weighted']
        lines.append(f"| {host} W2 | {u['A']:.3f} | {u['Az']:.3f} | {u['K']:.3f} | {u['Z']:.3f} | {u['A_via_future']:.3f} | "
                     f"{w['A_via_future']:.3f}（BF16 点 {r['head_share_sigma_weighted_bf16_point']['A_via_future']:.3f}） |")
    lines += ['', '各向同性单位噪声下的质量占比与第一阶段定义相同；σ² 加权用该宿主每层实际的舍入误差方差。', '',
              '| 宿主 | C80 student motor MSE | BF16 reader 读 student future 的 motor MSE | 损伤的 context ESS | 选中 16 层覆盖的损伤 |',
              '|---|---:|---:|---:|---:|']
    for host, r in sel['hosts'].items():
        lines.append(f"| {host} | {r['student_first10motor_mse']:.5f} | {r['bf16_reader_first10motor_mse']:.5f} | "
                     f"{r['context_ess']:.1f}/{r['contexts']} | {r['coverage']:.3f} |")
    lines.append(f"\nBF16 点（第一阶段）单位噪声质量的 context ESS：{sel['bf16_point']['context_ess_unit_noise']:.1f}/80。")
    lines += ['', '## 选中层上的 proxy（参数加权）', '', '| 宿主 | 权重 | H_U | H̃（BF16 点） | H̃_op |', '|---|---|---:|---:|---:|']
    for host, p in prox.items():
        for a, label in (('U', 'baseline'), ('G', 'H̃'), ('O', 'H̃_op')):
            lines.append(f"| {host} | {label} | {p[a]['H_U']:.5f} | {p[a]['H_G']:.5f} | {p[a]['H_O']:.5f} |")
    lines += ['', '## held-out 40×2（只换选中的 16 层）', '',
              'baseline 行取第一/三阶段存下的 endpoint；本作业重算其第一个 endpoint 的最大相对差：'
              + '，'.join(f'{h} {v:.1e}' for h, v in s['ucheck_relative_max'].items()) + '。', '',
              '| 宿主 | 臂 | mixed loss | future 在 motor MSE 里的占比 | first10 motor | future NMSE | gripper RMSE |',
              '|---|---|---:|---:|---:|---:|---:|']
    names = dict(U='baseline', G='16 层用 H̃', O='16 层用 H̃_op', G_full='全模型 H̃（第一/三阶段）')
    for host, arms in held.items():
        for k in ('U', 'G', 'O', 'G_full'):
            o = arms[k]
            lines.append(f"| {host} | {names[k]} | {o['mixed_loss']:.5f} | {future_share(o):.3f} | {o['first10motor_rmse']:.4f} | "
                         f"{o['future_latent_nmse']:.4f} | {o['first10gripper_rmse']:.4f} |")
    lines += ['', '## 判定（预先登记）', '',
              f"规则：至少 {dec['min_hosts']} 个宿主上，H̃_op 臂相对 baseline 的 mixed loss 与 future 占比都下降，则推进全量。", '',
              '| 宿主 | H̃_op：mixed loss 降 | H̃_op：future 占比降 | H̃：mixed loss 降 | H̃：future 占比降 |',
              '|---|---|---|---|---|']
    yes = {True: '是', False: '否'}
    for host, r in dec['hosts'].items():
        lines.append(f"| {host} | {yes[r['O']['mixed_down']]} | {yes[r['O']['future_share_down']]} | "
                     f"{yes[r['G']['mixed_down']]} | {yes[r['G']['future_share_down']]} |")
    lines.append(f"\n{dec['passed']}/3 个宿主满足，结论：{'推进全量' if dec['go'] else '不推进全量'}。")
    return '\n'.join(lines) + '\n'


def full_report(s):
    names = dict(U='baseline（H_U）', G='H̃（BF16 点）', O='H̃_op（工作点）')
    lines = ['# 工作点曲率 H̃_op 全量：三个 W2 宿主', '',
             f"作业 `{s['pbs_jobid']}`。门 1 判定推进（{s['gate']['passed']}/3 个宿主满足）。每个宿主在自己的 baseline W2 模型上取 "
             "A + λZ + βK 的 GN 探针（K 用冻结 BF16 reader 读 student 自己的 future），H̃_op 与第一阶段同形（γ = 0.1），614 层全部用宿主"
             "按 baseline 种子重量化，不训练、不加 bit。", '',
             '## 闭环（LIBERO-Plus 280；BF16 216）', '',
             '| 宿主 | baseline | H̃（BF16 点） | H̃_op | H̃_op 对 baseline 胜/负（p） | H̃_op 对 H̃ 胜/负（p） |',
             '|---|---:|---:|---:|---:|---:|']
    for host, r in s['hosts'].items():
        lp_ = r['loop']
        if lp_:
            u, g = lp_['vs_U'], lp_['vs_G']
            lines.append(f"| {host} | {lp_['U']} | {lp_['G']} | {lp_['O']} | {u['wins']}/{u['losses']}（{u['p']:.2f}） | "
                         f"{g['wins']}/{g['losses']}（{g['p']:.2f}） |")
    pu, pg = s['pooled']['vs_U'], s['pooled']['vs_G']
    lines.append(f"\n三宿主合计：H̃_op 对 baseline {pu['wins']} 胜 / {pu['losses']} 负（p = {pu['p']:.2f}），"
                 f"对 H̃ {pg['wins']} 胜 / {pg['losses']} 负（p = {pg['p']:.2f}）。")
    lines += ['', '## held-out 40×2', '',
              '| 宿主 | 臂 | mixed loss | future 在 motor MSE 里的占比 | first10 motor | future NMSE | gripper RMSE | 换回 BF16 future 后 motor |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for host, r in s['hosts'].items():
        for k in ('U', 'G', 'O'):
            o = r['heldout'].get(k)
            if o:
                lines.append(f"| {host} | {names[k]} | {o['mixed_loss']:.5f} | {future_share(o):.3f} | {o['first10motor_rmse']:.4f} | "
                             f"{o['future_latent_nmse']:.4f} | {o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} |")
    lines += ['', '## proxy（参数加权）', '', '| 宿主 | 权重 | bpw | H_U | H̃ | H̃_op |', '|---|---|---:|---:|---:|---:|']
    for host, r in s['hosts'].items():
        q = r['quant']
        for k in ('U', 'G', 'O'):
            p = q['proxies'][k]
            bpw = f"{q['bpw']:.3f}" if k == 'O' else '—'
            lines.append(f"| {host} | {names[k]} | {bpw} | {p['H_U']:.5f} | {p['H_G']:.5f} | {p['H_O']:.5f} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    cfg = wk.read(parser.parse_args().protocol)
    if cfg['protocol'] != wk.PROTOCOL:
        raise ValueError('Protocol identity differs')
    stage, shard, shards = os.environ['FW_STAGE'], int(os.environ['FW_SHARD']), int(os.environ['FW_SHARDS'])
    job = f"{os.environ['FW_JOB']}_{shard}"
    failure = Path(cfg['remote_root']) / 'results' / 'failures' / f'{job}.json'
    failure.unlink(missing_ok=True)
    try:
        GateWork(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
