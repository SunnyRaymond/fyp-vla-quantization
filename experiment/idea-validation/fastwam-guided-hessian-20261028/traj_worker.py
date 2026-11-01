"""Retrospective test of a trajectory-coherent layer objective (traj.json): on the C80 queries H_U was collected on, per
video-expert Linear the step-accumulated Hessian H_J = sum_token (sum_i delta_i x_i)(sum_i delta_i x_i)^T and its
incoherent counterpart H_S = sum_token sum_i delta_i^2 x_i x_i^T; then per saved W2 arm the normalized proxies under
H_U, H_S and H_J, and how well each ranks the arms' held-out future NMSE. One stage, one GPU -> results/traj/.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import worker as wk  # noqa: F401  (imports sampler first: the PBS allocation guard)
import lp_worker as lp
import torch

HERE = Path(__file__).resolve().parent
TRAJ = json.loads((HERE / 'traj.json').read_text(encoding='utf-8'))
STEPS = 10
VARIANTS = ('U', 'S', 'J')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Accumulator:
    """Per watched layer over the calls of each query. Stepped layer (STEPS + 1 calls): call i < STEPS adds
    delta_i^2 x_i^T x_i to H_S and delta_i x_i to the running token sum a; after call STEPS - 1, H_J += a^T a; the last
    call (prefill) is left out. Unstepped layer: every call's x^T x goes to H_S, which also stands for H_J. Check layers
    sum every call's x^T x into H_all. Rows count the tokens behind each sum."""

    def __init__(self, deltas, counts, device):
        if len(deltas) != STEPS:
            raise ValueError(f'Expected {STEPS} step deltas, got {len(deltas)}')
        self.deltas, self.counts, self.device = list(deltas), dict(counts), device
        self.state, self.check, self.calls = {}, {}, {}

    def stepped(self, name):
        return self.counts[name] == STEPS + 1

    def begin(self, sizes, check=None):
        self.state = {n: dict(hs=torch.zeros(k, k, device=self.device),
                              hj=torch.zeros(k, k, device=self.device) if self.stepped(n) else None,
                              rows_s=0, rows_j=0, acc=None) for n, k in sizes.items()}
        self.check = {n: dict(h=torch.zeros(k, k, device=self.device), rows=0) for n, k in (check or {}).items()}

    def start(self):
        self.calls = {n: 0 for n in set(self.state) | set(self.check)}

    def observe(self, name, x):
        if name not in self.calls:
            return
        n = self.calls[name]
        self.calls[name] = n + 1
        flat = x.detach().float().reshape(-1, x.shape[-1])
        c = self.check.get(name)
        if c is not None:
            c['h'].addmm_(flat.t(), flat)
            c['rows'] += flat.shape[0]
        s = self.state.get(name)
        if s is None:
            return
        if not self.stepped(name):
            s['hs'].addmm_(flat.t(), flat)
            s['rows_s'] += flat.shape[0]
            return
        if n >= STEPS:
            return
        d = self.deltas[n]
        s['hs'].addmm_(flat.t(), flat, alpha=d * d)
        s['rows_s'] += flat.shape[0]
        if n == 0:
            s['acc'] = flat * d
        else:
            if s['acc'] is None or s['acc'].shape != flat.shape:
                raise RuntimeError(f'{name}: step {n} input rows differ from step 0')
            s['acc'].add_(flat, alpha=d)
        if n == STEPS - 1:
            a, s['acc'] = s['acc'], None
            s['hj'].addmm_(a.t(), a)
            s['rows_j'] += a.shape[0]

    def finish(self):
        bad = {n: c for n, c in self.calls.items() if c != self.counts[n]}
        if bad:
            raise RuntimeError(f'Call counts differ from the dry query: {dict(list(bad.items())[:5])}')
        if any(s['acc'] is not None for s in self.state.values()):
            raise RuntimeError('A step accumulator was left open')

    def raw(self, name, variant):
        s = self.state[name]
        return s['hj'] if variant == 'J' and s['hj'] is not None else s['hs']

    def mean(self, name, variant):
        s = self.state[name]
        if variant == 'J' and s['hj'] is not None:
            return s['hj'] / max(s['rows_j'], 1)
        return s['hs'] / max(s['rows_s'], 1)


def energy(e, h):
    return float(((e @ h) * e).sum())


def ranks(values):
    """Average ranks (1-based), ties share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None  # undefined when one side is constant


def aggregate(layers, arms, shapes):
    """layers[name][arm][variant] = p; per arm and variant the weight-count-weighted and the plain mean over layers."""
    out = {}
    names = list(layers)
    weights = {n: shapes[n][0] * shapes[n][1] for n in names}
    total = sum(weights.values())
    for arm in arms:
        out[arm] = {}
        for v in VARIANTS:
            out[arm][v] = dict(weighted=sum(layers[n][arm][v] * weights[n] for n in names) / total,
                               plain=sum(layers[n][arm][v] for n in names) / len(names))
    return out


def ranking(agg, metrics, arms, key='weighted'):
    """Spearman rho of each variant's proxy against each held-out metric (and the closed loop where known)."""
    out = {}
    for metric in ('future_latent_nmse', 'mixed_loss', 'first10motor_rmse'):
        out[metric] = {v: spearman([agg[a][v][key] for a in arms], [metrics[a][metric] for a in arms]) for v in VARIANTS}
    loop = [a for a in arms if a in TRAJ['closed_loop']]
    if len(loop) >= 3:  # more successes = better: rank against minus successes so that + means the proxy agrees
        out['closed_loop'] = {v: spearman([agg[a][v][key] for a in loop], [-TRAJ['closed_loop'][a] for a in loop])
                              for v in VARIANTS}
        out['closed_loop_arms'] = loop
    return out


def decide(primary, rule):
    rho = primary['future_latent_nmse']
    if any(rho[v] is None for v in VARIANTS):
        return dict(rho=rho, go=False, coherence=False)
    go = rho['J'] >= rule['go_rho_min'] and rho['J'] - rho['U'] >= rule['go_gain_min']
    return dict(rho=rho, go=bool(go), coherence=bool(go and rho['J'] - rho['S'] >= rule['coherence_gain_min']))


class TrajWork(wk.Work):
    def __init__(self, cfg, job):
        super().__init__(cfg, job)
        self.out = self.base / 'traj'
        self.counting, self.dry_counts, self.acc, self.factors, self.selection = False, Counter(), None, {}, None

    # ---------------- queries ----------------
    def video_names(self):
        return [n for n in self.bank.names if n.startswith('video_expert')]

    def watch(self):
        for name in self.video_names():
            mod = self.bank.modules[name]
            plain = mod.forward

            def run(x, _name=name, _plain=plain):
                if self.counting:
                    self.dry_counts[_name] += 1
                elif self.acc is not None:
                    self.acc.observe(_name, x)
                return _plain(x)
            mod.forward = run

    def infer(self, case):
        value, description = self.datum(case)
        self.runtime.description = description
        self.count('formal_infer_queries')
        self.runtime.infer(value, self.rows[case]['sampler_seeds'][0], measure=False)

    def dry(self):
        """One C80 query: the video scheduler's deltas and each video Linear's calls per query."""
        sched = self.runtime.model.infer_video_scheduler
        existed, previous, deltas = 'step' in getattr(sched, '__dict__', {}), getattr(sched, '__dict__', {}).get('step'), []
        original = sched.step

        def step(model_output, delta, sample):
            deltas.append(float(delta))
            return original(model_output, delta, sample)
        sched.step = step
        self.dry_counts.clear()
        self.counting = True
        try:
            self.infer(lp.CASES[0])
        finally:
            self.counting = False
            if existed:
                sched.step = previous
            else:
                del sched.step
        counts = {n: self.dry_counts.get(n, 0) for n in self.video_names()}
        if len(deltas) != STEPS or any(c == 0 for c in counts.values()):
            raise RuntimeError(f'Dry query: {len(deltas)} video steps, {sum(c == 0 for c in counts.values())} unused layers')
        return deltas, counts

    # ---------------- arms ----------------
    def selected(self, host):
        if self.selection is None:
            self.selection = wk.read(self.base / 'gate' / 'selection.json')['hosts']
        return set(self.selection[host]['layers'])

    def arm_weight(self, arm, i, name):
        spec = TRAJ['arms'][arm]
        if spec['kind'] == 'file':
            root = (self.baseline if spec['root'] == 'baseline' else self.base) / 'quantized' / spec['dir']
            saved = torch.load(root / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
            if saved['module'] != name:
                raise RuntimeError(f'{arm}: stored weight identity differs at {i}')
            return saved['weight'].to(DEVICE, torch.float32)
        if spec['kind'] == 'gate':
            if i not in self.selected(spec['gate_host']):
                return self.arm_weight(spec['base'], i, name)
            if spec['selected'] == 'gate':
                saved = torch.load(self.base / 'gate' / spec['gate_host'] / 'weights' / f'{i:04d}.pt', map_location='cpu',
                                   weights_only=True)
                if saved['module'] != name:
                    raise RuntimeError(f'{arm}: stored weight identity differs at {i}')
                return saved['weight'].to(DEVICE, torch.float32)
            return self.arm_weight(spec['selected'], i, name)
        if spec['kind'] == 'lowrank':
            if arm not in self.factors:
                self.factors[arm] = torch.load(self.base / 'lowrank' / arm / 'factors.pt', map_location='cpu',
                                               weights_only=True)
            f = self.factors[arm][name]
            return self.arm_weight(spec['base'], i, name) + f['U'].to(DEVICE, torch.float32) @ f['V'].to(DEVICE, torch.float32)
        raise ValueError(spec['kind'])

    def gate_layers(self, arm):
        spec = TRAJ['arms'][arm]
        if spec['kind'] != 'gate':
            return None
        return self.selected(spec['gate_host'])

    # ---------------- passes ----------------
    def proxies(self, name, i):
        """Per arm and variant the normalized proxy of this layer, kappa for each arm and the master; saves H_J."""
        prep = self.baseline / 'prep'
        master = torch.load(prep / 'masters' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
        hu = torch.load(prep / 'hessians' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
        if master['module'] != name or hu['module'] != name:
            raise RuntimeError(f'Prep identity differs at {i}: {name}')
        m = master['weight'].to(DEVICE, torch.float32)
        hs = {'U': hu['hessian'].to(DEVICE, torch.float32), 'S': self.acc.raw(name, 'S'), 'J': self.acc.raw(name, 'J')}
        norm = {v: energy(m, h) for v, h in hs.items()}
        row = dict(index=i, shape=list(m.shape), stepped=self.acc.stepped(name),
                   kappa_master=energy(m, hs['J']) / max(energy(m, hs['S']), 1e-30), arms={})
        cache = {}
        for arm, spec in TRAJ['arms'].items():
            gate = self.gate_layers(arm)
            if gate is not None and i not in gate and spec['base'] in cache:
                row['arms'][arm] = dict(row['arms'][spec['base']])
                continue
            e = self.arm_weight(arm, i, name) - m
            raw = {v: energy(e, h) for v, h in hs.items()}
            row['arms'][arm] = dict({v: raw[v] / max(norm[v], 1e-30) for v in VARIANTS},
                                    kappa=raw['J'] / max(raw['S'], 1e-30))
            cache[arm] = True
            del e
        if row['stepped']:
            self.tsave(self.out / 'hessians' / f'{i:04d}.pt',
                       dict(module=name, rows=self.acc.state[name]['rows_j'], hessian=self.acc.mean(name, 'J').cpu(),
                            kind='traj'))
        del hs, m
        return row

    def collect(self, deltas, counts):
        """Memory passes over the C80 queries; after each pass the proxies of its layers. Returns per-layer rows and the
        H_all check."""
        names = self.video_names()
        index = {n: i for i, n in enumerate(self.bank.names)}
        sizes = {n: self.bank.modules[n].in_features for n in names}
        checks = {n: sizes[n] for n in TRAJ['checks']['h_all_layers']}
        budget = self.memory_budget() - sum(4 * k * k for k in checks.values())
        passes, current, used = [], [], 0
        for n in sorted(names, key=lambda n: (-sizes[n], index[n])):
            size = 4 * sizes[n] ** 2 * (2 if counts[n] == STEPS + 1 else 1)
            if current and used + size > budget:
                passes.append(current)
                current, used = [], 0
            current.append(n)
            used += size
        if current:
            passes.append(current)
        self.progress('traj_plan', passes=len(passes), budget_gb=budget / 2 ** 30)
        layers, check = {}, None
        for number, group in enumerate(passes):
            self.acc = Accumulator(deltas, counts, DEVICE)
            with torch.inference_mode():
                self.acc.begin({n: sizes[n] for n in group}, checks if number == 0 else {})
                for k, case in enumerate(lp.CASES):
                    self.enough()
                    self.acc.start()
                    self.infer(case)
                    self.acc.finish()
                    if k % 20 == 19:
                        self.progress('traj_collect', pass_number=number + 1, passes=len(passes), contexts=k + 1)
                if number == 0:
                    check = self.h_all_check()
                for n in group:
                    layers[n] = self.proxies(n, index[n])
                self.progress('traj_proxy', pass_number=number + 1, passes=len(passes), layers=len(layers))
            self.acc = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return layers, check

    def memory_budget(self):
        """Bytes free for the accumulators: the GPU's memory minus the peak a query reserved and 8 GB of headroom."""
        torch.cuda.empty_cache()
        return (torch.cuda.get_device_properties(0).total_memory - torch.cuda.max_memory_reserved() - 8 * 2 ** 30)

    def h_all_check(self):
        rows = []
        for name, c in self.acc.check.items():
            i = self.bank.names.index(name)
            hu = torch.load(self.baseline / 'prep' / 'hessians' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
            ref = hu['hessian'].to(DEVICE, torch.float32)
            mine = c['h'] / max(c['rows'], 1)
            rows.append(dict(module=name, rows=c['rows'], baseline_rows=hu['rows'],
                             rel=float((mine - ref).norm() / ref.norm().clamp_min(1e-30))))
        out = dict(rows=rows, worst=max(r['rel'] for r in rows), limit=TRAJ['checks']['h_all_rel_max'])
        wk.save(self.out / 'h_all_check.json', out)
        if out['worst'] > out['limit']:
            raise RuntimeError(f'H_all does not reproduce the baselines H_U: {out}')
        return out

    # ---------------- stage ----------------
    def run(self):
        if (self.out / 'summary.json').exists():
            return
        self.load()
        self.watch()
        deltas, counts = self.dry()
        layers, check = self.collect(deltas, counts)
        names = self.video_names()
        arms = list(TRAJ['arms'])
        metrics = {a: wk.read(self.base / 'heldout' / f'{a}.json')['overall'] for a in arms}
        shapes = {n: layers[n]['shape'] for n in names}
        table = {n: {a: layers[n]['arms'][a] for a in arms} for n in names}
        agg = aggregate(table, arms, shapes)
        qtip = [a for a in arms if TRAJ['arms'][a]['host'] == 'QTIP']
        rank = {'qtip': ranking(agg, metrics, qtip), 'pooled': ranking(agg, metrics, arms),
                'qtip_plain': ranking(agg, metrics, qtip, 'plain')}
        stepped = [n for n in names if layers[n]['stepped']]
        kappa = {a: sorted(layers[n]['arms'][a]['kappa'] for n in stepped)[len(stepped) // 2] for a in arms}
        kappa_master = sorted(layers[n]['kappa_master'] for n in stepped)[len(stepped) // 2]
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, traj=TRAJ, deltas=deltas,
                       calls=dict(Counter(counts.values())), unstepped=[n for n in names if not layers[n]['stepped']],
                       h_all_check=check, metrics={a: {k: metrics[a][k] for k in ('future_latent_nmse', 'mixed_loss',
                                                                                  'first10motor_rmse')} for a in arms},
                       proxies=agg, ranking=rank, kappa_median=kappa, kappa_master_median=kappa_master,
                       decision=decide(rank['qtip'], TRAJ['reading']))
        wk.save(self.out / 'layers.json', {n: layers[n] for n in names})
        wk.save(self.out / 'summary.json', summary)
        (self.out / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')
        self.progress('stage_complete', stage='traj')


def fmt(x, spec='.3f'):
    return '—' if x is None else format(x, spec)


def report(s):
    t, r, d, agg, m = s['traj'], s['ranking'], s['decision'], s['proxies'], s['metrics']
    rule = t['reading']
    lines = ['# 轨迹相干层目标的回溯检验', '',
             f"作业 `{s['pbs_jobid']}`。tfdiag：W2 的速度误差在 10 个去噪步上几乎同向（累加能量是各步之和的 7.1 倍）。"
             "若同一份权重误差每步重复，一层对 future 的贡献约为 ΔW Σ_i Δ_i x_i，目标就是 H_traj。这里不重新量化，"
             "只在已有的 W2 臂上算 H_U、H_S、H_traj 下的 proxy，看谁能给 held-out future NMSE 排对序。", '',
             '## 核对', '',
             f"- 去噪步 Δ_i：{', '.join(f'{x:.4f}' for x in s['deltas'])}。每层每次 query 的调用次数分布：{s['calls']}；"
             f"不分步的层 {len(s['unstepped'])} 个：{', '.join(s['unstepped'][:6])}{' …' if len(s['unstepped']) > 6 else ''}。",
             f"- 在 {len(s['h_all_check']['rows'])} 层上用全部调用收集的 Hessian 和基线 H_U 的相对差最大 "
             f"{s['h_all_check']['worst']:.2e}（上限 {s['h_all_check']['limit']:g}）。", '',
             '## 各臂', '',
             '| 臂 | held-out future NMSE | mixed | proxy H_U | proxy H_S | proxy H_traj | κ 中位数 |', '|---|---:|---:|---:|---:|---:|---:|']
    for a in t['arms']:
        lines.append(f"| {a} | {m[a]['future_latent_nmse']:.5f} | {m[a]['mixed_loss']:.5f} | {agg[a]['U']['weighted']:.5f} | "
                     f"{agg[a]['S']['weighted']:.5f} | {agg[a]['J']['weighted']:.5f} | {s['kappa_median'][a]:.2f} |")
    lines += ['', f"κ = tr(ΔW H_traj ΔWᵀ) / tr(ΔW H_S ΔWᵀ)：各步误差完全同相时为 10，不相关时为 1。"
              f"主权重 W 自己的 κ 中位数 {s['kappa_master_median']:.2f}。", '',
              '## 排序（Spearman ρ，proxy 越大应对应指标越大）', '']
    for key, title in (('qtip', f"QTIP 的 {sum(t['arms'][a]['host'] == 'QTIP' for a in t['arms'])} 个臂（主判据，按权重数加权）"),
                       ('qtip_plain', 'QTIP 的臂，各层不加权平均'), ('pooled', f"全部 {len(t['arms'])} 个臂（报告）")):
        lines += [f'### {title}', '', '| 指标 | ρ H_U | ρ H_S | ρ H_traj |', '|---|---:|---:|---:|']
        for metric in ('future_latent_nmse', 'mixed_loss', 'first10motor_rmse', 'closed_loop'):
            if metric in r[key]:
                v = r[key][metric]
                lines.append(f"| {metric} | {fmt(v['U'])} | {fmt(v['S'])} | {fmt(v['J'])} |")
        lines.append('')
    lines += ['闭环一行用的是负成功数：ρ 为正表示 proxy 越大、成功越少。', '', '## 判读（预先登记）', '',
              f"- QTIP 臂上 future NMSE 的 ρ：H_U {fmt(d['rho']['U'])}，H_S {fmt(d['rho']['S'])}，H_traj {fmt(d['rho']['J'])}。",
              f"- go 需要 ρ_traj ≥ {rule['go_rho_min']} 且 ρ_traj − ρ_U ≥ {rule['go_gain_min']}："
              f"{'go，下一步用保存的 H_traj 重新量化 QTIP W2。' if d['go'] else 'no-go。'}"]
    if d['go']:
        lines.append(f"- 归因于跨步项需要 ρ_traj − ρ_S ≥ {rule['coherence_gain_min']}："
                     f"{'成立。' if d['coherence'] else '不成立，增益来自只保留去噪步的调用。'}")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    cfg = wk.read(parser.parse_args().protocol)
    if cfg['protocol'] != wk.PROTOCOL:
        raise ValueError('Protocol identity differs')
    job = f"{os.environ['FW_JOB']}_0"
    failure = Path(cfg['remote_root']) / 'results' / 'failures' / f'{job}.json'
    failure.unlink(missing_ok=True)
    try:
        TrajWork(cfg, job).run()
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
