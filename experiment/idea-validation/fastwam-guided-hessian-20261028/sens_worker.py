"""FLAQ module 1 (sens.json): counterfactual per-group sensitivity of A + lambda Z + beta K on a host's W2 model.

From the host's W2 weights (A16 baselines, H_U), each of 64 groups of Linears in turn takes the same host's W3 weights;
on the C80 queries (seed index 0) the student query gives A (first10 motor MSE, = mixed loss - lambda Z), Z (future
NMSE) and the gripper against the cached BF16 teacher, and the frozen BF16 reader reads the student's own future for K
(soft-bin KL, the beta K of every C2 run) and its first10 motor MSE. Configs none (all W2) and all (all W3) bracket
the groups. Steps (FW_STAGE, shard FW_SHARD of FW_SHARDS):
  sens   one process per GPU; the 66 configs split evenly; results/sens/<host>/configs/<config>.json
  final  once every config file exists: results/sens/<host>/{summary.json, selection.json, RESULTS.zh.md}; with files
         missing it records what is missing and exits 0 (the g1 race jobs each run it after their shard)
0 training, 0 new teacher queries.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import re
import json

import gate_worker as gw  # owns the allocation guard (via worker), the frozen-reader context and the masters
import torch

import lp_worker as lp
import worker as wk

HERE = Path(__file__).resolve().parent
SENS = json.loads((HERE / 'sens.json').read_text(encoding='utf-8'))
TERMS = ('A', 'Z', 'K', 'T')
EXPERTS = ('video', 'action')


def group_of(name):
    m = re.match(r'(video|action)_expert\.blocks\.(\d+)\.', name)
    if m:
        return f'{m.group(1)}.b{int(m.group(2)):02d}'
    if name == 'video_expert.head.head':
        return 'video.head'
    if name.startswith('video_expert.'):
        return 'video.cond'
    if name in ('action_expert.action_encoder', 'action_expert.head', 'proprio_encoder'):
        return 'action.io'
    if name.startswith('action_expert.'):
        return 'action.cond'
    raise ValueError(f'No group for {name}')


def groups(names):
    """Group -> Linear indices in the bank's order; the layout must be sens.json's 64 groups."""
    out = {}
    for i, n in enumerate(names):
        out.setdefault(group_of(n), []).append(i)
    want = {**{f'{e}.b{b:02d}': 10 for e in EXPERTS for b in range(30)},
            'video.cond': 5, 'video.head': 1, 'action.cond': 5, 'action.io': 3}
    sizes = {g: len(v) for g, v in out.items()}
    if sizes != want or len(want) != SENS['groups']['count']:
        raise ValueError(f'Group layout differs: {sorted(set(sizes.items()) ^ set(want.items()))}')
    return out


def config_list(grp):
    return ['none', 'all'] + sorted(grp)


def assigned(n, shard, shards, cover=(), parts=None):
    """Config indices of process `shard` of `shards`. With `cover` (shards of the `parts`-way split that this job
    takes over from cancelled jobs), the union of those shards' configs is re-split over this job's processes."""
    if not cover:
        return wk.owned(shard, shards, lambda i: 1.0, range(n))
    pool = sorted(i for s in cover for i in wk.owned(s, parts, lambda i: 1.0, range(n)))
    return pool[shard::shards]


def members(config, grp, n):
    return [] if config == 'none' else list(range(n)) if config == 'all' else grp[config]


def scalars(m):
    """The metrics that are numbers (floats or 0-dim tensors)."""
    out = {}
    for key, v in m.items():
        try:
            out[key] = float(v)
        except (TypeError, ValueError, RuntimeError):
            pass
    return out


def terms(rows, lam, beta):
    """case -> A, lambda Z, beta K and their sum T (all in loss units, so A + Z + K = T)."""
    out = {}
    for r in rows:
        z = lam * r['future_latent_nmse']
        k = beta * r['reader_kl']
        out[r['case_id']] = dict(A=r['mixed_loss'] - z, Z=z, K=k, T=r['mixed_loss'] + k)
    return out


def benefits(none, other):
    """case -> value(none) - value(other) per term: positive = the upgrade helps."""
    return {c: {t: none[c][t] - other[c][t] for t in TERMS} for c in none}


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def mean_se(values):
    values = list(values)
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return m, math.sqrt(var / len(values))


def ranks(values):
    """Average ranks (1 = smallest), ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out, i = [0.0] * len(values), 0
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
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def densities(ben, bits, cases, term):
    """group -> mean benefit over cases / extra Mbit."""
    return {g: mean(ben[g][c][term] for c in cases) / (bits[g] / 1e6) for g in ben}


def top(dens, k):
    return [g for g, _ in sorted(dens.items(), key=lambda x: (-x[1], x[0]))[:k]]


def compare(ben, bits, first, second, term, k):
    d1, d2 = densities(ben, bits, first, term), densities(ben, bits, second, term)
    gs = sorted(ben)
    return dict(overlap=len(set(top(d1, k)) & set(top(d2, k))) / k,
                rho=spearman([d1[g] for g in gs], [d2[g] for g in gs]), top_first=top(d1, k), top_second=top(d2, k))


def suite_split(cases, suite, rng):
    first = []
    for s in sorted({suite[c] for c in cases}):
        mine = sorted(c for c in cases if suite[c] == s)
        rng.shuffle(mine)
        first += mine[:len(mine) // 2]
    return sorted(first), sorted(set(cases) - set(first))


def allocate(dens, means, bits, budget):
    """Greedy by density; a group enters if its mean benefit is positive and its extra bits still fit."""
    used, chosen = 0, []
    for g, _ in sorted(dens.items(), key=lambda x: (-x[1], x[0])):
        if means[g] > 0 and used + bits[g] <= budget:
            chosen.append(g)
            used += bits[g]
    return chosen, used


def analyze(cfg_rows, grp_bits, weights, base_bpw, suite, lam, beta):
    """cfg_rows: config -> rows; grp_bits: group -> extra bits; weights: target weight count. Everything the summary,
    the selection and the report need."""
    cfg, st = SENS['stability'], {}
    none = terms(cfg_rows['none'], lam, beta)
    cases = sorted(none)
    gs = sorted(grp_bits)
    ben = {g: benefits(none, terms(cfg_rows[g], lam, beta)) for g in gs}
    allben = benefits(none, terms(cfg_rows['all'], lam, beta))
    rows = {}
    for g in gs:
        row = dict(extra_bits=grp_bits[g], weight_share=grp_bits[g] / weights)
        for t in TERMS:
            m, se = mean_se(ben[g][c][t] for c in cases)
            row[t] = dict(mean=m, se=se, density=m / (grp_bits[g] / 1e6))
        rows[g] = row
    even, odd = [c for c in cases if c % 2 == 0], [c for c in cases if c % 2 == 1]
    k = cfg['top_k']
    for term in ('T', 'A'):
        st[term] = dict(even_odd=compare(ben, grp_bits, even, odd, term, k))
        rng, draws = random.Random(0), []
        for _ in range(cfg['random_splits']):
            first, second = suite_split(cases, suite, rng)
            draws.append(compare(ben, grp_bits, first, second, term, k))
        for key in ('overlap', 'rho'):
            vals = sorted(d[key] for d in draws)
            st[term][f'random_{key}_mean'] = mean(vals)
            st[term][f'random_{key}_p10'] = vals[int(0.1 * len(vals))]
    selection = {}
    for name, term in (('total', 'T'), ('action', 'A')):
        dens = {g: rows[g][term]['density'] for g in gs}
        means = {g: rows[g][term]['mean'] for g in gs}
        selection[name] = {}
        for b in SENS['allocation']['budgets_bpw']:
            chosen, used = allocate(dens, means, grp_bits, b * weights)
            selection[name][f'{b:g}'] = dict(groups=chosen, extra_bits=used, bpw=base_bpw + used / weights)
    experts = {}
    for e in EXPERTS:
        mine = [g for g in gs if g.startswith(e + '.')]
        experts[e] = {t: sum(rows[g][t]['mean'] for g in mine) for t in TERMS}
        experts[e]['weight_share'] = sum(rows[g]['weight_share'] for g in mine)
    all_mean = {t: mean(allben[c][t] for c in cases) for t in TERMS}
    absolute = {cfg_name: {t: mean(terms(cfg_rows[cfg_name], lam, beta)[c][t] for c in cases) for t in TERMS}
                for cfg_name in ('none', 'all')}
    raw = {cfg_name: {key: mean(r[key] for r in cfg_rows[cfg_name])
                      for key in ('future_latent_nmse', 'reader_kl', 'first10motor_rmse', 'reader_first10motor_mse')}
           for cfg_name in ('none', 'all')}
    stable = st['T']['even_odd']['overlap'] >= SENS['reading']['stable_overlap_min']
    return dict(groups=rows, stability=st, selection=selection, experts=experts, all=all_mean, absolute=absolute, raw=raw,
                additivity=dict(sum_single={t: sum(rows[g][t]['mean'] for g in gs) for t in TERMS}, all=all_mean),
                significant={t: sum(rows[g][t]['mean'] > 2 * rows[g][t]['se'] for g in gs) for t in TERMS},
                rank_total=top({g: rows[g]['T']['density'] for g in gs}, len(gs)),
                rank_action=top({g: rows[g]['A']['density'] for g in gs}, len(gs)),
                decision=dict(stable=stable, overlap=st['T']['even_odd']['overlap'],
                              min=SENS['reading']['stable_overlap_min'], go=stable))


def anchor_check(none_rows, fit_rows, tol):
    fit = {r['case_id']: r for r in fit_rows}
    worst = 0.0
    for r in none_rows:
        f = fit[r['case_id']]
        for key in ('mixed_loss', 'future_latent_nmse'):
            worst = max(worst, abs(r[key] - f[key]) / max(abs(f[key]), 1e-30))
    return dict(worst_relative=worst, passed=worst <= tol, cases=len(none_rows))


class SensWork(gw.GateWork):
    def setup_host(self):
        self.hname = os.environ.get('FW_HOSTS') or SENS['host']  # environment.sh passes FW_HOSTS into the container
        self.hcfg = SENS['hosts'][self.hname]
        self.sdir = self.base / 'sens' / self.hname
        for which in ('base', 'upgrade'):
            if not wk.read(self.arm_dir(which) / 'COMPLETE.json').get('complete'):
                raise RuntimeError(f'{self.hcfg[which + "_dir"]} is incomplete')

    def arm_dir(self, which):
        return self.baseline / 'quantized' / self.hcfg[f'{which}_dir']

    def put(self, which, indices):
        """The stored weights of `which` (base W2 / upgrade W3) into the given Linears."""
        root = self.arm_dir(which)
        with torch.no_grad():
            for i in indices:
                saved = torch.load(root / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                name = self.bank.names[i]
                if saved['module'] != name:
                    raise RuntimeError(f'Stored weight identity differs at {i}: {saved["module"]} != {name}')
                weight = self.bank.modules[name].weight
                weight.data.copy_(saved['weight'].to(weight.device))

    def measure(self, case, reader):
        """The C80 query at seed index 0 with the installed weights, then the frozen BF16 reader on its future."""
        from reader_loss import reader_terms
        target = self.teacher(case, 0)
        stats = {'action': self.p['reader_source']['action_stats']}
        with torch.no_grad():
            action, future, _ = self.query(case, 0)
            m = self.metrics(action, future, target)
            value, description = self.datum(case)
            self.runtime.description = description
            read, _ = reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], future.detach().float(),
                             checkpointed=False)
            self.count('reader_queries')
            k = reader_terms(torch, read, target['action'], stats, self.p['reader_source']['config'])['reader_kl']
        d = read.detach().float().cpu() - target['action'].float().cpu()
        return dict(case_id=case, suite=self.rows[case]['suite'], **scalars(m),
                    reader_kl=float(k), reader_first10motor_mse=float(d[:10, :6].square().mean()))

    def run_config(self, config, idx, reader):
        path = self.sdir / 'configs' / f'{config}.json'
        if path.exists():
            return
        self.put('upgrade', idx)
        try:
            rows = []
            for case in lp.CASES:
                self.enough()
                rows.append(self.measure(case, reader))
                self.progress('sens', host=self.hname, config=config, contexts=len(rows))
        finally:
            self.put('base', idx)
        wk.save(path, dict(protocol=wk.PROTOCOL, host=self.hname, config=config, pbs_jobid=self.job,
                           modules=[self.bank.names[i] for i in idx], rows=rows))

    def sens(self, shard, shards):
        self.setup_host()
        self.load()
        self.capture_masters()
        names = self.bank.names
        grp = groups(names)
        configs = config_list(grp)
        cover = [int(x) for x in os.environ.get('FW_ARMS', '').split(':') if x]  # run_sens.pbs: shards taken over
        mine = [configs[i] for i in assigned(len(configs), shard, shards, cover, SENS['allocation_jobs']['shards'])]
        self.put('base', range(len(names)))
        with self.bf16_reader() as reader:
            start = self.measure(lp.CASES[0], reader)
            for config in mine:
                self.run_config(config, members(config, grp, len(names)), reader)
            end = self.measure(lp.CASES[0], reader)
        same = start == end
        wk.save(self.sdir / 'shards' / f'{self.job_name}.json',
                dict(protocol=wk.PROTOCOL, host=self.hname, pbs_jobid=self.job, shard=shard, shards=shards,
                     cover=cover, configs=mine, start=start, end=end, restore_exact=same))
        if not same:
            raise RuntimeError('Base weights were not restored exactly')

    def final(self):
        self.setup_host()
        if (self.sdir / 'summary.json').exists():
            return
        names = [r['module'] for r in wk.read(self.baseline / 'prep' / 'modules.json')]
        grp = groups(names)
        missing = [c for c in config_list(grp) if not (self.sdir / 'configs' / f'{c}.json').exists()]
        if missing:
            wk.save(self.sdir / 'waiting' / f'{self.job_name}.json',
                    dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, missing=missing))
            return
        rows = {c: wk.read(self.sdir / 'configs' / f'{c}.json')['rows'] for c in config_list(grp)}
        lo, hi = self.arm_dir('base'), self.arm_dir('upgrade')
        extra = [wk.read(hi / 'stats' / f'{i:04d}.json')['bits_total'] - wk.read(lo / 'stats' / f'{i:04d}.json')['bits_total']
                 for i in range(len(names))]
        base = wk.read(lo / 'COMPLETE.json')
        grp_bits = {g: sum(extra[i] for i in idx) for g, idx in grp.items()}
        loss = self.cfg['loss']
        suite = {r['case_id']: r['suite'] for r in rows['none']}
        out = analyze(rows, grp_bits, base['target_weights'], base['bpw'], suite, loss['lambda'], loss['beta'])
        anchor = anchor_check(rows['none'], wk.read(self.base / self.hcfg['fit_anchor'])['rows'], SENS['checks']['anchor_tol'])
        shard_rows = [wk.read(p) for p in sorted((self.sdir / 'shards').glob('*.json'))]
        first = rows['none'][0]
        restore = dict(shards=len(shard_rows), exact=all(s['restore_exact'] for s in shard_rows),
                       start_equals_none=all(s['start'] == first for s in shard_rows))
        valid = anchor['passed'] and restore['exact'] and restore['start_equals_none']
        out['decision'].update(valid=valid, go=out['decision']['go'] and valid)
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, host=self.hname,
                       sens=SENS, loss=dict(lam=loss['lambda'], beta=loss['beta']), base_bpw=base['bpw'],
                       upgrade_bpw=wk.read(hi / 'COMPLETE.json')['bpw'], target_weights=base['target_weights'],
                       anchor=anchor, restore=restore, new_teacher_queries=0, **out)
        wk.save(self.sdir / 'selection.json', dict(protocol=wk.PROTOCOL, host=self.hname, pbs_jobid=self.job,
                                                   groups={g: [names[i] for i in idx] for g, idx in grp.items()},
                                                   selection=out['selection']))
        wk.save(self.sdir / 'summary.json', summary)
        (self.sdir / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')

    def stage(self, name, shard, shards):
        if name == 'sens':
            self.sens(shard, shards)
        elif name == 'final':
            self.final()
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


def fmt(x, digits=3):
    return f'{x:.{digits}g}'


def report(s):
    dec, st, g = s['decision'], s['stability'], s['groups']
    lam, beta = s['loss']['lam'], s['loss']['beta']
    host = s['host']
    lines = [f'# FLAQ 模块 1：{host} W2 上逐组换成 W3，测 C1 目标的敏感度', '',
             f"作业 `{s['pbs_jobid']}`。做法：从 {host} W2 出发，每次只把一组 Linear 换成同一宿主的 W3 权重，"
             "在 C80（80 个标定场景，seed 0）上跑一遍，记录这一组升级后三项各降了多少：", '',
             '- **A**：执行动作的 first10 motor MSE（动作误差）。',
             f'- **Z**：λ × 想象 future 的 NMSE（λ = {lam:.5g}）。',
             f'- **K**：β × 冻结 BF16 reader 读学生 future 得到的动作分布和 teacher 的 KL（β = {beta:.5g}），'
             '即 future 的误差传到动作上的那部分。',
             '- **T = A + Z + K**：C1 的总目标。', '',
             '分组：两个专家各 30 个块（每块 10 个 Linear），加上视频条件层、视频 head、动作条件层、动作输入输出层，共 64 组。'
             '"密度"= 平均下降 ÷ 这组升级多花的比特（每 Mbit）。', '',
             '## 检查', '',
             f"- 全 W2 配置与之前 C80 拟合逐场景对照：最大相对差 {s['anchor']['worst_relative']:.2e}"
             f"（要求 ≤ {s['sens']['checks']['anchor_tol']:g}），{'通过' if s['anchor']['passed'] else '不通过'}。",
             f"- 每个进程开头和结尾各测一次 case 2000，换回 W2 后结果{'逐位相同' if s['restore']['exact'] else '不同'}；"
             f"开头的值{'等于' if s['restore']['start_equals_none'] else '不等于'}全 W2 配置的 case 2000。", '',
             '## 整体：全 W2 与全 W3', '',
             '| 配置 | bpw | A | λZ | βK | T | future NMSE | reader KL | reader 读学生 future 的 motor MSE |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, bpw in (('none', s['base_bpw']), ('all', s['upgrade_bpw'])):
        a, r = s['absolute'][name], s['raw'][name]
        label = '全 W2' if name == 'none' else '全 W3'
        lines.append(f"| {label} | {bpw:.3f} | {fmt(a['A'])} | {fmt(a['Z'])} | {fmt(a['K'])} | {fmt(a['T'])} | "
                     f"{r['future_latent_nmse']:.4f} | {r['reader_kl']:.4f} | {r['reader_first10motor_mse']:.5f} |")
    add = s['additivity']
    lines += ['', f"64 组单独升级的 T 下降加起来是 {fmt(add['sum_single']['T'])}，全部升级是 {fmt(add['all']['T'])}"
                  f"（比值 {add['sum_single']['T'] / add['all']['T']:.2f}；接近 1 说明各组的效果大致可以相加，按组分配比特是合理的）。", '',
              '## 稳定性（预先登记的判据）', '',
              f"把 C80 按 case 号奇偶分成两半，各自算一次密度排序，看前 {s['sens']['stability']['top_k']} 组有多少重合。", '',
              '| 目标 | 奇偶两半前 10 重合 | 奇偶 Spearman ρ | 200 次随机对半：重合均值 / 10% 分位 | ρ 均值 / 10% 分位 |',
              '|---|---:|---:|---:|---:|']
    for term, label in (('T', 'C1 总目标 T'), ('A', '只看动作 A')):
        x = st[term]
        lines.append(f"| {label} | {x['even_odd']['overlap']:.1f} | {x['even_odd']['rho']:.2f} | "
                     f"{x['random_overlap_mean']:.2f} / {x['random_overlap_p10']:.1f} | "
                     f"{x['random_rho_mean']:.2f} / {x['random_rho_p10']:.2f} |")
    lines += ['', f"判定：T 的奇偶重合 {dec['overlap']:.1f}，要求 ≥ {dec['min']}；检查{'全部通过' if dec['valid'] else '没有全部通过'}。"
                  f"结论：{'稳定，进入步骤 2（按这个排序分配比特，在 held-out 上和对照比较）' if dec['go'] else '不进入步骤 2'}。", '',
              f"有多少组的平均下降超过 2 倍标准误：T {s['significant']['T']} 组，A {s['significant']['A']} 组，"
              f"Z {s['significant']['Z']} 组，K {s['significant']['K']} 组（共 64 组）。", '',
              '## 两个专家各承担多少', '',
              '| 专家 | 权重占比 | A 下降 | λZ 下降 | βK 下降 | T 下降 | T 中来自 Z + K 的比例 |', '|---|---:|---:|---:|---:|---:|---:|']
    for e, label in (('video', '视频专家'), ('action', '动作专家')):
        x = s['experts'][e]
        share = (x['Z'] + x['K']) / x['T'] if x['T'] else float('nan')
        lines.append(f"| {label} | {x['weight_share']:.3f} | {fmt(x['A'])} | {fmt(x['Z'])} | {fmt(x['K'])} | {fmt(x['T'])} | {share:.2f} |")
    lines += ['', '## 密度最高的 15 组（按 C1 总目标 T）', '',
              '| 名次 | 组 | 权重占比 | T 密度 | A 下降 | λZ 下降 | βK 下降 | T 下降 ± 标准误 | 按 A 排第几 |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    a_rank = {grp: i + 1 for i, grp in enumerate(s['rank_action'])}
    for i, grp in enumerate(s['rank_total'][:15]):
        r = g[grp]
        lines.append(f"| {i + 1} | {grp} | {r['weight_share']:.4f} | {fmt(r['T']['density'])} | {fmt(r['A']['mean'])} | "
                     f"{fmt(r['Z']['mean'])} | {fmt(r['K']['mean'])} | {fmt(r['T']['mean'])} ± {fmt(r['T']['se'], 2)} | {a_rank[grp]} |")
    lines += ['', '## 分配结果（给步骤 2 用）', '',
              '每个预算下按密度从高到低贪心地选组，直到额外比特用完。"C1" 按 T 排，"只看动作" 按 A 排。', '',
              '| 预算（+bpw） | 目标 | 实际 bpw | 选中的组 |', '|---:|---|---:|---|']
    for b in s['sens']['allocation']['budgets_bpw']:
        for name, label in (('total', 'C1（T）'), ('action', '只看动作（A）')):
            x = s['selection'][name][f'{b:g}']
            lines.append(f"| {b:g} | {label} | {x['bpw']:.3f} | {', '.join(x['groups'])} |")
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
        SensWork(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
