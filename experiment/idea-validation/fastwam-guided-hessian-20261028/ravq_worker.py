"""RAVQ, uncalibrated (ravq.json): QTIP rounding under a C1-derived module-level bit budget.

Every target Linear gets a role (interface, time table, text, video body, action body) and each arm a choice per role:
BF16 (interfaces), an exact per-timestep table (time conditioning; installed as the BF16 master, booked at table bits),
or QTIP with K bits. K = 2 / 3 are the A16 baselines' QTIP_W2 / QTIP_W3 layers (same port, seeds and H_U); K = 1 / 4
are quantized here by ravq_qtip (the port with K in 1..4). RAVQU arms move the action-body bits to video-body layers at
no more total bits than their reference arm, greedily by the weight-averaged H_U proxy reduction per stored bit.
Steps (FW_STAGE, shard FW_SHARD of FW_SHARDS):
  quant     the (layer, K) the plan needs with K in {1, 4} -> results/ravq/qtip_K<K>/{weights,codes,stats}/<index>
  assemble  per arm results/quantized/<arm>/{weights/<index>.pt (symlink to its source), sources.json, COMPLETE.json}
  eval      held-out 40x2 + C1 swap, the arms split over the shards
  final     results/ravq/{summary.json, RESULTS.zh.md, DECISION.json}
0 training, 0 new teacher queries.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import gate_worker as gw  # owns the allocation guard (via worker) and the install routing
import torch

import ravq_qtip as rq
import worker as wk

HERE = Path(__file__).resolve().parent
RV = json.loads((HERE / 'ravq.json').read_text(encoding='utf-8'))
INTERFACES = tuple(RV['modules']['interfaces'])
TIME = tuple(RV['modules']['time_tables'])
ROLES = ('interface', 'time', 'text', 'video_body', 'action_body')
ROLE_COUNTS = dict(interface=4, time=6, text=4, video_body=300, action_body=300)
NEW_BITS = (1, 4)
QUANT_COST = {1: 0.6, 4: 2.0}  # relative Viterbi cost per weight against K = 2 / 3 (2^(K V) candidates per state group)
METRICS = ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse', 'first10gripper_rmse', 'first10motor_rmse_swap',
           'first10motor_mse', 'first10motor_mse_swap', 'full32motor_rmse', 'first10gripper_disagreement')


def role(name):
    if name in INTERFACES:
        return 'interface'
    if name in TIME:
        return 'time'
    if name.startswith('video_expert.blocks.'):
        return 'video_body'
    if name.startswith('action_expert.blocks.'):
        return 'action_body'
    if name.startswith(('video_expert.text_embedding.', 'action_expert.text_embedding.')):
        return 'text'
    raise ValueError(f'No RAVQ role for {name}')


def planned_arms():
    return {a: c for a, c in RV['arms'].items() if not c.get('reference')}


def loop_arms(decision):
    """The closed-loop job's arms: DECISION.json's loops, restricted to loop_job.run when it is set (run_ravq_loop.pbs)."""
    run = RV['loop_job'].get('run')
    return [a for a in decision['loops'] if run is None or a in run]


def table_bits(layers, distinct):
    """index -> table bits of the time-conditioning Linears: per expert 16 x timesteps x (out of time_embedding.2 +
    out of time_projection.1), booked on the expert's time_embedding.0 (0 on the other two)."""
    by = {L['module']: L for L in layers}
    out = {}
    for head in ('video_expert', 'action_expert'):
        outs = by[f'{head}.time_embedding.2']['shape'][0] + by[f'{head}.time_projection.1']['shape'][0]
        out[by[f'{head}.time_embedding.0']['index']] = 16 * distinct[head] * outs
        out[by[f'{head}.time_embedding.2']['index']] = 0
        out[by[f'{head}.time_projection.1']['index']] = 0
    return out


def layer_bits(L, choice, storage, table):
    m, n = L['shape']
    if choice == 'bf16':
        return 16 * m * n
    if choice == 'table':
        return table[L['index']]
    if choice in L['bits']:
        return L['bits'][choice]
    return storage(m, n, choice)


def proxy_of(L, k):
    """H_U proxy of the layer at K bits; K = 4 from the layer's own W2 -> W3 ratio (selection only)."""
    if k in L['proxy']:
        return L['proxy'][k]
    if k == 4:
        p2, p3 = L['proxy'][2], L['proxy'][3]
        return p3 * p3 / p2 if p2 > 0 else 0.0
    raise ValueError(k)


def base_choices(layers, c):
    out = []
    for L in layers:
        r = L['role']
        out.append('bf16' if r == 'interface' else 'table' if r == 'time' else c['action_body'] if r == 'action_body'
                   else c['video_body'] if r == 'video_body' else c['family'])
    return out


def plan(layers, arms, storage, table):
    """arm -> dict(choices per layer, bits_total, upgraded indices, budget, used). Upgrades fill the budget greedily."""
    out = {}
    for arm, c in arms.items():
        choices = base_choices(layers, c)
        out[arm] = dict(choices=choices, bits_total=sum(layer_bits(L, k, storage, table) for L, k in zip(layers, choices)),
                        upgraded=[], budget=None, used=0)
    for arm, c in arms.items():
        if 'upgrade' not in c:
            continue
        row, up, base = out[arm], c['upgrade'], c['video_body']
        budget = out[c['budget_of']]['bits_total'] - row['bits_total']
        cands = []
        for L in layers:
            if L['role'] != 'video_body':
                continue
            cost = layer_bits(L, up, storage, table) - layer_bits(L, base, storage, table)
            gain = L['shape'][0] * L['shape'][1] * (proxy_of(L, base) - proxy_of(L, up))
            if cost > 0 and gain > 0:
                cands.append((-gain / cost, L['index'], cost))
        left = budget
        for _, i, cost in sorted(cands):
            if cost <= left:
                row['choices'][i] = up
                row['upgraded'].append(i)
                left -= cost
        row['upgraded'].sort()
        row['budget'], row['used'] = budget, budget - left
        row['bits_total'] += row['used']
    return out


def quant_list(plans):
    """Sorted (index, K) pairs with K in NEW_BITS that any planned arm uses."""
    return sorted({(i, k) for p in plans.values() for i, k in enumerate(p['choices']) if k in NEW_BITS})


def summarize_arm(layers, choices, bits, proxies):
    """bpw, weighted H_U proxy and per-role bpw / proxy / weight share of one arm."""
    n_all = sum(L['shape'][0] * L['shape'][1] for L in layers)
    roles = {}
    for L, k, b, p in zip(layers, choices, bits, proxies):
        n = L['shape'][0] * L['shape'][1]
        r = roles.setdefault(L['role'], dict(weights=0, bits=0.0, proxy=0.0))
        r['weights'] += n
        r['bits'] += b
        r['proxy'] += p * n
    by_role = {r: dict(weight_share=v['weights'] / n_all, bpw=v['bits'] / v['weights'], proxy=v['proxy'] / v['weights'])
               for r, v in roles.items()}
    return dict(target_weights=n_all, bpw=sum(bits) / n_all,
                proxy_nmse_weighted=sum(p * L['shape'][0] * L['shape'][1] for L, p in zip(layers, proxies)) / n_all,
                by_role=by_role)


class RavqWork(gw.GateWork):
    def rv(self, *parts):
        return self.base.joinpath('ravq', *parts)

    def arm_root(self, arm):
        if arm in planned_arms():
            return self.base / 'quantized' / arm
        if arm in RV['arms']:  # reference: the A16 baselines' arm
            return self.baseline / 'quantized' / arm
        return super().arm_root(arm)

    # ---------------- plan ----------------
    def layers(self):
        """614 rows: index, module, role, shape, bits {2, 3} and H_U proxy {2, 3} of the A16 baselines' QTIP layers."""
        rows = []
        for i in range(614):
            s2, s3 = (wk.read(self.baseline / 'quantized' / a / 'stats' / f'{i:04d}.json') for a in ('QTIP_W2', 'QTIP_W3'))
            if s2['module'] != s3['module'] or s2['shape'] != s3['shape']:
                raise RuntimeError(f'QTIP_W2 / QTIP_W3 layer identity differs at {i}')
            rows.append(dict(index=i, module=s2['module'], role=role(s2['module']), shape=s2['shape'],
                             bits={2: s2['bits_total'], 3: s3['bits_total']}, proxy={2: s2['proxy_nmse'], 3: s3['proxy_nmse']}))
        counts = {r: sum(L['role'] == r for L in rows) for r in ROLES}
        if counts != ROLE_COUNTS:
            raise RuntimeError(f'Role layout differs: {counts}')
        return rows

    def distinct_timesteps(self):
        check = wk.read(self.base / 'lowrank' / 'tcond' / 'input_check.json')
        if check['worst'] > 0:
            raise RuntimeError('Time-conditioning inputs differ across contexts')
        return {h: check['modules'][f'{h}.time_embedding.0']['distinct_rows'] for h in ('video_expert', 'action_expert')}

    def make_plan(self):
        self.Q = self.method('qtip')
        self.layer_rows = self.layers()
        self.table = table_bits(self.layer_rows, self.distinct_timesteps())
        self.plans = plan(self.layer_rows, planned_arms(), self.Q.storage_bits, self.table)
        for arm, c in planned_arms().items():
            if 'budget_of' in c and self.plans[arm]['bits_total'] > self.plans[c['budget_of']]['bits_total']:
                raise RuntimeError(f'{arm} exceeds the bits of {c["budget_of"]}')
        path = self.rv('plan.json')
        if not path.exists():
            wk.save(path, dict(protocol=wk.PROTOCOL, table_bits={str(k): v for k, v in self.table.items()},
                               arms={a: dict(p, choices=[str(k) for k in p['choices']]) for a, p in self.plans.items()},
                               quant=quant_list(self.plans)))

    # ---------------- quant ----------------
    def quant(self, shard, shards):
        todo = quant_list(self.plans)
        n = {L['index']: L['shape'][0] * L['shape'][1] for L in self.layer_rows}
        mine = wk.owned(shard, shards, lambda j: n[todo[j][0]] * QUANT_COST[todo[j][1]], range(len(todo)))
        prep = self.baseline / 'prep'
        torch.use_deterministic_algorithms(False)  # as the baselines' quant_worker: ported ops must not be refused
        try:
            for done, j in enumerate(mine):
                i, k = todo[j]
                root = self.rv(f'qtip_K{k}')
                if (root / 'stats' / f'{i:04d}.json').exists():
                    continue
                self.enough()
                name = self.layer_rows[i]['module']
                saved = {key: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                         for key, path in (('master', prep / 'masters'), ('H', prep / 'hessians'))}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                w = saved['master']['weight'].to('cuda', torch.float32)
                h = saved['H']['hessian'].to('cuda', torch.float32)
                begin = time.monotonic()
                result = rq.quantize_layer(self.Q, w, h, k, seed=RV['seed_base'] + i)
                seconds = time.monotonic() - begin
                dq = result['dequant'].to('cuda', torch.float32)
                if tuple(dq.shape) != tuple(w.shape) or not bool(torch.isfinite(dq).all()):
                    raise RuntimeError(f'QTIP K={k} produced an invalid weight at {name}')
                d = dq - w
                proxy = float(((d @ h) * d).sum() / ((w @ h) * w).sum().clamp_min(1e-30))
                self.tsave(root / 'weights' / f'{i:04d}.pt', dict(module=name, weight=dq.to(torch.bfloat16).cpu()))
                self.tsave(root / 'codes' / f'{i:04d}.pt', dict(module=name, **result['codes']))
                wk.save(root / 'stats' / f'{i:04d}.json', dict(
                    index=i, module=name, shape=list(w.shape), K=k, bits_total=result['bits_total'],
                    bpw=result['bits_total'] / w.numel(), proxy_nmse=proxy, seconds=seconds, config=result['config']))
                del saved, w, h, result, dq, d
                torch.cuda.empty_cache()
                self.progress('quant', layers=done + 1, of=len(mine), index=i, K=k, proxy=proxy)
        finally:
            torch.use_deterministic_algorithms(True)

    # ---------------- assemble ----------------
    def layer_source(self, i, k):
        """(weight file, bits_total, H_U proxy) of layer i under choice k."""
        L = self.layer_rows[i]
        if k in ('bf16', 'table'):
            return self.baseline / 'prep' / 'masters' / f'{i:04d}.pt', layer_bits(L, k, None, self.table), 0.0
        if k in (2, 3):
            root = self.baseline / 'quantized' / ('QTIP_W2' if k == 2 else 'QTIP_W3')
            return root / 'weights' / f'{i:04d}.pt', L['bits'][k], L['proxy'][k]
        s = wk.read(self.rv(f'qtip_K{k}', 'stats', f'{i:04d}.json'))
        if s['module'] != L['module'] or s['bits_total'] != self.Q.storage_bits(*L['shape'], k):
            raise RuntimeError(f'K={k} layer {i} differs from the plan')
        return self.rv(f'qtip_K{k}', 'weights', f'{i:04d}.pt'), s['bits_total'], s['proxy_nmse']

    def assemble(self):
        missing = [(i, k) for i, k in quant_list(self.plans) if not self.rv(f'qtip_K{k}', 'stats', f'{i:04d}.json').exists()]
        if missing:
            raise RuntimeError(f'{len(missing)} quantized layers missing, e.g. {missing[:3]}')
        for arm, p in self.plans.items():
            root = self.arm_root(arm)
            if (root / 'COMPLETE.json').exists():
                continue
            rows, bits, proxies = [], [], []
            for i, k in enumerate(p['choices']):
                path, b, pr = self.layer_source(i, k)
                if not path.exists():
                    raise RuntimeError(f'Source missing for {arm} layer {i}: {path}')
                link = root / 'weights' / f'{i:04d}.pt'
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    if Path(os.readlink(link)) != path:
                        raise RuntimeError(f'{link} points elsewhere')
                else:
                    os.symlink(path, link)
                rows.append(dict(index=i, module=self.layer_rows[i]['module'], role=self.layer_rows[i]['role'],
                                 choice=str(k), source=str(path), bits_total=b, proxy_nmse=pr))
                bits.append(b)
                proxies.append(pr)
            if abs(sum(bits) - p['bits_total']) > 0.5:
                raise RuntimeError(f'{arm}: assembled bits {sum(bits)} differ from the plan {p["bits_total"]}')
            wk.save(root / 'sources.json', rows)
            summary = summarize_arm(self.layer_rows, p['choices'], bits, proxies)
            upgraded = sum(self.layer_rows[i]['shape'][0] * self.layer_rows[i]['shape'][1] for i in p['upgraded'])
            wk.save(root / 'COMPLETE.json', dict(
                protocol=wk.PROTOCOL, complete=True, pbs_jobid=self.job, arm=arm, method='qtip',
                bits=RV['arms'][arm]['family'], config=RV['arms'][arm],
                bits_total=sum(bits), budget=p['budget'], budget_used=p['used'], upgraded_layers=len(p['upgraded']),
                upgraded_weight_share=upgraded / summary['target_weights'], **summary))
        self.progress('assemble', arms=len(self.plans))

    # ---------------- eval ----------------
    def eval_arms(self):
        return [a for a in RV['arms'] if RV['arms'][a].get('reference')] + list(planned_arms())

    def eval(self, shard, shards):
        mine = self.eval_arms()[shard::shards]
        if not mine:
            return
        self.load()
        for arm in mine:
            self.evaluate(arm)
            self.progress('eval_done', arm=arm)

    # ---------------- final ----------------
    def final(self):
        held = {}
        for arm in ['QTIP_U_W2'] + self.eval_arms():
            path = self.base / 'heldout' / f'{arm}.json'
            if not path.exists():
                raise RuntimeError(f'Held-out missing: {arm}')
            held[arm] = wk.read(path)['overall']
        bits = {arm: wk.read(self.arm_root(arm) / 'COMPLETE.json') for arm in planned_arms()}
        for arm in ('QTIP_W3',):
            bits[arm] = wk.read(self.arm_root(arm) / 'COMPLETE.json')
        bits['QTIP_U_W2'] = wk.read(self.baseline / 'quantized' / 'QTIP_W2' / 'COMPLETE.json')
        measured4 = []
        for i, k in quant_list(self.plans):
            if k == 4:
                s = wk.read(self.rv('qtip_K4', 'stats', f'{i:04d}.json'))
                measured4.append(dict(index=i, module=s['module'], estimated=proxy_of(self.layer_rows[i], 4),
                                      measured=s['proxy_nmse'], w3=self.layer_rows[i]['proxy'][3]))
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, ravq=RV, heldout=held,
                       bits={a: {k: v[k] for k in ('bpw', 'proxy_nmse_weighted') if k in v}
                             | {k: v[k] for k in ('by_role', 'upgraded_layers', 'upgraded_weight_share', 'budget',
                                                  'budget_used', 'bits_total') if k in v}
                             for a, v in bits.items()},
                       k4=measured4, reading=reading(held, bits))
        wk.save(self.rv('summary.json'), summary)
        wk.save(self.rv('DECISION.json'), dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, **summary['reading']['decision']))
        self.rv('RESULTS.zh.md').write_text(report(summary), encoding='utf-8')
        if summary['reading']['checks']['future_identity_worst'] > 1e-6:
            raise RuntimeError(f"Action-only change moved the future: {summary['reading']['checks']}")

    # ---------------- closed loop ----------------
    def loop_final(self):
        """Each looped arm paired against its fair reference (the protocol arm) and the A16 baseline arm."""
        def episodes(root):
            return ({int(f.stem): wk.read(f)['success'] for f in (root / 'episodes').glob('*.json')}
                    if (root / 'DONE').exists() else None)
        arms = loop_arms(wk.read(self.rv('DECISION.json')))
        refs = dict(QTIP_W2I=['QTIP_W2'], RAVQU_W2=['QTIP_W2I', 'QTIP_W2'], QTIP_W3I=['QTIP_W3'],
                    RAVQU_W3=['QTIP_W3I', 'QTIP_W3'])
        rows = {}
        for arm in arms:
            t = episodes(self.base / 'loops' / arm)
            if t is None:
                raise RuntimeError(f'Closed loop of {arm} is not done')
            row = dict(successes=sum(t.values()), episodes=len(t), pairs={})
            for ref in refs[arm]:
                u = episodes((self.base if ref in planned_arms() else self.baseline) / 'loops' / ref)
                if u is None:
                    continue
                wins = sum(t[k] and not u[k] for k in t if k in u)
                losses = sum(u[k] and not t[k] for k in t if k in u)
                row['pairs'][ref] = dict(reference=sum(u.values()), wins=wins, losses=losses, p=wk.mcnemar(wins, losses))
            rows[arm] = row
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, complete=True, arms=rows)
        wk.save(self.rv('loop', 'summary.json'), summary)
        lines = ['# RAVQ 未校准版：LIBERO-Plus 280 闭环', '', f'作业 `{self.job}`。', '',
                 '| 臂 | 成功/280 | 对照 | 对照成功 | 胜 / 负 | McNemar p |', '|---|---:|---|---:|---:|---:|']
        for arm, r in rows.items():
            for ref, q in r['pairs'].items():
                lines.append(f"| {LABELS[arm]} | {r['successes']} | {LABELS.get(ref, ref)} | {q['reference']} | "
                             f"{q['wins']} / {q['losses']} | {q['p']:.2f} |")
        self.rv('loop', 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return summary

    def stage(self, name, shard, shards):
        if name == 'loop_final':
            return self.loop_final()
        if name in ('quant', 'assemble', 'eval', 'final'):
            self.make_plan()
        if name == 'quant':
            self.quant(shard, shards)
        elif name == 'assemble':
            if shard == 0:
                self.assemble()
        elif name == 'eval':
            self.eval(shard, shards)
        elif name == 'final':
            self.final()
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


def ratio(held, a, b, k):
    return held[a][k] / held[b][k] if held[b][k] else float('nan')


def future_share(o):
    return 1 - o['first10motor_mse_swap'] / o['first10motor_mse'] if o['first10motor_mse'] > 0 else float('nan')


def reading(held, bits):
    r = RV['reading']
    keys = ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse', 'first10gripper_rmse', 'first10motor_rmse_swap')
    protocol = {a: {k: ratio(held, a, b, k) for k in keys} for a, b in (('QTIP_W2I', 'QTIP_U_W2'), ('QTIP_W3I', 'QTIP_W3'))}
    floor = {a: {k: ratio(held, a, b, k) for k in keys} for a, b in (('QTIP_A1W2I', 'QTIP_W2I'), ('QTIP_A2W3I', 'QTIP_W3I'))}
    identity = max(abs(ratio(held, a, b, 'future_latent_nmse') - 1)
                   for a, b in (('QTIP_A1W2I', 'QTIP_W2I'), ('QTIP_A2W3I', 'QTIP_W3I')))
    ours, decision = {}, {}
    for a, b in (('RAVQU_W2', 'QTIP_W2I'), ('RAVQU_W3', 'QTIP_W3I')):
        row = {k: ratio(held, a, b, k) for k in keys}
        row['bits_ratio'] = bits[a]['bits_total'] / bits[b]['bits_total']
        row['go'] = row['future_latent_nmse'] <= r['future_ratio_max'] and row['mixed_loss'] <= r['mixed_ratio_max']
        ours[a] = row
        decision[a] = dict(reference=b, go=row['go'], future_ratio=row['future_latent_nmse'], mixed_ratio=row['mixed_loss'])
    loops = ['QTIP_W2I']
    for a in ('RAVQU_W2', 'RAVQU_W3'):
        if decision[a]['go']:
            loops += [x for x in (decision[a]['reference'], a) if x not in loops]
    decision['loops'] = loops
    return dict(protocol=protocol, action_floor=floor, ravq=ours, decision=decision,
                future_share={a: future_share(o) for a, o in held.items()},
                checks=dict(future_identity_worst=identity,
                            budget_ok=all(bits[a]['bits_total'] <= bits[RV['arms'][a]['budget_of']]['bits_total']
                                          for a in planned_arms() if 'budget_of' in RV['arms'][a])))


LABELS = dict(QTIP_U_W2='QTIP W2（A16 基线）', QTIP_W2='QTIP W2（A16 基线）', QTIP_W3='QTIP W3（A16 基线）',
              QTIP_W2I='QTIP W2 + 接口 BF16 + 时间表', QTIP_A1W2I='同上，动作主体 1 bit',
              RAVQU_W2='RAVQ 未校准，2-bit 预算', QTIP_W3I='QTIP W3 + 接口 BF16 + 时间表',
              QTIP_A2W3I='同上，动作主体 2 bit', RAVQU_W3='RAVQ 未校准，3-bit 预算')


def report(s):
    held, bits, rd = s['heldout'], s['bits'], s['reading']
    order = ['QTIP_U_W2', 'QTIP_W2I', 'QTIP_A1W2I', 'RAVQU_W2', 'QTIP_W3', 'QTIP_W3I', 'QTIP_A2W3I', 'RAVQU_W3']

    def role_bpw(a, r):
        v = bits[a].get('by_role', {}).get(r)
        return f"{v['bpw']:.2f}" if v else '—'
    lines = ['# RAVQ 未校准版：QTIP 舍入 + C1 模块级比特预算', '',
             f"作业 `{s['pbs_jobid']}`。舍入全部是 QTIP（A16 基线的同一份移植、同一份 H_U、同样的逐层种子），不训练任何参数。"
             '各臂只在"每类层用几 bit"上不同：接口层（action_encoder、动作 head、proprio_encoder、视频 head）存 BF16，'
             '时间条件层按 timestep 存输出表（输出与 BF16 完全一致），RAVQ 臂把动作专家主体省下的比特给视频专家主体。', '',
             '## 比特', '',
             '| 臂 | 总 bpw | 视频主体 | 动作主体 | 文本条件 | 接口 | 时间表 | 升级的视频层（层数 / 权重占比） | H_U 误差代理 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in order:
        b = bits[a]
        up = f"{b['upgraded_layers']} / {b['upgraded_weight_share']:.3f}" if b.get('upgraded_layers') else '—'
        lines.append(f"| {LABELS[a]} | {b['bpw']:.4f} | {role_bpw(a, 'video_body')} | {role_bpw(a, 'action_body')} | "
                     f"{role_bpw(a, 'text')} | {role_bpw(a, 'interface')} | {role_bpw(a, 'time')} | {up} | "
                     f"{b['proxy_nmse_weighted']:.5f} |")
    lines += ['', '## held-out 40×2', '',
              '| 臂 | mixed loss | first10 motor RMSE | future NMSE | gripper RMSE | 换 BF16 future 后 motor RMSE | future 占比 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for a in order:
        o = held[a]
        lines.append(f"| {LABELS[a]} | {o['mixed_loss']:.5f} | {o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | "
                     f"{o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} | {rd['future_share'][a]:.3f} |")
    lines += ['', '"future 占比" = 1 − 换 BF16 future 后的 motor MSE / 原 motor MSE：动作误差里经由想象 future 进来的部分。', '',
              '## 判读', '']
    for a, v in rd['protocol'].items():
        lines.append(f"- 公平口径（{LABELS[a]}）相对原基线：mixed {v['mixed_loss']:.3f}×，motor {v['first10motor_rmse']:.3f}×，"
                     f"gripper {v['first10gripper_rmse']:.3f}×，future {v['future_latent_nmse']:.3f}×。")
    for a, v in rd['action_floor'].items():
        lines.append(f"- 动作主体降位（{LABELS[a]}）：换 BF16 future 后 motor {v['first10motor_rmse_swap']:.3f}×，"
                     f"motor {v['first10motor_rmse']:.3f}×，gripper {v['first10gripper_rmse']:.3f}×，future {v['future_latent_nmse']:.6f}×。")
    for a, v in rd['ravq'].items():
        d = rd['decision'][a]
        lines.append(f"- {LABELS[a]} 对 {LABELS[d['reference']]}（比特 {v['bits_ratio']:.4f}×）：future {v['future_latent_nmse']:.3f}×，"
                     f"mixed {v['mixed_loss']:.3f}×，motor {v['first10motor_rmse']:.3f}×，gripper {v['first10gripper_rmse']:.3f}× → "
                     f"{'上闭环' if d['go'] else '不上闭环'}（预先登记：future ≤ {RV['reading']['future_ratio_max']}× 且 mixed ≤ "
                     f"{RV['reading']['mixed_ratio_max']}×）。")
    lines.append(f"- 下一步闭环：{', '.join(LABELS[a] for a in rd['decision']['loops'])}。")
    if s['k4']:
        est = sum(r['estimated'] for r in s['k4']) / len(s['k4'])
        got = sum(r['measured'] for r in s['k4']) / len(s['k4'])
        lines.append(f"- K = 4 的视频层（{len(s['k4'])} 层）：选层时估的 H_U 代理均值 {est:.5f}，实测 {got:.5f}。")
    c = rd['checks']
    lines.append(f"- 检查：只改动作主体时 future NMSE 的最大相对差 {c['future_identity_worst']:.2e}；比特预算 {'满足' if c['budget_ok'] else '超出'}。")
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
        RavqWork(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
