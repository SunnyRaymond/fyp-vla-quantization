"""C2 continuous stage, first test (tune.json): Q0's budget on the host's continuous parameters.

Each host's W2 weight W (A16 baseline, H_U; indices frozen) becomes bf16(diag(e^s) W diag(e^t)) with per-Linear FP32
log-scales s (rows) and t (columns), zero at start, tuned with A + lambda Z + beta K through the imagine -> execute rollout:
the loss-pair C80 schedule, 8 updates of 10 contexts, K through the frozen BF16 reader, staged as the loss-pair worker
(no-grad queries -> A/Z cotangents, reader -> beta K cotangent at the future, grad replay of the same queries with both).
Updates are gradient-proportional, delta = -eta g / max|g|: the first run (26057132) showed Adam's sign-like first step
moving all ~3M scales at once raising the loss from 2e-3, and changes under ~0.2% rounding back to the same BF16 weight.
One process per host and GPU (FW_STAGE tune, FW_HOSTS = the job's hosts, default all three; FW_SHARD indexes them):
  fit before  C80 mixed loss of the baseline
  step        batch 1 at s = t = 0: realized vs predicted decrease of one step for each grid eta; pick eta
  train       8 updates, checkpointed
  fold        e^s, e^t into the BF16 weight -> results/quantized/<host>_T_W2/ (the training forward's weight, exactly)
  fit after   C80 mixed loss of the folded arm; held-out 40x2 + swap
FW_STAGE final: results/tune/{summary.json, RESULTS.zh.md, DECISION.json}; loop_final after the closed loops.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import gate_worker as gw  # owns the allocation guard (via worker), the frozen-reader context and the masters
import torch
import torch.nn.functional as F

import lp_worker as lp
import worker as wk

HERE = Path(__file__).resolve().parent
TUNE = json.loads((HERE / 'tune.json').read_text(encoding='utf-8'))
HOSTS = TUNE['hosts']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def tuned_arm(host):
    return host + TUNE['arm_suffix']


def effective(weight, s, t):
    """bf16(diag(e^s) W diag(e^t)): the weight the tuned forward uses and the one that is folded and deployed."""
    return (s.exp()[:, None] * weight.float() * t.exp()[None, :]).to(weight.dtype)


class ScaledLinear(torch.autograd.Function):
    """F.linear with the effective weight; saves only x (the BF16 weight and the scales are inputs), so non-reentrant
    checkpoint regions keep their activation footprint. d/ds_r = sum_c G_rc W'_rc, d/dt_c = sum_r G_rc W'_rc with
    G = g^T x (straight through the BF16 cast); the bias and W get no gradient."""

    @staticmethod
    def forward(ctx, x, s, t, weight, bias):
        ctx.save_for_backward(x, s, t, weight)
        return F.linear(x, effective(weight, s, t), bias)

    @staticmethod
    def backward(ctx, g):
        x, s, t, weight = ctx.saved_tensors
        w = effective(weight, s, t)
        gx = g @ w if ctx.needs_input_grad[0] else None
        m = g.reshape(-1, g.shape[-1]).float().t() @ x.reshape(-1, x.shape[-1]).float()
        m.mul_(w.float())
        return gx, m.sum(1), m.sum(0), None, None


def first_step_choice(rows, fraction):
    """The largest grid step whose realized decrease on the batch is at least fraction x the first-order prediction."""
    ok = [r['lr'] for r in rows if r['predicted'] > 0 and r['realized'] >= fraction * r['predicted']]
    return max(ok) if ok else None


def extra_bits(shape, free):
    return 0 if free else 16 * (shape[0] + shape[1])


def decide(hosts, rule):
    """Per host: C80 fit improved, held-out mixed loss <= ratio x baseline, held-out future NMSE below the baseline."""
    rows = {}
    for host, r in hosts.items():
        if not r.get('tuned'):
            rows[host] = dict(tuned=False, passed=False)
            continue
        u, t = r['heldout']['U'], r['heldout']['T']
        row = dict(tuned=True, fit_down=r['fit']['after'] < r['fit']['before'],
                   mixed_ratio=t['mixed_loss'] / u['mixed_loss'], future_down=t['future_latent_nmse'] < u['future_latent_nmse'])
        row['passed'] = row['fit_down'] and row['mixed_ratio'] <= rule['mixed_ratio_max'] and row['future_down']
        rows[host] = row
    passed = sum(r['passed'] for r in rows.values())
    return dict(hosts=rows, passed=passed, min_hosts=rule['min_hosts'], go=passed >= rule['min_hosts'])


class TuneWork(gw.GateWork):
    def tune_dir(self, host=None):
        return self.base / 'tune' / host if host else self.base / 'tune'

    def arm_root(self, arm):
        if arm in {tuned_arm(h) for h in HOSTS}:
            return self.base / 'quantized' / arm
        return super().arm_root(arm)

    # ---------------- scales ----------------
    def attach(self):
        """FP32 log-scales per target Linear and a forward that uses them while self.active (the Bank's plain forward
        otherwise, which equals the tuned one at s = t = 0 bitwise)."""
        self.active, self.s, self.t = False, {}, {}
        for name, mod in self.bank.modules.items():
            self.s[name] = torch.zeros(mod.out_features, device=DEVICE, requires_grad=True)
            self.t[name] = torch.zeros(mod.in_features, device=DEVICE, requires_grad=True)
            mod.forward = self._scaled(name, mod, mod.forward)

    def _scaled(self, name, mod, plain):
        def run(x):
            if not self.active:
                return plain(x)
            self.bank.seen.add(name)  # the sampler checks that every target Linear ran
            return ScaledLinear.apply(x, self.s[name], self.t[name], mod.weight, mod.bias)
        return run

    def parameters(self):
        return [self.s[n] for n in self.bank.names] + [self.t[n] for n in self.bank.names]

    def snapshot(self):
        return dict(s={n: v.detach().cpu() for n, v in self.s.items()}, t={n: v.detach().cpu() for n, v in self.t.items()})

    def restore(self, saved):
        with torch.no_grad():
            for n in self.bank.names:
                self.s[n].copy_(saved['s'][n].to(DEVICE))
                self.t[n].copy_(saved['t'][n].to(DEVICE))

    # ---------------- staged loss / gradient ----------------
    def batch(self, batch, grad):
        """mean over the batch of A + lambda Z + beta K at the current scales; with grad, its gradient accumulated into
        s.grad / t.grad by the staged replay (loss-pair kl_batch)."""
        from reader_loss import reader_terms
        beta, stats = self.cfg['loss']['beta'], {'action': self.p['reader_source']['action_stats']}
        staged = []
        for case in batch:
            self.enough()
            with torch.no_grad():
                action, future, _ = self.query(case, 0)
            target = self.teacher(case, 0)
            a_leaf = action.detach().float().clone().requires_grad_(True)
            z_leaf = future.detach().float().clone().requires_grad_(True)
            total, _, _ = self.loss(a_leaf, z_leaf, target)
            ga, gz = torch.autograd.grad(total, (a_leaf, z_leaf))
            staged.append(dict(case=case, action=action.detach().cpu(), future=future.detach().cpu(),
                               ga=ga.detach().cpu(), gz=gz.detach().cpu(), az=float(total.detach())))
            del action, future, a_leaf, z_leaf, total, ga, gz
        with self.bf16_reader() as reader:
            for row in staged:
                case = row['case']
                value, description = self.datum(case)
                self.runtime.description = description
                leaf = row['future'].to(DEVICE, torch.float32).requires_grad_(grad)
                with torch.set_grad_enabled(grad):
                    read, _ = reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], leaf, checkpointed=grad)
                    k = reader_terms(torch, read, self.teacher(case, 0)['action'], stats,
                                     self.p['reader_source']['config'])['reader_kl']
                self.count('reader_queries')
                row['k'] = float(k.detach())
                if grad:
                    rg, = torch.autograd.grad(beta * k, leaf)
                    row['gz'] += rg.detach().cpu()
                del leaf, read, k
        out = dict(az=sum(r['az'] for r in staged) / len(staged), k=sum(r['k'] for r in staged) / len(staged))
        out['total'] = out['az'] + beta * out['k']
        if grad:
            worst, n = 0.0, len(staged)
            for row in staged:
                action, future, _ = self.query(row['case'], 0, grad=True)
                if not (action.requires_grad and future.requires_grad):
                    raise RuntimeError('Tuned query lost its graph')
                worst = max(worst, wk.max_abs(action, row['action']), wk.max_abs(future, row['future']))
                torch.autograd.backward((action, future), ((row['ga'] / n).to(action), (row['gz'] / n).to(future)))
                self.counters['parameter_reverse_passes'] += 1
                del action, future
            if worst > TUNE['training']['replay_max_abs']:
                raise RuntimeError(f'Grad replay differs from the staged no-grad query: {worst}')
            out['replay_max_abs'] = worst
        torch.cuda.empty_cache()
        return out

    def step_check(self, host):
        """One gradient-proportional step from s = t = 0 on batch 1 per grid eta: delta = -eta g / max|g| changes the
        scale with the largest gradient by eta (relative) and the others in proportion; predicted decrease
        eta |g|^2 / max|g| against the realized one on the same batch. Also the gradient's spread over layers and how
        many scales move by more than the BF16 rounding threshold."""
        path = self.tune_dir(host) / 'step.json'
        if path.exists():
            return wk.read(path)
        cfg, params, batch = TUNE['training'], self.parameters(), self.schedule[0]
        for p in params:
            p.grad = None
        self.active = True
        base = self.batch(batch, grad=True)
        g = [p.grad.detach().clone() for p in params]
        top = float(max(x.abs().max() for x in g))
        l2 = float(sum(float(x.square().sum()) for x in g)) ** 0.5
        n = len(self.bank.names)
        layers = sorted(((float(g[i].square().sum() + g[n + i].square().sum()), self.bank.names[i]) for i in range(n)),
                        reverse=True)
        rows = []
        for eta in cfg['step_grid']:
            with torch.no_grad():
                for p, x in zip(params, g):
                    p.copy_(-eta * x / top)
            visible = sum(int((eta * x.abs() / top > cfg['bf16_visible']).sum()) for x in g)
            after = self.batch(batch, grad=False)
            rows.append(dict(lr=eta, predicted=eta * l2 ** 2 / top, realized=base['total'] - after['total'],
                             visible=visible, loss=after))
            self.progress('step', host=host, eta=eta)
        with torch.no_grad():
            for p in params:
                p.zero_()
                p.grad = None
        self.active = False
        out = dict(protocol=wk.PROTOCOL, host=host, pbs_jobid=self.job, batch=batch, base=base, grad_l2=l2, grad_max=top,
                   grad_l1=float(sum(float(x.abs().sum()) for x in g)), scales=sum(x.numel() for x in g),
                   top_layers=[dict(module=name, share=v / l2 ** 2) for v, name in layers[:10]], rows=rows,
                   chosen=first_step_choice(rows, cfg['min_realized_fraction']))
        wk.save(path, out)
        return out

    def train(self, host, eta):
        """8 updates delta = -eta g / max|g| on the loss-pair C80 batches; checkpointed after each."""
        cfg, root = TUNE['training'], self.tune_dir(host)
        params = self.parameters()
        state = dict(updates=0, history=[])
        if (root / 'checkpoint.pt').exists():
            saved = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=True)
            if saved['eta'] != eta:
                raise RuntimeError('Checkpoint step differs')
            self.restore(saved['scales'])
            state = saved['state']
        self.active = True
        while state['updates'] < cfg['updates']:
            self.enough()
            u, begin = state['updates'], time.monotonic()
            for p in params:
                p.grad = None
            before = self.batch(self.schedule[u], grad=True)
            top = float(max(p.grad.abs().max() for p in params))
            if not top < float('inf'):
                raise RuntimeError(f'Non-finite gradient at update {u + 1}')
            if top > 0:
                with torch.no_grad():
                    for p in params:
                        p.sub_(eta * p.grad / top)
            state['updates'] += 1
            state['history'].append(dict(update=u + 1, cases=self.schedule[u], before=before, grad_max=top,
                                         seconds=time.monotonic() - begin))
            self.tsave(root / 'checkpoint.pt', dict(eta=eta, scales=self.snapshot(), state=state))
            wk.save(root / 'training.json', dict(protocol=wk.PROTOCOL, host=host, eta=eta, **state))
            self.progress('train', host=host, updates=state['updates'], loss=before['total'])
        self.active = False
        return state

    # ---------------- fold / fit ----------------
    def fold(self, host):
        """The trained scales into the BF16 weights (the training forward's weight) with bits and the proxy under H_U."""
        c, out = HOSTS[host], self.arm_root(tuned_arm(host))
        if (out / 'COMPLETE.json').exists():
            return wk.read(out / 'COMPLETE.json')
        prep, base = self.baseline / 'prep', self.arm_root(c['baseline_arm'])
        stats = []
        with torch.no_grad():
            for i, name in enumerate(self.bank.names):
                saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                         for k, path in (('U', base / 'weights'), ('master', prep / 'masters'), ('H_U', prep / 'hessians'))}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                q = saved['U']['weight'].to(DEVICE)
                w = effective(q, self.s[name].detach(), self.t[name].detach())
                self.tsave(out / 'weights' / f'{i:04d}.pt', dict(module=name, weight=w.cpu()))
                m, h = saved['master']['weight'].to(DEVICE, torch.float32), saved['H_U']['hessian'].to(DEVICE, torch.float32)

                def proxy(x):
                    e = x.float() - m
                    return float(((e @ h) * e).sum() / ((m @ h) * m).sum().clamp_min(1e-30))
                shape = list(q.shape)
                bits = wk.read(base / 'stats' / f'{i:04d}.json')['bits_total'] + extra_bits(shape, c['free_scales'])
                stats.append(dict(index=i, module=name, shape=shape, bits_total=bits, proxy_U=proxy(w), baseline_proxy_U=proxy(q),
                                  s_abs_max=float(self.s[name].abs().max()), t_abs_max=float(self.t[name].abs().max()),
                                  s_rms=float(self.s[name].square().mean().sqrt()), t_rms=float(self.t[name].square().mean().sqrt())))
                del saved, q, w, m, h
        n = sum(s['shape'][0] * s['shape'][1] for s in stats)

        def weighted(key):
            return sum(s[key] * s['shape'][0] * s['shape'][1] for s in stats) / n
        wk.save(out / 'layers.json', stats)
        complete = dict(protocol=wk.PROTOCOL, complete=True, pbs_jobid=self.job, arm=tuned_arm(host), method=c['method'],
                        bits=c['bits'], hessian='U', tuned='e^s, e^t per Linear, A + lambda Z + beta K', target_weights=n,
                        bpw=sum(s['bits_total'] for s in stats) / n, proxy_nmse_weighted=weighted('proxy_U'),
                        baseline_proxy_U_weighted=weighted('baseline_proxy_U'),
                        s_abs_max=max(s['s_abs_max'] for s in stats), t_abs_max=max(s['t_abs_max'] for s in stats))
        wk.save(out / 'COMPLETE.json', complete)
        return complete

    def fit(self, host, arm, key):
        """C80 mixed loss (A + lambda Z) and its parts with the stored weights of arm installed."""
        path = self.tune_dir(host) / f'fit_{key}.json'
        if path.exists():
            return wk.read(path)
        self.install(arm)
        rows = []
        for case in lp.CASES:
            self.enough()
            with torch.no_grad():
                action, future, _ = self.query(case, 0)
            m = self.metrics(action, future, self.teacher(case, 0))
            rows.append(dict(case_id=case, **{k: m[k] for k in ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse')}))
            self.progress('fit', host=host, key=key, contexts=len(rows))
        out = dict(protocol=wk.PROTOCOL, host=host, arm=arm, rows=rows,
                   **{k: sum(r[k] for r in rows) / len(rows) for k in ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse')})
        wk.save(path, out)
        return out

    def tune(self, host):
        c = HOSTS[host]
        self.load()
        self.capture_masters()
        self.attach()
        self.fit(host, c['baseline_arm'], 'before')
        self.install(c['baseline_arm'])
        eta = self.step_check(host)['chosen']
        if eta is None:
            wk.save(self.tune_dir(host) / 'host.json', dict(protocol=wk.PROTOCOL, host=host, tuned=False, pbs_jobid=self.job))
            return
        self.train(host, eta)
        self.fold(host)
        self.fit(host, tuned_arm(host), 'after')
        self.evaluate(tuned_arm(host))
        wk.save(self.tune_dir(host) / 'host.json', dict(protocol=wk.PROTOCOL, host=host, tuned=True, eta=eta, pbs_jobid=self.job))

    # ---------------- reports ----------------
    def gather(self):
        hosts = {}
        for host, c in HOSTS.items():
            r = wk.read(self.tune_dir(host) / 'host.json')
            row = dict(tuned=r['tuned'], step=wk.read(self.tune_dir(host) / 'step.json'))
            if r['tuned']:
                fit = {k: wk.read(self.tune_dir(host) / f'fit_{k}.json') for k in ('before', 'after')}
                row.update(eta=r['eta'], training=wk.read(self.tune_dir(host) / 'training.json')['history'],
                           fit={k: v['mixed_loss'] for k, v in fit.items()},
                           fit_parts={k: {m: v[m] for m in ('first10motor_rmse', 'future_latent_nmse')} for k, v in fit.items()},
                           quant=wk.read(self.arm_root(tuned_arm(host)) / 'COMPLETE.json'),
                           heldout={k: wk.read(self.base / 'heldout' / f'{a}.json')['overall']
                                    for k, a in (('U', c['baseline_arm']), ('T', tuned_arm(host)))})
            hosts[host] = row
        return hosts

    def final(self):
        hosts = self.gather()
        decision = decide(hosts, TUNE['decision'])
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, tune=TUNE, hosts=hosts,
                       decision=decision, new_teacher_queries=0)
        wk.save(self.tune_dir() / 'summary.json', summary)
        wk.save(self.tune_dir() / 'DECISION.json', dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, **decision))
        (self.tune_dir() / 'RESULTS.zh.md').write_text(tune_report(summary), encoding='utf-8')
        return summary

    def loop_final(self):
        """Per host the tuned arm against the baseline W2 in the closed loop, McNemar; pooled over hosts."""
        def episodes(root):
            return {int(f.stem): wk.read(f)['success'] for f in (root / 'episodes').glob('*.json')} if (root / 'DONE').exists() else None
        rows, pooled = {}, [0, 0]
        for host, c in HOSTS.items():
            u, t = episodes(self.baseline / 'loops' / c['baseline_loop']), episodes(self.base / 'loops' / tuned_arm(host))
            if u is None or t is None:
                continue
            wins = sum(t[k] and not u[k] for k in t if k in u)
            losses = sum(u[k] and not t[k] for k in t if k in u)
            rows[host] = dict(baseline=sum(u.values()), tuned=sum(t.values()), episodes=len(t), wins=wins, losses=losses,
                              p=wk.mcnemar(wins, losses))
            pooled[0] += wins
            pooled[1] += losses
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, hosts=rows,
                       pooled=dict(wins=pooled[0], losses=pooled[1], p=wk.mcnemar(*pooled)),
                       tune=wk.read(self.tune_dir() / 'summary.json')['hosts'])
        wk.save(self.tune_dir() / 'loop' / 'summary.json', summary)
        (self.tune_dir() / 'loop' / 'RESULTS.zh.md').write_text(loop_report(summary), encoding='utf-8')
        return summary

    def stage(self, name, shard, shards):
        if name == 'tune':
            hosts = (os.environ.get('FW_HOSTS') or ':'.join(HOSTS)).split(':')  # ':' since apptainer --env splits on ','
            if shards != len(hosts) or not set(hosts) <= set(HOSTS):
                raise ValueError('One tune process per host')
            self.tune(hosts[shard])
        elif name == 'final':
            self.final()
        elif name == 'loop_final':
            self.loop_final()
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


def tune_report(s):
    dec, beta = s['decision'], s['tune']
    lines = ['# C2 连续阶段初测：每个宿主 W2 上用 A + λZ + βK 调两侧对角 scale', '',
             f"作业 `{s['pbs_jobid']}`。起点是 A16 baseline 的 W2 权重（H_U），码字冻结；每个 Linear 调 FP32 log-scale s（行）、t（列），"
             "W' = bf16(diag(e^s) W diag(e^t))；QTIP、QuIP# 并进 SU/SV 不加 bit，GPTVQ 每层多存两个 FP16 向量。预算照 Q0：C80 一遍，"
             "8 次更新、每次 10 个 context；更新按梯度比例，δ = −η g / max|g|（第一次运行 26057132 里 Adam 的近似符号步在 LR ≥ 2e-3 时"
             "让 loss 上升，小于约 0.2% 的变化又被 BF16 舍入吃掉）；K 用冻结 BF16 reader 读 student future。", '',
             '## 步长检查（第 1 批，s = t = 0）', '',
             '| 宿主 | η（最大相对变化） | 一阶预测下降 | 实际下降 | 比例 | 越过 BF16 阈值的 scale 数 |', '|---|---:|---:|---:|---:|---:|']
    for host, r in s['hosts'].items():
        st = r['step']
        for row in st['rows']:
            ratio = row['realized'] / row['predicted'] if row['predicted'] else float('nan')
            lines.append(f"| {host} | {row['lr']:g} | {row['predicted']:.3g} | {row['realized']:.3g} | {ratio:.2f} | "
                         f"{row['visible']}/{st['scales']} |")
        lines.append(f"| {host} | 选定 | {st['chosen'] if st['chosen'] is not None else '无（不调）'} | | | |")
    lines += ['', '梯度最集中的层（占 |g|² 的比例）：']
    for host, r in s['hosts'].items():
        lines.append(f"- {host}：" + '，'.join(f"{x['module']} {x['share']:.2f}" for x in r['step']['top_layers'][:5]))
    lines += ['', '## 训练与 C80 拟合', '', '| 宿主 | 各批更新前 loss | C80 mixed loss 前 → 后 | bpw | proxy(H_U) 前 → 后 | 最大 |s| / |t| |',
              '|---|---|---|---:|---|---|']
    for host, r in s['hosts'].items():
        if not r['tuned']:
            continue
        q = r['quant']
        losses = ' '.join(f"{h['before']['total']:.4g}" for h in r['training'])
        lines.append(f"| {host} | {losses} | {r['fit']['before']:.5f} → {r['fit']['after']:.5f} | {q['bpw']:.4f} | "
                     f"{q['baseline_proxy_U_weighted']:.5f} → {q['proxy_nmse_weighted']:.5f} | {q['s_abs_max']:.3f} / {q['t_abs_max']:.3f} |")
    lines += ['', '## held-out 40×2', '',
              '| 宿主 | 臂 | mixed loss | first10 motor | future NMSE | future 在 motor MSE 里的占比 | gripper RMSE | 换回 BF16 future 后 motor |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for host, r in s['hosts'].items():
        if not r['tuned']:
            continue
        for k, label in (('U', 'baseline'), ('T', '调后')):
            o = r['heldout'][k]
            lines.append(f"| {host} | {label} | {o['mixed_loss']:.5f} | {o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | "
                         f"{gw.future_share(o):.3f} | {o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} |")
    rule = beta['decision']
    lines += ['', '## 判定（预先登记）', '',
              f"规则：至少 {rule['min_hosts']} 个宿主上 C80 拟合下降、held-out mixed loss ≤ {rule['mixed_ratio_max']} × baseline、"
              "held-out future NMSE 低于 baseline，则跑闭环。", '',
              '| 宿主 | 已调 | C80 下降 | held-out mixed 比 | future 下降 | 满足 |', '|---|---|---|---:|---|---|']
    yes = {True: '是', False: '否'}
    for host, r in dec['hosts'].items():
        if not r['tuned']:
            lines.append(f'| {host} | 否 | — | — | — | 否 |')
            continue
        lines.append(f"| {host} | 是 | {yes[r['fit_down']]} | {r['mixed_ratio']:.3f} | {yes[r['future_down']]} | {yes[r['passed']]} |")
    lines.append(f"\n{dec['passed']}/3 个宿主满足，结论：{'跑闭环' if dec['go'] else '不跑闭环'}。")
    return '\n'.join(lines) + '\n'


def loop_report(s):
    lines = ['# C2 连续阶段初测：闭环（LIBERO-Plus 280；BF16 216）', '',
             f"作业 `{s['pbs_jobid']}`。每个宿主的调后臂与其 baseline W2 配对。", '',
             '| 宿主 | baseline | 调后 | 胜/负 | McNemar p |', '|---|---:|---:|---:|---:|']
    for host, r in s['hosts'].items():
        lines.append(f"| {host} | {r['baseline']} | {r['tuned']} | {r['wins']}/{r['losses']} | {r['p']:.2f} |")
    p = s['pooled']
    lines.append(f"\n三宿主合计：{p['wins']} 胜 / {p['losses']} 负，McNemar p = {p['p']:.2f}。")
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
        TuneWork(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
