"""Phase 4 (rows.json): the phase-1 H~ plus a zero-bit output-row scale on QTIP W2, one arm (QTIP_GR_W2), one GPU.

Row weight d_c = sum_t g_tc^2 |x_t|^2 of the phase-1 A + lambda Z + beta K probes (results/hessians/<i>.pt row_weight).
s_c^2 = sqrt(d_c)/|W_c| (probe.row_scale); QTIP quantizes diag(s) W on H~ with the baseline seed and the dequantized rows
are divided by s, so s folds into QTIP's fp32 SV: 0 extra bits, no gradient updates. Steps (FW_STAGE): quant / complete
/ eval / final; run_rows.pbs runs them with the 280 closed loop before final. The model, held-out panel and closed-loop
glue are worker.Work's (phases 1-3), unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import worker as wk  # owns the PBS allocation guard and the runtime imports
import torch

import probe as pr

HERE = Path(__file__).resolve().parent
ROWS = json.loads((HERE / 'rows.json').read_text(encoding='utf-8'))
ARM = ROWS['arm']
ARMS = ('QTIP_U_W2', wk.GUIDED, ARM)
HESSIANS = ('U', 'G', 'G_rows')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RowWork(wk.Work):
    def arm_root(self, arm):
        if arm == ARM:
            return self.base / 'quantized' / ARM
        return super().arm_root(arm)

    def modules(self):
        return wk.read(self.baseline / 'prep' / 'modules.json')

    def quantize_rows(self, shard=0, shards=1):
        """QTIP W2 on H~ with the row scale; per layer the proxies of the baseline QTIP_W2, phase-1 QTIP_G_W2 and this
        arm's weight under H_U, H~ and H~ with the row weights d (the guided loss the scale targets)."""
        impl, prep, bits = self.method(ROWS['method']), self.baseline / 'prep', ROWS['bits']
        rows, base = self.modules(), self.arm_root('QTIP_U_W2')
        if any(not (self.results / 'hessians' / f'{i:04d}.pt').exists() for i in range(len(rows))):
            raise RuntimeError('Phase-1 guided Hessians are incomplete')
        spent = [wk.read(base / 'stats' / f'{i:04d}.json')['seconds'] for i in range(len(rows))]
        out = self.arm_root(ARM)
        mine = wk.owned(shard, shards, spent.__getitem__, range(len(rows)))
        todo = [i for i in mine if not (out / 'stats' / f'{i:04d}.json').exists()]
        sources = (('master', prep / 'masters'), ('U', prep / 'hessians'), ('G', self.results / 'hessians'),
                   ('QTIP_U_W2', base / 'weights'), (wk.GUIDED, self.arm_root(wk.GUIDED) / 'weights'))
        torch.use_deterministic_algorithms(False)  # as the baselines' quant_worker
        try:
            for done, i in enumerate(todo):
                self.enough()
                name = rows[i]['module']
                saved = {k: torch.load(path / f'{i:04d}.pt', map_location='cpu', weights_only=True) for k, path in sources}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                w = saved['master']['weight'].to(DEVICE, torch.float32)
                h = {k: saved[k]['hessian'].to(DEVICE, torch.float32) for k in ('U', 'G')}
                d = saved['G'].get('row_weight')
                d = d.to(DEVICE, torch.float32) if d is not None and float(d.sum()) > 0 else None
                scale = (torch.ones(w.shape[0], device=DEVICE) if d is None
                         else pr.row_scale(d, w, ROWS['row_scale_limit']).to(DEVICE))
                begin = time.monotonic()
                result = impl.quantize_layer(scale[:, None] * w, h['G'], bits, seed=self.host['host']['seed_base'] + i)
                seconds = time.monotonic() - begin
                dq = result['dequant'].to(DEVICE, torch.float32) / scale[:, None]
                if tuple(dq.shape) != tuple(w.shape) or not bool(torch.isfinite(dq).all()):
                    raise RuntimeError(f'QTIP produced an invalid weight at {name}')
                q = {a: saved[a]['weight'].to(DEVICE, torch.float32) for a in ('QTIP_U_W2', wk.GUIDED)}
                q[ARM] = dq

                def proxy(x, hv, rw=None):
                    e = x - w
                    num, den = ((e @ hv) * e).sum(1), ((w @ hv) * w).sum(1)
                    if rw is not None:
                        num, den = num * rw, den * rw
                    return float(num.sum() / den.sum().clamp_min(1e-30))
                matrix = {a: dict(U=proxy(x, h['U']), G=proxy(x, h['G']), G_rows=proxy(x, h['G'], d)) for a, x in q.items()}
                self.tsave(out / 'weights' / f'{i:04d}.pt', dict(module=name, weight=dq.to(torch.bfloat16).cpu()))
                wk.save(out / 'stats' / f'{i:04d}.json', dict(
                    index=i, module=name, shape=list(w.shape), bits_total=result['bits_total'],
                    bpw=result['bits_total'] / w.numel(), seconds=seconds, config=result['config'], hessian='G',
                    row_scale=d is not None, row_ratio=pr.row_ratio(d, scale, w) if d is not None else 1.0,
                    row_scale_range=[float(scale.min()), float(scale.max())], proxy_U=matrix[ARM]['U'], proxies=matrix))
                del saved, w, h, q, dq, result
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                if done % 25 == 0:
                    self.progress('quant', layers=done + 1, of=len(todo))
        finally:
            torch.use_deterministic_algorithms(True)

    def complete_rows(self):
        """quantized/QTIP_GR_W2/COMPLETE.json (what the baselines' analysis.py reads) and the parameter-weighted proxies."""
        stats = [wk.read(self.arm_root(ARM) / 'stats' / f'{i:04d}.json') for i in range(len(self.modules()))]
        n = sum(s['shape'][0] * s['shape'][1] for s in stats)

        def weighted(f):
            return sum(f(s) * s['shape'][0] * s['shape'][1] for s in stats) / n
        matrix = {a: {k: weighted(lambda s: s['proxies'][a][k]) for k in HESSIANS} for a in ARMS}
        row = dict(arm=ARM, method=ROWS['method'], bits=ROWS['bits'], hessian='G', row_scale=True,
                   bpw=sum(s['bits_total'] for s in stats) / n, proxy_nmse_weighted=matrix[ARM]['U'],
                   quantize_seconds=sum(s['seconds'] for s in stats), row_ratio_weighted=weighted(lambda s: s['row_ratio']),
                   layers_scaled=sum(s['row_scale'] for s in stats), layers=len(stats))
        wk.save(self.arm_root(ARM) / 'COMPLETE.json', dict(protocol=wk.PROTOCOL, complete=True, pbs_jobid=self.job,
                                                          target_weights=n, config_example=stats[0]['config'], **row))
        return dict(arm=row, proxy_matrix=matrix)

    def final_rows(self):
        """The three QTIP W2 arms: proxies, held-out, closed loop by perturbation, paired counts of this arm."""
        quant = self.complete_rows()
        roots = {'QTIP_U_W2': self.baseline / 'loops' / self.host['host']['baseline_arm'],
                 wk.GUIDED: self.base / 'loops' / wk.GUIDED, ARM: self.base / 'loops' / ARM}
        episodes = {a: {int(f.stem): wk.read(f)['success'] for f in (r / 'episodes').glob('*.json')}
                    for a, r in roots.items() if (r / 'DONE').exists()}
        loops = {a: dict(successes=sum(e.values()), episodes=len(e), dimensions=wk.read(roots[a] / 'summary.json')['dimensions'])
                 for a, e in episodes.items()}
        if ARM in episodes:
            mine = episodes[ARM]
            for ref in ('QTIP_U_W2', wk.GUIDED):
                if ref in episodes:
                    r = episodes[ref]
                    wins = sum(mine[k] and not r[k] for k in mine if k in r)
                    losses = sum(r[k] and not mine[k] for k in mine if k in r)
                    loops[ARM][f'vs_{ref}'] = dict(wins=wins, losses=losses, p=wk.mcnemar(wins, losses))
        held = {a: wk.read(self.results / 'heldout' / f'{a}.json')['overall'] for a in ARMS
                if (self.results / 'heldout' / f'{a}.json').exists()}
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, rows=ROWS,
                       quant=quant, heldout=held, loops=loops, new_teacher_queries=0, training_updates=0)
        wk.save(self.base / 'rows' / 'summary.json', summary)
        (self.base / 'rows' / 'RESULTS.zh.md').write_text(rows_report(summary), encoding='utf-8')

    def stage(self, name, shard, shards):
        if name == 'quant':
            self.quantize_rows(shard, shards)
        elif name == 'complete':
            self.complete_rows()
        elif name == 'eval':
            self.load()
            self.evaluate(ARM)
        elif name == 'final':
            self.final_rows()
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


def rows_report(s):
    names = {'QTIP_U_W2': 'H_U（baseline QTIP_W2）', wk.GUIDED: 'H̃（第一阶段）', ARM: 'H̃ + 行 scale（本阶段）'}
    a, m = s['quant']['arm'], s['quant']['proxy_matrix']
    lines = ['# H̃ + 0 bit 输出行 scale：QTIP W2', '',
             f"作业 `{s['pbs_jobid']}`。H̃ 与第一阶段相同；行权重 d 取第一阶段 A + λZ + βK 探针的逐 token 行权重，"
             f"s² = √d/‖W_c‖，几何均值 1，限制在 [1/{s['rows']['row_scale_limit']}, {s['rows']['row_scale_limit']}]，"
             "并进 QTIP 的 fp32 SV，不加 bit、不训练。", '',
             f"bpw {a['bpw']:.4f}；{a['layers_scaled']}/{a['layers']} 层有行权重；按参数加权的预估 guided loss 比 "
             f"{a['row_ratio_weighted']:.3f}。", '',
             '## proxy（参数加权）', '', '| 权重 \\ Hessian | H_U | H̃ | H̃ + 行权重 |', '|---|---:|---:|---:|']
    lines += [f"| {names[x]} | " + ' | '.join(f'{m[x][k]:.5f}' for k in HESSIANS) + ' |' for x in m]
    lines += ['', '## held-out 40×2', '',
              '| 臂 | first10 motor | future NMSE | late22 motor | gripper RMSE | 换回 BF16 future 后 motor | mixed loss |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for x, o in s['heldout'].items():
        lines.append(f"| {names[x]} | {o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | "
                     f"{o['late22motor_rmse']:.4f} | {o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} | "
                     f"{o['mixed_loss']:.5f} |")
    loops = s['loops']
    if loops:
        dims = sorted(next(iter(loops.values()))['dimensions'])
        lines += ['', '## LIBERO-Plus 280 闭环（BF16 216）', '',
                  '| 臂 | 成功 | ' + ' | '.join(dims) + ' |', '|---|---:|' + '---:|' * len(dims)]
        lines += [f"| {names[x]} | {o['successes']}/{o['episodes']} | " + ' | '.join(str(o['dimensions'][k]) for k in dims)
                  + ' |' for x, o in loops.items()]
        mine = loops.get(ARM, {})
        for ref in ('QTIP_U_W2', wk.GUIDED):
            if f'vs_{ref}' in mine:
                v = mine[f'vs_{ref}']
                lines.append(f"\n本臂对 {names[ref]}：{v['wins']} 胜 {v['losses']} 负，McNemar p = {v['p']:.2f}。")
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
        RowWork(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
