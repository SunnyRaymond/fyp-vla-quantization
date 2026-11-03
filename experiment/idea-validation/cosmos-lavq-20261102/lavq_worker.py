"""LAVQ on Cosmos-Policy (protocol.json): one stage per call (LV_STAGE, shard LV_SHARD of LV_SHARDS); finished items on
disk are skipped, so a resubmission continues.

  quant          QTIP K = 1 and K = 4 of the 280 target Linears (weights / native codes / stats), the baselines' seeds
  quant_final    COMPLETE.json of QTIP_W1 / QTIP_W4
  teacher        contexts (C80 seed 0, held-out 40 x 2), BF16 teacher with the cut-candidate states, continuation
                 identity; future swap of QTIP_W2 / QTIP_W3 at every candidate cut on C80 (cut rule)
  teacher_final  C80 action statistics, the cut, results/teacher/summary.json
  metric         (1) per C80 context: probes J_R^T (eps sqrt(h/2)) of the BF16 reader at the teacher cut state
  metric_final   M (top-r + residual diagonal + rho I) and its split-half stability
  sens           (2) C80 queries + BF16 reader KL of every family base and every group one bit up / down
  sens_final     lambda, beta, lam_M; group gains / costs; exchange prefixes (results/alloc/plan.json)
  alloc          C80 queries + reader KL of the prefixes
  alloc_final    the prefix with the lowest C80 L per family -> results/quantized/LAVQU_W<b>
  heldout        held-out 40 x 2 of the hosts and LAVQU: errors, BF16 reader KL, future swap (rho)
  heldout_final  held-out table, LAVQU gate, calibration bases
  codes          pair codes of every layer of the calibration bases (native trellis, checked bit for bit)
  calib          (4) one family per shard: shared-table calibration (ravqc v3 rule), deploy LAVQ_W<b>, held-out
  final          DECISION.json (closed-loop arms), summary.json, RESULTS.zh.md
"""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from common import TARGETS, baseline, guard, merged_cache, prep, read, results, save, torch_save

HERE = Path(__file__).resolve().parent
P = read(HERE / 'protocol.json')
SMOKE = os.environ.get('LV_SMOKE') == '1'
if SMOKE:  # GPU smoke run of every stage on 2 C80 + 2 held-out contexts (results under smoke/), forced exchange and gates
    P['calibration']['cases'] = [P['calibration']['anchor_case'], P['calibration']['anchor_case'] + 1]
    P['heldout']['cases'] = [P['heldout']['cases'][0]] * 2
    P['metric'].update(probes_per_context=2, rank=2)
    P['allocation']['max_exchanges'] = 1
    P['training'].update(updates=2, contexts_per_update=1, step_grid=P['training']['step_grid'][:2])
R = Path(P['remote_root']) / 'smoke' if SMOKE else results(P)
CFG = P['reader']
HOSTS = ('QTIP', 'QUIPSHARP', 'GPTVQ')


class Deadline(Exception):
    pass


# ---------------- small helpers ----------------
def enough(margin=300):
    start = float(os.environ['COSMOS_JOB_START_EPOCH'])
    if time.time() + margin > start + float(os.environ['LV_SOFT_DEADLINE']):
        raise Deadline('soft deadline')


def progress(stage, **kw):
    print('LAVQ_PROGRESS ' + json.dumps(dict(stage=stage, t=round(time.time()), **kw), default=str), flush=True)


def owned(shard, shards, items, weight=lambda x: 1.0):
    """Greedy split by weight (largest first), stable."""
    loads, mine = [0.0] * shards, []
    for item in sorted(items, key=lambda x: -weight(x)):
        k = min(range(shards), key=lambda s: (loads[s], s))
        loads[k] += weight(item)
        if k == shard:
            mine.append(item)
    return sorted(mine, key=lambda x: str(x))


def c80():
    a, b = P['calibration']['cases']
    return [(c, P['calibration']['seed_index']) for c in range(a, b + 1)]


def heldout():
    a, b = P['heldout']['cases']
    return [(c, s) for c in range(a, b + 1) for s in P['heldout']['seed_indices']]


def key(ctx):
    return f'{ctx[0]}_{ctx[1]}'


def tload(path, device='cpu'):
    import torch
    return torch.load(path, map_location=device, weights_only=False)


def qtip_module():
    sys.path.insert(0, str(HERE / 'methods' / 'qtip'))
    Q = importlib.import_module('quant')
    return Q, Q.ldlq._PERMUTE.clone(), Q.ldlq._INV_PERMUTE.clone()


def modules():
    return read(prep(P) / 'modules.json')


def group_of(name):
    """(band, type) of a target Linear: net.blocks.<b>._checkpoint_wrapped_module.<type>.<proj>."""
    block = int(name.split('.')[2])
    tail = name.split('_checkpoint_wrapped_module.')[1]
    band = next(i for i, (lo, hi) in enumerate(P['allocation']['bands']) if lo <= block <= hi)
    kind = next(t for t, names in P['allocation']['types'].items() if tail in names)
    return f'b{band}_{kind}', kind


def arm_dir(arm):
    """Directory of a quantized arm: the baselines' arms, this job's QTIP_W1 / QTIP_W4 and assembled arms."""
    if arm in ('QTIP_W1', 'QTIP_W4'):
        return results(P) / 'quantized' / arm
    if arm.startswith('LAVQ'):
        return R / 'quantized' / arm
    return baseline(P) / 'quantized' / arm


def stats(arm, i):
    return read(arm_dir(arm) / 'stats' / f'{i:04d}.json')


def bits_table():
    """bits[i][K] and proxy[i][K] (stored bits, H-proxy) of every layer at K = 1..4."""
    rows = modules()
    bits, proxy = {}, {}
    for r in rows:
        i = r['index']
        bits[i], proxy[i] = {}, {}
        for k in (1, 2, 3, 4):
            s = stats(f'QTIP_W{k}', i)
            bits[i][k], proxy[i][k] = s['bits_total'], s['proxy_nmse']
    return bits, proxy


# ---------------- quant ----------------
def quant(shard, shards):
    import torch
    import ravq_qtip
    Q, _, _ = qtip_module()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = modules()
    items = [(k, r['index']) for k in P['qtip']['new_bits'] for r in rows]
    shape = {r['index']: r['shape'] for r in rows}
    mine = owned(shard, shards, items, lambda it: shape[it[1]][0] * shape[it[1]][1] + shape[it[1]][1] ** 2)
    started = time.monotonic()
    for done, (k, i) in enumerate(mine):
        out = R / 'quantized' / f'QTIP_W{k}'
        if (out / 'stats' / f'{i:04d}.json').exists():
            continue
        enough(900)
        name = rows[i]['module']
        master = torch.load(prep(P) / 'masters' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
        saved = torch.load(prep(P) / 'hessians' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
        if master['module'] != name or saved['module'] != name:
            raise RuntimeError(f'Prep identity differs at {i}')
        w = master['weight'].to('cuda', torch.float32)
        h = saved['hessian'].to('cuda', torch.float32)
        begin = time.monotonic()
        seed = P['qtip']['seed_base'] + i
        result = ravq_qtip.quantize_layer(Q, w, h, k, seed=seed)
        dq = result['dequant'].to('cuda', torch.float32)
        d = dq - w
        proxy = float(((d @ h) * d).sum() / ((w @ h) * w).sum().clamp_min(1e-30))
        torch_save(dict(module=name, method='qtip', bits=k, seed=seed, codes=result['codes']), out / 'codes' / f'{i:04d}.pt')
        torch_save(dict(module=name, weight=dq.to(torch.bfloat16).cpu()), out / 'weights' / f'{i:04d}.pt')
        save(out / 'stats' / f'{i:04d}.json', dict(index=i, module=name, shape=list(w.shape), bits_total=result['bits_total'],
                                                  bpw=result['bits_total'] / w.numel(), proxy_nmse=proxy,
                                                  seconds=time.monotonic() - begin, config=result['config']))
        del w, h, dq, d, result
        torch.cuda.empty_cache()
        if done % 10 == 0:
            progress('quant', K=k, layer=i, done=done + 1, of=len(mine), proxy=proxy,
                     elapsed=round(time.monotonic() - started))


def quant_final():
    for k in P['qtip']['new_bits']:
        arm = f'QTIP_W{k}'
        out = R / 'quantized' / arm
        st = [read(out / 'stats' / f'{i:04d}.json') for i in range(TARGETS)]
        weights = sum(s['shape'][0] * s['shape'][1] for s in st)
        save(out / 'COMPLETE.json', dict(protocol=P['protocol'], arm=arm, method='qtip', bits=k, complete=True,
                                         pbs_jobid=os.environ['PBS_JOBID'], target_weights=weights,
                                         bits_total=sum(s['bits_total'] for s in st),
                                         bpw=sum(s['bits_total'] for s in st) / weights,
                                         proxy_nmse_weighted=sum(s['proxy_nmse'] * s['shape'][0] * s['shape'][1] for s in st) / weights,
                                         quantize_gpu_hours=sum(s['seconds'] for s in st) / 3600))
        progress('quant_final', arm=arm)


# ---------------- the Cosmos runtime of a worker ----------------
class Work:
    def __init__(self, shard, shards):
        import torch
        self.torch, self.shard, self.shards = torch, shard, shards
        self.device = 'cuda'
        self.rt = None
        self.cache = {}

    def build(self, contexts):
        """BF16 runtime on a one-shard T5 cache (the official loader moves its whole file to the GPU under a 30 s lock;
        the merged 1.8 GB file times out with several workers, A16 loop 26079917)."""
        import cosmos_lavq as cl
        from common import calibration_input, dump_pickle, load_pickle, runtime
        full = load_pickle(merged_cache(P))
        texts = {calibration_input(P, c)[1]['description'] for c, _ in contexts}
        texts.add(calibration_input(P, P['calibration']['anchor_case'])[1]['description'])
        path = R / 't5' / f'{os.environ["PBS_JOBID"]}_{os.environ["LV_STAGE"]}_{self.shard}.pkl'
        dump_pickle({t: full[t] for t in texts}, path)
        del full
        self.rt = runtime(P, R / 'runtime' / f'{os.environ["LV_STAGE"]}_{self.shard}', path)
        self.names = list(self.rt.target_linears)
        self.mods = [self.rt.target_linears[n] for n in self.names]
        rows = modules()
        if [r['module'] for r in rows] != self.names:
            raise RuntimeError('Target Linear order differs from the baselines prep')
        self.masters = [m.weight.detach().clone() for m in self.mods]
        self.current = ['BF16'] * TARGETS
        self.cl = cl
        progress('build', of=os.environ['LV_STAGE'], shard=self.shard)

    # ---- weights ----
    def layer(self, src, i):
        torch = self.torch
        if src == 'BF16':
            return self.masters[i]
        if src.startswith('QTIP_W'):
            if src not in self.cache:
                self.cache[src] = [None] * TARGETS
            if self.cache[src][i] is None:
                w = torch.load(arm_dir(src) / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                if w['module'] != self.names[i]:
                    raise RuntimeError(f'{src} weight {i} identity differs')
                self.cache[src][i] = w['weight'].pin_memory()
            return self.cache[src][i]
        w = torch.load(arm_dir(src) / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
        if w['module'] != self.names[i]:
            raise RuntimeError(f'{src} weight {i} identity differs')
        return w['weight']

    def install(self, spec):
        """spec: 280 sources ('BF16', an arm name); copies only the layers that change."""
        with self.torch.no_grad():
            for i, src in enumerate(spec):
                if self.current[i] != src:
                    self.mods[i].weight.data.copy_(self.layer(src, i).to(self.device, non_blocking=True))
                    self.current[i] = src
        self.torch.cuda.synchronize()

    # ---- contexts / teachers ----
    def context(self, ctx):
        if ctx not in getattr(self, 'ctx_cache', {}):
            self.ctx_cache = getattr(self, 'ctx_cache', {})
            self.ctx_cache[ctx] = tload(R / 'contexts' / f'{key(ctx)}.pt')
        return self.ctx_cache[ctx]

    def teacher(self, ctx):
        return tload(R / 'teacher' / f'{key(ctx)}.pt')

    def readers(self, contexts, cut):
        """BF16 suffix readers of the contexts (kept: one teacher run per context for every arm), identity-checked."""
        self.install(['BF16'] * TARGETS)
        out = {}
        for ctx in contexts:
            enough()
            reader = self.cl.Reader(self.rt, self.context(ctx), cut)
            ident = reader.identity()
            t = self.teacher(ctx)
            parity = float((reader.action.cpu() - t['action']).abs().max())
            if ident > CFG['identity_max_abs'] or parity > CFG['identity_max_abs']:
                raise RuntimeError(f'Reader of {ctx} not identical: continuation {ident}, teacher {parity}')
            out[ctx] = reader
        return out

    def student(self, ctx, cut, swap=None):
        """No-grad query of the installed arm: action, cut future, final future slots; with swap (the teacher cut
        future), also the student's own remaining steps from its state with the teacher's future (rho)."""
        torch = self.torch
        result, cap = self.cl.run(self.rt, self.context(ctx), (cut,))
        row = dict(action=result.action.detach().float().cpu(), z=cap.future(cut).cpu(),
                   final=result.generated_latent[:, :, self.cl.FUT].detach().float().cpu())
        if swap is not None:
            x = self.cl.with_future(cap.states[cut][0], swap.to(self.device))
            with torch.no_grad():
                row['swap'] = self.cl.continue_action(self.rt, cap, cut, x, result.generated_latent.dtype).float().cpu()
        return row

    def reader_kl(self, reader, z):
        with self.torch.no_grad():
            a = reader.read(z.to(self.device))
        return float(self.cl.reader_kl(a, reader.action, self.astats, CFG))

    def load_stats(self):
        s = tload(R / 'teacher' / 'action_stats.pt')
        self.astats = {k: v.to(self.device) for k, v in s.items()}


# ---------------- teacher (+ cut swap) ----------------
def teacher(shard, shards):
    torch = __import__('torch')
    mine = owned(shard, shards, c80() + heldout(), lambda c: 1.0)
    w = Work(shard, shards)
    w.build(mine)
    cl = w.cl
    cuts = [int(c) for c in CFG['cut_candidates']]
    torch_save({k: torch.as_tensor(w.rt.dataset_stats[k]).float() for k in ('actions_min', 'actions_max')},
               R / 'teacher' / 'dataset_stats.pt')
    worst = 0.0
    for ctx in mine:
        out = R / 'teacher' / f'{key(ctx)}.pt'
        if out.exists():
            continue
        enough()
        cpath = R / 'contexts' / f'{key(ctx)}.pt'
        if not cpath.exists():
            torch_save(cl.build_context(P, w.rt, ctx[0], ctx[1]), cpath)
        c = w.context(ctx)
        slots = cl.slot_indices(w.rt, c)
        if (slots['action_latent_idx'], sorted((slots['future_wrist_image_latent_idx'], slots['future_image_latent_idx']))) != (4, [6, 7]):
            raise RuntimeError(f'Unexpected latent slots {slots}')
        result, cap = cl.run(w.rt, c, cuts)
        action = result.action.detach().float()
        ident = {}
        for cut in cuts:
            with torch.no_grad():
                again = cl.continue_action(w.rt, cap, cut, cap.states[cut][0], result.generated_latent.dtype)
            ident[cut] = float((again.float() - action).abs().max())
        worst = max(worst, *ident.values())
        if max(ident.values()) > CFG['identity_max_abs']:
            raise RuntimeError(f'Continuation identity of {ctx}: {ident}')
        torch_save(dict(ctx=list(ctx), action=action.cpu(), final=result.generated_latent[:, :, cl.FUT].detach().float().cpu(),
                        cut={cut: cap.future(cut).cpu() for cut in cuts}, identity=ident, slots=slots,
                        latent_shape=list(result.generated_latent.shape), latent_dtype=str(result.generated_latent.dtype)),
                   out)
        progress('teacher', ctx=ctx, identity=max(ident.values()))
    # anchor parity against the official get_action (baselines' prep anchor)
    anchor_ctx = (P['calibration']['anchor_case'], P['calibration']['seed_index'])
    if anchor_ctx in mine:
        ref = torch.load(prep(P) / 'bf16_anchor.pt', map_location='cpu', weights_only=True)
        t = w.teacher(anchor_ctx)
        save(R / 'teacher' / 'anchor_parity.json', dict(
            action_max_abs=float((t['action'][0] - ref['action']).abs().max()), tolerance=2e-3,
            passed=float((t['action'][0] - ref['action']).abs().max()) <= 2e-3))
    # future swap of the uniform hosts at every candidate cut (C80 of this shard)
    for arm in ('QTIP_W2', 'QTIP_W3'):
        todo = [ctx for ctx in mine if ctx in c80() and not (R / 'cut' / arm / f'{key(ctx)}.json').exists()]
        if not todo:
            continue
        w.install([arm] * TARGETS)
        for ctx in todo:
            enough()
            t = w.teacher(ctx)
            result, cap = cl.run(w.rt, w.context(ctx), cuts)
            a = result.action.detach().float()
            row = dict(A=float(cl.action_mse(a, t['action'])), swap={})
            for cut in cuts:
                x = cl.with_future(cap.states[cut][0], t['cut'][cut].to(w.device))
                with torch.no_grad():
                    s = cl.continue_action(w.rt, cap, cut, x, result.generated_latent.dtype)
                row['swap'][str(cut)] = float(cl.action_mse(s, t['action']))
                row.setdefault('Z', {})[str(cut)] = float(cl.nmse(cap.future(cut), t['cut'][cut]))
            save(R / 'cut' / arm / f'{key(ctx)}.json', row)
        progress('cut', arm=arm, contexts=len(todo))


def teacher_final():
    torch = __import__('torch')
    teachers = {ctx: tload(R / 'teacher' / f'{key(ctx)}.pt') for ctx in c80() + heldout()}
    ds = tload(R / 'teacher' / 'dataset_stats.pt')
    lo, hi = ds['actions_min'].float(), ds['actions_max'].float()
    acts = torch.cat([teachers[ctx]['action'][0] for ctx in c80()])  # [80 * 16, 7]
    norm = (2 * (acts - lo) / (hi - lo).clamp_min(1e-12) - 1)[:, :6]
    torch_save(dict(actions_min=lo, actions_max=hi, mean=norm.mean(0), scale=norm.std(0)), R / 'teacher' / 'action_stats.pt')
    cuts = [int(c) for c in CFG['cut_candidates']]
    rho = {}
    for arm in ('QTIP_W2', 'QTIP_W3'):
        rows = [read(R / 'cut' / arm / f'{key(ctx)}.json') for ctx in c80()]
        a = sum(r['A'] for r in rows)
        rho[arm] = {str(c): 1 - sum(r['swap'][str(c)] for r in rows) / a for c in cuts}
    mean = {c: sum(rho[arm][str(c)] for arm in rho) / len(rho) for c in cuts}
    best = max(cuts, key=lambda c: (mean[c], -c))
    ident = max(max(t['identity'].values()) for t in teachers.values())
    parity = read(R / 'teacher' / 'anchor_parity.json')
    if not parity['passed']:
        raise RuntimeError(f'Differentiable endpoint differs from the official anchor: {parity}')
    save(R / 'teacher' / 'summary.json', dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], cut=best,
                                              rho_c80=rho, rho_mean=mean, identity_max=ident, anchor_parity=parity,
                                              latent_shape=teachers[c80()[0]]['latent_shape'],
                                              latent_dtype=teachers[c80()[0]]['latent_dtype'],
                                              future_dims=int(teachers[c80()[0]]['cut'][best].numel())))
    progress('teacher_final', cut=best, rho=mean)


def the_cut():
    return int(read(R / 'teacher' / 'summary.json')['cut'])


# ---------------- (1) the reader metric ----------------
def metric(shard, shards):
    torch = __import__('torch')
    cut, cfg = the_cut(), P['metric']
    mine = [ctx for ctx in owned(shard, shards, c80()) if not (R / 'metric' / 'ctx' / f'{key(ctx)}.pt').exists()]
    if not mine:
        return
    w = Work(shard, shards)
    w.build(mine)
    w.load_stats()
    cl = w.cl
    w.install(['BF16'] * TARGETS)
    for ctx in mine:
        enough()
        reader = cl.Reader(w.rt, w.context(ctx), cut)
        ident = reader.identity()
        if ident > CFG['identity_max_abs']:
            raise RuntimeError(f'Reader identity of {ctx}: {ident}')
        h = cl.kl_curvature(reader.action, w.astats, CFG)
        gen = torch.Generator(device=w.device).manual_seed(cfg['seed'] + 10 * ctx[0])
        probes = []
        for _ in range(cfg['probes_per_context']):
            eps = torch.randint(0, 2, tuple(h.shape), generator=gen, device=w.device).float().mul_(2).sub_(1)
            leaf = reader.z.to(w.device).clone().requires_grad_(True)
            a = reader.read(leaf)
            g, = torch.autograd.grad(a[..., :6].float(), leaf, eps * (h / 2).sqrt())
            probes.append(g.detach().float().reshape(-1).cpu())
            del a, g, leaf
        P_ = torch.stack(probes)
        if not bool(torch.isfinite(P_).all()) or float(P_.abs().max()) == 0:
            raise RuntimeError(f'Reader probes of {ctx} are zero or non-finite')
        torch_save(dict(ctx=list(ctx), probes=P_, identity=ident, curvature_mean=float(h.mean()),
                        z_energy=float(reader.z.square().sum()), shape=list(reader.z.shape)),
                   R / 'metric' / 'ctx' / f'{key(ctx)}.pt')
        progress('metric', ctx=ctx, identity=ident)


def metric_final():
    import qmath
    cfg = P['metric']
    rows = [tload(R / 'metric' / 'ctx' / f'{key(ctx)}.pt') for ctx in c80()]
    vectors, cases = [r['probes'] for r in rows], [ctx[0] for ctx in c80()]
    saved, st, diag = qmath.build_metric(vectors, cases, cfg['rank'], cfg['rho'])
    stab = qmath.split_stability(vectors, cases, cfg['rank'])
    shape = rows[0]['shape']  # [1, 16, 2, H, W]
    d = diag.reshape(shape[1:])
    gaze = dict(views=[float(x) for x in d.sum((0, 2, 3)) / d.sum()],
                channels=[float(x) for x in d.sum((1, 2, 3)) / d.sum()],
                space=(d.sum((0, 1)) / d.sum()).tolist())
    saved['shape'] = shape
    torch_save(dict(saved, diag=diag), R / 'metric' / 'metric.pt')
    per = [float(r['probes'].square().sum(1).mean()) for r in rows]
    ess = sum(per) ** 2 / sum(x * x for x in per)
    save(R / 'metric' / 'summary.json', dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], cut=the_cut(),
                                             metric=cfg, stats=st, stability=stab, gaze=gaze,
                                             context_trace=dict(values=per, ess=ess),
                                             identity_max=max(r['identity'] for r in rows)))
    progress('metric_final', **st, **stab)


def load_metric(device):
    import qmath
    return qmath.Metric(tload(R / 'metric' / 'metric.pt'), device)


# ---------------- measured arms (C80 / held-out) ----------------
def arm_specs():
    """Every measured arm of the sensitivity stage: name -> 280 sources."""
    rows = modules()
    groups = sorted({group_of(r['module'])[0] for r in rows})
    specs = {}
    for b, fam in P['allocation']['families'].items():
        base = [fam['base']] * TARGETS
        specs[f'W{b}_base'] = base
        for g in groups:
            for direction in ('up', 'down'):
                k = fam[direction]
                specs[f'W{b}_{g}_{direction}'] = [f'QTIP_W{k}' if group_of(r['module'])[0] == g else fam['base']
                                                  for r in rows]
    return specs, groups


def measure(stage, contexts, specs, shard, shards, swap=False):
    """C80 / held-out queries + BF16 reader KL of the arms owned by this shard: results/<stage>/<arm>/<ctx>.pt."""
    torch = __import__('torch')
    cut = the_cut()
    names = owned(shard, shards, list(specs))
    todo = {a: [c for c in contexts if not (R / stage / a / f'{key(c)}.pt').exists()] for a in names}
    todo = {a: v for a, v in todo.items() if v}
    if not todo:
        return
    w = Work(shard, shards)
    w.build(contexts)
    w.load_stats()
    need = sorted({c for v in todo.values() for c in v})
    readers = w.readers(need, cut)
    for arm, cs in todo.items():
        w.install(specs[arm])
        rows = {}
        for ctx in cs:
            enough()
            t = w.teacher(ctx)
            rows[ctx] = w.student(ctx, cut, swap=t['cut'][cut] if swap else None)
        w.install(['BF16'] * TARGETS)
        for ctx, row in rows.items():
            row['K'] = w.reader_kl(readers[ctx], row['z'])
            torch_save(row, R / stage / arm / f'{key(ctx)}.pt')
        progress(stage, arm=arm, contexts=len(cs))
    # the readers must still be identical after every install
    ctx = need[0]
    w.install(['BF16'] * TARGETS)
    if readers[ctx].identity() > CFG['identity_max_abs']:
        raise RuntimeError('A kept reader drifted after the arm installs')
    del torch


def errors(stage, arm, contexts, M):
    """Per-context A, Z, Z_M, K, gripper, final-future NMSE (and the swap action error) of a measured arm."""
    import cosmos_lavq as cl
    cut = the_cut()
    out = {}
    for ctx in contexts:
        row = tload(R / stage / arm / f'{key(ctx)}.pt')
        t = tload(R / 'teacher' / f'{key(ctx)}.pt')
        e = dict(A=float(cl.action_mse(row['action'], t['action'])), Z=float(cl.nmse(row['z'], t['cut'][cut])),
                 ZM=float(M.nmse(row['z'], t['cut'][cut])), K=row['K'],
                 G=float(cl.gripper_mse(row['action'], t['action'])), F=float(cl.nmse(row['final'], t['final'])))
        if 'swap' in row:
            e['A_swap'] = float(cl.action_mse(row['swap'], t['action']))
        out[ctx] = e
    return out


def summarize(errs, lam_M, beta):
    n = len(errs)
    m = {k: sum(e[k] for e in errs.values()) / n for k in next(iter(errs.values()))}
    m['L'] = m['A'] + lam_M * m['ZM'] + beta * m['K']
    m['motor_rmse'] = math.sqrt(m['A'])
    if 'A_swap' in m:
        m['rho'] = 1 - m['A_swap'] / m['A']
    return m


# ---------------- (2) allocation ----------------
def sens(shard, shards):
    specs, _ = arm_specs()
    measure('sens', c80(), specs, shard, shards)


def exchanges(fam, groups, gain, cost, layers, bits, proxy, sizes, max_ex):
    """Prefixes of the exchange sequence (protocol allocation.rule); each prefix = {index: K}."""
    kb, kd, ku = fam['K'], fam['down'], fam['up']
    up_bits = {g: sum(bits[i][ku] - bits[i][kb] for i in layers[g]) for g in groups}
    dn_bits = {g: sum(bits[i][kb] - bits[i][kd] for i in layers[g]) for g in groups}
    gdens = {g: gain[g] / up_bits[g] for g in groups}
    cdens = {g: cost[g] / dn_bits[g] for g in groups}
    recv = sorted([g for g in groups if gain[g] > 0 or SMOKE], key=lambda g: (-gdens[g], g))
    lden = {i: sizes[i] * (proxy[i][kb] - proxy[i][ku]) / (bits[i][ku] - bits[i][kb]) for g in groups for i in layers[g]}
    queue = [(g, i) for g in recv for i in sorted(layers[g], key=lambda i: (-lden[i], i))]

    def allocation(donors):
        K = {i: kb for g in groups for i in layers[g]}
        budget = 0.0
        for d in donors:
            for i in layers[d]:
                K[i] = kd
                budget += bits[i][kb] - bits[i][kd]
        last = None
        for g, i in queue:
            if g in donors:
                continue
            add = bits[i][ku] - bits[i][kb]
            if add <= budget:
                K[i], budget, last = ku, budget - add, g
        return K, last, budget

    prefixes, donors, log = [allocation([])[0]], [], []
    for d in sorted(groups, key=lambda g: (cdens[g], g)):
        if len(donors) == max_ex:
            break
        trial = donors + [d]
        K, last, left = allocation(trial)
        if last is None or (cdens[d] >= gdens[last] and not SMOKE):
            log.append(dict(donor=d, accepted=False, cost_density=cdens[d],
                            marginal_gain_density=None if last is None else gdens[last]))
            break
        donors = trial
        prefixes.append(K)
        log.append(dict(donor=d, accepted=True, cost_density=cdens[d], marginal=last, marginal_gain_density=gdens[last],
                        raised=sum(1 for i in K if K[i] == ku), budget_left=left))
    return prefixes, donors, log, dict(up_bits=up_bits, down_bits=dn_bits, gain_density=gdens, cost_density=cdens)


def sens_final():
    specs, groups = arm_specs()
    M = load_metric('cpu')
    errs = {a: errors('sens', a, c80(), M) for a in specs}
    lc = P['loss']
    base_c = errs[{'QTIP_W2': 'W2_base', 'QTIP_W3': 'W3_base'}[lc['coefficient_arm']]]
    mean = lambda e, k: sum(x[k] for x in e.values()) / len(e)
    lam = lc['rho'] * mean(base_c, 'A') / mean(base_c, 'Z')
    beta = lc['beta_fraction'] * lam * mean(base_c, 'Z') / mean(base_c, 'K')
    rows = modules()
    layers = {g: [r['index'] for r in rows if group_of(r['module'])[0] == g] for g in groups}
    sizes = {r['index']: r['shape'][0] * r['shape'][1] for r in rows}
    bits, proxy = bits_table()
    plan, table = {}, {}
    for b, fam in P['allocation']['families'].items():
        base = errs[f'W{b}_base']
        lam_M = lam * mean(base, 'Z') / mean(base, 'ZM')
        S = {a: summarize(errs[a], lam_M, beta) for a in specs if a.startswith(f'W{b}_')}
        L0 = S[f'W{b}_base']['L']
        gain = {g: L0 - S[f'W{b}_{g}_up']['L'] for g in groups}
        cost = {g: S[f'W{b}_{g}_down']['L'] - L0 for g in groups}
        f = dict(fam, K=int(fam['base'][-1]))
        prefixes, donors, log, dens = exchanges(f, groups, gain, cost, layers, bits, proxy, sizes,
                                                P['allocation']['max_exchanges'])
        base_bits = sum(bits[i][f['K']] for i in bits)
        pre = []
        for j, K in enumerate(prefixes):
            total = sum(bits[i][K[i]] for i in K)
            if total > base_bits:
                raise RuntimeError(f'Prefix {j} of W{b} exceeds the budget: {total} > {base_bits}')
            pre.append(dict(name=f'W{b}_base' if j == 0 else f'W{b}_P{j}', K={str(i): K[i] for i in sorted(K)},
                            bits_total=total, budget=base_bits, donors=donors[:j],
                            raised=sum(1 for i in K if K[i] == f['up']), lowered=sum(1 for i in K if K[i] == f['down'])))
        plan[b] = dict(lam_M=lam_M, base=fam['base'], gain=gain, cost=cost, log=log, prefixes=pre, **dens)
        table[b] = S
    save(R / 'alloc' / 'plan.json', dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], cut=the_cut(),
                                         lam=lam, beta=beta, coefficient_arm=lc['coefficient_arm'], families=plan,
                                         sens=table))
    progress('sens_final', lam=lam, beta=beta, prefixes={b: len(v['prefixes']) for b, v in plan.items()})


def prefix_specs():
    plan = read(R / 'alloc' / 'plan.json')
    specs = {}
    for b, f in plan['families'].items():
        for pr in f['prefixes'][1:]:
            specs[pr['name']] = [f'QTIP_W{pr["K"][str(i)]}' for i in range(TARGETS)]
    return specs


def alloc(shard, shards):
    specs = prefix_specs()
    if specs:
        measure('alloc', c80(), specs, shard, shards)


def assemble(arm, K, budget, extra=None):
    """results/quantized/<arm>: weight links to the per-layer QTIP sources, sources.json, COMPLETE.json."""
    out = R / 'quantized' / arm
    rows = modules()
    src, total, weights = [], 0, 0
    for r in rows:
        i = r['index']
        k = int(K[str(i)])
        source = arm_dir(f'QTIP_W{k}')
        link = out / 'weights' / f'{i:04d}.pt'
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(source / 'weights' / f'{i:04d}.pt', link)
        s = stats(f'QTIP_W{k}', i)
        total += s['bits_total']
        weights += s['shape'][0] * s['shape'][1]
        src.append(dict(index=i, module=r['module'], K=k, role=group_of(r['module'])[1],
                        source=str(source / 'weights' / f'{i:04d}.pt'), codes=str(source / 'codes' / f'{i:04d}.pt'),
                        stats=str(source / 'stats' / f'{i:04d}.json')))
    if total > budget:
        raise RuntimeError(f'{arm}: {total} bits > budget {budget}')
    save(out / 'sources.json', src)
    save(out / 'COMPLETE.json', dict(protocol=P['protocol'], arm=arm, method='lavq', complete=True,
                                     pbs_jobid=os.environ['PBS_JOBID'], bits_total=total, budget=budget,
                                     target_weights=weights, bpw=total / weights, **(extra or {})))


def alloc_final():
    plan = read(R / 'alloc' / 'plan.json')
    M = load_metric('cpu')
    chosen = {}
    for b, f in plan['families'].items():
        rows = []
        for pr in f['prefixes']:
            stage = 'sens' if pr['name'].endswith('_base') else 'alloc'
            S = summarize(errors(stage, pr['name'], c80(), M), f['lam_M'], plan['beta'])
            rows.append(dict(pr, c80=S))
        best = len(rows) - 1 if SMOKE else min(range(len(rows)), key=lambda j: (rows[j]['c80']['L'], j))
        chosen[b] = dict(prefix=best, name=rows[best]['name'], rows=[{k: v for k, v in r.items() if k != 'K'} for r in rows])
        assemble(f'LAVQU_W{b}', rows[best]['K'], rows[0]['budget'],
                 dict(prefix=best, identical_to_host=best == 0, host=f['base']))
    save(R / 'alloc' / 'chosen.json', dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], families=chosen))
    progress('alloc_final', chosen={b: c['prefix'] for b, c in chosen.items()})


# ---------------- held-out ----------------
def heldout_arms():
    arms = [f'{h}_W{b}' for b in ('2', '3') for h in HOSTS]
    for b in ('2', '3'):
        c = read(R / 'quantized' / f'LAVQU_W{b}' / 'COMPLETE.json')
        if not c['identical_to_host']:
            arms.append(f'LAVQU_W{b}')
    return arms


def heldout_stage(shard, shards):
    specs = {}
    for arm in heldout_arms():
        if arm.startswith('LAVQU'):
            src = read(R / 'quantized' / arm / 'sources.json')
            specs[arm] = [f'QTIP_W{r["K"]}' for r in src]
        elif arm.startswith('QTIP'):
            specs[arm] = [arm] * TARGETS
        else:
            specs[arm] = [arm] * TARGETS
    measure('heldout', heldout(), specs, shard, shards, swap=True)


def heldout_final():
    plan = read(R / 'alloc' / 'plan.json')
    M = load_metric('cpu')
    table, gate, bases = {}, {}, {}
    for arm in heldout_arms():
        b = arm[-1]
        table[arm] = summarize(errors('heldout', arm, heldout(), M), plan['families'][b]['lam_M'], plan['beta'])
    for b in ('2', '3'):
        host = f'QTIP_W{b}'
        arm = f'LAVQU_W{b}'
        if arm in table:
            ok = table[arm]['L'] <= table[host]['L'] or SMOKE
            gate[b] = dict(passed=ok, L=table[arm]['L'], host_L=table[host]['L'],
                           ratios={k: table[arm][k] / table[host][k] for k in ('L', 'A', 'Z', 'ZM', 'K', 'G', 'F')
                                   if table[host][k] > 0})
            bases[b] = arm if ok else host
        else:
            gate[b] = dict(passed=None, note='prefix 0: LAVQU is the uniform host')
            bases[b] = host
    save(R / 'heldout' / 'summary.json', dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], cut=the_cut(),
                                              table=table, gate=gate, calib_base=bases))
    progress('heldout_final', gate={b: g.get('passed') for b, g in gate.items()}, bases=bases)


# ---------------- (4) shared-table calibration ----------------
def base_K(b):
    """{index: K} of the calibration base of family b (LAVQU_W<b> if it passed, else the uniform host)."""
    base = read(R / 'heldout' / 'summary.json')['calib_base'][b]
    if base.startswith('LAVQU'):
        return base, {r['index']: r['K'] for r in read(R / 'quantized' / base / 'sources.json')}
    return base, {i: int(b) for i in range(TARGETS)}


def codes(shard, shards):
    import torch
    import qmath
    Q, _, permute_inv = qtip_module()
    table = qmath.lut_values(Q.codebook(2, 'cpu').tlut.to(torch.bfloat16))
    items = sorted({(i, k) for b in ('2', '3') for i, k in base_K(b)[1].items()})
    shape = {r['index']: r['shape'] for r in modules()}
    mine = owned(shard, shards, items, lambda it: shape[it[0]][0] * shape[it[0]][1])
    for i, k in mine:
        out = R / 'codes' / f'K{k}' / f'{i:04d}.pt'
        if out.exists():
            continue
        enough()
        arm = f'QTIP_W{k}'
        c = torch.load(arm_dir(arm) / 'codes' / f'{i:04d}.pt', map_location='cpu', weights_only=False)
        w = torch.load(arm_dir(arm) / 'weights' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
        name = modules()[i]['module']
        if c['module'] != name or w['module'] != name:
            raise RuntimeError(f'Codes / weight identity differs at {i} K={k}')
        if not torch.equal(qmath.lut_values(c['codes']['tlut']), table):
            raise RuntimeError(f'Layer {i} K={k}: stored LUT differs from the global QTIP table')
        saved = qmath.layer_codes(Q, c['codes'], stats(arm, i)['config'], w['weight'], table, permute_inv, 'cuda')
        torch_save(dict(saved, module=name, index=i, K=k, source=str(arm_dir(arm) / 'weights' / f'{i:04d}.pt')), out)
        progress('codes', index=i, K=k)


class Calib(Work):
    """One family's shared-table calibration (ravqc v3: tables only, max-normalized step, FP32-decode fold)."""

    def attach(self, b):
        import torch
        import qmath
        Q, permute, permute_inv = qtip_module()
        self.base, K = base_K(b)
        self.install([f'QTIP_W{K[i]}' for i in range(TARGETS)])
        self.layers, self.tables, self.live = [], {}, {}
        for i, name in enumerate(self.names):
            saved = torch.load(R / 'codes' / f'K{K[i]}' / f'{i:04d}.pt', map_location='cpu', weights_only=True)
            if saved['module'] != name:
                raise RuntimeError(f'Codes of layer {i} do not belong to {name}')
            group = f'{group_of(name)[1]}@{K[i]}'
            L = qmath.QLayer(i, name, self.mods[i], group, saved, Q, permute, permute_inv, self.device)
            if not torch.equal(L.W0, self.layer(f'QTIP_W{K[i]}', i).cpu()):
                raise RuntimeError(f'Installed weight of {name} is not the base source')
            self.layers.append(L)
            if group not in self.tables:
                self.tables[group] = torch.zeros(qmath.ENTRIES, 2, device=self.device, requires_grad=True)
                self.live[group] = torch.zeros(qmath.ENTRIES, device=self.device)
            self.live[group] += saved['counts'].to(self.device).float()
            self.mods[i].forward = self._wrapped(L, self.mods[i].forward)
        self.table0 = qmath.lut_values(Q.codebook(2, 'cpu').tlut.to(torch.bfloat16)).to(self.device)
        self.lut_rms = float(self.table0.square().mean().sqrt())
        self.active = False
        self.student_w = [m.weight.data for m in self.mods]
        progress('attach', base=self.base, layers=len(self.layers), groups=sorted(self.tables))

    def _wrapped(self, L, plain):
        import qmath

        def run(x):
            if not self.active:
                return plain(x)
            return qmath.QTIPGrad.apply(x, self.tables[L.group], L)
        return run

    def bf16(self, on):
        """Pointer swap between the student's weights and the BF16 masters (the frozen reader reads BF16)."""
        for m, s, w in zip(self.mods, self.student_w, self.masters):
            m.weight.data = w if on else s

    def rebuild(self):
        for L in self.layers:
            L.rebuild(self.tables[L.group], self.table0)

    def snapshot(self):
        return {g: v.detach().cpu().clone() for g, v in self.tables.items()}

    def restore(self, saved):
        with self.torch.no_grad():
            for g, v in self.tables.items():
                v.copy_(saved[g].to(self.device))
        self.rebuild()

    def objective(self, batch, grad):
        """mean over the batch of A + lam_M Z_M + beta K; with grad, accumulated into the tables by the staged replay
        (no-grad queries -> A / Z_M cotangents, frozen BF16 reader -> beta K cotangent at the cut future, grad replay)."""
        torch, cl, cut = self.torch, self.cl, self.cut
        staged = []
        for ctx in batch:
            enough()
            t = self.teacher(ctx)
            result, cap = cl.run(self.rt, self.context(ctx), (cut,))
            a = result.action.detach().float().clone().requires_grad_(True)
            z = cap.future(cut).to(self.device).requires_grad_(True)
            A = cl.action_mse(a, t['action'])
            Z = cl.nmse(z, t['cut'][cut])
            ZM = self.M.nmse(z, t['cut'][cut])
            ga, gz = torch.autograd.grad(A + self.lam_M * ZM, (a, z))
            staged.append(dict(ctx=ctx, action=a.detach(), z=z.detach(), ga=ga, gz=gz, A=float(A), Z=float(Z), ZM=float(ZM)))
        self.bf16(True)
        try:
            for row in staged:
                reader = cl.Reader(self.rt, self.context(row['ctx']), cut)
                leaf = row['z'].clone().requires_grad_(grad)
                with torch.set_grad_enabled(grad):
                    k = cl.reader_kl(reader.read(leaf), reader.action, self.astats, CFG)
                row['K'] = float(k.detach())
                if grad:
                    rg, = torch.autograd.grad(self.beta * k, leaf)
                    row['gz'] = row['gz'] + rg.detach()
                del reader, leaf, k
        finally:
            self.bf16(False)
        n = len(staged)
        out = {k: sum(r[k] for r in staged) / n for k in ('A', 'Z', 'ZM', 'K')}
        out['total'] = out['A'] + self.lam_M * out['ZM'] + self.beta * out['K']
        if grad:
            worst = 0.0
            for row in staged:
                self.active = True
                try:
                    result, cap = cl.run(self.rt, self.context(row['ctx']), (cut,), retain=True, checkpoint=True)
                    action = result.action.float()
                    zl = cap.live[cut][:, :, cl.FUT]
                    if not (action.requires_grad and zl.requires_grad):
                        raise RuntimeError('Calibrated query lost its graph')
                    worst = max(worst, float((action.detach() - row['action']).abs().max()),
                                float((zl.detach() - row['z']).abs().max()))
                    torch.autograd.backward((action, zl), (row['ga'] / n, row['gz'] / n))
                finally:
                    self.active = False
                del result, cap, action, zl
            out['replay_max_abs'] = worst
            if worst > P['training']['replay_max_abs']:
                raise RuntimeError(f'Grad replay differs from the staged query: {worst}')
        torch.cuda.empty_cache()
        return out

    def step(self, eta):
        """delta u = -eta g_u / max|g_u| over every live table entry, u = entry / rms(tlut); predicted decrease."""
        torch = self.torch
        with torch.no_grad():
            gu = {g: (v.grad if v.grad is not None else torch.zeros_like(v)) * self.lut_rms
                  * (self.live[g] > 0)[:, None].float() for g, v in self.tables.items()}
            top = max(float(x.abs().max()) for x in gu.values())
            if not math.isfinite(top) or top == 0:
                raise RuntimeError(f'Table gradient scale {top}')
            pred = 0.0
            for g, v in self.tables.items():
                v.sub_(eta * gu[g] / top * self.lut_rms)
                pred += eta * float(gu[g].square().sum()) / top
        self.rebuild()
        return pred, top

    def zero(self):
        for v in self.tables.values():
            v.grad = None


def schedule():
    cs = c80()
    rng = random.Random(P['training']['schedule_seed'])
    rng.shuffle(cs)
    n = P['training']['contexts_per_update']
    return [cs[i * n:(i + 1) * n] for i in range(P['training']['updates'])]


def calib(shard, shards):
    torch = __import__('torch')
    fams = owned(shard, shards, ['2', '3'])
    plan = read(R / 'alloc' / 'plan.json')
    for b in fams:
        root = R / 'calib' / f'LAVQ_W{b}'
        if (root / 'arm.json').exists():
            continue
        w = Calib(shard, shards)
        w.build(c80() + heldout())
        w.load_stats()
        w.M = load_metric(w.device)
        w.cut, w.beta = the_cut(), plan['beta']
        w.attach(b)
        # lam_M at the base (its C80 measurement): lam mean Z / mean Z_M
        stage = 'alloc' if w.base.startswith('LAVQU') and read(R / 'quantized' / w.base / 'COMPLETE.json')['prefix'] > 0 else 'sens'
        name = (read(R / 'alloc' / 'chosen.json')['families'][b]['name'] if w.base.startswith('LAVQU') else f'W{b}_base')
        before = summarize(errors(stage, name, c80(), w.M), 0.0, 0.0)
        w.lam_M = plan['lam'] * before['Z'] / before['ZM']
        cfg, sched = P['training'], schedule()
        state = dict(updates=0, history=[], eta=None)
        if (root / 'checkpoint.pt').exists():
            saved = tload(root / 'checkpoint.pt')
            w.restore(saved['tables'])
            state = saved['state']
        if state['eta'] is None:
            w.zero()
            base = w.objective(sched[0], grad=True)
            start = w.snapshot()
            grads = {g: v.grad.detach().clone() for g, v in w.tables.items()}
            rows = []
            for eta in cfg['step_grid']:
                for g, v in w.tables.items():
                    v.grad = grads[g].clone()
                pred, top = w.step(eta)
                after = w.objective(sched[0], grad=False)
                rows.append(dict(lr=eta, predicted=pred, realized=base['total'] - after['total'], after=after, top=top))
                w.restore(start)
                progress('step', arm=f'LAVQ_W{b}', eta=eta, predicted=pred, realized=rows[-1]['realized'])
            import qmath
            chosen = qmath.first_step_choice(rows, cfg['min_realized_fraction'])
            if SMOKE and chosen is None:
                chosen = cfg['step_grid'][0]
            save(root / 'step.json', dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], base_arm=w.base,
                                          lam_M=w.lam_M, beta=w.beta, batch=[list(c) for c in sched[0]], before=base,
                                          rows=rows, chosen=chosen,
                                          table_grad_norm={g: float(v.norm()) for g, v in grads.items()}))
            state['eta'] = chosen
            if chosen is None:
                save(root / 'arm.json', dict(protocol=P['protocol'], arm=f'LAVQ_W{b}', base=w.base, tuned=False,
                                             pbs_jobid=os.environ['PBS_JOBID'], update_rule=cfg['update_rule_id']))
                w.rt.close()
                continue
            for g, v in w.tables.items():
                v.grad = grads[g].clone()
            pred, top = w.step(chosen)
            state['updates'] = 1
            state['history'].append(dict(update=1, batch=[list(c) for c in sched[0]], before=base, predicted=pred, top=top))
            torch_save(dict(tables=w.snapshot(), state=state), root / 'checkpoint.pt')
        while state['updates'] < cfg['updates']:
            u = state['updates']
            w.zero()
            out = w.objective(sched[u], grad=True)
            pred, top = w.step(state['eta'])
            state['updates'] += 1
            state['history'].append(dict(update=u + 1, batch=[list(c) for c in sched[u]], before=out, predicted=pred, top=top))
            torch_save(dict(tables=w.snapshot(), state=state), root / 'checkpoint.pt')
            save(root / 'training.json', dict(protocol=P['protocol'], arm=f'LAVQ_W{b}', lam_M=w.lam_M, **state))
            progress('train', arm=f'LAVQ_W{b}', updates=state['updates'], loss=out['total'])
        w.zero()
        deploy(w, b, state)
        # held-out (and the C80 fit after) of the calibrated arm, same measurement as the other arms
        arm = f'LAVQ_W{b}'
        readers = None
        for stage, contexts, swap in (('heldout', heldout(), True), ('calibfit', c80(), False)):
            todo = [c for c in contexts if not (R / stage / arm / f'{key(c)}.pt').exists()]
            if not todo:
                continue
            rows = {}
            for ctx in todo:
                enough()
                t = w.teacher(ctx)
                rows[ctx] = w.student(ctx, w.cut, swap=t['cut'][w.cut] if swap else None)
            w.bf16(True)
            try:
                for ctx, row in rows.items():
                    reader = w.cl.Reader(w.rt, w.context(ctx), w.cut)
                    row['K'] = w.reader_kl(reader, row['z'])
                    torch_save(row, R / stage / arm / f'{key(ctx)}.pt')
                    del reader
            finally:
                w.bf16(False)
            progress(stage, arm=arm, contexts=len(todo))
        save(root / 'arm.json', dict(protocol=P['protocol'], arm=arm, base=w.base, tuned=True, eta=state['eta'],
                                     lam_M=w.lam_M, beta=w.beta, pbs_jobid=os.environ['PBS_JOBID'],
                                     update_rule=cfg['update_rule_id']))
        w.rt.close()
        del readers


def deploy(w, b, state):
    """results/quantized/LAVQ_W<b>: the calibrated weights (exactly the training forward's) and COMPLETE.json (the base's
    bits plus the group tables)."""
    torch = w.torch
    arm, out = f'LAVQ_W{b}', R / 'quantized' / f'LAVQ_W{b}'
    torch_save(w.snapshot(), R / 'calib' / arm / 'tables.pt')
    changed = []
    for L in w.layers:
        weight = L.mod.weight.detach().cpu().clone()
        torch_save(dict(module=L.name, weight=weight), out / 'weights' / f'{L.index:04d}.pt')
        changed.append(dict(index=L.index, module=L.name, group=L.group,
                            rel=float((weight.float() - L.W0.float()).norm() / L.W0.float().norm())))
    base_bits = (read(R / 'quantized' / w.base / 'COMPLETE.json')['bits_total'] if w.base.startswith('LAVQU')
                 else sum(stats(w.base, i)['bits_total'] for i in range(TARGETS)))
    weights = sum(r['shape'][0] * r['shape'][1] for r in modules())
    extra = 16 * 512 * 2 * len(w.tables)
    save(out / 'sources.json', changed)
    save(out / 'COMPLETE.json', dict(protocol=P['protocol'], arm=arm, method='lavq', complete=True, base=w.base,
                                     pbs_jobid=os.environ['PBS_JOBID'], bits_total=base_bits + extra, extra_bits=extra,
                                     target_weights=weights, bpw=(base_bits + extra) / weights, eta=state['eta'],
                                     lam_M=w.lam_M, updates=state['updates'], groups=sorted(w.tables)))
    del torch


# ---------------- final ----------------
def final():
    plan = read(R / 'alloc' / 'plan.json')
    held = read(R / 'heldout' / 'summary.json')
    M = load_metric('cpu')
    table = dict(held['table'])
    decision, calib_rows = {}, {}
    for b in ('2', '3'):
        base = held['calib_base'][b]
        arm = f'LAVQ_W{b}'
        a = read(R / 'calib' / arm / 'arm.json')
        if a['tuned']:
            table[arm] = summarize(errors('heldout', arm, heldout(), M), plan['families'][b]['lam_M'], plan['beta'])
            base_row = table[base] if base in table else table[f'QTIP_W{b}']
            ok = table[arm]['L'] <= base_row['L']
            calib_rows[b] = dict(tuned=True, passed=ok, L=table[arm]['L'], base_L=base_row['L'],
                                 ratios={k: table[arm][k] / base_row[k] for k in ('L', 'A', 'Z', 'ZM', 'K', 'G', 'F')
                                         if base_row[k] > 0},
                                 c80_after=summarize(errors('calibfit', arm, c80(), M), plan['families'][b]['lam_M'],
                                                     plan['beta']))
        else:
            ok = False
            calib_rows[b] = dict(tuned=False, passed=False)
        chosen = arm if ok else base
        loop = chosen != f'QTIP_W{b}'
        decision[b] = dict(chosen=chosen, loop=loop, calib_base=base,
                           note=None if loop else 'uniform host: its A16 loop is the result')
    loop_arms = [d['chosen'] for d in decision.values() if d['loop']]
    for arm in loop_arms:
        if not (R / 'quantized' / arm / 'COMPLETE.json').exists():
            raise RuntimeError(f'{arm} has no assembled weights')
    save(R / 'DECISION.json', dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], families=decision,
                                   loop_arms=loop_arms))
    summary = dict(protocol=P['protocol'], pbs_jobid=os.environ['PBS_JOBID'], teacher=read(R / 'teacher' / 'summary.json'),
                   metric=read(R / 'metric' / 'summary.json'), lam=plan['lam'], beta=plan['beta'],
                   allocation={b: {k: v for k, v in f.items() if k != 'prefixes'} for b, f in plan['families'].items()},
                   chosen=read(R / 'alloc' / 'chosen.json'), heldout=table, lavqu_gate=held['gate'], calibration=calib_rows,
                   decision=decision)
    save(R / 'summary.json', summary)
    (R / 'RESULTS.zh.md').write_text(report(summary, plan), encoding='utf-8')
    progress('final', loop_arms=loop_arms)


def report(s, plan):
    t, m = s['teacher'], s['metric']
    L = ['# Cosmos-Policy 上的 LAVQ', '',
         f"协议 `{P['protocol']}`。reader 切点 = 第 {t['cut']} 次 denoiser 评估之前（候选 {CFG['cut_candidates']}，按 C80 上"
         f"未来替换去掉的动作误差份额 ρ 取最大）。", '',
         '## C80 上各切点的 future share ρ', '', '| 臂 | ' + ' | '.join(f'cut {c}' for c in CFG['cut_candidates']) + ' |',
         '|---|' + '---:|' * len(CFG['cut_candidates'])]
    for arm, r in t['rho_c80'].items():
        L.append(f'| {arm} | ' + ' | '.join(f"{r[str(c)]:.3f}" for c in CFG['cut_candidates']) + ' |')
    st, sb = m['stats'], m['stability']
    L += ['', '## ① reader metric', '',
          f"维度 {t['future_dims']}；有效秩 {st['effective_rank']:.1f}；top-{P['metric']['rank']} 迹占比 {st['top_share']:.2f}；"
          f"对角前 10% 占比 {st['diag_top10_share']:.2f}；两视角迹占比 {', '.join(f'{x:.2f}' for x in m['gaze']['views'])}；"
          f"半数对角相关 {sb['diag_corr']:.2f}，top-r 子空间重叠 {sb['subspace_overlap']:.2f}。", '',
          f"λ = {s['lam']:.4g}，β = {s['beta']:.4g}（QTIP_W2 C80 初值规则）。", '', '## ② 分组交换（C80）', '']
    for b, f in s['allocation'].items():
        L += [f"### W{b}（λ_M = {f['lam_M']:.4g}）", '', '| 组 | 升一位收益 | 降一位代价 | 收益密度 | 代价密度 |', '|---|---:|---:|---:|---:|']
        for g in sorted(f['gain']):
            L.append(f"| {g} | {f['gain'][g]:.3e} | {f['cost'][g]:.3e} | {f['gain_density'][g]:.3e} | {f['cost_density'][g]:.3e} |")
        ch = s['chosen']['families'][b]
        L += ['', '| 前缀 | 降位组 | 升位层数 | bits / 预算 | C80 L |', '|---|---|---:|---:|---:|']
        for r in ch['rows']:
            L.append(f"| {r['name']} | {', '.join(r['donors']) or '—'} | {r['raised']} | {r['bits_total'] / r['budget']:.4f} | "
                     f"{r['c80']['L']:.4e} |")
        L += ['', f"选中前缀 {ch['prefix']}（{ch['name']}）。", '']
    L += ['## held-out 40×2', '', '| 臂 | L | A | Z | Z_M | K | gripper | 最终未来 NMSE | ρ |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, r in s['heldout'].items():
        L.append(f"| {arm} | {r['L']:.4e} | {r['A']:.4e} | {r['Z']:.4f} | {r['ZM']:.4f} | {r['K']:.4e} | {r['G']:.4e} | "
                 f"{r['F']:.4f} | {r.get('rho', float('nan')):.3f} |")
    L += ['', '## 门槛与决定', '']
    for b in ('2', '3'):
        g, c, d = s['lavqu_gate'][b], s['calibration'][b], s['decision'][b]
        L.append(f"- W{b}：LAVQU 门槛 {g.get('passed')}（L 比 {g.get('ratios', {}).get('L', float('nan')):.3f}）；"
                 f"校准 tuned={c['tuned']} passed={c['passed']}"
                 + (f"（L 比 {c['ratios']['L']:.3f}）" if c.get('ratios') else '') + f"；闭环臂 {d['chosen']}（loop={d['loop']}）。")
    return '\n'.join(L) + '\n'


STAGES = dict(quant=quant, quant_final=quant_final, teacher=teacher, teacher_final=teacher_final, metric=metric,
              metric_final=metric_final, sens=sens, sens_final=sens_final, alloc=alloc, alloc_final=alloc_final,
              heldout=heldout_stage, heldout_final=heldout_final, codes=codes, calib=calib, final=final)
SHARDED = {'quant', 'teacher', 'metric', 'sens', 'alloc', 'heldout', 'codes', 'calib'}


def main():
    guard()
    stage = os.environ['LV_STAGE']
    shard, shards = int(os.environ.get('LV_SHARD', 0)), int(os.environ.get('LV_SHARDS', 1))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    begin = time.monotonic()
    try:
        if stage in SHARDED:
            STAGES[stage](shard, shards)
        else:
            STAGES[stage]()
    except Deadline:
        print(f'LAVQ_DEADLINE stage={stage} shard={shard}', flush=True)
        sys.exit(3)
    progress('done', stage_done=stage, shard=shard, seconds=round(time.monotonic() - begin))


if __name__ == '__main__':
    main()
