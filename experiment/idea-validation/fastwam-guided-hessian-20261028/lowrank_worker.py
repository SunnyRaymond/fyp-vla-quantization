"""C2 v2 gate (lowrank.json): does a C1-derived objective repair the imagined future once the tuned parameters have
capacity in every layer, and does decoupling the action term add to it?

QTIP W2 (A16 baseline, H_U) plus a rank-r BF16 branch on each of the 614 target Linears, y = x W^T + b + (x V^T) U^T:
a parametrization of any Linear of any host. Three arms share the branch, the start and the budget:
  E  EoRA-style init L = [dW C]_r C^-1 (dW = W_master - W_host, C C^T = damped H_U); no training, the prior-art reference
  J  joint objective A + lambda Z + beta K from E (tune_worker's staged batch)
  D  decoupled objective A_tf + lambda Z + beta K from E: A_tf is the student's action reading the BF16 teacher future
     (the C1 swap path), so the action side gets no reward for compensating a corrupted future
Updates are per-tensor relative steps P <- P - eta |P| g / |g| (every Linear moves by eta of its factor norm), eta from
the grid by the step rule on batch 1, 8 updates on the C80 batches (Q0 budget). Stages (FW_STAGE):
  init   shard 0: factors, E's BF16 factors, bits and proxy, C80 fit, held-out 40x2 + swap
  train  J and D (or FW_ARMS): one per GPU or in turn on one GPU
  final  results/lowrank/{summary.json, RESULTS.zh.md, DECISION.json}
  diag   (lowrank.json diag) why the C80 repair does not reach held-out: C80 at an unseen sampler seed for E, J, D, and
         per-(context, seed) gradients of D's objective at E split by path -> results/lowrank/diag/
  tfdiag (lowrank.json tfdiag) the per-step teacher-forced future objective at E: per-sample gradients next to the
         rollout ones, how much of them other contexts and seeds share, and what a step on it does to the rollout
         future of other contexts -> results/lowrank/tfdiag/
  tcond  (lowrank.json tcond) attribution: U and E with the time-conditioning Linears at their BF16 masters (outputs
         exact; their inputs are the timestep only), held-out 40x2 + swap -> results/lowrank/tcond/
  xinit / xtrain  (lowrank.json xfer) E, then J, unchanged on another host's W2 (GPTVQ_W2, QUIPSHARP_W2): one host per
         GPU, FW_HOSTS = the hosts in shard order; arms <HOST>_LRE_W2 / <HOST>_LRJ_W2, init files in lowrank/init_<HOST>
  xfinal results/lowrank/xfer/{summary.json, RESULTS.zh.md}: U, E, J per host, QTIP's from the gate for reference
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import inspect
import json
import math
import os
from pathlib import Path
import time
from types import MethodType

import torch
import torch.nn.functional as F

import gate_worker as gw
import lp_worker as lp
import tune_worker as tw
import worker as wk

HERE = Path(__file__).resolve().parent
LR = json.loads((HERE / 'lowrank.json').read_text(encoding='utf-8'))
ARMS = {k: v['arm'] for k, v in LR['arms'].items()}
TC_ARMS = {v['arm']: v for v in LR['tcond']['arms'].values()}
TRAINED = ('J', 'D')
DEVICE = tw.DEVICE
HOST, BASELINE_ARM, METHOD, INIT_DIR = LR['host'], LR['baseline_arm'], 'qtip', 'init'


def host_arms(host):
    if host == LR['host']:
        return {k: v['arm'] for k, v in LR['arms'].items()}
    return {k: f'{host}_LR{k}_W2' for k in LR['arms']}


def select_host(host):
    """The host whose W2 weights carry the branch: QTIP (the gate's own names and files) or an lowrank.json xfer host."""
    global HOST, BASELINE_ARM, METHOD, INIT_DIR, ARMS
    if host == LR['host']:
        HOST, BASELINE_ARM, METHOD, INIT_DIR = host, LR['baseline_arm'], 'qtip', 'init'
    else:
        c = LR['xfer']['hosts'][host]
        HOST, BASELINE_ARM, METHOD, INIT_DIR = host, c['baseline_arm'], c['method'], f'init_{host}'
    ARMS = host_arms(host)


class LowRank(torch.autograd.Function):
    """(x V^T) U^T in FP32, returned in x's dtype; saves x and the rank-r activation only. Gradients:
    dU = g^T h, dV = (g U)^T x, dx = (g U) V with h = x V^T."""

    @staticmethod
    def forward(ctx, x, u, v):
        h = F.linear(x.float(), v)
        ctx.save_for_backward(x, u, v, h)
        return F.linear(h, u).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, u, v, h = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1]).float()
        gh = g2 @ u
        gu = g2.t() @ h.reshape(-1, h.shape[-1])
        gv = gh.t() @ x.reshape(-1, x.shape[-1]).float()
        gx = (gh @ v).reshape(x.shape).to(x.dtype) if ctx.needs_input_grad[0] else None
        return gx, gu, gv


def eora(dw, h, rank, damping):
    """Rank-r L = U V minimizing tr((dW - L) H (dW - L)^T) with H damped to C C^T: L = [dW C]_r C^-1."""
    eye = torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    d = damping * float(h.diagonal().mean())
    for _ in range(6):
        c, info = torch.linalg.cholesky_ex(h + d * eye)
        if int(info) == 0:
            break
        d *= 10
    else:
        raise RuntimeError('H_U is not positive definite after damping')
    p, s, q = torch.linalg.svd(dw @ c, full_matrices=False)
    root = s[:rank].sqrt()
    u = p[:, :rank] * root[None, :]
    v = torch.linalg.solve_triangular(c.t(), (root[:, None] * q[:rank]).t(), upper=True).t()  # V C = S^1/2 Q^T
    return u, v, dict(damping=d, singular_values=[float(x) for x in s[:rank + 4]])


def energy(e, h):
    return float(((e @ h) * e).sum())


def relative_step(params, grads, eta):
    """In place: P <- P - eta |P| g / |g| per tensor (no step where g = 0)."""
    with torch.no_grad():
        for p, g in zip(params, grads):
            gn = float(g.norm())
            if not math.isfinite(gn):
                raise RuntimeError('Non-finite gradient')
            if gn > 0:
                p.sub_(eta * float(p.norm()) / gn * g)


def decide(rows, rule):
    """rows[k] = dict(tuned, heldout) for U, E, J, D."""
    u, e = rows['U']['heldout'], rows['E']['heldout']
    out = {}
    for k in TRAINED:
        r = rows[k]
        if not r['tuned']:
            out[k] = dict(tuned=False, passed=False)
            continue
        o = r['heldout']
        row = dict(tuned=True, future_ratio=o['future_latent_nmse'] / u['future_latent_nmse'],
                   mixed_ratio=o['mixed_loss'] / u['mixed_loss'],
                   below_E=o['future_latent_nmse'] < e['future_latent_nmse'] and o['mixed_loss'] < e['mixed_loss'])
        row['passed'] = (row['future_ratio'] <= rule['future_ratio_max'] and row['mixed_ratio'] <= rule['mixed_ratio_max']
                         and row['below_E'])
        out[k] = row
    j, d = rows['J'], rows['D']
    increment = (j['tuned'] and d['tuned'] and d['heldout']['future_latent_nmse'] < j['heldout']['future_latent_nmse']
                 and d['heldout']['mixed_loss'] <= j['heldout']['mixed_loss'])
    return dict(arms=out, decoupling_increment=bool(increment), go=out['D']['passed'])


GROUPS = ('video', 'video_time', 'action', 'action_time', 'other')


def tensor_group(name):
    """Group of a factor tensor '<module>.U' / '<module>.V': expert body, expert time conditioning, or other."""
    module = name.rsplit('.', 1)[0]
    head = module.split('.')[0]
    if head not in ('video_expert', 'action_expert'):
        return 'other'
    timed = '.time_embedding' in module or '.time_projection' in module
    return head.split('_')[0] + ('_time' if timed else '')


def diag_analysis(gram, norms, names, samples, batches, eta):
    """gram[t]: Gram matrix of tensor t's per-sample gradients stacked [F_1..F_S, A_1..A_S] over samples
    (case, seed, suite); norms[t] = |P_t| at the start point. Returns
      energy, cosines  per group and path (F, A, total = F + A): the group's share of mean |g|^2, and the mean cosine
                       between samples of the same context at the two seeds / other contexts of the suite / the rest
      snr              per path and batch, |mean g|^2 / mean |g|^2 over the batch's seed-0 samples (1/n if unrelated)
      batches          per batch, the first-order decrease of each category's F, A and total loss under one training
                       step on it (D's relative step, -eta |P_t| gbar_t / |gbar_t|, gbar = mean F + A of the batch's
                       seed-0 samples): in = the batch, seed = its contexts at seed 1, ctx / ctx_seed = the other
                       contexts at seed 0 / 1; F by group too; unit = sum_t |P_t| |gbar_t| (step.json's predicted_per_eta)
      T                per category and path, sum over batches of its decrease over that of the batch itself"""
    gram = gram.double()
    s = len(samples)
    f, a = slice(0, s), slice(s, 2 * s)
    group_of = [tensor_group(n) for n in names]
    index = {(c, sd): i for i, (c, sd, _) in enumerate(samples)}

    def blocks(g):
        return dict(F=g[f, f], A=g[a, a], total=g[f, f] + g[f, a] + g[a, f] + g[a, a])

    pairs = {}
    for i in range(s):
        for j in range(i + 1, s):
            (ci, _, ui), (cj, _, uj) = samples[i], samples[j]
            pairs.setdefault('seed' if ci == cj else 'suite' if ui == uj else 'other', []).append((i, j))
    pairs = {k: torch.tensor(v) for k, v in pairs.items()}
    sums = {grp: torch.zeros(2 * s, 2 * s, dtype=torch.float64) for grp in GROUPS + ('all',)}
    for t, grp in enumerate(group_of):
        sums[grp] += gram[t]
        sums['all'] += gram[t]
    whole = blocks(sums['all'])
    energy, cosines = {}, {}
    for grp, g in sums.items():
        for path, m in blocks(g).items():
            d = m.diagonal()
            energy[f'{grp}/{path}'] = float(d.mean() / whole[path].diagonal().mean().clamp_min(1e-300))
            if float(d.min()) <= 0:
                continue
            c = m / d.sqrt()[:, None] / d.sqrt()[None, :]
            cosines[f'{grp}/{path}'] = {k: float(c[v[:, 0], v[:, 1]].mean()) for k, v in pairs.items()}
    snr = {path: [] for path in whole}
    for b in batches:
        idx = [index[(c, 0)] for c in b]
        for path, m in whole.items():
            sub = m[idx][:, idx]
            snr[path].append(float(sub.mean() / sub.diagonal().mean()))
    out_batches = []
    for b in batches:
        idx = [index[(c, 0)] for c in b]
        cols = idx + [i + s for i in idx]
        dec = {p: torch.zeros(s, dtype=torch.float64) for p in ('F', 'A')}
        dec_f = {grp: torch.zeros(s, dtype=torch.float64) for grp in GROUPS}
        unit = 0.0
        for t, grp in enumerate(group_of):
            g = gram[t]
            fk = g[f][:, cols].sum(1) / len(idx)  # <F_k, gbar_t>
            ak = g[a][:, cols].sum(1) / len(idx)  # <A_k, gbar_t>
            gg = float(fk[idx].mean() + ak[idx].mean())  # |gbar_t|^2
            if gg <= 0:
                continue
            unit += float(norms[t]) * gg ** 0.5
            scale = eta * float(norms[t]) / gg ** 0.5
            dec['F'] += scale * fk
            dec['A'] += scale * ak
            dec_f[grp] += scale * fk
        dec['total'] = dec['F'] + dec['A']
        cats = {}
        for k, (c, sd, _) in enumerate(samples):
            cats.setdefault(('in' if sd == 0 else 'seed') if c in b else ('ctx' if sd == 0 else 'ctx_seed'), []).append(k)
        row = dict(cases=list(b), unit=unit)
        for key, ks in cats.items():
            row[key] = {p: float(v[ks].mean()) for p, v in dec.items()}
            row[key]['F_by_group'] = {grp: float(v[ks].mean()) for grp, v in dec_f.items()}
        out_batches.append(row)
    keys = [k for k in ('seed', 'ctx', 'ctx_seed') if all(k in r for r in out_batches)]
    T = {k: {p: sum(r[k][p] for r in out_batches) / sum(r['in'][p] for r in out_batches) for p in ('F', 'A', 'total')}
         for k in keys}
    return dict(energy=energy, cosines=cosines, pair_counts={k: int(v.shape[0]) for k, v in pairs.items()}, snr=snr,
                batches=out_batches, T=T)


def fit_suites(fits, rows):
    """Per suite, arm and seed: mean C80 future NMSE and mixed loss over the suite's contexts."""
    out = {}
    for key, f in fits.items():
        for sd, v in f.items():
            by = {}
            for r in v['rows']:
                by.setdefault(rows[r['case_id']]['suite'], []).append(r)
            for suite, rs in by.items():
                out.setdefault(suite, {}).setdefault(key, {})[sd] = {
                    m: sum(r[m] for r in rs) / len(rs) for m in ('future_latent_nmse', 'mixed_loss')}
    return out


def diag_reading(fits, analysis, rule):
    """fits[key][seed] = C80 fit summary of arm key at that sampler seed index (0 = the trained trajectories)."""
    seeds = sorted(fits['E'])
    rho = {}
    for key in fits:
        if key == 'E':
            continue
        r = {sd: {m: fits[key][sd][m] / fits['E'][sd][m] for m in ('future_latent_nmse', 'mixed_loss')} for sd in seeds}
        rho[key] = {m: dict(r0=r[seeds[0]][m], r1=r[seeds[1]][m], rho=(1 - r[seeds[1]][m]) / (1 - r[seeds[0]][m]))
                    for m in ('future_latent_nmse', 'mixed_loss')}
    value = rho['D']['future_latent_nmse']['rho']
    attached = 'context' if value >= rule['context_rho_min'] else 'noise' if value <= rule['noise_rho_max'] else 'both'
    shared = {k: analysis['T'][k]['F'] >= rule['shared_min'] for k in ('seed', 'ctx')}
    return dict(rho=rho, attached=attached, shared=shared)


class VideoDone(Exception):
    """Raised by the patched video scheduler after the last video step: the action stage is not needed."""


def run_video_stage(sampler, runtime, datum, seed, replace=None, grad=False, steps=10):
    """The IDM sampler's video stage (sampler.differentiable_action's inputs and patches), stopped after its last step.
    Per denoiser call: the state it read (replace[i] instead of the loop's own state when given: teacher forcing), its
    timestep and its velocity (connected when grad, through non-reentrant checkpoints); per scheduler step: delta and the
    stepped state before the first-frame clamp. The future is the last stepped state with frame 0 set to the observation,
    as the sampler returns it (the loop clamps the very tensor the sampler holds, in place). The scheduler steps on
    detached velocities: the loop's own states never carry a graph."""
    from torch.utils.checkpoint import checkpoint
    sampler._require_compute_allocation()
    model = runtime.model
    query = sampler._regular_inputs(torch, datum)
    kwargs = dict(prompt=runtime.ev.DEFAULT_PROMPT.format(task=runtime.description), input_image=query['image'],
                  proprio=query['proprio'], action_horizon=32, num_video_frames=9, num_inference_steps=steps,
                  sigma_shift=1.0, seed=int(seed), rand_device='cpu', tiled=False, compile_action_infer=False)
    rec, patches = dict(x=[], t=[], v=[], delta=[], out=[]), []

    def patch(obj, name, value):
        namespace = getattr(obj, '__dict__', {})
        patches.append((obj, name, name in namespace, namespace.get(name)))
        setattr(obj, name, value)

    original_video, original_step = model._denoise_video, model.infer_video_scheduler.step
    contexts = lambda: (nullcontext(), nullcontext())  # noqa: E731

    def denoise_video(*, latents_video, timestep_video, context, context_mask, video_self_attn_mask,
                      fuse_vae_embedding_in_latents):
        i = len(rec['v'])
        x = latents_video if replace is None else replace[i].to(latents_video.device, latents_video.dtype)

        def run(x_, t_, c_, m_):
            return original_video(latents_video=x_, timestep_video=t_, context=c_, context_mask=m_,
                                  video_self_attn_mask=video_self_attn_mask,
                                  fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents)
        v = (checkpoint(run, x, timestep_video, context, context_mask, use_reentrant=False, context_fn=contexts)
             if grad else run(x, timestep_video, context, context_mask))
        rec['x'].append(x.detach().clone())
        rec['t'].append(timestep_video.detach().clone())
        rec['v'].append(v)
        return v

    def step(model_output, delta, sample):
        out = original_step(model_output.detach(), delta, sample)
        rec['delta'].append(float(delta))
        rec['out'].append(out.detach().clone())
        if len(rec['out']) == steps:
            raise VideoDone
        return out

    patch(model, '_denoise_video', denoise_video)
    patch(model.infer_video_scheduler, 'step', step)
    if getattr(model, 'proprio_encoder', None) is not None:  # as the sampler: the proprio Linear must stay on the graph
        bound = getattr(model, '_append_proprio_to_context', None)
        function = getattr(bound, '__func__', bound)
        if function is not None and hasattr(function, '__wrapped__'):
            patch(model, '_append_proprio_to_context', MethodType(inspect.unwrap(function), model))
    try:
        raw = sampler._raw_idm_infer(model)
        with torch.inference_mode(False), (torch.enable_grad() if grad else torch.no_grad()):
            try:
                raw(model, **kwargs)
            except VideoDone:
                pass
            else:
                raise RuntimeError('The video stage did not stop after its last step')
    finally:
        for obj, name, existed, previous in reversed(patches):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
    if len(rec['v']) != steps or len(rec['out']) != steps:
        raise RuntimeError(f"Expected {steps} video steps, got {len(rec['v'])} calls and {len(rec['out'])} steps")
    rec['future'] = torch.cat((rec['x'][0][:, :, :1], rec['out'][-1][:, :, 1:]), dim=2)
    return rec


def tf_future(future, v_s, v_t, deltas):
    """x~ = future + sum_i delta_i (v_s,i - v_t,i) on frames 1: (frame 0 is the observation), in FP32."""
    d = sum(float(dl) * (vs.detach().float() - vt.to(vs.device).float())[:, :, 1:]
            for vs, vt, dl in zip(v_s, v_t, deltas))
    base = future.to(d.device, torch.float32)
    return torch.cat((base[:, :, :1], base[:, :, 1:] + d), dim=2)


def tf_analysis(gram, norms, names, samples, batches, eta):
    """gram[t]: Gram matrix of tensor t's per-sample gradients stacked [tf_1..tf_S, ro_1..ro_S] (the teacher-forced
    objective and diag's rollout F path) over samples (case, seed, suite); norms[t] = |P_t| at E. Returns
      energy, cosines  per group and path, as diag_analysis
      cross            per group, the mean cosine between a sample's tf and ro gradients (self) and between the tf and ro
                       gradients of the pair categories (seed / suite / other, both orders)
      snr              per path and batch, |mean g|^2 / mean |g|^2 over the batch's seed-0 samples
      batches          per batch and step source (tf, ro: D's relative step along the mean of the batch's seed-0
                       gradients of that path): unit = sum_t |P_t| |gbar_t| and, per category (in / seed / ctx /
                       ctx_seed as diag), the first-order decrease of both paths' losses
      T                per source, its own loss's decrease per category over that on the batch, summed over batches
      G                per source, the ro loss's decrease per category over the ro step's on its own batch"""
    gram = gram.double()
    s = len(samples)
    off = dict(tf=0, ro=s)
    group_of = [tensor_group(n) for n in names]
    index = {(c, sd): i for i, (c, sd, _) in enumerate(samples)}
    pairs = {}
    for i in range(s):
        for j in range(i + 1, s):
            (ci, _, ui), (cj, _, uj) = samples[i], samples[j]
            pairs.setdefault('seed' if ci == cj else 'suite' if ui == uj else 'other', []).append((i, j))
    pairs = {k: torch.tensor(v) for k, v in pairs.items()}
    sums = {grp: torch.zeros(2 * s, 2 * s, dtype=torch.float64) for grp in GROUPS + ('all',)}
    for t, grp in enumerate(group_of):
        sums[grp] += gram[t]
        sums['all'] += gram[t]
    whole = {p: sums['all'][o:o + s, o:o + s] for p, o in off.items()}
    energy, cosines, cross = {}, {}, {}
    for grp, g in sums.items():
        for p, o in off.items():
            d = g[o:o + s, o:o + s].diagonal()
            energy[f'{grp}/{p}'] = float(d.mean() / whole[p].diagonal().mean().clamp_min(1e-300))
            if float(d.min()) <= 0:
                continue
            c = g[o:o + s, o:o + s] / d.sqrt()[:, None] / d.sqrt()[None, :]
            cosines[f'{grp}/{p}'] = {k: float(c[v[:, 0], v[:, 1]].mean()) for k, v in pairs.items()}
        dt, dr = g[:s, :s].diagonal(), g[s:, s:].diagonal()
        if float(dt.min()) <= 0 or float(dr.min()) <= 0:
            continue
        c = g[:s, s:] / dt.sqrt()[:, None] / dr.sqrt()[None, :]
        cross[grp] = dict(self=float(c.diagonal().mean()),
                          **{k: float((c[v[:, 0], v[:, 1]].mean() + c[v[:, 1], v[:, 0]].mean()) / 2) for k, v in pairs.items()})
    snr = {p: [] for p in off}
    for b in batches:
        idx = [index[(c, 0)] for c in b]
        for p, m in whole.items():
            sub = m[idx][:, idx]
            snr[p].append(float(sub.mean() / sub.diagonal().mean()))
    out_batches = []
    for b in batches:
        idx = [index[(c, 0)] for c in b]
        cats = {}
        for k, (c, sd, _) in enumerate(samples):
            cats.setdefault(('in' if sd == 0 else 'seed') if c in b else ('ctx' if sd == 0 else 'ctx_seed'), []).append(k)
        row = dict(cases=list(b))
        for src, o in off.items():
            cols = [i + o for i in idx]
            dec = {p: torch.zeros(s, dtype=torch.float64) for p in off}
            unit = 0.0
            for t in range(len(names)):
                g = gram[t]
                gg = float(g[cols][:, cols].mean())  # |gbar_t|^2
                if gg <= 0:
                    continue
                unit += float(norms[t]) * gg ** 0.5
                scale = eta * float(norms[t]) / gg ** 0.5
                for p, q in off.items():
                    dec[p] += scale * g[q:q + s][:, cols].mean(1)  # <g_p,k, gbar_t>
            row[src] = dict(unit=unit, **{key: {p: float(v[ks].mean()) for p, v in dec.items()} for key, ks in cats.items()})
        out_batches.append(row)
    keys = [k for k in ('seed', 'ctx', 'ctx_seed') if all(k in r['tf'] for r in out_batches)]
    T = {src: {k: sum(r[src][k][src] for r in out_batches) / sum(r[src]['in'][src] for r in out_batches) for k in keys}
         for src in off}
    ro_in = sum(r['ro']['in']['ro'] for r in out_batches)
    G = {src: {k: sum(r[src][k]['ro'] for r in out_batches) / ro_in for k in ['in'] + keys} for src in off}
    return dict(energy=energy, cosines=cosines, cross=cross, pair_counts={k: int(v.shape[0]) for k, v in pairs.items()},
                snr=snr, batches=out_batches, T=T, G=G)


def tf_reading(analysis, rule):
    t, g = analysis['T']['tf']['ctx_seed'], analysis['G']['tf']['ctx_seed']
    shared, reaches = t >= rule['shared_min'], g >= rule['rollout_fresh_min']
    return dict(T_tf_ctx_seed=t, G_tf_ctx_seed=g, G_ro_ctx_seed=analysis['G']['ro']['ctx_seed'], shared=shared,
                reaches=reaches, go=shared and reaches)


class LowRankWork(tw.TuneWork):
    def lr_dir(self, arm=None):
        return self.base / 'lowrank' / arm if arm else self.base / 'lowrank'

    # ---------------- branch ----------------
    def attach(self):
        """FP32 factors per target Linear and a forward that adds the branch while self.active (the Bank's plain
        forward otherwise)."""
        self.active, self.U, self.V, rank = False, {}, {}, LR['rank']
        for name, mod in self.bank.modules.items():
            self.U[name] = torch.zeros(mod.out_features, rank, device=DEVICE, requires_grad=True)
            self.V[name] = torch.zeros(rank, mod.in_features, device=DEVICE, requires_grad=True)
            mod.forward = self._branched(name, mod, mod.forward)

    def _branched(self, name, mod, plain):
        def run(x):
            if not self.active:
                return plain(x)
            self.bank.seen.add(name)  # the sampler checks that every target Linear ran
            return F.linear(x, mod.weight, mod.bias) + LowRank.apply(x, self.U[name], self.V[name])
        return run

    def parameters(self):
        return [self.U[n] for n in self.bank.names] + [self.V[n] for n in self.bank.names]

    def snapshot(self):
        return {n: dict(U=self.U[n].detach().cpu().clone(), V=self.V[n].detach().cpu().clone()) for n in self.bank.names}

    def restore(self, saved):
        with torch.no_grad():
            for n in self.bank.names:
                self.U[n].copy_(saved[n]['U'].to(DEVICE, torch.float32))
                self.V[n].copy_(saved[n]['V'].to(DEVICE, torch.float32))

    def save_factors(self, arm):
        """The deployed BF16 factors."""
        path = self.lr_dir(arm) / 'factors.pt'
        if not path.exists():
            self.tsave(path, {n: {k: v.to(torch.bfloat16) for k, v in f.items()} for n, f in self.snapshot().items()})
        return path

    def install(self, arm):
        """A low-rank arm: the host's W2 weights plus its BF16 factors, branch on; a time-conditioning arm (tcond): its
        base arm, then the BF16 masters in the Linears of its groups (their branch factors zeroed); any other arm: plain
        install."""
        if arm in TC_ARMS:
            c = TC_ARMS[arm]
            self.install(c['base'])
            with torch.no_grad():
                for name in self.tcond_modules(c['groups']):
                    self.bank.modules[name].weight.data.copy_(self.masters[name].to(DEVICE))
                    self.U[name].zero_()
                    self.V[name].zero_()
            return
        if arm not in ARMS.values():
            self.active = False
            return super().install(arm)
        super().install(BASELINE_ARM)
        self.restore(torch.load(self.lr_dir(arm) / 'factors.pt', map_location='cpu', weights_only=True))
        self.active = True

    # ---------------- init ----------------
    def init_factors(self):
        path = self.lr_dir(INIT_DIR) / 'factors.pt'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True)
        prep, base = self.baseline / 'prep', self.arm_root(BASELINE_ARM)
        factors, stats = {}, []
        with torch.no_grad():
            for i, name in enumerate(self.bank.names):
                self.enough()
                saved = {k: torch.load(p / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                         for k, p in (('U', base / 'weights'), ('master', prep / 'masters'), ('H_U', prep / 'hessians'))}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                w, q = (saved[k]['weight'].to(DEVICE, torch.float32) for k in ('master', 'U'))
                h = saved['H_U']['hessian'].to(DEVICE, torch.float32)
                u, v, info = eora(w - q, h, LR['rank'], LR['init']['damping'])
                factors[name] = dict(U=u.cpu(), V=v.cpu())
                stats.append(dict(index=i, module=name, shape=list(q.shape), **info))
                if i % 50 == 0:
                    self.progress('init', layers=i + 1)
                del saved, w, q, h, u, v
        self.tsave(path, factors)
        wk.save(self.lr_dir(INIT_DIR) / 'layers.json', stats)
        return factors

    def complete(self, arm):
        """Bits (host + 16 r (out + in) per Linear) and the proxy of W_host + U V against the master under H_U."""
        path = self.lr_dir(arm) / 'COMPLETE.json'
        if path.exists():
            return wk.read(path)
        prep, base = self.baseline / 'prep', self.arm_root(BASELINE_ARM)
        factors = torch.load(self.lr_dir(arm) / 'factors.pt', map_location='cpu', weights_only=True)
        stats = []
        with torch.no_grad():
            for i, name in enumerate(self.bank.names):
                self.enough()
                saved = {k: torch.load(p / f'{i:04d}.pt', map_location='cpu', weights_only=True)
                         for k, p in (('U', base / 'weights'), ('master', prep / 'masters'), ('H_U', prep / 'hessians'))}
                if any(v['module'] != name for v in saved.values()):
                    raise RuntimeError(f'Layer identity differs at {i}: {name}')
                w, q = (saved[k]['weight'].to(DEVICE, torch.float32) for k in ('master', 'U'))
                h = saved['H_U']['hessian'].to(DEVICE, torch.float32)
                lo = factors[name]['U'].to(DEVICE, torch.float32) @ factors[name]['V'].to(DEVICE, torch.float32)
                ref = energy(w, h)
                shape = list(q.shape)
                bits = wk.read(base / 'stats' / f'{i:04d}.json')['bits_total'] + 16 * LR['rank'] * (shape[0] + shape[1])
                stats.append(dict(index=i, module=name, shape=shape, bits_total=bits,
                                  proxy_U=energy(q + lo - w, h) / max(ref, 1e-30), baseline_proxy_U=energy(q - w, h) / max(ref, 1e-30),
                                  branch_rel=float(lo.norm() / (q - w).norm().clamp_min(1e-30))))
                del saved, w, q, h, lo
        n = sum(s['shape'][0] * s['shape'][1] for s in stats)

        def weighted(key):
            return sum(s[key] * s['shape'][0] * s['shape'][1] for s in stats) / n
        wk.save(self.lr_dir(arm) / 'layers.json', stats)
        out = dict(protocol=wk.PROTOCOL, complete=True, pbs_jobid=self.job, arm=arm, method=METHOD, bits=2, hessian='U',
                   rank=LR['rank'], target_weights=n, bpw=sum(s['bits_total'] for s in stats) / n,
                   proxy_nmse_weighted=weighted('proxy_U'), baseline_proxy_U_weighted=weighted('baseline_proxy_U'))
        wk.save(path, out)
        return out

    def fit_arm(self, arm, seed=0, path=None):
        """C80 mixed loss (A + lambda Z) and its parts with the arm installed, at sampler seed index `seed`."""
        path = path or self.lr_dir(arm) / 'fit.json'
        if path.exists():
            return wk.read(path)
        self.install(arm)
        rows = []
        for case in lp.CASES:
            self.enough()
            with torch.no_grad():
                action, future, _ = self.query(case, seed)
            m = self.metrics(action, future, self.teacher(case, seed))
            rows.append(dict(case_id=case, **{k: m[k] for k in ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse')}))
            self.progress('fit', arm=arm, seed=seed, contexts=len(rows))
        out = dict(protocol=wk.PROTOCOL, arm=arm, seed_index=seed, rows=rows,
                   **{k: sum(r[k] for r in rows) / len(rows) for k in ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse')})
        wk.save(path, out)
        return out

    def init(self):
        arm = ARMS['E']
        if (self.lr_dir(arm) / 'arm.json').exists():
            return
        factors = self.init_factors()
        if not (self.lr_dir(arm) / 'factors.pt').exists():
            self.tsave(self.lr_dir(arm) / 'factors.pt',
                       {n: {k: v.to(torch.bfloat16) for k, v in f.items()} for n, f in factors.items()})
        self.complete(arm)
        self.fit_arm(arm)
        self.evaluate(arm)
        self.active = False
        wk.save(self.lr_dir(arm) / 'arm.json', dict(protocol=wk.PROTOCOL, arm=arm, tuned=False, pbs_jobid=self.job))

    # ---------------- objectives ----------------
    def batch_for(self, key):
        return self.batch if key == 'J' else self.batch_decoupled

    def batch_decoupled(self, batch, grad):
        """mean over the batch of A_tf + lambda Z + beta K: A_tf on the student's action reading the teacher future,
        Z and K on the student's own future; with grad, accumulated into the factors' .grad by a staged replay of the
        student query (future cotangent only) and of the teacher-future read (action cotangent)."""
        from reader_loss import reader_terms
        beta, stats = self.cfg['loss']['beta'], {'action': self.p['reader_source']['action_stats']}
        staged = []
        for case in batch:
            self.enough()
            target = self.teacher(case, 0)
            value, description = self.datum(case)
            seed = self.rows[case]['sampler_seeds'][0]
            with torch.no_grad():
                _, future, _ = self.query(case, 0)
                self.runtime.description = description
                action, _ = self.reader(self.runtime, value, seed, target['future'].to(DEVICE, torch.float32),
                                        checkpointed=False)
            self.count('reader_queries')
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
                case = row['case']
                _, future, _ = self.query(case, 0, grad=True)
                if not future.requires_grad:
                    raise RuntimeError('Tuned query lost its graph')
                worst = max(worst, wk.max_abs(future, row['future']))
                torch.autograd.backward(future, (row['gz'] / n).to(future))
                del future
                value, description = self.datum(case)
                self.runtime.description = description
                # a leaf future keeps the read's graph whatever checkpointing the reader uses; its gradient is unused
                given = self.teacher(case, 0)['future'].to(DEVICE, torch.float32).requires_grad_(True)
                action, _ = self.reader(self.runtime, value, self.rows[case]['sampler_seeds'][0], given, checkpointed=True)
                self.count('reader_queries')
                if not action.requires_grad:
                    raise RuntimeError('Teacher-future read lost its graph')
                worst = max(worst, wk.max_abs(action, row['action']))
                torch.autograd.backward(action, (row['ga'] / n).to(action))
                self.counters['parameter_reverse_passes'] += 2
                del action, given
            if worst > LR['training']['replay_max_abs']:
                raise RuntimeError(f'Grad replay differs from the staged no-grad pass: {worst}')
            out['replay_max_abs'] = worst
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    # ---------------- training ----------------
    def step_check(self, key):
        """Per grid eta, one relative step P - eta |P| g / |g| from E on batch 1: predicted decrease eta sum |P| |g|
        against the realized one on the same batch; also where |g|^2 sits."""
        arm = ARMS[key]
        path = self.lr_dir(arm) / 'step.json'
        if path.exists():
            return wk.read(path)
        cfg, params, batch = LR['training'], self.parameters(), self.schedule[0]
        for p in params:
            p.grad = None
        self.active = True
        base = self.batch_for(key)(batch, grad=True)
        start = [p.detach().clone() for p in params]
        g = [p.grad.detach().clone() for p in params]
        unit = sum(float(x.norm()) * float(y.norm()) for x, y in zip(start, g))
        g2 = sum(float(y.square().sum()) for y in g)
        n = len(self.bank.names)
        layers = sorted(((float(g[i].square().sum() + g[n + i].square().sum()), self.bank.names[i]) for i in range(n)),
                        reverse=True)
        rows = []
        for eta in cfg['step_grid']:
            with torch.no_grad():
                for p, x in zip(params, start):
                    p.copy_(x)
            relative_step(params, g, eta)
            after = self.batch_for(key)(batch, grad=False)
            rows.append(dict(lr=eta, predicted=eta * unit, realized=base['total'] - after['total'], loss=after))
            self.progress('step', arm=arm, eta=eta)
        with torch.no_grad():
            for p, x in zip(params, start):
                p.copy_(x)
                p.grad = None
        self.active = False
        out = dict(protocol=wk.PROTOCOL, arm=arm, pbs_jobid=self.job, batch=batch, base=base, predicted_per_eta=unit,
                   top_layers=[dict(module=name, share=v / max(g2, 1e-30)) for v, name in layers[:10]], rows=rows,
                   chosen=tw.first_step_choice(rows, cfg['min_realized_fraction']))
        wk.save(path, out)
        return out

    def train(self, key, eta):
        """8 relative steps on the loss-pair C80 batches; checkpointed after each."""
        arm, cfg = ARMS[key], LR['training']
        root, params = self.lr_dir(arm), self.parameters()
        state = dict(updates=0, history=[])
        if (root / 'checkpoint.pt').exists():
            saved = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=True)
            if saved['eta'] != eta:
                raise RuntimeError('Checkpoint step differs')
            self.restore(saved['factors'])
            state = saved['state']
        self.active = True
        while state['updates'] < cfg['updates']:
            self.enough()
            u, begin = state['updates'], time.monotonic()
            for p in params:
                p.grad = None
            before = self.batch_for(key)(self.schedule[u], grad=True)
            relative_step(params, [p.grad for p in params], eta)
            state['updates'] += 1
            state['history'].append(dict(update=u + 1, cases=self.schedule[u], before=before,
                                         seconds=time.monotonic() - begin))
            self.tsave(root / 'checkpoint.pt', dict(eta=eta, factors=self.snapshot(), state=state))
            wk.save(root / 'training.json', dict(protocol=wk.PROTOCOL, arm=arm, eta=eta, **state))
            self.progress('train', arm=arm, updates=state['updates'], loss=before['total'])
        self.active = False
        return state

    def run_arm(self, key):
        arm = ARMS[key]
        if (self.lr_dir(arm) / 'arm.json').exists():
            return
        super().install(BASELINE_ARM)
        self.restore(torch.load(self.lr_dir(ARMS['E']) / 'factors.pt', map_location='cpu', weights_only=True))
        eta = self.step_check(key)['chosen']
        if eta is None:
            wk.save(self.lr_dir(arm) / 'arm.json', dict(protocol=wk.PROTOCOL, arm=arm, tuned=False, pbs_jobid=self.job))
            return
        self.train(key, eta)
        self.save_factors(arm)
        self.complete(arm)
        self.fit_arm(arm)
        self.evaluate(arm)
        self.active = False
        wk.save(self.lr_dir(arm) / 'arm.json', dict(protocol=wk.PROTOCOL, arm=arm, tuned=True, eta=eta, pbs_jobid=self.job))

    # ---------------- transfer diagnosis (lowrank.json diag) ----------------
    def teacher(self, case, seed=0):
        """C80 at a sampler seed other than 0: the BF16 query written by diag_teachers (the cache holds seed 0 only)."""
        if case in lp.CASES and seed != 0:
            t = torch.load(self.lr_dir('diag') / 'teachers' / f'{case}_{seed}.pt', map_location='cpu', weights_only=True)
            if t['case_id'] != case or t['sampler_seed'] != self.rows[case]['sampler_seeds'][seed]:
                raise RuntimeError(f'Teacher identity differs: {case}, {seed}')
            return t
        return super().teacher(case, seed)

    def diag_teachers(self):
        """BF16 teachers of C80 at the diag seed, before any install (the modules hold the masters); first the cached
        seed-0 teachers of the first cases regenerated the same way."""
        cfg, root, seed = LR['diag'], self.lr_dir('diag'), LR['diag']['seed']
        self.active = False
        path = root / 'teacher_check.json'
        if not path.exists():
            rows = []
            for case in lp.CASES[:cfg['teacher_check']['cases']]:
                self.enough()
                with torch.no_grad():
                    action, future, _ = self.query(case, 0)
                cached = super().teacher(case, 0)
                ref = cached['future'].float()
                rows.append(dict(case_id=case, future_nmse=float((future.float().cpu() - ref).square().sum() / ref.square().sum()),
                                 action_max_abs=wk.max_abs(action, cached['action'])))
            wk.save(path, dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, rows=rows,
                               worst=max(r['future_nmse'] for r in rows), limit=cfg['teacher_check']['future_nmse_max']))
        check = wk.read(path)
        if check['worst'] > check['limit']:
            raise RuntimeError(f"Regenerated seed-0 teacher differs from the cache: future NMSE {check['worst']}")
        for case in lp.CASES:
            out = root / 'teachers' / f'{case}_{seed}.pt'
            if out.exists():
                continue
            self.enough()
            with torch.no_grad():
                action, future, _ = self.query(case, seed)
            self.tsave(out, dict(case_id=case, seed_index=seed, sampler_seed=self.rows[case]['sampler_seeds'][seed],
                                 action=action.detach().cpu(), future=future.detach().cpu()))
            self.progress('diag_teacher', case=case)

    def sample_grads(self, case, seed, paths=('F', 'A')):
        """D's objective on one (context, seed) at the current factors and its gradient split by path, each a list over
        self.parameters() on the CPU: F = lambda Z + beta K through the student query (the future), A = A_tf through the
        student reading the teacher future (the action). Same staging as batch_decoupled; F + A is its gradient.
        Returns ({path: gradient}, stats, the student future); without A the teacher action stands in for A_tf
        (stats az = lambda Z)."""
        from reader_loss import reader_terms
        beta, stats = self.cfg['loss']['beta'], {'action': self.p['reader_source']['action_stats']}
        params, target = self.parameters(), self.teacher(case, seed)
        value, description = self.datum(case)
        sampler = self.rows[case]['sampler_seeds'][seed]
        self.enough()
        with torch.no_grad():
            _, future, _ = self.query(case, seed)
            if 'A' in paths:
                self.runtime.description = description
                action, _ = self.reader(self.runtime, value, sampler, target['future'].to(DEVICE, torch.float32),
                                        checkpointed=False)
                self.count('reader_queries')
            else:
                action = target['action'].to(future.device)
        a_leaf = action.detach().float().clone().requires_grad_(True)
        z_leaf = future.detach().float().clone().requires_grad_(True)
        total, _, _ = self.loss(a_leaf, z_leaf, target)
        ga, gz = torch.autograd.grad(total, (a_leaf, z_leaf))
        with self.bf16_reader() as reader:
            self.runtime.description = description
            leaf = future.detach().float().clone().requires_grad_(True)
            read, _ = reader(self.runtime, value, sampler, leaf, checkpointed=True)
            k = reader_terms(torch, read, target['action'], stats, self.p['reader_source']['config'])['reader_kl']
            rg, = torch.autograd.grad(beta * k, leaf)
        self.count('reader_queries')
        gz = gz + rg
        grads, worst = {}, 0.0
        for path in paths:
            for p in params:
                p.grad = None
            if path == 'F':
                _, out, _ = self.query(case, seed, grad=True)
                reference, cotangent = future, gz
            else:
                self.runtime.description = description
                given = target['future'].to(DEVICE, torch.float32).requires_grad_(True)
                out, _ = self.reader(self.runtime, value, sampler, given, checkpointed=True)
                self.count('reader_queries')
                reference, cotangent = action, ga
            if not out.requires_grad:
                raise RuntimeError(f'Replay of path {path} lost its graph')
            worst = max(worst, wk.max_abs(out, reference))
            torch.autograd.backward(out, cotangent.to(out))
            self.counters['parameter_reverse_passes'] += 1
            grads[path] = [p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros(p.shape) for p in params]
            del out
        for p in params:
            p.grad = None
        if worst > LR['training']['replay_max_abs']:
            raise RuntimeError(f'Grad replay differs from the staged no-grad pass: {worst}')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return grads, dict(az=float(total.detach()), k=float(k.detach()), replay_max_abs=worst), future.detach()

    def diag_grads(self):
        """Per-sample gradients at E (contexts of the first training batches, both seeds) -> per-tensor Gram matrices of
        the stacked [F_1..F_S, A_1..A_S] and E's factor norms (the gradients themselves are not kept)."""
        cfg, path = LR['diag']['grad'], self.lr_dir('diag') / 'gram.pt'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True)
        super().install(LR['baseline_arm'])
        self.restore(torch.load(self.lr_dir(ARMS[cfg['arm']]) / 'factors.pt', map_location='cpu', weights_only=True))
        self.active = True
        params = self.parameters()
        cases = [c for b in self.schedule[:cfg['batches']] for c in b]
        samples = [(c, sd, self.rows[c]['suite']) for sd in cfg['seeds'] for c in cases]
        fs, As, stats = [], [], []
        for case, seed, _ in samples:
            g, st, _ = self.sample_grads(case, seed)
            fs.append(g['F'])
            As.append(g['A'])
            stats.append(dict(case_id=case, seed_index=seed, **st))
            self.progress('diag_grad', samples=len(stats))
        self.active = False
        gram = torch.empty(len(params), 2 * len(samples), 2 * len(samples), dtype=torch.float64)
        for t in range(len(params)):
            x = torch.stack([g[t].reshape(-1) for g in fs + As]).to(DEVICE, torch.float64)
            gram[t] = (x @ x.t()).cpu()
            for g in fs + As:
                g[t] = None
            del x
        names = [f'{n}.{k}' for k in ('U', 'V') for n in self.bank.names]  # self.parameters() order
        out = dict(names=names, samples=[list(x) for x in samples], gram=gram, stats=stats,
                   norms=torch.tensor([float(p.detach().norm()) for p in params], dtype=torch.float64))
        self.tsave(path, out)
        return out

    def diag(self):
        cfg, root = LR['diag'], self.lr_dir('diag')
        if (root / 'summary.json').exists():
            return
        self.diag_teachers()
        fits = {}
        for key in cfg['fit_arms']:
            arm = ARMS[key]
            fits[key] = {0: wk.read(self.lr_dir(arm) / 'fit.json'),
                         cfg['seed']: self.fit_arm(arm, cfg['seed'], root / f"fit_{arm}_s{cfg['seed']}.json")}
        g = self.diag_grads()
        analysis = diag_analysis(g['gram'], g['norms'], g['names'], [tuple(x) for x in g['samples']],
                                 self.schedule[:cfg['grad']['batches']], cfg['grad']['eta'])
        step = wk.read(self.lr_dir(ARMS[cfg['grad']['objective']]) / 'step.json')
        check = dict(unit=analysis['batches'][0]['unit'], step_json=step['predicted_per_eta'])
        check['rel'] = check['unit'] / check['step_json'] - 1
        check['passed'] = abs(check['rel']) <= cfg['grad']['predicted_check_rel']
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, diag=cfg, teacher_check=wk.read(root / 'teacher_check.json'),
                       fits={k: {sd: {m: v[m] for m in ('mixed_loss', 'first10motor_rmse', 'future_latent_nmse')}
                                 for sd, v in f.items()} for k, f in fits.items()},
                       by_suite=fit_suites(fits, self.rows), analysis=analysis, predicted_check=check,
                       grad_stats=g['stats'], reading=diag_reading(fits, analysis, cfg['reading']))
        wk.save(root / 'summary.json', summary)
        (root / 'RESULTS.zh.md').write_text(diag_report(summary), encoding='utf-8')
        if not check['passed']:
            raise RuntimeError(f"Batch-1 first-order decrease does not reproduce step.json: {check}")

    # ---------------- teacher-forced objective (lowrank.json tfdiag) ----------------
    def video_stage(self, case, seed, states=None, grad=False):
        value, description = self.datum(case)
        self.runtime.description = description
        self.count('student_queries')
        return run_video_stage(self.sampler, self.runtime, value, self.rows[case]['sampler_seeds'][seed],
                               replace=None if states is None else states['x'], grad=grad)

    def tf_samples(self):
        cfg = LR['diag']['grad']
        cases = [c for b in self.schedule[:cfg['batches']] for c in b]
        return [(c, sd, self.rows[c]['suite']) for sd in cfg['seeds'] for c in cases]

    def load_state(self, case, seed):
        s = torch.load(self.lr_dir('tfdiag') / 'states' / f'{case}_{seed}.pt', map_location='cpu', weights_only=True)
        if s['case_id'] != case or s['sampler_seed'] != self.rows[case]['sampler_seeds'][seed]:
            raise RuntimeError(f'Teacher state identity differs: {case}, {seed}')
        return s

    def tf_states(self, samples):
        """The BF16 teacher's video stage per sample, before any install (the modules hold the masters): the last
        stepped state must reproduce the teacher future and each recorded state the clamped step before it; then
        teacher forcing with the masters must return the recorded velocities."""
        lim, root = LR['tfdiag']['checks'], self.lr_dir('tfdiag')
        path = root / 'state_check.json'
        if not path.exists():
            self.active = False
            rows = []
            for case, seed, _ in samples:
                out = root / 'states' / f'{case}_{seed}.pt'
                if not out.exists():
                    self.enough()
                    with torch.no_grad():
                        loop = self.video_stage(case, seed)
                    chain = max(wk.max_abs(loop['x'][i + 1][:, :, 1:], loop['out'][i][:, :, 1:])
                                for i in range(len(loop['out']) - 1))
                    self.tsave(out, dict(case_id=case, seed_index=seed, sampler_seed=self.rows[case]['sampler_seeds'][seed],
                                         x=[x.cpu() for x in loop['x']], v=[v.detach().cpu() for v in loop['v']],
                                         t=[t.cpu() for t in loop['t']], delta=loop['delta'],
                                         future=loop['future'].cpu(), chain_max_abs=chain))
                    self.progress('tf_state', samples=len(rows) + 1)
                s = self.load_state(case, seed)
                full = self.teacher(case, seed)['future']
                ref = full.float()[:, :, 1:]
                nmse = float((s['future'].float()[:, :, 1:] - ref).square().sum() / ref.square().sum())
                rows.append(dict(case_id=case, seed_index=seed, future_nmse=nmse, chain_max_abs=s['chain_max_abs'],
                                 frame0_max_abs=wk.max_abs(s['future'][:, :, :1], full[:, :, :1])))
            case, seed, _ = samples[0]
            s = self.load_state(case, seed)
            with torch.no_grad():
                loop = self.video_stage(case, seed, s)
            wk.save(path, dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, rows=rows,
                               future_nmse_worst=max(r['future_nmse'] for r in rows),
                               chain_worst=max(r['chain_max_abs'] for r in rows),
                               frame0_worst=max(r['frame0_max_abs'] for r in rows),
                               bf16_tf_max_abs=max(wk.max_abs(a, b) for a, b in zip(loop['v'], s['v'])), limits=lim))
        check = wk.read(path)
        if (check['future_nmse_worst'] > lim['state_future_nmse_max'] or check['chain_worst'] > lim['state_chain_max_abs']
                or check['frame0_worst'] > lim['state_chain_max_abs'] or check['bf16_tf_max_abs'] > lim['bf16_tf_max_abs']):
            raise RuntimeError(f'Teacher states fail their checks: {check}')
        return check

    def tf_objective(self, case, seed, xt):
        """lambda Z(x~) + beta K(x~) and its gradient with respect to x~: Z the future NMSE against the teacher future,
        K the frozen BF16 reader at x~."""
        from reader_loss import reader_terms
        lam, beta = self.cfg['loss']['lambda'], self.cfg['loss']['beta']
        stats, target = {'action': self.p['reader_source']['action_stats']}, self.teacher(case, seed)
        value, description = self.datum(case)
        leaf = xt.detach().to(DEVICE, torch.float32).requires_grad_(True)
        z = self.latent_reference.future_nmse_loss(torch, leaf, target['future'].to(DEVICE, torch.float32))
        gz, = torch.autograd.grad(lam * z, leaf)
        with self.bf16_reader() as reader:
            self.runtime.description = description
            leaf = xt.detach().to(DEVICE, torch.float32).requires_grad_(True)
            read, _ = reader(self.runtime, value, self.rows[case]['sampler_seeds'][seed], leaf, checkpointed=True)
            k = reader_terms(torch, read, target['action'], stats, self.p['reader_source']['config'])['reader_kl']
            gk, = torch.autograd.grad(beta * k, leaf)
        self.count('reader_queries')
        return dict(z=float(z), k=float(k), total=float(lam * z + beta * k)), gz + gk

    def tf_grads(self, case, seed, states):
        """The teacher-forced objective on one sample at the current factors and its gradient (a list over
        self.parameters() on the CPU): no-grad pass -> x~ and its cotangent g; replay of the 10 denoiser calls with
        grad (bitwise check), each velocity's cotangent delta_i g on frames 1:. Also the per-step displacements."""
        params = self.parameters()
        self.enough()
        with torch.no_grad():
            loop = self.video_stage(case, seed, states)
        xt = tf_future(states['future'], loop['v'], states['v'], states['delta'])
        values, gx = self.tf_objective(case, seed, xt)
        for p in params:
            p.grad = None
        replay = self.video_stage(case, seed, states, grad=True)
        if not all(v.requires_grad for v in replay['v']):
            raise RuntimeError('Teacher-forced replay lost its graph')
        worst = max(wk.max_abs(a, b) for a, b in zip(replay['v'], loop['v']))
        cotangents = []
        for v, dl in zip(replay['v'], states['delta']):
            c = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            c[:, :, 1:] = dl * gx[:, :, 1:].to(v.device)
            cotangents.append(c.to(v.dtype))
        torch.autograd.backward(replay['v'], cotangents)
        self.counters['parameter_reverse_passes'] += 1
        grads = [p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros(p.shape) for p in params]
        for p in params:
            p.grad = None
        if worst > LR['training']['replay_max_abs']:
            raise RuntimeError(f'Teacher-forced replay differs from the no-grad pass: {worst}')
        steps = [(dl * (vs.detach().float() - vt.to(vs.device).float())[:, :, 1:]).cpu()
                 for vs, vt, dl in zip(loop['v'], states['v'], states['delta'])]
        del loop, replay, cotangents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return grads, dict(values, replay_max_abs=worst), dict(xt=xt.cpu(), steps=steps)

    def tf_step_check(self, case, seed, states, grads, before):
        """Reported: the tf objective's realized decrease from a relative step along its own gradient on one sample,
        against the first-order prediction eta sum_t |P_t| |g_t|."""
        eta, params, start = LR['tfdiag']['checks']['step_check_eta'], self.parameters(), self.snapshot()
        predicted = eta * sum(float(p.detach().norm()) * float(g.norm()) for p, g in zip(params, grads))
        relative_step(params, [g.to(p.device) for p, g in zip(params, grads)], eta)
        with torch.no_grad():
            loop = self.video_stage(case, seed, states)
        after, _ = self.tf_objective(case, seed, tf_future(states['future'], loop['v'], states['v'], states['delta']))
        self.restore(start)
        realized = before['total'] - after['total']
        return dict(case_id=case, seed_index=seed, eta=eta, before=before['total'], after=after['total'],
                    predicted=predicted, realized=realized, ratio=realized / predicted if predicted > 0 else None)

    def tf_grams(self, samples):
        """At E: per sample the tf and ro gradients -> per-tensor Gram matrices of the stacked [tf_1..tf_S, ro_1..ro_S],
        E's factor norms, per-sample values (Z, K of both objectives; the tf displacement against the rollout one, per
        step and summed) and the step check on the first sample."""
        root = self.lr_dir('tfdiag')
        path = root / 'gram.pt'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True)
        super().install(LR['baseline_arm'])
        self.restore(torch.load(self.lr_dir(ARMS[LR['diag']['grad']['arm']]) / 'factors.pt', map_location='cpu',
                                weights_only=True))
        self.active = True
        params = self.parameters()
        lam = self.cfg['loss']['lambda']
        tfs, ros, rows, check = [], [], [], None
        for case, seed, _ in samples:
            states = self.load_state(case, seed)
            gt, tv, parts = self.tf_grads(case, seed, states)
            if check is None:
                check = self.tf_step_check(case, seed, states, gt, tv)
            g, st, future = self.sample_grads(case, seed, paths=('F',))
            tfs.append(gt)
            ros.append(g['F'])
            ref = states['future'].float()[:, :, 1:]
            d_ro = future.float().cpu()[:, :, 1:] - ref
            d_tf = parts['xt'][:, :, 1:] - ref
            norm2 = float(ref.square().sum())

            def cos(a, b):
                return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
            rows.append(dict(case_id=case, seed_index=seed, z_tf=tv['z'], k_tf=tv['k'], total_tf=tv['total'],
                             z_ro=float(d_ro.square().sum()) / norm2, k_ro=st['k'], total_ro=st['az'] + self.cfg['loss']['beta'] * st['k'],
                             cos_tf_ro=cos(d_tf, d_ro), norm_tf_ro=float(d_tf.norm() / d_ro.norm().clamp_min(1e-30)),
                             step_energy=[float(e.square().sum()) / norm2 for e in parts['steps']],
                             step_cos_ro=[cos(e, d_ro) for e in parts['steps']],
                             replay_max_abs=max(tv['replay_max_abs'], st['replay_max_abs'])))
            del parts, future
            self.progress('tf_grad', samples=len(rows))
        self.active = False
        gram = torch.empty(len(params), 2 * len(samples), 2 * len(samples), dtype=torch.float64)
        for t in range(len(params)):
            x = torch.stack([g[t].reshape(-1) for g in tfs + ros]).to(DEVICE, torch.float64)
            gram[t] = (x @ x.t()).cpu()
            for g in tfs + ros:
                g[t] = None
            del x
        names = [f'{n}.{k}' for k in ('U', 'V') for n in self.bank.names]  # self.parameters() order
        out = dict(names=names, samples=[list(x) for x in samples], gram=gram, rows=rows, step_check=check, lam=lam,
                   norms=torch.tensor([float(p.detach().norm()) for p in params], dtype=torch.float64))
        self.tsave(path, out)
        return out

    def tfdiag(self):
        cfg, root = LR['tfdiag'], self.lr_dir('tfdiag')
        if (root / 'summary.json').exists():
            return
        samples = self.tf_samples()
        state_check = self.tf_states(samples)
        g = self.tf_grams(samples)
        old = torch.load(self.lr_dir('diag') / 'gram.pt', map_location='cpu', weights_only=True)
        if [tuple(x) for x in old['samples']] != [tuple(x) for x in g['samples']]:
            raise RuntimeError('diag and tfdiag samples differ')
        s = len(samples)
        new_ro, old_f = g['gram'][:, s:, s:].double(), old['gram'][:, :s, :s].double()
        ro_check = dict(rel=float((new_ro - old_f).norm() / old_f.norm()), limit=cfg['checks']['ro_gram_rel_max'])
        ro_check['passed'] = ro_check['rel'] <= ro_check['limit']
        analysis = tf_analysis(g['gram'], g['norms'], g['names'], [tuple(x) for x in g['samples']],
                               self.schedule[:LR['diag']['grad']['batches']], cfg['eta'])
        rows = g['rows']

        def mean(key):
            return sum(r[key] for r in rows) / len(rows)
        steps = len(rows[0]['step_energy'])
        values = dict(z_tf=mean('z_tf'), z_ro=mean('z_ro'), k_tf=mean('k_tf'), k_ro=mean('k_ro'),
                      total_tf=mean('total_tf'), total_ro=mean('total_ro'), cos_tf_ro=mean('cos_tf_ro'),
                      norm_tf_ro=mean('norm_tf_ro'),
                      step_energy=[sum(r['step_energy'][i] for r in rows) / len(rows) for i in range(steps)],
                      step_cos_ro=[sum(r['step_cos_ro'][i] for r in rows) / len(rows) for i in range(steps)],
                      by_seed={sd: {k: sum(r[k] for r in rows if r['seed_index'] == sd) / sum(r['seed_index'] == sd for r in rows)
                                    for k in ('z_tf', 'z_ro', 'cos_tf_ro')} for sd in sorted({r['seed_index'] for r in rows})})
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, tfdiag=cfg, state_check=state_check, ro_check=ro_check,
                       step_check=g['step_check'], values=values, rows=rows, analysis=analysis,
                       reading=tf_reading(analysis, cfg['reading']))
        wk.save(root / 'summary.json', summary)
        (root / 'RESULTS.zh.md').write_text(tf_report(summary), encoding='utf-8')
        if not ro_check['passed']:
            raise RuntimeError(f'Recomputed rollout gradients differ from diag/gram.pt: {ro_check}')

    # ---------------- time-conditioning attribution (lowrank.json tcond) ----------------
    def tcond_modules(self, groups):
        names = [n for n in self.bank.names if tensor_group(n + '.U') in groups]
        if len(names) != LR['tcond']['modules_per_group'] * len(groups):
            raise RuntimeError(f'Time-conditioning Linears of {groups}: {names}')
        return names

    def tcond_inputs(self):
        """BF16, before any install: the inputs of the time-conditioning Linears per call for two samples; they must
        match call by call. Distinct input rows of each expert's first Linear = timesteps a table has to hold."""
        cfg, path = LR['tcond']['input_check'], self.lr_dir('tcond') / 'input_check.json'
        if path.exists():
            return wk.read(path)
        self.active = False
        names = self.tcond_modules(['video_time', 'action_time'])
        seen, originals = {}, {}
        for n in names:
            mod = self.bank.modules[n]
            originals[n] = mod.forward

            def record(x, n=n, plain=mod.forward, in_features=mod.in_features):
                seen.setdefault(n, []).append(x.detach().reshape(-1, in_features).float().clone())
                return plain(x)
            mod.forward = record
        calls = []
        try:
            for case, seed in cfg['samples']:
                seen.clear()
                with torch.no_grad():
                    self.query(case, seed)
                calls.append({n: list(v) for n, v in seen.items()})
        finally:
            for n, f in originals.items():
                self.bank.modules[n].forward = f
        a, b = calls
        rows = {}
        for n in names:
            same = (len(a.get(n, [])) == len(b.get(n, [])) > 0
                    and all(x.shape == y.shape for x, y in zip(a[n], b[n])))
            diff = max(float((x - y).abs().max()) for x, y in zip(a[n], b[n])) if same else 1e30  # call pattern differs
            distinct = int(torch.unique(torch.cat(a[n]), dim=0).shape[0])
            rows[n] = dict(calls=len(a[n]), max_abs=diff, distinct_rows=distinct,
                           out_features=self.bank.modules[n].out_features, in_features=self.bank.modules[n].in_features)
        out = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, samples=cfg['samples'], modules=rows,
                   worst=max(r['max_abs'] for r in rows.values()))
        wk.save(path, out)
        return out

    def tcond_bits(self, check):
        """bpw of each arm with the group Linears as BF16 weights and as per-timestep tables of their outputs."""
        e = wk.read(self.lr_dir(ARMS['E']) / 'COMPLETE.json')
        n_total, rank = e['target_weights'], LR['rank']
        base = {LR['baseline_arm']: wk.read(self.arm_root(LR['baseline_arm']) / 'COMPLETE.json')['bpw'], ARMS['E']: e['bpw']}
        index = {n: i for i, n in enumerate(self.bank.names)}
        out = {}
        for arm, c in TC_ARMS.items():
            names = self.tcond_modules(c['groups'])
            freed = 0
            for n in names:
                mod = self.bank.modules[n]
                host = wk.read(self.arm_root(LR['baseline_arm']) / 'stats' / f'{index[n]:04d}.json')['bits_total']
                freed += host + (16 * rank * (mod.out_features + mod.in_features) if c['base'] == ARMS['E'] else 0)
            weights = sum(16 * self.bank.modules[n].out_features * self.bank.modules[n].in_features for n in names)
            table = 0
            for grp in c['groups']:
                head = grp.split('_')[0] + '_expert'
                steps = check['modules'][f'{head}.time_embedding.0']['distinct_rows']
                table += 16 * steps * sum(self.bank.modules[f'{head}.{m}'].out_features
                                          for m in ('time_embedding.2', 'time_projection.1'))
            out[arm] = dict(base=c['base'], base_bpw=base[c['base']], bpw_weights=base[c['base']] + (weights - freed) / n_total,
                            bpw_table=base[c['base']] + (table - freed) / n_total, modules=names)
        return out

    def tcond(self):
        cfg, root = LR['tcond'], self.lr_dir('tcond')
        if (root / 'summary.json').exists():
            return
        check = self.tcond_inputs()
        if check['worst'] > 0:
            raise RuntimeError(f"Time-conditioning inputs differ across contexts: {check['worst']}")
        for arm in TC_ARMS:
            self.evaluate(arm)
        self.active = False
        held = {arm: wk.read(self.base / 'heldout' / f'{arm}.json')['overall']
                for arm in list(TC_ARMS) + [LR['baseline_arm'], ARMS['E']]}
        ratios = {arm: {k: held[arm][k] / held[c['base']][k] for k in ('future_latent_nmse', 'mixed_loss', 'first10motor_rmse',
                                                                       'first10gripper_rmse', 'first10motor_rmse_swap')}
                  for arm, c in TC_ARMS.items()}
        r = ratios[cfg['arms']['U_TV']['arm']]['future_latent_nmse']
        rule = cfg['reading']
        verdict = 'carries' if r <= rule['carries_max'] else 'negligible' if r > rule['negligible_min'] else 'partial'
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, tcond=cfg, input_check=check, bits=self.tcond_bits(check),
                       heldout=held, ratios=ratios, reading=dict(r=r, verdict=verdict))
        wk.save(root / 'summary.json', summary)
        (root / 'RESULTS.zh.md').write_text(tcond_report(summary), encoding='utf-8')

    # ---------------- report ----------------
    def gather(self):
        fit = wk.read(self.base / 'tune' / LR['host'] / 'fit_before.json')
        rows = dict(U=dict(tuned=False, arm=LR['baseline_arm'], fit={k: v for k, v in fit.items() if k != 'rows'},
                           heldout=wk.read(self.base / 'heldout' / f"{LR['baseline_arm']}.json")['overall']))
        for key, arm in ARMS.items():
            r = wk.read(self.lr_dir(arm) / 'arm.json')
            row = dict(tuned=r['tuned'], arm=arm, eta=r.get('eta'))
            if key == 'E' or r['tuned']:
                row.update(fit=wk.read(self.lr_dir(arm) / 'fit.json'), quant=wk.read(self.lr_dir(arm) / 'COMPLETE.json'),
                           heldout=wk.read(self.base / 'heldout' / f'{arm}.json')['overall'])
            if key in TRAINED:
                row['step'] = wk.read(self.lr_dir(arm) / 'step.json')
                if r['tuned']:
                    row['training'] = wk.read(self.lr_dir(arm) / 'training.json')['history']
            for part in ('fit', 'quant'):
                if part in row:
                    row[part] = {k: v for k, v in row[part].items() if k != 'rows'}
            rows[key] = row
        return rows

    def final(self):
        rows = self.gather()
        decision = decide(rows, LR['decision'])
        summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, status='complete', complete=True, lowrank=LR, rows=rows,
                       decision=decision, new_teacher_queries=0)
        wk.save(self.lr_dir() / 'summary.json', summary)
        wk.save(self.lr_dir() / 'DECISION.json', dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, **decision))
        (self.lr_dir() / 'RESULTS.zh.md').write_text(report(summary), encoding='utf-8')
        return summary

    # ---------------- transfer to other hosts (lowrank.json xfer) ----------------
    def xfer_rows(self, host):
        """U, E and J of one host: bits, C80 fit and held-out; J's step choice. U's fit is tune's fit_before."""
        select_host(host)

        def strip(d):
            return {k: v for k, v in d.items() if k != 'rows'}
        rows = dict(U=dict(arm=BASELINE_ARM, tuned=False, quant=strip(wk.read(self.arm_root(BASELINE_ARM) / 'COMPLETE.json')),
                           fit=strip(wk.read(self.base / 'tune' / host / 'fit_before.json')),
                           heldout=wk.read(self.base / 'heldout' / f'{BASELINE_ARM}.json')['overall']))
        for key in ('E', 'J'):
            arm = ARMS[key]
            r = wk.read(self.lr_dir(arm) / 'arm.json')
            row = dict(arm=arm, tuned=r['tuned'], eta=r.get('eta'))
            if key == 'E' or r['tuned']:
                row.update(quant=strip(wk.read(self.lr_dir(arm) / 'COMPLETE.json')), fit=strip(wk.read(self.lr_dir(arm) / 'fit.json')),
                           heldout=wk.read(self.base / 'heldout' / f'{arm}.json')['overall'])
            if key == 'J':
                step = wk.read(self.lr_dir(arm) / 'step.json')
                row['step'] = dict(chosen=step['chosen'], rows=[{k: s[k] for k in ('lr', 'predicted', 'realized')} for s in step['rows']])
            for k in ('mixed_loss', 'future_latent_nmse', 'first10motor_rmse', 'first10gripper_rmse'):
                if 'heldout' in row:
                    row.setdefault('heldout_vs_U', {})[k] = row['heldout'][k] / rows['U']['heldout'][k]
            rows[key] = row
        return rows

    def xfinal(self):
        hosts = [LR['host']] + list(LR['xfer']['hosts'])
        try:
            summary = dict(protocol=wk.PROTOCOL, pbs_jobid=self.job, complete=True, xfer=LR['xfer'],
                           hosts={h: self.xfer_rows(h) for h in hosts})
        finally:
            select_host(LR['host'])
        root = self.lr_dir('xfer')
        wk.save(root / 'summary.json', summary)
        (root / 'RESULTS.zh.md').write_text(xfer_report(summary), encoding='utf-8')
        return summary

    def stage(self, name, shard, shards):
        if name in ('xinit', 'xtrain'):  # one xfer host per GPU, in FW_HOSTS order
            hosts = (os.environ.get('FW_HOSTS') or ':'.join(LR['xfer']['hosts'])).split(':')
            if not set(hosts) <= set(LR['xfer']['hosts']) or shards != len(hosts):
                raise ValueError(f'xfer runs one host per GPU: {hosts}, {shards} shards')
            select_host(hosts[shard])
        if name in ('init', 'train', 'diag', 'tfdiag', 'tcond', 'xinit', 'xtrain'):
            self.load()
            self.capture_masters()
            self.attach()
        if name == 'xinit':
            self.init()
        elif name == 'xtrain':
            self.run_arm('J')
        elif name == 'xfinal':
            self.xfinal()
        elif name == 'diag':
            self.diag()
        elif name == 'tfdiag':
            self.tfdiag()
        elif name == 'tcond':
            self.tcond()
        elif name == 'init':
            if shard == 0:
                self.init()
        elif name == 'train':
            arms = (os.environ.get('FW_ARMS') or ':'.join(TRAINED)).split(':')  # ':' since apptainer --env splits on ','
            if not set(arms) <= set(TRAINED):
                raise ValueError(f'Unknown arms: {arms}')
            if shards == 1:
                for key in arms:
                    self.run_arm(key)
            elif shards == len(arms):
                self.run_arm(arms[shard])
            else:
                raise ValueError('train runs on 1 GPU (the arms in turn) or one GPU per arm')
        elif name == 'final':
            self.final()
        else:
            raise ValueError(f'Unknown step {name}')
        self.progress('stage_complete', stage=name, shard=shard, shards=shards)


LABELS = dict(U='baseline QTIP W2', E='EoRA 初始化（不训练）', J='联合目标 A + λZ + βK', D='解耦目标 A_tf + λZ + βK')
HOST_LABELS = dict(QTIP='QTIP W2（参照，C2 v2 门）', GPTVQ='GPTVQ W2', QUIPSHARP='QuIP# W2')


def xfer_report(s):
    lines = ['# EoRA-J 迁移到其他 W2 宿主（lowrank.json xfer）', '',
             f"作业 `{s['pbs_jobid']}`。E、J 原样搬到每个宿主上：rank {LR['rank']}，阻尼 {LR['init']['damping']}，"
             f"λ、β 不变，同一步长网格和 0.5 规则，C80 上 8 次更新。U 的 C80 拟合来自 tune 的 fit_before。", '',
             '| 宿主 | 臂 | bpw | η | C80 mixed | held-out mixed（×U） | future NMSE（×U） | first10 motor（×U） | gripper（×U） | 换 BF16 future 后 motor |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for host, rows in s['hosts'].items():
        for key in ('U', 'E', 'J'):
            r = rows[key]
            name = f"{HOST_LABELS.get(host, host)}" if key == 'U' else ''
            if 'heldout' not in r:
                lines.append(f"| {name} | {key} | 未训练（步长检查没过） | | | | | | | |")
                continue
            o, ratio = r['heldout'], r.get('heldout_vs_U', {})

            def cell(k, fmt):
                return f"{o[k]:{fmt}}" + (f"（{ratio[k]:.3f}）" if k in ratio else '')
            eta = f"{r['eta']:g}" if r.get('eta') is not None else '—'
            lines.append(f"| {name} | {key} | {r['quant']['bpw']:.4f} | {eta} | {r['fit']['mixed_loss']:.5f} | "
                         f"{cell('mixed_loss', '.5f')} | {cell('future_latent_nmse', '.4f')} | {cell('first10motor_rmse', '.4f')} | "
                         f"{cell('first10gripper_rmse', '.4f')} | {o['first10motor_rmse_swap']:.4f} |")
    lines += ['', '## J 的步长检查（第 1 批，从 E 出发）', '', '| 宿主 | η | 一阶预测下降 | 实际下降 | 比例 |', '|---|---:|---:|---:|---:|']
    for host, rows in s['hosts'].items():
        st = rows['J'].get('step')
        if not st:
            continue
        for row in st['rows']:
            ratio = row['realized'] / row['predicted'] if row['predicted'] else float('nan')
            lines.append(f"| {HOST_LABELS.get(host, host)} | {row['lr']:g} | {row['predicted']:.3g} | {row['realized']:.3g} | {ratio:.2f} |")
        lines.append(f"| {HOST_LABELS.get(host, host)} | 选定 | {st['chosen'] if st['chosen'] is not None else '无（不训练）'} | | |")
    return '\n'.join(lines) + '\n'


def report(s):
    rows, dec, rule = s['rows'], s['decision'], s['lowrank']['decision']
    lines = ['# C2 v2 门：每个目标 Linear 挂 rank-%d BF16 低秩支路，联合目标 vs 解耦目标' % s['lowrank']['rank'], '',
             f"作业 `{s['pbs_jobid']}`。宿主 QTIP W2（H_U），614 个目标 Linear 都挂同样的支路 y = xWᵀ + b + (xVᵀ)Uᵀ。"
             "E 是 EoRA 式闭式初始化（激活加权的误差低秩近似），J、D 从 E 出发，各 8 次逐张量相对步长更新（C80 一遍）。"
             "D 的动作项 A_tf 是量化模型读 BF16 future 时的动作误差，动作一侧不会因为迁就坏掉的 future 得到奖励。", '',
             '| 臂 | bpw | proxy(H_U) | C80 mixed | C80 future NMSE | held-out mixed | first10 motor RMSE | future NMSE | future 占比 | gripper RMSE | 换 BF16 future 后 motor |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('U', 'E', 'J', 'D'):
        r = rows[key]
        if 'heldout' not in r:
            lines.append(f'| {LABELS[key]} | 未训练（步长检查没过） | | | | | | | | | |')
            continue
        q, f, o = r.get('quant', {}), r['fit'], r['heldout']
        bpw = f"{q['bpw']:.4f}" if q else '—'
        proxy = f"{q['proxy_nmse_weighted']:.5f}" if q else '—'
        lines.append(f"| {LABELS[key]} | {bpw} | {proxy} | {f['mixed_loss']:.5f} | {f['future_latent_nmse']:.4f} | "
                     f"{o['mixed_loss']:.5f} | {o['first10motor_rmse']:.4f} | {o['future_latent_nmse']:.4f} | "
                     f"{gw.future_share(o):.3f} | {o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} |")
    lines += ['', '## 步长检查（第 1 批，从 E 出发）', '', '| 臂 | η | 一阶预测下降 | 实际下降 | 比例 |', '|---|---:|---:|---:|---:|']
    for key in TRAINED:
        st = rows[key]['step']
        for row in st['rows']:
            ratio = row['realized'] / row['predicted'] if row['predicted'] else float('nan')
            lines.append(f"| {key} | {row['lr']:g} | {row['predicted']:.3g} | {row['realized']:.3g} | {ratio:.2f} |")
        lines.append(f"| {key} | 选定 | {st['chosen'] if st['chosen'] is not None else '无（不训练）'} | | |")
    lines += ['', '梯度最集中的层（占 |g|² 的比例）：']
    for key in TRAINED:
        lines.append(f"- {key}：" + '，'.join(f"{x['module']} {x['share']:.2f}" for x in rows[key]['step']['top_layers'][:5]))
    lines += ['', '各批更新前 loss：']
    for key in TRAINED:
        if rows[key]['tuned']:
            lines.append(f"- {key}（η = {rows[key]['eta']}）：" + ' '.join(f"{h['before']['total']:.4g}" for h in rows[key]['training']))
    yes = {True: '是', False: '否'}
    lines += ['', '## 判定（预先登记）', '',
              f"一臂通过 = 训练过，且 held-out future NMSE ≤ {rule['future_ratio_max']} × U、mixed loss ≤ {rule['mixed_ratio_max']} × U、"
              "两者都低于 E；D 通过则上闭环。解耦增量（只报告）= D 的 future NMSE 低于 J 且 mixed 不高于 J。", '',
              '| 臂 | future / U | mixed / U | 两项都低于 E | 通过 |', '|---|---:|---:|---|---|']
    for key in TRAINED:
        r = dec['arms'][key]
        if not r['tuned']:
            lines.append(f'| {key} | — | — | — | 否（未训练） |')
            continue
        lines.append(f"| {key} | {r['future_ratio']:.3f} | {r['mixed_ratio']:.3f} | {yes[r['below_E']]} | {yes[r['passed']]} |")
    lines.append(f"\n解耦增量：{yes[dec['decoupling_increment']]}。结论：{'D 通过，上闭环' if dec['go'] else 'D 未通过，不上闭环'}。")
    return '\n'.join(lines) + '\n'


GROUP_LABELS = dict(video='视频专家主体', video_time='视频专家时间条件', action='动作专家主体', action_time='动作专家时间条件',
                    other='其他（proprio）', all='全部')
CATEGORY_LABELS = dict(seed='同一批 context，seed 1', ctx='其他 context，seed 0', ctx_seed='其他 context，seed 1')


def diag_report(s):
    a, rd, cfg, fits = s['analysis'], s['reading'], s['diag'], s['fits']
    sd = cfg['seed']
    tc, pc = s['teacher_check'], s['predicted_check']
    lines = ['# C2 v2 低秩支路：C80 上的 future 修复为什么到不了 held-out', '',
             f"作业 `{s['pbs_jobid']}`。C80 和 held-out 是同样的 40 个任务（每个任务 2 个 / 1 个不相交的 demo 快照，相位位置相同），"
             "J 在每批更新前对这批新 context 的 loss 只有 E 的 0.93–1.07 倍，训过的批次却降到 0.64–0.77 倍："
             "修复留在训过的 context 上。这里问它跟着 context 走还是跟着那一条 sampler seed 轨迹走，以及一批的梯度有多少是别的样本共有的。", '',
             f"## 1. 同样的 C80 context，换 seed {sd}（没训练过的噪声）", '',
             f"seed {sd} 的 BF16 teacher 在本作业里生成；先用同样的方法重生成前 {len(tc['rows'])} 个 context 的 seed 0 teacher，"
             f"和缓存比 future NMSE 最大 {tc['worst']:.2e}（上限 {tc['limit']:g}）。", '',
             '| 臂 | future NMSE s0 | future NMSE s%d | s0 / E | s%d / E | mixed s0 | mixed s%d | ρ future | ρ mixed |' % (sd, sd, sd),
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in fits:
        f = fits[key]
        r = rd['rho'].get(key)
        fl, ml = 'future_latent_nmse', 'mixed_loss'
        cells = [f"{f[0][fl]:.4f}", f"{f[sd][fl]:.4f}", f"{r[fl]['r0']:.3f}" if r else '—', f"{r[fl]['r1']:.3f}" if r else '—',
                 f"{f[0][ml]:.5f}", f"{f[sd][ml]:.5f}", f"{r[fl]['rho']:.2f}" if r else '—', f"{r[ml]['rho']:.2f}" if r else '—']
        lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'ρ = (1 − r₁) / (1 − r₀)，r = 臂 / E：换噪声后保留下来的那部分改进。', '', '按 suite（future NMSE / E）：', '',
              '| suite | ' + ' | '.join(f'{k} s0 | {k} s{sd}' for k in fits if k != 'E') + ' |',
              '|---|' + '---:|' * (2 * (len(fits) - 1))]
    for suite, by in s['by_suite'].items():
        e = by['E']
        lines.append(f'| {suite} | ' + ' | '.join(f"{by[k][0]['future_latent_nmse'] / e[0]['future_latent_nmse']:.3f} | "
                                                 f"{by[k][sd]['future_latent_nmse'] / e[sd]['future_latent_nmse']:.3f}"
                                                 for k in fits if k != 'E') + ' |')
    n = len(a['batches'][0]['cases'])
    lines += ['', f"## 2. E 处 D 目标的逐样本梯度（前 {cfg['grad']['batches']} 个训练批次的 {cfg['grad']['batches'] * n} 个 context × seed 0/{sd}）", '',
              'F = λZ + βK 经学生 query 回传（future 一侧），A = A_tf 经读 teacher future 的学生回传（动作一侧）。', '',
              '| 组 | F 能量占比 | A 能量占比 | F cos 同 context 两 seed | F cos 同 suite | F cos 其他 | A cos 同 context | A cos 同 suite | A cos 其他 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for grp in GROUPS + ('all',):
        def cos(path, k):
            c = a['cosines'].get(f'{grp}/{path}')
            return f'{c[k]:.3f}' if c and k in c else '—'
        lines.append(f"| {GROUP_LABELS[grp]} | {a['energy'][f'{grp}/F']:.3f} | {a['energy'][f'{grp}/A']:.3f} | "
                     f"{cos('F', 'seed')} | {cos('F', 'suite')} | {cos('F', 'other')} | "
                     f"{cos('A', 'seed')} | {cos('A', 'suite')} | {cos('A', 'other')} |")
    lines += ['', '一批 seed 0 样本的 |平均梯度|² / 平均 |梯度|²（互不相关时为 1/%d = %.2f）：' % (n, 1 / n) + '，'.join(
        f"{p} " + ' / '.join(f'{x:.3f}' for x in v) for p, v in a['snr'].items()), '',
              f"一步训练（D 的逐张量相对步长，η = {cfg['grad']['eta']}，方向 = 这批 seed 0 样本的平均梯度）的一阶 loss 下降，"
              '相对于这批样本自身的下降（4 批合计）：', '',
              '| 样本 | F（future 一侧） | A（动作一侧） | 合计 |', '|---|---:|---:|---:|']
    for k, label in CATEGORY_LABELS.items():
        if k in a['T']:
            t = a['T'][k]
            lines.append(f"| {label} | {t['F']:.3f} | {t['A']:.3f} | {t['total']:.3f} |")
    ins = {p: sum(b['in'][p] for b in a['batches']) / len(a['batches']) for p in ('F', 'A', 'total')}
    lines += ['', f"这批样本自身的一阶下降（4 批平均）：F {ins['F']:.3g}，A {ins['A']:.3g}，合计 {ins['total']:.3g}。", '',
              'F 一侧一阶下降按组（4 批平均；这批自身 / 其他 context seed 0）：' + '，'.join(
                  f"{GROUP_LABELS[grp]} {sum(b['in']['F_by_group'][grp] for b in a['batches']) / len(a['batches']):.3g} / "
                  f"{sum(b['ctx']['F_by_group'][grp] for b in a['batches']) / len(a['batches']):.3g}" for grp in GROUPS), '',
              f"核对：第 1 批的 Σ‖P‖‖ḡ‖ = {pc['unit']:.5g}，step.json 的 predicted_per_eta = {pc['step_json']:.5g}"
              f"（相对差 {pc['rel']:+.2e}，{'通过' if pc['passed'] else '不通过'}）。", '',
              '## 判读（预先登记）', '']
    rule = cfg['reading']
    rho = rd['rho']['D']['future_latent_nmse']['rho']
    label = dict(context='跟着 context 走', noise='跟着噪声轨迹走', both='两者都有')[rd['attached']]
    lines += [f"- D 的 ρ（future）= {rho:.2f}（≥ {rule['context_rho_min']} 记为跟着 context，≤ {rule['noise_rho_max']} 记为跟着噪声）：{label}。",
              f"- 一阶共享（F，门槛 {rule['shared_min']}）：同 context 换 seed {a['T']['seed']['F']:.3f}"
              f"（{'共享' if rd['shared']['seed'] else '不共享'}），其他 context {a['T']['ctx']['F']:.3f}"
              f"（{'共享' if rd['shared']['ctx'] else '不共享'}）。"]
    return '\n'.join(lines) + '\n'


PATH_LABELS = dict(tf='逐步 teacher forcing（tf）', ro='自己的 rollout（ro）')


def tf_report(s):
    a, rd, cfg, v = s['analysis'], s['reading'], s['tfdiag'], s['values']
    sc, rc, st = s['state_check'], s['ro_check'], s['step_check']
    rule, n = cfg['reading'], len(a['batches'][0]['cases'])
    lines = ['# C2 v2：逐步 teacher forcing 的 future 目标有没有通用部分', '',
             f"作业 `{s['pbs_jobid']}`。fwld1（26063678）：E 处 rollout future 梯度 67% 属于这一条去噪轨迹、33% 属于这个 context、"
             "通用部分 0.2%。这里在同样的 80 个样本上，把 future 目标换成逐步 teacher forcing：学生在 BF16 teacher 每一步的状态上算速度，"
             "x~ = x_T + Σ_i Δ_i (v_S − v_T)，目标 λZ(x~) + βK(x~)。问它的梯度有多少是别的 context、别的噪声共有的，"
             "以及沿它走一步对别的 context 的 rollout future 有多大作用。", '',
             '## 核对', '',
             f"- teacher 状态：{len(sc['rows'])} 个样本的视频阶段最后一步和 teacher future 的 NMSE 最大 {sc['future_nmse_worst']:.2e}"
             f"（上限 {cfg['checks']['state_future_nmse_max']:g}），第 0 帧最大差 {sc['frame0_worst']:g}；"
             f"每个记录的状态和前一步截断后的结果最大差 {sc['chain_worst']:g}；"
             f"装 BF16 权重做 teacher forcing，速度和记录值最大差 {sc['bf16_tf_max_abs']:g}。",
             f"- 重算的 ro 梯度和 diag/gram.pt 的 F 块：相对差 {rc['rel']:.2e}（上限 {rc['limit']:g}，{'通过' if rc['passed'] else '不通过'}）。",
             f"- 步长核对（报告，不作门槛）：样本 {st['case_id']}/s{st['seed_index']}，η = {st['eta']}，tf 目标实际下降 {st['realized']:.3g}，"
             f"一阶预测 {st['predicted']:.3g}（比值 {st['ratio']:.2f}）。" if st['ratio'] is not None else '- 步长核对：预测下降为 0。',
             f"- 回放最大差 {max(r['replay_max_abs'] for r in s['rows']):g}。", '',
             '## 1. 两个目标在 E 处的值（80 个样本平均）', '',
             '| | Z（future NMSE） | K | λZ + βK |', '|---|---:|---:|---:|',
             f"| ro | {v['z_ro']:.4f} | {v['k_ro']:.4f} | {v['total_ro']:.3g} |",
             f"| tf | {v['z_tf']:.4f} | {v['k_tf']:.4f} | {v['total_tf']:.3g} |", '',
             f"tf 位移 x~ − x_T 和 rollout 位移 x_S − x_T：余弦 {v['cos_tf_ro']:.3f}，长度比 {v['norm_tf_ro']:.3f}。"
             + '按 seed：' + '；'.join(f"seed {sd}：Z_ro {b['z_ro']:.4f}，Z_tf {b['z_tf']:.4f}，余弦 {b['cos_tf_ro']:.3f}"
                                     for sd, b in v['by_seed'].items()) + '。', '',
             '逐步（第 i 步的位移 Δ_i (v_S − v_T) 的能量 / |x_T|²，以及它和 rollout 位移的余弦）：', '',
             '| 步 | ' + ' | '.join(str(i + 1) for i in range(len(v['step_energy']))) + ' |',
             '|---|' + '---:|' * len(v['step_energy']),
             '| 能量 | ' + ' | '.join(f'{x:.2e}' for x in v['step_energy']) + ' |',
             '| 余弦 | ' + ' | '.join(f'{x:.3f}' for x in v['step_cos_ro']) + ' |', '',
             f"## 2. 逐样本梯度（前 {LR['diag']['grad']['batches']} 个训练批次的 {LR['diag']['grad']['batches'] * n} 个 context × seed 0/1）", '',
             '| 组 | tf 能量占比 | ro 能量占比 | tf cos 同 context 两 seed | tf cos 同 suite | tf cos 其他 | ro cos 同 context | ro cos 同 suite | ro cos 其他 | tf·ro 同样本 | tf·ro 其他 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for grp in GROUPS + ('all',):
        if f'{grp}/tf' not in a['energy']:
            continue

        def cos(path, k):
            c = a['cosines'].get(f'{grp}/{path}')
            return f'{c[k]:.3f}' if c and k in c else '—'
        x = a['cross'].get(grp)
        lines.append(f"| {GROUP_LABELS[grp]} | {a['energy'][f'{grp}/tf']:.3f} | {a['energy'][f'{grp}/ro']:.3f} | "
                     f"{cos('tf', 'seed')} | {cos('tf', 'suite')} | {cos('tf', 'other')} | "
                     f"{cos('ro', 'seed')} | {cos('ro', 'suite')} | {cos('ro', 'other')} | "
                     + (f"{x['self']:.3f} | {x['other']:.3f} |" if x else '— | — |'))
    lines += ['', '一批 seed 0 样本的 |平均梯度|² / 平均 |梯度|²（互不相关时为 1/%d = %.2f）：' % (n, 1 / n) + '，'.join(
        f"{p} " + ' / '.join(f'{x:.3f}' for x in vals) for p, vals in a['snr'].items()), '',
              f"## 3. 走一步（D 的逐张量相对步长，η = {cfg['eta']}，方向 = 这批 seed 0 样本的平均梯度）", '',
              'T：这一步对自己目标的一阶下降 / 对这批样本自身的下降。G：这一步对 rollout 目标（ro）的一阶下降 / '
              'ro 自己那一步对这批样本自身的下降。4 批合计。', '',
              '| 样本 | T（tf 步，tf 目标） | T（ro 步，ro 目标） | G（tf 步，ro 目标） | G（ro 步，ro 目标） |', '|---|---:|---:|---:|---:|']
    labels = dict(CATEGORY_LABELS, **{'in': '这批样本自身'})
    for k in ('in', 'seed', 'ctx', 'ctx_seed'):
        tt = a['T']['tf'].get(k, 1.0 if k == 'in' else None)
        tr = a['T']['ro'].get(k, 1.0 if k == 'in' else None)
        lines.append(f"| {labels[k]} | {tt:.3f} | {tr:.3f} | {a['G']['tf'][k]:.3f} | {a['G']['ro'][k]:.3f} |")
    units = {src: [b[src]['unit'] for b in a['batches']] for src in ('tf', 'ro')}
    lines += ['', '每批 Σ‖P‖‖ḡ‖：' + '；'.join(f"{src} " + ' / '.join(f'{u:.4g}' for u in us) for src, us in units.items()) + '。', '',
              '## 判读（预先登记）', '',
              f"- tf 的通用部分：T（tf 步，其他 context，seed 1）= {rd['T_tf_ctx_seed']:.3f}（门槛 {rule['shared_min']}）："
              f"{'共享' if rd['shared'] else '不共享'}。",
              f"- 对 held-out 类样本 rollout future 的作用：G（tf 步，其他 context，seed 1）= {rd['G_tf_ctx_seed']:.3f}"
              f"（门槛 {rule['rollout_fresh_min']}；ro 自己那一步是 {rd['G_ro_ctx_seed']:.3f}）：{'够' if rd['reaches'] else '不够'}。",
              f"- {'go：在 A_tf + λZ(x~) + βK(x~) 上训练一个臂。' if rd['go'] else 'no-go。'}"]
    if not rd['go']:
        lines.append('- ' + ('tf 的局部误差是共享的，但别的 context 的 rollout future 误差不由它构成（后续步的传播占主导），只做 teacher forcing 到不了 held-out future。'
                             if rd['shared'] else
                             '局部误差同样按 context 各自不同：从 C80 规模的标定集出发、经权重修 future 这条路走不通。'))
    return '\n'.join(lines) + '\n'


TC_LABELS = dict(U_TV='U + 视频时间条件 BF16', U_TA='U + 动作时间条件 BF16', U_TVA='U + 两者 BF16', E_TVA='E + 两者 BF16')


def tcond_report(s):
    cfg, ic, bits, held, ratios, rd = s['tcond'], s['input_check'], s['bits'], s['heldout'], s['ratios'], s['reading']
    rule = cfg['reading']
    lines = ['# 时间条件层归因：W2 的 held-out future 误差有多少来自时间条件路径', '',
             f"作业 `{s['pbs_jobid']}`。tfdiag 里逐步 teacher forcing 梯度能量的 86% 和它的共享部分都在视频专家的 3 个时间条件 Linear 上。"
             "这些层的输入只是 timestep 的正弦编码，与 context 无关；把它们换回 BF16 权重，输出就和 BF16 模型完全一样。"
             "固定 10 步采样时，这也等于按 timestep 存一张输出表。", '',
             '## 输入核对（BF16，两个不同 context、不同 seed）', '',
             '| Linear | 调用次数 | 两个样本逐次最大差 | 不同输入行数 |', '|---|---:|---:|---:|']
    for n, r in ic['modules'].items():
        lines.append(f"| {n} | {r['calls']} | {r['max_abs']:g} | {r['distinct_rows']} |")
    lines += ['', '## held-out 40×2', '',
              '| 臂 | bpw（BF16 权重） | bpw（按 timestep 存表） | future NMSE | / 基臂 | mixed | / 基臂 | first10 motor | gripper | 换 BF16 future 后 motor | future 占比 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    base_label = {LR['baseline_arm']: 'U（QTIP W2）', ARMS['E']: 'E（EoRA）'}
    for arm in (LR['baseline_arm'], ARMS['E']):
        o = held[arm]
        lines.append(f"| {base_label[arm]} | — | — | {o['future_latent_nmse']:.4f} | — | {o['mixed_loss']:.5f} | — | "
                     f"{o['first10motor_rmse']:.4f} | {o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} | "
                     f"{gw.future_share(o):.3f} |")
    for key, c in cfg['arms'].items():
        arm, o, q, b = c['arm'], held[c['arm']], ratios[c['arm']], bits[c['arm']]
        lines.append(f"| {TC_LABELS[key]} | {b['bpw_weights']:.4f} | {b['bpw_table']:.4f} | {o['future_latent_nmse']:.4f} | "
                     f"{q['future_latent_nmse']:.3f} | {o['mixed_loss']:.5f} | {q['mixed_loss']:.3f} | {o['first10motor_rmse']:.4f} | "
                     f"{o['first10gripper_rmse']:.4f} | {o['first10motor_rmse_swap']:.4f} | {gw.future_share(o):.3f} |")
    label = dict(carries='时间条件路径承载了至少十分之一的 W2 future 误差，修复精确且与 context 无关：C2 v3 以按 timestep 的条件表为基础，下一步上闭环（U_TV 对 U）',
                 partial='只占一小部分，和 E_TVA 一起看再定',
                 negligible='tfdiag 找到的共享局部误差不是 W2 future 误差的来源；这个宿主上经权重修 future 已到头，C2 转向动作一侧')[rd['verdict']]
    lines += ['', '## 判读（预先登记）', '',
              f"- r = U_TV 的 held-out future NMSE / U = {rd['r']:.3f}（≤ {rule['carries_max']} 承载，> {rule['negligible_min']} 可忽略）：{label}。"]
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
        LowRankWork(cfg, job).stage(stage, shard, shards)
    except BaseException as error:
        wk.save(failure, dict(protocol=wk.PROTOCOL, pbs_jobid=os.environ.get('PBS_JOBID'), job=job,
                              error=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
