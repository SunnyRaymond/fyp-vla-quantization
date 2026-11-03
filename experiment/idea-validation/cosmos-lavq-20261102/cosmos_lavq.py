"""Cosmos-Policy side of LAVQ: contexts, the differentiable five-evaluation endpoint with solver-boundary captures, the
future-slot swap and the frozen BF16 suffix reader, action errors, the soft-bin reader KL and its curvature.

Cosmos denoises the nine latent slots of LIBERO (observation, proprio, action = slot 4, future proprio, future wrist /
third-person images = slots 6 / 7, value) jointly with one DiT. The reader starts at a cut: the solver state just before
the cut-th denoiser evaluation (res_multistep: four loop steps and a final denoise, cuts 2..4 lie inside the loop). It
runs the remaining steps from a state whose future-image slots are replaced, as cosmos-policy-vqwam-20261024/reader.py
(teacher identity 1e-6 and connected input gradients there).
"""
from __future__ import annotations

import inspect

import torch

FUT = slice(6, 8)  # future wrist / future third-person image slots of the nine-slot layout


def _counter(function, seen=None, depth=0):
    """The endpoint's sampler-evaluation counter, found through the sampler closures."""
    seen = set() if seen is None else seen
    if depth > 10 or id(function) in seen:
        return None
    seen.add(id(function))
    if not inspect.isfunction(function):
        return None
    for cell in function.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, dict) and 'sampler_evaluations' in value:
            return value
        if inspect.isfunction(value):
            found = _counter(value, seen, depth + 1)
            if found is not None:
                return found
    return None


def _clone(value):
    if torch.is_tensor(value):
        with torch.inference_mode(False):
            return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    return value


class Capture:
    """Solver states at the requested cuts of one endpoint run, with the solver body to continue from them."""

    def __init__(self, cuts=(), retain=False):
        self.cuts, self.retain = {int(c) for c in cuts}, bool(retain)
        self.body = self.nonlocals = self.upper = None
        self.states, self.starts, self.live = {}, {}, {}

    def __enter__(self):
        from cosmos_policy._src.imaginaire.modules import res_sampler
        self.module, self.original = res_sampler, res_sampler.fori_loop

        def loop(lower, upper, body_fun, init_val):
            counter = _counter(body_fun)
            if counter is None:
                raise RuntimeError('Cannot locate the sampler evaluation counter')

            def wrapped(index, state):
                cut = int(counter['sampler_evaluations']) + 1
                if cut in self.cuts and cut not in self.states:
                    if self.body is None:
                        self.body, self.upper = body_fun, int(upper)
                        self.nonlocals = inspect.getclosurevars(body_fun).nonlocals
                    self.starts[cut] = int(index)
                    self.states[cut] = _clone(state)
                    if self.retain:
                        self.live[cut] = state[0]
                return body_fun(index, state)
            return self.original(lower, upper, wrapped, init_val)

        res_sampler.fori_loop = loop
        return self

    def __exit__(self, *_):
        self.module.fori_loop = self.original

    def require(self):
        missing = self.cuts - set(self.states)
        if missing:
            raise RuntimeError(f'Cuts {sorted(missing)} are not solver-loop boundaries')
        if self.cuts and ('x0_fn' not in self.nonlocals or 'sigmas_L' not in self.nonlocals):
            raise RuntimeError('Solver continuation lacks the denoiser or the sigma schedule')
        return self

    def future(self, cut):
        """Future slots of the cut state in the solver's own float64: a float32 round trip flips BF16 roundings of the
        remaining denoiser calls (reader identity 3e-3 in smoke 26095411)."""
        return self.states[cut][0][:, :, FUT].detach().clone()

    def continue_from(self, cut, x):
        """The final denoised latent of the remaining solver steps from the captured state at cut, state[0] = x."""
        state = _clone(self.states[cut])
        state = type(state)((x, *state[1:]))
        for index in range(self.starts[cut], self.upper):
            state = self.body(index, state)
        v = self.nonlocals
        ones = v.get('ones_B')
        if ones is None:
            ones = torch.ones(state[0].shape[0], device=state[0].device, dtype=torch.float64)
        return v['x0_fn'](state[0], v['sigmas_L'][-1] * ones)


def with_future(x, z):
    """x with its future-image slots set to z (differentiable in z)."""
    full = x.detach().clone()
    full[:, :, FUT] = z.to(full.dtype)
    return full


def build_context(p, rt, case, seed_index):
    """The official get_action input batch of a frozen calibration / held-out case, cut at the sampler boundary (no
    model forward), with the RNG state after preprocessing (cosmos-policy-vqwam get_context)."""
    import calibrate
    from common import calibration_input
    from cosmos_runtime import _cpu_tree
    observation, metadata, row = calibration_input(p, case)
    seed = int(row['sampler_seeds'][int(seed_index)])
    language = metadata['description']
    if language not in rt.cosmos_utils.t5_text_embeddings_cache:
        raise RuntimeError(f'T5 cache miss: {language!r}')
    rt.cfg.seed = seed
    rt.cfg.task_suite_name = str(metadata['suite'])
    rt.run_libero_eval.set_seed_everywhere(seed)
    obs = rt._policy_observation(observation)
    captured = {}

    class Stop(Exception):
        pass

    def stop(data_batch, *args, **kwargs):
        captured['data_batch'] = data_batch
        raise Stop()

    original = rt.model.generate_samples_from_batch
    rt.model.generate_samples_from_batch = stop
    try:
        try:
            with torch.inference_mode():
                rt.cosmos_utils.get_action(
                    rt.cfg, rt.model, rt.dataset_stats, obs, language, seed=seed, randomize_seed=False,
                    num_denoising_steps_action=int(rt.protocol['inference']['denoising_steps']),
                    generate_future_state_and_value_in_parallel=bool(
                        rt.protocol['inference']['parallel_action_future_value']))
        except Stop:
            pass
    finally:
        rt.model.generate_samples_from_batch = original
    if 'data_batch' not in captured:
        raise RuntimeError('get_action did not reach its sampler boundary')
    batch = _cpu_tree(captured['data_batch'], torch)
    return dict(case=int(case), seed_index=int(seed_index), seed=seed, data_batch=batch,
                latent_indices=_cpu_tree(batch.get('latent_indices'), torch),
                pre_forward_rng_state=calibrate._rng_state(torch), suite=str(metadata['suite']), description=language)


def run(rt, ctx, cuts=(), retain=False, checkpoint=False):
    """One differentiable official endpoint run (grad_enabled: a graph only where parameters or inputs require grad)."""
    import calibrate
    import endpoint as legacy
    calibrate._restore_rng(ctx['pre_forward_rng_state'], torch)
    with Capture(cuts, retain) as cap:
        result = legacy.run_endpoint(rt.model, ctx['data_batch'], int(ctx['seed']), rt.protocol, grad_enabled=True,
                                     checkpoint_denoiser=checkpoint, dataset_stats=rt.dataset_stats,
                                     latent_indices=ctx.get('latent_indices'))
    return result, cap.require()


def slot_indices(rt, ctx, device='cpu'):
    import endpoint as legacy
    batch = ctx['data_batch']
    out = {}
    for key in ('action_latent_idx', 'future_wrist_image_latent_idx', 'future_image_latent_idx'):
        out[key] = int(legacy._tensor_index(batch, key, 1, device, torch, ctx.get('latent_indices'))[0])
    return out


def action_from_latent(rt, generated):
    """The unnormalized [B, 16, 7] action chunk from the action slot of a generated latent (official extraction)."""
    size = generated.shape[0]
    width = int(rt.protocol['inference']['chunk_size']) * 7
    slot = generated[:, :, 4].reshape(size, -1)
    chunks = slot.shape[-1] // width
    action = slot[:, :chunks * width].reshape(size, chunks, width // 7, 7).mean(1)
    if rt.protocol['inference']['unnormalize_actions']:
        lo = torch.as_tensor(rt.dataset_stats['actions_min'], device=action.device).float()
        hi = torch.as_tensor(rt.dataset_stats['actions_max'], device=action.device).float()
        action = .5 * (action.float() + 1) * (hi - lo) + lo
    return action


def continue_action(rt, cap, cut, x, dtype):
    return action_from_latent(rt, cap.continue_from(cut, x).to(dtype))


def action_mse(a, t):
    return (a[..., :6].float() - t.to(a.device)[..., :6].float()).square().mean()


def gripper_mse(a, t):
    return (a[..., 6].float() - t.to(a.device)[..., 6].float()).square().mean()


def nmse(z, t):
    t = t.to(z.device).float()
    return (z.float() - t).square().sum() / t.square().sum().clamp_min(1e-12)


def normalized_motor(a, stats):
    lo, hi = stats['actions_min'].to(a.device), stats['actions_max'].to(a.device)
    return (2 * (a.float() - lo) / (hi - lo).clamp_min(1e-12) - 1)[..., :6]


def reader_kl(a, t, stats, cfg):
    """Mean per-scalar soft-bin KL(teacher || a) over the 16 x 6 normalized motor actions."""
    import softbin_loss
    config = {k: cfg[k] for k in ('interior_min', 'interior_max', 'sigma', 'std_floor')}
    config['bins'] = cfg['soft_bins']
    return softbin_loss.pointwise_kl(torch, normalized_motor(a, stats), normalized_motor(t, stats), stats['mean'],
                                     stats['scale'], -1, config)


def kl_curvature(t, stats, cfg):
    """Diagonal of d^2 K / da^2 at a = t in motor units (K as reader_kl, without its clamp_min(0), which would cut the
    gradient at the exact zero); [B, 16, 6]."""
    import softbin_loss
    t = t.detach().float()
    a = t.clone().requires_grad_(True)
    m = stats['mean'].to(t.device).float().reshape(-1)
    s = stats['scale'].to(t.device).float().reshape(-1).clamp_min(cfg['std_floor'])
    edges = torch.linspace(cfg['interior_min'], cfg['interior_max'], 15, device=t.device)
    tlp = softbin_loss.softbin_log_probs(torch, (normalized_motor(t, stats) - m) / s, edges, cfg['sigma'])
    alp = softbin_loss.softbin_log_probs(torch, (normalized_motor(a, stats) - m) / s, edges, cfg['sigma'])
    k = (tlp.exp() * (tlp - alp)).sum(-1).mean()
    grad, = torch.autograd.grad(k, a, create_graph=True)
    curvature, = torch.autograd.grad(grad[..., :6].sum(), a)
    return curvature[..., :6].detach().clamp_min(0)


class Reader:
    """The frozen BF16 suffix reader of one context at a cut. Build and read with the BF16 weights installed."""

    def __init__(self, rt, ctx, cut):
        self.rt, self.cut = rt, int(cut)
        result, self.cap = run(rt, ctx, (self.cut,))
        self.action = result.action.detach().float()
        self.dtype = result.generated_latent.dtype
        self.z = self.cap.future(self.cut)

    def read(self, z):
        x = with_future(self.cap.states[self.cut][0], z)
        return continue_action(self.rt, self.cap, self.cut, x, self.dtype)

    def identity(self):
        return float((self.read(self.z).detach() - self.action).abs().max())
