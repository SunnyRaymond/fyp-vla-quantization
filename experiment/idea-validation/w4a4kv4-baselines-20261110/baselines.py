"""The four W4A4KV4 baselines: per-layer fitting (weights) and the per-layer forward path (activations).

QuaRot     fixed randomized signed block-Hadamard R128 per stream, GPTQ weights in rotated coordinates, A4 symmetric
SpinQuant  the same base times the learned Cayley rotation (reused from the earlier W4A4 runs), GPTQ, A4 symmetric
SVDQuant   channel smoothing alpha 0.5, BF16 rank-32 branch, residual W4, A4 on the smoothed input
ViDiT-Q    channel balancing alpha 0.5665, random signs + full Hadamard, affine W4 (two-stage), affine A4
Every scale is per group of 32 channels; rotations and smoothing use the frozen per-model calibration statistics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import scalarq as sq

ROTATED = {"quarot": "fixed", "spinquant": "learned"}


class Coords:
    """x -> xR and W -> WR with one RotationState (fixed or learned)."""

    def __init__(self, state, prefix, stream_of):
        self.state, self.prefix, self.stream_of = state, prefix, stream_of

    def x(self, value, name):
        return self.state.transform(value.float(), self.stream_of(self.prefix + name))

    def sandwich(self, h, name):
        """R^T H R."""
        return self.x(self.x(h, name).t(), name).t()


def build_coords(cfg, arm, shapes, device):
    """RotationState for a rotated arm. shapes: {module: (out_features, in_features)}."""
    import rotation as rot
    r = cfg["rotation"]
    prefix = r.get("alias_prefix", "")
    mods = {prefix + name: torch.nn.Linear(shape[1], 1, bias=False) for name, shape in shapes.items()}
    mode = ROTATED[arm]
    seed = r["seed"]
    state = rot.build_rotation_state(mods, mode=mode, seed=seed)
    if mode == "learned":
        saved = torch.load(r["learned"], map_location="cpu", weights_only=True)
        sd = saved.get(r["learned_key"])
        if sd is None or saved.get("seed", seed) != seed:
            raise RuntimeError(f"Learned rotation binding differs: {r['learned']}")
        state.load_state_dict(sd)
    for value in state.parameters():
        value.requires_grad_(False)
    state.to(device)
    state.refresh()
    return Coords(state, prefix, rot._stream_for_name)


def fit_layer(arm, w, h, amax, coords, name, index, quant, methods):
    """Quantize one layer. w [out, in] FP32 and h [in, in] FP32 on the GPU, amax [in] for the channel methods.
    Returns (bank, proxy_nmse)."""
    group = quant["group"]
    m = methods[arm]
    if arm in ROTATED:
        wc = coords.x(w, name)
        hc = coords.sandwich(h, name)
        codes, scales = sq.gptq(wc, hc, group, quant["gptq_damp"])
        bank = sq.encode_sym(codes, scales, group)
        return bank, sq.proxy_nmse(sq.decode(bank, torch.float32), wc, hc)
    wmax = w.abs().amax(0).clamp_min(1e-8)
    amax = amax.to(w.device, dtype=torch.float32).clamp_min(1e-8)
    if arm == "svdquant":
        alpha = m["alpha"]
        s = amax.pow(alpha) / wmax.pow(1 - alpha)
        t = w * s
        retained = min(m["rank"], *w.shape)
        with torch.random.fork_rng(devices=[w.device.index or 0] if w.device.type == "cuda" else []):
            torch.manual_seed(quant["seed"] + index * 1009)
            u, sv, v = torch.svd_lowrank(t, q=min(retained + m["oversampling"], *w.shape), niter=m["power_iterations"])
        root = sv[:retained].sqrt()
        up = (u[:, :retained] * root).to(torch.bfloat16)
        down = (root[:, None] * v[:, :retained].t()).to(torch.bfloat16)
        low = up.float() @ down.float()
        codes, scales = sq.rtn_sym(t - low, group)
        bank = dict(sq.encode_sym(codes, scales, group), smoothing=s, up=up, down=down, rank=retained)
        hs = h / s[None, :] / s[:, None]
        return bank, sq.proxy_nmse(sq.decode(bank, torch.float32) + low, t, hs)
    if arm == "viditq":
        alpha = m["alpha"]
        mask = wmax.pow(alpha) / amax.pow(1 - alpha)
        signs = sq.random_signs(w.shape[1], quant["seed"] + index * 17, w.device)
        first = sq.encode_aff(w / mask, group)
        target = sq.rotate(w / mask, signs)
        bank = dict(sq.encode_aff(sq.rotate(sq.decode(first, torch.float32), signs), group), smoothing=mask, signs=signs)
        hz = sq.rotate(sq.rotate(h * mask[None, :] * mask[:, None], signs).t(), signs).t()
        return bank, sq.proxy_nmse(sq.decode(bank, torch.float32), target, hz)
    raise ValueError(arm)


def forward_for(arm, module, bank, coords, name, group):
    """The fake-quantized forward of one Linear; module.weight already holds the dequantized BF16 weight."""
    if arm in ROTATED:
        def run(x):
            return F.linear(sq.q_sym(coords.x(x, name), group), module.weight, module.bias)
        return run
    if arm == "svdquant":
        s, up, down = bank["smoothing"], bank["up"], bank["down"]

        def run(x):
            n = x.float() / s
            main = F.linear(sq.q_sym(n, group), module.weight, module.bias)
            return main + F.linear(F.linear(n.to(torch.bfloat16), down), up)
        return run
    if arm == "viditq":
        mask, signs = bank["smoothing"], bank["signs"]

        def run(x):
            return F.linear(sq.q_aff(sq.rotate(x.float() * mask, signs), group), module.weight, module.bias)
        return run
    raise ValueError(arm)


def install_arm(modules, index_of, bank_dir, arm, cfg, group, device):
    """Decode every weight of `arm` into its Linear and replace the forwards. Returns the set of Linears seen by a query."""
    shapes = {n: tuple(m.weight.shape) for n, m in modules.items()}
    coords = build_coords(cfg, arm, shapes, device) if arm in ROTATED else None
    seen = set()
    for name, module in modules.items():
        bank = torch.load(bank_dir / f"{index_of[name]:04d}.pt", map_location=device, weights_only=True)
        if bank["module"] != name or tuple(bank["shape"]) != tuple(module.weight.shape):
            raise RuntimeError(f"Bank of {arm} does not match {name}")
        with torch.no_grad():
            module.weight.copy_(sq.decode(bank).to(module.weight.dtype))
        run = forward_for(arm, module, bank, coords, name, group)

        def counted(x, _run=run, _name=name):
            seen.add(_name)
            return _run(x)
        module.forward = counted
    return seen
