"""Read-only gradient and Adam direction helpers for Fast-WAM."""

import math


def gradient_summary(torch, grads_a, grads_z, groups):
    """Summarize paired flattened gradients overall and by ratio/channel."""
    if len(grads_a) != len(grads_z):
        raise ValueError("gradient lists must have equal length")
    pairs = {"overall": [], "ratio": [], "channel": []}
    offset = 0
    for group in groups:
        name = group["name"]
        if name not in ("ratio", "channel"):
            raise ValueError("group name must be ratio or channel")
        end = offset + len(group["params"])
        if end > len(grads_a):
            raise ValueError("groups contain more parameters than gradients")
        part = list(zip(grads_a[offset:end], grads_z[offset:end]))
        pairs[name].extend(part)
        pairs["overall"].extend(part)
        offset = end
    if offset != len(grads_a):
        raise ValueError("groups and gradients must have matching flattened lengths")

    def summarize(items):
        live = [g for pair in items for g in pair if g is not None]
        dtype = torch.float64 if any(g.dtype == torch.float64 for g in live) else torch.float32
        aa, zz, dots, finite_a, finite_z = [], [], [], [], []
        connected_a = connected_z = 0
        for a, z in items:
            if a is not None:
                connected_a += 1
                aa.append(a.detach().to(dtype=dtype).square().sum())
                finite_a.append(torch.isfinite(a).all())
            if z is not None:
                connected_z += 1
                zz.append(z.detach().to(dtype=dtype).square().sum())
                finite_z.append(torch.isfinite(z).all())
            if a is not None and z is not None:
                if a.shape != z.shape:
                    raise ValueError("paired gradients must have matching shapes")
                dots.append((a.detach().to(dtype=dtype) * z.detach().to(dtype=dtype)).sum())

        def total(values):
            return torch.stack(values).sum() if values else None

        sa, sz, sd = total(aa), total(zz), total(dots)
        na = float(torch.sqrt(sa).item()) if sa is not None else 0.0
        nz = float(torch.sqrt(sz).item()) if sz is not None else 0.0
        dot = float(sd.item()) if sd is not None else 0.0
        return {
            "norm_a": na, "norm_z": nz, "dot": dot,
            "cosine": dot / (na * nz) if na and nz else 0.0,
            "finite": {
                "a": bool(torch.stack(finite_a).all().item()) if finite_a else True,
                "z": bool(torch.stack(finite_z).all().item()) if finite_z else True,
            },
            "connected": {"a": connected_a, "z": connected_z, "total": len(items)},
        }

    return {name: summarize(items) for name, items in pairs.items()}


def adam_delta(torch, gradients, param_groups, state_dict=None):
    """Return next-step deltas without mutation; saved IDs map by group/list order.

    Supports standard Adam with zero weight decay, no AMSGrad, and no maximize.
    """
    saved_groups = state_dict["param_groups"] if state_dict is not None else None
    saved_state = state_dict["state"] if state_dict is not None else {}
    if len(gradients) != sum(len(group["params"]) for group in param_groups):
        raise ValueError("gradient list and parameter groups must have matching lengths")
    if saved_groups is not None and len(saved_groups) != len(param_groups):
        raise ValueError("live and saved optimizer group counts differ")
    result = []
    index = 0
    for gi, live_group in enumerate(param_groups):
        params = list(live_group["params"])
        cfg = saved_groups[gi] if saved_groups is not None else live_group
        ids = cfg.get("params", []) if saved_groups is not None else [None] * len(params)
        if len(ids) != len(params):
            raise ValueError("saved and live parameter group lengths differ")
        lr = float(cfg.get("lr", 1e-3))
        beta1, beta2 = cfg.get("betas", (0.9, 0.999))
        eps = float(cfg.get("eps", 1e-8))
        if cfg.get("amsgrad", False) or cfg.get("maximize", False) or cfg.get("weight_decay", 0) != 0:
            raise NotImplementedError("only standard Adam defaults are supported")
        for param, pid in zip(params, ids):
            grad = gradients[index]
            index += 1
            if grad is None:
                result.append(torch.zeros_like(param))
                continue
            if grad.shape != param.shape:
                raise ValueError("gradient and parameter shapes differ")
            state = saved_state.get(pid, {}) if state_dict is not None else {}
            if state:
                m0 = state["exp_avg"].to(device=grad.device, dtype=grad.dtype)
                v0 = state["exp_avg_sq"].to(device=grad.device, dtype=grad.dtype)
                step = state["step"]
                step = int(step.item()) if torch.is_tensor(step) else int(step)
            else:
                m0, v0, step = torch.zeros_like(param), torch.zeros_like(param), 0
            g = grad.detach()
            m = m0.mul(beta1).add(g, alpha=1.0 - beta1)
            v = v0.mul(beta2).add(g.square(), alpha=1.0 - beta2)
            step += 1
            denom = torch.sqrt(v).div(math.sqrt(1.0 - beta2 ** step)).add(eps)
            result.append(m.div(denom).mul(-lr / (1.0 - beta1 ** step)))
    return result
