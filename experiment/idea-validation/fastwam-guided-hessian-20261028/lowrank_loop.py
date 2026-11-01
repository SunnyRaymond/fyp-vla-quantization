"""LIBERO-Plus 280 closed loop of a low-rank arm (lowrank.json "loop"): the A16 baselines' loop_worker with the host's
dense W2 weights and the arm's BF16 factors on every target Linear, y = x W^T + b + (x V^T) U^T, in one of three forms
(--lr-mode) in the eager anchor and inside the cache + CUDA Graph regions:
  branch  lowrank_worker's own computation (branch in FP32 from the BF16 factors, cast to the activation dtype, added to
          the BF16 output); bitwise the C80 fit, 614 extra FP32 passes per forward (26066178: policy call 0.745 s vs 0.505 s);
  fused   BF16 x V^T, then one in-place addmm of (x V^T) U^T into the BF16 output (FP32 accumulation);
  merged  W' = bf16(W + U V), a plain BF16 Linear like every A16 arm (the baselines also run dense dequantized weights).
(--lr-mode host installs the host weights alone, for latency references.)

Before the branch goes on, the host weights alone must reproduce the host's own closed-loop anchor (first10 motor RMSE of
case 2000 against the BF16 anchor); with the exact branch, the anchor must reproduce the arm's C80 fit row for case 2000.
A fused or merged install must then reproduce the exact branch's change of the anchor to within FAST_REL: first10 motor
RMSE(fast - branch) / RMSE(branch - host) and ||future_fast - future_branch|| / ||future_branch - future_host||. All of it
is written to --lr-check. Arguments: loop_worker's, plus --lr-mode, --lr-host (host weights directory), --lr-factors
(factors.pt), --lr-host-anchor (the host loop's startup.json), --lr-fit (the arm's fit.json) and --lr-check; passed on
the command line because run_python starts the container with --cleanenv."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import loop_worker as lw
from common import TARGETS, read, results, save

CHECK_REL = 1e-4
FAST_REL = 0.05
MODES = ("branch", "fused", "merged", "host")
FORWARD, CFG = {}, {}


def branch(x, u, v):
    import torch.nn.functional as F
    return F.linear(F.linear(x.float(), v), u).to(x.dtype)


def branched(module, u, v):
    import torch.nn.functional as F
    return lambda x: F.linear(x, module.weight, module.bias) + branch(x, u, v)


def fused(module, v, ut):
    """v: (r, in) and ut: (r, out), both in the activation dtype."""
    import torch.nn.functional as F

    def run(x):
        x2 = x.reshape(-1, x.shape[-1])
        y = F.linear(x2, module.weight, module.bias)
        y.addmm_(F.linear(x2, v), ut)
        return y.reshape(*x.shape[:-1], y.shape[-1])
    return run


def plain(module):
    import torch.nn.functional as F
    return lambda x: F.linear(x, module.weight, module.bias)


def motor_rmse(a, b):
    return float((a - b)[:10, :6].pow(2).mean().sqrt())


def distance(a, b):
    return float((a - b).pow(2).sum().sqrt())


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def install(modules, forwards):
    for name, module in modules.items():
        FORWARD[name] = module.forward = forwards[name]


def install_weights(p, runtime, arm):
    import torch
    mode = CFG.get("lr_mode") or "branch"
    if mode not in MODES:
        raise ValueError(f"Unknown low-rank mode: {mode}")
    host = Path(CFG["lr_host"])
    if not read(host / "COMPLETE.json").get("complete"):
        raise RuntimeError("Host quantization is incomplete")
    modules, seen = runtime.q.modules, set()
    for i in range(TARGETS):
        saved = torch.load(host / "weights" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        module = modules[saved["module"]]
        w = saved["weight"]
        if w.dtype != torch.bfloat16 or w.shape != module.weight.shape or not bool(torch.isfinite(w).all()):
            raise RuntimeError(f"Invalid stored weight for {saved['module']}")
        module.weight.data.copy_(w.to(module.weight.device))
        seen.add(saved["module"])
    factors = {} if mode == "host" else torch.load(CFG["lr_factors"], map_location="cpu", weights_only=True)
    if seen != set(modules) or (mode != "host" and set(factors) != set(modules)):
        raise RuntimeError("Host weights and factors must cover every target Linear exactly once")
    reference = torch.load(results(p) / "prep" / "bf16_anchor.pt", map_location="cpu", weights_only=True)["action"]
    install(modules, {name: plain(module) for name, module in modules.items()})
    host_action, host_future = lw.capture(p, runtime)
    anchor = read(CFG["lr_host_anchor"])["anchor_action_vs_bf16"]["first10_motor_rmse"]
    check = dict(mode=mode, host_first10_motor_rmse=motor_rmse(host_action, reference), host_loop_anchor=anchor,
                 limit=CHECK_REL)
    check["host_rel"] = rel(check["host_first10_motor_rmse"], anchor)
    passed = check["host_rel"] <= CHECK_REL
    if mode != "host":
        exact, fast = {}, {}
        for name, module in modules.items():
            u = factors[name]["U"].to(module.weight.device, torch.float32)
            v = factors[name]["V"].to(module.weight.device, torch.float32)
            if (u.shape != (module.weight.shape[0], v.shape[0]) or v.shape[1] != module.weight.shape[1]
                    or factors[name]["U"].dtype != torch.bfloat16 or not bool(torch.isfinite(u).all() & torch.isfinite(v).all())):
                raise RuntimeError(f"Invalid factors for {name}")
            exact[name] = branched(module, u, v)
            if mode == "fused":
                fast[name] = fused(module, v.to(module.weight.dtype), u.t().contiguous().to(module.weight.dtype))
            elif mode == "merged":
                fast[name] = (u, v)
        install(modules, exact)
        arm_action, arm_future = lw.capture(p, runtime)
        fit = next(r for r in read(CFG["lr_fit"])["rows"] if r["case_id"] == p["panel"]["anchor_case"])
        check.update(arm_first10_motor_rmse=motor_rmse(arm_action, reference), arm_fit_row=fit["first10motor_rmse"])
        check["arm_rel"] = rel(check["arm_first10_motor_rmse"], fit["first10motor_rmse"])
        passed = passed and check["arm_rel"] <= CHECK_REL
        if mode == "merged":
            for name, (u, v) in fast.items():
                w = modules[name].weight
                w.data.copy_((w.float() + u @ v).to(w.dtype))
            fast = {name: plain(module) for name, module in modules.items()}
        if mode in ("fused", "merged"):
            install(modules, fast)
            fast_action, fast_future = lw.capture(p, runtime)
            check.update(
                fast_first10_motor_rmse=motor_rmse(fast_action, reference), fast_limit=FAST_REL,
                fast_action_rel=motor_rmse(fast_action, arm_action) / max(motor_rmse(arm_action, host_action), 1e-30),
                fast_future_rel=distance(fast_future, arm_future) / max(distance(arm_future, host_future), 1e-30))
            passed = passed and check["fast_action_rel"] <= FAST_REL and check["fast_future_rel"] <= FAST_REL
    check["passed"] = passed
    save(Path(CFG["lr_check"]), check)
    if not check["passed"]:
        raise RuntimeError(f"Low-rank install check failed: {check}")


def _forward(self, name, module):
    run = FORWARD[name]

    def counted(x):
        if self.track:
            self.calls[name] += 1
            self.seen.add(name)
        return run(x)
    return counted


def configure(argv):
    """Takes the --lr-* arguments, patches loop_worker, returns the rest."""
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("--lr-host", "--lr-factors", "--lr-host-anchor", "--lr-fit", "--lr-check"):
        parser.add_argument(name)
    parser.add_argument("--lr-mode", choices=MODES, default="branch")
    known, rest = parser.parse_known_args(argv)
    CFG.update(vars(known))
    lw.install_weights = install_weights
    lw.CountingLinears._forward = _forward
    return rest


def main(argv=None):
    rest = configure(sys.argv[1:] if argv is None else argv)
    sys.argv = [sys.argv[0]] + rest
    lw.main()


if __name__ == "__main__":
    main()
