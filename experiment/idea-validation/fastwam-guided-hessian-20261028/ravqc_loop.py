"""LIBERO-Plus 280 closed loop of a calibrated RAVQ arm (ravqc.json loop_job): the A16 baselines' loop_worker with the
arm's dense BF16 weights and its calibrated biases (stored next to the weights) installed, on the same cache + CUDA Graph
runtime as every A16 arm. Before the loop the anchor query (case 2000) with the arm installed must reproduce the arm's
C80 fit row for case 2000 (first10 motor RMSE against the BF16 anchor, relative CHECK_REL), which fails if a bias did not
land. Arguments: loop_worker's, plus --rc-fit (the arm's fit_after.json) and --rc-check (the check written here); passed
on the command line because run_python starts the container with --cleanenv."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import loop_worker as lw
from common import TARGETS, read, results, save

CHECK_REL = 1e-4
CFG = {}


def install_weights(p, runtime, arm):
    import torch
    root = results(p) / "quantized" / arm
    if not read(root / "COMPLETE.json").get("complete"):
        raise RuntimeError(f"{arm} is incomplete")
    modules, seen, biases = runtime.q.modules, set(), 0
    for i in range(TARGETS):
        saved = torch.load(root / "weights" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        module = modules[saved["module"]]
        w = saved["weight"]
        if w.dtype != torch.bfloat16 or w.shape != module.weight.shape or not bool(torch.isfinite(w).all()):
            raise RuntimeError(f"Invalid stored weight for {saved['module']}")
        module.weight.data.copy_(w.to(module.weight.device))
        if "bias" in saved:
            b = saved["bias"]
            if module.bias is None or b.shape != module.bias.shape or b.dtype != module.bias.dtype:
                raise RuntimeError(f"Invalid stored bias for {saved['module']}")
            module.bias.data.copy_(b.to(module.bias.device))
            biases += 1
        seen.add(saved["module"])
    if seen != set(modules):
        raise RuntimeError("Stored weights do not cover every target Linear exactly once")
    reference = torch.load(results(p) / "prep" / "bf16_anchor.pt", map_location="cpu", weights_only=True)["action"]
    action, _ = lw.capture(p, runtime)
    fit = next(r for r in read(CFG["rc_fit"])["rows"] if r["case_id"] == p["panel"]["anchor_case"])
    got = float((action - reference)[:10, :6].pow(2).mean().sqrt())
    check = dict(arm=arm, biases=biases, first10_motor_rmse=got, fit_row=fit["first10motor_rmse"], limit=CHECK_REL)
    check["rel"] = abs(got - fit["first10motor_rmse"]) / max(abs(fit["first10motor_rmse"]), 1e-30)
    check["passed"] = check["rel"] <= CHECK_REL
    save(Path(CFG["rc_check"]), check)
    if not check["passed"]:
        raise RuntimeError(f"Calibrated arm install check failed: {check}")


def configure(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rc-fit", required=True)
    parser.add_argument("--rc-check", required=True)
    known, rest = parser.parse_known_args(argv)
    CFG.update(vars(known))
    lw.install_weights = install_weights
    return rest


def main(argv=None):
    rest = configure(sys.argv[1:] if argv is None else argv)
    sys.argv = [sys.argv[0]] + rest
    lw.main()


if __name__ == "__main__":
    main()
