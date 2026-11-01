"""840-episode closed loop of one RAVQ arm on one worker slot, on the A16 runtime (fastwam-a16-baselines
loop_worker), as in fastwam-plus-panelc-20261031's panel_worker.

The A16 build_runtime runs unchanged against this job's results root: model without T5 + the merged A/B/C T5 cache must
reproduce the full-T5 BF16 case-2000 anchor bitwise; then the arm's dense BF16 weights (and calibrated biases) go in
(fastwam-guided-hessian-20261028 results/quantized/<arm>), the case-2000 anchor must reproduce the arm's reference
(first10 motor RMSE against the BF16 anchor, relative CHECK_REL: a calibrated arm's C80 fit row catches a bias that did
not land; an uncalibrated arm's panel-A loop startup ties B + C to the same weights as its A), and eager vs cache +
CUDA Graph must be bitwise equal. Episodes are claimed with an atomic mkdir of loops/<arm>/claims/<index>; the slot
index (A 0-279, B 280-559, C 560-839) drives the env and sampler seeds exactly as in panels A, B, C.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import loop_worker as lw
from common import TARGETS, guard, read, results, save
import panel

SETTLING = "Task completed during settling"
CHECK_REL = 1e-4


def arm_rows(p, arm):
    return [row for name in p["arms"][arm]["panels"] for row in panel.load(p, name)]


def task_order(rows):
    """Longest suite first, as in panel A, so the tail of each queue is short episodes."""
    return sorted(rows, key=lambda r: (r["suite"] != "libero_10", r["index"]))


def wait_t5(p, poll=10.0):
    prep, limit = results(p) / "prep", p["schedule"]["t5_wait_seconds"]
    start = time.monotonic()
    while not (prep / "T5_COMPLETE.json").exists():
        if (prep / "T5_FAILED.json").exists():
            raise RuntimeError("T5 cache merge failed (prep/T5_FAILED.json)")
        if time.monotonic() - start > limit:
            raise TimeoutError(f"T5 cache not ready after {limit}s")
        time.sleep(poll)
    return time.monotonic() - start


def anchor_reference(p, arm):
    """first10 motor RMSE of the case-2000 anchor that the installed arm must reproduce: a calibrated arm's C80 fit row,
    an uncalibrated RAVQU arm's own panel-A loop startup (same A16 runtime)."""
    if p["arms"][arm].get("uncalibrated"):
        path = Path(p["panel_a"]["ravq_loops"]) / arm / "worker_0" / "startup.json"
        start = read(path)
        if not start.get("passed") or start.get("arm") != arm:
            raise RuntimeError(f"{path} is not a passed startup of {arm}")
        return start["anchor_action_vs_bf16"]["first10_motor_rmse"]
    pa = read(p["a16"]["protocol"])
    fit_path = Path(p["ravqc"]["root"]) / "results" / "ravqc" / arm / "fit_after.json"
    return next(r for r in read(fit_path)["rows"] if r["case_id"] == pa["panel"]["anchor_case"])["first10motor_rmse"]


def install_arm(p, runtime, arm, out):
    """Weights (+ biases of a calibrated arm), then the case-2000 anchor check (anchor_reference)."""
    import torch
    root = Path(p["ravqc"]["root"]) / "results" / "quantized" / arm
    if not read(root / "COMPLETE.json").get("complete"):
        raise RuntimeError(f"{arm} is incomplete")
    uncalibrated = bool(p["arms"][arm].get("uncalibrated"))
    modules, seen, biases = runtime.q.modules, set(), 0
    for i in range(TARGETS):
        saved = torch.load(root / "weights" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        module = modules[saved["module"]]
        w = saved["weight"]
        if w.dtype != torch.bfloat16 or w.shape != module.weight.shape or not bool(torch.isfinite(w).all()):
            raise RuntimeError(f"Invalid stored weight for {saved['module']}")
        module.weight.data.copy_(w.to(module.weight.device))
        if "bias" in saved:
            if uncalibrated:
                raise RuntimeError(f"Uncalibrated {arm} stores a bias for {saved['module']}")
            b = saved["bias"]
            if module.bias is None or b.shape != module.bias.shape or b.dtype != module.bias.dtype:
                raise RuntimeError(f"Invalid stored bias for {saved['module']}")
            module.bias.data.copy_(b.to(module.bias.device))
            biases += 1
        seen.add(saved["module"])
    if seen != set(modules):
        raise RuntimeError("Stored weights do not cover every target Linear exactly once")
    pa = read(p["a16"]["protocol"])
    reference = torch.load(results(p) / "prep" / "bf16_anchor.pt", map_location="cpu", weights_only=True)["action"]
    action, _ = lw.capture(dict(pa, remote_root=p["remote_root"]), runtime)
    want = anchor_reference(p, arm)
    got = float((action - reference)[:10, :6].pow(2).mean().sqrt())
    check = dict(arm=arm, biases=biases, first10_motor_rmse=got, reference=want, limit=CHECK_REL,
                 reference_source="panel-A loop startup" if uncalibrated else "C80 fit row")
    check["rel"] = abs(got - want) / max(abs(want), 1e-30)
    check["passed"] = check["rel"] <= CHECK_REL
    save(out / "install_check.json", check)
    if not check["passed"]:
        raise RuntimeError(f"Arm install check failed: {check}")


def patch(p, out):
    """Install the arm (weights, biases when calibrated, anchor check) and wait for the T5 cache after loading."""
    original_runtime = lw.bf16_runtime
    waited = {}

    def install_weights(_p, runtime, arm):
        install_arm(p, runtime, arm, out)

    def bf16_runtime(out_dir, text_encoder=True):
        runtime = original_runtime(out_dir, text_encoder=text_encoder)
        waited["t5_wait_seconds"] = wait_t5(p)
        return runtime

    lw.install_weights, lw.bf16_runtime = install_weights, bf16_runtime
    return waited


def claim(root, order, start, worker):
    """Next unclaimed slot at or after position start in order; returns (row, next_start) or (None, len)."""
    claims = root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for k in range(start, len(order)):
        target = claims / str(order[k]["index"])
        try:
            target.mkdir()
        except FileExistsError:
            continue
        (target / f"worker_{worker}").touch()
        return order[k], k + 1
    return None, len(order)


def run_episode(runtime, row, arm):
    try:
        return lw.episode(runtime, row, row["index"], arm)
    except RuntimeError as exc:
        if SETTLING not in str(exc):
            raise
        if runtime.env is not None:
            runtime.env.close()
            runtime.env = None
        return dict(arm=arm, variant_id=str(row["variant_id"]), index=row["index"], suite=row["suite"],
                    dimension=row["dimension"], task_id=int(row["task_id"]), success=False, invalid=True,
                    termination="settling_success", steps=0, replans=0, episode_seconds=0.0, policy_call_seconds=[],
                    pbs_jobid=os.environ["PBS_JOBID"])


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    if args.arm not in p["arms"]:
        raise ValueError(args.arm)
    pa = read(p["a16"]["protocol"])
    order = task_order(arm_rows(p, args.arm))
    root = results(p) / "loops" / args.arm
    out = root / f"worker_{args.worker}"
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    started = time.monotonic()
    waited = patch(p, out)
    runtime = lw.build_runtime(dict(pa, remote_root=p["remote_root"], protocol=p["protocol"]), args.arm, out)
    load_seconds, completed, position = time.monotonic() - started, [], 0
    try:
        while True:
            row, position = claim(root, order, position, args.worker)
            if row is None:
                break
            save(out / "progress.json", dict(phase="episode", index=row["index"], completed=len(completed),
                                             updated_epoch=time.time()))
            record = run_episode(runtime, row, args.arm)
            record["worker"], record["host"] = args.worker, os.uname().nodename
            save(root / "episodes" / f"{row['index']}.json", record)
            completed.append(row["index"])
            print(f"EPISODE arm={args.arm} worker={args.worker} index={row['index']} success={record['success']} "
                  f"steps={record['steps']}" + (" invalid" if record.get("invalid") else ""), flush=True)
    finally:
        if runtime.env is not None:
            runtime.env.close()
    save(out / "summary.json", dict(protocol=p["protocol"], arm=args.arm, pbs_jobid=os.environ["PBS_JOBID"],
                                    complete=True, worker=args.worker, host=os.uname().nodename, episodes=completed,
                                    load_and_anchor_seconds=load_seconds, **waited))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
