"""LIBERO-Plus panel A closed loop of a LAQ arm on Fast-WAM (eager fake quantization, decoded weights).

Startup: the BF16 model with the LAQ hooks on the T5 cache must reproduce the A16 BF16 case-2000 anchor bitwise; then
the 4-bit weights go in and the arm's allocation is installed with the offline gaze; every block-level Linear and each
K/V site must be hit by the anchor query. The first worker of an arm stores the arm anchor and later workers must
reproduce it bitwise. In an episode the first query uses the offline gaze and every later query the gaze of the query
before it. --smoke stops after the startup checks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

from fw_common import bf16_runtime, capture, guard, install_t5_cache, max_abs, read, save, torch_save
from loopcommon import claim, walltime_seconds


def build(p, m, arm, out, root):
    import torch
    import laq
    c = p["laq"]
    prep = Path(m["a16_prep"])
    alloc = read(Path(p["remote_root"]) / "results" / "fastwam" / "calib" / "allocation.json")
    if not read(Path(p["remote_root"]) / "results" / "fastwam" / "calib" / "COMPLETE.json").get("complete"):
        raise RuntimeError("The calibration is not complete")
    runtime = bf16_runtime(out, text_encoder=False)
    install_t5_cache(runtime, torch.load(prep / "t5_cache.pt", map_location="cpu", weights_only=True))
    device = torch.device("cuda")
    st = laq.State(c["steps"], m["frame_tokens"], m["video_tokens"], read(c["arch"]), m["rotation_seed"], device)
    modules = dict(runtime.q.modules)
    if laq.install(st, runtime.model, modules) != m["targets"]:
        raise RuntimeError("Block-level Linear count differs from the protocol")
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    action, future = capture(m, runtime)
    bf16 = dict(action=max_abs(action, reference["action"]), future=max_abs(future, reference["future"]))
    if bf16["action"] != 0 or bf16["future"] != 0:
        raise RuntimeError(f"BF16 anchor differs from the A16 anchor: {bf16}")
    index_of = {r["module"]: r["index"] for r in read(prep / "modules.json")}
    laq.load_weights(st, modules, index_of, Path(c["bank"]))
    st.seen.clear()
    st.counts.clear()
    st.offline = (torch.tensor(alloc["offline_gaze"]["fut"], device=device),
                  torch.tensor(alloc["offline_gaze"]["obs"], device=device))
    st.watch = True
    st.use(alloc["arms"][arm]["spec"], online=c["online"][arm])
    action, future = capture(m, runtime)
    st.reset()
    counts = dict(st.counts)
    if len(st.seen) != m["targets"] or not all(counts.get(s, 0) > 0 for s in ("text", "observation", "future")):
        raise RuntimeError(f"Coverage failed: {len(st.seen)} Linears, KV sites {counts}")
    runtime.laq = st
    seen = st.seen
    anchor = dict(action=action.float().cpu(), future=future.float().cpu())
    path = root / "anchor.pt"
    if not path.exists():
        torch_save(anchor, path)
    first = torch.load(path, map_location="cpu", weights_only=True)
    if not (torch.equal(anchor["action"], first["action"]) and torch.equal(anchor["future"], first["future"])):
        raise RuntimeError(f"{arm} anchor differs from the arm's first worker")
    damage = anchor["action"] - reference["action"].float()
    check = dict(protocol=p["protocol"], arm=arm, pbs_jobid=os.environ["PBS_JOBID"], bf16_vs_a16_anchor=bf16,
                 linears_seen=len(seen), kv_hits_per_query=counts, allocation=alloc["arms"][arm],
                 anchor_action_max_abs=float(damage.abs().max()),
                 anchor_first10_motor_rmse=float(damage[:10, :6].pow(2).mean().sqrt()),
                 anchor_future_nmse=float((anchor["future"] - reference["future"].float()).pow(2).sum()
                                          / reference["future"].float().pow(2).sum()),
                 peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
    save(out / "startup.json", check)
    torch.cuda.empty_cache()
    return runtime, check


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def episode(runtime, row, index, arm):
    """Same seeds, settling, caps and 10-action replans as the A16 and earlier W4A4 evaluations."""
    import numpy as np
    import torch
    import runner as pilot
    started = time.monotonic()
    runtime.laq.reset()
    obs = runtime.task(row, index)
    cap = 700 if row["suite"] == "libero_10" else 400
    steps, replans, success, calls = 0, 0, False, []
    while steps < cap and not success:
        datum = runtime.datum(obs)
        begin = time.monotonic()
        action, _ = runtime.infer(datum, pilot.seed_for(index, replans), measure=False)
        torch.cuda.synchronize()
        calls.append(time.monotonic() - begin)
        commands = runtime.command(action)
        if not np.isfinite(commands).all():
            raise RuntimeError("Non-finite controller commands")
        for command in commands[:min(10, cap - steps)]:
            obs, _, done, _ = runtime.env.step(command.tolist())
            steps += 1
            if done:
                success = True
                break
        replans += 1
    runtime.env.close()
    runtime.env = None
    return dict(arm=arm, variant_id=str(row["variant_id"]), index=index, suite=row["suite"],
                dimension=row["dimension"], task_id=int(row["task_id"]), success=success,
                termination="success" if success else "control_step_timeout", steps=steps, replans=replans,
                env_seed=pilot.seed_for(index, environment=True), sampler_seed=pilot.seed_for(index),
                episode_seconds=time.monotonic() - started, policy_call_seconds=calls,
                gaze_overlap_previous=_mean([t["previous"] for t in runtime.laq.trace]),
                gaze_overlap_offline=_mean([t["offline"] for t in runtime.laq.trace]),
                pbs_jobid=os.environ["PBS_JOBID"])


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    m, arm, worker = p["models"]["fastwam"], args.arm, args.worker
    if arm not in p["arms"]:
        raise ValueError(arm)
    root = Path(p["remote_root"]) / "results" / "fastwam" / "loops" / arm
    out = root / ("smoke" if args.smoke else f"worker_{worker}")
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    deadline = float(os.environ["FW_JOB_START_EPOCH"]) + walltime_seconds(m["jobs"]["loop"]["walltime"]) - 900
    runtime, check = build(p, m, arm, out, root)
    if args.smoke:
        print(f"SMOKE fastwam {arm} motor_rmse={check['anchor_first10_motor_rmse']:.4f} "
              f"future_nmse={check['anchor_future_nmse']:.4f} peak={check['peak_reserved_gib']:.1f}GiB", flush=True)
        save(out / "progress.json", dict(phase="smoke_passed", updated_epoch=time.time()))
        return
    rows = read(m["panel"]["manifest"])["variants"]
    rows = [dict(r, index=i) for i, r in enumerate(rows)]
    if len(rows) != 280:
        raise RuntimeError("Expected the frozen 280-variant manifest")
    order = sorted(rows, key=lambda r: (r["suite"] != "libero_10", r["index"]))
    completed, position = [], 0
    try:
        while time.time() < deadline - 600:
            row, position = claim(root, order, position, worker)
            if row is None:
                break
            index = row["index"]
            save(out / "progress.json", dict(phase="episode", index=index, completed=len(completed),
                                             updated_epoch=time.time()))
            try:
                record = episode(runtime, row, index, arm)
            except BaseException:
                shutil.rmtree(root / "claims" / str(index), ignore_errors=True)
                raise
            record.update(worker=worker, host=os.uname().nodename)
            save(root / "episodes" / f"{index}.json", record)
            completed.append(index)
            print(f"EPISODE arm={arm} worker={worker} index={index} success={record['success']} "
                  f"steps={record['steps']} seconds={record['episode_seconds']:.0f}", flush=True)
    finally:
        if getattr(runtime, "env", None) is not None:
            runtime.env.close()
    save(out / "summary.json", dict(protocol=p["protocol"], arm=arm, pbs_jobid=os.environ["PBS_JOBID"], worker=worker,
                                    episodes=completed))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
