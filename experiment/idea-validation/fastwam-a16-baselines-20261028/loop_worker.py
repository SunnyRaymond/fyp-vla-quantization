"""LIBERO-Plus 280 closed loop for one A16 arm: dense dequantized BF16 weights on the cache + CUDA Graph runtime.

Startup per worker: model without T5 + T5 cache must reproduce the full-T5 BF16 case-2000 action/future bitwise;
then the arm's weights are installed and the case-2000 eager and cache+Graph outputs must be bitwise equal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import time
from types import SimpleNamespace

from common import (TARGETS, bf16_runtime, capture, guard, install_t5_cache, max_abs, read, results, save)


class CountingLinears:
    """Plain BF16 F.linear with the counters graph_runtime expects from engine banks (bench_worker 26028322)."""
    def __init__(self, modules):
        self.modules, self.calls, self.seen = modules, Counter(), set()
        self.diagnostics, self.track, self.record = False, True, None
        for name, module in modules.items():
            module.forward = self._forward(name, module)

    def _forward(self, name, module):
        import torch.nn.functional as F

        def run(x):
            if self.track:
                self.calls[name] += 1
                self.seen.add(name)
            return F.linear(x, module.weight, module.bias)
        return run


def install_weights(p, runtime, arm):
    import torch
    root = results(p) / "quantized" / arm
    if not read(root / "COMPLETE.json").get("complete"):
        raise RuntimeError(f"{arm} quantization is incomplete")
    modules, seen = runtime.q.modules, set()
    for i in range(TARGETS):
        saved = torch.load(root / "weights" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
        module = modules[saved["module"]]
        w = saved["weight"]
        if w.dtype != torch.bfloat16 or w.shape != module.weight.shape or not bool(torch.isfinite(w).all()):
            raise RuntimeError(f"Invalid stored weight for {saved['module']}")
        module.weight.data.copy_(w.to(module.weight.device))
        seen.add(saved["module"])
    if seen != set(modules):
        raise RuntimeError("Stored weights do not cover every target Linear exactly once")


def build_runtime(p, arm, out):
    import torch
    import condition_cache
    import graph_runtime
    prep = results(p) / "prep"
    runtime = bf16_runtime(out, text_encoder=False)
    install_t5_cache(runtime, torch.load(prep / "t5_cache.pt", map_location="cpu", weights_only=True))
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    action, future = capture(p, runtime)
    t5 = dict(action=max_abs(action, reference["action"]), future=max_abs(future, reference["future"]))
    if t5["action"] != 0 or t5["future"] != 0:
        raise RuntimeError(f"T5-cache BF16 anchor differs from the full-T5 anchor: {t5}")
    install_weights(p, runtime, arm)
    eager_action, eager_future = capture(p, runtime)
    banks = CountingLinears(runtime.q.modules)
    runtime.banks = banks
    cache = condition_cache.install(runtime.model, diagnostics=True, stable_buffers=True)
    optimization = SimpleNamespace(cache=cache, graph=None, compiled={"enabled": False, "regions": {}})
    runtime.optimization = optimization
    graph_runtime.install(runtime, optimization, compiled=False)
    graph_action, graph_future = capture(p, runtime)
    regions = optimization.graph.report()["regions"]
    engine = dict(action=max_abs(graph_action, eager_action), future=max_abs(graph_future, eager_future))
    if (engine["action"] != 0 or engine["future"] != 0 or len(banks.seen) != TARGETS
            or set(regions) != {"video_step", "video_prefill", "action_step"}
            or cache.metadata()["diagnostic_mismatches"]):
        raise RuntimeError(f"Engine parity failed: {engine}, regions {sorted(regions)}, seen {len(banks.seen)}")
    cache.diagnostics = False
    for slot in cache._slots.values():
        slot["input"] = None
    optimization.graph.reset_metrics()
    banks.calls.clear()
    damage = eager_action - reference["action"]
    props = torch.cuda.get_device_properties(0)
    save(out / "startup.json", dict(
        protocol=p["protocol"], arm=arm, pbs_jobid=os.environ["PBS_JOBID"], passed=True,
        t5_cache_vs_full_t5=t5, cache_graph_vs_eager=engine, gpu=props.name, gpu_uuid=str(props.uuid),
        anchor_action_vs_bf16=dict(max_abs=float(damage.abs().max()),
                                   first10_motor_rmse=float(damage[:10, :6].pow(2).mean().sqrt())),
        anchor_future_nmse_vs_bf16=float((eager_future - reference["future"]).pow(2).sum()
                                         / reference["future"].pow(2).sum())))
    torch.cuda.empty_cache()
    return runtime


def episode(runtime, row, index, arm):
    """Same seeds, settling, caps and 10-action replans as fastwam-plus-engine-eval-20261027."""
    import numpy as np
    import torch
    import runner as pilot
    started = time.monotonic()
    obs = runtime.task(row, index)
    cap = 700 if row["suite"] == "libero_10" else 400
    steps, replans, success, call_seconds = 0, 0, False, []
    while steps < cap and not success:
        datum = runtime.datum(obs)
        torch.cuda.synchronize()
        begin = time.monotonic()
        action, _ = runtime.infer(datum, pilot.seed_for(index, replans), measure=False)
        torch.cuda.synchronize()
        call_seconds.append(time.monotonic() - begin)
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
    return dict(
        arm=arm, variant_id=str(row["variant_id"]), index=index, suite=row["suite"],
        dimension=row["dimension"], task_id=int(row["task_id"]), success=success,
        termination="success" if success else "control_step_timeout", steps=steps, replans=replans,
        env_seed=pilot.seed_for(index, environment=True), sampler_seed=pilot.seed_for(index),
        episode_seconds=time.monotonic() - started, policy_call_seconds=call_seconds,
        pbs_jobid=os.environ["PBS_JOBID"])


def claim(root, worker):
    import fcntl
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue, tasks = read(root / "task_queue.json"), read(root / "tasks.json")
        if queue["next"] >= len(tasks):
            return None
        index = tasks[queue["next"]]
        queue["next"] += 1
        queue["claims"].append(dict(index=index, worker=worker, epoch=time.time()))
        save(root / "task_queue.json", queue)
        return index


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    if args.arm not in p["arms"]:
        raise ValueError(args.arm)
    rows = read(p["panel"]["manifest"])["variants"]
    if len(rows) != 280:
        raise RuntimeError("Expected the frozen 280-variant manifest")
    root = results(p) / "loops" / args.arm
    episodes = root / "episodes"
    if args.init:
        done = {int(f.stem) for f in episodes.glob("*.json")}
        pending = sorted((s for s in range(280) if s not in done), key=lambda s: (rows[s]["suite"] != "libero_10", s))
        save(root / "tasks.json", pending)
        save(root / "task_queue.json", dict(next=0, claims=[]))
        print(f"QUEUE arm={args.arm} pending={len(pending)} done={len(done)}", flush=True)
        return
    out = root / f"worker_{args.worker}"
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    started = time.monotonic()
    runtime = build_runtime(p, args.arm, out)
    load_seconds, completed = time.monotonic() - started, []
    try:
        while (index := claim(root, args.worker)) is not None:
            save(out / "progress.json", dict(phase="episode", index=index, completed=len(completed),
                                             updated_epoch=time.time()))
            record = episode(runtime, rows[index], index, args.arm)
            record["worker"] = args.worker
            save(episodes / f"{index}.json", record)
            completed.append(index)
            print(f"EPISODE arm={args.arm} worker={args.worker} index={index} "
                  f"success={record['success']} steps={record['steps']}", flush=True)
    finally:
        if runtime.env is not None:
            runtime.env.close()
    save(out / "summary.json", dict(protocol=p["protocol"], arm=args.arm, pbs_jobid=os.environ["PBS_JOBID"],
         complete=True, worker=args.worker, episodes=completed, load_and_anchor_seconds=load_seconds))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
