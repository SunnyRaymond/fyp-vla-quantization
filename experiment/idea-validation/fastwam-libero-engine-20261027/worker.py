"""Four persistent inference workers; standard LIBERO and frozen C+decoder+buffer."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import socket
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if (not os.environ.get("PBS_JOBID") or "login" in host.lower() or not nodefile
            or host not in {x.split(".")[0] for x in Path(nodefile).read_text().split()}):
        raise RuntimeError("Approved PBS compute allocation required")


def task_plan(p):
    suites = p["evaluation"]["suites"]
    tasks = [{"suite": "libero_10", "task_id": task} for task in range(10)]
    tasks += [{"suite": suite, "task_id": task} for task in range(10)
              for suite in suites if suite != "libero_10"]
    assert len(tasks) == 40 and len({(r["suite"], r["task_id"]) for r in tasks}) == 40
    return tasks


def claim(root, worker):
    import fcntl
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = read(root / "task_queue.json")
        tasks = read(root / "tasks.json")
        index = queue["next"]
        if index >= len(tasks):
            return None
        queue["next"] += 1
        row = dict(tasks[index], queue_index=index, worker=worker)
        queue["claims"].append(row)
        save(root / "task_queue.json", queue)
        return row


def build_runtime(p, out):
    import numpy as np
    import torch
    import abc_runtime
    import calibration_source
    import decoder_runtime
    import export_runtime
    import fused_vq
    from profile_worker import _capture_anchor
    from reference_profile import freeze_numerics

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated GPU per persistent worker is required")
    props = torch.cuda.get_device_properties(0)
    mask = os.environ["CUDA_VISIBLE_DEVICES"]
    if str(props.uuid).removeprefix("GPU-").lower() != mask.removeprefix("GPU-").lower():
        raise RuntimeError("Worker GPU differs from its assigned PBS GPU")
    freeze_numerics(torch)
    ep = read(Path(p["engine"]["source_artifacts"]) / "protocol.json")
    bundle = export_runtime.load_bundle(p["engine"]["bundle"])
    runtime = abc_runtime.make_runtime(out, ep, bundle, direct=True)
    banks = abc_runtime.install(runtime, bundle, "c_buffer")
    del bundle
    tuning = read(Path(p["engine"]["source_results"]) / "tuning.json")
    identity = tuning["checkpoint_identity"]
    if (identity["source_job"] != p["model"]["source_job"]
            or identity["bundle_path"] != p["engine"]["bundle"]
            or tuning["mode"] != "c_buffer"
            or tuning["parity"]["bitwise_passed"] != 614
            or tuning["device"]["name"] != props.name
            or {r["name"] for r in tuning["modules"]} != set(banks.modules)):
        raise RuntimeError("Frozen decoder receipt does not match this checkpoint/GPU/modules")
    decisions = {tuple(r["shape"]): r["selected"] for r in tuning["shapes"]}
    banks.decoder = decoder_runtime.DecoderDispatcher(
        decisions, tuning, out / "reused_decoder.json", fused_vq._decode_kernel)
    gc.collect()
    torch.cuda.empty_cache()
    plan_path = Path(p["engine"]["calibration_plan"])
    row = next(r for r in read(plan_path)["inputs"] if r["case_id"] == 2000)
    observation, metadata = calibration_source.load_observation(np, plan_path.parent, row)
    runtime.description = metadata["description"]
    action, _, future, calls = _capture_anchor(
        runtime, runtime.datum(observation), row["sampler_seeds"][0], torch)
    anchor = torch.load(p["engine"]["anchor_tensor"], map_location="cpu", weights_only=True)
    action_error = float((action.detach().float().cpu() - anchor["action"].float()).abs().max())
    future_error = float((future.float().cpu() - anchor["future"].float()).abs().max())
    if action_error != 0 or future_error != 0 or calls != 1 or len(banks.ever_seen) != 614:
        raise RuntimeError(f"Frozen full action/future anchor failed: {action_error}, {future_error}")
    if any(t.requires_grad for t in runtime.model.parameters()):
        raise RuntimeError("Inference model must be frozen")
    if any(m.weight is not None for m in banks.modules.values()):
        raise RuntimeError("Engine unexpectedly retains dense target weights")
    if set(runtime.optimization.graph.regions) != {"video_step", "video_prefill", "action_step"}:
        raise RuntimeError("Expected all three CUDA Graph regions")
    runtime.optimization.diagnostics = False
    runtime.optimization.reset_metrics()
    from libero.libero import get_libero_path
    expected = Path("/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero")
    if Path(get_libero_path("bddl_files")).resolve() != (expected / "bddl_files").resolve():
        raise RuntimeError("Evaluation must use original LIBERO, not LIBERO-Plus")
    startup = dict(protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"],
        passed=True, source_job=p["model"]["source_job"], engine="c_buffer",
        action_max_abs=action_error, future_max_abs=future_error,
        modules=614, model_loads=1, gpu=props.name, gpu_uuid=str(props.uuid),
        original_libero_root=str(expected), reused_decoder=tuning["pbs_jobid"],
        graphs=sorted(runtime.optimization.graph.regions), installation=runtime.install_metadata)
    save(out / "startup.json", startup)
    del action, future, anchor, observation
    gc.collect()
    torch.cuda.empty_cache()
    return runtime


def run_task(runtime, p, row, out):
    import numpy as np
    import torch
    from libero.libero import get_libero_path
    from fastwam.utils.pytorch_utils import set_global_seed

    seed = p["evaluation"]["seed"]
    set_global_seed(seed, get_worker_init_fn=False)
    random.seed(seed)
    suite_name, task_id = row["suite"], row["task_id"]
    suite = runtime.benchmark.get_benchmark_dict()[suite_name]()
    if suite.get_num_tasks() != 10:
        raise RuntimeError("Expected standard ten-task LIBERO suite")
    task = suite.get_task(task_id)
    runtime.description = task.language
    env, description = runtime.ev.get_libero_env(task, runtime.ev.LIBERO_ENV_RESOLUTION, seed)
    if description != runtime.description:
        raise RuntimeError("Task description differs")
    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
    if len(initial_states) < 10:
        raise RuntimeError("Original initial-state bank lacks ten trials")
    results = []
    try:
        for trial in p["evaluation"]["trial_ids"]:
            save(out / "progress.json", dict(phase="running", task=row, trial=trial,
                 completed=sum(1 for _ in out.glob("episodes/*.json")), updated_epoch=time.time()))
            started = time.monotonic()
            env.reset()
            obs = env.set_init_state(initial_states[trial])
            for _ in range(p["evaluation"]["settling_steps"]):
                obs, _, done, _ = env.step(runtime.ev.get_libero_dummy_action())
                if done:
                    raise RuntimeError("Task completed during settling")
            steps, success, replans = 0, False, 0
            cap = p["evaluation"]["max_steps"][suite_name]
            query_times = []
            while steps < cap and not success:
                datum = runtime.datum(obs)
                torch.cuda.synchronize()
                begin = time.monotonic()
                action, _ = runtime.infer(datum, p["evaluation"]["sampler_seed"], measure=False)
                torch.cuda.synchronize()
                query_times.append(time.monotonic() - begin)
                commands = runtime.command(action)
                if not np.isfinite(commands).all():
                    raise RuntimeError("Non-finite controller commands")
                for command in commands[:min(p["evaluation"]["replan_steps"], cap - steps)]:
                    obs, _, done, _ = env.step(command.tolist())
                    steps += 1
                    if done:
                        success = True
                        break
                replans += 1
            result = dict(protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"],
                worker=row["worker"], suite=suite_name, task_id=task_id, task_name=task.name,
                trial_id=trial, initial_state_index=trial, initial_state_path=str(init_path),
                seed=seed, sampler_seed=p["evaluation"]["sampler_seed"], complete=True,
                success=bool(success), steps=steps, replans=replans,
                episode_seconds=time.monotonic()-started, policy_call_seconds=query_times,
                termination="success" if success else "control_step_timeout", engine="c_buffer")
            save(out / "episodes" / f"{suite_name}_{task_id}_{trial}.json", result)
            results.append(result)
            print(f"EPISODE_COMPLETE worker={row['worker']} suite={suite_name} task={task_id} trial={trial} success={success} steps={steps}", flush=True)
    finally:
        env.close()
    return results


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()
    p = read(args.protocol)
    root = Path(p["remote_root"]) / "results"
    if args.init:
        if (root / "task_queue.json").exists():
            raise RuntimeError("A queue already exists; refuse duplicate evaluation")
        save(root / "tasks.json", task_plan(p))
        save(root / "task_queue.json", dict(next=0, claims=[]))
        return
    if args.worker not in range(4):
        raise ValueError("Expected worker 0..3")
    out = root / f"worker_{args.worker}"
    out.mkdir(parents=True, exist_ok=True)
    save(out / "progress.json", dict(phase="loading", completed=0, updated_epoch=time.time()))
    started = time.monotonic()
    runtime = build_runtime(p, out)
    load_seconds = time.monotonic() - started
    results = []
    try:
        while (row := claim(root, args.worker)) is not None:
            results.extend(run_task(runtime, p, row, out))
    finally:
        runtime.q.close()
    full_report = runtime.optimization.report()
    report = dict(mode=full_report["mode"], logical_coverage=full_report["logical_coverage"],
                  graph_kind=full_report["graph"].get("kind"),
                  graph_regions={name: {k: r.get(k) for k in ("captures", "replays", "capture_target_calls")}
                                 for name, r in full_report["graph"]["regions"].items()},
                  workspace_layout=full_report["workspace_layout"])
    save(out / "summary.json", dict(protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"],
        complete=True, worker=args.worker, episodes=len(results), successes=sum(r["success"] for r in results),
        load_and_anchor_seconds=load_seconds, model_loads=1, engine=report))
    save(out / "progress.json", dict(phase="complete", completed=len(results), updated_epoch=time.time()))
    print(f"WORKER_COMPLETE worker={args.worker} episodes={len(results)}", flush=True)


if __name__ == "__main__":
    main()
