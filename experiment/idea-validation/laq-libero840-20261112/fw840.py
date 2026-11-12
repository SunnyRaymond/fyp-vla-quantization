"""Standard-LIBERO closed loop (840 episodes) of the LAQ arm on Fast-WAM, on an EEE GPU.

Same model, weights, allocation, hooks, online gaze and CUDA graphs as the panel-A loop (laq-fastwam-20261111,
fastwam_loop.py), only the episodes change: original LIBERO suites, official initial state = trial index.

Startup checks. ASPIRE2A ran on A100s and every anchor was bitwise; on another GPU model cuBLAS picks other kernels, so
the A100 anchors only bound the drift: the BF16 anchor of this GPU must lie within `bf16_drift_fraction` of the A100 LAQ
damage (first-10 motor RMSE and future NMSE), and the LAQ damage on this GPU (LAQ vs this GPU's BF16 anchor) must stay
within `damage_ratio` of the A100 damage. Within one GPU model everything is bitwise again: the first worker stores the
arm anchor per GPU model and every later worker must reproduce it, eager and through the captured graphs.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import random
import time

import l840
from fw_common import bf16_runtime, capture, guard, install_t5_cache, max_abs, read, save, torch_save

MODEL = "fastwam"


def build(P, p, m, arm, out, res):
    import torch
    import laq
    c = p["laq"]
    prep = Path(m["a16_prep"])
    calib = Path(p["remote_root"]) / "results" / "fastwam" / "calib"
    if not read(calib / "COMPLETE.json").get("complete"):
        raise RuntimeError("The calibration is not complete")
    alloc = read(calib / "allocation.json")
    runtime = bf16_runtime(out, text_encoder=False)
    # A16 (A100) entries first; the standard-LIBERO prompts it lacks come from t5prep840.py
    cache = torch.load(res / "t5_libero40.pt", map_location="cpu", weights_only=True)
    cache.update(torch.load(prep / "t5_cache.pt", map_location="cpu", weights_only=True))
    install_t5_cache(runtime, cache)
    device = torch.device("cuda")
    st = laq.State(c["steps"], m["frame_tokens"], m["video_tokens"], read(c["arch"]), m["rotation_seed"], device)
    modules = dict(runtime.q.modules)
    if laq.install(st, runtime.model, modules) != m["targets"]:
        raise RuntimeError("Block-level Linear count differs from the protocol")
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    bf16_action, bf16_future = capture(m, runtime)
    drift = l840.metrics(bf16_action, bf16_future, reference["action"], reference["future"])
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
    anchor = dict(action=action.float().cpu(), future=future.float().cpu())
    damage = l840.metrics(anchor["action"], anchor["future"], bf16_action, bf16_future)
    cfg = P["models"][MODEL]
    a100 = read(cfg["a100_startup"])
    a100_damage = dict(first10_motor_rmse=a100["anchor_first10_motor_rmse"], future_nmse=a100["anchor_future_nmse"])
    a100_anchor = torch.load(cfg["a100_anchor"], map_location="cpu", weights_only=True)
    vs_a100_laq = l840.metrics(anchor["action"], anchor["future"], a100_anchor["action"], a100_anchor["future"])
    g = P["gates"]
    ratio = {k: damage[k] / a100_damage[k] for k in a100_damage}
    gate = dict(drift_ok=all(drift[k] <= g["bf16_drift_fraction"] * a100_damage[k] for k in a100_damage),
                ratio_ok=all(g["damage_ratio"][0] <= ratio[k] <= g["damage_ratio"][1] for k in ratio))
    gpu = torch.cuda.get_device_name(0)
    check = dict(protocol=P["protocol"], model=MODEL, arm=arm, job=l840.owner_tag(), host=os.uname().nodename, gpu=gpu,
                 bf16_vs_a100_bf16=drift, laq_vs_this_gpu_bf16=damage, a100_laq_damage=a100_damage,
                 damage_ratio=ratio, laq_vs_a100_laq=vs_a100_laq, gate=gate, linears_seen=len(st.seen),
                 kv_hits_per_query=counts, allocation=alloc["arms"][arm],
                 peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
    save(out / "startup.json", check)
    if not (gate["drift_ok"] and gate["ratio_ok"]):
        raise RuntimeError(f"Cross-GPU gate failed: drift {drift}, damage ratio {ratio}, A100 damage {a100_damage}")
    path = res / "startup" / f"anchor_{l840.gpu_slug(gpu)}.pt"
    if not path.exists():
        torch_save(anchor, path)
    first = torch.load(path, map_location="cpu", weights_only=True)
    if not (torch.equal(anchor["action"], first["action"]) and torch.equal(anchor["future"], first["future"])):
        raise RuntimeError(f"{arm} anchor differs from the first worker on {gpu}")
    if c.get("graphs"):
        from graphs import Graphs
        st.freeze()
        runtime.graphs = Graphs(st)
        runtime.graphs.install()
        timing = []
        for label in ("capture", "replay"):
            st.reset()
            torch.cuda.synchronize()
            begin = time.monotonic()
            again, future_again = capture(m, runtime)
            torch.cuda.synchronize()
            timing.append(time.monotonic() - begin)
            if not (torch.equal(again.float().cpu(), anchor["action"])
                    and torch.equal(future_again.float().cpu(), anchor["future"])):
                raise RuntimeError(f"{arm}: the graph {label} query differs from the eager anchor")
        st.reset()
        check.update(graphs=dict(capture_seconds=timing[0], replay_seconds=timing[1], regions=len(runtime.graphs.regions),
                                 peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))
    save(out / "startup.json", check)
    torch.cuda.empty_cache()
    return runtime, check


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


class Suites:
    def __init__(self, runtime):
        self.runtime, self.cache = runtime, {}

    def task(self, row):
        name = row["suite"]
        if name not in self.cache:
            self.cache[name] = self.runtime.benchmark.get_benchmark_dict()[name]()
        task = self.cache[name].get_task(int(row["task_id"]))
        if task.name != row["task_name"] or task.language != row["language"]:
            raise RuntimeError(f"Manifest task identity changed: {row} vs {task.name}")
        return task


def episode(runtime, suites, row, B, arm):
    """Official initial state `trial`, 30 no-op settling steps, 10-action replans, the panel-A sampler seeds."""
    import numpy as np
    import torch
    from libero.libero import get_libero_path
    import runner as pilot
    started = time.monotonic()
    runtime.laq.reset()
    task = suites.task(row)
    runtime.description = task.language
    index, env_seed = row["index"], row["env_seed"]
    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    if str(init_path) != row["init_state_path"]:
        raise RuntimeError(f"Initial-state path changed: {init_path}")
    states = torch.load(init_path, map_location="cpu", weights_only=False)
    env, description = runtime.ev.get_libero_env(task, runtime.ev.LIBERO_ENV_RESOLUTION, env_seed)
    runtime.env = env
    if description != task.language:
        raise RuntimeError("Task description differs")
    np.random.seed(env_seed)
    random.seed(env_seed)
    env.reset()
    obs = env.set_init_state(states[int(row["trial"])])
    for _ in range(B["settling_steps"]):
        obs, _, done, _ = env.step(runtime.ev.get_libero_dummy_action())
        if done:
            raise RuntimeError("Task completed during settling; invalid evaluation slot")
    cap = B["max_steps"][row["suite"]]
    steps, replans, success, calls = 0, 0, False, []
    while steps < cap and not success:
        datum = runtime.datum(obs)
        seed = pilot.seed_for(index, replans)
        if seed != row["sampler_seed"] + replans:
            raise RuntimeError("Sampler seed rule differs from the protocol")
        begin = time.monotonic()
        action, _ = runtime.infer(datum, seed, measure=False)
        torch.cuda.synchronize()
        calls.append(time.monotonic() - begin)
        commands = runtime.command(action)
        if not np.isfinite(commands).all():
            raise RuntimeError("Non-finite controller commands")
        for command in commands[:min(B["replan_steps"][MODEL], cap - steps)]:
            obs, _, done, _ = env.step(command.tolist())
            steps += 1
            if done:
                success = True
                break
        replans += 1
    env.close()
    runtime.env = None
    return dict(arm=arm, model=MODEL, index=index, suite=row["suite"], task_id=int(row["task_id"]),
                task_name=row["task_name"], trial=int(row["trial"]), initial_state_index=int(row["trial"]),
                success=success, termination="success" if success else "control_step_timeout", steps=steps,
                replans=replans, env_seed=env_seed, sampler_seed=row["sampler_seed"],
                episode_seconds=time.monotonic() - started, policy_call_seconds=calls,
                gaze_overlap_previous=_mean([t["previous"] for t in runtime.laq.trace]),
                gaze_overlap_offline=_mean([t["offline"] for t in runtime.laq.trace]))


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    P = l840.read(args.protocol)
    cfg = P["models"][MODEL]
    p = read(cfg["laq_protocol"])
    m, arm = p["models"]["fastwam"], cfg["arm"]
    res = l840.results(P, MODEL)
    tag = l840.owner_tag()
    out = res / "workers" / f"{tag['job']}.{tag['restart']}.{args.worker}"
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    deadline = float(os.environ["L840_DEADLINE_EPOCH"])
    # one build at a time per job: two concurrent Fast-WAM loads overflow the 64 GB a 6000 Ada GPU brings
    lock = os.open(os.environ["L840_BUILD_LOCK"], os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        runtime, check = build(P, p, m, arm, out, res)
    finally:
        os.close(lock)
    g = check.get("graphs") or {}
    print(f"STARTUP {MODEL} worker={args.worker} gpu={check['gpu']} drift={check['bf16_vs_a100_bf16']} "
          f"damage={check['laq_vs_this_gpu_bf16']} ratio={check['damage_ratio']} "
          f"replay={g.get('replay_seconds', float('nan')):.2f}s peak={check['peak_reserved_gib']:.1f}GiB", flush=True)
    if args.smoke:
        save(out / "progress.json", dict(phase="smoke_passed", updated_epoch=time.time()))
        return
    rows = l840.manifest_rows(P, MODEL)
    ordered = l840.order(rows)
    suites = Suites(runtime)
    completed = []
    try:
        while time.time() < deadline and (args.max_episodes is None or len(completed) < args.max_episodes):
            row = l840.next_row(P, res, ordered, args.worker)
            if row is None:
                break
            index = row["index"]
            save(out / "progress.json", dict(phase="episode", index=index, completed=len(completed),
                                             updated_epoch=time.time()))
            try:
                record = episode(runtime, suites, row, P["benchmark"], arm)
            except BaseException:
                l840.release(res, index)
                raise
            record.update(worker=args.worker, job=tag, host=os.uname().nodename, gpu=check["gpu"])
            save(res / "episodes" / f"{index}.json", record)
            completed.append(index)
            print(f"EPISODE {MODEL} worker={args.worker} index={index} {row['suite']} task={row['task_id']} "
                  f"trial={row['trial']} success={record['success']} steps={record['steps']} "
                  f"seconds={record['episode_seconds']:.0f}", flush=True)
    finally:
        if getattr(runtime, "env", None) is not None:
            runtime.env.close()
    save(out / "summary.json", dict(protocol=P["protocol"], worker=args.worker, job=tag, episodes=completed))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
