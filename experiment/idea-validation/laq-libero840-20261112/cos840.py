"""Standard-LIBERO closed loop (840 episodes) of the LAQ arm on Cosmos-Policy, on an EEE GPU.

Same model, weights, allocation, hooks and online gaze as the panel-A loop (laq-cosmos-20261111, cosmos_loop.py); the
episode runner is the same CosmosRuntime.episode (30 settling steps, 16-action replans, caps 400/700) on the original
LIBERO suites, with the official initial state = trial index.

Startup checks follow fw840.py: on this GPU model the BF16 anchor must lie within `bf16_drift_fraction` of the A100 LAQ
damage (action relative RMSE, future-image NMSE), the LAQ damage against this GPU's BF16 anchor must stay within
`damage_ratio` of the A100 damage, and every worker on one GPU model must reproduce the first worker's arm anchor bitwise.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import time
import types

import l840
from cos_common import anchor, anchor_errors, guard, read, runtime, same_anchor, save, torch_save
from cosmos_loop import SETTLING, stage_language, t5_cache

MODEL = "cosmos"
KEYS = ("action_rel_rmse", "future_image_nmse")


def original_states(self, suite, task_id, smoke_original):
    """Official initial states of the original LIBERO task (replaces the LIBERO-Plus loader)."""
    import torch
    from libero.libero import get_libero_path
    task = suite.get_task(int(task_id))
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, map_location="cpu", weights_only=False), str(path)


def startup(P, p, m, rt, arm, res):
    import torch
    import laqc
    c = p["laq"]
    prep = Path(m["a16_prep"])
    calib = Path(p["remote_root"]) / "results" / "cosmos" / "calib"
    if not read(calib / "COMPLETE.json").get("complete"):
        raise RuntimeError("The calibration is not complete")
    alloc = read(calib / "allocation.json")
    device = torch.device("cuda")
    st = laqc.State(c["steps"], c["cut"], read(c["arch"]), m["rotation_seed"], device)
    modules = dict(rt.target_linears)
    installed, wrapped = laqc.install(st, rt.model, modules)
    if installed != m["targets"] or wrapped != m["attentions"]:
        raise RuntimeError(f"Hooked {installed} Linears and {wrapped} attentions, expected {m['targets']} and {m['attentions']}")
    bf16 = anchor(m, rt)
    reference = torch.load(prep / "bf16_anchor.pt", map_location="cpu", weights_only=True)
    drift = anchor_errors(bf16, reference)
    index_of = {r["module"]: r["index"] for r in read(prep / "modules.json")}
    laqc.load_weights(st, modules, index_of, Path(c["bank"]))
    st.seen.clear()
    st.counts.clear()
    st.offline = torch.tensor(alloc["offline_gaze"], device=device)
    if st.offline.numel() != st.layout["S"]:
        raise RuntimeError(f"The offline gaze covers {st.offline.numel()} tokens, the model has {st.layout['S']}")
    st.watch = True
    st.use(alloc["arms"][arm]["spec"], online=c["online"][arm])
    torch.cuda.synchronize()
    begin = time.monotonic()
    mine = anchor(m, rt)
    torch.cuda.synchronize()
    seconds = time.monotonic() - begin
    st.reset()
    counts = dict(st.counts)
    if len(st.seen) != m["targets"] or not all(counts.get(s, 0) > 0 for s in ("self", "text")):
        raise RuntimeError(f"Coverage failed: {len(st.seen)} Linears, KV sites {counts}")
    rt.laq = st
    damage = anchor_errors(mine, bf16)
    cfg = P["models"][MODEL]
    a100_damage = read(cfg["a100_startup"])["anchor_vs_bf16"]
    vs_a100_laq = anchor_errors(mine, torch.load(cfg["a100_anchor"], map_location="cpu", weights_only=True))
    g = P["gates"]
    ratio = {k: damage[k] / a100_damage[k] for k in KEYS}
    gate = dict(drift_ok=all(drift[k] <= g["bf16_drift_fraction"] * a100_damage[k] for k in KEYS),
                ratio_ok=all(g["damage_ratio"][0] <= ratio[k] <= g["damage_ratio"][1] for k in KEYS))
    gpu = torch.cuda.get_device_name(0)
    check = dict(protocol=P["protocol"], model=MODEL, arm=arm, job=l840.owner_tag(), host=os.uname().nodename, gpu=gpu,
                 bf16_vs_a100_bf16=drift, laq_vs_this_gpu_bf16=damage, a100_laq_damage=a100_damage, damage_ratio=ratio,
                 laq_vs_a100_laq=vs_a100_laq, gate=gate, linears_seen=len(st.seen), kv_hits_per_query=counts,
                 allocation=alloc["arms"][arm], anchor_query_seconds=seconds,
                 peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
    if not (gate["drift_ok"] and gate["ratio_ok"]):
        return check, False
    path = res / "startup" / f"anchor_{l840.gpu_slug(gpu)}.pt"
    if not path.exists():
        torch_save(mine, path)
    first = torch.load(path, map_location="cpu", weights_only=True)
    if not same_anchor(mine, first):
        raise RuntimeError(f"{arm} anchor differs from the first worker on {gpu}: {anchor_errors(mine, first)}")
    return check, True


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


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
    m, arm, B = p["models"]["cosmos"], cfg["arm"], P["benchmark"]
    res = l840.results(P, MODEL)
    tag = l840.owner_tag()
    out = res / "workers" / f"{tag['job']}.{tag['restart']}.{args.worker}"
    save(out / "progress.json", dict(phase="loading", updated_epoch=time.time()))
    deadline = float(os.environ["L840_DEADLINE_EPOCH"])
    full, keep, small = t5_cache(m, out)
    lock = os.open(os.environ["L840_BUILD_LOCK"], os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        rt = runtime(m, out / "runtime", small)
        inference = rt.protocol["inference"]
        if (inference["settling_steps"] != B["settling_steps"] or inference["max_steps_long"] != B["max_steps"]["libero_10"]
                or any(inference["max_steps_normal"] != B["max_steps"][s] for s in B["suites"] if s != "libero_10")
                or inference["execute_per_replan"] != B["replan_steps"][MODEL]):
            raise RuntimeError("The Cosmos runtime protocol differs from the benchmark settings")
        rt._initial_states = types.MethodType(original_states, rt)
        rt.soft_deadline_epoch = deadline
        check, passed = startup(P, p, m, rt, arm, res)
        import torch
        torch.cuda.empty_cache()
    finally:
        os.close(lock)
    save(out / "startup.json", check)
    print(f"STARTUP {MODEL} worker={args.worker} gpu={check['gpu']} drift={check['bf16_vs_a100_bf16']} "
          f"damage={check['laq_vs_this_gpu_bf16']} ratio={check['damage_ratio']} query={check['anchor_query_seconds']:.1f}s "
          f"peak={check['peak_reserved_gib']:.1f}GiB", flush=True)
    if not passed:
        rt.close()
        raise RuntimeError(f"Cross-GPU gate failed: {check['gate']}")
    if args.smoke:
        rt.close()
        save(out / "progress.json", dict(phase="smoke_passed", updated_epoch=time.time()))
        return
    rows = l840.manifest_rows(P, MODEL)
    ordered = l840.order(rows)
    completed, stopped = [], False
    try:
        while time.time() < deadline and (args.max_episodes is None or len(completed) < args.max_episodes):
            row = l840.next_row(P, res, ordered, args.worker)
            if row is None:
                break
            index = row["index"]
            save(out / "progress.json", dict(phase="episode", index=index, completed=len(completed),
                                             updated_epoch=time.time()))
            task_row = dict(suite=row["suite"], task_id=row["task_id"], task_name=row["task_name"],
                            env_seed=row["env_seed"], sampler_seed=row["sampler_seed"], state_id=row["trial"],
                            variant_id=f"{row['suite']}/{row['task_id']}/{row['trial']}", dimension="original")
            stage_language(rt, full, keep, task_row)
            rt.laq.reset()
            try:
                record = rt.episode(task_row, index)
            except BaseException:
                l840.release(res, index)
                raise
            error = record.get("error") or ""
            if record["termination"] == "soft_deadline":
                l840.release(res, index)
                stopped = True
                break
            invalid = SETTLING in error
            if not invalid and not (record["complete"] and record["closed_loop_complete"]):
                l840.release(res, index)
                raise RuntimeError(f"episode {index} incomplete: {record['termination']} {error}")
            if record["initial_state_source"] != row["init_state_path"] or record["language"] != row["language"]:
                l840.release(res, index)
                raise RuntimeError(f"episode {index} ran another task or initial-state file: {record['initial_state_source']}")
            trace = rt.laq.trace
            record.update(model=MODEL, arm=arm, trial=row["trial"], initial_state_index=row["trial"], invalid=invalid,
                          success=bool(record["success"]) and not invalid, worker=args.worker, job=tag,
                          host=os.uname().nodename, gpu=check["gpu"],
                          gaze_overlap_previous=_mean([t["previous"] for t in trace]),
                          gaze_overlap_offline=_mean([t["offline"] for t in trace]))
            save(res / "episodes" / f"{index}.json", record)
            completed.append(index)
            print(f"EPISODE {MODEL} worker={args.worker} index={index} {row['suite']} task={row['task_id']} "
                  f"trial={row['trial']} success={record['success']} steps={record['steps']} "
                  f"seconds={record['episode_seconds']:.0f}" + (" invalid" if invalid else ""), flush=True)
    finally:
        rt.close()
    save(out / "summary.json", dict(protocol=P["protocol"], worker=args.worker, job=tag, episodes=completed,
                                    stopped_at_deadline=stopped))
    save(out / "progress.json", dict(phase="complete", completed=len(completed), updated_epoch=time.time()))


if __name__ == "__main__":
    main()
