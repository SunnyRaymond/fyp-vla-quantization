"""CPU checks of one stack before any GPU job: test840.py --protocol P --model fastwam|cosmos [--write-manifest]

1. every file the worker reads exists (code, calibration, banks, prep anchors, A100 startup / anchor);
2. the worker and the panel-A LAQ modules import;
3. the manifest: 40 original LIBERO tasks x 21 initial states, names, languages, initial-state files (>= 21 states);
   written once (fastwam), and the other stack must rebuild it identically;
4. the model's T5 cache holds the prompt of every task;
5. the original LIBERO environments: every task (Fast-WAM) / the first task of each suite (Cosmos) builds, takes initial
   states 0 and 20 and settles 30 no-op steps without finishing;
6. claim / release / recover on a temporary directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import l840


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL {msg}")
    print(f"ok {msg}", flush=True)


def files(P, model):
    cfg = P["models"][model]
    p = l840.read(cfg["laq_protocol"])
    m = p["models"][model]
    c = p["laq"]
    calib = Path(p["remote_root"]) / "results" / model / "calib"
    prep = Path(m["a16_prep"])
    need = [cfg["a100_startup"], cfg["a100_anchor"], calib / "allocation.json", calib / "COMPLETE.json", c["arch"],
            prep / "bf16_anchor.pt", prep / "modules.json"]
    need += [prep / "t5_cache.pt"] if model == "fastwam" else [prep / "base_extra_cache.pkl"]
    for path in need:
        check(Path(path).is_file() and Path(path).stat().st_size > 0, f"file {path}")
    import methods as mt
    # load_weights decodes only the block-level Linears (methods.parse); interface layers stay BF16
    modules = [r for r in l840.read(prep / "modules.json") if mt.parse(r["module"]) is not None]
    check(len(modules) == m["targets"], f"{len(modules)} block-level Linears = protocol targets {m['targets']}")
    bank = Path(c["bank"])
    missing = [r["index"] for r in modules if not (bank / f"{r['index']:04d}.pt").is_file()]
    check(not missing, f"bank {bank}: {len(modules)} module files present (missing {missing[:5]})")
    check(l840.read(calib / "COMPLETE.json").get("complete") is True, "calibration COMPLETE")
    plan = Path(m.get("calibration", {}).get("plan") or m["panel"]["calibration_plan"])
    check(plan.is_file(), f"calibration plan {plan}")
    return p, m


def imports(model):
    names = (["laq", "graphs", "fw_common", "runner", "reference_profile", "profile_worker", "calibration_source",
              "quantization", "fw840", "experiments.libero.eval_libero_single"] if model == "fastwam"
             else ["laqc", "cos_common", "cosmos_loop", "cosmos_runtime", "endpoint", "quantization", "cos840",
                   "cosmos_policy.experiments.robot.libero.libero_utils"])
    for name in names:
        module = __import__(name, fromlist=["_"])
        check(True, f"import {name} from {getattr(module, '__file__', '?')}")


def build_manifest(P):
    import torch
    from libero.libero import benchmark, get_libero_path
    check("LIBERO-plus" not in benchmark.__file__ and "fastwam-smoke/libero" in benchmark.__file__,
          f"original LIBERO benchmark module {benchmark.__file__}")
    rows = l840.index_rows(P)
    info = {}
    for suite in P["benchmark"]["suites"]:
        bench = benchmark.get_benchmark_dict()[suite]()
        check(bench.get_num_tasks() == P["benchmark"]["tasks_per_suite"], f"{suite} has 10 tasks")
        for task_id in range(bench.get_num_tasks()):
            task = bench.get_task(task_id)
            path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            states = torch.load(path, map_location="cpu", weights_only=False)
            if len(states) < P["benchmark"]["trials_per_task"]:
                raise SystemExit(f"FAIL {suite} {task.name}: only {len(states)} initial states")
            info[suite, task_id] = dict(task_name=task.name, language=task.language, init_state_path=str(path),
                                        initial_states_available=len(states))
    out = [dict(r, **info[r["suite"], r["task_id"]]) for r in rows]
    check(len({(r["suite"], r["task_id"]) for r in out}) == 40 and len(out) == 840, "840 rows over 40 tasks")
    return dict(protocol=P["protocol"], benchmark=P["benchmark"], rows=out)


def manifest(P, model, write):
    doc = build_manifest(P)
    target = l840.results(P, model) / "manifest.json"
    first = l840.results(P, "fastwam") / "manifest.json"
    if write:
        l840.save(target, doc)
        check(True, f"wrote {target}")
    else:
        check(first.is_file(), "the fastwam manifest exists")
        check(l840.read(first)["rows"] == doc["rows"], "this stack rebuilds the fastwam manifest exactly")
        l840.save(target, doc)
    rows = l840.manifest_rows(P, model)
    check(len(rows) == 840, "manifest_rows accepts the manifest")
    return rows


def t5(P, model, m, rows):
    languages = sorted({r["language"] for r in rows})
    check(len(languages) == 40, "40 distinct task languages")
    if model == "fastwam":
        import torch
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        cache = torch.load(l840.results(P, model) / "t5_libero40.pt", map_location="cpu", weights_only=True)
        cache.update(torch.load(Path(m["a16_prep"]) / "t5_cache.pt", map_location="cpu", weights_only=True))
        missing = [x for x in languages if DEFAULT_PROMPT.format(task=x) not in cache]
    else:
        from cos_common import load_pickle
        cache = load_pickle(Path(m["a16_prep"]) / "base_extra_cache.pkl")
        missing = [x for x in languages if x not in cache]
    check(not missing, f"T5 cache covers all 40 prompts (missing {missing})")


def environments(P, model, rows):
    import numpy as np
    import torch
    B = P["benchmark"]
    from libero.libero import benchmark
    picks = [r for r in rows if r["trial"] in (0, B["trials_per_task"] - 1)]
    if model == "cosmos":
        picks = [r for r in picks if r["task_id"] == 0 and r["trial"] == B["trials_per_task"] - 1]
    suites = {}
    for r in picks:
        begin = time.monotonic()
        suite = suites.setdefault(r["suite"], benchmark.get_benchmark_dict()[r["suite"]]())
        task = suite.get_task(r["task_id"])
        if model == "fastwam":
            from experiments.libero import eval_libero_single as ev
            env, _ = ev.get_libero_env(task, ev.LIBERO_ENV_RESOLUTION, r["env_seed"])
            dummy = ev.get_libero_dummy_action()
            states = torch.load(r["init_state_path"], map_location="cpu", weights_only=False)
            size = ev.LIBERO_ENV_RESOLUTION
        else:
            import types
            from cos840 import original_states
            from cosmos_policy.experiments.robot.libero import libero_utils
            from libero.libero.envs import OffScreenRenderEnv
            env = OffScreenRenderEnv(bddl_file_name=str(suite.get_task_bddl_file_path(r["task_id"])),
                                     camera_heights=256, camera_widths=256)
            env.seed(r["env_seed"])
            states, source = original_states(types.SimpleNamespace(), suite, r["task_id"], False)
            check(source == r["init_state_path"], f"cosmos initial-state loader path {source}")
            dummy = libero_utils.get_libero_dummy_action("cosmos")
            size = 256
        np.random.seed(r["env_seed"])
        env.reset()
        obs = env.set_init_state(states[r["trial"]])
        for _ in range(B["settling_steps"]):
            obs, _, done, _ = env.step(dummy)
            if done:
                raise SystemExit(f"FAIL {r['suite']} {r['task_name']} trial {r['trial']} finished during settling")
        image = obs["agentview_image"]
        env.close()
        check(image.shape == (size, size, 3) and float(image.std()) > 0,
              f"env {r['suite']} task {r['task_id']} trial {r['trial']} settles ({time.monotonic() - begin:.1f}s)")


def claims():
    os.environ.setdefault("SLURM_JOB_ID", "999")
    os.environ["SLURM_RESTART_COUNT"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [dict(index=i, suite="libero_10" if i >= 3 else "libero_goal") for i in range(5)]
        ordered = l840.order(rows)
        check([r["index"] for r in ordered] == [3, 4, 0, 1, 2], "order puts libero_10 first")
        a, _ = l840.claim(root, ordered, 0, "w0")
        b, _ = l840.claim(root, ordered, 0, "w1")
        check((a["index"], b["index"]) == (3, 4), "claims are exclusive")
        l840.save(root / "episodes" / "3.json", {})
        # a claim of a dead job and one of this job's earlier restart are stale; a live job's claim is kept
        for index, job, restart in ((0, "123", 0), (1, os.environ["SLURM_JOB_ID"], 0), (2, "456", 0)):
            (root / "claims" / str(index)).mkdir()
            l840.save(root / "claims" / str(index) / "owner.json", dict(job=job, restart=restart))
        dropped = l840.recover(root, running={os.environ["SLURM_JOB_ID"], "456"})
        check(sorted(dropped) == [0, 1], f"recover drops dead and earlier-restart claims ({dropped})")
        check((root / "claims" / "3").exists() and (root / "claims" / "4").exists(), "recover keeps finished and live claims")
        c, _ = l840.claim(root, ordered, 0, "w2")
        check(c["index"] == 0, "a recovered row is claimed again")
        l840.release(root, 4)
        d, _ = l840.claim(root, ordered, 0, "w3")
        check(d["index"] == 4, "a released row is claimed again")
    check(l840.walltime_seconds("1-02:03:04") == 93784 and l840.walltime_seconds("12:00:00") == 43200, "walltime parse")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", choices=("fastwam", "cosmos"), required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    P = l840.read(args.protocol)
    print(f"TEST840 {args.model} python {sys.version.split()[0]}", flush=True)
    claims()
    p, m = files(P, args.model)
    imports(args.model)
    rows = manifest(P, args.model, args.write_manifest)
    t5(P, args.model, m, rows)
    environments(P, args.model, rows)
    print(f"TEST840_PASS {args.model}", flush=True)


if __name__ == "__main__":
    main()
