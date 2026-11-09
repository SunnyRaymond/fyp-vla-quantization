"""Open-loop check of the AgileX Fast-WAM policy on recorded HDF5 episodes (raw frames, no video re-encoding).

    python openloop_check.py --bundle <bundle> --episodes ep1.hdf5 ep2.hdf5 ... --out report.json
        [--configs idm idm_shift1 first_frame] [--stride 10] [--execute 10]
        [--write-reference ref.json | --reference ref.json]

For every query time t = 0, stride, 2*stride, ... (t <= T-2) the policy sees the recorded frames and state at t and
predicts 32 actions; they are compared with the recorded actions action[t:t+32] (action[t] == state[t+1]) over the
steps that exist. "hold" is the do-nothing baseline that keeps state[t]. Errors are mean absolute errors: joints in rad
(mean over the 6 joints), gripper in m; "exec" = the first --execute steps (what the robot runs before replanning),
"full" = all predicted steps that have ground truth.

Pass rule for the default config (fixed before looking at results), per task and pooled:
  joints exec MAE <= 0.5 x hold, gripper exec MAE <= max(0.5 x hold, 2 mm), every prediction finite.
With --reference the run also checks that pooled joints exec MAE <= 1.5 x reference + 0.005 rad (environment check).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agilex_policy import AgileXFastWAMPolicy  # noqa: E402

CONFIGS = {
    "idm": dict(mode="idm", sigma_shift=None),
    "idm_shift1": dict(mode="idm", sigma_shift=1.0),
    "first_frame": dict(mode="first_frame", sigma_shift=None),
}
JOINTS = slice(0, 6)


def load_episode(path: Path):
    with h5py.File(path, "r") as h:
        return dict(fixed=h["observation/images/realsense_fixed"][:], wrist=h["observation/images/realsense_wrist"][:],
                    state=h["observation/state"][:].astype(np.float32), action=h["action"][:].astype(np.float32),
                    instruction=str(h.attrs["instruction"]))


def errors(pred: np.ndarray, gt: np.ndarray, execute: int):
    n = len(gt)
    d = np.abs(pred[:n] - gt)
    joint, grip = d[:, JOINTS].mean(axis=1), d[:, 6]
    k = min(execute, n)
    return dict(joint_exec=float(joint[:k].mean()), joint_full=float(joint.mean()),
                grip_exec=float(grip[:k].mean()), grip_full=float(grip.mean()),
                joint_end_exec=float(joint[k - 1]), per_step_joint=joint.tolist())


def pooled(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def summarize(rows):
    out = {}
    for key in ("joint_exec", "joint_full", "grip_exec", "grip_full", "joint_end_exec"):
        out[key] = pooled([r["model"] for r in rows], key)
        out["hold_" + key] = pooled([r["hold"] for r in rows], key)
    out["joint_exec_ratio"] = out["joint_exec"] / max(out["hold_joint_exec"], 1e-12)
    out["grip_exec_ratio"] = out["grip_exec"] / max(out["hold_grip_exec"], 1e-12)
    out["queries"] = len(rows)
    steps = np.zeros(32)
    counts = np.zeros(32)
    for r in rows:
        v = np.asarray(r["model"]["per_step_joint"])
        steps[:len(v)] += v
        counts[:len(v)] += 1
    out["joint_mae_by_step"] = (steps / np.maximum(counts, 1)).round(5).tolist()
    return out


def passes(summary):
    return bool(summary["joint_exec"] <= 0.5 * summary["hold_joint_exec"]
                and summary["grip_exec"] <= max(0.5 * summary["hold_grip_exec"], 0.002))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=["idm"], choices=sorted(CONFIGS))
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--execute", type=int, default=10)
    ap.add_argument("--seed-spread", type=int, default=0, help="extra seeds at each episode's first query (default cfg)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-reference")
    ap.add_argument("--reference")
    args = ap.parse_args()

    policy = AgileXFastWAMPolicy(args.bundle, checkpoint=args.checkpoint)
    print(f"loaded step {policy.checkpoint_step} in {policy.load_seconds:.1f}s; instructions {policy.instructions}",
          flush=True)
    episodes = [(Path(p), load_episode(Path(p))) for p in args.episodes]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report = dict(checkpoint_step=policy.checkpoint_step, load_seconds=round(policy.load_seconds, 1),
                  episodes=[str(p) for p, _ in episodes], stride=args.stride, execute=args.execute,
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", configs={})
    all_finite = True
    for name in args.configs:
        policy.mode, policy.sigma_shift = CONFIGS[name]["mode"], CONFIGS[name]["sigma_shift"]
        # warm-up so the first timed call does not include CUDA/cuBLAS initialisation
        _, ep = episodes[0]
        policy.predict(ep["fixed"][0], ep["wrist"][0], ep["state"][0], ep["instruction"])
        rows, latencies, spread = [], [], []
        for path, ep in episodes:
            T = len(ep["state"])
            for t in range(0, T - 1, args.stride):
                out = policy.predict(ep["fixed"][t], ep["wrist"][t], ep["state"][t], ep["instruction"])
                pred = out["actions"]
                latencies.append(out["latency_s"])
                all_finite &= bool(np.isfinite(pred).all())
                gt = ep["action"][t:t + policy.action_horizon]
                hold = np.repeat(ep["state"][t][None], policy.action_horizon, axis=0)
                rows.append(dict(episode=path.name, task=ep["instruction"], t=t, model=errors(pred, gt, args.execute),
                                 hold=errors(hold, gt, args.execute),
                                 gripper_pred_range=[float(pred[:, 6].min()), float(pred[:, 6].max())]))
                if t == 0 and name == args.configs[0]:
                    for s in range(1, args.seed_spread + 1):
                        other = policy.predict(ep["fixed"][t], ep["wrist"][t], ep["state"][t], ep["instruction"],
                                               seed=s)["actions"]
                        spread.append(float(np.abs(other[:args.execute, JOINTS] - pred[:args.execute, JOINTS]).mean()))
            print(f"[{name}] {path.name} T={T} queries={sum(r['episode'] == path.name for r in rows)} "
                  f"joint_exec={pooled([r['model'] for r in rows if r['episode'] == path.name], 'joint_exec'):.4f}",
                  flush=True)
        tasks = sorted({r["task"] for r in rows})
        cfg_report = dict(**CONFIGS[name], pooled=summarize(rows),
                          per_task={task: summarize([r for r in rows if task == r["task"]]) for task in tasks},
                          per_episode={p.name: summarize([r for r in rows if r["episode"] == p.name])
                                       for p, _ in episodes},
                          latency_s=dict(median=float(np.median(latencies)), p90=float(np.percentile(latencies, 90)),
                                         max=float(np.max(latencies))),
                          gripper_pred_range=[min(r["gripper_pred_range"][0] for r in rows),
                                              max(r["gripper_pred_range"][1] for r in rows)])
        if spread:
            cfg_report["seed_spread_joint_exec"] = float(np.mean(spread))
        cfg_report["pass"] = passes(cfg_report["pooled"]) and all(passes(v) for v in cfg_report["per_task"].values())
        report["configs"][name] = cfg_report
        p = cfg_report["pooled"]
        print(f"[{name}] joints exec {p['joint_exec']:.4f} rad (hold {p['hold_joint_exec']:.4f}, ratio "
              f"{p['joint_exec_ratio']:.3f}); gripper exec {p['grip_exec'] * 1000:.2f} mm (hold "
              f"{p['hold_grip_exec'] * 1000:.2f}); latency median {cfg_report['latency_s']['median']:.3f}s; "
              f"pass {cfg_report['pass']}", flush=True)
    if torch.cuda.is_available():
        report["peak_gpu_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report["peak_gpu_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    report["all_finite"] = all_finite
    default = args.configs[0]
    report["pass"] = bool(all_finite and report["configs"][default]["pass"])
    if args.write_reference:
        ref = dict(config=default, joint_exec=report["configs"][default]["pooled"]["joint_exec"],
                   grip_exec=report["configs"][default]["pooled"]["grip_exec"],
                   per_episode={k: v["joint_exec"] for k, v in report["configs"][default]["per_episode"].items()},
                   device=report["device"], checkpoint_step=policy.checkpoint_step)
        Path(args.write_reference).write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    if args.reference:
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        got = report["configs"][ref["config"]]["pooled"]["joint_exec"] if ref["config"] in report["configs"] else None
        ok = got is not None and got <= 1.5 * ref["joint_exec"] + 0.005
        report["reference_check"] = dict(reference=ref["joint_exec"], got=got, pass_=bool(ok))
        report["pass"] = bool(report["pass"] and ok)
        print(f"reference joints exec {ref['joint_exec']:.4f} rad, this machine {got}, ok {ok}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("OPENLOOP_PASS" if report["pass"] else "OPENLOOP_FAIL", flush=True)
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
