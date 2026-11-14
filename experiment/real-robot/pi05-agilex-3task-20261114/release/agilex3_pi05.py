"""pi0.5 (openpi, full fine-tuning) for the AgileX Piper, one model for three tasks: inference, policy server, self-test.

Run with the Python of an openpi checkout at commit 215abfb217dbac7d5f1273282331b9b1866c0479 (its .venv). This file
only needs the bundle next to it: checkpoint/ (params + norm stats) and openpi_cache/ (PaliGemma tokenizer, so nothing
is downloaded).

  python agilex3_pi05.py selftest --reference selftest_reference.json --out selftest_local.json
  python agilex3_pi05.py serve --port 8000          # openpi websocket policy server (openpi-client on the robot)

In your own robot code:

  from agilex3_pi05 import AgileXPi05Policy
  policy = AgileXPi05Policy()                                  # loads checkpoint/ next to this file
  actions = policy.predict(fixed, wrist, state, instruction)   # (10, 7): absolute joint targets (rad) + gripper (m)

Inputs are what the recorder wrote into the HDF5 files: fixed = observation/images/realsense_fixed, wrist =
observation/images/realsense_wrist, each (240, 320, 3) uint8 in the same channel order; state = observation/state
(7: six joints in rad, gripper opening in m); instruction = one of TASKS. actions[k] is the target state 0.1 * (k + 1) s
ahead (the model was trained at 10 Hz).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("OPENPI_DATA_HOME", str(HERE / "openpi_cache"))

NAME = "pi05_agilex3"
REPO_ID = "local/agilex3_10hz"
CHECKPOINT = HERE / "checkpoint"
CHECKPOINT_STEP = 5999
HORIZON = 10
RAW_FPS, FPS = 30, 10
STRIDE = RAW_FPS // FPS
IMAGE_SHAPE = (240, 320, 3)
TASKS = {
    "cluster_grasp": "place the red apple and green cube to the box",
    "cabinet_lemon": "open the bin and put the lemon in it",
    "circle": "place the cup in the tape",
}
NOISE_SEED_BASE = 1000


def train_config():
    """The training config pi05_agilex3, minus the training-only fields. Model and transforms must stay as trained."""
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config

    return _config.TrainConfig(
        name=NAME,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=HORIZON),
        data=_config.LeRobotLiberoDataConfig(
            repo_id=REPO_ID,
            base_config=_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        assets_base_dir=str(CHECKPOINT / "assets"),
    )


class AgileXPi05Policy:
    """openpi Policy for pi05_agilex3. Norm stats come from <checkpoint>/assets/local/agilex3_10hz."""

    def __init__(self, checkpoint: str | Path = CHECKPOINT):
        from openpi.policies import policy_config

        t0 = time.time()
        self.config = train_config()
        self.policy = policy_config.create_trained_policy(self.config, Path(checkpoint))
        self.action_dim = self.config.model.action_dim
        self.load_seconds = time.time() - t0
        self.last_timing: dict = {}

    def predict(self, fixed, wrist, state, instruction: str, noise: np.ndarray | None = None) -> np.ndarray:
        fixed, wrist = np.asarray(fixed), np.asarray(wrist)
        state = np.asarray(state, dtype=np.float32)
        for name, image in (("fixed", fixed), ("wrist", wrist)):
            if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
                raise ValueError(f"{name} image must be {IMAGE_SHAPE} uint8 like the recorded HDF5 frames, "
                                 f"got {image.shape} {image.dtype}")
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(f"state must be 7 finite values (6 joints in rad + gripper in m), got {state}")
        if instruction not in TASKS.values():
            print(f"warning: instruction {instruction!r} is not one of the trained instructions", file=sys.stderr)
        out = self.policy.infer({"observation/image": fixed, "observation/wrist_image": wrist,
                                 "observation/state": state, "prompt": instruction}, noise=noise)
        self.last_timing = dict(out.get("policy_timing", {}))
        return np.asarray(out["actions"], dtype=np.float32)


def load_episode(path: str) -> dict:
    """A raw 30 Hz recording read at 10 Hz from frame 0, exactly as in training: frame k = raw frame 3k, its action =
    action[min(3k + 2, T - 1)] = state[3k + 3]."""
    import h5py

    with h5py.File(path, "r") as h:
        state = h["observation/state"][:].astype(np.float32)
        action = h["action"][:].astype(np.float32)
        fixed = h["observation/images/realsense_fixed"][:]
        wrist = h["observation/images/realsense_wrist"][:]
        instruction = h.attrs["instruction"]
        fps = float(h.attrs["fps"])
    instruction = (instruction.decode() if isinstance(instruction, bytes) else str(instruction)).strip()
    if fps != RAW_FPS:
        raise ValueError(f"{path}: fps {fps}, expected {RAW_FPS}")
    idx = np.arange(0, len(state), STRIDE)
    return dict(state=state[idx], action=action[np.minimum(idx + STRIDE - 1, len(state) - 1)], fixed=fixed[idx],
                wrist=wrist[idx], instruction=instruction)


def errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    return dict(joint=float(np.abs(pred[:, :6] - gt[:, :6]).mean()), grip=float(np.abs(pred[:, 6] - gt[:, 6]).mean()))


def summarize(rows: list[dict]) -> dict:
    joint = float(np.mean([r["model"]["joint"] for r in rows]))
    hold = float(np.mean([r["hold"]["joint"] for r in rows]))
    return dict(queries=len(rows), joint=joint, grip=float(np.mean([r["model"]["grip"] for r in rows])),
                hold_joint=hold, hold_grip=float(np.mean([r["hold"]["grip"] for r in rows])),
                joint_ratio=joint / hold)


def selftest(args) -> int:
    import jax

    episodes = sorted(glob.glob(args.episodes)) if isinstance(args.episodes, str) else sorted(args.episodes)
    if not episodes:
        raise SystemExit(f"no episodes match {args.episodes}")
    policy = AgileXPi05Policy(args.checkpoint)
    device = jax.devices()[0]
    print(f"loaded {args.checkpoint} in {policy.load_seconds:.1f}s on {device.device_kind}", flush=True)
    data = {Path(p).name: load_episode(p) for p in episodes}

    first = next(iter(data.values()))
    t0 = time.time()
    policy.predict(first["fixed"][0], first["wrist"][0], first["state"][0], first["instruction"],
                   noise=np.random.default_rng(0).standard_normal((HORIZON, policy.action_dim)).astype(np.float32))
    first_call = time.time() - t0  # includes XLA compilation

    rows, preds, latencies, model_ms, finite = [], [], [], [], True
    for name, ep in data.items():
        for t in range(0, len(ep["state"]) - HORIZON + 1, HORIZON):
            q = len(rows)
            noise = np.random.default_rng(NOISE_SEED_BASE + q).standard_normal((HORIZON, policy.action_dim))
            s0 = time.time()
            pred = policy.predict(ep["fixed"][t], ep["wrist"][t], ep["state"][t], ep["instruction"],
                                  noise=noise.astype(np.float32))
            latencies.append(time.time() - s0)
            model_ms.append(policy.last_timing.get("infer_ms", float("nan")))
            finite &= bool(np.isfinite(pred).all())
            gt = ep["action"][t:t + HORIZON]
            hold = np.repeat(ep["state"][t][None], HORIZON, axis=0)
            rows.append(dict(episode=name, task=ep["instruction"], t=t, model=errors(pred, gt), hold=errors(hold, gt),
                             grip_range=[float(pred[:, 6].min()), float(pred[:, 6].max())]))
            preds.append(pred)
        mine = [r for r in rows if r["episode"] == name]
        print(f"{name}: {len(ep['state'])} frames at 10 Hz, {len(mine)} queries, joint error "
              f"{summarize(mine)['joint']:.4f} rad (hold {summarize(mine)['hold_joint']:.4f})", flush=True)

    per_task = {task: summarize([r for r in rows if r["task"] == task]) for task in sorted({r["task"] for r in rows})}
    pooled = summarize(rows)
    stats = device.memory_stats() or {}
    report = dict(checkpoint=str(args.checkpoint), checkpoint_step=CHECKPOINT_STEP, device=device.device_kind,
                  jax=jax.__version__, load_seconds=round(policy.load_seconds, 1), first_call_seconds=round(first_call, 1),
                  latency_s=dict(median=float(np.median(latencies)), p90=float(np.percentile(latencies, 90)),
                                 max=float(np.max(latencies))),
                  model_infer_ms_median=float(np.nanmedian(model_ms)),
                  peak_gpu_memory_gib=round(stats["peak_bytes_in_use"] / 2 ** 30, 2) if "peak_bytes_in_use" in stats
                  else None,
                  noise_seed_base=NOISE_SEED_BASE, all_finite=finite, pooled=pooled, per_task=per_task,
                  per_episode={n: summarize([r for r in rows if r["episode"] == n]) for n in data},
                  grip_pred_range=[min(r["grip_range"][0] for r in rows), max(r["grip_range"][1] for r in rows)],
                  rows=rows)
    ok = finite and all(s["joint_ratio"] <= 0.5 for s in [pooled, *per_task.values()])
    if args.write_reference:
        ref = dict(checkpoint_step=CHECKPOINT_STEP, device=device.device_kind, noise_seed_base=NOISE_SEED_BASE,
                   joint=pooled["joint"], grip=pooled["grip"], per_task={k: v["joint"] for k, v in per_task.items()},
                   queries=[dict(episode=r["episode"], t=r["t"], actions=np.round(p, 6).tolist())
                            for r, p in zip(rows, preds)])
        Path(args.write_reference).write_text(json.dumps(ref) + "\n", encoding="utf-8")
    if args.reference:
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        keyed = {(q["episode"], q["t"]): np.asarray(q["actions"], dtype=np.float32) for q in ref["queries"]}
        matched = [(keyed[(r["episode"], r["t"])], p) for r, p in zip(rows, preds) if (r["episode"], r["t"]) in keyed]
        diff_joint = float(np.mean([np.abs(a[:, :6] - b[:, :6]).mean() for a, b in matched])) if matched else None
        diff_grip = float(np.mean([np.abs(a[:, 6] - b[:, 6]).mean() for a, b in matched])) if matched else None
        ref_ok = (len(matched) == len(ref["queries"]) == len(rows) and pooled["joint"] <= 1.5 * ref["joint"] + 0.005
                  and diff_joint <= 0.02)
        report["reference_check"] = dict(reference_device=ref["device"], reference_joint=ref["joint"],
                                         joint=pooled["joint"], matched_queries=len(matched),
                                         pred_diff_joint=diff_joint, pred_diff_grip=diff_grip, ok=bool(ref_ok))
        print(f"reference ({ref['device']}): joint error {ref['joint']:.4f} rad, here {pooled['joint']:.4f}; "
              f"predictions differ by {diff_joint} rad (joints), {diff_grip} m (gripper)", flush=True)
        ok = ok and ref_ok
    report["pass"] = bool(ok)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for task, s in per_task.items():
        print(f"[{task}] {s['queries']} queries: joint {s['joint']:.4f} rad (hold {s['hold_joint']:.4f}, ratio "
              f"{s['joint_ratio']:.3f}), gripper {s['grip'] * 1000:.2f} mm", flush=True)
    print(f"pooled: joint {pooled['joint']:.4f} rad (hold {pooled['hold_joint']:.4f}, ratio {pooled['joint_ratio']:.3f}), "
          f"gripper {pooled['grip'] * 1000:.2f} mm (hold {pooled['hold_grip'] * 1000:.2f}); latency median "
          f"{report['latency_s']['median']:.3f}s; first call {first_call:.1f}s; peak GPU memory "
          f"{report['peak_gpu_memory_gib']} GiB", flush=True)
    print("SELFTEST_PASS" if ok else "SELFTEST_FAIL", flush=True)
    return 0 if ok else 3


def serve(args) -> int:
    import logging

    from openpi.serving import websocket_policy_server

    logging.basicConfig(level=logging.INFO, force=True)
    p = AgileXPi05Policy(args.checkpoint)
    logging.info("loaded %s in %.1fs; serving on 0.0.0.0:%d", args.checkpoint, p.load_seconds, args.port)
    websocket_policy_server.WebsocketPolicyServer(policy=p.policy, host="0.0.0.0", port=args.port,
                                                  metadata=p.policy.metadata).serve_forever()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    st = sub.add_parser("selftest", help="open-loop check on the sample recordings")
    st.add_argument("--episodes", nargs="+", default=[str(p) for p in sorted((HERE / "samples").glob("*.hdf5"))])
    st.add_argument("--reference", help="selftest_reference.json from the bundle: compare with our A100 run")
    st.add_argument("--write-reference")
    st.add_argument("--out", default="selftest_local.json")
    st.add_argument("--checkpoint", default=str(CHECKPOINT))
    sv = sub.add_parser("serve", help="openpi websocket policy server")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = ap.parse_args()
    return selftest(args) if args.action == "selftest" else serve(args)


if __name__ == "__main__":
    sys.exit(main())
