"""Exploratory paired error decomposition; execute only in an approved PBS allocation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


SCREEN_DIR = Path(__file__).resolve().parents[1] / "mechanism-screen"
sys.path.append(str(SCREEN_DIR))
from runner import SAMPLER_SEEDS, Screen, guard, save  # noqa: E402


TEST_EPISODES = tuple(range(8, 16))
RTN_DRAWS = (2101, 2102, 2103, 2104)
LEARNED_DRAWS = (2101, 2102, 2103, 2104)
BOOTSTRAP_DRAWS = 2000
IDENTITY_TOL = 1e-7


def load_test_data(path: Path, torch):
    data = torch.load(path, map_location="cpu", weights_only=False)
    data = [d for d in data if int(d["episode"]) in TEST_EPISODES]
    keys = [(int(d["episode"]), int(d["position"])) for d in data]
    expected = {(episode, position) for episode in TEST_EPISODES for position in (0, 1)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"frozen TEST must contain exactly 8x2 observations; got {keys}")
    for datum in data:
        if any(seed not in datum["refs"] and str(seed) not in datum["refs"] for seed in SAMPLER_SEEDS):
            raise ValueError(f"missing cached BF16 sampler labels for {datum['episode']}/{datum['position']}")
    return sorted(data, key=lambda d: (int(d["episode"]), int(d["position"])))


def cached_ref(datum, seed):
    refs = datum["refs"]
    return refs[seed] if seed in refs else refs[str(seed)]


def environment_action(screen, normalized):
    """Apply the official evaluator's complete inverse-normalize/gripper transform."""
    import numpy as np

    action = screen.ev._denormalize_action(normalized.clone(), screen.processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = screen.ev.invert_gripper_action(action)
    if bool(screen.cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] < 10 or action.shape[1] != 7:
        raise ValueError(f"expected official LIBERO [horizon,7] command, got {action.shape}")
    return action.copy()


def record(arm, datum, sampler_seed, draw_seed, output, real_action):
    return {
        "arm": arm,
        "episode": int(datum["episode"]),
        "position": int(datum["position"]),
        "sampler_seed": int(sampler_seed),
        "draw_seed": None if draw_seed is None else int(draw_seed),
        "normalized_action": output.detach().float().cpu().clone(),
        "environment_action": real_action,
    }


def command_metrics(action, reference):
    import numpy as np

    command = action[:10]
    target = reference[:10]
    mismatches = np.sign(command[:, -1]) != np.sign(target[:, -1])
    return {
        "continuous6_command_mse_first10": float(np.square(command[:, :6] - target[:, :6]).mean()),
        "gripper_sign_mismatch_count_first10": int(mismatches.sum()),
        "gripper_sign_mismatch_rate_first10": float(mismatches.mean()),
    }


def decompose(reference, w4, quantized):
    ref = reference.detach().double()
    base = w4.detach().double()
    pred = quantized.detach().double()
    if ref.shape != base.shape or ref.shape != pred.shape:
        raise ValueError(f"action shape mismatch: {tuple(ref.shape)}, {tuple(base.shape)}, {tuple(pred.shape)}")
    dw = base - ref
    da = pred - base
    total = pred - ref
    mse_w = float(dw.square().mean())
    mse_a = float(da.square().mean())
    twice_inner = float((2 * dw * da).mean())
    total_mse = float(total.square().mean())
    residual = total_mse - (mse_w + mse_a + twice_inner)
    return {
        "mse_w": mse_w,
        "mse_a": mse_a,
        "twice_inner_product": twice_inner,
        "total_mse": total_mse,
        "identity_residual": residual,
    }


def mean(values):
    return sum(values) / len(values)


def bootstrap_ci(values, seed=91573):
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_DRAWS))
    return [samples[49], samples[1949]]


def summarize(rows):
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    summaries = {}
    for arm, arm_rows in by_arm.items():
        names = [key for key in arm_rows[0] if key not in {"arm", "episode", "position", "sampler_seed", "draw_seed"}]
        trajectory = {}
        for episode in TEST_EPISODES:
            episode_rows = [r for r in arm_rows if r["episode"] == episode]
            trajectory[str(episode)] = {name: mean([float(r[name]) for r in episode_rows]) for name in names}
        overall = {name: mean([trajectory[str(e)][name] for e in TEST_EPISODES]) for name in names}
        intervals = {name: bootstrap_ci([trajectory[str(e)][name] for e in TEST_EPISODES]) for name in names}
        summaries[arm] = {
            "conditions": len(arm_rows),
            "per_trajectory": trajectory,
            "overall_equal_trajectory_mean": overall,
            "trajectory_bootstrap_95pct_conditional_on_frozen_draws": intervals,
        }
    return summaries


def run(args):
    guard()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    data = load_test_data(args.data, torch)
    locked = json.loads(args.locked.read_text(encoding="utf-8"))
    learned = locked["learned"]
    screen = Screen(args.out)
    records = []
    metric_rows = []
    try:
        # Rebuild the runtime site trace and verify every cached TEST reference before W4 mutation.
        refs, ref_commands = {}, {}
        ref_gate_max = 0.0
        for datum in data:
            key_base = (int(datum["episode"]), int(datum["position"]))
            for sampler_seed in SAMPLER_SEEDS:
                output, _, _ = screen.infer(datum, "bf16", sampler_seed)
                cached = cached_ref(datum, sampler_seed).detach().float().cpu()
                error = float((output - cached).abs().max())
                ref_gate_max = max(ref_gate_max, error)
                if error > 1e-6:
                    raise AssertionError(("CACHED_BF16_LABEL", key_base, sampler_seed, error))
                refs[(*key_base, sampler_seed)] = output.detach().float().cpu()
                command = environment_action(screen, output)
                ref_commands[(*key_base, sampler_seed)] = command
                records.append(record("bf16", datum, sampler_seed, None, output, command))

        first = data[0]
        first_key = (int(first["episode"]), int(first["position"]), SAMPLER_SEEDS[0])
        repeat, _, _ = screen.infer(first, "bf16", SAMPLER_SEEDS[0])
        unhooked, _, _ = screen.infer(first, "bf16", SAMPLER_SEEDS[0], trace=False)
        repeat_error = float((repeat - refs[first_key]).abs().max())
        identity_error = float((unhooked - refs[first_key]).abs().max())
        if repeat_error > 1e-6 or identity_error > 1e-6:
            raise AssertionError(("BF16_REPEAT_OR_HOOK_IDENTITY", repeat_error, identity_error))

        sites = set(screen.expected_sites or ())
        if not sites or set(learned) != sites:
            raise ValueError("locked learned phase site table does not exactly match rebuilt TEST execution sites")
        if any(not math.isfinite(float(v)) or not 0 <= float(v) < 1 for v in learned.values()):
            raise ValueError("locked learned phases must be finite and in [0,1)")
        save(args.out / "reference_gate.json", {
            "cached_bf16_label_max_abs": ref_gate_max,
            "repeat_max_abs": repeat_error,
            "hook_disabled_identity_max_abs": identity_error,
            "passed_before_weight_quantize": True,
            "test_observations": len(data),
            "executed_site_count": len(sites),
        })

        screen.q.weight_quantize()
        w4_outputs = {}
        for datum in data:
            base_key = (int(datum["episode"]), int(datum["position"]))
            for sampler_seed in SAMPLER_SEEDS:
                output, stats, _ = screen.infer(datum, "w4", sampler_seed)
                key = (*base_key, sampler_seed)
                w4_outputs[key] = output.detach().float().cpu()
                command = environment_action(screen, output)
                records.append(record("w4", datum, sampler_seed, 0, output, command))
                metric_rows.append({
                    "arm": "w4", "episode": base_key[0], "position": base_key[1],
                    "sampler_seed": sampler_seed, "draw_seed": 0,
                    "total_mse": float((output.double() - refs[key].double()).square().mean()),
                    **command_metrics(command, ref_commands[key]),
                    "overload_codes": int(stats["overload_codes"]),
                })

        for arm, mode, draw_seeds in (
            ("rtn", "rtn", RTN_DRAWS),
            ("learned", "phase", LEARNED_DRAWS),
        ):
            for datum in data:
                base_key = (int(datum["episode"]), int(datum["position"]))
                for i, draw_seed in enumerate(draw_seeds):
                    sampler_seed = SAMPLER_SEEDS[i % len(SAMPLER_SEEDS)]
                    phases = learned if arm == "learned" else None
                    output, stats, _ = screen.infer(
                        datum, mode, sampler_seed, draw_seed, phases=phases
                    )
                    key = (*base_key, sampler_seed)
                    command = environment_action(screen, output)
                    records.append(record(arm, datum, sampler_seed, draw_seed, output, command))
                    metrics = decompose(refs[key], w4_outputs[key], output)
                    if abs(metrics["identity_residual"]) > IDENTITY_TOL:
                        raise AssertionError(("DECOMPOSITION_IDENTITY", arm, base_key, draw_seed, metrics))
                    metric_rows.append({
                        "arm": arm, "episode": base_key[0], "position": base_key[1],
                        "sampler_seed": sampler_seed, "draw_seed": draw_seed,
                        **metrics,
                        **command_metrics(command, ref_commands[key]),
                        "overload_codes": int(stats["overload_codes"]),
                    })

        summary = summarize(metric_rows)
        decomposition_rows = [r for r in metric_rows if r["arm"] in {"rtn", "learned"}]
        identity_max = max(abs(r["identity_residual"]) for r in decomposition_rows)
        result = {
            "scope": "exploratory decomposition on previously observed TEST IDs 8-15, fixed frozen observations only",
            "interpretation_limits": [
                "This is not an independent validation or a replacement gate; the existing recipe-specific NO_GO remains frozen.",
                "Trajectory bootstrap conditions on these frozen action-sampler and quantizer draws.",
                "MSE components describe an exact algebraic error decomposition, not causal shares or a learned weight-error floor.",
                "W4-only keeps the model's original BF16 activation path; it is not uniform A16 activation quantization.",
                "No closed-loop outcomes, latency benefit, or native packed-kernel claim is measured.",
            ],
            "cohort": {"episodes": list(TEST_EPISODES), "observations_per_episode": 2,
                       "sampler_seeds": list(SAMPLER_SEEDS)},
            "draws": {"w4_sampler_conditions": 2, "rtn_paired_draw_seeds": list(RTN_DRAWS),
                      "learned_new_draw_seeds": list(LEARNED_DRAWS),
                      "sampler_assignment": "alternate 2026/2027 by draw index"},
            "reference_gate": {"cached_label_max_abs": ref_gate_max,
                               "repeat_max_abs": repeat_error,
                               "hook_disabled_identity_max_abs": identity_error},
            "decomposition_identity": {"max_abs_residual": identity_max,
                                       "tolerance": IDENTITY_TOL,
                                       "passed": identity_max <= IDENTITY_TOL},
            "per_trajectory_and_overall": summary,
            "decomposition_rows": decomposition_rows,
            "forward_counts": dict(screen.forward_count),
            "forward_budget": {"bf16_label_replay": 32, "bf16_repeat_and_hook_identity": 2,
                                "w4_only": 32, "rtn": 64, "learned": 64, "total": 194},
            "gate_status": "DIAGNOSTIC_ONLY_EXISTING_NO_GO_UNCHANGED",
        }
        torch.save({
            "schema_version": 1,
            "scope": result["scope"],
            "records": records,
        }, args.out / "actions.pt")
        save(args.out / "result.json", result)
        print("DECOMPOSITION_COMPLETE", result["decomposition_identity"], result["forward_counts"], flush=True)
    finally:
        screen.q.close()
        screen.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="existing frozen_observations.pt")
    parser.add_argument("--locked", type=Path, required=True, help="existing locked_selection.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
