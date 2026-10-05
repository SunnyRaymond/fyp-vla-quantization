"""Fixed-observation W4A4/A8 action-MSE worker; run inside a real PBS allocation."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path


PROTOCOL_ID = "fastwam-activation-bit-screen-v1"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MECHANISM_DIR = ROOT / "mechanism-screen"
PAIRED_DIR = ROOT / "paired-followup"
CAL_EPISODES = (4, 5)
DEV_EPISODES = (6, 7)
TEST_EPISODES = tuple(range(8, 16))
SAMPLER_SEEDS = (2026, 2027)
CAL_DRAWS = (101, 102)
DEV_DRAWS = (201, 202)
TEST_DRAWS = (2101, 2102, 2103, 2104)
GROUP_COUNT = 8
SITE_COUNT = 6446
IDENTITY_TOL = 1e-7
PHASE_VALUES = (0.0, 0.25, 0.5, 0.75)
DIRECT_VALUES = (0.94, 0.98, 1.0, 1.02)


def calibration_budget():
    # 16 labels + two identities + 8 groups * 2 sweeps * 4 values * 2 draws * 4 CAL rows + DEV.
    return 18 + 8 * 2 * 4 * 2 * 4 + 2 * 4 * 2


def evaluation_budget(bits):
    arms = 8 if bits == 4 else 7
    return 32 + 2 + 32 + arms * 16 * 4


def save_progress(out, state, status="running", exception=None):
    payload = {
        "stage": state.get("stage", "startup"),
        "completed": int(state.get("completed", 0)),
        "expected": int(state.get("expected", 0)),
        "current": state.get("current"),
        "status": status,
    }
    if exception is not None:
        payload["exception"] = str(exception).replace("\n", " ")[:400]
    (Path(out) / "progress.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def self_check():
    assert calibration_budget() == 546
    assert evaluation_budget(4) == 578
    assert evaluation_budget(8) == 514
    assert 8 * 16 * 4 == 512 and 7 * 16 * 4 == 448
    sites = [f"s{i}" for i in range(8)]
    learned = {site: i / 8 for i, site in enumerate(sites)}
    shuffled = [learned[site] for site in sites]
    random.Random(8842).shuffle(shuffled)
    permuted = dict(zip(sites, shuffled))
    deltas = {round((permuted[s] - learned[s]) % 1, 8) for s in sites}
    assert set(permuted) == set(learned) and len(deltas) > 1


def import_helpers():
    # PBS snapshots may flatten these three scripts into one ARTIFACTS directory.
    for directory in (MECHANISM_DIR, PAIRED_DIR):
        value = str(directory)
        if directory.exists() and value not in sys.path:
            sys.path.append(value)
    script_path = str(SCRIPT_DIR)
    if script_path in sys.path:
        sys.path.remove(script_path)
    sys.path.insert(0, script_path)
    runner = importlib.import_module("runner")
    decompose = importlib.import_module("decompose")
    return runner, decompose


def install_quantizer(screen, quantizer_type, bits):
    old = screen.q
    old_modules = set(old.modules)
    old.close()
    quantizer = quantizer_type(screen.model, group_size=128, activation_bits=bits)
    if set(quantizer.modules) != old_modules:
        quantizer.close()
        raise AssertionError("new quantizer module coverage differs from Screen's original modules")
    if not hasattr(quantizer, "rtn_full_range"):
        quantizer.close()
        raise TypeError("quant.Quantizer must expose public bool rtn_full_range")
    quantizer.rtn_full_range = False
    screen.q = quantizer


def tracked_infer(screen, out, state, expected):
    raw_infer = screen.infer
    context = {"current": None, "stage": "startup"}
    state["expected"] = expected

    def infer(datum, mode, sampler_seed, draw_seed=0, phases=None, clips=None, trace=True):
        current = {
            "stage": context["stage"],
            "arm": context.get("arm"),
            "objective": context.get("objective"),
            "mode": mode,
            "episode": int(datum["episode"]),
            "position": int(datum["position"]),
            "sampler_seed": int(sampler_seed),
            "draw_seed": int(draw_seed),
        }
        state["stage"], state["current"] = context["stage"], current
        save_progress(out, state)
        try:
            output, stats, seconds = raw_infer(
                datum, mode, sampler_seed, draw_seed, phases=phases, clips=clips, trace=trace
            )
            if int(stats.get("nonfinite_values", 0)):
                raise AssertionError(("NONFINITE_QUANTIZER_VALUES", current, stats))
            if mode == "rtn" and context["stage"] == "calibrate-direct":
                state["direct_activation_overload_count"] = state.get(
                    "direct_activation_overload_count", 0
                ) + int(stats.get("activation_overload_count", 0))
            if mode == "phase":
                state["phase_activation_overload_count"] = state.get(
                    "phase_activation_overload_count", 0
                ) + int(stats.get("activation_overload_count", 0))
            if mode in {"phase", "independent"} and int(
                stats.get("activation_overload_count", stats.get("overload_codes", 0))
            ) != 0:
                raise AssertionError(("DITHER_OVERLOAD", current, stats))
            state["completed"] += 1
            state["current"] = current
            save_progress(out, state)
            return output, stats, seconds
        except Exception as exc:
            save_progress(out, state, status="failed", exception=exc)
            raise

    screen.infer = infer
    raw_objective = screen.objective

    def objective(data, mode, parameters, draw_seeds, label):
        context["objective"] = str(label)
        return raw_objective(data, mode, parameters, draw_seeds, label)

    screen.objective = objective
    return context


def load_protocol_data(torch, path, episodes, expected_total):
    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(loaded, (list, tuple)):
        raise TypeError("frozen_observations.pt must contain a list of observation rows")
    all_keys = [(int(d["episode"]), int(d["position"])) for d in loaded]
    all_expected = {(episode, position) for episode in range(4, 16) for position in (0, 1)}
    if len(all_keys) != 24 or set(all_keys) != all_expected:
        raise ValueError(f"frozen file must contain exactly the 24 protocol rows; got {all_keys}")
    selected = [d for d in loaded if int(d["episode"]) in set(episodes)]
    keys = [(int(d["episode"]), int(d["position"])) for d in selected]
    expected = {(episode, position) for episode in episodes for position in (0, 1)}
    if len(keys) != expected_total or set(keys) != expected:
        raise ValueError(f"frozen cohort mismatch: expected {sorted(expected)}, got {keys}")
    for datum in selected:
        for seed in SAMPLER_SEEDS:
            if seed not in datum["refs"] and str(seed) not in datum["refs"]:
                raise ValueError(
                    f"missing cached BF16 label at {datum['episode']}/{datum['position']}/{seed}"
                )
    return sorted(selected, key=lambda d: (int(d["episode"]), int(d["position"])))


def replay_labels(screen, decompose, data, context):
    labels = {}
    max_error = 0.0
    for datum in data:
        base = (int(datum["episode"]), int(datum["position"]))
        for seed in SAMPLER_SEEDS:
            output, _, _ = screen.infer(datum, "bf16", seed)
            cached = decompose.cached_ref(datum, seed).detach().float().cpu()
            error = float((output - cached).abs().max())
            max_error = max(max_error, error)
            if error > 1e-6:
                raise AssertionError(("CACHED_BF16_LABEL", base, seed, error))
            labels[(*base, seed)] = cached.clone()

    first = data[0]
    seed = SAMPLER_SEEDS[0]
    expected = labels[(int(first["episode"]), int(first["position"]), seed)]
    repeat, _, _ = screen.infer(first, "bf16", seed)
    unhooked, _, _ = screen.infer(first, "bf16", seed, trace=False)
    repeat_error = float((repeat - expected).abs().max())
    identity_error = float((unhooked - expected).abs().max())
    if repeat_error > 1e-6 or identity_error > 1e-6:
        raise AssertionError(("BF16_REPEAT_OR_HOOK_IDENTITY", repeat_error, identity_error))
    gate = {
        "cached_bf16_label_max_abs": max_error,
        "repeat_max_abs": repeat_error,
        "hook_disabled_identity_max_abs": identity_error,
        "cached_label_replays": len(data) * len(SAMPLER_SEEDS),
        "identity_forwards": 2,
        "passed_before_weight_quantize": True,
    }
    return labels, gate


def validate_sites(screen):
    sites = list(screen.expected_sites or ())
    if len(sites) != SITE_COUNT or len(set(sites)) != SITE_COUNT:
        raise AssertionError(f"expected {SITE_COUNT} unique executed sites, got {len(sites)}")
    groups = sorted({screen.group(site) for site in sites} - {None, "video.condition"})
    if len(groups) != GROUP_COUNT:
        raise AssertionError(f"expected exactly {GROUP_COUNT} calibration groups, got {groups}")
    return sites, groups


def permute_like_runner(screen, learned):
    effective_sites = [site for site in screen.expected_sites if screen.group(site) is not None]
    shuffled = [learned[site] for site in effective_sites]
    random.Random(8842).shuffle(shuffled)
    permuted = dict(learned)
    permuted.update(zip(effective_sites, shuffled))
    deltas = {round((permuted[site] - learned[site]) % 1, 8) for site in effective_sites}
    return permuted, len(deltas) > 1, effective_sites


def source_protocol(mode, bits):
    return {
        "protocol_id": PROTOCOL_ID,
        "worker": "activation_screen.py",
        "source_helpers": [
            "mechanism-screen/runner.py::Screen, calibrate, objective, table, infer",
            "paired-followup/decompose.py::cached_ref, environment_action, command_metrics, decompose, record, summarize, bootstrap_ci",
        ],
        "mode": mode,
        "bits": bits,
        "group_size": 128,
        "steps": 20,
        "task_id": 0,
        "action_mode": "first_frame",
        "sigma_shift": 1.0,
        "compiled": False,
        "selection_boundary": "CAL episodes 4-5; DEV 6-7; TEST 8-15",
    }


def update_runtime_config(out, bits):
    path = Path(out) / "runtime_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(
        {
            "activation_bits": int(bits),
            "activation_quantization_grid": {
                "headroom": {
                    "delta": "row_maxabs/6" if bits == 4 else "row_maxabs/126",
                    "clamp": [-6, 7] if bits == 4 else [-126, 127],
                },
                "standard_rtn": {
                    "delta": "row_maxabs/7" if bits == 4 else "row_maxabs/127",
                    "clamp": [-7, 7] if bits == 4 else [-127, 127],
                },
            },
            "rtn_full_range_default": False,
        }
    )
    path.write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")


def run_calibration(args, runner, decompose, quantizer_type, torch, state):
    data = load_protocol_data(torch, args.data, (*CAL_EPISODES, *DEV_EPISODES), 8)
    cal = [d for d in data if int(d["episode"]) in CAL_EPISODES]
    dev = [d for d in data if int(d["episode"]) in DEV_EPISODES]
    if len(cal) != 4 or len(dev) != 4:
        raise AssertionError("calibration requires exactly four CAL and four DEV observations")

    screen = runner.Screen(args.out)
    try:
        screen.cfg.EVALUATION.num_inference_steps = 20
        install_quantizer(screen, quantizer_type, args.bits)
        update_runtime_config(args.out, args.bits)
        screen.q.rtn_full_range = False
        context = tracked_infer(screen, args.out, state, calibration_budget())
        context["stage"] = "calibration-reference-replay"
        labels, reference_gate = replay_labels(screen, decompose, data, context)
        sites, groups = validate_sites(screen)
        if state["completed"] != 18:
            raise AssertionError(("CALIBRATION_REFERENCE_BUDGET", state["completed"]))
        screen.q.weight_quantize()

        phase_mode = args.mode == "calibrate-phase"
        mode = "phase" if phase_mode else "rtn"
        context["stage"] = args.mode
        candidates = PHASE_VALUES if phase_mode else DIRECT_VALUES
        checkpoints = screen.calibrate(cal, mode, groups, candidates)
        dev_scores = [
            screen.objective(dev, mode, checkpoint, DEV_DRAWS, f"{args.mode}/dev/{i}")
            for i, checkpoint in enumerate(checkpoints)
        ]
        selected_index = min(range(len(dev_scores)), key=dev_scores.__getitem__)
        selected_parameters = dict(checkpoints[selected_index])
        phase_table = screen.table(selected_parameters) if phase_mode else None
        clip_table = (
            {site: float(selected_parameters.get(screen.group(site), 1.0)) for site in sites}
            if not phase_mode
            else None
        )
        permuted, nongauge, effective_sites = (
            permute_like_runner(screen, phase_table) if phase_mode else (None, None, [])
        )
        if state["completed"] != calibration_budget():
            raise AssertionError(("CALIBRATION_FORWARD_BUDGET", state["completed"]))
        if phase_mode and state.get("phase_activation_overload_count", 0):
            raise AssertionError("phase calibration observed activation overload")

        result = {
            "protocol_id": PROTOCOL_ID,
            "mode": args.mode,
            "bits": args.bits,
            "complete": True,
            "forward_count": state["completed"],
            "forward_counts": dict(screen.forward_count),
            "forward_budget": calibration_budget(),
            "groups": groups,
            "executed_site_count": len(sites),
            "selected_checkpoint": selected_index,
            "selected_parameters": selected_parameters,
            "checkpoints": checkpoints,
            "dev_scores": dev_scores,
            "sourceprotocol": source_protocol(args.mode, args.bits),
            "reference_gate": reference_gate,
            "rtn_full_range": False,
            "calibration": {
                "cal_observations": len(cal),
                "dev_observations": len(dev),
                "sweeps": 2,
                "candidate_values": list(candidates),
                "cal_draws": list(CAL_DRAWS),
                "dev_draws": list(DEV_DRAWS),
            },
        }
        if phase_mode:
            result.update(
                {
                    "learned": phase_table,
                    "permuted": permuted,
                    "nongauge": nongauge,
                    "permutation_nongauge": nongauge,
                    "effective_site_count": len(effective_sites),
                    "phase_overload_count": 0,
                }
            )
        else:
            result.update(
                {
                    "clips": clip_table,
                    "direct_activation_overload_count": int(
                        state.get("direct_activation_overload_count", 0)
                    ),
                }
            )
        if len(sites) != SITE_COUNT or state["completed"] != calibration_budget():
            result["complete"] = False
        runner.save(args.out / "selection.json", result)
        state["stage"] = "complete"
        state["current"] = None
        save_progress(args.out, state, status="complete")
    finally:
        screen.q.close()
        screen.env.close()


def validate_site_table(name, table, sites, value_min=None, value_max=None):
    if not isinstance(table, dict) or set(table) != set(sites) or len(table) != SITE_COUNT:
        raise ValueError(f"{name} must contain exactly the {SITE_COUNT} rebuilt site keys")
    values = [float(v) for v in table.values()]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} contains nonfinite values")
    if value_min is not None and any(
        v < value_min or (v >= value_max if value_max == 1.0 else v > value_max)
        for v in values
    ):
        raise ValueError(f"{name} values are outside [{value_min},{value_max}]")
    return {str(k): float(v) for k, v in table.items()}


def load_selections(args):
    locked = json.loads(args.locked.read_text(encoding="utf-8"))
    phase_selection = direct_selection = None
    if args.bits == 4:
        phase_selection = json.loads(args.phase_selection.read_text(encoding="utf-8"))
        direct_selection = json.loads(args.direct_selection.read_text(encoding="utf-8"))
        for selection, mode in ((phase_selection, "calibrate-phase"), (direct_selection, "calibrate-direct")):
            if (
                selection.get("protocol_id") != PROTOCOL_ID
                or selection.get("mode") != mode
                or int(selection.get("bits", -1)) != 4
                or selection.get("complete") is not True
                or int(selection.get("forward_count", -1)) != calibration_budget()
            ):
                raise ValueError(f"incomplete or mismatched A4 {mode} selection")
    return locked, phase_selection, direct_selection


def prepare_arm_tables(screen, bits, locked, phase_selection, direct_selection, sites, groups):
    old_learned = validate_site_table("locked.learned", locked.get("learned"), sites, 0.0, 1.0)
    old_permuted = validate_site_table("locked.permuted", locked.get("permuted"), sites, 0.0, 1.0)
    old_direct_groups = locked.get("direct")
    if not isinstance(old_direct_groups, dict) or set(old_direct_groups) != set(groups):
        raise ValueError("locked.direct must contain exactly the eight rebuilt group keys")
    old_direct = {site: float(old_direct_groups.get(screen.group(site), 1.0)) for site in sites}
    if not all(math.isfinite(v) and v > 0 for v in old_direct.values()):
        raise ValueError("locked direct clip table contains invalid values")

    if bits == 4:
        learned_a4 = validate_site_table("phase_selection.learned", phase_selection.get("learned"), sites, 0.0, 1.0)
        permuted_a4 = validate_site_table("phase_selection.permuted", phase_selection.get("permuted"), sites, 0.0, 1.0)
        clips_a4 = validate_site_table("direct_selection.clips", direct_selection.get("clips"), sites)
        if any(v <= 0 for v in clips_a4.values()):
            raise ValueError("A4 direct clips must be positive")
        if sorted(phase_selection.get("groups", [])) != groups or sorted(direct_selection.get("groups", [])) != groups:
            raise ValueError("A4 selections do not use the rebuilt eight groups")
        return old_learned, old_permuted, old_direct, learned_a4, permuted_a4, clips_a4
    return old_learned, old_permuted, old_direct, None, None, None


def finite_metric_row(row):
    for key, value in row.items():
        if key in {"arm", "episode", "position", "sampler_seed", "draw_seed"}:
            continue
        if isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            raise AssertionError(("NONFINITE_METRIC", key, value))


def metric_row(decompose, helper, arm, datum, sampler_seed, draw_seed, reference, w4, output, stats, seconds, reference_command, command):
    components = decompose.decompose(reference, w4, output)
    if abs(components["identity_residual"]) > IDENTITY_TOL:
        raise AssertionError(("DECOMPOSITION_IDENTITY", arm, datum["episode"], datum["position"], draw_seed, components))
    row = {
        "arm": arm,
        "episode": int(datum["episode"]),
        "position": int(datum["position"]),
        "sampler_seed": int(sampler_seed),
        "draw_seed": int(draw_seed),
        **components,
        **helper.command_metrics(command, reference_command),
        "overload_codes": int(stats.get("overload_codes", 0)),
        "activation_overload_count": int(stats.get("activation_overload_count", 0)),
        "forward_seconds_fake_quant": float(seconds),
    }
    finite_metric_row(row)
    return row


def paired_gains(summary, method, baseline, bootstrap_ci):
    method_values = [summary[method]["per_trajectory"][str(e)]["total_mse"] for e in TEST_EPISODES]
    baseline_values = [summary[baseline]["per_trajectory"][str(e)]["total_mse"] for e in TEST_EPISODES]
    absolute = [b - m for b, m in zip(baseline_values, method_values)]
    relative = [(b - m) / max(b, 1e-20) for b, m in zip(baseline_values, method_values)]
    return {
        "method": method,
        "baseline": baseline,
        "absolute_gain_equal_trajectory_mean": decompose_mean(absolute),
        "relative_gain_ratio_of_equal_trajectory_means": (decompose_mean(baseline_values) - decompose_mean(method_values)) / max(decompose_mean(baseline_values), 1e-20),
        "per_trajectory_absolute_gain": {str(e): value for e, value in zip(TEST_EPISODES, absolute)},
        "per_trajectory_relative_gain": {str(e): value for e, value in zip(TEST_EPISODES, relative)},
        "absolute_gain_trajectory_bootstrap_95pct_conditional_on_frozen_draws": bootstrap_ci(absolute),
        "relative_gain_trajectory_bootstrap_95pct_conditional_on_frozen_draws": bootstrap_ci(relative),
        "trajectories_improved": sum(value > 0 for value in absolute),
    }


def decompose_mean(values):
    return sum(values) / len(values)


def run_evaluation(args, runner, decompose, quantizer_type, torch, state):
    data = load_protocol_data(torch, args.data, TEST_EPISODES, 16)
    locked, phase_selection, direct_selection = load_selections(args)
    expected = evaluation_budget(args.bits)
    screen = runner.Screen(args.out)
    records, metric_rows = [], []
    try:
        screen.cfg.EVALUATION.num_inference_steps = 20
        install_quantizer(screen, quantizer_type, args.bits)
        update_runtime_config(args.out, args.bits)
        screen.q.rtn_full_range = False
        context = tracked_infer(screen, args.out, state, expected)
        context["stage"] = "evaluation-reference-replay"
        labels, reference_gate = replay_labels(screen, decompose, data, context)
        sites, groups = validate_sites(screen)
        old_learned, old_permuted, old_direct, learned_a4, permuted_a4, clips_a4 = prepare_arm_tables(
            screen, args.bits, locked, phase_selection, direct_selection, sites, groups
        )
        if state["completed"] != 34:
            raise AssertionError(("EVALUATION_REFERENCE_BUDGET", state["completed"]))

        ref_commands = {}
        for datum in data:
            for seed in SAMPLER_SEEDS:
                key = (int(datum["episode"]), int(datum["position"]), seed)
                ref_commands[key] = decompose.environment_action(screen, labels[key])
                records.append(
                    decompose.record("bf16", datum, seed, None, labels[key], ref_commands[key])
                )

        # One in-place W4 mutation follows all teacher labels and identity checks.
        screen.q.weight_quantize()
        w4_outputs = {}
        context["stage"] = "w4-original-activation"
        for datum in data:
            base = (int(datum["episode"]), int(datum["position"]))
            for seed in SAMPLER_SEEDS:
                key = (*base, seed)
                output, stats, seconds = screen.infer(datum, "w4", seed)
                w4_outputs[key] = output.detach().float().cpu().clone()
                command = decompose.environment_action(screen, output)
                records.append(decompose.record("w4", datum, seed, 0, output, command))
                row = metric_row(
                    decompose, decompose, "w4", datum, seed, 0, labels[key], output, output,
                    stats, seconds, ref_commands[key], command,
                )
                metric_rows.append(row)
        if state["completed"] != 66:
            raise AssertionError(("W4_FORWARD_BUDGET", state["completed"]))

        arms = [
            ("rtn", "rtn", {}, {}, False),
            ("rtn_standard", "rtn", {}, {}, True),
            ("independent", "independent", {}, {}, False),
            ("shared0", "phase", {}, {}, False),
            ("learned_a8_transfer", "phase", old_learned, {}, False),
        ]
        if args.bits == 4:
            arms.extend(
                [
                    ("learned_a4", "phase", learned_a4, {}, False),
                    ("permuted", "phase", permuted_a4, {}, False),
                    ("direct", "rtn", {}, clips_a4, False),
                ]
            )
        else:
            arms.extend(
                [
                    ("permuted", "phase", old_permuted, {}, False),
                    ("direct", "rtn", {}, old_direct, False),
                ]
            )

        context["stage"] = "evaluation-quantized-arms"
        for datum in data:
            base = (int(datum["episode"]), int(datum["position"]))
            for draw_index, draw_seed in enumerate(TEST_DRAWS):
                sampler_seed = SAMPLER_SEEDS[draw_index % len(SAMPLER_SEEDS)]
                order = list(arms)
                random.Random(9381 + base[0] * 100 + base[1] * 10 + draw_index).shuffle(order)
                for arm, mode, phases, clips, full_range in order:
                    context["arm"] = arm
                    screen.q.rtn_full_range = full_range
                    key = (*base, sampler_seed)
                    try:
                        output, stats, seconds = screen.infer(
                            datum, mode, sampler_seed, draw_seed,
                            phases=phases or None, clips=clips or None,
                        )
                    finally:
                        screen.q.rtn_full_range = False
                    activation_overload = int(stats.get("activation_overload_count", 0))
                    if arm != "direct" and activation_overload != 0:
                        raise AssertionError(("UNEXPECTED_ACTIVATION_OVERLOAD", arm, base, draw_seed, stats))
                    reference = labels[key]
                    command = decompose.environment_action(screen, output)
                    records.append(
                        decompose.record(arm, datum, sampler_seed, draw_seed, output, command)
                    )
                    metric_rows.append(
                        metric_row(
                            decompose, decompose, arm, datum, sampler_seed, draw_seed,
                            reference, w4_outputs[key], output, stats, seconds,
                            ref_commands[key], command,
                        )
                    )
        if state["completed"] != expected:
            raise AssertionError(("EVALUATION_FORWARD_BUDGET", state["completed"], expected))

        expected_rows = 32 + len(arms) * 64
        if len(metric_rows) != expected_rows:
            raise AssertionError(("METRIC_ROW_COUNT", len(metric_rows), expected_rows))
        condition_keys = [
            (r["arm"], r["episode"], r["position"], r["sampler_seed"], r["draw_seed"])
            for r in metric_rows if r["arm"] != "w4"
        ]
        if len(condition_keys) != len(set(condition_keys)):
            raise AssertionError("duplicate quantized metric condition")
        arm_counts = {arm[0]: sum(r["arm"] == arm[0] for r in metric_rows) for arm in arms}
        if any(count != 64 for count in arm_counts.values()):
            raise AssertionError(("ARM_ROW_COUNT", arm_counts))
        summary = decompose.summarize(metric_rows)

        methods = ["learned_a8_transfer"] + (["learned_a4"] if args.bits == 4 else [])
        baselines = ["rtn", "rtn_standard", "independent", "shared0", "direct"]
        gains = {
            method: {baseline: paired_gains(summary, method, baseline, decompose.bootstrap_ci) for baseline in baselines}
            for method in methods
        }
        learned_arm = "learned_a4" if args.bits == 4 else "learned_a8_transfer"
        permutation = paired_gains(summary, learned_arm, "permuted", decompose.bootstrap_ci)
        permutation_check = {
            "nongauge": bool(
                phase_selection.get("permutation_nongauge") if args.bits == 4 else locked.get("permutation_nongauge", False)
            ),
            "paired_gain_learned_vs_permuted": permutation,
            "mean_mse_learned": summary[learned_arm]["overall_equal_trajectory_mean"]["total_mse"],
            "mean_mse_permuted": summary["permuted"]["overall_equal_trajectory_mean"]["total_mse"],
            "trajectories_permuted_worse": permutation["trajectories_improved"],
        }
        permutation_check["pass"] = bool(
            permutation_check["nongauge"]
            and permutation_check["mean_mse_permuted"] > permutation_check["mean_mse_learned"]
            and permutation_check["trajectories_permuted_worse"] >= 6
        )

        main_baseline = min(
            ("rtn", "rtn_standard", "independent", "direct"),
            key=lambda arm: summary[arm]["overall_equal_trajectory_mean"]["total_mse"],
        )
        main_gain = gains[learned_arm][main_baseline]
        science_gates = {
            "relative_gain_at_least_10pct": main_gain["relative_gain_ratio_of_equal_trajectory_means"] >= 0.10,
            "improves_at_least_6_of_8_trajectories": main_gain["trajectories_improved"] >= 6,
            "absolute_gain_bootstrap_lower_positive": main_gain[
                "absolute_gain_trajectory_bootstrap_95pct_conditional_on_frozen_draws"
            ][0] > 0,
            "permutation_nongauge_and_removes_gain": permutation_check["pass"],
        }
        gates = {
            "cached_bf16_labels_pass": reference_gate["cached_bf16_label_max_abs"] <= 1e-6,
            "repeat_and_identity_pass": max(
                reference_gate["repeat_max_abs"], reference_gate["hook_disabled_identity_max_abs"]
            ) <= 1e-6,
            "executed_site_count_6446": len(sites) == SITE_COUNT,
            "group_count_8": len(groups) == GROUP_COUNT,
            "finite_outputs_and_metrics": all(
                math.isfinite(float(value))
                for row in metric_rows
                for key, value in row.items()
                if key not in {"arm", "episode", "position", "sampler_seed", "draw_seed"}
                and not isinstance(value, bool)
            ),
            "phase_independent_and_standard_rtn_overload_zero": all(
                int(row.get("activation_overload_count", 0)) == 0
                for row in metric_rows
                if row["arm"] in {"rtn", "rtn_standard", "independent", "shared0", "learned_a8_transfer", "learned_a4", "permuted"}
            ),
            "decomposition_identity_within_1e-7": max(
                abs(float(row["identity_residual"])) for row in metric_rows
            ) <= IDENTITY_TOL,
            "condition_rows_complete": len(metric_rows) == expected_rows and all(v == 64 for v in arm_counts.values()),
            "forward_budget_exact": state["completed"] == expected,
        }
        if not all(gates.values()):
            raise AssertionError(("EVALUATION_PROTOCOL_GATE", gates))
        result = {
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "bits": args.bits,
            "mode": "evaluate",
            "metric_rows": metric_rows,
            "per_trajectory_and_overall": summary,
            "reference_gate": reference_gate,
            "forward_counts": dict(screen.forward_count),
            "forward_budget": {"expected": expected, "observed": state["completed"]},
            "forward_count": state["completed"],
            "arm_row_counts": arm_counts,
            "gates": gates,
            "science_gates": science_gates,
            "paired_gains": gains,
            "permutation_check": permutation_check,
            "strongest_matched_baseline": main_baseline,
            "scope": "fixed-observation action MSE on previously observed TEST IDs 8-15; task0 only",
            "interpretation_limits": [
                "TEST IDs were observed by the earlier frozen task0 experiment; this is a reused-data diagnostic, not independent validation.",
                "No closed-loop outcome, native latency, or packed-kernel claim is measured.",
                "The W4 reference arm retains original BF16 activations; it is not W4A16 terminology evidence.",
                "Trajectory bootstrap intervals condition on the frozen sampler and quantizer draws.",
                "MSE decomposition is an algebraic identity, not a causal attribution.",
            ],
            "cohort": {"test_episodes": list(TEST_EPISODES), "observations_per_episode": 2, "sampler_seeds": list(SAMPLER_SEEDS)},
            "draws": {"dither_seeds": list(TEST_DRAWS), "sampler_assignment": "alternate 2026/2027 by draw index"},
            "sourceprotocol": source_protocol("evaluate", args.bits),
        }
        torch.save({"schema_version": 1, "scope": result["scope"], "records": records}, args.out / "actions.pt")
        runner.save(args.out / "result.json", result)
        state["stage"], state["current"] = "complete", None
        save_progress(args.out, state, status="complete")
    finally:
        screen.q.close()
        screen.env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate-phase", "calibrate-direct", "evaluate"), required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="frozen_observations.pt")
    parser.add_argument("--locked", type=Path, required=True, help="previous A8 locked_selection.json")
    parser.add_argument("--phase-selection", type=Path)
    parser.add_argument("--direct-selection", type=Path)
    args = parser.parse_args()
    if args.mode == "evaluate" and args.bits == 4 and (args.phase_selection is None or args.direct_selection is None):
        parser.error("evaluate --bits 4 requires --phase-selection and --direct-selection")
    return args


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state = {"stage": "startup", "completed": 0, "expected": 0, "current": None}
    try:
        runner, decompose = import_helpers()
        runner.guard()
        self_check()
        from quant import Quantizer

        quantizer_type = Quantizer
        import torch

        if args.mode.startswith("calibrate-"):
            state["expected"] = calibration_budget()
            run_calibration(args, runner, decompose, quantizer_type, torch, state)
        else:
            state["expected"] = evaluation_budget(args.bits)
            run_evaluation(args, runner, decompose, quantizer_type, torch, state)
    except Exception as exc:
        save_progress(args.out, state, status="failed", exception=exc)
        raise


if __name__ == "__main__":
    main()
