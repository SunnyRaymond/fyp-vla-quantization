"""Resumable two-round action+future search over fixed microscale-VQ banks."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import statistics
import tempfile
import time
import traceback


ACTION_METRICS = (
    "motor_rmse_first10_vs_bf16",
    "gripper_rmse_first10_vs_bf16",
    "motor_rmse_32_vs_bf16",
    "motor_rmse_late22_vs_bf16",
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def _mean_metric(values):
    if isinstance(values[0], list):
        return [_mean_metric([value[index] for value in values]) for index in range(len(values[0]))]
    return statistics.mean(values)


def aggregate_future(rows):
    if not rows:
        return {"case_count": 0, "context_count": 0, "per_case": {}}
    names = ("future_latent_rmse", "future_latent_nmse", "future_rmse", "future_nmse",
             "temporal_delta_global_rmse", "temporal_delta_global_nmse",
             "temporal_delta_rmse", "temporal_delta_nmse", "clamp_firstframe_max_abs")
    groups = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    per_case = {str(case): {name: _mean_metric([row[name] for row in records]) for name in names}
                for case, records in groups.items()}
    return {"case_count": len(groups), "context_count": len(rows),
            "metrics": {name: _mean_metric([value[name] for value in per_case.values()]) for name in names},
            "per_case": per_case}


def aggregate_action(rows):
    if not rows:
        return {"case_count": 0, "context_count": 0, "per_case": {}}
    groups = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    per_case = {str(case): {name: statistics.mean(float(row[name]) for row in records)
                            for name in ACTION_METRICS}
                for case, records in groups.items()}
    means = {name: statistics.mean(values[name] for values in per_case.values())
             for name in ACTION_METRICS}
    return {"case_count": len(groups), "context_count": len(rows), "per_case": per_case,
            "mean_motor_rmse_first10": means[ACTION_METRICS[0]],
            "mean_gripper_rmse_first10": means[ACTION_METRICS[1]],
            "mean_motor_rmse_32": means[ACTION_METRICS[2]],
            "mean_motor_rmse_late22": means[ACTION_METRICS[3]]}


def _p90(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("P90 requires at least one case")
    position = (len(ordered) - 1) * 0.9
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def selection_entry(epoch, action_rows, future_rows):
    action = aggregate_action(action_rows)
    if action["case_count"] != 4 or action["context_count"] != 8:
        raise RuntimeError("Each selection epoch must cover exactly four cases and eight contexts")
    return {"epoch": int(epoch), "case_count": action["case_count"], "context_count": action["context_count"],
            "mean_motor_rmse_first10": action["mean_motor_rmse_first10"],
            "mean_gripper_rmse_first10": action["mean_gripper_rmse_first10"],
            "mean_motor_rmse_32": action["mean_motor_rmse_32"],
            "mean_motor_rmse_late22": action["mean_motor_rmse_late22"],
            "p90_motor_rmse_first10": _p90(
                [row[ACTION_METRICS[0]] for row in action["per_case"].values()]),
            "per_case": action["per_case"], "future_metrics": aggregate_future(future_rows)}


def numerical_contract(protocol):
    keys = ("protocol", "source_root", "source_full_job", "roots", "initial_arm", "target_linears",
            "calibration_cases", "calibration_seed_indices", "selection_cases", "test_cases",
            "evaluation_seed_indices", "teacher", "loss", "trainables", "fixed", "optimizer",
            "learning_rate", "gradient_norm_clip", "maximum_epochs", "updates_per_epoch", "case_order",
            "checkpoint_selection", "training_gradients", "checkpointing", "walltime_seconds",
            "budget_rule", "resume_state", "parity_max_abs_tolerance",
            "parity_rmse_tolerance", "reuse_boundary", "claim_scope", "reference_cache",
            "reference_summary", "teacher_reference_cache", "teacher_reference_summary", "bank_root",
            "bank_full_job", "microscaling", "search", "candidate_run_id", "rho", "round",
            "normalization_c", "latent_weight")
    return {key: protocol[key] for key in keys}


def latent_weight_for_rho(rho, normalization_c):
    rho, normalization_c = float(rho), float(normalization_c)
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    if not math.isfinite(normalization_c) or normalization_c <= 0:
        raise ValueError("normalization_c must be finite and positive")
    weight = rho * normalization_c
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("rho * normalization_c must be finite and nonnegative")
    return weight


def advance_training(state, protocol):
    state["updates_completed"] += 1
    state["next_train_index"] += 1
    if state["next_train_index"] == len(protocol["calibration_cases"]):
        state["epochs_completed"] = state["epoch"]
        state["phase"] = "selection"


def evaluation_pairs(cases, seeds, rows):
    expected = [(case, seed) for case in cases for seed in seeds]
    completed = [(row["case_id"], row["seed_index"]) for row in rows]
    if completed != expected[:len(completed)]:
        raise RuntimeError("Saved evaluation is not a unique completed prefix of the frozen cohort")
    return expected[len(completed):]


def validate_run_binding(protocol, run_id, rho, round_number, mode, freeze=None, resume_dir=None):
    """Validate CLI identity against the frozen first-round, chooser, or final binding."""
    rho = float(rho)
    round_number = int(round_number)
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    search = protocol["search"]
    if mode == "search":
        if not run_id or not math.isfinite(rho):
            raise ValueError("Search requires a finite rho and nonempty run ID")
        if round_number == 1:
            expected = {f"r1-r{index:03d}": float(value)
                        for index, value in enumerate(search["round1_rhos"])}
            additional = protocol.get("additional_round1_candidates", [])
            if not isinstance(additional, list):
                raise ValueError("Additional round 1 candidates must be a frozen list")
            for item in additional:
                if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
                    raise ValueError("Additional round 1 candidate identity is malformed")
                candidate_rho = float(item.get("rho"))
                candidate_id = item["run_id"]
                if (not candidate_id.startswith("r1-r") or not math.isfinite(candidate_rho)
                        or candidate_rho < 0 or candidate_id in expected
                        or candidate_rho in expected.values()):
                    raise ValueError("Additional round 1 candidate is duplicate or invalid")
                expected[candidate_id] = candidate_rho
            if (freeze is not None or len(set(expected.values())) != len(expected)
                    or expected.get(run_id) != rho):
                raise ValueError("Round 1 run ID/rho must match one frozen candidate and no chooser freeze file")
        elif round_number == 2:
            if not isinstance(freeze, dict) or freeze.get("protocol") != protocol["protocol"] or freeze.get("round") != 2:
                raise ValueError("Round 2 run requires the frozen round2_candidates.json")
            if not freeze.get("selection_job_id"):
                raise ValueError("Round 2 candidate freeze is missing its chooser job ID")
            candidates = freeze.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 3:
                raise ValueError("Round 2 freeze must contain exactly three candidates")
            identities = [(item.get("run_id"), float(item.get("rho"))) for item in candidates]
            if len(set(identities)) != 3 or (run_id, rho) not in identities:
                raise ValueError("Round 2 run ID/rho is not one of the three frozen candidates")
        else:
            raise ValueError("Search round must be 1 or 2")
        return {"candidate_run_id": run_id, "rho": rho, "round": round_number,
                "selected_run_id": None, "selected_epoch": None}
    if mode != "final":
        raise ValueError("mode must be search or final")
    required = ("selected_run_id", "selected_job_id", "selected_result_dir", "selected_epoch", "selected_round",
                "rho", "latent_weight", "normalization_c", "selection_job_id", "selection_source_jobs")
    if (not isinstance(freeze, dict) or freeze.get("protocol") != protocol["protocol"]
            or freeze.get("rounds_complete") != 2 or any(key not in freeze for key in required)):
        raise ValueError("Final mode requires the verified two-round optimal_selection.json binding")
    epoch = freeze["selected_epoch"]
    if run_id != "final" or round_number != 2 or not isinstance(epoch, int) or epoch not in (0, 1, 2):
        raise ValueError("Final mode must use run ID 'final', round 2, and selected epoch 0/1/2")
    if not math.isclose(rho, float(freeze["rho"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Final CLI rho differs from the frozen global selection")
    if (not freeze["selected_job_id"] or not freeze["selection_job_id"] or not freeze["selection_source_jobs"]
            or freeze["selected_round"] not in (1, 2)):
        raise ValueError("Final selection is missing its verified chooser/search job bindings")
    return {"candidate_run_id": freeze["selected_run_id"], "rho": rho, "round": 2,
            "selected_run_id": freeze["selected_run_id"], "selected_epoch": epoch,
            "selected_round": freeze["selected_round"]}


def _reference_key(case, seed):
    return f"{case}:{seed}"


def _parity(torch, actual, expected, future=False):
    left, right = actual.detach().float().cpu(), expected.detach().float().cpu()
    if future:
        left, right = left[:, :, 1:], right[:, :, 1:]
    delta = left - right
    return {"max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt())}


def _metrics_for_action(torch, metrics, action, reference):
    values = metrics.action_metrics(torch, action, reference)
    return {"motor_rmse_first10_vs_bf16": values["first10motor_rmse"],
            "gripper_rmse_first10_vs_bf16": values["first10gripper_rmse"],
            "motor_rmse_32_vs_bf16": values["full32motor_rmse"],
            "motor_rmse_late22_vs_bf16": values["late22motor_rmse"]}


def _assert_reference_tensor(torch, row, key, shape):
    value = row.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise RuntimeError(f"Invalid cached reference {key} at {row.get('case_id')}:{row.get('seed_index')}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"Nonfinite cached reference {key} at {row.get('case_id')}:{row.get('seed_index')}")


def _validate_reference_cache(torch, p, cache, summary, latent_reference, metrics):
    if (cache.get("schema_version") != 2 or cache.get("protocol") != p["protocol"]
            or cache.get("source_root") != p["source_root"]
            or cache.get("source_full_job") != p["source_full_job"]
            or cache.get("bank_root") != p["bank_root"]
            or cache.get("bank_full_job") != p["bank_full_job"]
            or cache.get("initial_arm") != p["initial_arm"]):
        raise RuntimeError("Schema-2 reference cache identity differs from the frozen source/bank roots")
    ref = latent_reference.resume_reference_contract(summary)
    if (summary.get("protocol") != p["protocol"] or summary.get("source_root") != p["source_root"]
            or summary.get("source_full_job") != p["source_full_job"]
            or summary.get("bank_root") != p["bank_root"]
            or summary.get("bank_full_job") != p["bank_full_job"]
            or summary.get("initial_arm") != p["initial_arm"]):
        raise RuntimeError("Reference summary identity differs from the frozen source/bank roots")
    if (cache.get("loss_normalization") != summary.get("loss_normalization")
            or ref["calibration_ids"] != p["calibration_cases"]):
        raise RuntimeError("Cache normalization or calibration IDs differ from its summary")
    expected_teacher = {_reference_key(row["case_id"], row["seed_index"]) for row in metrics.frozen_contexts(p)}
    expected_parent = {_reference_key(case, seed) for case in p["calibration_cases"] + p["selection_cases"]
                       for seed in ([0] if case in p["calibration_cases"] else p["evaluation_seed_indices"])}
    if set(cache.get("teacher", {})) != expected_teacher or set(cache.get("parent", {})) != expected_parent:
        raise RuntimeError("Reference cache does not cover exactly teacher60 and micro parent16")
    expected_selection = {_reference_key(case, seed) for case in p["selection_cases"]
                          for seed in p["evaluation_seed_indices"]}
    if set(cache.get("initial_selection_future_metrics", {})) != expected_selection:
        raise RuntimeError("Initial selection future metrics do not cover the frozen eight contexts")
    for cache_name in ("teacher", "parent"):
        for key, row in cache[cache_name].items():
            if row.get("case_id") is None or row.get("seed_index") is None:
                raise RuntimeError(f"Reference metadata is missing at {cache_name}/{key}")
            _assert_reference_tensor(torch, row, "action", (32, 7))
            _assert_reference_tensor(torch, row, "future", (1, 48, 3, 14, 28))
    return ref


def _load_micro_actions(torch, p, source, source_root):
    bank_full = Path(p["bank_root"]) / "results" / "full"
    bank_summary = json.loads((bank_full / "summary.json").read_text(encoding="utf-8"))
    bank_protocol = json.loads((bank_full / "protocol.json").read_text(encoding="utf-8"))
    if (bank_summary.get("status") != "complete" or bank_summary.get("job_id") != p["bank_full_job"]
            or bank_protocol.get("protocol") != "fastwam-vq-microscale-v1"
            or bank_protocol.get("source_root") != p["source_root"]
            or bank_protocol.get("source_full_job") != p["source_full_job"]
            or bank_protocol.get("new_arm") != p["initial_arm"]):
        raise RuntimeError("Existing micro bank run is not the declared completed source")
    old_plan = json.loads((Path(p["roots"]["old_inputs"]) / "plan.json").read_text(encoding="utf-8"))
    fresh_plan = json.loads((source_root / "plan.json").read_text(encoding="utf-8"))
    fresh = {int(row["case_id"]): row for row in fresh_plan["inputs"]}

    def manifest(case):
        return source.validate_plan(old_plan, case) if case < 1000 else fresh[case]

    rows = {}
    with (bank_full / "actions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("arm") != p["initial_arm"]:
                continue
            key = (int(row["case_id"]), int(row["seed_index"]))
            if key in rows:
                raise RuntimeError(f"Duplicate cached micro action {key}")
            if row.get("split") not in ("selection", "test") or row.get("observed_module_count") != 614:
                raise RuntimeError(f"Unexpected cached micro action metadata at {key}")
            expected_seed = manifest(key[0])["sampler_seeds"][key[1]]
            action = torch.tensor(row["action"], dtype=torch.float32)
            if (row.get("sampler_seed") != expected_seed or tuple(action.shape) != (32, 7)
                    or not bool(torch.isfinite(action).all())):
                raise RuntimeError(f"Cached micro action identity/value differs at {key}")
            rows[key] = {**row, "action_tensor": action}
    expected = {(case, seed) for case in p["selection_cases"] + p["test_cases"]
                for seed in p["evaluation_seed_indices"]}
    if set(rows) != expected:
        raise RuntimeError("Existing micro run must contain exactly selection8 and test44 actions")
    return rows, manifest


def _initial_action_rows(torch, metrics, p, cache, micro_actions, split):
    cases = p["selection_cases"] if split == "selection" else p["test_cases"]
    rows = []
    for case in cases:
        for seed in p["evaluation_seed_indices"]:
            key = (case, seed)
            target = cache["teacher"][_reference_key(case, seed)]["action"].float()
            action = micro_actions[key]["action_tensor"]
            rows.append({"case_id": case, "seed_index": seed, "split": split,
                         "arm": p["initial_arm"], **_metrics_for_action(torch, metrics, action, target)})
    return rows


def _verify_selection_cache_parity(p, cache, micro_actions):
    for case in p["selection_cases"]:
        for seed in p["evaluation_seed_indices"]:
            key = _reference_key(case, seed)
            actual = micro_actions[(case, seed)]["action_tensor"]
            expected = cache["parent"][key]["action"]
            parity = _parity(None, actual, expected)
            if (parity["max_abs"] > p["parity_max_abs_tolerance"]
                    or parity["rmse"] > p["parity_rmse_tolerance"]):
                raise RuntimeError(f"Existing micro selection action differs from reference cache: {case}:{seed} {parity}")


def run(args):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import metrics
    import latent_reference
    from train_banks import TrainableBanks
    from checkpointing import atomic_torch_save, capture_rng, optimizer_to_device, restore_rng, to_cpu

    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p.get("protocol") != "fastwam-vq-micro-latent-search-v1":
        raise ValueError("Unexpected micro latent search protocol")
    freeze = json.loads(args.freeze_file.read_text(encoding="utf-8")) if args.freeze_file else None
    selected_protocol = None
    final_resume_meta = None
    if args.mode == "final" and freeze is not None:
        selected_dir = Path(freeze["selected_result_dir"])
        selected_protocol = json.loads((selected_dir / "protocol.json").read_text(encoding="utf-8"))
        if (selected_protocol.get("candidate_run_id") != freeze.get("selected_run_id")
                or selected_protocol.get("rho") != float(freeze["rho"])
                or selected_protocol.get("round") not in (1, 2)
                or selected_protocol.get("round") != freeze.get("selected_round")):
            raise RuntimeError("Frozen final candidate directory has a different run/rho/round binding")
        if args.resume_dir and args.resume_dir.resolve() != selected_dir.resolve():
            resume_dir = args.resume_dir.resolve()
            if resume_dir.parent != (Path(p["remote_root"]) / "results").resolve():
                raise RuntimeError("Final evaluation resume directory is outside this campaign")
            final_resume_meta = {
                "protocol": json.loads((resume_dir / "protocol.json").read_text(encoding="utf-8")),
                "summary": json.loads((resume_dir / "summary.json").read_text(encoding="utf-8")),
            }
            old_summary = final_resume_meta["summary"]
            if (old_summary.get("status") != "resumable" or old_summary.get("mode") != "final"
                    or old_summary.get("selected_run_id") != freeze["selected_run_id"]
                    or old_summary.get("selected_job_id") != freeze["selected_job_id"]
                    or old_summary.get("selected_epoch") != freeze["selected_epoch"]
                    or old_summary.get("selected_round") != freeze["selected_round"]
                    or old_summary.get("selected_result_dir") != freeze["selected_result_dir"]
                    or old_summary.get("rounds_complete") != 2):
                raise RuntimeError("Final resume directory is not an unfinished result for this frozen selection")
    binding = validate_run_binding(p, args.run_id, args.rho, args.round, args.mode, freeze,
                                   args.resume_dir if args.mode == "final" else None)
    p["execution_run_id"] = args.run_id
    p["candidate_run_id"] = binding["candidate_run_id"]
    p["rho"] = binding["rho"]
    p["round"] = selected_protocol["round"] if selected_protocol is not None else binding["round"]
    p["execution_round"] = binding["round"]
    if args.out.exists():
        raise FileExistsError(f"Existing result directory: {args.out}")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    allocation_started = float(os.environ.get("FW_JOB_START_EPOCH", time.time()))
    deadline = started + p["walltime_seconds"] - (time.time() - allocation_started)
    job_id = os.environ["PBS_JOBID"]
    interrupted = []
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupted.append(signum))
    source_root = Path(p["source_root"])
    source_dir = source_root / "results" / "full"
    old_root = Path(p["roots"]["old_inputs"])
    stage = "reference_loading"
    state, banks, optimizer, runtime = None, None, None, None
    epoch_states = {}
    torch.backends.cuda.matmul.allow_tf32 = False

    def progress(**extra):
        record = {"status": "running", "stage": stage, "job_id": job_id,
                  "run_id": p["execution_run_id"], "candidate_run_id": p["candidate_run_id"],
                  "rho": p["rho"], "round": p["execution_round"],
                  "updates_completed": state["updates_completed"] if state else 0,
                  "epochs_completed": state["epochs_completed"] if state else 0,
                  "elapsed_seconds": time.monotonic() - started, **extra}
        save(args.out / "progress.json", record)
        print("MICRO_LATENT_PROGRESS " + json.dumps(record), flush=True)

    try:
        reference_summary = json.loads(Path(p["reference_summary"]).read_text(encoding="utf-8"))
        reference_cache = torch.load(Path(p["reference_cache"]), map_location="cpu", weights_only=True, mmap=True)
        ref_contract = _validate_reference_cache(torch, p, reference_cache, reference_summary, latent_reference, metrics)
        if not p.get("baseline_optimizer_reset_after_update1"):
            raise RuntimeError("Matched first-update Adam reset must remain enabled")
        p["normalization_c"] = ref_contract["normalization_c"]
        p["latent_weight"] = latent_weight_for_rho(p["rho"], p["normalization_c"])
        if args.mode == "final" and (
                not math.isclose(p["normalization_c"], float(freeze["normalization_c"]), rel_tol=1e-8, abs_tol=1e-12)
                or not math.isclose(p["latent_weight"], float(freeze["latent_weight"]), rel_tol=1e-8, abs_tol=1e-12)):
            raise RuntimeError("Final global selection normalization/lambda differs from the current reference cache")
        contract = numerical_contract(p)
        if final_resume_meta and numerical_contract(final_resume_meta["protocol"]) != contract:
            raise RuntimeError("Final resume changes the selected candidate numerical contract")
        save(args.out / "protocol.json", {**p, "mode": args.mode, "selected_epoch": binding["selected_epoch"]})

        stage = "source_validation"
        progress()
        full_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
        if full_summary.get("status") != "complete" or full_summary.get("pbs_jobid") != p["source_full_job"]:
            raise RuntimeError("BF16 teacher source is not the declared completed run")
        micro_actions, manifest = _load_micro_actions(torch, p, source, source_root)
        _verify_selection_cache_parity(p, reference_cache, micro_actions)
        selection0_rows = _initial_action_rows(torch, metrics, p, reference_cache, micro_actions, "selection")
        test0_rows = _initial_action_rows(torch, metrics, p, reference_cache, micro_actions, "test")
        initial_selection_future_rows = []
        for case in p["selection_cases"]:
            for seed in p["evaluation_seed_indices"]:
                key = _reference_key(case, seed)
                initial_selection_future_rows.append({"case_id": case, "seed_index": seed,
                                                      **reference_cache["initial_selection_future_metrics"][key]})
        selection0 = selection_entry(0, selection0_rows, initial_selection_future_rows)
        initial_selection = aggregate_action(selection0_rows)
        initial_test = aggregate_action(test0_rows)

        initial_calibration_rows = []
        for case in p["calibration_cases"]:
            key = _reference_key(case, 0)
            teacher = reference_cache["teacher"][key]
            parent = reference_cache["parent"][key]
            action_mse = (parent["action"][:10, :6].float() - teacher["action"][:10, :6].float()).square().mean()
            future_nmse = latent_reference.future_nmse_loss(torch, parent["future"], teacher["future"])
            initial_calibration_rows.append({"case_id": case, "seed_index": 0, "split": "calibration",
                                             "action_mse": float(action_mse), "future_nmse": float(future_nmse)})
        derived_norm = latent_reference.calibration_loss_scale(initial_calibration_rows,
                                                               p["calibration_cases"], 1.0)
        if (not math.isclose(derived_norm["normalization_c"], ref_contract["normalization_c"], rel_tol=1e-8, abs_tol=1e-12)
                or not math.isclose(derived_norm["mean_cal_initial_action_mse"], ref_contract["mean_cal_initial_action_mse"], rel_tol=1e-8, abs_tol=1e-12)
                or not math.isclose(derived_norm["mean_cal_initial_future_nmse"], ref_contract["mean_cal_initial_future_nmse"], rel_tol=1e-8, abs_tol=1e-12)):
            raise RuntimeError("Recomputed micro calibration normalization differs from the reference summary")
        p["normalization_c"] = derived_norm["normalization_c"]
        p["latent_weight"] = latent_weight_for_rho(p["rho"], p["normalization_c"])
        contract = numerical_contract(p)
        save(args.out / "protocol.json", {**p, "mode": args.mode, "selected_epoch": binding["selected_epoch"]})

        if args.mode == "search":
            stage = "checkpoint_self_checks"
            progress()
            from test_checkpointing import self_check as checkpoint_test
            from test_driver import self_check as driver_test
            from test_latent_loss import self_check as latent_test
            from test_micro_training import self_check as micro_test
            with tempfile.TemporaryDirectory(prefix="checkpoint-test-", dir=args.out) as directory:
                checkpoint_test(torch, directory)
            driver_test(p)
            latent_test(torch, p)
            micro_test(torch)
            print("MICRO_TRAINING_AND_CURSOR_SELF_CHECKS_PASSED", flush=True)

        payload = None
        final_resume_payload = None
        if args.mode == "search" and args.resume_dir:
            resume_dir = args.resume_dir.resolve()
            if resume_dir.parent != (Path(p["remote_root"]) / "results").resolve():
                raise RuntimeError("Search resume directory is outside this campaign")
            previous = json.loads((resume_dir / "protocol.json").read_text(encoding="utf-8"))
            previous_summary = json.loads((resume_dir / "summary.json").read_text(encoding="utf-8"))
            if (previous_summary.get("status") != "resumable"
                    or numerical_contract(previous) != contract
                    or previous.get("mode") != "search"):
                raise RuntimeError("Resume changes candidate identity or the frozen numerical contract")
            payload = torch.load(resume_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
            if payload.get("schema_version") != 2 or payload.get("numerical_contract") != contract:
                raise RuntimeError("Unsupported or mismatched schema-2 resume checkpoint")
            state = payload["run_state"]
            if (state["latent_weight"] != p["latent_weight"] or state["normalization_c"] != p["normalization_c"]
                    or state["candidate_run_id"] != p["candidate_run_id"] or state["rho"] != p["rho"]
                    or state["round"] != p["round"]):
                raise RuntimeError("Checkpoint rho, run, or normalization binding differs")
            epoch_states = payload["epoch_states"]
        elif args.mode == "final" and final_resume_meta:
            final_resume_payload = torch.load(args.resume_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
            if (final_resume_payload.get("schema_version") != 2
                    or final_resume_payload.get("numerical_contract") != contract):
                raise RuntimeError("Final resume checkpoint has an unsupported schema or changed binding")
            state = final_resume_payload["run_state"]
            if (state.get("mode") != "final" or state.get("selected_run_id") != freeze["selected_run_id"]
                    or state.get("selected_job_id") != freeze["selected_job_id"]
                    or state.get("selected_epoch") != binding["selected_epoch"]):
                raise RuntimeError("Final checkpoint selection binding differs from optimal_selection.json")
            epoch_states = final_resume_payload["epoch_states"]

        stage = "model_loading"
        progress()
        runtime = pilot.Runtime(args.out)
        runtime.arm("bf16")
        banks = TrainableBanks(runtime, p["bank_root"], p["initial_arm"])
        parameters = banks.parameters()
        if args.mode == "search":
            optimizer = torch.optim.Adam(parameters, lr=p["learning_rate"])

        if args.mode == "search" and payload is None:
            epoch_states = {"0": banks.snapshot()}
            state = {"mode": "search", "phase": "train", "epoch": 1, "next_train_index": 0,
                     "updates_completed": 0, "epochs_completed": 0, "candidate_run_id": p["candidate_run_id"],
                     "rho": p["rho"], "round": p["round"], "normalization_c": p["normalization_c"],
                     "latent_weight": p["latent_weight"], "parity_records": [], "parity_validated": False,
                     "gradient_parity": None, "training_history": [], "evaluation_rows": [],
                     "selection_history": [selection0], "gradient_seconds": [], "forward_seconds": [],
                     "resume_lineage": [], "prior_wall_seconds": 0.0, "optimizer_reset_after_update1": False,
                     "initial_selection": initial_selection, "initial_test": initial_test,
                     "reference_summary_job_id": reference_summary["job_id"]}
        elif args.mode == "search":
            banks.restore(payload["master_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            optimizer_to_device(torch, optimizer, "cuda")
            restore_rng(torch, payload["rng_state"])
            print("FULL_SEARCH_CHECKPOINT_RESTORED " + json.dumps(
                {"phase": state["phase"], "updates": state["updates_completed"]}), flush=True)
        elif args.mode == "final" and final_resume_payload is not None:
            banks.restore(final_resume_payload["master_state"])
            restore_rng(torch, final_resume_payload["rng_state"])
            print("FINAL_EVALUATION_CHECKPOINT_RESTORED " + json.dumps(
                {"phase": state["phase"], "contexts": len(state["evaluation_rows"])}), flush=True)
        else:
            selected_epoch = binding["selected_epoch"]
            selected_dir = Path(freeze["selected_result_dir"]).resolve()
            selected_summary = json.loads((selected_dir / "summary.json").read_text(encoding="utf-8"))
            selected_protocol = json.loads((selected_dir / "protocol.json").read_text(encoding="utf-8"))
            if (selected_summary.get("status") != "search_complete"
                    or selected_summary.get("job_id") != freeze["selected_job_id"]
                    or selected_summary.get("run_id") != freeze["selected_run_id"]
                    or selected_summary.get("rho") != p["rho"]
                    or selected_summary.get("round") != p["round"]
                    or selected_summary.get("latent_weight") != float(freeze["latent_weight"])
                    or selected_summary.get("normalization_c") != float(freeze["normalization_c"])
                    or selected_protocol.get("candidate_run_id") != freeze["selected_run_id"]
                    or selected_protocol.get("rho") != p["rho"]
                    or numerical_contract(selected_protocol) != contract):
                raise RuntimeError("Optimal selection does not match the verified complete candidate summary")
            candidate = torch.load(selected_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
            if candidate.get("schema_version") != 2 or candidate.get("numerical_contract") != contract:
                raise RuntimeError("Selected candidate checkpoint is not schema-2 or has a different binding")
            epoch_states = candidate.get("epoch_states", {})
            key = str(selected_epoch)
            if key not in epoch_states:
                raise RuntimeError(f"Selected epoch {selected_epoch} is absent from the saved epoch_states")
            if len(selected_summary.get("selection_history", [])) != 3:
                raise RuntimeError("Selected search summary is missing one or more epoch selection points")
            banks.restore(epoch_states[key])
            state = {"mode": "final", "phase": "calibration", "selected_epoch": selected_epoch,
                     "candidate_run_id": freeze["selected_run_id"], "selected_run_id": freeze["selected_run_id"],
                     "selected_job_id": freeze["selected_job_id"],
                     "rho": p["rho"], "round": 2, "normalization_c": p["normalization_c"],
                     "latent_weight": float(freeze["latent_weight"]), "updates_completed": 0,
                     "epochs_completed": 2, "evaluation_rows": [], "forward_seconds": [],
                     "resume_lineage": [{"job_id": job_id, "selected_search_dir": str(selected_dir)}],
                     "prior_wall_seconds": 0.0, "selection_summary": next(
                         row for row in selected_summary["selection_history"] if row["epoch"] == selected_epoch),
                     "selected_training_history": selected_summary["training_history"],
                     "reference_summary_job_id": reference_summary["job_id"]}
            epoch_states = {key: epoch_states[key]}
            restore_rng(torch, candidate["rng_state"])
            print("GLOBAL_SELECTED_EPOCH_RESTORED " + json.dumps(
                {"run_id": freeze["selected_run_id"], "epoch": selected_epoch}), flush=True)

        state["resume_lineage"].append({"job_id": job_id,
                                        "resume_from": args.resume_dir.name if args.resume_dir else None})
        del payload, final_resume_payload
        gc.collect()
        data = {}

        def persist():
            snapshot = {"schema_version": 2, "numerical_contract": contract,
                        "master_state": banks.snapshot(),
                        "optimizer_state": to_cpu(optimizer.state_dict()) if optimizer else None,
                        "epoch_states": epoch_states, "rng_state": capture_rng(torch),
                        "run_state": {**state, "prior_wall_seconds": state["prior_wall_seconds"] + time.monotonic() - started}}
            atomic_torch_save(torch, args.out / "checkpoint.pt", snapshot)
            receipt = {"schema_version": 2, "job_id": job_id, "mode": args.mode,
                       "run_id": p["execution_run_id"], "candidate_run_id": p["candidate_run_id"],
                       "rho": p["rho"], "round": p["execution_round"],
                       "selected_round": p["round"] if args.mode == "final" else None,
                       "target_linears": len(banks.modules), "phase": state["phase"],
                       "epoch": state.get("epoch"), "next_train_index": state.get("next_train_index"),
                       "updates_completed": state["updates_completed"], "epochs_completed": state["epochs_completed"],
                       "evaluation_contexts": len(state["evaluation_rows"]),
                       "selected_epoch": state.get("selected_epoch"), "latent_weight": state["latent_weight"],
                       "normalization_c": state["normalization_c"], "checkpoint": str(args.out / "checkpoint.pt"),
                       "saved_utc_epoch": time.time()}
            save(args.out / "checkpoint_receipt.json", receipt)
            (args.out / "CHECKPOINT_READY").touch()
            if args.mode == "search":
                save(args.out / "selection_history.json", {"history": state["selection_history"]})
            print("CHECKPOINT_SAVED " + json.dumps(receipt), flush=True)

        def datum(case):
            if case not in data:
                row = manifest(case)
                observation, metadata = source.load_case(old_root if case < 1000 else source_root, row)
                data[case] = (runtime.datum(observation), metadata["description"], row)
            return data[case]

        def query(case, seed_index, checkpointed, return_future=False):
            value, description, row = datum(case)
            runtime.description, runtime.current_arm = description, "micro_latent_search"
            banks.seen.clear()
            output = sampler.differentiable_action(runtime, value, row["sampler_seeds"][seed_index],
                                                   checkpointed=checkpointed, return_future=return_future)
            if len(banks.seen) != p["target_linears"]:
                raise RuntimeError(f"Only {len(banks.seen)}/{p['target_linears']} target Linears were called")
            return output

        def pause_before_next_operation(training):
            timings = state["gradient_seconds"] if training else state["forward_seconds"]
            estimated = max(timings[-8:]) * 1.05 if timings else (360 if training else 100)
            if interrupted or time.monotonic() + estimated + p["save_reserve_seconds"] >= deadline:
                persist()
                save(args.out / "summary.json", {"status": "resumable", "protocol": p["protocol"],
                     "job_id": job_id, "mode": args.mode, "run_id": p["execution_run_id"],
                     "candidate_run_id": p["candidate_run_id"],
                     "execution_run_id": p["execution_run_id"],
                     "selected_run_id": state.get("candidate_run_id") if args.mode == "final" else None,
                     "selected_job_id": state.get("selected_job_id") if args.mode == "final" else None,
                     "selected_result_dir": freeze["selected_result_dir"] if args.mode == "final" else None,
                     "selected_epoch": state.get("selected_epoch") if args.mode == "final" else None,
                     "selected_round": selected_protocol["round"] if args.mode == "final" else None,
                     "rounds_complete": 2 if args.mode == "final" else None,
                     "selection_job_id": freeze["selection_job_id"] if args.mode == "final" else None,
                     "selection_source_jobs": freeze["selection_source_jobs"] if args.mode == "final" else None,
                     "rho": p["rho"], "round": p["execution_round"],
                     "normalization_c": p["normalization_c"],
                     "latent_weight": p["latent_weight"], "target_linears": 614,
                     "completed_epochs": state["epochs_completed"], "updates": state["updates_completed"],
                     "phase": state["phase"], "next_train_index": state.get("next_train_index"),
                     "evaluation_contexts": len(state["evaluation_rows"]),
                     "resume_lineage": state["resume_lineage"], "wall_seconds": time.monotonic() - started,
                     "estimated_next_operation_seconds": estimated,
                     "reason": "termination signal" if interrupted else "next operation reaches allocation end",
                     "checkpoint": str(args.out / "checkpoint.pt"), "complete": False})
                (args.out / "RESUME_READY").touch()
                print("MICRO_LATENT_RESUME_READY", flush=True)
                return True
            return False

        def evaluated(split, epoch=None):
            return [row for row in state["evaluation_rows"]
                    if row["split"] == split and (epoch is None or row.get("epoch") == epoch)]

        def evaluate_search_selection(epoch):
            rows = evaluated("selection", epoch)
            for case, seed_index in evaluation_pairs(p["selection_cases"], p["evaluation_seed_indices"], rows):
                if pause_before_next_operation(False):
                    return None
                progress(case_id=case, seed_index=seed_index, split="selection", epoch=epoch)
                tick = time.monotonic()
                action, counts, future = query(case, seed_index, False, return_future=True)
                action = action.detach().float().cpu()
                target = reference_cache["teacher"][_reference_key(case, seed_index)]
                action_summary = _metrics_for_action(torch, metrics, action, target["action"].float())
                future_summary = metrics.latent_metrics(torch, future, target["future"].to(device="cuda"))
                duration = time.monotonic() - tick
                state["forward_seconds"].append(duration)
                record = {"case_id": case, "seed_index": seed_index, "split": "selection", "epoch": epoch,
                          "arm": "micro_latent_search", **action_summary, **future_summary,
                          "forward_seconds": duration, "action": action.tolist(), **counts}
                rows.append(record)
                state["evaluation_rows"].append(record)
                del future
                persist()
                append(args.out / "actions.jsonl", record)
            future_rows = [{name: row[name] for name in (
                "case_id", "future_latent_rmse", "future_latent_nmse", "future_rmse", "future_nmse",
                "temporal_delta_global_rmse", "temporal_delta_global_nmse",
                "temporal_delta_rmse", "temporal_delta_nmse", "clamp_firstframe_max_abs")}
                for row in rows]
            return selection_entry(epoch, rows, future_rows)

        def evaluate_final_split(split, cases, seeds):
            rows = evaluated(split)
            for case, seed_index in evaluation_pairs(cases, seeds, rows):
                if pause_before_next_operation(False):
                    return None
                progress(case_id=case, seed_index=seed_index, split=split,
                         epoch=state["selected_epoch"])
                tick = time.monotonic()
                action, counts, future = query(case, seed_index, False, return_future=True)
                action = action.detach().float().cpu()
                target = reference_cache["teacher"][_reference_key(case, seed_index)]
                action_summary = _metrics_for_action(torch, metrics, action, target["action"].float())
                future_summary = metrics.latent_metrics(torch, future, target["future"].to(device="cuda"))
                duration = time.monotonic() - tick
                state["forward_seconds"].append(duration)
                record = {"case_id": case, "seed_index": seed_index, "split": split,
                          "epoch": state["selected_epoch"], "arm": "micro_latent_search_selected",
                          **action_summary, **future_summary, "forward_seconds": duration,
                          "action": action.tolist(), **counts}
                rows.append(record)
                state["evaluation_rows"].append(record)
                del future
                persist()
                append(args.out / "actions.jsonl", record)
            return aggregate_action(rows), aggregate_future(rows)

        persist()
        if args.mode == "search" and not state["parity_validated"]:
            stage = "cached_micro_initialization_parity"
            parity_cases = p["calibration_cases"][:2]
            completed = [row["case_id"] for row in state["parity_records"]]
            if completed != parity_cases[:len(completed)]:
                raise RuntimeError("Invalid micro initialization-parity checkpoint cursor")
            for case in parity_cases[len(completed):]:
                if pause_before_next_operation(False):
                    return
                tick = time.monotonic()
                action, counts, future = query(case, 0, False, return_future=True)
                state["forward_seconds"].append(time.monotonic() - tick)
                expected = reference_cache["parent"][_reference_key(case, 0)]
                action_parity = _parity(torch, action, expected["action"])
                future_parity = _parity(torch, future, expected["future"], future=True)
                record = {"case_id": case, "action": action_parity, "future": future_parity, **counts}
                record["passed"] = (action_parity["max_abs"] <= p["parity_max_abs_tolerance"]
                                     and action_parity["rmse"] <= p["parity_rmse_tolerance"]
                                     and future_parity["max_abs"] <= p["parity_max_abs_tolerance"]
                                     and future_parity["rmse"] <= p["parity_rmse_tolerance"])
                if not record["passed"]:
                    raise RuntimeError(f"Micro initial action/future parity failed: {record}")
                state["parity_records"].append(record)
                del action, future
                persist()
            state["parity_validated"] = True
            persist()
        if args.mode == "search":
            save(args.out / "parity.json", {"records": state["parity_records"],
                 "gradient_forward": state["gradient_parity"], "reused_on_resume": bool(args.resume_dir)})

        if args.mode == "search":
            while state["phase"] not in ("finalize", "complete"):
                stage = state["phase"]
                if stage == "train":
                    if pause_before_next_operation(True):
                        return
                    case = p["calibration_cases"][state["next_train_index"]]
                    progress(case_id=case, epoch=state["epoch"], update_in_epoch=state["next_train_index"])
                    optimizer.zero_grad(set_to_none=True)
                    tick = time.monotonic()
                    action, counts, future = query(case, 0, True, return_future=True)
                    if not future.requires_grad:
                        raise RuntimeError("Student future is disconnected from the full gradient graph")
                    if state["updates_completed"] == 0:
                        expected = reference_cache["parent"][_reference_key(case, 0)]
                        action_parity = _parity(torch, action, expected["action"])
                        future_parity = _parity(torch, future, expected["future"], future=True)
                        gradient_parity = {"case_id": case, "action": action_parity, "future": future_parity, **counts}
                        gradient_parity["passed"] = (
                            action_parity["max_abs"] <= p["parity_max_abs_tolerance"]
                            and action_parity["rmse"] <= p["parity_rmse_tolerance"]
                            and future_parity["max_abs"] <= p["parity_max_abs_tolerance"]
                            and future_parity["rmse"] <= p["parity_rmse_tolerance"])
                        save(args.out / "gradient_forward_parity.json", gradient_parity)
                        if not gradient_parity["passed"]:
                            raise RuntimeError("Checkpointed gradient forward differs from the micro parent capture")
                        state["gradient_parity"] = gradient_parity
                    teacher = reference_cache["teacher"][_reference_key(case, 0)]
                    teacher_action = teacher["action"].to(device="cuda", dtype=torch.float32)
                    teacher_future = teacher["future"].to(device="cuda")
                    action_mse = (action[:10, :6].float() - teacher_action[:10, :6]).square().mean()
                    future_nmse = latent_reference.future_nmse_loss(torch, future, teacher_future)
                    scaled_latent_loss = state["latent_weight"] * future_nmse
                    total_loss = action_mse + scaled_latent_loss
                    if (not bool(torch.isfinite(total_loss)) or not future_nmse.requires_grad
                            or not bool(torch.isfinite(action_mse)) or not bool(torch.isfinite(future_nmse))):
                        raise RuntimeError("Nonfinite or disconnected action+future loss")
                    values = {"action_mse": float(action_mse.detach()), "future_nmse": float(future_nmse.detach()),
                              "scaled_latent_loss": float(scaled_latent_loss.detach()),
                              "total_loss": float(total_loss.detach()), "latent_weight": state["latent_weight"]}
                    total_loss.backward()
                    report = banks.gradient_report()
                    if report["connected"] != 614 or report["finite"] != 614 or report["nonzero"] == 0:
                        save(args.out / "gradient_coverage_failure.json", report)
                        raise RuntimeError("Trainable micro bank gradient coverage is incomplete or nonfinite")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, p["gradient_norm_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    reset_now = latent_reference.reset_adam_after_update1(optimizer, state)
                    del action, future, teacher_action, teacher_future, action_mse, future_nmse, scaled_latent_loss, total_loss
                    banks.refresh()
                    torch.cuda.synchronize()
                    duration = time.monotonic() - tick
                    state["gradient_seconds"].append(duration)
                    record = {"update": state["updates_completed"] + 1, "case_id": case, "epoch": state["epoch"],
                              **values, "gradient_norm_before_clip": float(norm), "gradient_seconds": duration,
                              "optimizer_reset_after_update1": state["optimizer_reset_after_update1"],
                              "optimizer_reset_performed_this_update": reset_now, **counts, **report}
                    state["training_history"].append(record)
                    advance_training(state, p)
                    persist()
                    append(args.out / "training.jsonl", record)
                    if state["updates_completed"] == 1:
                        save(args.out / "first_backward.json", record)
                elif stage == "selection":
                    entry = evaluate_search_selection(state["epoch"])
                    if entry is None:
                        return
                    state["selection_history"].append(entry)
                    epoch_states[str(state["epoch"])] = banks.snapshot()
                    if state["epoch"] == p["maximum_epochs"]:
                        state["phase"] = "finalize"
                    else:
                        state["epoch"] += 1
                        state["next_train_index"] = 0
                        state["phase"] = "train"
                    persist()
                elif stage in ("finalize", "complete"):
                    break
                else:
                    raise RuntimeError(f"Invalid search resume phase: {stage}")
            if state["updates_completed"] != p["maximum_epochs"] * p["updates_per_epoch"]:
                raise RuntimeError("Search cannot complete before all 16 updates")
            if [row["epoch"] for row in state["selection_history"]] != [0, 1, 2]:
                raise RuntimeError("Search must save selection points for epochs 0, 1, and 2")
            if set(epoch_states) != {"0", "1", "2"}:
                raise RuntimeError("Search checkpoint is missing one of the three master states")
            if not state["optimizer_reset_after_update1"]:
                raise RuntimeError("Matched Adam reset after update one was not committed")
            state["phase"] = "complete"
            summary = {"status": "search_complete", "protocol": p["protocol"],
                       "job_id": job_id, "run_id": p["candidate_run_id"], "rho": p["rho"], "round": p["round"],
                       "latent_weight": p["latent_weight"], "normalization_c": p["normalization_c"],
                       "target_linears": 614, "completed_epochs": state["epochs_completed"],
                       "updates": state["updates_completed"], "selection_history": state["selection_history"],
                       "training_history": state["training_history"], "teacher_forward_count": 0,
                       "test_forward_count": 0, "final_test_context_count": 0,
                       "selection_context_count": sum(row["split"] == "selection" and row["epoch"] in (1, 2)
                                                       for row in state["evaluation_rows"]),
                       "initial_selection_context_count": 8, "selection_cache_parity": True,
                       "parity_records": state["parity_records"], "gradient_forward_parity": state["gradient_parity"],
                       "gradient_seconds": state["gradient_seconds"], "forward_seconds": state["forward_seconds"],
                       "wall_seconds": time.monotonic() - started,
                       "cumulative_wall_seconds": state["prior_wall_seconds"] + time.monotonic() - started,
                       "resume_lineage": state["resume_lineage"],
                       "optimizer_schedule": "Adam reset once immediately after committed update1; retained thereafter",
                       "optimizer_reset_after_update1": state["optimizer_reset_after_update1"],
                       "reference_summary_job_id": reference_summary["job_id"],
                       "reference_model_forward_count": reference_summary["model_forward_count"],
                       "bank_root": p["bank_root"], "bank_full_job": p["bank_full_job"],
                       "episodes": 0, "native_low_bit_kernel": False}
            save(args.out / "summary.json", summary)
            persist()
            (args.out / "SEARCH_COMPLETE").touch()
            save(args.out / "SEARCH_COMPLETE.json", {"job_id": job_id, "run_id": p["candidate_run_id"],
                 "rho": p["rho"], "round": p["round"], "selection_epochs": [0, 1, 2],
                 "test_forward_count": 0})
            print("MICRO_LATENT_SEARCH_COMPLETE " + json.dumps(summary), flush=True)
            return

        if state["phase"] == "calibration":
            stage = "final_calibration"
            result = evaluate_final_split("calibration", p["calibration_cases"], [0])
            if result is None:
                return
            state["calibration_summary"], state["calibration_future_summary"] = result
            state["phase"] = "test"
            persist()
        if state["phase"] == "test":
            stage = "final_test"
            result = evaluate_final_split("test", p["test_cases"], p["evaluation_seed_indices"])
            if result is None:
                return
            state["test_summary"], state["test_future_summary"] = result
            state["phase"] = "complete"
            persist()
        if state["phase"] != "complete":
            raise RuntimeError(f"Invalid final evaluation phase: {state['phase']}")
        if (state["calibration_summary"]["case_count"] != 8
                or state["calibration_summary"]["context_count"] != 8
                or state["test_summary"]["case_count"] != 22
                or state["test_summary"]["context_count"] != 44):
            raise RuntimeError("Final evaluation must cover the complete frozen calibration8/test44 cohorts")
        initial_calibration_action = aggregate_action([
            {"case_id": case, "seed_index": 0, "split": "calibration", "arm": p["initial_arm"],
             **_metrics_for_action(torch, metrics,
                 reference_cache["parent"][_reference_key(case, 0)]["action"],
                 reference_cache["teacher"][_reference_key(case, 0)]["action"])}
            for case in p["calibration_cases"]])
        initial_calibration_future = aggregate_future([
            {"case_id": case, **metrics.latent_metrics(
                torch, reference_cache["parent"][_reference_key(case, 0)]["future"],
                reference_cache["teacher"][_reference_key(case, 0)]["future"])}
            for case in p["calibration_cases"]])
        summary = {"status": "complete", "mode": "final", "protocol": p["protocol"],
                   "job_id": job_id, "run_id": "final", "selected_run_id": freeze["selected_run_id"],
                   "selected_job_id": freeze["selected_job_id"], "selected_epoch": binding["selected_epoch"],
                   "selected_result_dir": freeze["selected_result_dir"],
                   "selection_job_id": freeze["selection_job_id"],
                   "selection_source_jobs": freeze["selection_source_jobs"], "rounds_complete": 2,
                   "round": 2, "selected_round": selected_protocol["round"],
                   "rho": p["rho"], "latent_weight": p["latent_weight"],
                   "normalization_c": p["normalization_c"], "target_linears": 614,
                   "completed_epochs": 2, "updates": 0, "selection": state["selection_summary"],
                   "initial_selection": initial_selection,
                   "initial_selection_future_metrics": selection0["future_metrics"],
                   "initial_calibration": {"action": initial_calibration_action,
                                           "future_metrics": initial_calibration_future},
                   "calibration": state["calibration_summary"],
                   "calibration_future_metrics": state["calibration_future_summary"],
                   "initial_test": initial_test, "test": state["test_summary"],
                   "test_future_metrics": state["test_future_summary"],
                   "calibration_context_count": state["calibration_summary"]["context_count"],
                   "test_context_count": state["test_summary"]["context_count"],
                   "test_forward_count": state["test_summary"]["context_count"],
                   "final_test_context_count": state["test_summary"]["context_count"],
                   "teacher_forward_count": 0, "parent_test_future_forward_count": 0,
                   "selection_context_count": 0, "training_history": state["selected_training_history"],
                   "training_performed": False, "reference_summary_job_id": reference_summary["job_id"],
                   "reference_model_forward_count": reference_summary["model_forward_count"],
                   "wall_seconds": time.monotonic() - started,
                   "cumulative_wall_seconds": state["prior_wall_seconds"] + time.monotonic() - started,
                   "resume_lineage": state["resume_lineage"], "episodes": 0, "native_low_bit_kernel": False,
                   "test_independence_boundary": p["reuse_boundary"],
                   "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())}
        save(args.out / "summary.json", summary)
        persist()
        print("MICRO_LATENT_FINAL_COMPLETE " + json.dumps(summary), flush=True)
    except Exception as exc:
        save(args.out / "failure.json", {"status": "failed", "stage": stage, "message": str(exc),
             "updates": state["updates_completed"] if state else 0,
             "epochs": state["epochs_completed"] if state else 0,
             "checkpoint_available": (args.out / "checkpoint.pt").is_file(), "traceback": traceback.format_exc()})
        raise
    finally:
        if runtime and runtime.env:
            runtime.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--mode", choices=("search", "final"), required=True)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--freeze-file", type=Path)
    run(parser.parse_args())

