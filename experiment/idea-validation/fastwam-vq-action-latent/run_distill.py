"""Resumable action+future latent VQ validation on the frozen Fast-WAM cohort."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import tempfile
import time
import traceback


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    metrics = ("motor_rmse_first10_vs_bf16", "gripper_rmse_first10_vs_bf16", "motor_rmse_32_vs_bf16")
    values = {str(case): {metric: statistics.mean(r[metric] for r in records) for metric in metrics}
              for case, records in groups.items()}
    return {"case_count": len(groups), "context_count": len(rows),
            "mean_motor_rmse_first10": statistics.mean(v[metrics[0]] for v in values.values()),
            "mean_gripper_rmse_first10": statistics.mean(v[metrics[1]] for v in values.values()),
            "mean_motor_rmse_32": statistics.mean(v[metrics[2]] for v in values.values()), "per_case": values}


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


def numerical_contract(protocol):
    keys = ("source_root", "source_full_job", "roots", "initial_arm", "target_linears",
            "calibration_cases", "calibration_seed_indices", "selection_cases", "test_cases",
            "evaluation_seed_indices", "teacher", "loss", "latent_loss", "trainables", "fixed",
            "optimizer", "learning_rate", "gradient_norm_clip", "maximum_epochs", "updates_per_epoch",
            "checkpoint_selection", "training_gradients", "checkpointing",
            "latent_diagnostic_root", "latent_diagnostic_jobs", "reference_cache", "reference_summary",
            "latent_loss_fraction", "latent_loss_weight_policy", "baseline_optimizer_reset_after_update1",
            "baseline_action_distill_job", "optimizer_schedule")
    return {key: protocol[key] for key in keys}


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


def _reference_key(case, seed):
    return f"{case}:{seed}"


def _parity(torch, actual, expected, future=False):
    left = actual.detach().float().cpu()
    right = expected.detach().float().cpu()
    if future:
        left, right = left[:, :, 1:], right[:, :, 1:]
    delta = left - right
    return {"max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt())}


def run(args):
    import sampler
    sampler._require_compute_allocation()
    import torch
    import runner as pilot
    import run_diagnostic as source
    import metrics
    import latent_reference
    from train_banks import TrainableBanks, self_check
    from checkpointing import atomic_torch_save, capture_rng, optimizer_to_device, restore_rng, to_cpu

    p = json.loads(args.protocol.read_text(encoding="utf-8"))
    if p["protocol"] != "fastwam-vq-action-latent-v1":
        raise ValueError("Unexpected action-latent protocol")
    if not p["baseline_optimizer_reset_after_update1"]:
        raise RuntimeError("The matched first-update Adam reset must remain enabled")
    if args.out.exists():
        raise FileExistsError(f"Existing result directory: {args.out}")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    allocation_started = float(os.environ.get("FW_JOB_START_EPOCH", time.time()))
    deadline = started + p["walltime_seconds"] - (time.time() - allocation_started)
    job_id = os.environ["PBS_JOBID"]
    interrupted = []
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupted.append(signum))
    source_root, old_root = Path(p["source_root"]), Path(p["roots"]["old_inputs"])
    source_dir = source_root / "results" / "full"
    stage, state, banks, optimizer, runtime = "reference_loading", None, None, None, None
    torch.backends.cuda.matmul.allow_tf32 = False

    def progress(**extra):
        record = {"status": "running", "stage": stage, "job_id": job_id,
                  "updates_completed": state["updates_completed"] if state else 0,
                  "epochs_completed": state["epochs_completed"] if state else 0,
                  "elapsed_seconds": time.monotonic() - started, **extra}
        save(args.out / "progress.json", record)
        print("ACTION_LATENT_PROGRESS " + json.dumps(record), flush=True)

    try:
        reference_summary = json.loads(Path(p["reference_summary"]).read_text(encoding="utf-8"))
        reference_contract = latent_reference.resume_reference_contract(reference_summary)
        if (reference_summary.get("protocol") != p["protocol"]
                or reference_summary.get("latent_diagnostic_jobs") != p["latent_diagnostic_jobs"]
                or reference_contract["fraction"] != float(p["latent_loss_fraction"])
                or reference_contract["calibration_ids"] != p["calibration_cases"]):
            raise RuntimeError("Reference summary differs from the declared frozen protocol")
        p["latent_loss"] = reference_contract
        reference_cache = torch.load(Path(p["reference_cache"]), map_location="cpu", weights_only=True, mmap=True)
        if (reference_cache.get("schema_version") != 1 or reference_cache.get("protocol") != p["protocol"]
                or reference_cache.get("source_root") != p["source_root"]
                or reference_cache.get("source_full_job") != p["source_full_job"]
                or reference_cache.get("initial_arm") != p["initial_arm"]
                or reference_cache.get("calibration_ids") != p["calibration_cases"]
                or reference_cache.get("latent_loss") != {
                    **reference_contract,
                    "mean_cal_initial_action_mse": reference_summary["mean_cal_initial_action_mse"],
                    "mean_cal_initial_future_nmse": reference_summary["mean_cal_initial_future_nmse"],
                }):
            raise RuntimeError("Reference cache does not match its preflight summary")
        expected_keys = {_reference_key(row["case_id"], row["seed_index"])
                         for row in metrics.frozen_contexts(p)}
        if set(reference_cache["teacher"]) != expected_keys or set(reference_cache["parent"]) != expected_keys:
            raise RuntimeError("Reference cache does not cover the complete frozen 60-context cohort")
        for key, ref in reference_cache["teacher"].items():
            if tuple(ref["action"].shape) != (32, 7) or tuple(ref["future"].shape) != (1, 48, 3, 14, 28):
                raise RuntimeError(f"Invalid teacher reference shape at {key}")
            if not bool(torch.isfinite(ref["action"]).all()) or not bool(torch.isfinite(ref["future"]).all()):
                raise RuntimeError(f"Nonfinite teacher reference at {key}")
        for key, ref in reference_cache["parent"].items():
            if tuple(ref["action"].shape) != (32, 7) or tuple(ref["future"].shape) != (1, 48, 3, 14, 28):
                raise RuntimeError(f"Invalid parent reference shape at {key}")
            if not bool(torch.isfinite(ref["action"]).all()) or not bool(torch.isfinite(ref["future"]).all()):
                raise RuntimeError(f"Nonfinite parent reference at {key}")
        if set(reference_cache["initial_selection_future_metrics"]) != {
                _reference_key(case, seed) for case in p["selection_cases"] for seed in p["evaluation_seed_indices"]}:
            raise RuntimeError("Initial selection future metrics are not complete")
        contract = numerical_contract(p)
        save(args.out / "protocol.json", p)

        stage = "source_validation"
        progress()
        full_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
        if full_summary["status"] != "complete" or full_summary["pbs_jobid"] != p["source_full_job"]:
            raise RuntimeError("Source VQ run is not the declared completed run")
        old_plan = json.loads((old_root / "plan.json").read_text(encoding="utf-8"))
        fresh = {row["case_id"]: row for row in json.loads((source_root / "plan.json").read_text(encoding="utf-8"))["inputs"]}

        stage = "checkpoint_self_checks"
        progress()
        self_check()
        from test_checkpointing import self_check as checkpoint_test
        from test_driver import self_check as driver_test
        from test_latent_loss import self_check as latent_test
        with tempfile.TemporaryDirectory(prefix="checkpoint-test-", dir=args.out) as directory:
            checkpoint_test(torch, directory)
        driver_test(p)
        latent_test(torch, p)
        print("CHECKPOINT_CURSOR_AND_LATENT_SELF_CHECKS_PASSED", flush=True)

        payload = None
        if args.resume_dir:
            if args.resume_dir.parent.resolve() != (Path(p["remote_root"]) / "results").resolve():
                raise RuntimeError("Resume directory is outside this campaign")
            previous_protocol = json.loads((args.resume_dir / "protocol.json").read_text(encoding="utf-8"))
            if numerical_contract(previous_protocol) != contract:
                raise RuntimeError("Resume changes the frozen numerical contract or calibration-derived weight")
            if not (args.resume_dir / "checkpoint.pt").is_file():
                raise RuntimeError("Schema-2 resume requires a complete checkpoint; legacy recovery is disabled")
            payload = torch.load(args.resume_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
            if payload["schema_version"] != 2 or payload["numerical_contract"] != contract:
                raise RuntimeError("Unsupported or mismatched resume checkpoint")
            state = payload["run_state"]
            if state["phase"] == "complete":
                raise RuntimeError("A completed run must not be resumed")
            if (state["latent_weight"] != reference_contract["latent_weight"]
                    or state["latent_loss_policy"] != reference_contract["policy"]
                    or state["latent_loss_fraction"] != reference_contract["fraction"]
                    or state["calibration_ids"] != reference_contract["calibration_ids"]
                    or state["optimizer_reset_after_update1"] != (state["updates_completed"] >= 1)):
                raise RuntimeError("Checkpoint loss or matched-optimizer schedule differs from reference")

        stage = "model_loading"
        progress()
        runtime = pilot.Runtime(args.out)
        runtime.arm("bf16")
        stage = "loading_existing_vq_banks"
        progress()
        banks = TrainableBanks(runtime, source_dir, p["initial_arm"])
        parameters = banks.parameters()
        optimizer = torch.optim.Adam(parameters, lr=p["learning_rate"])
        initial = banks.snapshot()

        initial_action_rows, initial_future_rows = [], []
        for case in p["selection_cases"]:
            for seed in p["evaluation_seed_indices"]:
                key = _reference_key(case, seed)
                target, start = reference_cache["teacher"][key], reference_cache["parent"][key]
                initial_action_rows.append({
                    "case_id": case, "seed_index": seed,
                    **source.action_metrics(torch, start["action"], target["action"]),
                })
                initial_future_rows.append({"case_id": case, "seed_index": seed,
                                            **reference_cache["initial_selection_future_metrics"][key]})
        selection0 = aggregate(initial_action_rows)
        selection0_future = aggregate_future(initial_future_rows)
        initial_test_rows = []
        initial_test_future_rows = []
        for case in p["test_cases"]:
            for seed in p["evaluation_seed_indices"]:
                key = _reference_key(case, seed)
                target, start = reference_cache["teacher"][key], reference_cache["parent"][key]
                initial_test_rows.append({"case_id": case, "seed_index": seed,
                                          **source.action_metrics(torch, start["action"], target["action"])})
                initial_test_future_rows.append({"case_id": case, "seed_index": seed,
                                                 **metrics.latent_metrics(torch, start["future"], target["future"])})
        initial_test = aggregate(initial_test_rows)
        initial_test_future = aggregate_future(initial_test_future_rows)

        if state is None:
            state = {"phase": "train", "epoch": 1, "next_train_index": 0,
                     "updates_completed": 0, "epochs_completed": 0,
                     "parity_records": [], "parity_validated": False, "gradient_parity": None,
                     "training_history": [], "evaluation_rows": [], "gradient_seconds": [], "forward_seconds": [],
                     "selection_history": [{"epoch": 0, "source": "cached parent actions and diagnostic futures",
                                            **selection0, "future_metrics": selection0_future}],
                     "selected_epoch": 0, "best_score": selection0["mean_motor_rmse_first10"],
                     "frozen": False, "resume_lineage": [], "prior_wall_seconds": 0.0,
                     "latent_weight": reference_contract["latent_weight"],
                     "latent_loss_policy": reference_contract["policy"],
                     "latent_loss_fraction": reference_contract["fraction"],
                     "calibration_ids": reference_contract["calibration_ids"],
                     "reference_summary_job_id": reference_summary["job_id"],
                     "optimizer_reset_after_update1": False}
        best, last_trained = initial, None
        if payload:
            banks.restore(payload["master_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            optimizer_to_device(torch, optimizer, "cuda")
            best, last_trained = payload["best_state"], payload["last_trained_state"]
            restore_rng(torch, payload["rng_state"])
            print("FULL_CHECKPOINT_RESTORED " + json.dumps(
                {"phase": state["phase"], "updates": state["updates_completed"]}), flush=True)
        state["resume_lineage"].append({"job_id": job_id, "resume_from": args.resume_dir.name if args.resume_dir else None})
        del payload, initial
        gc.collect()
        data = {}

        def persist():
            snapshot = {"schema_version": 2, "numerical_contract": contract,
                        "master_state": banks.snapshot(), "optimizer_state": to_cpu(optimizer.state_dict()),
                        "best_state": best, "last_trained_state": last_trained, "rng_state": capture_rng(torch),
                        "run_state": {**state, "prior_wall_seconds": state["prior_wall_seconds"] + time.monotonic() - started}}
            atomic_torch_save(torch, args.out / "checkpoint.pt", snapshot)
            receipt = {"schema_version": 2, "job_id": job_id, "target_linears": len(banks.modules),
                       "phase": state["phase"], "epoch": state["epoch"], "next_train_index": state["next_train_index"],
                       "updates_completed": state["updates_completed"], "epochs_completed": state["epochs_completed"],
                       "evaluation_contexts": len(state["evaluation_rows"]), "selected_epoch": state["selected_epoch"],
                       "latent_weight": state["latent_weight"], "calibration_ids": state["calibration_ids"],
                       "optimizer_reset_after_update1": state["optimizer_reset_after_update1"],
                       "checkpoint": str(args.out / "checkpoint.pt"), "saved_utc_epoch": time.time()}
            save(args.out / "checkpoint_receipt.json", receipt)
            (args.out / "CHECKPOINT_READY").touch()
            save(args.out / "selection_history.json", {"history": state["selection_history"]})
            print("CHECKPOINT_SAVED " + json.dumps(receipt), flush=True)

        def datum(case):
            if case not in data:
                row = source.validate_plan(old_plan, case) if case < 1000 else fresh[case]
                observation, metadata = source.load_case(old_root if case < 1000 else source_root, row)
                data[case] = (runtime.datum(observation), metadata["description"], row)
            return data[case]

        def query(case, seed_index, checkpointed, return_future=False):
            value, description, row = datum(case)
            runtime.description, runtime.current_arm = description, "action_latent_distill"
            banks.seen.clear()
            output = sampler.differentiable_action(
                runtime, value, row["sampler_seeds"][seed_index], checkpointed=checkpointed,
                return_future=return_future)
            if len(banks.seen) != p["target_linears"]:
                raise RuntimeError(f"Only {len(banks.seen)}/614 targets called")
            return output

        def pause_before_next_operation(training):
            timings = state["gradient_seconds"] if training else state["forward_seconds"]
            estimated = max(timings[-8:]) * 1.05 if timings else (360 if training else 100)
            if interrupted or time.monotonic() + estimated + p["save_reserve_seconds"] >= deadline:
                persist()
                save(args.out / "summary.json", {"status": "resumable", "protocol": p["protocol"], "job_id": job_id,
                     "target_linears": 614, "completed_epochs": state["epochs_completed"], "updates": state["updates_completed"],
                     "phase": state["phase"], "next_train_index": state["next_train_index"],
                     "evaluation_contexts": len(state["evaluation_rows"]), "resume_lineage": state["resume_lineage"],
                     "latent_weight": state["latent_weight"], "calibration_ids": state["calibration_ids"],
                     "optimizer_reset_after_update1": state["optimizer_reset_after_update1"],
                     "wall_seconds": time.monotonic() - started, "estimated_next_operation_seconds": estimated,
                     "reason": "termination signal" if interrupted else "next operation reaches allocation end",
                     "checkpoint": str(args.out / "checkpoint.pt"), "complete": False})
                print("ACTION_LATENT_RESUME_READY", flush=True)
                return True
            return False

        def evaluated(split, epoch):
            return [row for row in state["evaluation_rows"] if row["split"] == split and row["epoch"] == epoch]

        def evaluate(cases, seeds, split, epoch):
            if split == "test" and not state["frozen"]:
                raise RuntimeError("Test evaluation before freeze")
            rows = evaluated(split, epoch)
            for case, seed_index in evaluation_pairs(cases, seeds, rows):
                if pause_before_next_operation(False):
                    return None
                progress(case_id=case, seed_index=seed_index, split=split, epoch=epoch)
                tick = time.monotonic()
                action, counts, future = query(case, seed_index, False, return_future=True)
                action = action.detach().float().cpu()
                target = reference_cache["teacher"][_reference_key(case, seed_index)]
                action_target = target["action"].to(device="cuda", dtype=torch.float32)
                future_target = target["future"].to(device="cuda")
                action_summary = source.action_metrics(torch, action, action_target.cpu())
                action_summary.update(metrics.action_metrics(torch, action, action_target.cpu()))
                future_summary = metrics.latent_metrics(torch, future, future_target)
                duration = time.monotonic() - tick
                state["forward_seconds"].append(duration)
                record = {"case_id": case, "seed_index": seed_index, "split": split, "epoch": epoch,
                          "arm": "action_latent_distill", **action_summary, **future_summary,
                          "forward_seconds": duration, "action": action.tolist(), **counts}
                rows.append(record)
                state["evaluation_rows"].append(record)
                del future, future_target, action_target
                persist()
                append(args.out / "actions.jsonl", record)
            return aggregate(rows), aggregate_future(rows)

        persist()
        if not state["parity_validated"]:
            stage = "cached_initialization_action_future_parity"
            parity_cases = p["calibration_cases"][:2]
            completed = [row["case_id"] for row in state["parity_records"]]
            if completed != parity_cases[:len(completed)]:
                raise RuntimeError("Invalid initialization-parity checkpoint cursor")
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
                    raise RuntimeError(f"Cached initialization action/future parity failed: {record}")
                state["parity_records"].append(record)
                del action, future
                persist()
            state["parity_validated"] = True
            persist()
        save(args.out / "parity.json", {"records": state["parity_records"],
             "gradient_forward": state["gradient_parity"], "reused_on_resume": bool(args.resume_dir)})

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
                    raise RuntimeError("Student future latent is disconnected from the full gradient graph")
                if state["updates_completed"] == 0:
                    expected = reference_cache["parent"][_reference_key(case, 0)]
                    action_parity = _parity(torch, action, expected["action"])
                    future_parity = _parity(torch, future, expected["future"], future=True)
                    grad_parity = {"case_id": case, "action": action_parity, "future": future_parity, **counts}
                    grad_parity["passed"] = (action_parity["max_abs"] <= p["parity_max_abs_tolerance"]
                                              and action_parity["rmse"] <= p["parity_rmse_tolerance"]
                                              and future_parity["max_abs"] <= p["parity_max_abs_tolerance"]
                                              and future_parity["rmse"] <= p["parity_rmse_tolerance"])
                    save(args.out / "gradient_forward_parity.json", grad_parity)
                    if not grad_parity["passed"]:
                        raise RuntimeError("Checkpointed gradient action/future differs from parent capture")
                    state["gradient_parity"] = grad_parity

                ref = reference_cache["teacher"][_reference_key(case, 0)]
                teacher_action = ref["action"].to(device="cuda", dtype=torch.float32)
                teacher_future = ref["future"].to(device="cuda")
                action_mse = (action[:10, :6].float() - teacher_action[:10, :6]).square().mean()
                future_nmse = latent_reference.future_nmse_loss(torch, future, teacher_future)
                scaled_latent_loss = state["latent_weight"] * future_nmse
                total_loss = action_mse + scaled_latent_loss
                if not bool(torch.isfinite(total_loss)) or not future_nmse.requires_grad:
                    raise RuntimeError("Nonfinite or disconnected action+future training loss")
                values = {"action_mse": float(action_mse.detach()), "future_nmse": float(future_nmse.detach()),
                          "scaled_latent_loss": float(scaled_latent_loss.detach()),
                          "total_loss": float(total_loss.detach()), "latent_weight": state["latent_weight"]}
                total_loss.backward()
                report = banks.gradient_report()
                if report["connected"] != 614 or report["finite"] != 614 or report["nonzero"] == 0:
                    save(args.out / "gradient_coverage_failure.json", report)
                    raise RuntimeError("Trainable codebook gradient coverage is incomplete or nonfinite")
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
                result = evaluate(p["selection_cases"], p["evaluation_seed_indices"], "selection", state["epoch"])
                if result is None:
                    return
                score, future_score = result
                state["selection_history"].append({"epoch": state["epoch"], **score, "future_metrics": future_score})
                if score["mean_motor_rmse_first10"] < state["best_score"]:
                    state["best_score"], state["selected_epoch"], best = score["mean_motor_rmse_first10"], state["epoch"], banks.snapshot()
                if state["epoch"] == p["maximum_epochs"]:
                    state["phase"] = "freeze"
                else:
                    state["epoch"] += 1
                    state["next_train_index"] = 0
                    state["phase"] = "train"
                persist()
            elif stage == "freeze":
                if state["updates_completed"] != p["maximum_epochs"] * p["updates_per_epoch"]:
                    raise RuntimeError("Freeze before full declared training completion")
                last_trained = banks.snapshot()
                banks.restore(best)
                state["frozen"], state["phase"] = True, "final_calibration"
                persist()
            elif stage in ("final_calibration", "test"):
                split = "calibration" if stage == "final_calibration" else "test"
                cases = p["calibration_cases"] if split == "calibration" else p["test_cases"]
                seeds = [0] if split == "calibration" else p["evaluation_seed_indices"]
                result = evaluate(cases, seeds, split, state["selected_epoch"])
                if result is None:
                    return
                state[split + "_summary"], state[split + "_future_summary"] = result
                state["phase"] = "test" if split == "calibration" else "finalize"
                persist()
            else:
                raise RuntimeError(f"Invalid resume phase: {stage}")

        atomic_torch_save(torch, args.out / "selected_codebook_overlay.pt",
                         {"source_root": str(source_root), "source_arm": p["initial_arm"],
                          "module_names": list(banks.modules), "master_state": best,
                          "fixed_indices_and_transforms": "reuse corresponding source bank files"})
        atomic_torch_save(torch, args.out / "last_trained_master.pt", last_trained)
        save(args.out / "frozen_checkpoint.json", {"selected_epoch": state["selected_epoch"],
             "completed_epochs": state["epochs_completed"], "updates": state["updates_completed"],
             "selection_score": state["best_score"], "test_queries_before_freeze": 0})
        changed_books, changed_scales = 0, 0
        for name in banks.modules:
            encoding = banks.stored_encoding(name)
            changed_books += int(torch.count_nonzero(encoding["books"] != banks.encodings[name]["books"]))
            changed_scales += int(torch.count_nonzero(encoding["row_scales"] != banks.encodings[name]["row_scales"]))
        if state["updates_completed"] != 16 or not state["optimizer_reset_after_update1"]:
            raise RuntimeError("Final checkpoint does not meet the frozen update/reset schedule")
        test = state["test_summary"]
        summary = {"status": "complete", "protocol": p["protocol"], "job_id": job_id,
                   "target_linears": 614, "completed_epochs": state["epochs_completed"], "updates": state["updates_completed"],
                   "selected_epoch": state["selected_epoch"], "selection_history": state["selection_history"],
                   "calibration": state["calibration_summary"], "calibration_future_metrics": state["calibration_future_summary"],
                   "test": test, "test_future_metrics": state["test_future_summary"],
                   "cached_initial_test": initial_test, "cached_initial_test_future_metrics": initial_test_future,
                   "relative_test_motor_rmse_change": test["mean_motor_rmse_first10"] / initial_test["mean_motor_rmse_first10"] - 1,
                   "changed_stored_book_scalars": changed_books, "changed_stored_row_scales": changed_scales,
                   "evaluation_contexts": len(state["evaluation_rows"]),
                   "selection_contexts": sum(row["split"] == "selection" for row in state["evaluation_rows"]),
                   "calibration_context_count": sum(row["split"] == "calibration" for row in state["evaluation_rows"]),
                   "test_context_count": sum(row["split"] == "test" for row in state["evaluation_rows"]),
                   "test_queries_before_freeze": 0, "teacher_inference_queries": 0,
                   "teacher_source": p["teacher"], "reference_cache": p["reference_cache"],
                   "latent_diagnostic_root": p["latent_diagnostic_root"],
                   "latent_diagnostic_jobs": p["latent_diagnostic_jobs"],
                   "reference_preflight_job_id": reference_summary["job_id"],
                   "reference_model_forward_count": reference_summary["model_forward_count"],
                   "latent_weight_policy": reference_contract["policy"],
                   "latent_loss_fraction": reference_contract["fraction"],
                   "latent_weight": state["latent_weight"], "latent_weight_calibration_ids": state["calibration_ids"],
                   "training_history": state["training_history"],
                   "training_seconds_per_update": state["gradient_seconds"], "forward_seconds": state["forward_seconds"],
                   "wall_seconds": time.monotonic() - started,
                   "cumulative_wall_seconds": state["prior_wall_seconds"] + time.monotonic() - started,
                   "resume_lineage": state["resume_lineage"],
                   "optimizer_schedule": "matched_baseline_optimizer_reset_after_update1",
                   "optimizer_reset_after_update1": state["optimizer_reset_after_update1"],
                   "episodes": 0, "native_low_bit_kernel": False,
                   "test_independence_boundary": p["reuse_boundary"],
                   "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())}
        save(args.out / "summary.json", summary)
        state["phase"] = "complete"
        persist()
        print("ACTION_LATENT_DISTILL_COMPLETE " + json.dumps(summary), flush=True)
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
    parser.add_argument("--resume-dir", type=Path)
    run(parser.parse_args())
