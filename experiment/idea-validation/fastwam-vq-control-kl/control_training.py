"""Allocation-guarded paired C80 D/r training with update-boundary resume."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import time

from control_common import Deadline, guard, read, save


UNIT_PREFIX = "train_"
TRAINING_PROTOCOL = "control-kl-training-v1"
CASE_IDS = list(range(2000, 2080))
VARIANTS = ("clean", "noisy")
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)
NOISE_SHAPE = (1, 32, 7)
COUNTS = {"video_steps": 10, "action_steps": 10}
TRAINING_KEYS = {
    "protocol", "parent_protocol", "data_config", "checkpoint_arm",
    "update_order", "source_jobs", "arms", "optimizer",
}
SOURCE_KEYS = {"training_teacher", "training_loss_selfcheck",
               "aggregate_c80_parameter_gate"}
OPTIMIZER_KEYS = {"betas", "eps", "gradient_norm_clip",
                  "reset_after_update1", "gate_fraction"}


def _finite_number(value, label, *, positive=False, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return result


def _same_tensor_mapping(torch, left, right):
    return (set(left) == set(right)
            and all(torch.equal(torch.as_tensor(left[key]).detach().cpu(),
                                torch.as_tensor(right[key]).detach().cpu())
                    for key in left))


def _validate_tensor_mapping(torch, saved, current, label):
    if set(saved) != set(current):
        raise ValueError(f"{label} checkpoint tensor names differ")
    for name, value in saved.items():
        reference = current[name]
        if (not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(reference.shape)
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{label} checkpoint tensor is malformed: {name}")


def validate_training_protocol(parent, data_config, config, arm, gate_config):
    """Reject unfrozen/implicit experiment settings before model construction."""
    if not isinstance(config, dict) or set(config) != TRAINING_KEYS:
        raise ValueError("training_protocol.json must contain exactly the frozen training fields")
    if (config.get("protocol") != TRAINING_PROTOCOL
            or config.get("parent_protocol") != parent.get("protocol")
            or config.get("data_config") != data_config
            or config.get("checkpoint_arm") != "per_channel_clip"):
        raise ValueError("Training protocol, parent, data config, or checkpoint arm differs")
    order = parent.get("source_closed_protocol", {}).get("calibration", {}).get("update_order")
    if (not isinstance(order, list) or order != config.get("update_order")
            or len(order) != 80 or len(set(order)) != 80
            or set(order) != set(CASE_IDS)):
        raise ValueError("Training update order must be the original C80 order, once each")
    sources = config.get("source_jobs")
    if not isinstance(sources, dict) or set(sources) != SOURCE_KEYS or any(
            not isinstance(value, str) or not value for value in sources.values()):
        raise ValueError("source_jobs must bind the three completed training prerequisites")
    arms = config.get("arms")
    if not isinstance(arms, dict) or not arms or arm not in arms:
        raise ValueError(f"No explicitly configured training arm: {arm}")
    from training_objective import validate_config as validate_loss_config
    for name, value in arms.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,25}", name):
            raise ValueError("Training arm names must be simple lowercase identifiers")
        validate_loss_config(value)

    optimizer = config.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != OPTIMIZER_KEYS:
        raise ValueError("optimizer must explicitly contain betas, eps, clipping, reset, and gate fraction")
    betas = optimizer.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.betas must contain two explicit values")
    parsed_betas = [_finite_number(value, "optimizer beta", maximum=1.0) for value in betas]
    if any(value >= 1.0 for value in parsed_betas):
        raise ValueError("optimizer betas must be less than one")
    _finite_number(optimizer["eps"], "optimizer.eps", positive=True)
    _finite_number(optimizer["gradient_norm_clip"], "optimizer.gradient_norm_clip", positive=True)
    if not isinstance(optimizer["reset_after_update1"], bool):
        raise ValueError("optimizer.reset_after_update1 must be explicit boolean")
    fraction = _finite_number(optimizer["gate_fraction"], "optimizer.gate_fraction",
                              positive=True, maximum=1.0)
    fractions = gate_config.get("fractions")
    if not isinstance(fractions, list) or fraction not in fractions:
        raise ValueError("optimizer.gate_fraction was not evaluated by the frozen C80 gate")
    return arms[arm], optimizer


def validate_prerequisite_receipts(parent, training_config, gate_config, receipts):
    """Bind completed receipts and require both full-C80 finite-step descents."""
    for key in SOURCE_KEYS:
        receipt = receipts.get(key)
        unit = key
        if (not isinstance(receipt, dict) or receipt.get("protocol") != parent.get("protocol")
                or receipt.get("unit") != unit or receipt.get("complete") is not True
                or receipt.get("pbs_jobid") != training_config["source_jobs"][key]):
            raise ValueError(f"Completed prerequisite binding differs: {key}")
    teacher = receipts["training_teacher"]
    if (teacher.get("expected_records") != 160 or teacher.get("completed_records") != 160
            or teacher.get("contexts") != 80 or teacher.get("planned_new_teacher_queries") != 48
            or teacher.get("unique_new_teacher_outputs") != 48
            or teacher.get("total_new_query_completed") != 48
            or teacher.get("initial_noise_parity_valid") is not True
            or teacher.get("initial_noise_parity_contexts") != 80
            or teacher.get("frozen_model_unchanged") is not True
            or teacher.get("training_updates") != 0
            or teacher.get("heldout_inputs_used") is not False
            or teacher.get("evaluation_trajectory_inputs_used") is not False):
        raise ValueError("Training-teacher cache does not satisfy all 160-record C80 gates")
    loss_check = receipts["training_loss_selfcheck"]
    if ("test_training_objective.py" not in loss_check.get("passed_checks", [])
            or loss_check.get("model_queries") != 0 or loss_check.get("training_updates") != 0):
        raise ValueError("Training loss selfcheck did not pass without model work")
    gate = receipts["aggregate_c80_parameter_gate"]
    if (gate.get("gate_protocol") != gate_config.get("protocol")
            or gate.get("full_c80_finite_step_evaluated") is not True
            or gate.get("training_updates") != 0
            or gate.get("heldout_inputs_used") is not False
            or gate.get("sensor_evaluation_inputs_used") is not False):
        raise ValueError("Aggregate C80 receipt is not a complete calibration-only finite-step gate")
    fraction = training_config["optimizer"]["gate_fraction"]
    expected_objectives = set(gate_config.get("objective_keys", {}).values())
    objectives = gate.get("objectives")
    if not isinstance(objectives, dict) or not expected_objectives.issubset(objectives):
        raise ValueError("Aggregate C80 receipt is missing either gated objective")
    for objective in expected_objectives:
        result = objectives[objective]
        if result.get("cases") != 80:
            raise ValueError(f"C80 gate objective does not cover all 80 cases: {objective}")
        candidates = [row for row in result.get("candidates", [])
                     if row.get("fraction") == fraction]
        if len(candidates) != 1 or candidates[0].get("cases") != 80 or candidates[0].get("actual_descent") is not True:
            raise ValueError(f"Configured gate fraction lacks a full-C80 actual descent: {objective}")
    return True


def validate_resume_checkpoint(checkpoint, contract, update_order):
    """Validate exact frozen bindings and a contiguous, unreordered case prefix."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Training checkpoint must be a mapping")
    for key, value in contract.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Training checkpoint binding differs: {key}")
    if checkpoint.get("update_order") != update_order:
        raise ValueError("Training checkpoint update order differs")
    index = checkpoint.get("update_index")
    history = checkpoint.get("history")
    if (isinstance(index, bool) or not isinstance(index, int) or index < 0
            or index > len(update_order) or not isinstance(history, list)
            or len(history) != index):
        raise ValueError("Training checkpoint index/history length differs")
    if [row.get("case_id") for row in history] != update_order[:index] or any(
            row.get("update") != position for position, row in enumerate(history, 1)):
        raise ValueError("Training checkpoint history is not the exact completed prefix")
    for key in ("ratios", "shadow_log2_scales", "optimizer", "rng"):
        if key not in checkpoint:
            raise ValueError(f"Training checkpoint is missing {key}")
    return index


def _atomic_torch_save(torch, path, payload):
    from checkpointing import atomic_torch_save
    atomic_torch_save(torch, path, payload)


def _record_job_segment(history, segment):
    result = list(history or [])
    if not segment:
        return result
    job = segment.get("pbs_jobid")
    if not job or any(item.get("pbs_jobid") == job for item in result):
        return result
    result.append(dict(segment))
    return result


def _metadata_teacher_bindings(torch, np, switching, p, root, data_config,
                               cases, plan_by_id, teacher_plan_job, calibration_jobs):
    import training_teacher as teacher
    from diagnostic_runtime import datum_identity

    targets, datums_meta, bindings = {}, {}, {}
    output_root = root / "results" / "training_teacher"
    for case_id in CASE_IDS:
        case = cases[case_id]
        planned = plan_by_id[case_id]
        training_meta = read(root / data_config["dataset_subdir"] / str(case_id) / "metadata.json")
        sampler_seed = int(training_meta["sampler_seed"])
        description = training_meta["description"]
        observations = {
            "clean": teacher._load_npz(np, teacher._resolve(root, case["clean_observation"])),
            "noisy": teacher._load_npz(np, teacher._resolve(root, training_meta["noisy_observation"])),
        }
        datums = {variant: switching.datum(observations[variant]) for variant in VARIANTS}
        identities = {variant: datum_identity(torch, datums[variant]) for variant in VARIANTS}
        if identities["clean"] == identities["noisy"]:
            raise ValueError(f"Training clean/noisy input identities coincide for {case_id}")
        eligibility = teacher._source_eligibility(planned)
        source_job = planned["source_calibration_pbs_jobid"]
        if source_job != calibration_jobs[f"calibration_{planned['shard']}"]:
            raise ValueError(f"Training teacher calibration source job differs for {case_id}")
        for variant in VARIANTS:
            binding = teacher._binding(p["protocol"], case_id, variant, sampler_seed,
                                       description, identities[variant])
            source_path, variant_eligibility = teacher._reuse_source(planned, variant)
            source_path = teacher._resolve(root, source_path) if source_path is not None else None
            tensor_path, metadata_path = teacher._record_paths(output_root, case_id, variant)
            if not tensor_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Missing cached training teacher record: {case_id}/{variant}")
            payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
            metadata = read(metadata_path)
            outputs = teacher._validate_training_record(
                torch, payload, metadata, binding, data_config, teacher_plan_job,
                source_job, source_path, variant_eligibility)
            if variant == "clean":
                clean_noise = outputs["action_initial_noise"]
            else:
                teacher._noise_equal(torch, clean_noise, outputs["action_initial_noise"])
            payload.update(outputs)
            targets[(case_id, variant)] = payload
            bindings[f"{case_id}:{variant}"] = {
                "binding": binding,
                "producer_job": metadata["producer_job"],
                "source_calibration_job": metadata["source_calibration_job"],
                "source_file": metadata["source_file"],
                "source_kind": metadata["source_kind"],
                "source_eligibility": metadata["source_eligibility"],
            }
        datums_meta[case_id] = {
            "description": description, "sampler_seed": sampler_seed,
            "identities": identities,
        }
        del observations, datums
    if len(targets) != 160 or len(bindings) != 160:
        raise RuntimeError("Cached training teacher coverage is not exactly 160 records")
    return targets, datums_meta, bindings


def _save_progress(out, state, stage, **extra):
    state["stage"] = stage
    state.update(extra)
    save(out / "summary.json", state)
    save(out / "progress.json", state)
    print("CONTROL_TRAINING_PROGRESS " + json.dumps({
        "unit": state["unit"], "pbs_jobid": state["pbs_jobid"],
        "stage": stage, "completed_updates": state["completed_updates"],
        "this_job": state["this_job"],
    }, sort_keys=True), flush=True)


def _student_forward(torch, switching, runtime, banks, datum, seed, description,
                     capture_rng, restore_rng, job_state, out,
                     variant, reference=None, gradient=False):
    from diagnostic_runtime import _capture_action_initial_noise

    if banks.cached and gradient:
        raise RuntimeError("Student VJP cannot run after deployment cache folding")
    runtime.description = description
    before_rng = capture_rng(torch)
    banks.seen.clear()
    job_state["student_query_attempted"] += 1
    state = job_state["summary"]
    state["this_job"] = {key: value for key, value in job_state.items()
                         if key != "summary"}
    _save_progress(out, state, "student_forward",
                   active_variant=variant, gradient=gradient)
    try:
        with _capture_action_initial_noise(runtime.model, torch) as captured:
            with torch.set_grad_enabled(gradient):
                action, counts, future = switching.sampler.differentiable_action(
                    runtime, datum, seed, checkpointed=gradient, return_future=True)
        noise = captured["value"]
        if (len(banks.seen) != 614 or counts != COUNTS
                or tuple(action.shape) != ACTION_SHAPE
                or tuple(future.shape) != FUTURE_SHAPE
                or not bool(torch.isfinite(action).all())
                or not bool(torch.isfinite(future).all())
                or not isinstance(noise, torch.Tensor)
                or tuple(noise.shape) != NOISE_SHAPE
                or not bool(torch.isfinite(noise).all())):
            raise RuntimeError("Dynamic endpoint coverage, 10+10 counts, shape, or finiteness failed")
        if reference is not None:
            if (action.dtype != reference["action"].dtype
                    or future.dtype != reference["future"].dtype
                    or not torch.equal(action.detach().cpu(), reference["action"])
                    or not torch.equal(future.detach().cpu(), reference["future"])
                    or not torch.equal(noise, reference["action_initial_noise"])):
                raise RuntimeError("Same-parameter endpoint/dtype or initial-noise parity failed")
        job_state["student_query_completed"] += 1
        if gradient:
            return action, future
        return {
            "action": action.detach().cpu().clone(),
            "future": future.detach().cpu().clone(),
            "counts": dict(counts),
            "action_initial_noise": noise.detach().cpu().clone(),
        }
    finally:
        restore_rng(torch, before_rng)


def _reader_validate(torch, result, expected_noise, label):
    action, counts = result
    noise = counts.get("action_initial_noise")
    if (tuple(action.shape) != ACTION_SHAPE or not bool(torch.isfinite(action).all())
            or counts.get("action_scheduler_steps") != 10
            or counts.get("video_scheduler_steps") != 10
            or counts.get("teacher_future_injection_verified") is not True
            or counts.get("future_tail_matches") is not True
            or counts.get("first_frame_clamp_verified") is not True
            or not isinstance(noise, torch.Tensor)
            or not torch.equal(noise, expected_noise)):
        raise RuntimeError(f"BF16 reader counts/noise/future checks failed: {label}")
    return action, counts


@contextmanager
def _reader_context(torch, switching, runtime, device, job_state, out):
    from parameter_gate import parked_student_weights, resident_weights
    from frozen_reader import BF16Reader

    with parked_student_weights(torch, switching):
        with resident_weights(torch, switching.original_weights, device) as weights:
            reader = BF16Reader(runtime, weights)
            try:
                yield reader
            finally:
                reader.original_weights.clear()


def run(unit):
    here = Path(__file__).resolve().parent
    p = read(here / "protocol.json")
    root = Path(p["remote_root"])
    out = root / "results" / unit
    out.mkdir(parents=True, exist_ok=True)
    current_job = os.environ["PBS_JOBID"]
    arm = unit[len(UNIT_PREFIX):]
    previous_summary = read(out / "summary.json") if (out / "summary.json").is_file() else None
    if previous_summary and previous_summary.get("pbs_jobid") != current_job:
        old_job = previous_summary.get("pbs_jobid")
        if old_job:
            save(out / "jobs" / f"{old_job}.json", previous_summary)
    job_history = list(previous_summary.get("job_history", [])) if previous_summary else []
    if previous_summary:
        job_history = _record_job_segment(job_history, previous_summary.get("this_job"))
    state = {
        "protocol": p.get("protocol"), "unit": unit, "arm": arm,
        "pbs_jobid": current_job, "complete": False, "checkpointed": False,
        "stage": "preflight", "status": "running", "completed_updates": 0,
        "training_updates": 0, "saved_trained_checkpoints": 0,
        "new_teacher_queries": 0, "full_bf16_teacher_queries": 0,
        "heldout_inputs_used": False, "evaluation_trajectory_inputs_used": False,
        "environment_steps": 0, "reader_native_anchor_performed": False,
        "job_history": job_history,
        "this_job": {
            "pbs_jobid": current_job, "student_query_attempted": 0,
            "student_query_completed": 0, "reader_pass_attempted": 0,
            "reader_pass_completed": 0, "reader_weight_stagings": 0,
            "updates_attempted": 0, "updates_completed": 0,
            "stage": "preflight",
        },
    }
    save(out / "summary.json", state)
    switching = None
    failure = None
    checkpoint_path = out / "checkpoint.pt"
    try:
        config_path = here / "training_protocol.json"
        if not config_path.is_file():
            raise FileNotFoundError("Frozen training_protocol.json is required before model loading")
        data_config = read(here / "training_data_protocol.json")
        training_config = read(config_path)
        gate_config = read(here / "c80_gate_protocol.json")
        arm_loss, optimizer_config = validate_training_protocol(
            p, data_config, training_config, arm, gate_config)
        state["training_protocol"] = training_config
        state["data_config"] = data_config

        from aggregate import completed
        receipts = {
            "training_teacher": completed(root, "training_teacher"),
            "training_loss_selfcheck": completed(root, "training_loss_selfcheck"),
            "aggregate_c80_parameter_gate": completed(root, "aggregate_c80_parameter_gate"),
        }
        validate_prerequisite_receipts(p, training_config, gate_config, receipts)
        # This CPU selfcheck is also a scheduler prerequisite for train_*.
        runtime_check = completed(root, "training_runtime_selfcheck")
        if (runtime_check.get("protocol") != p["protocol"]
                or "test_control_training.py" not in runtime_check.get("passed_checks", [])
                or runtime_check.get("model_queries") != 0
                or runtime_check.get("training_updates") != 0):
            raise ValueError("Training runtime selfcheck receipt differs")
        pairs_summary = completed(root, "training_pairs")
        plan_summary = completed(root, "training_teacher_plan")
        calibration_receipts = {
            shard: completed(root, f"calibration_{shard}") for shard in (0, 1)
        }
        if (receipts["training_teacher"].get("source_jobs", {}).get("training_pairs")
                != pairs_summary.get("pbs_jobid")
                or receipts["training_teacher"].get("source_jobs", {}).get("training_teacher_plan")
                != plan_summary.get("pbs_jobid")):
            raise ValueError("Training teacher source pair/plan job bindings differ")
        for shard in (0, 1):
            if receipts["training_teacher"].get("source_jobs", {}).get(
                    f"calibration_{shard}") != calibration_receipts[shard].get("pbs_jobid"):
                raise ValueError(f"Training teacher calibration source job differs: {shard}")
        if (receipts["training_teacher"].get("new_teacher_queries")
                != receipts["training_teacher"].get("total_new_query_attempted")
                or receipts["training_teacher"].get("successful_new_teacher_queries")
                != receipts["training_teacher"].get("total_new_query_completed")):
            raise ValueError("Training teacher query-attempt ledger counters differ")
        import numpy as np
        import training_teacher as teacher
        _, cases, plan_by_id, calibration_jobs, planned_queries = teacher._validate_training_definition(
            root, p, data_config, pairs_summary, plan_summary, calibration_receipts, np)
        if planned_queries != 48:
            raise ValueError("The frozen training plan must contain 48 unique noisy queries")
        state["source_jobs"] = {
            **training_config["source_jobs"],
            "training_pairs": pairs_summary["pbs_jobid"],
            "training_teacher_plan": plan_summary["pbs_jobid"],
            "calibration_0": calibration_receipts[0]["pbs_jobid"],
            "calibration_1": calibration_receipts[1]["pbs_jobid"],
            "training_runtime_selfcheck": runtime_check["pbs_jobid"],
        }
        state["gate_fraction"] = optimizer_config["gate_fraction"]
        state["new_teacher_queries"] = 0
        state["full_bf16_teacher_queries"] = 0
        _save_progress(out, state, "preflight_complete")

        import torch
        from checkpointing import (capture_rng, optimizer_to_device, restore_rng, to_cpu)
        from diagnostic_runtime import SwitchingRuntime, datum_identity
        from parameter_gate import dynamic_student, frozen_snapshot, paired_vjp
        from training_objective import paired_endpoint_objective, weighted_reader_objective

        deadline = Deadline(6600)
        state["deadline_seconds"] = 6600
        state["status"] = "loading_model"
        _save_progress(out, state, "loading_model")
        switching = SwitchingRuntime(p, out)
        runtime, banks = switching.runtime, switching.banks
        # Preserve the original BF16 masters on CPU. Reader calls stage one immutable GPU copy.
        device = next(runtime.model.parameters()).device
        dynamic_student(switching, "per_channel_clip")
        switching.original_weights = {
            name: value.detach().cpu().clone()
            for name, value in switching.original_weights.items()
        }
        torch.cuda.empty_cache()

        targets, case_meta, teacher_bindings = _metadata_teacher_bindings(
            torch, np, switching, p, root, data_config, cases, plan_by_id,
            plan_summary["pbs_jobid"], {str(k): v for k, v in calibration_jobs.items()})
        contract = {
            "protocol": p, "unit": unit, "arm": arm,
            "training_protocol": training_config, "data_config": data_config,
            "gate_protocol": gate_config, "source_jobs": state["source_jobs"],
            "teacher_bindings": teacher_bindings,
        }
        groups = banks.optimizer_groups(p["source_closed_protocol"]["training"],
                                        p["source_closed_protocol"]["scaling"])
        if len(groups) != 2 or len(groups[0]["params"]) != 614 or len(groups[1]["params"]) != 614:
            raise RuntimeError("Expected 614 ratio and 614 channel-shadow tensors")
        base_lrs = (p["source_closed_protocol"]["training"]["clip_learning_rate"],
                    p["source_closed_protocol"]["scaling"]["learning_rate"])
        for group, base_lr in zip(groups, base_lrs):
            if group["lr"] != base_lr:
                raise ValueError("ChannelBanks optimizer learning rates differ from frozen source")
            group["lr"] = float(base_lr) * float(optimizer_config["gate_fraction"])
        parameters = [parameter for group in groups for parameter in group["params"]]
        expected_trainables = list(banks.ratios.values()) + list(banks.shadows.values())
        if ({id(value) for value in parameters} != {id(value) for value in expected_trainables}
                or len(parameters) != 1228
                or any(not value.requires_grad for value in parameters)
                or any(value.requires_grad for value in runtime.model.parameters())
                or any(value.requires_grad for value in list(banks.books.values()) + list(banks.logs.values()))):
            raise RuntimeError("Only the 614 ratios and 614 channel shadows may be trainable")
        baseline_path = p["source_closed_protocol"]["sources"]["baseline_deployment"]
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        if (not _same_tensor_mapping(
                    torch, banks.clip_snapshot(), switching.deployments["per_channel_clip"]["ratios"])
                or not _same_tensor_mapping(
                    torch, banks.scale_snapshot(), switching.deployments["per_channel_clip"]["log2_scales"])):
            raise RuntimeError("Initial D/r differs from the learned per_channel_clip checkpoint")
        frozen_before = frozen_snapshot(switching, parameters, relocatable_student_weights=True)
        stats = read(Path(p["source_relational_protocol"]["source_dual_root"])
                     / "results/fit_stats/summary.json")
        if stats.get("complete") is not True or stats.get("heldout_used") is not False:
            raise ValueError("Frozen calibration-only loss statistics receipt is incomplete")
        optimizer = torch.optim.Adam(
            groups, betas=tuple(float(value) for value in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]))

        history = []
        update_index = 0
        if checkpoint_path.is_file():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            update_index = validate_resume_checkpoint(saved, {
                **contract, "update_order": training_config["update_order"],
            }, training_config["update_order"])
            history = saved["history"]
            _validate_tensor_mapping(torch, saved["ratios"], banks.clip_snapshot(), "ratio")
            _validate_tensor_mapping(torch, saved["shadow_log2_scales"],
                                     banks.scale_snapshot(), "channel scale")
            banks.restore_clips(saved["ratios"])
            banks.restore_scales(saved["shadow_log2_scales"])
            optimizer.load_state_dict(saved["optimizer"])
            optimizer_to_device(torch, optimizer, device)
            restore_rng(torch, saved["rng"])
            # Per-job counters are audit history, not a resume gate. Exact frozen
            # bindings and the completed update prefix control recovery.
        else:
            history = []
            update_index = 0
        state["completed_updates"] = update_index
        state["training_updates"] = update_index
        state["saved_trained_checkpoints"] = update_index
        state["checkpoint_path"] = str(checkpoint_path)
        state["teacher_record_count"] = len(teacher_bindings)
        state["memory_strategy"] = "CPU immutable BF16 reader masters; parked student dense weights during reader calls"
        job_state = {"summary": state, **state["this_job"]}

        def persist(stage, **extra):
            job_state["stage"] = stage
            state["this_job"] = {key: value for key, value in job_state.items()
                                 if key != "summary"}
            _save_progress(out, state, stage, **extra)

        def student_call(case_id, variant, datum, gradient=False, reference=None):
            meta = case_meta[case_id]
            return _student_forward(torch, switching, runtime, banks, datum,
                                    meta["sampler_seed"], meta["description"],
                                    capture_rng, restore_rng, job_state, out,
                                    variant, reference, gradient)

        if not checkpoint_path.is_file():
            _atomic_torch_save(torch, checkpoint_path, {
                **contract, "update_order": training_config["update_order"],
                "update_index": 0, "history": [],
                "ratios": banks.clip_snapshot(),
                "shadow_log2_scales": banks.scale_snapshot(),
                "optimizer": to_cpu(optimizer.state_dict()),
                "rng": capture_rng(torch), "job_history": job_history,
            })

        # Each PBS process repeats the native reader anchor on its first pair;
        # previous anchor results remain in history as per-job evidence.
        anchor_done = False
        state["reader_native_anchor_performed"] = False
        recent_seconds = [float(row["seconds"]) for row in history[-3:] if row.get("seconds")]
        while update_index < len(training_config["update_order"]):
            reserve = max(300.0, (max(recent_seconds) * 1.2)
                          if recent_seconds else 360.0)
            if deadline.near(reserve):
                state["status"] = "checkpointed"
                state["checkpointed"] = True
                state["complete"] = False
                persist("checkpointed", reason="soft deadline before next complete paired update")
                return 0
            case_id = training_config["update_order"][update_index]
            case = cases[case_id]
            meta_path = root / data_config["dataset_subdir"] / str(case_id) / "metadata.json"
            training_meta = read(meta_path)
            observations = {
                "clean": teacher._load_npz(np, teacher._resolve(root, case["clean_observation"])),
                "noisy": teacher._load_npz(np, teacher._resolve(root, training_meta["noisy_observation"])),
            }
            datums = {variant: switching.datum(observations[variant]) for variant in VARIANTS}
            identities = {variant: datum_identity(torch, datums[variant]) for variant in VARIANTS}
            expected_identities = case_meta[case_id]["identities"]
            if identities != expected_identities:
                raise RuntimeError(f"Training input identity changed after prevalidation: {case_id}")
            start = time.monotonic()
            job_state["updates_attempted"] += 1
            persist("paired_forward", active_case=case_id, update=update_index + 1)
            endpoints = {}
            for variant in VARIANTS:
                endpoints[variant] = student_call(case_id, variant, datums[variant])
                if not torch.equal(endpoints[variant]["action_initial_noise"],
                                   targets[(case_id, variant)]["action_initial_noise"]):
                    raise RuntimeError(f"Student/teacher initial-noise parity failed: {case_id}/{variant}")
            teacher._noise_equal(torch, endpoints["clean"]["action_initial_noise"],
                                 endpoints["noisy"]["action_initial_noise"])
            if update_index == 0:
                clean_case = {**cases[case_id], "description": case_meta[case_id]["description"]}
                source_unit = f"calibration_{clean_case['shard']}"
                source_path = root / "results" / source_unit / "records" / (
                    f"{case_id}_per_channel_clip_clean.pt")
                from calibration_worker import _pair_binding
                from diagnostic_runtime import validate_binding
                cached = torch.load(source_path, map_location="cpu", weights_only=True)
                source_binding = _pair_binding(p["protocol"], source_unit, clean_case,
                                               "per_channel_clip", "clean")
                source_binding["input_identity"] = identities["clean"]
                validate_binding(cached, source_binding)
                if (not torch.equal(endpoints["clean"]["action"].float(), cached["action"].float())
                        or not torch.equal(endpoints["clean"]["future"].float(), cached["future"].float())):
                    raise RuntimeError("Initial clean dynamic endpoint differs from calibration cache")
                state["initial_clean_calibration_cache_parity"] = True
            leaves_action = {v: endpoints[v]["action"].to(device).detach().requires_grad_(True)
                             for v in VARIANTS}
            leaves_future = {v: endpoints[v]["future"].to(device).detach().requires_grad_(True)
                             for v in VARIANTS}
            endpoint_loss, raw_terms = paired_endpoint_objective(
                torch, leaves_action, leaves_future,
                {v: targets[(case_id, v)] for v in VARIANTS}, stats, arm_loss)
            if not endpoint_loss.requires_grad or not bool(torch.isfinite(endpoint_loss)):
                raise RuntimeError("Paired endpoint objective is disconnected or non-finite")
            raw_record = {key: float(value.detach()) for key, value in raw_terms.items()}
            endpoint_loss_value = float(endpoint_loss.detach())
            vector_order = [leaves_action["clean"], leaves_action["noisy"],
                            leaves_future["clean"], leaves_future["noisy"]]
            vectors = torch.autograd.grad(endpoint_loss, vector_order)
            cotangents = {
                "clean": [vectors[0].detach(), vectors[2].detach()],
                "noisy": [vectors[1].detach(), vectors[3].detach()],
            }
            del endpoint_loss, raw_terms, vectors, vector_order

            reader_raw = {}
            reader_weighted = {}
            reader_anchor = False
            if float(arm_loss["control_weight"]) > 0:
                job_state["reader_weight_stagings"] += 1
                with _reader_context(torch, switching, runtime, device, job_state, out) as reader:
                    for variant in VARIANTS:
                        if not anchor_done and variant == "clean":
                            for anchor_variant in VARIANTS:
                                job_state["reader_pass_attempted"] += 1
                                persist("reader_native_anchor", active_case=case_id,
                                        active_variant=anchor_variant)
                                output = reader(runtime, datums[anchor_variant],
                                                case_meta[case_id]["sampler_seed"],
                                                targets[(case_id, anchor_variant)]["future"],
                                                checkpointed=False)
                                action, counts = _reader_validate(
                                    torch, output,
                                    targets[(case_id, anchor_variant)]["action_initial_noise"],
                                    f"native anchor {case_id}/{anchor_variant}")
                                if not torch.equal(action.detach().float().cpu(),
                                                   targets[(case_id, anchor_variant)]["action"].float()):
                                    raise RuntimeError("BF16 future->native teacher action anchor failed")
                                job_state["reader_pass_completed"] += 1
                            reader_anchor = True
                            anchor_done = True
                        job_state["reader_pass_attempted"] += 1
                        persist("reader_gradient", active_case=case_id,
                                active_variant=variant)
                        reader_action, reader_counts = reader(
                            runtime, datums[variant], case_meta[case_id]["sampler_seed"],
                            leaves_future[variant], checkpointed=True)
                        _reader_validate(torch, (reader_action, reader_counts),
                                         endpoints[variant]["action_initial_noise"],
                                         f"gradient reader {case_id}/{variant}")
                        reader_loss, raw_control = weighted_reader_objective(
                            torch, reader_action, targets[(case_id, variant)]["action"],
                            stats, arm_loss)
                        vector, = torch.autograd.grad(reader_loss, leaves_future[variant])
                        if (tuple(vector.shape) != FUTURE_SHAPE
                                or not bool(torch.isfinite(vector).all())
                                or bool(vector[:, :, 0].ne(0).any())):
                            raise RuntimeError("Reader future cotangent shape/finite/frame-zero gate failed")
                        # Preserve endpoint dtype so BF16 cotangents use the same rounding before addition.
                        cotangents[variant][1] = cotangents[variant][1] + vector.detach().to(
                            dtype=cotangents[variant][1].dtype)
                        reader_raw[variant] = float(raw_control.detach())
                        reader_weighted[variant] = float(reader_loss.detach())
                        job_state["reader_pass_completed"] += 1
                        del reader_action, reader_counts, reader_loss, raw_control, vector
                if reader_anchor:
                    state["reader_native_anchor_performed"] = True

            for variant in VARIANTS:
                action_vector, future_vector = cotangents[variant]
                if (bool(action_vector[10:, :].ne(0).any())
                        or bool(action_vector[:10, 6].ne(0).any())
                        or bool(future_vector[:, :, 0].ne(0).any())):
                    raise RuntimeError("Endpoint cotangents escaped first10-motor6/frame-zero mask")

            gradients = paired_vjp(
                torch, parameters,
                lambda variant: student_call(
                    case_id, variant, datums[variant], gradient=True,
                    reference=endpoints[variant]),
                {variant: tuple(cotangents[variant]) for variant in VARIANTS})
            if (len(gradients) != len(parameters)
                    or any(value is None or not bool(torch.isfinite(value).all())
                           for value in gradients)):
                raise RuntimeError("D/r paired VJP has disconnected or non-finite parameter gradients")
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient.detach()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(optimizer_config["gradient_norm_clip"]),
                error_if_nonfinite=True)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("D/r gradient norm is non-finite")
            optimizer.step()
            if update_index == 0 and optimizer_config["reset_after_update1"]:
                optimizer.state.clear()
            banks.project()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            new_index = update_index + 1
            record = {
                "update": new_index, "case_id": case_id,
                "seed": case_meta[case_id]["sampler_seed"],
                "input_identities": identities,
                "raw_endpoint_terms": raw_record,
                "reader_control_kl": reader_raw,
                "reader_weighted_control_loss": reader_weighted,
                "endpoint_loss": endpoint_loss_value,
                "optimization_loss": endpoint_loss_value + sum(reader_weighted.values()),
                "reader_native_anchor": reader_anchor,
                "gradient_norm_before_clip": float(norm), "seconds": elapsed,
                "pbs_jobid": current_job,
            }
            history.append(record)
            update_index = new_index
            job_state["updates_completed"] += 1
            state["completed_updates"] = update_index
            state["training_updates"] = update_index
            state["last_update"] = record
            recent_seconds = (recent_seconds + [elapsed])[-3:]
            checkpoint = {
                **contract, "update_order": training_config["update_order"],
                "update_index": update_index, "history": history,
                "ratios": banks.clip_snapshot(),
                "shadow_log2_scales": banks.scale_snapshot(),
                "optimizer": to_cpu(optimizer.state_dict()),
                "rng": capture_rng(torch), "job_history": job_history,
            }
            _atomic_torch_save(torch, checkpoint_path, checkpoint)
            state["saved_trained_checkpoints"] += 1
            persist("update_checkpointed", last_update=record)
            del gradients, cotangents, leaves_action, leaves_future, endpoints
            del observations, datums, identities, training_meta

        # Strict frozen checks happen before the deployment fold changes D/r flags and W.
        frozen_result = banks.verify_frozen(baseline["module_masters"])
        if frozen_snapshot(switching, parameters, relocatable_student_weights=True) != frozen_before:
            raise RuntimeError("Frozen model/reader/bank snapshot changed during C80 training")
        if any(value.requires_grad for value in runtime.model.parameters()) or any(
                value.requires_grad for value in list(banks.books.values()) + list(banks.logs.values())):
            raise RuntimeError("Frozen model or VQ bank parameters became trainable")
        if deadline.near(max(300.0, (max(recent_seconds) * 1.2)
                            if recent_seconds else 300.0)):
            state["status"] = "checkpointed"
            state["checkpointed"] = True
            state["complete"] = False
            persist("checkpointed", reason="soft deadline before final deployment parity")
            return 0

        first_case = training_config["update_order"][0]
        meta_path = root / data_config["dataset_subdir"] / str(first_case) / "metadata.json"
        training_meta = read(meta_path)
        first_observations = {
            "clean": teacher._load_npz(np, teacher._resolve(root, cases[first_case]["clean_observation"])),
            "noisy": teacher._load_npz(np, teacher._resolve(root, training_meta["noisy_observation"])),
        }
        first_datums = {variant: switching.datum(first_observations[variant]) for variant in VARIANTS}
        first_dynamic = student_call(first_case, "clean", first_datums["clean"])
        persist("deployment_cache_fold", case_id=first_case)
        rounding = banks.cache_compensated_weights()
        state["post_fold_parameters_require_grad"] = sum(
            bool(parameter.requires_grad) for parameter in parameters)
        if state["post_fold_parameters_require_grad"] != 0:
            raise RuntimeError("Deployment fold did not freeze D/r parameters")
        first_cached = student_call(first_case, "clean", first_datums["clean"])
        folded_parity = {
            "action_max_abs": float((first_dynamic["action"].float()
                                      - first_cached["action"].float()).abs().max()),
            "future_max_abs": float((first_dynamic["future"].float()
                                      - first_cached["future"].float()).abs().max()),
            "initial_noise_max_abs": float((first_dynamic["action_initial_noise"]
                                             - first_cached["action_initial_noise"]).abs().max()),
        }
        if any(value != 0 for value in folded_parity.values()):
            raise RuntimeError("Dynamic versus native cached first-C80 action/future parity failed")
        deployment_path = out / "deployment.pt"
        reference_path = out / "reference_endpoint.pt"
        reference_identity = datum_identity(torch, first_datums["clean"])
        if reference_identity != case_meta[first_case]["identities"]["clean"]:
            raise RuntimeError("Final reference input differs from the frozen training input")
        _atomic_torch_save(torch, reference_path, {
            "protocol": p["protocol"], "training_protocol": training_config,
            "unit": unit, "arm": arm, "pbs_jobid": current_job,
            "completed_updates": 80, "case_id": first_case, "variant": "clean",
            "sampler_seed": case_meta[first_case]["sampler_seed"],
            "description": case_meta[first_case]["description"],
            "input_identity": reference_identity, **first_cached,
        })
        deployment = {
            "protocol": p, "training_protocol": training_config,
            "data_config": data_config, "unit": unit, "arm": arm,
            "checkpoint_arm": "per_channel_clip", "updates": 80,
            "update_order": training_config["update_order"],
            "source_jobs": state["source_jobs"],
            "teacher_bindings": teacher_bindings,
            "ratios": banks.clip_snapshot(), "log2_scales": banks.scale_snapshot(),
            "scale_metadata_dtype": "float32_log2",
            "folded_rounding": rounding,
            "reference_endpoint": str(reference_path),
        }
        _atomic_torch_save(torch, deployment_path, deployment)
        state.update({
            "complete": True, "checkpointed": False, "status": "complete",
            "stage": "complete", "completed_updates": 80, "training_updates": 80,
            "new_teacher_queries": 0, "full_bf16_teacher_queries": 0,
            "deployment": str(deployment_path),
            "reference_endpoint": str(reference_path),
            "native_cached_parity": folded_parity,
            "frozen_model_unchanged": True, "frozen_vq": frozen_result,
            "heldout_inputs_used": False, "evaluation_trajectory_inputs_used": False,
            "evaluation_queries": 0, "environment_steps": 0,
            "this_job": {key: value for key, value in job_state.items() if key != "summary"},
        })
        save(out / "summary.json", state)
        save(out / "progress.json", state)
        return 0
    except Exception as exc:
        failure = exc
        state.update({"complete": False, "checkpointed": False,
                     "status": "failed", "stage": "failed",
                     "failure": f"{type(exc).__name__}: {exc}"})
        if "job_state" in locals():
            state["this_job"] = {key: value for key, value in job_state.items()
                                 if key != "summary"}
        save(out / "summary.json", state)
        save(out / "progress.json", state)
        raise
    finally:
        if switching is not None:
            switching.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"train_[a-z][a-z0-9_]{0,25}", args.unit):
        parser.error("--unit must be train_<arm>")
    guard()
    return run(args.unit)


if __name__ == "__main__":
    raise SystemExit(main())
