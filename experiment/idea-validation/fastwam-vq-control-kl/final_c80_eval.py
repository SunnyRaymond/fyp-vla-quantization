"""Resumable raw-loss evaluation on the complete original training C80 pool."""
import argparse
import math
import os
from pathlib import Path
import re

from control_common import Deadline, guard, read, save


VARIANTS = ("clean", "noisy")
CASE_IDS = tuple(range(2000, 2080))
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)
NOISE_SHAPE = (1, 32, 7)
COUNTS = {"video_steps": 10, "action_steps": 10}
RAW_KEYS = {
    "action_absolute_kl", "action_response_kl", "latent_absolute_kl",
    "control_kl_clean", "control_kl_noisy",
}


class SoftStop(RuntimeError):
    """Deadline reached before a whole pair is ready to commit."""


def case_ids_for_shard(cases, shard):
    if isinstance(shard, bool) or not isinstance(shard, int) or shard not in (0, 1):
        raise ValueError("C80 shard must be 0 or 1")
    selected = sorted(int(case_id) for case_id, case in cases.items()
                      if int(case.get("shard", -1)) == shard)
    expected = [case_id for case_id in CASE_IDS if case_id % 2 == shard]
    if selected != expected or len(selected) != 40:
        raise ValueError("C80 shard must retain its original 40 of 80 training cases")
    return selected


def compute_raw_terms(torch, endpoints, reader_actions, targets, stats, loss_config):
    """Compute unweighted pair and reader losses from one complete pair."""
    if any(set(mapping) != set(VARIANTS) for mapping in
           (endpoints, reader_actions, targets)):
        raise ValueError("Raw-loss evaluation requires clean and noisy variants")
    from training_objective import paired_endpoint_objective, weighted_reader_objective

    actions = {variant: endpoints[variant]["action"] for variant in VARIANTS}
    futures = {variant: endpoints[variant]["future"] for variant in VARIANTS}
    with torch.no_grad():
        _, raw = paired_endpoint_objective(
            torch, actions, futures, targets, stats, loss_config)
        result = {key: float(raw[key].detach().cpu()) for key in (
            "action_absolute_kl", "action_response_kl", "latent_absolute_kl")}
        for variant in VARIANTS:
            _, control = weighted_reader_objective(
                torch, reader_actions[variant], targets[variant]["action"],
                stats, loss_config)
            result[f"control_kl_{variant}"] = float(control.detach().cpu())
    if set(result) != RAW_KEYS or not all(_finite_number(value) for value in result.values()):
        raise ValueError("Raw C80 loss terms must be finite scalars")
    return result


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _tensor(torch, value, shape, label, *, cpu=True):
    if (not torch.is_tensor(value) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())
            or (cpu and value.device.type != "cpu")):
        raise ValueError(f"Invalid {label}; expected finite {shape} CPU tensor")


def validate_record(torch, record, expected_binding):
    """Validate one atomically saved pair against its exact immutable binding."""
    if not isinstance(record, dict) or not isinstance(expected_binding, dict):
        raise ValueError("Pair record and expected binding must be mappings")
    mismatch = [key for key, value in expected_binding.items()
                if record.get(key) != value]
    if mismatch:
        raise ValueError("C80 pair binding differs: " + ", ".join(mismatch))
    if record.get("complete") is not True or record.get("status") != "complete":
        raise ValueError("Partial C80 pair cannot be reused")
    if not isinstance(record.get("producer_job"), str) or not record["producer_job"]:
        raise ValueError("C80 pair lacks its producer PBS job")
    case_id = record.get("case_id")
    seed, description = record.get("sampler_seed"), record.get("description")
    teacher_bindings = record.get("teacher_bindings")
    identities = record.get("input_identities")
    if (not isinstance(case_id, int) or isinstance(case_id, bool)
            or record.get("index") != case_id
            or not isinstance(seed, int) or not isinstance(description, str)
            or not description or not isinstance(teacher_bindings, dict)
            or set(teacher_bindings) != set(VARIANTS)
            or not isinstance(identities, dict) or set(identities) != set(VARIANTS)):
        raise ValueError("C80 case, input, or teacher binding is incomplete")
    for variant in VARIANTS:
        teacher = teacher_bindings[variant]
        binding = teacher.get("binding") if isinstance(teacher, dict) else None
        if (not isinstance(binding, dict)
                or binding.get("unit") != "training_teacher"
                or binding.get("index") != case_id or binding.get("variant") != variant
                or binding.get("arm") != "bf16" or binding.get("seed") != seed
                or binding.get("description") != description
                or binding.get("input_identity") != identities.get(variant)
                or not isinstance(teacher.get("producer_job"), str)
                or not teacher.get("producer_job")):
            raise ValueError(f"Teacher identity/provenance differs for {case_id}/{variant}")
    endpoints = record.get("endpoints")
    readers = record.get("reader_actions")
    reader_counts = record.get("reader_counts")
    if (not isinstance(endpoints, dict) or set(endpoints) != set(VARIANTS)
            or not isinstance(readers, dict) or set(readers) != set(VARIANTS)
            or not isinstance(reader_counts, dict) or set(reader_counts) != set(VARIANTS)):
        raise ValueError("C80 pair endpoints or reader actions are incomplete")
    for variant in VARIANTS:
        endpoint = endpoints[variant]
        if not isinstance(endpoint, dict) or endpoint.get("counts") != COUNTS:
            raise ValueError(f"Endpoint sampler counts differ for {case_id}/{variant}")
        for key, shape in (("action", ACTION_SHAPE), ("future", FUTURE_SHAPE),
                           ("action_initial_noise", NOISE_SHAPE)):
            _tensor(torch, endpoint.get(key), shape, f"endpoint {variant}/{key}")
        _tensor(torch, readers[variant], ACTION_SHAPE, f"reader action {variant}")
        counts = reader_counts[variant]
        if (not isinstance(counts, dict)
                or counts.get("video_prefill_calls") != 1
                or counts.get("action_scheduler_steps") != 10
                or counts.get("video_scheduler_steps") != 10
                or counts.get("teacher_future_injection_verified") is not True
                or counts.get("teacher_future_injection_steps") != 10
                or counts.get("future_tail_matches") is not True
                or counts.get("first_frame_clamp_verified") is not True):
            raise ValueError(f"Reader counts missing for {case_id}/{variant}")
    if not torch.equal(endpoints["clean"]["action_initial_noise"],
                       endpoints["noisy"]["action_initial_noise"]):
        raise ValueError("C80 clean/noisy endpoint initial noise differs")
    raw = record.get("raw_terms")
    if (not isinstance(raw, dict) or set(raw) != RAW_KEYS
            or not all(_finite_number(value) for value in raw.values())):
        raise ValueError("C80 pair raw loss terms are incomplete or non-finite")
    return record


def _load_case(torch, np, switching, teacher, root, data_config, case):
    metadata_path = root / data_config["dataset_subdir"] / str(int(case["case_id"])) / "metadata.json"
    metadata = read(metadata_path)
    _set_case_description(switching.runtime, metadata)
    observations = {
        "clean": teacher._load_npz(np, teacher._resolve(root, case["clean_observation"])),
        "noisy": teacher._load_npz(np, teacher._resolve(root, metadata["noisy_observation"])),
    }
    datums = {variant: switching.datum(observations[variant]) for variant in VARIANTS}
    from diagnostic_runtime import datum_identity

    identities = {variant: datum_identity(torch, datums[variant]) for variant in VARIANTS}
    if identities["clean"] == identities["noisy"]:
        raise ValueError(f"Training views have identical input identity for {case['case_id']}")
    return datums, identities, metadata


def _set_case_description(runtime, metadata):
    description = metadata.get("description") if isinstance(metadata, dict) else None
    if not isinstance(description, str) or not description:
        raise ValueError("Training-pair task description is missing")
    runtime.description = description
    return description


def _reader_pass(torch, reader, runtime, datum, seed, future, expected_noise,
                 deadline, state, publish, control_training, label):
    if deadline.near(300):
        raise SoftStop(f"deadline before BF16 reader pass: {label}")
    state["reader_pass_attempted"] += 1
    publish()
    with torch.no_grad():
        result = reader(runtime, datum, seed, future, checkpointed=False)
    action, counts = control_training._reader_validate(torch, result, expected_noise, label)
    if (counts.get("video_prefill_calls") != 1
            or counts.get("teacher_future_injection_steps") != 10
            or counts.get("action_conditioning_future_matches") is not True
            or counts.get("action_conditioning_first_frame_clamped") is not True
            or counts.get("future_first_frame_matches_observation") is not True):
        raise RuntimeError(f"BF16 reader injection/prefill checks failed: {label}")
    action = action.detach().float().cpu()
    state["reader_pass_completed"] += 1
    publish()
    return action, {key: counts[key] for key in (
        "video_prefill_calls", "action_scheduler_steps", "video_scheduler_steps",
        "teacher_future_injection_steps", "teacher_future_injection_verified",
        "future_tail_matches", "first_frame_clamp_verified",
        "action_conditioning_future_matches",
        "action_conditioning_first_frame_clamped",
        "future_first_frame_matches_observation")}


def _atomic_torch_save(torch, path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _archive_summary(out, job_id):
    path = out / "summary.json"
    if not path.is_file():
        return None
    previous = read(path)
    old_job = previous.get("pbs_jobid")
    if old_job == job_id:
        return previous
    if not isinstance(old_job, str) or not old_job or Path(old_job).name != old_job:
        raise ValueError("Cannot archive prior final-C80 summary without a safe PBS job id")
    archived = out / "jobs" / f"{old_job}.json"
    if archived.exists() and read(archived) != previous:
        raise ValueError("Existing archived final-C80 PBS summary conflicts")
    save(archived, previous)
    path.unlink()
    return None


def _publish(out, state, *, summary=True):
    state["completed_pairs"] = len(state["completed_case_ids"])
    state["completed_endpoints"] = 2 * state["completed_pairs"]
    state["this_job"].update({key: state[key] for key in (
        "startup_query_attempted", "startup_query_completed",
        "endpoint_query_attempted", "endpoint_query_completed",
        "reader_pass_attempted", "reader_pass_completed",
        "reader_anchor_passes_attempted", "reader_anchor_passes_completed",
        "calibration_source_reuse_endpoints", "existing_pair_reuse")})
    save(out / "progress.json", state)
    if summary:
        save(out / "summary.json", state)


def _cached_mse_endpoint(torch, teacher, calibration_worker, root, parent,
                         calibration_jobs, case, variant, identity, description):
    shard = int(case["shard"])
    unit = f"calibration_{shard}"
    path = root / "results" / unit / "records" / (
        f"{int(case['case_id'])}_per_channel_clip_{variant}.pt")
    binding = calibration_worker._pair_binding(
        parent["protocol"], unit,
        {"case_id": int(case["case_id"]), "sampler_seed": int(case["sampler_seed"]),
         "description": description}, "per_channel_clip", variant)
    binding["input_identity"] = identity
    outputs, metadata = teacher._read_source_record(torch, path, binding)
    if calibration_jobs.get(shard) is None:
        raise ValueError(f"Missing calibration source PBS for case {case['case_id']}")
    return outputs, {
        "kind": "calibration_reuse", "source_file": str(path),
        "source_calibration_job": calibration_jobs[shard],
        "source_binding": binding,
        "source_counts": metadata.get("counts"),
    }


def _new_state(parent, unit, arm, shard, job_id, config, data_config):
    return {
        "protocol": parent["protocol"], "unit": unit, "arm": arm, "shard": shard,
        "pbs_jobid": job_id, "status": "running", "complete": False,
        "checkpointed": False, "expected_pairs": 40, "completed_pairs": 0,
        "expected_endpoints": 80, "completed_endpoints": 0,
        "completed_case_ids": [], "remaining_case_ids": [],
        "training_protocol": config, "data_config": data_config,
        "deployment_binding": None, "source_jobs": {}, "teacher_job": None,
        "startup_query_attempted": 0, "startup_query_completed": 0,
        "endpoint_query_attempted": 0, "endpoint_query_completed": 0,
        "reader_pass_attempted": 0, "reader_pass_completed": 0,
        "reader_anchor_passes_attempted": 0, "reader_anchor_passes_completed": 0,
        "reader_native_anchor_performed": False,
        "calibration_source_reuse_endpoints": 0, "existing_pair_reuse": 0,
        "planned_fresh_endpoint_outputs": None,
        "new_bf16_teacher_queries": 0, "training_updates": 0,
        "saved_trained_checkpoints": 0, "environment_steps": 0,
        "heldout_inputs_used": False, "evaluation_trajectory_inputs_used": False,
        "all_initial_noise_parity_valid": False,
        "this_job": {"pbs_jobid": job_id},
    }


def _source_jobs(pairs, plan, teacher, calibrations, losses, gate, runtime_check):
    expected = {
        "training_pairs": pairs, "training_teacher_plan": plan,
        "training_teacher": teacher, "training_loss_selfcheck": losses,
        "aggregate_c80_parameter_gate": gate,
        "calibration_0": calibrations[0], "calibration_1": calibrations[1],
        "training_runtime_selfcheck": runtime_check,
    }
    return {key: receipt["pbs_jobid"] for key, receipt in expected.items()}


def run(unit):
    guard()
    import numpy as np
    import torch
    from aggregate import completed
    import control_training
    import training_teacher as teacher
    import training_evaluation
    import calibration_worker
    from diagnostic_runtime import SwitchingRuntime
    from frozen_reader import BF16Reader

    match = re.fullmatch(r"final_c80_(mse_reference|[a-z][a-z0-9_]{0,25})_([01])", unit)
    if not match:
        raise ValueError("Expected final_c80_<arm>_[01]")
    arm, shard = match.group(1), int(match.group(2))
    here = Path(__file__).resolve().parent
    parent, data_config = read(here / "protocol.json"), read(here / "training_data_protocol.json")
    training_config = read(here / "training_protocol.json")
    gate_config = read(here / "c80_gate_protocol.json")
    candidate_arms = training_config.get("arms")
    if not isinstance(candidate_arms, dict) or not candidate_arms:
        raise ValueError("Frozen training protocol must define at least one explicit loss arm")
    if arm != "mse_reference" and arm not in candidate_arms:
        raise ValueError(f"Unknown trained C80 arm: {arm}")
    loss_arm = arm if arm != "mse_reference" else next(iter(candidate_arms))
    loss_config, _ = control_training.validate_training_protocol(
        parent, data_config, training_config, loss_arm, gate_config)
    job_id = os.environ["PBS_JOBID"]
    root, unit_path = Path(parent["remote_root"]), Path(parent["remote_root"]) / "results" / unit
    unit_path.mkdir(parents=True, exist_ok=True)
    previous = _archive_summary(unit_path, job_id)
    state = _new_state(parent, unit, arm, shard, job_id, training_config, data_config)
    if previous is not None:
        if any(previous.get(key) != state.get(key) for key in (
                "protocol", "unit", "arm", "training_protocol", "data_config")):
            raise ValueError("Same-PBS final-C80 continuation has a different frozen binding")
        for key in (
                "startup_query_attempted", "startup_query_completed",
                "endpoint_query_attempted", "endpoint_query_completed",
                "reader_pass_attempted", "reader_pass_completed",
                "reader_anchor_passes_attempted", "reader_anchor_passes_completed",
                "calibration_source_reuse_endpoints", "existing_pair_reuse"):
            state[key] = int(previous.get(key, 0))
    _publish(unit_path, state)

    switching = None
    try:
        receipts = {
            "training_pairs": completed(root, "training_pairs"),
            "training_teacher_plan": completed(root, "training_teacher_plan"),
            "training_teacher": completed(root, "training_teacher"),
            "training_loss_selfcheck": completed(root, "training_loss_selfcheck"),
            "aggregate_c80_parameter_gate": completed(root, "aggregate_c80_parameter_gate"),
            "training_runtime_selfcheck": completed(root, "training_runtime_selfcheck"),
            "calibration_0": completed(root, "calibration_0"),
            "calibration_1": completed(root, "calibration_1"),
        }
        control_training.validate_prerequisite_receipts(
            parent, training_config, gate_config,
            {key: receipts[key] for key in (
                "training_teacher", "training_loss_selfcheck",
                "aggregate_c80_parameter_gate")})
        runtime_check = receipts["training_runtime_selfcheck"]
        if (runtime_check.get("protocol") != parent["protocol"]
                or "test_control_training.py" not in runtime_check.get("passed_checks", [])
                or runtime_check.get("model_queries") != 0
                or runtime_check.get("training_updates") != 0):
            raise ValueError("Training runtime selfcheck receipt differs")
        if (receipts["training_teacher"].get("source_jobs", {}).get("training_pairs")
                != receipts["training_pairs"].get("pbs_jobid")
                or receipts["training_teacher"].get("source_jobs", {}).get("training_teacher_plan")
                != receipts["training_teacher_plan"].get("pbs_jobid")):
            raise ValueError("Teacher source pair/plan PBS bindings differ")
        cal_by_index = {0: receipts["calibration_0"], 1: receipts["calibration_1"]}
        for index in (0, 1):
            if (receipts["training_teacher"].get("source_jobs", {}).get(f"calibration_{index}")
                    != cal_by_index[index].get("pbs_jobid")):
                raise ValueError(f"Teacher calibration source PBS differs for shard {index}")

        from training_teacher import _validate_training_definition
        definition, cases, plan_by_id, calibration_jobs, _ = _validate_training_definition(
            root, parent, data_config, receipts["training_pairs"],
            receipts["training_teacher_plan"], cal_by_index, np)
        shard_cases = [cases[case_id] for case_id in case_ids_for_shard(cases, shard)]
        del definition
        stats = read(Path(parent["source_relational_protocol"]["source_dual_root"])
                     / "results" / "fit_stats" / "summary.json")
        if stats.get("complete") is not True or stats.get("heldout_used") is not False:
            raise ValueError("Frozen calibration-only dual-KL loss statistics are required")

        state["source_jobs"] = _source_jobs(
            receipts["training_pairs"], receipts["training_teacher_plan"],
            receipts["training_teacher"], cal_by_index,
            receipts["training_loss_selfcheck"],
            receipts["aggregate_c80_parameter_gate"], runtime_check)
        state["teacher_job"] = receipts["training_teacher"]["pbs_jobid"]
        state["planned_fresh_endpoint_outputs"] = (
            80 if arm != "mse_reference" else sum(
                int(teacher._source_eligibility(plan_by_id[int(case["case_id"])])
                    ["new_noisy_teacher_query_required"]) for case in shard_cases))

        switching = SwitchingRuntime(parent, unit_path)
        runtime = switching.runtime
        if arm == "mse_reference":
            switching.activate("per_channel_clip")
            deployment_binding = {
                "label": "original_learned_per_channel_clip_mse_reference",
                "evaluation_arm": arm,
                "source_deployment_binding": dict(
                    switching.deployment_bindings["per_channel_clip"]),
                "training_protocol": training_config, "data_config": data_config,
                "source_jobs": state["source_jobs"],
            }
        else:
            state["startup_query_attempted"] = 0
            state["startup_query_completed"] = 0
            _publish(unit_path, state)
            deployment_binding = training_evaluation.prepare_runtime(
                switching, parent, training_config, arm, unit_path, state)
            state["startup_query_completed"] += 2
            _publish(unit_path, state)
        state["deployment_binding"] = deployment_binding
        targets, case_metadata, teacher_bindings = control_training._metadata_teacher_bindings(
            torch, np, switching, parent, root, data_config, cases, plan_by_id,
            receipts["training_teacher_plan"]["pbs_jobid"],
            {str(index): receipt["pbs_jobid"] for index, receipt in cal_by_index.items()})
        if len(targets) != 160 or len(teacher_bindings) != 160:
            raise RuntimeError("C80 evaluation must consume all 160 cached BF16 teacher views")
        reader = BF16Reader(runtime, switching.original_weights)
        state["reader_original_weights"] = "immutable BF16 masters"
        if any(parameter.requires_grad for parameter in runtime.model.parameters()):
            raise RuntimeError("Reader/model parameters must remain frozen")
        state["deployment_binding"] = deployment_binding
        state["completed_case_ids"] = []
        state["remaining_case_ids"] = [int(case["case_id"]) for case in shard_cases]
        _publish(unit_path, state)
        records_dir = unit_path / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        permitted = {f"{int(case['case_id'])}.pt" for case in shard_cases}
        extras = {path.name for path in records_dir.glob("*.pt")} - permitted
        if extras:
            raise ValueError(f"Final C80 directory contains out-of-shard records: {sorted(extras)}")
        deadline = Deadline(6600)

        # Every PBS segment repeats the two-view native teacher-future reader anchor.
        first_case = shard_cases[0]
        first_id = int(first_case["case_id"])
        first_datums, first_ids, first_meta = _load_case(
            torch, np, switching, teacher, root, data_config, first_case)
        if first_ids != case_metadata[first_id]["identities"]:
            raise ValueError("First C80 input identities differ from teacher bindings")
        _set_case_description(runtime, first_meta)
        for variant in VARIANTS:
            target = targets[(first_id, variant)]
            state["reader_anchor_passes_attempted"] += 1
            action, _ = _reader_pass(
                torch, reader, runtime, first_datums[variant], int(first_meta["sampler_seed"]),
                target["future"], target["action_initial_noise"], deadline, state,
                lambda: _publish(unit_path, state), control_training,
                f"native teacher anchor {first_id}/{variant}")
            state["reader_anchor_passes_completed"] += 1
            _publish(unit_path, state)
            if not torch.equal(action, target["action"].detach().float().cpu()):
                raise RuntimeError(f"Native BF16 reader anchor differs for {first_id}/{variant}")
        state["reader_native_anchor_performed"] = True
        _publish(unit_path, state)
        del first_datums

        for case in shard_cases:
            case_id = int(case["case_id"])
            path = records_dir / f"{case_id}.pt"
            datums, identities, metadata = _load_case(
                torch, np, switching, teacher, root, data_config, case)
            if identities != case_metadata[case_id]["identities"]:
                raise ValueError(f"Training input identity differs from teacher cache for {case_id}")
            if (int(metadata["sampler_seed"]) != int(case_metadata[case_id]["sampler_seed"])
                    or metadata["description"] != case_metadata[case_id]["description"]):
                raise ValueError(f"Training seed/description differs for case {case_id}")
            _set_case_description(runtime, metadata)
            binding = {
                "protocol": parent["protocol"], "unit": unit, "arm": arm,
                "index": case_id, "case_id": case_id, "suite": case.get("suite"),
                "deployment_binding": deployment_binding,
                "teacher_bindings": {
                    variant: teacher_bindings[f"{case_id}:{variant}"] for variant in VARIANTS},
                "input_identities": identities,
                "sampler_seed": int(metadata["sampler_seed"]),
                "description": metadata["description"],
            }
            if path.is_file():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                validate_record(torch, saved, binding)
                for variant in VARIANTS:
                    teacher._noise_equal(
                        torch, saved["endpoints"][variant]["action_initial_noise"],
                        targets[(case_id, variant)]["action_initial_noise"])
                state["existing_pair_reuse"] += 1
                state["completed_case_ids"].append(case_id)
                state["remaining_case_ids"].remove(case_id)
                state["all_initial_noise_parity_valid"] = True
                _publish(unit_path, state)
                continue
            if deadline.near(300):
                raise SoftStop(f"deadline before C80 pair {case_id}")
            endpoints = {}
            for variant in VARIANTS:
                target = targets[(case_id, variant)]
                source = None
                should_reuse_mse = arm == "mse_reference" and (
                    variant == "clean" or teacher._source_eligibility(
                        plan_by_id[case_id])["noisy_teacher_reuse_eligible"])
                if should_reuse_mse:
                    endpoint, source = _cached_mse_endpoint(
                        torch, teacher, calibration_worker, root, parent,
                        {0: receipts["calibration_0"]["pbs_jobid"],
                         1: receipts["calibration_1"]["pbs_jobid"]}, case, variant,
                        identities[variant], metadata["description"])
                    state["calibration_source_reuse_endpoints"] += 1
                else:
                    if deadline.near(300):
                        raise SoftStop(f"deadline before native endpoint {case_id}/{variant}")
                    state["endpoint_query_attempted"] += 1
                    _publish(unit_path, state)
                    runtime.description = metadata["description"]
                    action, future, counts, noise = switching.query(
                        datums[variant], int(metadata["sampler_seed"]), metadata["description"])
                    endpoint = teacher._validated_outputs(torch, {
                        "action": action, "future": future,
                        "counts": counts, "action_initial_noise": noise,
                    }, f"Final C80 endpoint {case_id}/{variant}")
                    state["endpoint_query_completed"] += 1
                    _publish(unit_path, state)
                    source = {"kind": "new_endpoint_query", "producer_job": job_id}
                teacher._noise_equal(
                    torch, endpoint["action_initial_noise"], target["action_initial_noise"])
                endpoints[variant] = {**endpoint, "source": source}
            teacher._noise_equal(torch, endpoints["clean"]["action_initial_noise"],
                                 endpoints["noisy"]["action_initial_noise"])

            reader_actions, reader_counts = {}, {}
            for variant in VARIANTS:
                reader_actions[variant], reader_counts[variant] = _reader_pass(
                    torch, reader, runtime, datums[variant], int(metadata["sampler_seed"]),
                    endpoints[variant]["future"], endpoints[variant]["action_initial_noise"],
                    deadline, state, lambda: _publish(unit_path, state), control_training,
                    f"student future {case_id}/{variant}")
            raw_terms = compute_raw_terms(
                torch, endpoints, reader_actions,
                {variant: targets[(case_id, variant)] for variant in VARIANTS},
                stats, loss_config)
            record = {
                **binding, "producer_job": job_id, "complete": True, "status": "complete",
                "endpoints": endpoints, "reader_actions": reader_actions,
                "reader_counts": reader_counts, "raw_terms": raw_terms,
            }
            validate_record(torch, record, binding)
            _atomic_torch_save(torch, path, record)
            state["completed_case_ids"].append(case_id)
            state["completed_case_ids"].sort()
            state["remaining_case_ids"].remove(case_id)
            state["all_initial_noise_parity_valid"] = True
            _publish(unit_path, state, summary=True)
            del datums, endpoints, reader_actions, reader_counts

        expected = [int(case["case_id"]) for case in shard_cases]
        if (state["completed_case_ids"] != expected or len(expected) != 40
                or state["completed_endpoints"] != 80
                or state["reader_native_anchor_performed"] is not True
                or any(parameter.requires_grad for parameter in runtime.model.parameters())):
            raise RuntimeError("Final C80 shard completion, reader anchor, or frozen-model gate failed")
        state.update(status="complete", complete=True, checkpointed=False,
                     expected_pairs=40, completed_pairs=40,
                     expected_endpoints=80, completed_endpoints=80,
                     remaining_case_ids=[], all_initial_noise_parity_valid=True,
                     teacher_bindings_complete=True)
        _publish(unit_path, state, summary=True)
        return 0
    except SoftStop as exc:
        state.update(status="checkpointed", complete=False, checkpointed=True,
                     stop_reason=str(exc))
        _publish(unit_path, state, summary=True)
        return 0
    except Exception as exc:
        state.update(status="failed", complete=False, checkpointed=False,
                     failure=f"{type(exc).__name__}: {exc}")
        _publish(unit_path, state, summary=True)
        raise
    finally:
        if switching is not None:
            switching.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args(argv)
    return run(args.unit)


if __name__ == "__main__":
    raise SystemExit(main())
