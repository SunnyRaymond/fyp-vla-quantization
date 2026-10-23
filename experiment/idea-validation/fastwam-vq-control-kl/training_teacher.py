"""Prepare frozen BF16 action/future labels for the training-only C80 pool."""
import argparse
import os
from pathlib import Path

from control_common import Deadline, guard, read, save


UNIT = "training_teacher"
CHILD_PROTOCOL = "control-kl-training-data-v1"
CASE_IDS = list(range(2000, 2080))
VARIANTS = ("clean", "noisy")
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)
NOISE_SHAPE = (1, 32, 7)
COUNTS = {"video_steps": 10, "action_steps": 10}
MAX_NEW_QUERIES = 48
EXPECTED_RECORDS = 160
RESERVE_SECONDS = 180


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _same_arrays(np, left, right):
    return (set(left) == set(right)
            and all(left[key].dtype == right[key].dtype
                    and left[key].shape == right[key].shape
                    and np.array_equal(left[key], right[key]) for key in left))


def _load_npz(np, path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _source_eligibility(plan_record):
    clean = plan_record.get("clean_teacher_reuse_eligible") is True
    noisy_equal = plan_record.get("noisy_observations_exactly_equal") is True
    noisy_reuse = plan_record.get("noisy_teacher_reuse_eligible") is True
    new_noisy = plan_record.get("new_noisy_teacher_query_required") is True
    if not clean or noisy_reuse != noisy_equal or new_noisy == noisy_reuse:
        raise ValueError(f"Invalid teacher reuse eligibility for {plan_record.get('case_id')}")
    if not plan_record.get("clean_teacher_source"):
        raise ValueError(f"Clean teacher source is missing for {plan_record.get('case_id')}")
    if noisy_reuse != bool(plan_record.get("noisy_teacher_source")):
        raise ValueError(f"Noisy teacher source differs from eligibility for {plan_record.get('case_id')}")
    return {
        "clean_teacher_reuse_eligible": clean,
        "noisy_observations_exactly_equal": noisy_equal,
        "noisy_teacher_reuse_eligible": noisy_reuse,
        "new_noisy_teacher_query_required": new_noisy,
    }


def _reuse_source(plan_record, variant):
    eligibility = _source_eligibility(plan_record)
    if variant == "clean":
        return plan_record["clean_teacher_source"], eligibility
    if variant != "noisy":
        raise ValueError(f"Unknown training variant: {variant}")
    if eligibility["noisy_teacher_reuse_eligible"]:
        return plan_record["noisy_teacher_source"], eligibility
    if eligibility["new_noisy_teacher_query_required"]:
        return None, eligibility
    raise ValueError(f"Noisy teacher source is neither reusable nor queryable for {plan_record['case_id']}")


def _binding(protocol, case_id, variant, seed, description, identity):
    return {
        "protocol": protocol,
        "unit": UNIT,
        "index": int(case_id),
        "arm": "bf16",
        "variant": variant,
        "seed": int(seed),
        "description": description,
        "input_identity": identity,
    }


def _float_tensor(torch, value, shape, label):
    tensor = torch.as_tensor(value).detach().float().cpu()
    if (tuple(tensor.shape) != tuple(shape)
            or not bool(torch.isfinite(tensor).all())):
        raise ValueError(f"{label} has an invalid shape or non-finite values")
    return tensor


def _validated_outputs(torch, record, label):
    if not isinstance(record, dict) or record.get("counts") != COUNTS:
        raise ValueError(f"{label} has invalid 10+10 sampler counts")
    return {
        "action": _float_tensor(torch, record.get("action"), ACTION_SHAPE, label + " action"),
        "future": _float_tensor(torch, record.get("future"), FUTURE_SHAPE, label + " future"),
        "action_initial_noise": _float_tensor(
            torch, record.get("action_initial_noise"), NOISE_SHAPE, label + " initial noise"),
        "counts": dict(COUNTS),
    }


def _validate_bound_record(torch, payload, metadata, binding):
    from diagnostic_runtime import validate_binding

    validate_binding(payload, binding)
    validate_binding(metadata, binding)
    if metadata.get("counts") != COUNTS:
        raise ValueError("Teacher record metadata counts differ")
    outputs = _validated_outputs(torch, payload, "Teacher record")
    for field, tensor in (("action_shape", outputs["action"]),
                          ("future_shape", outputs["future"]),
                          ("action_initial_noise_shape", outputs["action_initial_noise"])):
        if metadata.get(field) != list(tensor.shape):
            raise ValueError(f"Teacher record metadata {field} differs")
    return outputs


def _noise_equal(torch, clean_noise, noisy_noise):
    if (tuple(clean_noise.shape) != NOISE_SHAPE
            or tuple(noisy_noise.shape) != NOISE_SHAPE
            or not bool(torch.isfinite(clean_noise).all())
            or not bool(torch.isfinite(noisy_noise).all())
            or not torch.equal(clean_noise, noisy_noise)):
        raise RuntimeError("Clean/noisy training views do not have exactly equal initial action noise")
    return True


def _record_paths(out, case_id, variant):
    stem = f"{int(case_id)}_bf16_{variant}"
    return out / "records" / f"{stem}.pt", out / "records" / f"{stem}.json"


def _save_torch_atomic(torch, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _read_source_record(torch, path, binding):
    tensor_path = Path(path)
    json_path = tensor_path.with_suffix(".json")
    if not tensor_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Calibration teacher source pair is incomplete: {tensor_path}")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    metadata = read(json_path)
    return _validate_bound_record(torch, payload, metadata, binding), metadata


def _validate_training_definition(root, p, config, pairs_summary, plan_summary,
                                  calibration_receipts, np):
    if (p.get("protocol") != "fastwam-vq-control-kl-v1"
            or config.get("protocol") != CHILD_PROTOCOL
            or config.get("parent_protocol") != p.get("protocol")
            or config.get("case_ids") != CASE_IDS
            or config.get("dataset_subdir") != "data/training_pairs"):
        raise ValueError("Frozen training child/parent protocol differs")
    if (pairs_summary.get("protocol") != p["protocol"]
            or pairs_summary.get("unit") != "training_pairs"
            or pairs_summary.get("config") != config
            or pairs_summary.get("contexts") != 80
            or pairs_summary.get("training_started") is not False
            or pairs_summary.get("training_updates") != 0):
        raise ValueError("Completed training-pairs receipt differs from the frozen protocol")
    if (plan_summary.get("protocol") != p["protocol"]
            or plan_summary.get("unit") != "training_teacher_plan"
            or plan_summary.get("complete") is not True
            or plan_summary.get("training_data_protocol") != config["protocol"]
            or plan_summary.get("contexts") != 80
            or plan_summary.get("source_training_pairs_job") != pairs_summary.get("pbs_jobid")
            or plan_summary.get("eligible_clean_reuses") != 80
            or plan_summary.get("eligible_noisy_reuses") != 32
            or plan_summary.get("planned_new_teacher_queries") != 48
            or plan_summary.get("new_model_queries") != 0
            or plan_summary.get("training_updates") != 0):
        raise ValueError("Completed training-teacher plan differs from the frozen 80-context plan")
    definition = read(root / config["dataset_subdir"] / "definition.json")
    if (definition.get("protocol") != config["protocol"]
            or definition.get("parent_protocol") != p["protocol"]
            or definition.get("config") != config
            or definition.get("parent_config") != p
            or definition.get("training_use") is not True
            or sorted(int(row["case_id"]) for row in definition.get("cases", [])) != CASE_IDS):
        raise ValueError("Training-pairs definition differs from the frozen child configuration")
    diagnostic = read(root / "data/calibration_pairs/definition.json")
    calibration_cases = {int(row["case_id"]): row for row in diagnostic.get("cases", [])}
    if len(calibration_cases) != 80 or set(calibration_cases) != set(CASE_IDS):
        raise ValueError("Calibration source definition does not cover C80 exactly")
    plan_records = plan_summary.get("records", [])
    plan_by_id = {int(row["case_id"]): row for row in plan_records}
    if len(plan_by_id) != 80 or set(plan_by_id) != set(CASE_IDS):
        raise ValueError("Teacher plan records do not cover C80 exactly")
    source_jobs = {f"calibration_{shard}": calibration_receipts[shard]["pbs_jobid"]
                   for shard in (0, 1)}
    for shard, receipt in calibration_receipts.items():
        if (receipt.get("protocol") != p["protocol"]
                or receipt.get("unit") != f"calibration_{shard}"
                or receipt.get("complete") is not True
                or not receipt.get("pbs_jobid")):
            raise ValueError(f"Calibration {shard} completed receipt binding differs")
    if len(set(source_jobs.values())) != 2:
        raise ValueError("Calibration shards must have distinct PBS source receipts")

    cases = {int(row["case_id"]): row for row in definition["cases"]}
    new_queries = noisy_reuses = clean_reuses = 0
    for case_id in CASE_IDS:
        case, planned = cases[case_id], plan_by_id[case_id]
        shard = int(case["shard"])
        old_case = calibration_cases[case_id]
        if (planned.get("shard") != shard
                or planned.get("source_calibration_pbs_jobid") != source_jobs[f"calibration_{shard}"]
                or case.get("row") != old_case.get("row")
                or case.get("shard") != old_case.get("shard")
                or case.get("clean_observation") != old_case.get("clean_observation")
                or case.get("clean_metadata") != old_case.get("clean_metadata")
                or planned.get("training_metadata") != str(
                    root / config["dataset_subdir"] / str(case_id) / "metadata.json")):
            raise ValueError(f"Training/calibration source plan differs for case {case_id}")
        eligibility = _source_eligibility(planned)
        clean_reuses += 1
        noisy_reuses += int(eligibility["noisy_teacher_reuse_eligible"])
        new_queries += int(eligibility["new_noisy_teacher_query_required"])
        clean_path = root / "results" / f"calibration_{shard}" / "records" / f"{case_id}_bf16_clean.pt"
        noisy_path = root / "results" / f"calibration_{shard}" / "records" / f"{case_id}_bf16_noisy.pt"
        if Path(planned["clean_teacher_source"]).resolve() != clean_path.resolve():
            raise ValueError(f"Clean teacher source path differs for case {case_id}")
        if eligibility["noisy_teacher_reuse_eligible"]:
            if Path(planned["noisy_teacher_source"]).resolve() != noisy_path.resolve():
                raise ValueError(f"Noisy teacher source path differs for case {case_id}")
        elif planned.get("noisy_teacher_source") is not None:
            raise ValueError(f"Ineligible noisy source must not be named for case {case_id}")

        training_meta_path = root / config["dataset_subdir"] / str(case_id) / "metadata.json"
        training_meta = read(training_meta_path)
        source_meta = read(_resolve(root, case["clean_metadata"]))
        calibration_meta = read(root / "data/calibration_pairs" / str(case_id) / "metadata.json")
        sampler_index = int(config["sampler_seed_index"])
        for meta in (source_meta, calibration_meta, training_meta):
            context = meta if "sampler_seeds" in meta else meta.get("row", {})
            if (meta.get("case_id") != case_id
                    or context.get("demo_id") != case["row"].get("demo_id")
                    or context.get("frame_index") != case["row"].get("frame_index")
                    or context.get("sampler_seeds") != case["row"].get("sampler_seeds")
                    or meta.get("description") != source_meta.get("description")):
                raise ValueError(f"Clean source case/demo/frame/sampler metadata differs for {case_id}")
        if (calibration_meta.get("training_use") is not False
                or training_meta.get("training_use") is not True):
            raise ValueError(f"Training-use metadata differs for case {case_id}")
        if (training_meta.get("protocol") != config["protocol"]
                or training_meta.get("parent_protocol") != p["protocol"]
                or training_meta.get("row") != case["row"]
                or training_meta.get("description") != source_meta.get("description")
                or training_meta.get("sampler_seed") != case.get("sampler_seed")
                or training_meta.get("sampler_seed") != case["row"]["sampler_seeds"][sampler_index]
                or training_meta.get("sampler_seed_index") != sampler_index
                or training_meta.get("training_use") is not True
                or calibration_meta.get("row") != case["row"]
                or calibration_meta.get("sampler_seed") != training_meta["sampler_seed"]
                or calibration_meta.get("description") != training_meta["description"]):
            raise ValueError(f"Training metadata child/parent/case/seed binding differs for {case_id}")

        clean_arrays = _load_npz(np, _resolve(root, case["clean_observation"]))
        noisy_arrays = _load_npz(np, _resolve(root, training_meta["noisy_observation"]))
        if "agentview_image" not in clean_arrays or not _same_arrays_except_agentview(
                np, clean_arrays, noisy_arrays):
            raise ValueError(f"Non-agentview training arrays changed for case {case_id}")
        cal_noisy = _load_npz(np, _resolve(root, calibration_meta["noisy_observation"]))
        noisy_equal = _same_arrays(np, noisy_arrays, cal_noisy)
        if (noisy_equal is not eligibility["noisy_observations_exactly_equal"]
                or noisy_equal is not eligibility["noisy_teacher_reuse_eligible"]):
            raise ValueError(f"Noisy teacher reuse eligibility drifted for case {case_id}")

    if (clean_reuses != 80 or noisy_reuses != 32 or new_queries != 48
            or new_queries > MAX_NEW_QUERIES):
        raise ValueError("Teacher reuse/query totals differ from the frozen 80+32+48 plan")
    return definition, cases, plan_by_id, source_jobs, new_queries


def _same_arrays_except_agentview(np, left, right):
    keys = set(left) | set(right)
    return ("agentview_image" in keys and set(left) == set(right)
            and all(left[key].dtype == right[key].dtype
                    and left[key].shape == right[key].shape
                    and np.array_equal(left[key], right[key])
                    for key in keys if key != "agentview_image"))


def _progress_binding(p, config, teacher_plan_job):
    return {"protocol": p["protocol"], "unit": UNIT, "data_config": config,
            "teacher_plan_pbs_jobid": teacher_plan_job}


def _load_progress(path, binding):
    if not path.is_file():
        return {**binding, "query_attempts": [], "completed_keys": [],
                "new_query_completed_total": 0, "reuse_calibration_total": 0}
    previous = read(path)
    if any(previous.get(key) != value for key, value in binding.items()):
        raise ValueError("Existing training-teacher progress belongs to a different frozen plan")
    if "query_attempts" not in previous and isinstance(previous.get("attempted_query_keys"), list):
        prior_summary = read(path.with_name("summary.json"))
        prior_job = prior_summary.get("pbs_jobid")
        if not prior_job:
            raise ValueError("Legacy query attempts have no producer PBS job binding")
        previous["query_attempts"] = [
            {"key": str(key), "pbs_jobid": prior_job, "status": "attempted"}
            for key in previous.pop("attempted_query_keys")
        ]
    attempts = previous.get("query_attempts")
    completed = previous.get("completed_keys")
    if (not isinstance(attempts, list)
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("key"), str)
                   or not isinstance(item.get("pbs_jobid"), str)
                   or item.get("status") not in ("attempted", "completed",
                                                  "retried_after_terminal_repair")
                   for item in attempts)
            or not isinstance(completed, list) or len(completed) != len(set(completed))
            or not isinstance(previous.get("new_query_completed_total"), int)
            or not isinstance(previous.get("reuse_calibration_total"), int)):
        raise ValueError("Existing training-teacher progress counters are malformed")
    return previous


def _start_query_attempt(progress, key, pbs_jobid, planned_queries, repair_of=None):
    attempts = progress["query_attempts"]
    previous = [item for item in attempts if item["key"] == key]
    if previous:
        latest = previous[-1]
        if (latest["status"] != "attempted"
                or latest["pbs_jobid"] == pbs_jobid
                or repair_of != latest["pbs_jobid"]):
            raise RuntimeError(
                f"Unresolved query needs a verified terminal repair of {latest['pbs_jobid']}: {key}")
        latest["status"] = "retried_after_terminal_repair"
    elif len({item["key"] for item in attempts}) >= planned_queries:
        raise RuntimeError("Teacher query keys would exceed the frozen plan")
    attempt = {"key": key, "pbs_jobid": pbs_jobid, "status": "attempted"}
    attempts.append(attempt)
    return attempt


def _complete_query_attempt(progress, key, producer_job):
    matches = [item for item in progress["query_attempts"]
               if item["key"] == key and item["pbs_jobid"] == producer_job]
    if not matches or matches[-1]["status"] not in ("attempted", "completed"):
        raise RuntimeError(f"Saved query output has no matching attempt record: {key}")
    matches[-1]["status"] = "completed"


def _persist(out, summary, progress):
    save(out / "summary.json", summary)
    save(out / "progress.json", progress)


def _new_summary(protocol, pbs_jobid):
    return {
        "protocol": protocol, "unit": UNIT, "pbs_jobid": pbs_jobid,
        "complete": False, "checkpointed": False, "status": "running",
        "expected_records": EXPECTED_RECORDS, "completed_records": 0,
        "contexts": 80, "planned_new_teacher_queries": None,
        "new_query_attempted": 0, "new_query_completed": 0,
        "total_new_query_attempted": 0, "new_query_retry_attempts": 0,
        "total_new_query_completed": 0,
        "successful_new_teacher_queries": 0, "unique_new_teacher_outputs": 0,
        "new_teacher_queries_scope": "BF16 query attempts, including failed/retried attempts",
        "reuse_calibration": 0, "reuse_existing": 0,
        "training_updates": 0, "saved_trained_checkpoints": 0,
        "heldout_inputs_used": False, "evaluation_trajectory_inputs_used": False,
        "source_jobs": {}, "initial_noise_parity_valid": False,
        "initial_noise_parity_contexts": 0, "frozen_model_unchanged": None,
    }


def _sync_counts(summary, progress, segment, records_seen, planned_queries):
    attempts = progress["query_attempts"]
    unique_attempts = len({item["key"] for item in attempts})
    summary.update({
        "planned_new_teacher_queries": planned_queries,
        "completed_records": records_seen,
        "new_query_attempted": segment["new_query_attempted"],
        "new_query_completed": segment["new_query_completed"],
        "total_new_query_attempted": len(attempts),
        "new_query_retry_attempts": len(attempts) - unique_attempts,
        "total_new_query_completed": progress["new_query_completed_total"],
        "reuse_calibration": segment["reuse_calibration"],
        "reuse_existing": segment["reuse_existing"],
        "reuse_calibration_total": progress["reuse_calibration_total"],
        # These legacy counters report actual query attempts, including failed
        # attempts and terminal-repair retries. Planned unique queries and
        # saved outputs are reported independently below.
        "new_model_queries": len(attempts),
        "new_teacher_queries": len(attempts),
        "successful_new_teacher_queries": sum(
            item["status"] == "completed" for item in attempts),
        "unique_new_teacher_outputs": progress["new_query_completed_total"],
    })


def _validate_training_record(torch, payload, metadata, binding, config,
                              teacher_plan_job, source_calibration_job, source_file,
                              eligibility):
    outputs = _validate_bound_record(torch, payload, metadata, binding)
    expected = {
        "data_config": config,
        "teacher_plan_pbs_jobid": teacher_plan_job,
        "source_calibration_job": source_calibration_job,
        "source_file": str(source_file) if source_file is not None else None,
        "source_eligibility": eligibility,
    }
    for field, value in expected.items():
        if payload.get(field) != value or metadata.get(field) != value:
            raise ValueError(f"Existing training teacher record {field} differs")
    source_kind = "calibration_reuse" if source_file is not None else "new_query"
    if payload.get("source_kind") != source_kind or metadata.get("source_kind") != source_kind:
        raise ValueError("Existing training teacher source kind differs")
    producer_job = payload.get("producer_job")
    if (not isinstance(producer_job, str) or not producer_job
            or metadata.get("producer_job") != producer_job):
        raise ValueError("Existing training teacher producer PBS job differs")
    return outputs


def _all_frozen(switching):
    params = list(switching.runtime.model.parameters())
    for collection in (switching.banks.books, switching.banks.logs,
                       switching.banks.ratios, switching.banks.shadows):
        params.extend(collection.values())
    return all(not parameter.requires_grad for parameter in params)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    guard()
    import numpy as np
    import torch
    from aggregate import completed
    from diagnostic_runtime import SwitchingRuntime, datum_identity
    from parameter_gate import frozen_snapshot

    here = Path(__file__).resolve().parent
    p = read(here / "protocol.json")
    config = read(here / "training_data_protocol.json")
    root = Path(p["remote_root"])
    out = root / "results" / UNIT
    out.mkdir(parents=True, exist_ok=True)
    summary = _new_summary(p.get("protocol"), os.environ.get("PBS_JOBID"))
    progress = {"protocol": p.get("protocol"), "unit": UNIT,
                "data_config": config, "teacher_plan_pbs_jobid": None,
                "query_attempts": [], "completed_keys": [],
                "new_query_completed_total": 0, "reuse_calibration_total": 0}
    segment = {"new_query_attempted": 0, "new_query_completed": 0,
               "reuse_calibration": 0, "reuse_existing": 0}
    records = {}
    noise_parity_cases = set()
    switching = None
    snapshot_before = None
    frozen_unchanged = None
    stop_for_deadline = False
    failure = None
    planned_queries = None
    progress_ready = False
    try:
        plan_summary = completed(root, "training_teacher_plan")
        pairs_summary = completed(root, "training_pairs")
        calibration_receipts = {
            shard: completed(root, f"calibration_{shard}") for shard in (0, 1)
        }
        if (plan_summary.get("pbs_jobid") is None
                or pairs_summary.get("pbs_jobid") is None
                or pairs_summary.get("protocol") != p.get("protocol")
                or pairs_summary.get("unit") != "training_pairs"
                or plan_summary.get("protocol") != p.get("protocol")
                or plan_summary.get("unit") != "training_teacher_plan"):
            raise ValueError("Training-teacher prerequisite receipt binding differs")
        definition, cases, plan_by_id, source_jobs, planned_queries = _validate_training_definition(
            root, p, config, pairs_summary, plan_summary, calibration_receipts, np)
        source_jobs.update({"training_pairs": pairs_summary["pbs_jobid"],
                            "training_teacher_plan": plan_summary["pbs_jobid"]})
        summary["source_jobs"] = source_jobs
        summary["source_calibration_jobs"] = {
            str(shard): calibration_receipts[shard]["pbs_jobid"] for shard in (0, 1)
        }
        summary["planned_new_teacher_queries"] = planned_queries
        progress_binding = _progress_binding(p, config, plan_summary["pbs_jobid"])
        progress = _load_progress(out / "progress.json", progress_binding)
        progress_ready = True
        progress["completed_keys"] = [str(value) for value in progress["completed_keys"]]
        _persist(out, summary, progress)

        switching = SwitchingRuntime(p, out)
        switching.activate("bf16")
        if not _all_frozen(switching):
            raise RuntimeError("A model or bank parameter is trainable before teacher queries")
        snapshot_before = frozen_snapshot(switching, [])
        deadline = Deadline(6600)
        current_job = os.environ["PBS_JOBID"]

        for case_id in CASE_IDS:
            case, planned = cases[case_id], plan_by_id[case_id]
            meta_path = root / config["dataset_subdir"] / str(case_id) / "metadata.json"
            training_meta = read(meta_path)
            clean_arrays = _load_npz(np, _resolve(root, case["clean_observation"]))
            noisy_arrays = _load_npz(np, _resolve(root, training_meta["noisy_observation"]))
            datums = {
                "clean": switching.datum(clean_arrays),
                "noisy": switching.datum(noisy_arrays),
            }
            identities = {variant: datum_identity(torch, datums[variant])
                          for variant in VARIANTS}
            if identities["clean"] == identities["noisy"]:
                raise ValueError(f"Clean/noisy model inputs are identical for case {case_id}")
            clean_noise = None
            eligibility = _source_eligibility(planned)
            source_job = planned["source_calibration_pbs_jobid"]
            source_shard = int(planned["shard"])
            if source_job != calibration_receipts[source_shard]["pbs_jobid"]:
                raise ValueError(f"Calibration source job changed for case {case_id}")

            for variant in VARIANTS:
                query_key = f"{case_id}:noisy" if variant == "noisy" else None
                description = training_meta["description"]
                binding = _binding(p["protocol"], case_id, variant,
                                   case["sampler_seed"], description,
                                   identities[variant])
                source_file, eligibility = _reuse_source(planned, variant)
                source_file = _resolve(root, source_file) if source_file is not None else None
                tensor_path, json_path = _record_paths(out, case_id, variant)
                if tensor_path.is_file() != json_path.is_file():
                    raise RuntimeError(f"Partial training-teacher record exists: {tensor_path.stem}")
                existed = tensor_path.is_file()
                if existed:
                    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
                    metadata = read(json_path)
                    outputs = _validate_training_record(
                        torch, payload, metadata, binding, config,
                        plan_summary["pbs_jobid"], source_job, source_file, eligibility)
                    segment["reuse_existing"] += 1
                    source_kind = payload["source_kind"]
                    if source_kind == "new_query":
                        _complete_query_attempt(progress, query_key, payload["producer_job"])
                elif source_file is not None:
                    source_binding = _binding(p["protocol"], case_id, variant,
                                              case["sampler_seed"], description,
                                              identities[variant])
                    source_binding["unit"] = f"calibration_{source_shard}"
                    outputs, _ = _read_source_record(torch, source_file, source_binding)
                    source_kind = "calibration_reuse"
                    segment["reuse_calibration"] += 1
                else:
                    if deadline.near(RESERVE_SECONDS):
                        stop_for_deadline = True
                        break
                    switching.activate("bf16")
                    _start_query_attempt(
                        progress, query_key, current_job, planned_queries,
                        os.environ.get("FW_CONTROL_REPAIR_OF"))
                    segment["new_query_attempted"] += 1
                    _sync_counts(summary, progress, segment, len(records), planned_queries)
                    _persist(out, summary, progress)
                    action, future, counts, noise = switching.query(
                        datums[variant], case["sampler_seed"], description)
                    outputs = _validated_outputs(torch, {
                        "action": action, "future": future,
                        "action_initial_noise": noise, "counts": counts,
                    }, f"New BF16 query {case_id}/{variant}")
                    source_kind = "new_query"

                if variant == "clean":
                    clean_noise = outputs["action_initial_noise"]
                elif clean_noise is None:
                    raise RuntimeError(f"No clean teacher noise was validated for case {case_id}")
                else:
                    _noise_equal(torch, clean_noise, outputs["action_initial_noise"])

                if not existed:
                    extras = {
                        "data_config": config,
                        "teacher_plan_pbs_jobid": plan_summary["pbs_jobid"],
                        "source_calibration_job": source_job,
                        "producer_job": current_job,
                        "source_file": str(source_file) if source_file is not None else None,
                        "source_eligibility": eligibility,
                        "source_kind": source_kind,
                    }
                    payload = {**binding, **extras, **outputs}
                    metadata = {
                        **binding, **extras, "counts": dict(COUNTS),
                        "action_shape": list(ACTION_SHAPE),
                        "future_shape": list(FUTURE_SHAPE),
                        "action_initial_noise_shape": list(NOISE_SHAPE),
                    }
                    _save_torch_atomic(torch, tensor_path, payload)
                    save(json_path, metadata)
                    key = f"{case_id}:{variant}"
                    if key not in progress["completed_keys"]:
                        progress["completed_keys"].append(key)
                        if source_kind == "new_query":
                            _complete_query_attempt(progress, query_key, current_job)
                            progress["new_query_completed_total"] += 1
                            segment["new_query_completed"] += 1
                        else:
                            progress["reuse_calibration_total"] += 1
                    outputs = _validate_training_record(
                        torch, payload, metadata, binding, config,
                        plan_summary["pbs_jobid"], source_job, source_file, eligibility)
                elif f"{case_id}:{variant}" not in progress["completed_keys"]:
                    key = f"{case_id}:{variant}"
                    progress["completed_keys"].append(key)
                    if source_kind == "new_query":
                        _complete_query_attempt(progress, query_key, payload["producer_job"])
                        progress["new_query_completed_total"] += 1
                    else:
                        progress["reuse_calibration_total"] += 1

                records[(case_id, variant)] = outputs
                _sync_counts(summary, progress, segment, len(records), planned_queries)
                summary["status"] = "running"
                summary["complete"] = False
                _persist(out, summary, progress)
            if stop_for_deadline:
                break
            if (case_id, "clean") in records and (case_id, "noisy") in records:
                _noise_equal(torch, records[(case_id, "clean")]["action_initial_noise"],
                             records[(case_id, "noisy")]["action_initial_noise"])
                noise_parity_cases.add(case_id)
                summary["initial_noise_parity_contexts"] = len(noise_parity_cases)
                summary["initial_noise_parity_valid"] = len(noise_parity_cases) == 80
                _persist(out, summary, progress)
            del datums, identities, clean_arrays, noisy_arrays

        if switching is not None and snapshot_before is not None:
            switching.activate("bf16")
            snapshot_after = frozen_snapshot(switching, [])
            frozen_unchanged = snapshot_before == snapshot_after and _all_frozen(switching)
            summary["frozen_model_unchanged"] = frozen_unchanged
            if not frozen_unchanged:
                raise RuntimeError("Frozen model/bank snapshot or requires_grad state changed")

        total_attempted = len(progress["query_attempts"])
        total_completed = progress["new_query_completed_total"]
        attempted_keys = {item["key"] for item in progress["query_attempts"]}
        expected_query_keys = {
            f"{case_id}:noisy" for case_id in CASE_IDS
            if _source_eligibility(plan_by_id[case_id])["new_noisy_teacher_query_required"]
        }
        attempts_resolved = all(item["status"] != "attempted"
                                for item in progress["query_attempts"])
        complete = (
            len(records) == EXPECTED_RECORDS
            and planned_queries <= MAX_NEW_QUERIES
            and total_completed == planned_queries
            and attempted_keys == expected_query_keys
            and attempts_resolved
            and len(progress["completed_keys"]) == EXPECTED_RECORDS
            and len(noise_parity_cases) == 80
            and frozen_unchanged is True
            and _all_frozen(switching)
        )
        if stop_for_deadline:
            summary["status"] = "checkpointed"
            summary["checkpointed"] = True
            summary["complete"] = False
        elif complete:
            summary["status"] = "complete"
            summary["complete"] = True
        else:
            raise RuntimeError("Training teacher completion gates did not pass")
    except Exception as exc:
        failure = exc
    finally:
        if switching is not None:
            try:
                if snapshot_before is not None:
                    switching.activate("bf16")
                    snapshot_after = frozen_snapshot(switching, [])
                    frozen_unchanged = snapshot_before == snapshot_after and _all_frozen(switching)
                    summary["frozen_model_unchanged"] = frozen_unchanged
                switching.close()
            except Exception as exc:
                if failure is None:
                    failure = exc

    _sync_counts(summary, progress, segment, len(records), planned_queries)
    if summary.get("complete") is not True and summary.get("checkpointed") is not True:
        summary["status"] = "failed"
        summary["complete"] = False
    if failure is not None:
        summary["status"] = "failed"
        summary["complete"] = False
        summary["checkpointed"] = False
        summary["failure"] = f"{type(failure).__name__}: {failure}"
        if progress_ready:
            _persist(out, summary, progress)
        else:
            save(out / "summary.json", summary)
        raise failure

    if progress_ready:
        _persist(out, summary, progress)
    else:
        save(out / "summary.json", summary)
    print({key: summary[key] for key in (
        "unit", "status", "complete", "completed_records",
        "new_query_attempted", "new_query_completed",
        "reuse_calibration", "reuse_existing")}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
