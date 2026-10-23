"""Run fixed-input clean/noisy C80 queries for all four frozen arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from control_common import Deadline, guard, read, save


ARMS = ("bf16", "per_channel_clip", "softbin", "relational")
VARIANTS = ("clean", "noisy")
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)
NOISE_SHAPE = (1, 32, 7)
EXPECTED_CASES = 40
EXPECTED_TOTAL_CASES = 80
SOFT_DEADLINE_SECONDS = 6600
RESERVE_SECONDS = 180


class DeadlineReached(RuntimeError):
    pass


def _pair_binding(protocol, unit, case, arm, variant):
    return {
        "protocol": protocol,
        "unit": unit,
        "index": int(case["case_id"]),
        "arm": arm,
        "variant": variant,
        "seed": int(case["sampler_seed"]),
        "description": case["description"],
    }


def _pair_catalog(protocol, definition, pair_summary, original_rows,
                  expected_ids, shard, expected_per_shard=EXPECTED_CASES,
                  expected_total=EXPECTED_TOTAL_CASES):
    expected_ids = [int(value) for value in expected_ids]
    expected_set = set(expected_ids)
    if (len(expected_set) != int(expected_total)
            or definition.get("protocol") != protocol
            or definition.get("definition") != "calibration-noise-pairs-v1"
            or definition.get("training_use") is not False
            or definition.get("environment_steps") != 0):
        raise ValueError("Frozen all-C80 calibration-pairs definition is required")
    if (pair_summary.get("protocol") != protocol
            or pair_summary.get("unit") != "calibration_pairs"
            or pair_summary.get("complete") is not True
            or pair_summary.get("contexts") != int(expected_total)
            or pair_summary.get("training_updates") != 0
            or pair_summary.get("environment_steps") != 0
            or pair_summary.get("heldout_inputs_used") is not False
            or pair_summary.get("training_pool_modified") is not False):
        raise ValueError("Complete CPU calibration-pairs summary with no heldout/training use is required")

    cases = definition.get("cases")
    records = pair_summary.get("records")
    if not isinstance(cases, list) or len(cases) != int(expected_total):
        raise ValueError("Calibration-pairs definition must contain all 80 C80 cases")
    if not isinstance(records, list) or len(records) != int(expected_total):
        raise ValueError("Calibration-pairs CPU summary must contain all 80 case records")
    case_map = {int(case["case_id"]): case for case in cases}
    record_map = {int(record["case_id"]): record for record in records}
    if set(case_map) != expected_set or set(record_map) != expected_set:
        raise ValueError("Calibration-pairs case IDs differ from the original C80 cohort")
    if len(case_map) != len(cases) or len(record_map) != len(records):
        raise ValueError("Calibration-pairs definition or summary contains duplicate case IDs")

    shard_cases = []
    for case_id in expected_ids:
        case, summary_record = case_map[case_id], record_map[case_id]
        row = original_rows.get(case_id)
        if row is None or case.get("row") != row:
            raise ValueError(f"Calibration case {case_id} no longer matches the original C80 plan row")
        if (int(case.get("sampler_seed", -1)) != int(row["sampler_seeds"][0])
                or int(case.get("shard", -1)) not in (0, 1)
                or case.get("suite") != row.get("suite")):
            raise ValueError(f"Calibration case {case_id} row, seed, or training boundary differs")
        for key in ("case_id", "suite", "row", "family", "severity", "noise_level",
                    "corruption_seed", "sampler_seed", "shard", "clean_observation",
                    "clean_metadata"):
            if summary_record.get(key) != case.get(key):
                raise ValueError(f"CPU pair summary binding differs for case {case_id}: {key}")
        if summary_record.get("training_use") is not False:
            raise ValueError(f"CPU pair summary allows training use for case {case_id}")
        if (not isinstance(summary_record.get("description"), str)
                or not summary_record.get("description")
                or not isinstance(summary_record.get("noisy_observation"), str)
                or summary_record.get("camera_changed") is not True):
            raise ValueError(f"CPU pair summary lacks prepared observation binding for case {case_id}")
        case = {
            **case,
            "description": summary_record["description"],
            "noisy_observation": summary_record["noisy_observation"],
            "camera_changed": summary_record["camera_changed"],
            "training_use": False,
        }
        case_map[case_id] = case
        if int(case["shard"]) == int(shard):
            shard_cases.append(case)

    if len(shard_cases) != int(expected_per_shard):
        raise ValueError(f"calibration_{shard} must own {expected_per_shard} cases")
    if len({int(case["case_id"]) for case in shard_cases}) != len(shard_cases):
        raise ValueError("Shard contains duplicate calibration case IDs")
    return shard_cases


def _validate_pair_metadata(case, pair_metadata, source_metadata):
    case_id = int(case["case_id"])
    for key, value in case.items():
        if pair_metadata.get(key) != value:
            raise ValueError(f"Per-case calibration metadata differs for {case_id}: {key}")
    if (pair_metadata.get("description") != case.get("description")
            or pair_metadata.get("camera_changed") is not True
            or pair_metadata.get("training_use") is not False):
        raise ValueError(f"Per-case calibration pair metadata is incomplete for {case_id}")
    row = case["row"]
    for key in ("case_id", "demo_id", "frame_index", "sampler_seeds"):
        if source_metadata.get(key) != row.get(key):
            raise ValueError(f"Original clean metadata differs for case {case_id}: {key}")
    if source_metadata.get("description") != case.get("description"):
        raise ValueError(f"Original clean description differs for case {case_id}")


def _validate_sensor0_gate(protocol, sensor_plan, sensor_summary,
                           expected_cases=20):
    if (sensor_plan.get("protocol") != protocol
            or sensor_plan.get("complete") is not True
            or sensor_plan.get("training_use") is not False):
        raise ValueError("Frozen sensor plan is incomplete")
    cases = sensor_plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 40:
        raise ValueError("The complete 40-case sensor source plan is required")
    expected_indices = sorted(int(case["index"]) for case in cases
                              if int(case["shard"]) == 0)
    if len(expected_indices) != expected_cases:
        raise ValueError("sensor_0 source plan does not contain exactly 20 cases")
    records = sensor_summary.get("records")
    if (sensor_summary.get("protocol") != protocol
            or sensor_summary.get("unit") != "sensor_0"
            or sensor_summary.get("complete") is not True
            or sensor_summary.get("expected") != expected_cases
            or sensor_summary.get("completed") != expected_cases
            or sensor_summary.get("expected_base_queries") != 160
            or sensor_summary.get("expected_repeat_queries") != 1
            or sensor_summary.get("training_updates") != 0
            or sensor_summary.get("training_use") is not False
            or not isinstance(records, list)
            or sorted(int(record["index"]) for record in records) != expected_indices):
        raise ValueError("The full sensor_0 CPU gate summary is required before calibration queries")
    parity = sensor_summary.get("source_first_chunk_parity")
    if (not isinstance(parity, list) or len(parity) != expected_cases
            or any(not isinstance(item, dict)
                   or set(item) != set(ARMS)
                   or any(value != 0 for value in item.values()) for item in parity)):
        raise ValueError("sensor_0 must pass exact source first-chunk parity for all four arms")
    repeats = sensor_summary.get("same_input_repeats")
    if (not isinstance(repeats, list) or len(repeats) != 1
            or repeats[0].get("action_max_abs") != 0
            or repeats[0].get("future_max_abs") != 0
            or repeats[0].get("initial_noise_max_abs") != 0):
        raise ValueError("sensor_0 exact same-input BF16 repeat gate is required")


def _max_abs(torch, left, right, shape, label):
    left = torch.as_tensor(left).detach().float().cpu()
    right = torch.as_tensor(right).detach().float().cpu()
    if tuple(left.shape) != tuple(shape) or tuple(right.shape) != tuple(shape):
        raise ValueError(f"{label} shape differs from {tuple(shape)}")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError(f"{label} contains non-finite values")
    return float((left - right).abs().max())


def _load_npz(np, path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_case_inputs(torch, np, switching, root, case, pair_metadata_path):
    pair_metadata = read(pair_metadata_path)
    source_metadata = read(_resolve(root, case["clean_metadata"]))
    _validate_pair_metadata(case, pair_metadata, source_metadata)
    paths = {
        "clean": _resolve(root, case["clean_observation"]),
        "noisy": _resolve(root, case["noisy_observation"]),
    }
    for variant, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared {variant} C80 observation for {case['case_id']}")
    observations = {variant: _load_npz(np, path) for variant, path in paths.items()}
    datums = {variant: switching.datum(observation)
              for variant, observation in observations.items()}
    from diagnostic_runtime import datum_identity

    identities = {variant: datum_identity(torch, datum)
                  for variant, datum in datums.items()}
    if identities["clean"] == identities["noisy"]:
        raise ValueError(f"Prepared clean/noisy inputs are identical for {case['case_id']}")
    return datums, identities, paths


def _teacher_source(p, expected_ids):
    source_protocol = p["source_relational_protocol"]
    teacher_root = Path(source_protocol["control_root"]) / "results" / "teacher" / "calibration"
    receipt_path = teacher_root / "receipt.json"
    receipt = read(receipt_path)
    if (receipt.get("complete") is not True
            or receipt.get("queries") != len(expected_ids)
            or receipt.get("case_ids") != list(expected_ids)
            or receipt.get("seed_indices") != [0]):
        raise ValueError("Original C80 BF16 teacher receipt is incomplete or mismatched")
    return teacher_root, receipt_path


def _load_teacher(torch, teacher_root, case, original_row):
    case_id = int(case["case_id"])
    path = teacher_root / f"{case_id}_0.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original C80 BF16 teacher cache: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("plan_row") != original_row
            or int(saved.get("sampler_seed", -1)) != int(case["sampler_seed"])):
        raise ValueError(f"Original BF16 teacher plan-row/seed binding differs for {case_id}")
    action = saved.get("action")
    future = saved.get("future")
    _max_abs(torch, action, action, ACTION_SHAPE, f"teacher action {case_id}")
    _max_abs(torch, future, future, FUTURE_SHAPE, f"teacher future {case_id}")
    counts = saved.get("counts")
    if counts is not None and counts != {"video_steps": 10, "action_steps": 10}:
        raise ValueError(f"Original BF16 teacher counts differ for {case_id}")
    return saved, path


def _record_paths(out, index, arm, variant):
    stem = f"{int(index)}_{arm}_{variant}"
    return out / "records" / f"{stem}.pt", out / "records" / f"{stem}.json"


def _noise_parity(torch, results):
    for variant in VARIANTS:
        if not all((variant, arm) in results for arm in ARMS):
            continue
        reference = results[(variant, "bf16")]["action_initial_noise"]
        if tuple(reference.shape) != NOISE_SHAPE:
            raise ValueError(f"Initial action noise shape differs for {variant}")
        for arm in ARMS[1:]:
            candidate = results[(variant, arm)]["action_initial_noise"]
            if not torch.equal(reference, candidate):
                delta = _max_abs(torch, reference, candidate, NOISE_SHAPE,
                                 f"initial noise {variant}/{arm}")
                raise RuntimeError(f"Four-arm action initial noise differs: {variant}/{arm}, {delta}")


def _write_summary(path, protocol, unit, shard, cases, pair_summary_path,
                   sensor_summary_path, teacher_receipt_path, native_records,
                   case_results, counters, status):
    expected_keys = {
        (int(case["case_id"]), variant, arm)
        for case in cases for variant in VARIANTS for arm in ARMS
    }
    observed_keys = set(native_records)
    all_records = observed_keys == expected_keys
    counts_valid = all(record.get("counts") == {"video_steps": 10, "action_steps": 10}
                       and record.get("action_shape") == list(ACTION_SHAPE)
                       and record.get("future_shape") == list(FUTURE_SHAPE)
                       and record.get("action_initial_noise_shape") == list(NOISE_SHAPE)
                       for record in native_records.values())
    noise_groups = {}
    for key, record in native_records.items():
        noise_groups.setdefault(key[:2], {})[key[2]] = record.get(
            "action_initial_noise_identity")
    noise_parity_valid = (
        len(noise_groups) == len(cases) * len(VARIANTS)
        and all(set(group) == set(ARMS)
                and len({json.dumps(identity, sort_keys=True)
                         for identity in group.values()}) == 1
                for group in noise_groups.values())
    )
    clean_teacher_parity = (
        len(case_results) == len(cases)
        and all(item.get("clean_bf16_teacher_parity", {}).get("action_max_abs") == 0.0
                and item.get("clean_bf16_teacher_parity", {}).get("future_max_abs") == 0.0
                for item in case_results.values())
    )
    complete = (status == "complete" and all_records and counts_valid
                and noise_parity_valid and clean_teacher_parity)
    records = []
    for case in cases:
        case_id = int(case["case_id"])
        keys = {(case_id, variant, arm) for variant in VARIANTS for arm in ARMS}
        item = case_results.get(case_id, {})
        parity = item.get("clean_bf16_teacher_parity") or {}
        records.append({
            "index": case_id,
            "case_id": case_id,
            "suite": case["suite"],
            "shard": int(case["shard"]),
            "row": case["row"],
            "sampler_seed": int(case["sampler_seed"]),
            "native_queries_expected": len(keys),
            "native_queries_saved": len(observed_keys & keys),
            "clean_bf16_teacher_parity": parity or None,
            "initial_noise_parity_valid": item.get("initial_noise_parity_valid", False),
        })
    summary = {
        "protocol": protocol,
        "unit": unit,
        "status": "complete" if complete else status,
        "complete": complete,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "shard": shard,
        "expected": len(cases),
        "completed": sum(record["native_queries_saved"] == record["native_queries_expected"]
                         and record["initial_noise_parity_valid"]
                         and (record.get("clean_bf16_teacher_parity") or {}).get("action_max_abs") == 0.0
                         and (record.get("clean_bf16_teacher_parity") or {}).get("future_max_abs") == 0.0
                         for record in records),
        "records": records,
        "core_queries_expected": len(expected_keys),
        "core_queries_saved": len(observed_keys),
        "core_queries_new_this_segment": counters["full_model_queries_new"],
        "core_queries_attempted_new_this_segment": counters["full_model_queries_attempted_new"],
        "core_queries_reused": counters["query_records_reused"],
        "new_model_queries": counters["full_model_queries_new"],
        "new_student_queries": counters["non_bf16_queries_new"],
        "new_teacher_queries": counters["bf16_queries_new"],
        "full_model_queries_new": counters["full_model_queries_new"],
        "new_teacher_queries_scope": "BF16 arm queries, a subset of full_model_queries_new; no teacher-only forward",
        "cached_source_teacher_records_read": counters["source_teacher_records_read"],
        "source_teacher_cache": "control_root/results/teacher/calibration/{case_id}_0.pt",
        "source_teacher_receipt": str(teacher_receipt_path),
        "source_teacher_binding": "original C80 plan_row and sampler_seeds[0]; source cache has no input digest",
        "source_cpu_summary": str(pair_summary_path),
        "source_cpu_summary_complete": True,
        "sensor_0_gate_summary": str(sensor_summary_path),
        "sensor_0_gate_passed": True,
        "counts_valid": counts_valid,
        "target_linears_covered_per_native_query": 614,
        "initial_noise_parity_all_arms_per_variant": noise_parity_valid,
        "clean_bf16_action_future_exact_parity_all_cases": clean_teacher_parity,
        "new_training_updates": 0,
        "heldout_inputs_used": False,
        "training_use": False,
        "source_scope": "prepared original C80 clean/noisy fixed-input pairs; no new observations or environment steps",
        "clean_noisy_seed_rule": "both variants use the single original sampler_seeds[0] binding for each case",
        "updated_at_epoch": time.time(),
    }
    save(path, summary)
    return summary


def _run_unit(args):
    p = read(args.protocol)
    protocol = p["protocol"]
    shard = int(args.unit[-1])
    root = Path(p["remote_root"])
    relp = p["source_relational_protocol"]

    definition_path = root / "data" / "calibration_pairs" / "definition.json"
    pair_summary_path = root / "results" / "calibration_pairs" / "summary.json"
    sensor_plan_path = root / "data" / "sensor_plan.json"
    sensor_summary_path = root / "results" / "sensor_0" / "summary.json"
    definition, pair_summary = read(definition_path), read(pair_summary_path)
    sensor_plan, sensor_summary = read(sensor_plan_path), read(sensor_summary_path)
    _validate_sensor0_gate(protocol, sensor_plan, sensor_summary)

    original_plan_path = Path(relp["calibration"]["plan_file"])
    original_plan = read(original_plan_path)
    expected_ids = [int(value) for value in relp["calibration"]["case_ids"]]
    if (original_plan.get("status") != "complete"
            or original_plan.get("calibration_cohorts", {}).get("c80") != expected_ids):
        raise ValueError("Original complete C80 calibration plan is required")
    original_rows = {int(row["case_id"]): row for row in original_plan.get("inputs", [])
                     if int(row["case_id"]) in expected_ids}
    if set(original_rows) != set(expected_ids):
        raise ValueError("Original C80 plan rows do not cover the frozen cohort exactly")
    cases = _pair_catalog(protocol, definition, pair_summary, original_rows,
                          expected_ids, shard)

    teacher_root, teacher_receipt_path = _teacher_source(p, expected_ids)
    out = root / "results" / args.unit
    out.mkdir(parents=True, exist_ok=True)
    records_dir = out / "records"
    summary_path = out / "summary.json"
    run_binding = {
        "protocol": protocol,
        "unit": args.unit,
        "shard": shard,
        "case_ids": [int(case["case_id"]) for case in cases],
        "source_cpu_definition": str(definition_path),
        "source_cpu_summary": str(pair_summary_path),
        "sensor_0_gate_summary": str(sensor_summary_path),
        "original_c80_plan": str(original_plan_path),
        "teacher_cache": str(teacher_root),
        "arms": list(ARMS),
        "variants": list(VARIANTS),
    }
    binding_path = out / "run_binding.json"
    if binding_path.is_file() and read(binding_path) != run_binding:
        raise ValueError("Existing calibration output belongs to a different frozen C80 binding")
    if not binding_path.is_file():
        save(binding_path, run_binding)

    pair_metadata_paths = {
        int(case["case_id"]): root / "data" / "calibration_pairs"
        / str(int(case["case_id"])) / "metadata.json"
        for case in cases
    }
    for case in cases:
        metadata_path = pair_metadata_paths[int(case["case_id"])]
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing calibration pair metadata: {metadata_path}")
        _validate_pair_metadata(case, read(metadata_path),
                                read(_resolve(root, case["clean_metadata"])))

    native_records, case_results = {}, {}
    counters = {
        "full_model_queries_attempted_new": 0,
        "full_model_queries_new": 0,
        "non_bf16_queries_new": 0,
        "bf16_queries_new": 0,
        "query_records_reused": 0,
        "source_teacher_records_read": 0,
    }
    status = "running"
    _write_summary(summary_path, protocol, args.unit, shard, cases,
                   pair_summary_path, sensor_summary_path, teacher_receipt_path,
                   native_records, case_results, counters, status)

    import numpy as np
    import torch
    from diagnostic_runtime import SwitchingRuntime
    from sensor_worker import _query_paths, _query_record

    switching = None
    deadline = Deadline(SOFT_DEADLINE_SECONDS)
    try:
        switching = SwitchingRuntime(p, out)
        if len(switching.runtime.q.modules) != 614:
            raise RuntimeError("Expected all 614 frozen target Linears")
        for case in cases:
            case_id = int(case["case_id"])
            datums, identities, observation_paths = _load_case_inputs(
                torch, np, switching, root, case, pair_metadata_paths[case_id]
            )
            switching.runtime.description = case["description"]
            source_teacher, teacher_path = _load_teacher(
                torch, teacher_root, case, original_rows[case_id]
            )
            counters["source_teacher_records_read"] += 1
            case_native = {}
            stop_for_deadline = False
            for variant in VARIANTS:
                for arm in ARMS:
                    tensor_path, json_path = _query_paths(out, case_id, arm, variant)
                    if tensor_path.is_file() != json_path.is_file():
                        raise RuntimeError(f"Partial endpoint pair exists: {tensor_path.stem}")
                    cached = tensor_path.is_file() and json_path.is_file()
                    if not cached and deadline.near(RESERVE_SECONDS):
                        stop_for_deadline = True
                        break

                    source_action = None
                    if arm == "bf16" and variant == "clean":
                        source_action = source_teacher["action"]
                    if not cached:
                        counters["full_model_queries_attempted_new"] += 1
                    result, reused = _query_record(
                        switching, torch, args.unit, case_id, arm, variant,
                        int(case["sampler_seed"]), case["description"],
                        datums[variant], identities[variant], out,
                        source_action=source_action,
                    )
                    if reused:
                        counters["query_records_reused"] += 1
                    else:
                        counters["full_model_queries_new"] += 1
                        counters["bf16_queries_new" if arm == "bf16"
                                 else "non_bf16_queries_new"] += 1
                    metadata = read(json_path)
                    if (tuple(result["action"].shape) != ACTION_SHAPE
                            or tuple(result["future"].shape) != FUTURE_SHAPE
                            or tuple(result["action_initial_noise"].shape) != NOISE_SHAPE
                            or result.get("counts") != {"video_steps": 10, "action_steps": 10}
                            or metadata.get("action_initial_noise_identity") is None):
                        raise RuntimeError(f"Native endpoint shape/count/noise binding differs: {tensor_path.stem}")
                    result["action"] = torch.as_tensor(result["action"]).detach().float().cpu()
                    result["future"] = torch.as_tensor(result["future"]).detach().float().cpu()
                    result["action_initial_noise"] = torch.as_tensor(
                        result["action_initial_noise"]).detach().float().cpu()
                    case_native[(variant, arm)] = result
                    native_records[(case_id, variant, arm)] = metadata

                    if arm == "bf16" and variant == "clean":
                        action_error = _max_abs(
                            torch, result["action"], source_teacher["action"],
                            ACTION_SHAPE, f"clean BF16 action parity {case_id}",
                        )
                        future_error = _max_abs(
                            torch, result["future"], source_teacher["future"],
                            FUTURE_SHAPE, f"clean BF16 future parity {case_id}",
                        )
                        if action_error != 0.0 or future_error != 0.0:
                            raise RuntimeError(
                                f"Clean BF16 full action/future parity failed for {case_id}: "
                                f"action={action_error}, future={future_error}"
                            )
                        parity = {
                            "action_max_abs": action_error,
                            "future_max_abs": future_error,
                            "reference": str(teacher_path),
                            "reference_binding": "original C80 plan_row and sampler_seeds[0]",
                        }
                    _write_summary(
                        summary_path, protocol, args.unit, shard, cases,
                        pair_summary_path, sensor_summary_path, teacher_receipt_path,
                        native_records, case_results, counters, status,
                    )
                if stop_for_deadline:
                    break
                _noise_parity(torch, case_native)
            if stop_for_deadline:
                status = "resumable_deadline"
                break
            if len(case_native) != len(VARIANTS) * len(ARMS):
                raise RuntimeError(f"Incomplete native matrix for C80 case {case_id}")
            _noise_parity(torch, case_native)
            case_results[case_id] = {
                "clean_bf16_teacher_parity": parity,
                "initial_noise_parity_valid": True,
                "source_teacher_cache": str(teacher_path),
                "observation_paths": {
                    variant: str(path) for variant, path in observation_paths.items()
                },
            }
            _write_summary(
                summary_path, protocol, args.unit, shard, cases,
                pair_summary_path, sensor_summary_path, teacher_receipt_path,
                native_records, case_results, counters, status,
            )
            del datums, case_native, source_teacher
        if status == "running":
            status = "complete"
    except Exception:
        status = "failed"
        _write_summary(
            summary_path, protocol, args.unit, shard, cases,
            pair_summary_path, sensor_summary_path, teacher_receipt_path,
            native_records, case_results, counters, status,
        )
        raise
    finally:
        if switching is not None:
            switching.close()
        summary = _write_summary(
            summary_path, protocol, args.unit, shard, cases,
            pair_summary_path, sensor_summary_path, teacher_receipt_path,
            native_records, case_results, counters, status,
        )
    print("CALIBRATION_WORKER_SUMMARY " + json.dumps(summary), flush=True)
    return summary


def selfcheck():
    previous_start = os.environ.get("FW_JOB_START_EPOCH")
    os.environ["FW_JOB_START_EPOCH"] = str(int(time.time()))
    try:
        assert isinstance(Deadline(SOFT_DEADLINE_SECONDS).near(RESERVE_SECONDS), bool)
    finally:
        if previous_start is None:
            os.environ.pop("FW_JOB_START_EPOCH", None)
        else:
            os.environ["FW_JOB_START_EPOCH"] = previous_start

    protocol = "toy-calibration-v1"
    original_rows, cases, summary_records = {}, [], []
    for case_id in range(2000, 2004):
        row = {"case_id": case_id, "suite": "toy", "demo_id": case_id,
               "frame_index": 1, "sampler_seeds": [900000000 + case_id]}
        original_rows[case_id] = row
        case = {
            "case_id": case_id, "suite": "toy", "row": row,
            "family": "fog", "severity": 2, "noise_level": 32,
            "corruption_seed": 760000000 + case_id * 100 + 17,
            "sampler_seed": row["sampler_seeds"][0], "shard": (case_id - 2000) % 2,
            "clean_observation": f"/toy/clean_{case_id}.npz",
            "clean_metadata": f"/toy/clean_{case_id}.json",
            "description": f"toy task {case_id}",
            "noisy_observation": f"/toy/noisy_{case_id}.npz", "camera_changed": True,
        }
        cases.append({key: value for key, value in case.items()
                      if key not in ("description", "noisy_observation", "camera_changed")})
        summary_records.append({**case, "training_use": False})
    definition = {"protocol": protocol, "definition": "calibration-noise-pairs-v1",
                  "cases": cases, "training_use": False, "environment_steps": 0}
    pair_summary = {
        "protocol": protocol, "unit": "calibration_pairs", "complete": True,
        "contexts": 4, "training_updates": 0, "environment_steps": 0,
        "heldout_inputs_used": False, "training_pool_modified": False,
        "records": summary_records,
    }
    selected = _pair_catalog(protocol, definition, pair_summary, original_rows,
                             list(original_rows), 0, expected_per_shard=2,
                             expected_total=4)
    assert [case["case_id"] for case in selected] == [2000, 2002]
    case = selected[0]
    pair_metadata = {**case, "description": "toy task 2000",
                     "noisy_observation": "/toy/noisy_2000.npz", "camera_changed": True}
    source_metadata = {key: case["row"][key]
                       for key in ("case_id", "demo_id", "frame_index", "sampler_seeds")}
    source_metadata["description"] = "toy task 2000"
    _validate_pair_metadata(case, pair_metadata, source_metadata)
    binding = _pair_binding(protocol, "calibration_0", pair_metadata, "bf16", "clean")
    assert binding == {
        "protocol": protocol, "unit": "calibration_0", "index": 2000,
        "arm": "bf16", "variant": "clean", "seed": 900002000,
        "description": "toy task 2000",
    }

    sensor_plan = {"protocol": protocol, "complete": True, "training_use": False,
                   "cases": [{"index": i, "shard": i % 2} for i in range(40)]}
    sensor_summary = {
        "protocol": protocol, "unit": "sensor_0", "complete": True,
        "expected": 20, "completed": 20, "expected_base_queries": 160,
        "expected_repeat_queries": 1, "training_updates": 0, "training_use": False,
        "records": [{"index": i} for i in range(0, 40, 2)],
        "source_first_chunk_parity": [{arm: 0.0 for arm in ARMS} for _ in range(20)],
        "same_input_repeats": [{"action_max_abs": 0.0, "future_max_abs": 0.0,
                                 "initial_noise_max_abs": 0.0}],
    }
    _validate_sensor0_gate(protocol, sensor_plan, sensor_summary)
    assert _pair_binding(protocol, "calibration_0", pair_metadata, "bf16", "clean")["seed"] == \
        _pair_binding(protocol, "calibration_0", pair_metadata, "bf16", "noisy")["seed"]
    print(json.dumps({
        "status": "passed",
        "deadline_api": True,
        "pair_catalog_and_case_binding": True,
        "sensor_0_gate_binding": True,
        "clean_noisy_share_sampler_seed": True,
        "model_or_cuda_initialized": False,
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", choices=("calibration_0", "calibration_1"))
    parser.add_argument("--protocol", default=str(Path(__file__).with_name("protocol.json")))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.unit is None:
        parser.error("--unit is required unless --selfcheck is used")
    guard()
    _run_unit(args)


if __name__ == "__main__":
    main()
