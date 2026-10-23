"""Original heldout80 evaluation for completed paired-C80 training arms."""
import argparse
import os
from pathlib import Path
import re
import math
import statistics

from control_common import Deadline, guard, read, save


COUNTS = {"video_steps": 10, "action_steps": 10}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
METRICS = ("first10motor_rmse", "full32motor_rmse", "late22motor_rmse",
           "first10gripper_rmse", "future_latent_nmse", "action_kl", "latent_kl",
           "action_temporal_kl", "latent_spatiotemporal_kl", "softbin_objective",
           "relational_objective", "command_rmse", "command_chunk_bias_rms",
           "command_prefix_cumsum_max_abs")


def heldout_queries(parent, shard=None):
    evaluation = parent["source_relational_protocol"]["evaluation"]
    if (evaluation["case_ids"] != list(range(3000, 3040))
            or evaluation["seed_indices"] != [0, 1]
            or shard not in (None, 0, 1)):
        raise ValueError("Original 40 cases and both heldout seeds are required")
    return [(case, seed) for case in evaluation["case_ids"]
            for seed in evaluation["seed_indices"] if shard is None or seed == shard]


def validate_endpoint(torch, saved, binding):
    if any(saved.get(key) != value for key, value in binding.items()):
        raise ValueError("Heldout endpoint source/input binding differs")
    for key, shape in (("action", (32, 7)), ("future", (1, 48, 3, 14, 28)),
                       ("action_initial_noise", (1, 32, 7))):
        value = saved.get(key)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError("Invalid heldout endpoint tensor: " + key)
    if saved.get("counts") != COUNTS:
        raise ValueError("Heldout endpoint sampler counts differ")
    if binding.get("split") == "heldout":
        for key in ("commands", "teacher_commands"):
            value = saved.get(key)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (32, 7)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid heldout execution commands: " + key)
    return saved


def aggregate_records(parent, records):
    actual = [(row["case_id"], row["seed_index"]) for row in records]
    if len(actual) != len(set(actual)) or set(actual) != set(heldout_queries(parent)):
        raise ValueError("Heldout aggregate requires all original 40 x 2 queries")
    by_suite = {}
    for suite in SUITES:
        rows = [row for row in records if row["suite"] == suite]
        cases = {row["case_id"] for row in rows}
        if len(rows) != 20 or len(cases) != 10 or any(
                {row["seed_index"] for row in rows if row["case_id"] == case} != {0, 1}
                for case in cases):
            raise ValueError("Each suite requires ten cases and both seeds")
        by_suite[suite] = {key: statistics.fmean(
            statistics.fmean(row[key] for row in rows if row["case_id"] == case)
            for case in cases) for key in METRICS}
    overall = {key: statistics.fmean(by_suite[suite][key] for suite in SUITES)
               for key in METRICS}
    if any(not math.isfinite(row[key]) or row[key] < 0 for row in records for key in METRICS):
        raise ValueError("Endpoint metrics must be finite and nonnegative")
    return {"queries": 80, "cases": 40, "seed_indices": [0, 1],
            "aggregation": "mean seeds per case, mean cases per suite, mean four suites",
            "by_suite": by_suite, "overall": overall}


def prepare_runtime(switching, parent, config, arm, out, state):
    """Install completed D/r and replay its existing final training reference."""
    import numpy as np
    from aggregate import completed
    from diagnostic_runtime import datum_identity
    from trained_deployment import install_training_deployment
    from training_teacher import _load_npz, _resolve

    torch, root = switching.torch, Path(parent["remote_root"])
    unit = "train_" + arm
    summary = completed(root, unit)
    deployment_path = root / "results" / unit / "deployment.pt"
    deployment = torch.load(deployment_path, map_location="cpu", weights_only=True)
    binding = install_training_deployment(
        switching, parent, config, arm, deployment, summary, str(deployment_path))
    reference_path = root / "results" / unit / "reference_endpoint.pt"
    if deployment.get("reference_endpoint") != str(reference_path):
        raise ValueError("Training reference endpoint path differs")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    first = config["update_order"][0]
    definition = read(root / config["data_config"]["dataset_subdir"] / "definition.json")
    cases = {row["case_id"]: row for row in definition["cases"]}
    case = cases[first]
    metadata = read(_resolve(root, case["clean_metadata"]))
    seed = case["sampler_seed"]
    datum = switching.datum(_load_npz(np, _resolve(root, case["clean_observation"])))
    expected = {
        "protocol": parent["protocol"], "training_protocol": config,
        "unit": unit, "arm": arm, "pbs_jobid": summary["pbs_jobid"],
        "completed_updates": 80, "case_id": first, "variant": "clean",
        "sampler_seed": seed, "description": metadata["description"],
        "input_identity": datum_identity(torch, datum),
    }
    validate_endpoint(torch, reference, expected)
    state["startup_query_attempted"] += 1
    save(out / "progress.json", state)
    action, future, counts, noise = switching.query(datum, seed, metadata["description"])
    differences = {
        "source_action_max_abs": float((action - reference["action"].float()).abs().max()),
        "source_future_max_abs": float((future - reference["future"].float()).abs().max()),
        "source_initial_noise_max_abs": float((noise - reference["action_initial_noise"]).abs().max()),
    }
    switching.banks.seen.clear()
    state["startup_query_attempted"] += 1
    save(out / "progress.json", state)
    actual, _ = switching.runtime.infer(datum, seed, measure=False)
    differences["actual_infer_action_max_abs"] = float(
        (actual.detach().float().cpu() - reference["action"].float()).abs().max())
    if (any(value != 0 for value in differences.values()) or counts != COUNTS
            or len(switching.banks.seen) != 614):
        raise RuntimeError("New deployment source/native-infer parity failed: " + str(differences))
    receipt = {"protocol": parent["protocol"], "arm": arm, "binding": binding,
               "pbs_jobid": os.environ["PBS_JOBID"], "passed": True,
               "observed_modules": 614, "startup_queries": 2, **differences}
    save(out / "setup_receipt.json", receipt)
    return binding


def run(unit, arm, shard):
    import numpy as np
    import torch
    import calibration_source
    import clipping_reference
    from aggregate import completed
    from checkpointing import atomic_torch_save
    from diagnostic_runtime import SwitchingRuntime, datum_identity

    here = Path(__file__).resolve().parent
    parent, config = read(here / "protocol.json"), read(here / "training_protocol.json")
    root, job = Path(parent["remote_root"]), os.environ["PBS_JOBID"]
    if arm not in config["arms"]:
        raise ValueError("Unknown frozen training arm")
    completed(root, "train_" + arm)
    queries = heldout_queries(parent, shard)
    out = root / "results" / unit
    out.mkdir(parents=True, exist_ok=True)
    state = {"protocol": parent["protocol"], "unit": unit, "arm": arm,
             "pbs_jobid": job, "complete": False, "status": "running",
             "expected_queries": 40, "completed_queries": 0,
             "startup_query_attempted": 0, "heldout_query_attempted": 0,
             "reused_queries": 0, "training_updates": 0, "new_teacher_queries": 0}
    deadline, switching = Deadline(6600), None
    save(out / "progress.json", state)
    try:
        relp = parent["source_relational_protocol"]
        teacher_root = Path(relp["clipping_root"]) / "results/teacher"
        teacher_receipt = read(teacher_root / "receipt.json")
        if (teacher_receipt.get("complete") is not True or teacher_receipt.get("queries") != 80
                or teacher_receipt.get("case_ids") != list(range(3000, 3040))
                or teacher_receipt.get("seed_indices") != [0, 1]):
            raise ValueError("Original shared heldout80 teacher is incomplete")
        data_root, rows, cache, baseline, reference = clipping_reference.load_inputs(
            relp["source_clipping_protocol"], torch)
        del cache, baseline, reference
        switching = SwitchingRuntime(parent, out)
        binding = prepare_runtime(switching, parent, config, arm, out, state)
        state["deployment_binding"] = binding
        for case_id, seed_index in queries:
            if deadline.near(240):
                state.update(status="checkpointed", checkpointed=True)
                save(out / "summary.json", state)
                return 0
            row = rows[case_id]
            seed = row["sampler_seeds"][seed_index]
            observation, metadata = calibration_source.load_observation(np, data_root, row)
            datum = switching.datum(observation)
            target_path = teacher_root / f"{case_id}_{seed_index}.pt"
            target = torch.load(target_path, map_location="cpu", weights_only=True)
            if target.get("plan_row") != row or target.get("sampler_seed") != seed:
                raise ValueError("Original heldout teacher row/seed differs")
            expected = {"protocol": parent["protocol"], "unit": unit, "arm": arm,
                        "deployment_binding": binding, "split": "heldout",
                        "case_id": case_id, "seed_index": seed_index, "sampler_seed": seed,
                        "suite": row["suite"], "plan_row": row,
                        "description": metadata["description"],
                        "input_identity": datum_identity(torch, datum),
                        "teacher_path": str(target_path)}
            path = out / "records" / f"{case_id}_{seed_index}.pt"
            if path.is_file():
                validate_endpoint(torch, torch.load(path, map_location="cpu", weights_only=True), expected)
                state["reused_queries"] += 1
            else:
                state["heldout_query_attempted"] += 1
                state["current_case"] = [case_id, seed_index]
                save(out / "progress.json", state)
                action, future, counts, noise = switching.query(datum, seed, metadata["description"])
                payload = {**expected, "producer_pbs_jobid": job,
                           "action": action, "future": future, "counts": counts,
                           "action_initial_noise": noise,
                           "commands": torch.as_tensor(switching.runtime.command(action.clone())).float().cpu(),
                           "teacher_commands": torch.as_tensor(
                               switching.runtime.command(target["action"].clone())).float().cpu()}
                validate_endpoint(torch, payload, expected)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_torch_save(torch, path, payload)
            state["completed_queries"] += 1
            save(out / "progress.json", state)
        state.update(complete=True, status="complete", checkpointed=False)
        save(out / "summary.json", state)
        return 0
    except Exception as exc:
        state.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        save(out / "summary.json", state)
        raise
    finally:
        if switching is not None:
            switching.close()


def aggregate(unit, arm):
    import torch
    import latent_reference
    from aggregate import completed, endpoint_metrics, loss_definition
    from metrics_reference import metrics

    here = Path(__file__).resolve().parent
    parent, config = read(here / "protocol.json"), read(here / "training_protocol.json")
    root = Path(parent["remote_root"])
    relp = parent["source_relational_protocol"]
    receipts = {shard: completed(root, f"heldout_{arm}_{shard}") for shard in (0, 1)}
    binding = receipts[0]["deployment_binding"]
    if receipts[1]["deployment_binding"] != binding or arm not in config["arms"]:
        raise ValueError("Heldout shards do not bind the same trained deployment")
    stats = read(Path(relp["source_dual_root"]) / "results/fit_stats/summary.json")
    definition, records = loss_definition(parent), []
    teacher_root = Path(relp["clipping_root"]) / "results/teacher"
    for case, seed_index in heldout_queries(parent):
        source_unit = f"heldout_{arm}_{seed_index}"
        path = root / "results" / source_unit / "records" / f"{case}_{seed_index}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"protocol": parent["protocol"], "unit": source_unit, "arm": arm,
                    "deployment_binding": binding, "split": "heldout", "case_id": case,
                    "seed_index": seed_index,
                    "teacher_path": str(teacher_root / f"{case}_{seed_index}.pt")}
        validate_endpoint(torch, saved, expected)
        target = torch.load(expected["teacher_path"], map_location="cpu", weights_only=True)
        if (target.get("plan_row") != saved["plan_row"]
                or target.get("sampler_seed") != saved["sampler_seed"]):
            raise ValueError("Shared teacher row or sampler seed differs during aggregation")
        traditional = metrics(torch, latent_reference, saved["action"], saved["future"],
                              target, relp["training"])
        common = endpoint_metrics(torch, saved, {**target, "commands": saved["teacher_commands"]},
                                  stats, definition)
        records.append({"case_id": case, "seed_index": seed_index, "suite": saved["suite"],
                        **traditional, **{key: common[key] for key in METRICS if key in common}})
    result = {"protocol": parent["protocol"], "unit": unit, "arm": arm,
              "pbs_jobid": os.environ["PBS_JOBID"], "complete": True,
              "deployment_binding": binding,
              "source_jobs": {f"heldout_{arm}_{s}": row["pbs_jobid"] for s, row in receipts.items()},
              "training_updates": 0, "new_model_queries": 0, "new_teacher_queries": 0,
              "scope": "original fixed-input heldout80 and original common metrics; no closed-loop claim",
              **aggregate_records(parent, records)}
    save(root / "results" / unit / "details.json", {"records": records})
    save(root / "results" / unit / "summary.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    aggregate_match = re.fullmatch(r"aggregate_heldout_([a-z][a-z0-9_]{0,25})", args.unit)
    if aggregate_match:
        guard()
        return aggregate(args.unit, aggregate_match[1])
    match = re.fullmatch(r"heldout_([a-z][a-z0-9_]{0,25})_([01])", args.unit)
    if not match:
        parser.error("Expected heldout_<training_arm>_<seed_index>")
    guard()
    return run(args.unit, match[1], int(match[2]))


if __name__ == "__main__":
    raise SystemExit(main())
