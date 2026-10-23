"""Evaluate frozen final C80 checkpoints under every explicitly configured loss."""
import os
import math
from pathlib import Path
import statistics

from control_common import guard, read, save
from training_objective import validate_config


RAW_KEYS = ("action_absolute_kl", "action_response_kl", "latent_absolute_kl",
            "control_kl_clean", "control_kl_noisy")
METRIC_KEYS = ("clean_motor_rmse", "noisy_motor_rmse", "clean_future_nmse", "noisy_future_nmse")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def compose(raw, config):
    weights = validate_config(config)
    if any(not math.isfinite(raw[key]) or raw[key] < 0 for key in RAW_KEYS):
        raise ValueError("Raw objective terms must be finite and nonnegative")
    return (weights["action_absolute_weight"] * raw["action_absolute_kl"]
            + weights["response_weight"] * raw["action_response_kl"]
            + weights["lambda_latent"] * (
                weights["latent_absolute_weight"] * raw["latent_absolute_kl"]
                + weights["control_weight"] * 0.5 * (
                    raw["control_kl_clean"] + raw["control_kl_noisy"])))


def summarize(records, loss_configs):
    if not isinstance(loss_configs, dict) or not loss_configs:
        raise ValueError("Explicit frozen loss configurations are required")
    ids = [row["case_id"] for row in records]
    if len(ids) != 80 or len(set(ids)) != 80 or set(ids) != set(range(2000, 2080)):
        raise ValueError("Final C80 summary requires exactly the original 80 contexts")
    if any(not math.isfinite(row[key]) or row[key] < 0
           for row in records for key in (*RAW_KEYS, *METRIC_KEYS)):
        raise ValueError("Final C80 metrics must be finite and nonnegative")
    by_suite = {}
    for suite in SUITES:
        selected = [row for row in records if row["suite"] == suite]
        if len(selected) != 20:
            raise ValueError("Final C80 requires twenty contexts in each original suite")
        by_suite[suite] = {key: statistics.fmean(row[key] for row in selected)
                           for key in (*RAW_KEYS, *METRIC_KEYS)}
        by_suite[suite]["objective_by_loss_config"] = {
            name: statistics.fmean(compose(row, config) for row in selected)
            for name, config in loss_configs.items()}
    return {
        "cases": 80, "views_per_case": 2,
        "aggregation": "mean twenty cases per suite, mean four suites",
        "by_suite": by_suite,
        "overall": {
            **{key: statistics.fmean(by_suite[suite][key] for suite in SUITES)
               for key in (*RAW_KEYS, *METRIC_KEYS)},
            "objective_by_loss_config": {
                name: statistics.fmean(by_suite[suite]["objective_by_loss_config"][name]
                                       for suite in SUITES) for name in loss_configs},
        },
    }


def main():
    guard()
    import torch
    import latent_reference
    from aggregate import completed
    from final_c80_eval import validate_record
    from training_objective import paired_endpoint_objective, weighted_reader_objective

    here = Path(__file__).resolve().parent
    parent, config = read(here / "protocol.json"), read(here / "training_protocol.json")
    root = Path(parent["remote_root"])
    definition = read(root / config["data_config"]["dataset_subdir"] / "definition.json")
    cases = {case["case_id"]: case for case in definition["cases"]}
    stats = read(Path(parent["source_relational_protocol"]["source_dual_root"])
                 / "results/fit_stats/summary.json")
    loss_configs = config["arms"]
    if "mse_reference" in loss_configs:
        raise ValueError("mse_reference is reserved for the original frozen checkpoint")
    checkpoints, source_jobs, details = {}, {}, {}
    for arm in ["mse_reference", *loss_configs]:
        receipts = {shard: completed(root, f"final_c80_{arm}_{shard}") for shard in (0, 1)}
        binding = receipts[0]["deployment_binding"]
        if (receipts[1]["deployment_binding"] != binding
                or any(receipt.get("training_protocol") != config for receipt in receipts.values())):
            raise ValueError("Final C80 shards must bind the same checkpoint and training inputs")
        rows = []
        for case_id in range(2000, 2080):
            case = cases[case_id]
            shard = int(case["shard"])
            unit = f"final_c80_{arm}_{shard}"
            receipt = receipts[shard]
            expected_ids = sorted(c["case_id"] for c in cases.values() if c["shard"] == shard)
            if (receipt.get("unit") != unit or receipt.get("completed_case_ids") != expected_ids
                    or receipt.get("expected_pairs") != 40 or receipt.get("completed_pairs") != 40):
                raise ValueError("Final C80 worker coverage differs")
            saved = torch.load(root / "results" / unit / "records" / f"{case_id}.pt",
                               map_location="cpu", weights_only=True)
            expected = {"protocol": parent["protocol"], "unit": unit, "arm": arm,
                        "case_id": case_id, "suite": case["suite"], "deployment_binding": binding}
            validate_record(torch, saved, expected)
            targets = {}
            for variant in ("clean", "noisy"):
                target = torch.load(root / "results/training_teacher/records" / f"{case_id}_bf16_{variant}.pt",
                                    map_location="cpu", weights_only=True)
                metadata = read(root / "results/training_teacher/records" / f"{case_id}_bf16_{variant}.json")
                expected_input = target["input_identity"]
                if (saved["input_identities"][variant] != expected_input
                        or saved["sampler_seed"] != target["seed"]
                        or saved["description"] != target["description"]
                        or saved["teacher_bindings"][variant] != {
                            "binding": {key: target[key] for key in ("protocol", "unit", "index", "arm", "variant", "seed", "description", "input_identity")},
                            **{key: metadata[key] for key in ("producer_job", "source_calibration_job", "source_file", "source_kind", "source_eligibility")}}):
                    raise ValueError("Final C80 training-teacher input/provenance binding differs")
                targets[variant] = target
            endpoints = saved["endpoints"]
            # Raw KL values are independent of this explicit config's weights.
            metric_config = next(iter(loss_configs.values()))
            _, raw = paired_endpoint_objective(
                torch, {v: endpoints[v]["action"] for v in targets},
                {v: endpoints[v]["future"] for v in targets}, targets, stats, metric_config)
            recalculated = {key: float(value) for key, value in raw.items()}
            metrics = {}
            for variant in targets:
                _, control = weighted_reader_objective(
                    torch, saved["reader_actions"][variant], targets[variant]["action"], stats, metric_config)
                recalculated["control_kl_" + variant] = float(control)
                error = endpoints[variant]["action"].float()[:10, :6] - targets[variant]["action"].float()[:10, :6]
                metrics[variant + "_motor_rmse"] = float(error.square().mean().sqrt())
                metrics[variant + "_future_nmse"] = float(latent_reference.future_nmse_loss(
                    torch, endpoints[variant]["future"], targets[variant]["future"]))
            if any(abs(recalculated[key] - saved["raw_terms"][key]) > 1e-6 * max(1.0, abs(recalculated[key]))
                   for key in RAW_KEYS):
                raise ValueError("Saved final C80 raw loss does not reproduce from its endpoint tensors")
            rows.append({"case_id": case_id, "suite": case["suite"], **recalculated, **metrics})
        checkpoints[arm] = {"deployment_binding": binding, **summarize(rows, loss_configs)}
        details[arm] = rows
        source_jobs.update({f"final_c80_{arm}_{shard}": value["pbs_jobid"] for shard, value in receipts.items()})
    baseline = checkpoints["mse_reference"]["overall"]["objective_by_loss_config"]
    changes = {arm: {loss: {
        "delta": result["overall"]["objective_by_loss_config"][loss] - value,
        "percent": 100 * (result["overall"]["objective_by_loss_config"][loss] / value - 1) if value else None,
    } for loss, value in baseline.items()} for arm, result in checkpoints.items() if arm != "mse_reference"}
    out = root / "results/aggregate_final_c80"
    save(out / "details.json", details)
    save(out / "summary.json", {
        "protocol": parent["protocol"], "unit": "aggregate_final_c80",
        "pbs_jobid": os.environ["PBS_JOBID"], "complete": True, "training_protocol": config,
        "source_jobs": source_jobs, "checkpoints": checkpoints, "change_vs_mse": changes,
        "new_model_queries": 0, "training_updates": 0,
        "scope": "complete original C80 training views and shared teacher; no heldout or closed-loop claim"})


if __name__ == "__main__":
    main()
