"""Prepare fixed teacher and initialization references inside a PBS CPU allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket


DIAGNOSTIC_PROTOCOL = "fastwam-vq-latent-diagnostic-v1"
LATENT_SHAPE = (1, 48, 3, 14, 28)
ACTION_SHAPE = (32, 7)


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cpu_allocation_guard() -> str:
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("A genuine PBS compute allocation is required for reference preparation")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")
    if os.environ.get("PBS_GPUFILE"):
        raise RuntimeError("Reference preparation must use a CPU-only PBS allocation")
    return host


def key(case_id: int, seed_index: int) -> str:
    return f"{case_id}:{seed_index}"


def action_error(torch, actual, reference) -> dict:
    delta = actual.to(dtype=torch.float32) - reference.to(dtype=torch.float32)
    return {"max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt())}


def _tensor(torch, value, shape, label):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise RuntimeError(f"{label} must have shape {tuple(shape)}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{label} must contain finite floating-point values")
    return value.detach().cpu().clone()


def _source_actions(torch, protocol, expected_contexts):
    source_root = Path(protocol["source_root"])
    source_dir = source_root / "results" / "full"
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("pbs_jobid") != protocol["source_full_job"]:
        raise RuntimeError("The declared source action run is not complete")

    actions = {"bf16": {}, protocol["initial_arm"]: {}}
    with (source_dir / "actions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            arm = row.get("arm")
            if arm not in actions:
                continue
            identity = row["case_id"], row["seed_index"]
            if identity in actions[arm]:
                raise RuntimeError(f"Duplicate {arm} source action at {identity}")
            action = torch.tensor(row["action"], dtype=torch.float32)
            actions[arm][identity] = {
                "action": _tensor(torch, action, ACTION_SHAPE, f"{arm} source action {identity}"),
                "sampler_seed": row.get("sampler_seed"),
            }

    for context in expected_contexts:
        identity = context["case_id"], context["seed_index"]
        if identity not in actions[protocol["initial_arm"]]:
            raise RuntimeError(f"Missing source initialization action at {identity}")
        if identity not in actions["bf16"] and context["split"] == "calibration":
            capture = torch.load(source_dir / "sensitivity" / f"case_{identity[0]}.pt",
                                 map_location="cpu", weights_only=True, mmap=True)
            baseline = _tensor(torch, capture["baseline_action"], ACTION_SHAPE,
                               f"BF16 sensitivity action {identity}")
            actions["bf16"][identity] = {"action": baseline, "sampler_seed": None}
            del capture
        if identity not in actions["bf16"]:
            raise RuntimeError(f"Missing BF16 source action at {identity}")

    old_plan = json.loads((Path(protocol["roots"]["old_inputs"]) / "plan.json").read_text(encoding="utf-8"))
    new_plan = json.loads((source_root / "plan.json").read_text(encoding="utf-8"))
    old_rows = {row["case_id"]: row for row in old_plan["inputs"]}
    new_rows = {row["case_id"]: row for row in new_plan["inputs"]}
    for context in expected_contexts:
        identity = context["case_id"], context["seed_index"]
        row = (old_rows if context["case_id"] < 1000 else new_rows)[context["case_id"]]
        seed = int(row["sampler_seeds"][context["seed_index"]])
        for arm in ("bf16", protocol["initial_arm"]):
            record = actions[arm][identity]
            if record["sampler_seed"] is None:
                record["sampler_seed"] = seed
            if int(record["sampler_seed"]) != seed:
                raise RuntimeError(f"{arm} sampler seed differs from the frozen plan at {identity}")
    return actions


def run(protocol_path: Path, output_root: Path) -> None:
    host = cpu_allocation_guard()
    import torch
    import metrics
    import latent_reference
    from test_latent_loss import self_check as latent_loss_self_check

    latent_loss_self_check(torch)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != "fastwam-vq-action-latent-v1":
        raise RuntimeError("Unexpected action-latent protocol")
    if protocol.get("latent_diagnostic_protocol", DIAGNOSTIC_PROTOCOL) != DIAGNOSTIC_PROTOCOL:
        raise RuntimeError("Unexpected latent diagnostic protocol")
    contexts = metrics.frozen_contexts(protocol)
    if len(contexts) != 60:
        raise RuntimeError("The frozen cohort must contain exactly 60 contexts")
    context_by_key = {key(row["case_id"], row["seed_index"]): row for row in contexts}
    action_sources = _source_actions(torch, protocol, contexts)

    diagnostic_root = Path(protocol["latent_diagnostic_root"])
    job_ids = protocol["latent_diagnostic_jobs"]
    arms = ("bf16", "parent_self", "parent_teacher_future", "distilled_self", "distilled_teacher_future")
    expected_cal = set(protocol["calibration_cases"])
    teacher, parent, matched = {}, {}, []
    max_abs_tol = float(protocol["parity_max_abs_tolerance"])
    rmse_tol = float(protocol["parity_rmse_tolerance"])
    source_parity = {}

    for shard in (0, 1):
        shard_root = diagnostic_root / "results" / f"shard-{shard}"
        summary = json.loads((shard_root / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("status") != "complete" or summary.get("protocol") != DIAGNOSTIC_PROTOCOL
                or summary.get("job_id") != job_ids[str(shard)] or summary.get("shard") != shard
                or summary.get("context_count") != 30 or summary.get("expected_context_count") != 30
                or summary.get("native_context_queries") != 90 or summary.get("teacher_future_queries") != 60
                or summary.get("target_linears") != 614 or summary.get("selected_epoch") != 2):
            raise RuntimeError(f"Latent diagnostic shard {shard} summary is incomplete or mismatched")

    for cache_key, context in context_by_key.items():
        shard = context["shard"]
        stem = f"case_{context['case_id']}_seed_{context['seed_index']}"
        shard_root = diagnostic_root / "results" / f"shard-{shard}"
        receipt = json.loads((shard_root / f"{stem}.json").read_text(encoding="utf-8"))
        if (receipt.get("protocol") != DIAGNOSTIC_PROTOCOL or receipt.get("context") != context
                or receipt.get("arms") != list(arms) or receipt.get("native_queries") != 3
                or receipt.get("teacher_future_queries") != 2):
            raise RuntimeError(f"Invalid diagnostic receipt for {cache_key}")
        payload = torch.load(shard_root / f"{stem}.pt", map_location="cpu", weights_only=True, mmap=True)
        expected_seed = action_sources["bf16"][(context["case_id"], context["seed_index"])]["sampler_seed"]
        if (payload.get("protocol") != DIAGNOSTIC_PROTOCOL or payload.get("context") != context
                or int(payload.get("sampler_seed", -1)) != int(expected_seed)):
            raise RuntimeError(f"Diagnostic input/seed identity differs for {cache_key}")
        captures = payload.get("captures")
        if not isinstance(captures, dict) or set(captures) != set(arms):
            raise RuntimeError(f"Diagnostic arm coverage differs for {cache_key}")

        for arm in arms:
            capture = captures[arm]
            action = _tensor(torch, capture.get("action"), ACTION_SHAPE, f"{arm} action {cache_key}")
            future = _tensor(torch, capture.get("video_latents"), LATENT_SHAPE, f"{arm} future {cache_key}")
            if arm == "bf16":
                parity = action_error(torch, action, action_sources["bf16"][(context["case_id"], context["seed_index"])]["action"])
                source_parity.setdefault("bf16", []).append(parity)
                if parity["max_abs"] > max_abs_tol or parity["rmse"] > rmse_tol:
                    raise RuntimeError(f"BF16 capture differs from original source action at {cache_key}: {parity}")
            elif arm == "parent_self":
                parent_reference = action_sources[protocol["initial_arm"]][(context["case_id"], context["seed_index"])]["action"]
                parity = action_error(torch, action, parent_reference)
                source_parity.setdefault("parent", []).append(parity)
                if parity["max_abs"] > max_abs_tol or parity["rmse"] > rmse_tol:
                    raise RuntimeError(f"Parent capture differs from original source action at {cache_key}: {parity}")

        identity = context["case_id"], context["seed_index"]
        bf16 = captures["bf16"]
        parent_self = captures["parent_self"]
        teacher[cache_key] = {
            "case_id": int(context["case_id"]), "seed_index": int(context["seed_index"]),
            "sampler_seed": int(expected_seed),
            "action": action_sources["bf16"][identity]["action"],
            "future": _tensor(torch, bf16["video_latents"], LATENT_SHAPE, f"teacher future {cache_key}"),
        }
        parent[cache_key] = {
            "case_id": int(context["case_id"]), "seed_index": int(context["seed_index"]),
            "sampler_seed": int(expected_seed),
            "action": _tensor(torch, parent_self["action"], ACTION_SHAPE, f"parent action {cache_key}"),
            "future": _tensor(torch, parent_self["video_latents"], LATENT_SHAPE, f"parent future {cache_key}"),
        }
        matched.append({"case_id": int(context["case_id"]), "seed_index": int(context["seed_index"]),
                        "sampler_seed": int(expected_seed), "shard": int(shard), "split": context["split"]})
        del payload, captures

    calibration = [teacher[key(case, 0)] for case in protocol["calibration_cases"]]
    parent_calibration = [parent[key(case, 0)] for case in protocol["calibration_cases"]]
    calibration_rows = []
    for target, initial in zip(calibration, parent_calibration):
        action_delta = initial["action"][:10, :6].float() - target["action"][:10, :6].float()
        calibration_rows.append({
            "case_id": initial["case_id"], "seed_index": initial["seed_index"],
            "split": "calibration", "action_mse": float(action_delta.square().mean()),
            "future_nmse": metrics.latent_metrics(torch, initial["future"], target["future"])["future_latent_nmse"],
        })
    fraction = float(protocol["latent_loss_fraction"])
    loss_scale = latent_reference.calibration_loss_scale(
        calibration_rows, protocol["calibration_cases"], fraction)
    mean_action_mse = loss_scale["mean_cal_initial_action_mse"]
    mean_future_nmse = loss_scale["mean_cal_initial_future_nmse"]
    latent_weight = loss_scale["latent_weight"]

    selection_initial_metrics = {}
    for context in contexts:
        if context["split"] != "selection":
            continue
        cache_key = key(context["case_id"], context["seed_index"])
        selection_initial_metrics[cache_key] = metrics.latent_metrics(
            torch, parent[cache_key]["future"], teacher[cache_key]["future"])

    cache = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "source_root": protocol["source_root"],
        "source_full_job": protocol["source_full_job"],
        "initial_arm": protocol["initial_arm"],
        "calibration_ids": list(protocol["calibration_cases"]),
        "teacher": teacher,
        "parent": parent,
        "initial_selection_future_metrics": selection_initial_metrics,
        "latent_loss": {
            **loss_scale,
        },
    }
    cache_path, summary_path = Path(protocol["reference_cache"]), Path(protocol["reference_summary"])
    output_root = output_root.resolve()
    if cache_path.parent.resolve() != output_root or summary_path.parent.resolve() != output_root:
        raise RuntimeError("Reference outputs must be written directly under the declared output root")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, temporary)
    temporary.replace(cache_path)
    summary = {
        "status": "complete", "protocol": protocol["protocol"], "job_id": os.environ["PBS_JOBID"],
        "host": host, "latent_diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "latent_diagnostic_jobs": job_ids, "context_count": len(teacher),
        "calibration_context_count": len(calibration), "parent_context_count": len(parent),
        "model_forward_count": 0, "target_linears": 614,
        "latent_loss_self_check": "passed",
        "matched_contexts": matched,
        "action_source_parity": {
            arm: {"contexts": len(records), "max_abs": max(r["max_abs"] for r in records),
                  "max_context_rmse": max(r["rmse"] for r in records)}
            for arm, records in source_parity.items()
        },
        "latent_weight_policy": cache["latent_loss"]["policy"],
        "latent_loss_fraction": fraction, "latent_weight": latent_weight,
        "mean_cal_initial_action_mse": mean_action_mse,
        "mean_cal_initial_future_nmse": mean_future_nmse,
        "calibration_ids": list(protocol["calibration_cases"]),
        "reference_cache": str(cache_path),
    }
    save_json(summary_path, summary)
    print("LATENT_REFERENCES_READY " + json.dumps({key: summary[key] for key in
          ("job_id", "context_count", "calibration_context_count", "model_forward_count", "latent_weight")},
          allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.out)
