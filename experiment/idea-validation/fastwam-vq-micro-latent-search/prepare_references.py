"""Build schema-2 BF16 teacher and B32 micro parent references in a guarded GPU allocation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import time


PROTOCOL = "fastwam-vq-micro-latent-search-v1"
SOURCE_PROTOCOL = "fastwam-vq-action-latent-v1"
MICRO_PROTOCOL = "fastwam-vq-microscale-v1"
ACTION_SHAPE = (32, 7)
LATENT_SHAPE = (1, 48, 3, 14, 28)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def allocation_guard() -> str:
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile or "login" in host.lower():
        raise RuntimeError("A genuine PBS compute allocation is required for micro references")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not present in PBS_NODEFILE")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("A GPU allocation is required for micro parent references")
    return host


def context_rows(protocol: dict, *, include_test: bool) -> list[dict]:
    rows = []
    for case in protocol["calibration_cases"]:
        rows.append({"split": "calibration", "case_id": int(case), "seed_index": 0})
    for case in protocol["selection_cases"]:
        for seed in protocol["evaluation_seed_indices"]:
            rows.append({"split": "selection", "case_id": int(case), "seed_index": int(seed)})
    if include_test:
        for case in protocol["test_cases"]:
            for seed in protocol["evaluation_seed_indices"]:
                rows.append({"split": "test", "case_id": int(case), "seed_index": int(seed)})
    return rows


def validate_reference_payload(torch, cache: dict, teacher_contexts: list[dict],
                               parent_contexts: list[dict]) -> None:
    """Validate schema and exact key/context coverage for both reference dictionaries."""
    if cache.get("schema_version") != 2 or cache.get("status") != "complete":
        raise RuntimeError("Reference cache must use complete schema_version=2")
    for field in ("protocol", "initial_arm", "bank_root", "bank_full_job", "source_root", "source_full_job"):
        if not cache.get(field):
            raise RuntimeError(f"Reference cache is missing {field}")
    if cache["protocol"] != PROTOCOL or cache["initial_arm"] != "micro_scale_vq_a4":
        raise RuntimeError("Reference cache protocol or initial arm differs")
    if (cache.get("teacher_context_count") != 60 or cache.get("parent_context_count") != 16
            or cache.get("calibration_context_count") != 8 or cache.get("selection_context_count") != 8
            or cache.get("model_forward_count") != 16 or cache.get("teacher_forward_count") != 0
            or cache.get("target_linears") != 614):
        raise RuntimeError("Reference cache counts differ from the frozen 60/16/8/8 contract")

    for name, expected in (("teacher", teacher_contexts), ("parent", parent_contexts)):
        records = cache.get(name)
        expected_by_key = {f"{row['case_id']}:{row['seed_index']}": row for row in expected}
        if not isinstance(records, dict) or set(records) != set(expected_by_key):
            raise RuntimeError(f"Reference {name} keys differ from the frozen cohort")
        for cache_key, context in expected_by_key.items():
            record = records[cache_key]
            if (record.get("case_id") != context["case_id"]
                    or record.get("seed_index") != context["seed_index"]
                    or type(record.get("sampler_seed")) is not int
                    or record["sampler_seed"] != context["sampler_seed"]):
                raise RuntimeError(f"Reference {name} identity differs at {cache_key}")
            for tensor_name, shape in (("action", ACTION_SHAPE), ("future", LATENT_SHAPE)):
                tensor = record.get(tensor_name)
                if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                        or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
                    raise RuntimeError(f"Reference {name} {tensor_name} is invalid at {cache_key}")

    normalization = cache.get("loss_normalization")
    required = {"policy", "normalization_c", "mean_cal_initial_action_mse",
                "mean_cal_initial_future_nmse", "calibration_ids"}
    if not isinstance(normalization, dict) or not required.issubset(normalization):
        raise RuntimeError("Reference cache is missing micro calibration normalization")
    if normalization["policy"] != (
            "normalization_c = mean_8cal_micro_initial_action_mse / mean_8cal_micro_initial_future_nmse"):
        raise RuntimeError("Reference cache uses an unexpected normalization policy")
    if (not math.isfinite(float(normalization["normalization_c"]))
            or float(normalization["normalization_c"]) <= 0
            or not math.isfinite(float(normalization["mean_cal_initial_action_mse"]))
            or float(normalization["mean_cal_initial_action_mse"]) < 0
            or not math.isfinite(float(normalization["mean_cal_initial_future_nmse"]))
            or float(normalization["mean_cal_initial_future_nmse"]) <= 0):
        raise RuntimeError("Reference cache normalization contains invalid values")


def _same_value(torch, left, right) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    return left == right


def _finite_tensor(torch, value, shape, label):
    if (not torch.is_tensor(value) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise RuntimeError(f"{label} must be a finite floating tensor with shape {tuple(shape)}")
    return value.detach().cpu().clone()


def _json_action_tensor(torch, value, label):
    """Decode the float32 JSON action representation used by the frozen micro run."""
    try:
        tensor = torch.tensor(value, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"{label} is not a numeric JSON action array") from exc
    return _finite_tensor(torch, tensor, ACTION_SHAPE, label)


def _load_protocol_inputs(protocol: dict, source, old_root: Path, source_root: Path):
    old_plan = json.loads((old_root / "plan.json").read_text(encoding="utf-8"))
    fresh_rows = json.loads((source_root / "plan.json").read_text(encoding="utf-8"))["inputs"]
    fresh = {row["case_id"]: row for row in fresh_rows}
    all_contexts = context_rows(protocol, include_test=True)
    rows = {}
    for context in all_contexts:
        case = context["case_id"]
        row = source.validate_plan(old_plan, case) if case < 1000 else fresh.get(case)
        if row is None:
            raise RuntimeError(f"Frozen plan is missing case {case}")
        seed_index = context["seed_index"]
        seeds = row.get("sampler_seeds")
        if not isinstance(seeds, list) or seed_index >= len(seeds) or type(seeds[seed_index]) is not int:
            raise RuntimeError(f"Frozen plan lacks sampler seed for {case}:{seed_index}")
        rows[(case, seed_index)] = {**context, "sampler_seed": int(seeds[seed_index]), "plan_row": row}
    return rows


def _validate_protocol(protocol: dict) -> None:
    if protocol.get("protocol") != PROTOCOL or protocol.get("initial_arm") != "micro_scale_vq_a4":
        raise RuntimeError("Unexpected micro latent-search protocol or initial arm")
    if protocol.get("target_linears") != 614:
        raise RuntimeError("Expected exactly 614 target Linears")
    expected = (("calibration_cases", 8), ("selection_cases", 4), ("test_cases", 22))
    for name, count in expected:
        cases = protocol.get(name)
        if not isinstance(cases, list) or len(cases) != count or len(set(cases)) != count:
            raise RuntimeError(f"{name} must contain exactly {count} unique cases")
    if protocol.get("calibration_seed_indices") != [0] or protocol.get("evaluation_seed_indices") != [0, 1]:
        raise RuntimeError("Frozen reference seed cohorts differ")
    for name in ("source_root", "source_full_job", "bank_root", "bank_full_job",
                 "teacher_reference_cache", "teacher_reference_summary", "reference_cache", "reference_summary"):
        if not protocol.get(name):
            raise RuntimeError(f"Protocol is missing {name}")


def _load_teacher(torch, protocol: dict, expected_rows: dict):
    source_root = Path(protocol["source_root"])
    source_full = source_root / "results" / "full"
    source_summary = json.loads((source_full / "summary.json").read_text(encoding="utf-8"))
    if source_summary.get("status") != "complete" or source_summary.get("pbs_jobid") != protocol["source_full_job"]:
        raise RuntimeError("The declared BF16 source run is not complete")

    summary_path = Path(protocol["teacher_reference_summary"])
    cache_path = Path(protocol["teacher_reference_cache"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("status") != "complete" or summary.get("protocol") != SOURCE_PROTOCOL
            or summary.get("reference_cache") != str(cache_path)
            or summary.get("context_count") != 60 or summary.get("model_forward_count") != 0
            or summary.get("target_linears") != 614):
        raise RuntimeError("The shared teacher reference summary is incomplete or mismatched")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    if (cache.get("schema_version") != 1 or cache.get("protocol") != SOURCE_PROTOCOL
            or cache.get("source_root") != protocol["source_root"]
            or cache.get("source_full_job") != protocol["source_full_job"]):
        raise RuntimeError("The shared teacher cache source identity differs")
    expected_keys = {f"{case}:{seed}" for case, seed in expected_rows}
    records = cache.get("teacher")
    if not isinstance(records, dict) or set(records) != expected_keys:
        raise RuntimeError("The shared teacher cache must cover exactly the frozen 60 contexts")
    teacher = {}
    for cache_key, row in records.items():
        context = expected_rows[(int(row["case_id"]), int(row["seed_index"]))]
        if (cache_key != f"{context['case_id']}:{context['seed_index']}"
                or type(row.get("sampler_seed")) is not int
                or row["sampler_seed"] != context["sampler_seed"]):
            raise RuntimeError(f"Teacher sampler identity differs at {cache_key}")
        teacher[cache_key] = {
            "case_id": context["case_id"], "seed_index": context["seed_index"],
            "sampler_seed": context["sampler_seed"],
            "action": _finite_tensor(torch, row.get("action"), ACTION_SHAPE, f"teacher action {cache_key}"),
            "future": _finite_tensor(torch, row.get("future"), LATENT_SHAPE, f"teacher future {cache_key}"),
        }
    return teacher


def _load_micro_source(torch, protocol: dict, expected_rows: dict):
    bank_root = Path(protocol["bank_root"])
    run_root = bank_root / "results" / "full"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "complete" or summary.get("protocol") != MICRO_PROTOCOL
            or summary.get("job_id") != protocol["bank_full_job"]
            or summary.get("source_full_job") != protocol["source_full_job"]
            or summary.get("fitted_linears") != 614 or summary.get("new_queries") != 52):
        raise RuntimeError("The declared micro bank run is incomplete or mismatched")

    actions = {}
    with (run_root / "actions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("arm") != protocol["initial_arm"]:
                continue
            identity = (int(row["case_id"]), int(row["seed_index"]))
            if identity in actions:
                raise RuntimeError(f"Duplicate cached micro action at {identity}")
            expected = expected_rows.get(identity)
            if (expected is None or row.get("split") != expected["split"]
                    or type(row.get("sampler_seed")) is not int
                    or row["sampler_seed"] != expected["sampler_seed"]):
                raise RuntimeError(f"Cached micro action identity differs at {identity}")
            actions[identity] = _json_action_tensor(torch, row.get("action"),
                                                    f"cached micro action {identity}")
    expected_actions = {(case, seed) for case in protocol["selection_cases"] + protocol["test_cases"]
                        for seed in protocol["evaluation_seed_indices"]}
    if set(actions) != expected_actions:
        raise RuntimeError("Cached micro actions must cover the original selection8/test44 cohort")
    return run_root, actions


def _load_capture(torch, path: Path, protocol: dict, context: dict):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema_version") != 1 or payload.get("protocol") != PROTOCOL
            or payload.get("initial_arm") != protocol["initial_arm"]
            or payload.get("bank_root") != protocol["bank_root"]
            or payload.get("bank_full_job") != protocol["bank_full_job"]
            or payload.get("source_root") != protocol["source_root"]
            or payload.get("source_full_job") != protocol["source_full_job"]):
        raise RuntimeError(f"Parent capture provenance differs: {path.name}")
    for name in ("case_id", "seed_index", "sampler_seed", "split"):
        if payload.get(name) != context[name]:
            raise RuntimeError(f"Parent capture {name} differs: {path.name}")
    return {
        "case_id": context["case_id"], "seed_index": context["seed_index"],
        "sampler_seed": context["sampler_seed"],
        "action": _finite_tensor(torch, payload.get("action"), ACTION_SHAPE, f"parent action {path.name}"),
        "future": _finite_tensor(torch, payload.get("future"), LATENT_SHAPE, f"parent future {path.name}"),
    }


def _capture_path(root: Path, context: dict) -> Path:
    return root / "reference_progress" / "parent" / f"case_{context['case_id']}_seed_{context['seed_index']}.pt"


def _save_parent(torch, path: Path, protocol: dict, context: dict, action, future,
                 elapsed_seconds: float) -> None:
    from checkpointing import atomic_torch_save

    payload = {
        "schema_version": 1, "protocol": PROTOCOL, "initial_arm": protocol["initial_arm"],
        "bank_root": protocol["bank_root"], "bank_full_job": protocol["bank_full_job"],
        "source_root": protocol["source_root"], "source_full_job": protocol["source_full_job"],
        **{name: context[name] for name in ("split", "case_id", "seed_index", "sampler_seed")},
        "action": action.detach().cpu().clone(), "future": future.detach().cpu().clone(),
        "forward_seconds": float(elapsed_seconds),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(torch, path, payload)


def _calibration_normalization(torch, metrics, latent_reference, protocol, teacher, parent):
    rows = []
    for case in protocol["calibration_cases"]:
        cache_key = f"{case}:0"
        target, initial = teacher[cache_key], parent[cache_key]
        action_mse = float((initial["action"][:10, :6].float() - target["action"][:10, :6].float()).square().mean())
        future_nmse = metrics.latent_metrics(torch, initial["future"], target["future"])["future_latent_nmse"]
        rows.append({"case_id": int(case), "seed_index": 0, "split": "calibration",
                     "action_mse": action_mse, "future_nmse": future_nmse})
    scale = latent_reference.calibration_loss_scale(rows, protocol["calibration_cases"], 1.0)
    return {name: scale[name] for name in (
        "policy", "normalization_c", "mean_cal_initial_action_mse",
        "mean_cal_initial_future_nmse", "calibration_ids")}


def _selection_parity(torch, protocol, parent, old_micro_actions):
    rows = []
    for case in protocol["selection_cases"]:
        for seed in protocol["evaluation_seed_indices"]:
            cache_key = f"{case}:{seed}"
            actual = parent[cache_key]["action"].float()
            reference = old_micro_actions[(case, seed)].float()
            delta = actual - reference
            record = {"case_id": case, "seed_index": seed,
                      "max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt())}
            record["passed"] = (record["max_abs"] <= float(protocol["parity_max_abs_tolerance"])
                                 and record["rmse"] <= float(protocol["parity_rmse_tolerance"]))
            rows.append(record)
    return {
        "contexts": len(rows), "passed": len(rows) == 8 and all(row["passed"] for row in rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "max_context_rmse": max(row["rmse"] for row in rows), "records": rows,
    }


def _base_record(protocol: dict, *, job_id: str, host: str, status: str,
                 parent_count: int, self_check: str,
                 calibration_count: int = 8, selection_count: int = 8) -> dict:
    return {
        "status": status, "protocol": protocol["protocol"], "job_id": job_id, "host": host,
        "initial_arm": protocol["initial_arm"], "bank_root": protocol["bank_root"],
        "bank_full_job": protocol["bank_full_job"], "source_root": protocol["source_root"],
        "source_full_job": protocol["source_full_job"], "teacher_context_count": 60,
        "parent_context_count": parent_count, "expected_parent_context_count": 16,
        "calibration_context_count": calibration_count, "selection_context_count": selection_count,
        "model_forward_count": parent_count, "teacher_forward_count": 0,
        "target_linears": 614, "self_check": self_check, "self_check_passed": self_check == "passed",
        "reference_cache": protocol["reference_cache"],
    }


def run(protocol_path: Path, output_root: Path) -> None:
    host = allocation_guard()
    import torch
    import metrics
    import latent_reference
    from test_micro_reference import self_check

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA GPU is required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    self_check(torch, protocol)
    output_root = output_root.resolve()
    cache_path = Path(protocol["reference_cache"])
    summary_path = Path(protocol["reference_summary"])
    if cache_path.parent.resolve() != output_root or summary_path.parent.resolve() != output_root:
        raise RuntimeError("Reference outputs must be written directly under the declared ROOT")
    if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "complete":
        raise FileExistsError("Reference cache is already complete; refusing to overwrite it")

    progress_path = output_root / "reference_progress.json"
    save_json(progress_path, {"status": "running", "protocol": PROTOCOL,
                             "job_id": os.environ["PBS_JOBID"],
                             "completed_parent_contexts": 0, "expected_parent_contexts": 16,
                             "pending_parent_context_count": 16, "model_forward_count": 0,
                             "teacher_forward_count": 0})

    source_root = Path(protocol["source_root"])
    old_root = Path(protocol["roots"]["old_inputs"])
    import run_diagnostic as source

    expected_rows = _load_protocol_inputs(protocol, source, old_root, source_root)
    teacher_contexts = [{**expected_rows[(row["case_id"], row["seed_index"])]}
                        for row in context_rows(protocol, include_test=True)]
    parent_contexts = [{**expected_rows[(row["case_id"], row["seed_index"])]}
                       for row in context_rows(protocol, include_test=False)]
    teacher = _load_teacher(torch, protocol, expected_rows)
    micro_run_root, old_micro_actions = _load_micro_source(torch, protocol, expected_rows)

    parent = {}
    completed_durations = []
    for context in parent_contexts:
        path = _capture_path(output_root, context)
        if path.is_file():
            key = f"{context['case_id']}:{context['seed_index']}"
            parent[key] = _load_capture(torch, path, protocol, context)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(saved.get("forward_seconds"), (int, float)):
                completed_durations.append(float(saved["forward_seconds"]))

    artifact = Path(os.environ.get("ARTIFACTS", output_root / "artifacts" / os.environ["PBS_JOBID"]))
    artifact.mkdir(parents=True, exist_ok=True)
    save_json(progress_path, {"status": "running", "protocol": PROTOCOL,
                             "job_id": os.environ["PBS_JOBID"],
                             "completed_parent_contexts": len(parent), "expected_parent_contexts": 16,
                             "pending_parent_context_count": 16 - len(parent),
                             "model_forward_count": len(parent), "teacher_forward_count": 0})
    start_epoch = float(os.environ.get("FW_JOB_START_EPOCH", time.time()))
    allocation_elapsed = max(0.0, time.time() - start_epoch)
    soft_limit = int(os.environ.get("FW_SOFT_DEADLINE_SECONDS", min(int(protocol["walltime_seconds"]), 3300)))
    deadline = time.monotonic() + max(0.0, soft_limit - allocation_elapsed)
    interrupted = []
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupted.append(signum))
    runtime = banks = None
    try:
        pending = [context for context in parent_contexts
                   if f"{context['case_id']}:{context['seed_index']}" not in parent]
        if pending:
            import runner as pilot
            import run_experiment as original
            import micro_codebook as micro
            import sampler

            torch.backends.cuda.matmul.allow_tf32 = False
            runtime = pilot.Runtime(artifact)
            if "A100" not in torch.cuda.get_device_name(0):
                raise RuntimeError("Frozen micro parent parity requires the existing A100 hardware class")
            runtime.arm("bf16")
            banks = original.FrozenBanks(runtime, artifact, protocol)
            if len(banks.modules) != 614:
                raise RuntimeError(f"Expected 614 target Linears, found {len(banks.modules)}")
            bank_dir = micro_run_root / "banks" / protocol["initial_arm"]
            with torch.no_grad():
                for index, (name, module) in enumerate(banks.modules.items()):
                    bank = torch.load(bank_dir / f"{index:04d}.pt", map_location="cuda", weights_only=True)
                    if bank.get("module") != name or tuple(bank["encoded"].get("shape", ())) != tuple(module.weight.shape):
                        raise RuntimeError(f"Micro bank identity/shape differs at {index}: {name}")
                    for key, value in banks.transforms[name].items():
                        if key not in bank["transform"] or not _same_value(torch, value, bank["transform"][key]):
                            raise RuntimeError(f"Frozen transform differs from the micro bank: {name} {key}")
                    if bank["encoded"].get("format") != micro.FORMAT:
                        raise RuntimeError(f"Unexpected micro encoding at {name}")
                    module.weight.data = micro.decode_vq(bank["encoded"])
                    del bank
            banks.transformed = True

            data = {}

            def datum(context):
                case = context["case_id"]
                if case not in data:
                    row = context["plan_row"]
                    root = old_root if case < 1000 else source_root
                    observation, metadata = source.load_case(root, row)
                    data[case] = (runtime.datum(observation), metadata["description"])
                return data[case]

            for context in pending:
                remaining = deadline - time.monotonic()
                estimated = max(180.0, max(completed_durations[-3:], default=0.0) * 1.25)
                reserve = max(float(protocol.get("save_reserve_seconds", 90)), 90.0)
                if interrupted or remaining <= estimated + reserve:
                    break
                case, seed_index = context["case_id"], context["seed_index"]
                value, description = datum(context)
                runtime.description, runtime.current_arm = description, protocol["initial_arm"]
                banks.seen.clear()
                tick = time.monotonic()
                action, counts, future = sampler.differentiable_action(
                    runtime, value, context["sampler_seed"], checkpointed=False, return_future=True)
                if len(banks.seen) != 614 or counts != {"video_steps": 10, "action_steps": 10}:
                    raise RuntimeError(f"Micro parent forward did not cover all targets at {case}:{seed_index}")
                action = _finite_tensor(torch, action, ACTION_SHAPE, f"micro parent action {case}:{seed_index}")
                future = _finite_tensor(torch, future, LATENT_SHAPE, f"micro parent future {case}:{seed_index}")
                duration = time.monotonic() - tick
                key = f"{case}:{seed_index}"
                _save_parent(torch, _capture_path(output_root, context), protocol, context,
                             action, future, duration)
                parent[key] = {"case_id": case, "seed_index": seed_index,
                               "sampler_seed": context["sampler_seed"], "action": action, "future": future}
                completed_durations.append(duration)
                save_json(progress_path, {
                    "status": "running", "protocol": PROTOCOL, "job_id": os.environ["PBS_JOBID"],
                    "completed_parent_contexts": len(parent), "expected_parent_contexts": 16,
                    "pending_parent_context_count": 16 - len(parent),
                    "model_forward_count": len(parent), "teacher_forward_count": 0,
                    "last_completed": {"case_id": case, "seed_index": seed_index,
                                        "forward_seconds": duration},
                })
                print("MICRO_REFERENCE_PROGRESS " + json.dumps({
                    "job_id": os.environ["PBS_JOBID"], "completed_parent_contexts": len(parent),
                    "expected_parent_contexts": 16, "case_id": case, "seed_index": seed_index,
                    "forward_seconds": duration,
                }, allow_nan=False), flush=True)
                del action, future

        if len(parent) != 16:
            completed_keys = set(parent)
            calibration_count = sum(
                f"{context['case_id']}:{context['seed_index']}" in completed_keys
                for context in parent_contexts if context["split"] == "calibration")
            selection_count = sum(
                f"{context['case_id']}:{context['seed_index']}" in completed_keys
                for context in parent_contexts if context["split"] == "selection")
            summary = _base_record(protocol, job_id=os.environ["PBS_JOBID"], host=host,
                                   status="resumable", parent_count=len(parent), self_check="passed",
                                   calibration_count=calibration_count, selection_count=selection_count)
            summary.update({"teacher_forward_count": 0, "pending_parent_context_count": 16 - len(parent),
                            "continuation": "rerun prepare_references.py --protocol PATH --out ROOT",
                            "parent_progress_dir": str(output_root / "reference_progress" / "parent")})
            save_json(summary_path, summary)
            save_json(progress_path, {"status": "resumable", "protocol": PROTOCOL,
                                     "job_id": os.environ["PBS_JOBID"],
                                     "completed_parent_contexts": len(parent),
                                     "expected_parent_contexts": 16,
                                     "pending_parent_context_count": 16 - len(parent),
                                     "model_forward_count": len(parent), "teacher_forward_count": 0})
            print("MICRO_REFERENCES_RESUMABLE " + json.dumps({
                "job_id": os.environ["PBS_JOBID"], "completed_parent_contexts": len(parent),
                "expected_parent_contexts": 16,
            }), flush=True)
            return

        normalization = _calibration_normalization(torch, metrics, latent_reference,
                                                   protocol, teacher, parent)
        parity = _selection_parity(torch, protocol, parent, old_micro_actions)
        if not parity["passed"]:
            raise RuntimeError(f"Micro selection action parity failed: {parity}")
        selection_future_metrics = {}
        for context in parent_contexts:
            if context["split"] != "selection":
                continue
            cache_key = f"{context['case_id']}:{context['seed_index']}"
            selection_future_metrics[cache_key] = metrics.latent_metrics(
                torch, parent[cache_key]["future"], teacher[cache_key]["future"])

        cache = {
            "schema_version": 2, "status": "complete", "protocol": PROTOCOL,
            "initial_arm": protocol["initial_arm"], "bank_root": protocol["bank_root"],
            "bank_full_job": protocol["bank_full_job"], "source_root": protocol["source_root"],
            "source_full_job": protocol["source_full_job"], "teacher_context_count": 60,
            "parent_context_count": 16, "calibration_context_count": 8,
            "selection_context_count": 8, "model_forward_count": 16,
            "teacher_forward_count": 0, "target_linears": 614,
            "loss_normalization": normalization, "teacher": teacher, "parent": parent,
            "initial_selection_future_metrics": selection_future_metrics,
        }
        validate_reference_payload(torch, cache, teacher_contexts, parent_contexts)

        from checkpointing import atomic_torch_save
        atomic_torch_save(torch, cache_path, cache)
        summary = _base_record(protocol, job_id=os.environ["PBS_JOBID"], host=host,
                               status="complete", parent_count=16, self_check="passed")
        summary.update({
            "loss_normalization": normalization, "selection_cache_parity": True,
            "selection_cache_parity_metrics": parity, "initial_selection_future_metrics": selection_future_metrics,
            "teacher_forward_count": 0, "reference_cache": str(cache_path),
        })
        save_json(summary_path, summary)
        save_json(progress_path, {"status": "complete", "protocol": PROTOCOL,
                                 "job_id": os.environ["PBS_JOBID"], "completed_parent_contexts": 16,
                                 "expected_parent_contexts": 16, "model_forward_count": 16,
                                 "teacher_forward_count": 0})
        print("MICRO_REFERENCES_READY " + json.dumps({
            "job_id": summary["job_id"], "teacher_context_count": 60,
            "parent_context_count": 16, "model_forward_count": 16,
            "normalization_c": normalization["normalization_c"],
            "selection_cache_parity": True,
        }, allow_nan=False), flush=True)
    finally:
        if banks is not None:
            banks.close()
        if runtime is not None and runtime.env is not None:
            runtime.env.close()
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Shared campaign ROOT; completed parent captures are resumed here")
    args = parser.parse_args()
    run(args.protocol, args.out)
