"""Paired fixed-input, five-NFE timing for one Cosmos arm per PBS process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)]
    return {"count": len(ordered), "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered), "p95": p95, "max": ordered[-1]}


def validate_arm(p: dict, arm: str) -> None:
    if arm not in p["arms"]:
        raise ValueError(f"Arm is absent from the frozen protocol: {arm}")
    if p.get("mode") != "libero_direct_policy" or p["inference"].get("denoising_steps") != 5:
        raise ValueError("The benchmark is frozen to direct-policy five-NFE Cosmos inference")
    panel = p["performance"]
    cases, seeds = panel["case_ids"], panel["seed_indices"]
    expected = len(cases) * len(seeds)
    if (cases != list(range(3000, 3040)) or seeds != [0, 1]
            or panel.get("queries") != expected or panel.get("repetitions") != 3
            or panel.get("blocks") != 3 or panel.get("warmup_queries") != 3):
        raise ValueError("Performance panel or repetition counts differ from the frozen 40 x 2 x 3 x 3 design")


def run_one(common, runtime, context, *, torch, device) -> dict:
    import endpoint as legacy_endpoint

    sampler = legacy_endpoint._sampler_module(runtime.model)
    original = sampler.forward
    interval = {"calls": 0, "seconds": None}

    def timed_forward(*args, **kwargs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        interval["calls"] += 1
        try:
            return original(*args, **kwargs)
        finally:
            torch.cuda.synchronize(device)
            interval["seconds"] = time.perf_counter() - start

    sampler.forward = timed_forward
    torch.cuda.synchronize(device)
    complete_start = time.perf_counter()
    try:
        result = common.endpoint(runtime, context)
        torch.cuda.synchronize(device)
    finally:
        sampler.forward = original
    full_query_seconds = time.perf_counter() - complete_start
    expected_nfe = int(runtime.protocol["inference"]["denoising_steps"])
    if interval["calls"] != 1 or interval["seconds"] is None:
        raise RuntimeError(f"Expected one complete sampler.forward interval, observed {interval['calls']}")
    if int(result.sampler_evaluations) != expected_nfe:
        raise RuntimeError(f"Sampler completed {result.sampler_evaluations} NFE, expected {expected_nfe}")
    if int(result.denoiser_calls) != expected_nfe:
        raise RuntimeError('Actual denoiser calls differ from the frozen five-NFE path')
    if not math.isfinite(interval["seconds"]) or interval["seconds"] <= 0:
        raise RuntimeError("Invalid synchronized denoising wall interval")
    return {"denoising_seconds": float(interval["seconds"]),
            "complete_endpoint_query_seconds": float(full_query_seconds),
            "sampler_forward_calls": interval["calls"],
            "sampler_evaluations": int(result.sampler_evaluations),
            "denoiser_calls": int(result.denoiser_calls)}


def sample_path(out: Path, case_id: int, seed_index: int, block: int, repetition: int) -> Path:
    return out / "samples" / f"{case_id}_{seed_index}_b{block}_r{repetition}.json"


def load_sample(path: Path, expected: dict) -> dict:
    sample = read_json(path)
    for key, value in expected.items():
        if sample.get(key) != value:
            raise RuntimeError(f"Saved performance sample binding differs at {key}: {path}")
    return sample


def aggregate_completed_panel(p: dict, root: Path) -> dict | None:
    arms = p["arms"]
    if len(arms) != 6:
        raise ValueError("The frozen performance panel must contain six arms")
    summaries = {}
    expected_samples = (len(p["performance"]["case_ids"]) * len(p["performance"]["seed_indices"])
                        * p["performance"]["blocks"] * p["performance"]["repetitions"])
    for arm in arms:
        path = root / "results" / "performance" / arm / "summary.json"
        if not path.is_file():
            return None
        item = read_json(path)
        if (item.get("protocol") != p["protocol"] or item.get("arm") != arm
                or item.get("status") != "complete"
                or item.get("coverage", {}).get("samples") != expected_samples
                or item.get("coverage", {}).get("expected_samples") != expected_samples
                or item.get("timing_scope") != p["performance"]["latency_scope"]):
            return None
        if arm != "bf16" and item.get("runtime_label", {}).get("native_int4_kernel_used") is not False:
            raise RuntimeError(f"Low-bit arm lacks its numerical execution boundary: {arm}")
        median = item.get("denoising_wall_seconds", {}).get("median")
        if isinstance(median, bool) or not isinstance(median, (int, float)) or not math.isfinite(median) or median <= 0:
            raise RuntimeError(f"Invalid denoising median for performance arm {arm}")
        summaries[arm] = item
    hardware = summaries['bf16'].get('hardware')
    if not hardware or any(item.get('hardware') != hardware for item in summaries.values()):
        raise RuntimeError('Six-arm speedup requires matching GPU model and runtime versions')
    reference = float(summaries["bf16"]["denoising_wall_seconds"]["mean"])
    speedups = {arm: reference / float(item["denoising_wall_seconds"]["mean"])
                for arm, item in summaries.items()}
    panel = {
        "protocol": p["protocol"], "status": "complete", "reference_arm": "bf16",
        "model": p["model"], "mode": p["mode"],
        "hardware": hardware,
        "timing_scope": p["performance"]["latency_scope"],
        "query_design": {"case_ids": p["performance"]["case_ids"],
                         "seed_indices": p["performance"]["seed_indices"],
                         "blocks": p["performance"]["blocks"],
                         "repetitions": p["performance"]["repetitions"]},
        "arms": {arm: {"denoising_wall_seconds": item["denoising_wall_seconds"],
                       "complete_endpoint_query_seconds": item["complete_endpoint_query_seconds"],
                       "peak_allocated_gb_decimal": item["peak_allocated_gb_decimal"],
                       "packed_storage_bytes": item["packed_storage_bytes"],
                       "packed_storage_gib_binary": item["packed_storage_gib_binary"],
                       "runtime_label": item["runtime_label"],
                       "coverage": item["coverage"],
                       "speedup_over_bf16": speedups[arm]}
                for arm, item in summaries.items()},
        "execution_boundary": p["execution_boundary"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(root / "results" / "performance" / "summary.json", panel)
    for arm, item in summaries.items():
        item["speedup_over_bf16"] = speedups[arm]
        item["speedup_status"] = "aggregated_after_complete_six-arm-panel"
        atomic_json(root / "results" / "performance" / arm / "summary.json", item)
    return panel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--soft-deadline-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    p = read_json(args.protocol)
    validate_arm(p, args.arm)
    if args.soft_deadline_seconds <= 0:
        raise ValueError("A positive soft deadline is required")

    import common
    import runtime as arm_runtime
    torch = common.torch_module()
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = common.prepare_runtime(p, args.root, args.out)
    try:
        from validation import ValidationCheckpointed
        try:
            metadata = arm_runtime.install_arm(runtime, p, args.root, args.arm,
                                               optimized=(args.arm == "ours_vqwam"))
        except ValidationCheckpointed:
            atomic_json(args.out / 'summary.json', {
                'protocol': p['protocol'], 'arm': args.arm, 'status': 'checkpointed',
                'pbs_job_id': os.environ['PBS_JOBID'], 'phase': 'engine_validation',
                'complete': False})
            return 2
        if not isinstance(metadata, dict):
            raise TypeError("runtime.install_arm must return a metadata mapping")
        storage_bytes = metadata.get("packed_storage_bytes")
        if isinstance(storage_bytes, bool) or not isinstance(storage_bytes, int) or storage_bytes <= 0:
            raise RuntimeError("runtime.install_arm lacks packed_storage_bytes; refusing to infer packed storage from a checkpoint")
        execution_scope = metadata.get("execution_scope") or metadata.get("execution")
        if not isinstance(execution_scope, str) or not execution_scope.strip():
            raise RuntimeError("runtime.install_arm must label the measured execution path")
        if args.arm != "bf16" and metadata.get("native_int4_kernel_used") is not False:
            raise RuntimeError("Low-bit numerical benchmark requires an explicit native_int4_kernel_used=false receipt")
        common.save_json(args.out / "install_metadata.json", metadata)

        device = next(runtime.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"Expected a CUDA runtime model, found {device}")
        props = torch.cuda.get_device_properties(device)
        hardware = {'gpu_name': props.name, 'total_memory_bytes': props.total_memory,
                    'capability': [props.major, props.minor], 'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda}
        cases = [int(case) for case in p["performance"]["case_ids"]]
        seeds = [int(seed) for seed in p["performance"]["seed_indices"]]
        contexts = []
        for case_id in cases:
            for seed_index in seeds:
                context = common.get_context(runtime, case_id, seed_index=seed_index)
                if int(context.get("case_id", -1)) != case_id or int(context.get("seed_index", -1)) != seed_index:
                    raise RuntimeError("Prepared performance context has a different frozen key")
                contexts.append((case_id, seed_index, context))

        try:
            allocation_start = float(os.environ["COSMOS_JOB_START_EPOCH"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError("COSMOS_JOB_START_EPOCH is required for the allocation soft deadline") from exc
        remaining = args.soft_deadline_seconds - (time.time() - allocation_start)
        deadline = time.monotonic() + max(0.0, remaining)
        warmup_count = int(p["performance"]["warmup_queries"])
        warmups = []
        for case_id, seed_index, context in contexts[:warmup_count]:
            if time.monotonic() + 120 >= deadline:
                break
            warmups.append({"case_id": case_id, "seed_index": seed_index,
                            **run_one(common, runtime, context, torch=torch, device=device)})

        blocks = int(p["performance"]["blocks"])
        repetitions = int(p["performance"]["repetitions"])
        expected_samples = {}
        for block in range(blocks):
            for repetition in range(repetitions):
                for case_id, seed_index, _ in contexts:
                    path = sample_path(args.out, case_id, seed_index, block, repetition)
                    expected_samples[path] = {"protocol": p["protocol"], "arm": args.arm,
                                              "case_id": case_id, "seed_index": seed_index,
                                              "block": block, "repetition": repetition,
                                              "hardware": hardware}
        resumed_samples = 0
        for path, expected in expected_samples.items():
            if path.is_file():
                load_sample(path, expected)
                resumed_samples += 1
        if len(warmups) == warmup_count:
            for block in range(blocks):
                for repetition in range(repetitions):
                    for case_id, seed_index, context in contexts:
                        path = sample_path(args.out, case_id, seed_index, block, repetition)
                        if path.is_file():
                            continue
                        if time.monotonic() + 120 >= deadline:
                            break
                        torch.cuda.reset_peak_memory_stats(device)
                        measurement = run_one(common, runtime, context, torch=torch, device=device)
                        peak_gb = float(torch.cuda.max_memory_allocated(device)) / 1e9
                        record = {**expected_samples[path], "seed": int(context["seed"]), **measurement,
                                  "peak_allocated_gb_decimal": peak_gb,
                                  "timing_scope": p["performance"]["latency_scope"],
                                  "pbs_job_id": os.environ.get("PBS_JOBID")}
                        if not math.isfinite(peak_gb) or peak_gb <= 0:
                            raise RuntimeError("Invalid CUDA peak allocated measurement")
                        atomic_json(path, record)
                    if time.monotonic() + 120 >= deadline:
                        break
                if time.monotonic() + 120 >= deadline:
                    break

        records, coverage = [], []
        for path, expected in expected_samples.items():
            if path.is_file():
                records.append(load_sample(path, expected))
        for case_id, seed_index, _ in contexts:
            count = sum(1 for record in records
                        if record["case_id"] == case_id and record["seed_index"] == seed_index)
            coverage.append({"case_id": case_id, "seed_index": seed_index,
                             "completed_repetitions": count,
                             "expected_repetitions": blocks * repetitions})
        missing = [row for row in coverage if row["completed_repetitions"] != row["expected_repetitions"]]
        complete = not missing and len(records) == len(contexts) * blocks * repetitions
        if args.arm == 'ours_vqwam':
            live = runtime.ours_vqwam_engine.metadata()
            if live['parity_status'] != 'validated_full80':
                raise RuntimeError('Ours engine lost validated graph status during measured queries')
            metadata.update(live)
            common.save_json(args.out / 'install_metadata.json', metadata)
        denoising = summarize([record["denoising_seconds"] for record in records])
        complete_endpoint = summarize([record["complete_endpoint_query_seconds"] for record in records])
        peak = summarize([record["peak_allocated_gb_decimal"] for record in records])
        summary = {
            "protocol": p["protocol"], "status": "complete" if complete else "checkpointed",
            "complete": complete, "arm": args.arm, "model": p["model"], "mode": p["mode"],
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "hardware": hardware,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_label": {"implementation": metadata.get("implementation"),
                              "execution_scope": execution_scope,
                              "native_int4_kernel_used": metadata.get("native_int4_kernel_used"),
                              "optimized_engine": args.arm == "ours_vqwam"},
            "engine_validation": ({'parity_status': metadata['parity_status'],
                                   'cuda_graph': metadata['cuda_graph']}
                                  if args.arm == 'ours_vqwam' else None),
            "coverage": {"queries": len(contexts), "completed_queries": len(contexts) - len({
                (row["case_id"], row["seed_index"]) for row in missing}),
                         "samples": len(records), "expected_samples": len(contexts) * blocks * repetitions,
                         "reused_samples": resumed_samples, "by_query": coverage, "missing": missing},
            "repetitions_per_query": repetitions, "blocks": blocks,
            "warmup_queries": warmups,
            "denoising_wall_seconds": denoising,
            "complete_endpoint_query_seconds": complete_endpoint,
            "timing_scope": p["performance"]["latency_scope"],
            "endpoint_scope_label": "complete common.endpoint call; includes endpoint copies/extraction and is not denoising-only latency",
            "peak_allocated_gb_decimal": peak,
            "packed_storage_bytes": storage_bytes,
            "packed_storage_gib_binary": storage_bytes / (2 ** 30),
            "storage_scope": p["performance"]["storage_scope"],
            "speedup_over_bf16": None,
            "speedup_status": "computed only after all six arm receipts are aggregated",
            "execution_boundary": p["execution_boundary"],
        }
        atomic_json(args.out / "summary.json", summary)
        panel = aggregate_completed_panel(p, args.root) if complete else None
        if panel is not None:
            summary = read_json(args.out / "summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0 if complete else 2
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
