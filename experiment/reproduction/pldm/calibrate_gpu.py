#!/usr/bin/env python3
"""Run an isolated 10-warmup/100-measured PLDM training cost calibration."""
from __future__ import annotations

import csv
import copy
import io
import json
import importlib.metadata
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SHA = "1bd7e564ecd961205bc18b23067b19e9ca24ac90"
WARMUP_UPDATES = 10
MEASURED_UPDATES = 100
TOTAL_UPDATES = WARMUP_UPDATES + MEASURED_UPDATES
DEFAULT_ROOT = Path("/scratch/users/ntu/yguo017/pldm-reproduction")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise SystemExit("Refusing GPU work without PBS_JOBID and PBS_NODEFILE")
    host = socket.gethostname().split(".")[0].lower()
    if any(word in host for word in ("login", "head", "submit")):
        raise SystemExit(f"Refusing probable login node: {host}")
    nodes = {
        line.split()[0].split(".")[0].lower()
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in nodes:
        raise SystemExit(f"Current host {host} is not listed in PBS_NODEFILE")
    return job_id, host


def atomic_json(path: Path, value: dict, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{job_id}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def gpu_identity(torch_module) -> dict:
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise RuntimeError("GPU identity gate failed: no CUDA device visible to PyTorch")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True
    )
    rows = list(csv.reader(io.StringIO(output)))
    devices = [{"index": row[0].strip(), "uuid": row[1].strip(), "name": row[2].strip()} for row in rows if len(row) >= 3]
    if not devices:
        raise RuntimeError("GPU identity gate failed: NVML returned no GPU rows")
    token = visible.split(",", 1)[0].strip() if visible else ""
    selected = next((device for device in devices if token and token in (device["index"], device["uuid"])), devices[0] if not token else None)
    if selected is None:
        raise RuntimeError(f"GPU identity gate failed: CUDA_VISIBLE_DEVICES token {token!r} has no NVML index/UUID match: {devices}")
    torch_name = torch_module.cuda.get_device_properties(0).name.strip()
    if torch_name != selected["name"]:
        raise RuntimeError(f"GPU identity gate failed: torch cuda:0 name={torch_name!r} does not match selected NVML device={selected}")
    return {"CUDA_VISIBLE_DEVICES": visible or "<unset>", "torch_logical_device": "cuda:0", "torch_device_name": torch_name, "nvml_device": selected}


def validate_preparation(root: Path) -> tuple[dict, dict, Path, Path]:
    prep = json.loads((root / "PREPARATION.json").read_text(encoding="utf-8"))
    upstream_identity = json.loads(
        (root / "upstream" / "source_identity.json").read_text(encoding="utf-8")
    )
    staged_identity = json.loads(
        (root / "staged_source_identity.json").read_text(encoding="utf-8")
    )
    if prep.get("status") != "PASS":
        raise RuntimeError(f"CPU preparation gate is not PASS: {prep.get('status')}")
    if prep.get("source_sha") != SOURCE_SHA:
        raise RuntimeError("CPU preparation source SHA differs from the frozen source")
    if upstream_identity.get("source_sha") != SOURCE_SHA:
        raise RuntimeError("Upstream source identity differs from the frozen source")
    if staged_identity.get("source_sha") != SOURCE_SHA:
        raise RuntimeError("Staged source identity differs from the frozen source")
    if staged_identity.get("algorithm_or_config_changes") != []:
        raise RuntimeError("Staged source identity reports algorithm/config changes")
    for name in ("config_composition", "render"):
        if prep.get("stages", {}).get(name, {}).get("status") != "PASS":
            raise RuntimeError(f"CPU preparation stage {name} is not PASS")
    loader = prep.get("actual_training_loader", {})
    if loader.get("status") != "PASS" or loader.get("actual_passes") != 3:
        raise RuntimeError("CPU actual-loader / three-pass gate is not PASS")
    if loader.get("expected_optimizer_updates", 0) <= 0:
        raise RuntimeError("CPU preparation did not record a positive official update count")
    probing_loader = prep.get("actual_probing_loader", {})
    if probing_loader.get("status") != "PASS" or probing_loader.get("total_training_passes") != 50:
        raise RuntimeError("CPU probing-loader metadata / 50-pass gate is not PASS")
    dataset = root / "datasets" / "good_quality_data_memmap"
    metadata_path = dataset / "render_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Prepared memmap metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_commit") != SOURCE_SHA or metadata.get("states_dtype") != "uint8":
        raise RuntimeError("Prepared memmap identity/dtype gate failed")
    if metadata.get("frame_count") != 3_686_400:
        raise RuntimeError("Prepared frame count differs from the released full dataset")
    return prep, staged_identity, dataset, metadata


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty timing series")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def current_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def rng_snapshot(np, torch) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def rng_snapshot_equal(before: dict, after: dict, np, torch) -> bool:
    before_numpy, after_numpy = before["numpy"], after["numpy"]
    numpy_equal = (
        before_numpy[0] == after_numpy[0]
        and np.array_equal(before_numpy[1], after_numpy[1])
        and before_numpy[2:] == after_numpy[2:]
    )
    cuda_equal = len(before["torch_cuda"]) == len(after["torch_cuda"]) and all(
        torch.equal(left, right)
        for left, right in zip(before["torch_cuda"], after["torch_cuda"])
    )
    return (
        before["python"] == after["python"]
        and numpy_equal
        and torch.equal(before["torch_cpu"], after["torch_cpu"])
        and cuda_equal
    )


def storage_equivalence_gate(trainer, dataset_path: Path, config, np, torch) -> dict:
    """Compare eager native storage with mmap storage at 64 deterministic windows."""
    wrapped = trainer.ds
    base = wrapped.dataloader.dataset
    eager_states = base.states
    if not isinstance(eager_states, np.ndarray) or isinstance(eager_states, np.memmap):
        raise RuntimeError(f"Expected eager host ndarray for states, got {type(eager_states).__name__}")
    if eager_states.dtype != np.uint8:
        raise RuntimeError(f"Expected eager states dtype uint8, got {eager_states.dtype}")

    trajectories, trajectory_frames = eager_states.shape[:2]
    windows_per_trajectory = trajectory_frames - int(config.data.offline_wall_config.n_steps) + 1
    dataset_length = len(base)
    if windows_per_trajectory <= 0 or dataset_length != trajectories * windows_per_trajectory:
        raise RuntimeError(
            "Eager dataset length does not match trajectory/window layout: "
            f"len={dataset_length}, trajectories={trajectories}, "
            f"windows_per_trajectory={windows_per_trajectory}"
        )

    selected = {0, dataset_length - 1}
    boundary_trajectories = np.linspace(1, trajectories - 1, 20, dtype=np.int64).tolist()
    for trajectory in boundary_trajectories:
        boundary_index = int(trajectory) * windows_per_trajectory
        selected.update((boundary_index - 1, boundary_index))
    for index in np.linspace(0, dataset_length - 1, 256, dtype=np.int64).tolist():
        if len(selected) >= 64:
            break
        selected.add(int(index))
    indices = sorted(selected)
    if len(indices) != 64 or indices[0] != 0 or indices[-1] != dataset_length - 1:
        raise RuntimeError(f"Could not construct the frozen 64-window storage probe: {indices}")

    lazy_reference = copy.copy(base)
    lazy_reference.states = np.load(dataset_path / "states.npy", mmap_mode="r")
    if not isinstance(lazy_reference.states, np.memmap):
        raise RuntimeError("Lazy-reference states.npy did not open as a NumPy memmap")
    if lazy_reference.states.shape != eager_states.shape or lazy_reference.states.dtype != eager_states.dtype:
        raise RuntimeError(
            "Eager/mmap states metadata mismatch: "
            f"eager={eager_states.shape}/{eager_states.dtype}, "
            f"mmap={lazy_reference.states.shape}/{lazy_reference.states.dtype}"
        )
    mmap_type = type(lazy_reference.states).__name__

    torch.cuda.synchronize()
    rng_before = rng_snapshot(np, torch)
    started = time.perf_counter()
    eager_samples = [base[index] for index in indices]
    lazy_samples = [lazy_reference[index] for index in indices]
    sample_fields = tuple(getattr(eager_samples[0], "_fields", ()))
    if not {"states", "actions", "locations"}.issubset(sample_fields):
        raise RuntimeError(f"Unexpected WallSample fields: {sample_fields}")
    if any(tuple(getattr(sample, "_fields", ())) != sample_fields for sample in eager_samples + lazy_samples):
        raise RuntimeError("Eager/mmap WallSample field definitions differ")

    collate = torch.utils.data.default_collate
    eager_raw = collate(eager_samples)
    lazy_raw = collate(lazy_samples)
    raw_equal = {
        name: bool(torch.equal(getattr(eager_raw, name), getattr(lazy_raw, name)))
        for name in sample_fields
    }
    with torch.no_grad():
        eager_normalized = wrapped.normalizer.normalize_sample(eager_raw)
        lazy_normalized = wrapped.normalizer.normalize_sample(lazy_raw)
    normalized_fields = tuple(getattr(eager_normalized, "_fields", ()))
    lazy_normalized_fields = tuple(getattr(lazy_normalized, "_fields", ()))
    normalized_core_fields = {"states", "actions", "locations"}
    normalized_structure_equal = (
        normalized_fields == lazy_normalized_fields
        and normalized_core_fields.issubset(normalized_fields)
    )
    normalized_equal = {
        name: bool(torch.equal(getattr(eager_normalized, name), getattr(lazy_normalized, name)))
        for name in normalized_fields
        if name in lazy_normalized_fields
    }
    torch.cuda.synchronize()
    rng_after = rng_snapshot(np, torch)
    rng_unchanged = rng_snapshot_equal(rng_before, rng_after, np, torch)
    elapsed = time.perf_counter() - started
    metadata = {
        name: {
            "shape": list(getattr(eager_raw, name).shape),
            "dtype": str(getattr(eager_raw, name).dtype),
        }
        for name in sample_fields
    }
    normalized_metadata = {
        name: {
            "shape": list(getattr(eager_normalized, name).shape),
            "dtype": str(getattr(eager_normalized, name).dtype),
        }
        for name in normalized_fields
    }
    del eager_samples, lazy_samples, eager_raw, lazy_raw, eager_normalized, lazy_normalized, lazy_reference
    torch.cuda.synchronize()
    failure_reasons = []
    if not all(raw_equal.values()):
        failure_reasons.append("raw WallSample fields differ")
    if not all(normalized_equal.values()):
        failure_reasons.append("normalized WallSample fields differ")
    if not normalized_structure_equal:
        failure_reasons.append("normalized WallSample field definitions differ")
    if not normalized_core_fields.issubset(normalized_fields):
        failure_reasons.append("normalized WallSample lacks states/actions/locations")
    if not rng_unchanged:
        random.setstate(rng_before["python"])
        np.random.set_state(rng_before["numpy"])
        torch.set_rng_state(rng_before["torch_cpu"])
        torch.cuda.set_rng_state_all(rng_before["torch_cuda"])
        failure_reasons.append("Python/NumPy/Torch CPU/CUDA RNG state changed (state restored)")
    return {
        "status": "PASS" if not failure_reasons else "FAIL",
        "failure_reasons": failure_reasons,
        "storage_mode": "native eager uint8 states versus read-only mmap uint8 states",
        "normalizer_reused": True,
        "data_loader_iterated": False,
        "normalizer_rescanned": False,
        "sample_count": len(indices),
        "indices": indices,
        "boundary_trajectory_ids": [int(value) for value in boundary_trajectories],
        "windows_per_trajectory": windows_per_trajectory,
        "raw_fields": list(sample_fields),
        "normalized_fields": list(normalized_fields),
        "eager_states": {
            "type": type(eager_states).__name__,
            "shape": list(eager_states.shape),
            "dtype": str(eager_states.dtype),
            "nbytes": int(eager_states.nbytes),
            "current_process_rss_bytes_after_gate": current_rss_bytes(),
        },
        "mmap_states": {
            "type": mmap_type,
            "shape": list(eager_states.shape),
            "dtype": str(eager_states.dtype),
        },
        "raw_field_equal": raw_equal,
        "normalized_field_equal": normalized_equal,
        "normalized_structure_equal": normalized_structure_equal,
        "raw_batch_metadata": metadata,
        "normalized_batch_metadata": normalized_metadata,
        "rng_states_unchanged": rng_unchanged,
        "elapsed_seconds": elapsed,
        "formal_storage_override_decision": (
            "ELIGIBLE_FOR_ROOT_REVIEW_AFTER_PASS"
            if not failure_reasons
            else "REJECT_STORAGE_OVERRIDE"
        ),
    }


class CalibrationStop(Exception):
    pass


class BoundedLoader:
    """Forward the official loader and stop before an epoch can validate/save."""

    def __init__(self, base, torch, records, limit: int, deadline: float, training_started: float):
        self.base = base
        self.torch = torch
        self.records = records
        self.limit = limit
        self.deadline = deadline
        self.training_started = training_started
        self.stop_reason = ""
        self.first_batch_start_offset_seconds = None
        self.first_batch = None

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __iter__(self):
        iterator = iter(self.base)
        for index in range(self.limit):
            if time.monotonic() >= self.deadline:
                self.stop_reason = "internal_calibration_time_limit_before_next_update"
                raise CalibrationStop(self.stop_reason)
            step_start = time.perf_counter()
            if index == 0:
                self.first_batch_start_offset_seconds = step_start - self.training_started
            try:
                batch = next(iterator)
            except StopIteration:
                self.stop_reason = "official_loader_exhausted_before_110_updates"
                raise CalibrationStop(self.stop_reason)
            batch_ready = time.perf_counter()
            batch_wait_seconds = batch_ready - step_start
            if index == 0:
                self.first_batch = batch
            update_body_start = time.perf_counter()
            yield batch
            self.torch.cuda.synchronize()
            update_body_seconds = time.perf_counter() - update_body_start
            self.records.append(
                {
                    "index": index + 1,
                    "batch_wait_seconds": batch_wait_seconds,
                    "update_body_seconds_including_cuda_completion": update_body_seconds,
                    "seconds_including_batch_wait_and_cuda_completion": time.perf_counter()
                    - step_start,
                }
            )
            if index + 1 == self.limit:
                self.stop_reason = "completed_10_warmup_plus_100_timed_updates"
                raise CalibrationStop(self.stop_reason)
            if time.monotonic() >= self.deadline:
                self.stop_reason = "internal_calibration_time_limit"
                raise CalibrationStop(self.stop_reason)
        self.stop_reason = "completed_limit"
        raise CalibrationStop(self.stop_reason)


def run_native_planner_cost_probe(trainer, config, bounded_loader, torch, process_started: float) -> dict:
    """Time official MPPIPlanner.plan calls without running native evaluation."""
    deadline = process_started + float(os.environ.get("PLDM_PLANNER_CALIBRATION_DEADLINE_SECONDS", "810"))
    report = {
        "status": "SKIPPED_TIME_BUDGET",
        "purpose": "cost only; no native evaluation, success, or quality metric",
        "requested": {"warmup_calls": 1, "steady_calls_max": 3, "batch_size": 20, "plan_size": 96, "mppi_samples": 2000},
        "calls": [],
            "native_evaluation_prober_used": False,
            "projected_cost": bool(config.eval_cfg.wall_planning.level1.projected_cost),
            "notes": [
                "uses the official MPCEvaluator._construct_planner helper and MPPIPlanner.plan",
                "prober=None; pinned projected_cost is false, so MPPI running cost is unchanged; output predicted-location projection work is omitted",
                "current and target observations come from the first actual frozen training-loader batch",
                "reset_targets calls objective.set_target only; nominal per-environment MPPI controls remain stateful between calls, matching native planner reuse",
            ],
    }
    if bounded_loader.first_batch is None:
        report["status"] = "SKIPPED_NO_FROZEN_DATA_BATCH"
        return report
    if config.eval_cfg.wall_planning.level1.projected_cost:
        report["status"] = "SKIPPED_PROBER_REQUIRED"
        report["notes"].append("projected_cost=true requires the official trained location prober")
        return report
    remaining = deadline - time.monotonic()
    if remaining < 330:
        report["remaining_seconds_at_gate"] = max(0.0, remaining)
        return report

    batch = bounded_loader.first_batch
    if batch.states.ndim != 5 or batch.states.shape[0] < 20 or batch.states.shape[1] < 2:
        report["status"] = "FAIL_INPUT_SHAPE"
        report["input_state_shape"] = list(batch.states.shape)
        return report

    from pldm.planning.mpc import MPCEvaluator
    from pldm.models.utils import flatten_conv_output

    model = trainer.model.level1
    trainer.model.eval()
    config_mpc = config.eval_cfg.wall_planning
    helper = MPCEvaluator(
        config=config_mpc,
        model=model,
        prober=None,
        normalizer=trainer.ds.normalizer,
        quick_debug=False,
    )
    planner = helper._construct_planner(n_envs=20)
    observations = batch.states[:20]
    current_observations = observations[:, 0].cuda()
    goal_observations = observations[:, -1].float().cuda()
    with torch.no_grad():
        goal_component = model.backbone(goal_observations).obs_component.detach()
        target_encodings = flatten_conv_output(goal_component)
        planner.reset_targets(target_encodings, repr_input=True)

    report.update(
        {
            "status": "RUNNING",
            "input_state_shape": list(batch.states.shape),
            "input_state_dtype": str(batch.states.dtype),
            "current_observation_shape": list(current_observations.shape),
            "goal_observation_shape": list(goal_observations.shape),
            "target_encoding_shape": list(target_encodings.shape),
            "remaining_seconds_before_first_call": deadline - time.monotonic(),
            "peak_memory_baseline_allocated_bytes": torch.cuda.memory_allocated(0),
            "peak_memory_baseline_reserved_bytes": torch.cuda.memory_reserved(0),
            "model_mode": "eval",
            "model_mode_matches_native_validate": True,
        }
    )
    torch.cuda.reset_peak_memory_stats(0)

    def timed_call(kind: str, ordinal: int) -> dict:
        # Use the same fixed goal for every sample. The official planner keeps
        # nominal MPPI state between calls, as it does during native evaluation.
        planner.reset_targets(target_encodings, repr_input=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = planner.plan(
            current_observations,
            plan_size=96,
            repr_input=False,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        finite = {
            "actions": bool(torch.isfinite(result.actions).all().item()),
            "pred_encs": bool(torch.isfinite(result.pred_encs).all().item()),
            "pred_obs": bool(torch.isfinite(result.pred_obs).all().item()),
        }
        if not all(finite.values()):
            raise RuntimeError(f"Native planner cost-only call returned non-finite tensors: {finite}")
        return {
            "kind": kind,
            "ordinal": ordinal,
            "seconds_including_cuda_completion": seconds,
            "actions_shape": list(result.actions.shape),
            "pred_encs_shape": list(result.pred_encs.shape),
            "pred_obs_shape": list(result.pred_obs.shape),
            "finite": finite,
        }

    try:
        report["calls"].append(timed_call("warmup", 1))
        warmup_seconds = report["calls"][0]["seconds_including_cuda_completion"]
        for ordinal in range(1, 4):
            remaining = deadline - time.monotonic()
            minimum_next_call_budget = max(30.0, warmup_seconds * 1.25 + 15.0)
            if remaining < minimum_next_call_budget:
                break
            report["calls"].append(timed_call("steady", ordinal))
        steady = [call["seconds_including_cuda_completion"] for call in report["calls"] if call["kind"] == "steady"]
        report["status"] = "PASS" if len(steady) == 3 else "PARTIAL_STEADY_CALLS"
        report["steady_call_count"] = len(steady)
        report["steady_mean_seconds"] = sum(steady) / len(steady) if steady else None
        report["native_eval_planner_calls_per_invocation_from_source"] = 1000
        report["source_work_estimate"] = {
            "evaluation_invocation_planner_calls": 1000,
            "total_candidate_latent_transitions": 2_928_000_000,
            "h96_call_candidate_latent_transitions": 3_840_000,
            "h96_call_equivalents_by_transition_count": 762.5,
            "linear_transition_scaled_runtime_seconds_estimate": (
                762.5 * report["steady_mean_seconds"] if steady else None
            ),
            "planning_only_horizon_96_proxy_seconds": (
                1000 * report["steady_mean_seconds"] if steady else None
            ),
            "limitation": "linear transition scaling omits 1000-call overhead, environment stepping, trained-prober projection, native probing, reporting, and early termination",
        }
        report["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        report["remaining_seconds_after_probe"] = deadline - time.monotonic()
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()[-5000:]}
    return report


def main() -> int:
    job_id, host = assert_allocation()
    started_utc = utc_now()
    process_started = time.monotonic()
    root = Path(os.environ.get("PLDM_ROOT", str(DEFAULT_ROOT))).resolve()
    report_dir = root / "reports" / f"calibration-{job_id}"
    report_path = report_dir / "CALIBRATION.json"
    report = {
        "status": "RUNNING",
        "job_id": job_id,
        "host": host,
        "started_utc": started_utc,
        "source_sha": SOURCE_SHA,
        "remote_root": str(root),
        "scope": "10 warmup + 100 timed official PLDM optimizer updates, native eager-uint8 storage equivalence gate, and bounded cost-only native MPPI planner samples when time permits; no native evaluation or quality claim",
    }
    atomic_json(report_path, report, job_id)

    try:
        prep, staged_identity, dataset, render_metadata = validate_preparation(root)
        staged = (root / "staged").resolve()
        if not staged.is_dir():
            raise RuntimeError(f"Staged source tree is missing: {staged}")
        formal_output = root / "checkpoints" / "tworooms-seqlen90-3M-seed101"
        if formal_output.exists():
            raise RuntimeError("Formal checkpoint path already exists; calibration refuses to touch it")
        calibration_root = root / "calibration_output"
        calibration_dir_name = f"tworooms-calibration-{job_id}"
        calibration_output = calibration_root / calibration_dir_name
        if calibration_output.exists():
            raise RuntimeError(f"Refusing to reuse calibration output: {calibration_output}")

        overlays = [
            *[Path(record["overlay"]) for record in prep.get("observed_dependency_installs", [])],
            root / "runtime_overlay" / "numpy-1.26.4-scipy-1.10.0-pinned",
            root / "runtime_overlay" / "pandas-2.0.1-pinned",
            root / "runtime_overlay" / "zarr-2.14.2-pinned",
            root / "runtime_overlay" / "gdown",
            root / "runtime_overlay" / "html-parser",
            root / "runtime_overlay" / "arm-pytorch-utilities-0.4.3",
            root / "runtime_overlay" / "pytorch-seed-0.2.0",
            root / "runtime_overlay" / "statsmodels-patsy-pinned",
            Path("/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runtime-overlay-minimal"),
        ]
        overlays = [path.resolve() for path in overlays if path.is_dir()]
        sys.path[:0] = [str(staged), *map(str, overlays)]

        import numpy as np
        import scipy
        import pandas
        import pytz
        import torch
        import zarr
        import numcodecs
        import gym
        import gym_notices
        import arm_pytorch_utilities
        import pytorch_seed
        import statsmodels
        import statsmodels.api
        import patsy
        from tqdm.auto import tqdm as tqdm_impl

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("A visible GPU is required for the calibration allocation")

        import pldm.train as train_module
        from pldm.logger import Logger
        from pldm.optimizers.optimizer_factory import OptimizerFactory
        import pldm.data.dataset_factory as dataset_factory_module
        import pldm.models.hjepa as hjepa_module
        import pldm_envs.wall.data.offline_wall as offline_wall_module

        source_modules = {
            "pldm.train": train_module,
            "pldm.data.dataset_factory": dataset_factory_module,
            "pldm.models.hjepa": hjepa_module,
            "pldm_envs.wall.data.offline_wall": offline_wall_module,
        }
        module_origins = {}
        for name, module in source_modules.items():
            origin = Path(module.__file__).resolve()
            if not inside(origin, staged):
                raise RuntimeError(f"Import escaped staged source for {name}: {origin}")
            module_origins[name] = str(origin)
        zarr_overlay = root / "runtime_overlay" / "zarr-2.14.2-pinned"
        for name, module in (("zarr", zarr), ("numcodecs", numcodecs)):
            origin = Path(module.__file__).resolve()
            if not inside(origin, zarr_overlay):
                raise RuntimeError(f"Import escaped the isolated pinned overlay for {name}: {origin}")
            module_origins[name] = str(origin)
        gym_overlay = Path("/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runtime-overlay-minimal")
        for name, module in (("gym", gym), ("gym_notices", gym_notices)):
            origin = Path(module.__file__).resolve()
            if not inside(origin, gym_overlay):
                raise RuntimeError(f"Import escaped the shared read-only gym overlay for {name}: {origin}")
            module_origins[name] = str(origin)
        arm_overlay = root / "runtime_overlay" / "arm-pytorch-utilities-0.4.3"
        arm_origin = Path(arm_pytorch_utilities.__file__).resolve()
        if not inside(arm_origin, arm_overlay):
            raise RuntimeError(f"Import escaped the isolated pinned arm_pytorch_utilities overlay: {arm_origin}")
        module_origins["arm_pytorch_utilities"] = str(arm_origin)
        seed_overlay = root / "runtime_overlay" / "pytorch-seed-0.2.0"
        seed_origin = Path(pytorch_seed.__file__).resolve()
        if not inside(seed_origin, seed_overlay):
            raise RuntimeError(f"Import escaped the isolated pinned pytorch-seed overlay: {seed_origin}")
        module_origins["pytorch_seed"] = str(seed_origin)
        statsmodels_overlay = root / "runtime_overlay" / "statsmodels-patsy-pinned"
        for module in (statsmodels, statsmodels.api, patsy):
            origin = Path(module.__file__).resolve()
            if not inside(origin, statsmodels_overlay):
                raise RuntimeError(f"Import escaped the isolated pinned statsmodels overlay: {origin}")
            module_origins[module.__name__] = str(origin)
        scientific_overlay = root / "runtime_overlay" / "numpy-1.26.4-scipy-1.10.0-pinned"
        for module, expected_version in ((np, "1.26.4"), (scipy, "1.10.0")):
            origin = Path(module.__file__).resolve()
            if not inside(origin, scientific_overlay):
                raise RuntimeError(f"Import escaped the isolated pinned scientific overlay: {module.__name__}: {origin}")
            if module.__version__ != expected_version:
                raise RuntimeError(f"Pinned scientific stack version mismatch for {module.__name__}: {module.__version__} != {expected_version}")
            module_origins[module.__name__] = str(origin)
        pandas_overlay = root / "runtime_overlay" / "pandas-2.0.1-pinned"
        if pandas.__version__ != "2.0.1" or not inside(Path(pandas.__file__).resolve(), pandas_overlay):
            raise RuntimeError(f"Pandas import/version escaped its official pinned overlay: version={pandas.__version__} origin={pandas.__file__}")
        module_origins["pandas"] = str(Path(pandas.__file__).resolve())
        if importlib.metadata.version("pytz") != "2022.7.1" or not inside(Path(pytz.__file__).resolve(), pandas_overlay):
            raise RuntimeError(f"Pytz import/version escaped its official pinned overlay: version={importlib.metadata.version('pytz')} origin={pytz.__file__}")
        module_origins["pytz"] = str(Path(pytz.__file__).resolve())
        observed_dependency_origins = {}
        for record in prep.get("observed_dependency_installs", []):
            module = __import__(record["trigger_module_not_found"])
            version = importlib.metadata.version(record["distribution"])
            origin = Path(module.__file__).resolve()
            overlay = Path(record["overlay"]).resolve()
            if version != record["version"] or not inside(origin, overlay):
                raise RuntimeError(f"Observed dependency import escaped its pinned task overlay: {record} origin={origin} version={version}")
            module_origins[record["trigger_module_not_found"]] = str(origin)
            observed_dependency_origins[record["distribution"]] = {"version": version, "origin": str(origin), "overlay": str(overlay)}

        identity_path = root / "upstream" / "source_identity.json"
        assert json.loads(identity_path.read_text(encoding="utf-8"))["source_sha"] == SOURCE_SHA
        assert json.loads((root / "staged_source_identity.json").read_text(encoding="utf-8"))["source_sha"] == SOURCE_SHA
        report.update(
            {
                "status": "RUNNING",
                "prep_job_id": prep.get("cpu_job_id"),
                "dataset_path": str(dataset),
                "released_frame_count": render_metadata["frame_count"],
                "released_observation_dtype": render_metadata["states_dtype"],
                "module_origins": module_origins,
                "observed_dependency_origins": observed_dependency_origins,
                "runtime": {
                    "python": sys.executable,
                    "python_version": sys.version,
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "pandas": pandas.__version__,
                    "pytz": importlib.metadata.version("pytz"),
                    "zarr": zarr.__version__,
                    "numcodecs": numcodecs.__version__,
                    "gym": importlib.metadata.version("gym"),
                    "gym-notices": importlib.metadata.version("gym-notices"),
                    "arm_pytorch_utilities": importlib.metadata.version("arm_pytorch_utilities"),
                    "pytorch-seed": importlib.metadata.version("pytorch-seed"),
                    "statsmodels": importlib.metadata.version("statsmodels"),
                    "patsy": importlib.metadata.version("patsy"),
                    "cuda_device": torch.cuda.get_device_name(0),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu_identity": gpu_identity(torch),
                    "staged_compatibility_patches": staged_identity.get("patches", []),
                },
            }
        )
        atomic_json(report_path, report, job_id)

        config_started = time.perf_counter()
        config_cli_values = [
            f"output_root={calibration_root}",
            f"output_dir={calibration_dir_name}",
            f"run_name=calibration-{job_id}",
            f"data.offline_wall_config.offline_data_path={dataset}",
            "data.offline_wall_config.lazy_load=false",
            "data.offline_wall_config.device=cuda",
            "wandb=false",
            "resume_if_possible=false",
            "eval_at_beginning=false",
            "eval_during_training=false",
        ]
        sys.argv = [
            "train.py",
            "--configs",
            "configs/wall/icml/seqlen90_3M.yaml",
            "--values",
            *config_cli_values,
        ]
        config = train_module.TrainConfig.parse_from_command_line()
        config_parse_seconds = time.perf_counter() - config_started
        frozen = {
            "seed": 101,
            "epochs": 2,
            "quick_debug": False,
            "n_steps": 16,
            "objectives": [getattr(value, "name", str(value)) for value in config.objectives_l1.objectives],
            "optimizer_type": str(config.optimizer_type),
            "base_lr": config.base_lr,
            "compile_model": config.compile_model,
            "data_path": str(dataset),
            "lazy_load": config.data.offline_wall_config.lazy_load,
            "offline_dataset_device": config.data.offline_wall_config.device,
            "eval_at_beginning": config.eval_at_beginning,
            "eval_during_training": config.eval_during_training,
            "native_eval": {
                "level": config.eval_cfg.wall_planning.levels,
                "n_envs": config.eval_cfg.wall_planning.n_envs,
                "n_steps": config.eval_cfg.wall_planning.n_steps,
                "n_envs_batch_size": config.eval_cfg.wall_planning.n_envs_batch_size,
                "replan_every": config.eval_cfg.wall_planning.replan_every,
                "max_plan_length": config.eval_cfg.wall_planning.level1.max_plan_length,
                "mppi_samples": config.eval_cfg.wall_planning.level1.mppi.num_samples,
                "eval_mpcs_field_value_unused_by_official_evaluator": config.eval_mpcs,
                "planner_calls_per_invocation_from_source": 1000,
            },
        }
        if (
            config.seed != 101
            or config.epochs != 2
            or config.quick_debug
            or config.n_steps != 16
            or frozen["objectives"] != ["VICReg", "IDM"]
            or config.hjepa.train_l1 is not True
            or config.wandb is not False
            or config.data.offline_wall_config.n_steps != 16
            or config.data.offline_wall_config.lazy_load is not False
            or config.data.offline_wall_config.device != "cuda"
            or config.eval_cfg.wall_planning.levels != "medium"
            or config.eval_cfg.wall_planning.n_envs != 100
            or config.eval_cfg.wall_planning.n_steps != 200
            or config.eval_cfg.wall_planning.n_envs_batch_size != 20
            or config.eval_cfg.wall_planning.replan_every != 1
            or config.eval_cfg.wall_planning.level1.mppi.num_samples != 2000
            or config.eval_cfg.wall_planning.level1.projected_cost is not False
            or config.eval_mpcs != 20
        ):
            raise RuntimeError(f"Calibration config gate failed: {frozen}")

        print("PREPARATION_CPU_GATE=PASS", flush=True)
        print("STAGED_IMPORT_ORIGINS=" + json.dumps(module_origins), flush=True)
        print("OFFICIAL_CONFIG_GATE=PASS " + json.dumps(frozen), flush=True)

        host_rss_before_init = current_rss_bytes()
        torch.cuda.synchronize()
        init_started = time.perf_counter()
        trainer = train_module.Trainer(config)
        torch.cuda.synchronize()
        trainer_init_seconds = time.perf_counter() - init_started
        loader_length = len(trainer.ds)
        prepared_loader = prep["actual_training_loader"]
        if loader_length != prepared_loader.get("optimizer_updates_per_epoch"):
            raise RuntimeError(
                "GPU Trainer loader length differs from the guarded CPU loader probe: "
                f"{loader_length} vs {prepared_loader.get('optimizer_updates_per_epoch')}"
            )
        epoch_indices = list(range(0, int(config.epochs) + 1))
        expected_updates = loader_length * len(epoch_indices)
        if expected_updates != prepared_loader.get("expected_optimizer_updates"):
            raise RuntimeError(
                "GPU official update count differs from the guarded CPU loader calculation: "
                f"{expected_updates} vs {prepared_loader.get('expected_optimizer_updates')}"
            )
        base_dataset = trainer.ds.dataloader.dataset
        host_rss_after_init = current_rss_bytes()
        try:
            storage_gate_report = storage_equivalence_gate(trainer, dataset, config, np, torch)
        except Exception as exc:
            report["storage_equivalence_gate"] = {
                "status": "FAIL",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc()[-5000:],
                },
            }
            report["host_memory"] = {
                "rss_before_trainer_init_bytes": host_rss_before_init,
                "rss_after_trainer_init_bytes": host_rss_after_init,
                "eager_states_shape": list(base_dataset.states.shape),
                "eager_states_dtype": str(base_dataset.states.dtype),
                "eager_states_nbytes": int(base_dataset.states.nbytes),
            }
            atomic_json(report_path, report, job_id)
            raise
        report["storage_equivalence_gate"] = storage_gate_report
        report["host_memory"] = {
            "rss_before_trainer_init_bytes": host_rss_before_init,
            "rss_after_trainer_init_bytes": host_rss_after_init,
            "rss_after_storage_gate_bytes": current_rss_bytes(),
            "trainer_initialization_seconds_including_native_eager_materialization": trainer_init_seconds,
            "eager_arrays": {
                name: {
                    "shape": list(getattr(base_dataset, name).shape),
                    "dtype": str(getattr(base_dataset, name).dtype),
                    "nbytes": int(getattr(base_dataset, name).nbytes),
                }
                for name in ("states", "actions", "locations")
            },
        }
        atomic_json(report_path, report, job_id)
        if storage_gate_report["status"] != "PASS":
            raise RuntimeError(
                "Storage-only equivalence gate failed: "
                f"{storage_gate_report['failure_reasons']}"
            )
        torch.cuda.reset_peak_memory_stats(0)
        report["config"] = frozen
        report["timing"] = {
            "config_parse_seconds": config_parse_seconds,
            "trainer_initialization_seconds": trainer_init_seconds,
            "optimizer_updates_per_epoch": loader_length,
            "actual_epoch_indices": epoch_indices,
            "actual_passes": len(epoch_indices),
            "expected_official_optimizer_updates": expected_updates,
            "calibration_update_budget": {
                "warmup": WARMUP_UPDATES,
                "timed": MEASURED_UPDATES,
                "total": TOTAL_UPDATES,
            },
            "internal_time_limit_seconds": float(
                os.environ.get("PLDM_CALIBRATION_SECONDS", "720")
            ),
        }
        probing_loader = prep["actual_probing_loader"]
        probe_work_batches = probing_loader["total_batch_work_metadata_estimate"]
        probe_batch_size = probing_loader["configured_batch_size"]
        train_batch_size = prepared_loader["batch_size"]
        train_timed_mean = None
        report["probing_cost_estimate"] = {
            "status": "PENDING_TRAINING_TIMING",
            "official_probe_metadata": probing_loader,
            "measured_probe_forward_or_prober_training": False,
            "cost_proxy_basis": "Use the measured full official training update mean as a per-sample resource proxy scaled by actual probing/training batch sizes; no prober forward/train batch is timed in this allocation.",
            "probe_train_and_validation_batch_work": probe_work_batches,
            "probe_batch_size": probe_batch_size,
            "training_batch_size": train_batch_size,
        }
        report["memory"] = {
            "after_trainer_init_allocated_bytes": torch.cuda.memory_allocated(0),
            "after_trainer_init_reserved_bytes": torch.cuda.memory_reserved(0),
        }
        atomic_json(report_path, report, job_id)

        # The official Trainer's logger has no training effect. Keep in-memory calls
        # but disable scalar persistence so this is a timing run, not a quality report.
        Logger.run().output_path = None

        def quiet_tqdm(*args, **kwargs):
            kwargs["disable"] = True
            return tqdm_impl(*args, **kwargs)

        train_module.tqdm = quiet_tqdm
        validation_calls = 0

        def forbidden_validation(*args, **kwargs):
            nonlocal validation_calls
            validation_calls += 1
            raise RuntimeError("Native evaluation is forbidden in calibration")

        trainer.validate = forbidden_validation

        records: list[dict] = []
        max_seconds = float(os.environ.get("PLDM_CALIBRATION_SECONDS", "720"))
        deadline = process_started + max_seconds
        training_started = time.perf_counter()
        bounded_loader = BoundedLoader(
            trainer.ds, torch, records, TOTAL_UPDATES, deadline, training_started
        )
        trainer.ds = bounded_loader
        optimizer_update_count = 0
        original_create_optimizer = OptimizerFactory.create_optimizer

        def instrumented_create_optimizer(factory):
            nonlocal optimizer_update_count
            optimizer = original_create_optimizer(factory)
            original_step = optimizer.step

            def counted_step(*args, **kwargs):
                nonlocal optimizer_update_count
                result = original_step(*args, **kwargs)
                optimizer_update_count += 1
                return result

            optimizer.step = counted_step
            return optimizer

        OptimizerFactory.create_optimizer = instrumented_create_optimizer
        stop_reason = ""
        training_error = None
        try:
            trainer.train()
        except CalibrationStop as stop:
            stop_reason = str(stop)
        except Exception as exc:
            training_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-7000:],
            }
        finally:
            OptimizerFactory.create_optimizer = original_create_optimizer
        torch.cuda.synchronize()
        training_elapsed_seconds = time.perf_counter() - training_started
        train_records = [
            item["seconds_including_batch_wait_and_cuda_completion"] for item in records
        ]
        batch_wait_records = [item["batch_wait_seconds"] for item in records]
        update_body_records = [item["update_body_seconds_including_cuda_completion"] for item in records]
        warmup_times = train_records[:WARMUP_UPDATES]
        timed_times = train_records[WARMUP_UPDATES : TOTAL_UPDATES]
        timed_batch_wait = batch_wait_records[WARMUP_UPDATES : TOTAL_UPDATES]
        timed_update_body = update_body_records[WARMUP_UPDATES : TOTAL_UPDATES]
        complete_timing_sample = (
            training_error is None
            and optimizer_update_count == TOTAL_UPDATES
            and len(records) == TOTAL_UPDATES
            and len(timed_times) == MEASURED_UPDATES
        )

        def summarize(values: list[float]) -> dict | None:
            if not values:
                return None
            return {
                "count": len(values),
                "mean_seconds": sum(values) / len(values),
                "median_seconds": percentile(values, 0.5),
                "p95_seconds": percentile(values, 0.95),
            }

        training_summary = {
            "stop_reason": stop_reason,
            "training_error": training_error,
            "optimizer_updates_per_epoch": loader_length,
            "actual_epoch_indices": epoch_indices,
            "actual_passes": len(epoch_indices),
            "expected_full_official_updates": expected_updates,
            "calibration_updates_completed": optimizer_update_count,
            "step_timing_records_completed": len(records),
            "warmup_updates_requested": WARMUP_UPDATES,
            "warmup_updates_completed": len(warmup_times),
            "timed_updates_requested": MEASURED_UPDATES,
            "timed_updates_completed": len(timed_times),
            "per_update_wall_seconds_include": [
                "batch_wait_seconds: time in next(official_iterator)",
                "update_body_seconds_including_cuda_completion: time after next() through official update body and the existing torch.cuda.synchronize()",
            ],
            "warmup_seconds": warmup_times,
            "timed_update_seconds": timed_times,
            "batch_wait_seconds": batch_wait_records,
            "update_body_seconds_including_cuda_completion": update_body_records,
            "timed_batch_wait_seconds": timed_batch_wait,
            "timed_update_body_seconds_including_cuda_completion": timed_update_body,
            "timing_summary": summarize(timed_times),
            "timed_batch_wait_summary": summarize(timed_batch_wait),
            "timed_update_body_summary": summarize(timed_update_body),
            "full_training_cost_estimate_status": "READY" if complete_timing_sample else "NOT_ESTIMATED_INCOMPLETE_CALIBRATION",
            "total_calibration_training_seconds": training_elapsed_seconds,
            "optimizer_scheduler_setup_before_first_batch_seconds": bounded_loader.first_batch_start_offset_seconds,
            "estimated_official_training_update_seconds_at_timed_mean": (
                expected_updates * (sum(timed_times) / len(timed_times))
                if complete_timing_sample
                else None
            ),
            "native_evaluation_calls": validation_calls,
        }
        report["timing"].update(training_summary)
        if complete_timing_sample:
            train_timed_mean = sum(timed_times) / len(timed_times)
            probe_sample_equivalents = probe_work_batches * probe_batch_size / train_batch_size
            report["probing_cost_estimate"].update(
                {
                    "status": "PROXY_REPORTED",
                    "training_update_mean_seconds": train_timed_mean,
                    "probe_training_sample_equivalents_at_training_batch_size": probe_sample_equivalents,
                    "conservative_extra_probing_seconds_proxy": probe_sample_equivalents * train_timed_mean,
                    "limitation": "Not a measured prober cost or guaranteed upper bound. The full training update includes backbone backward work, while probing uses different forwards and may add visualization/plotting. Retain extra walltime margin and do not call this a measured evaluation estimate.",
                }
            )
        else:
            report["probing_cost_estimate"].update(
                {
                    "status": "NOT_ESTIMATED_INCOMPLETE_CALIBRATION",
                    "limitation": "The 110-update calibration did not complete; no probing cost proxy is derived from partial update timings.",
                }
            )
        report["memory"].update(
            {
                "training_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "training_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "after_training_allocated_bytes": torch.cuda.memory_allocated(0),
                "after_training_reserved_bytes": torch.cuda.memory_reserved(0),
            }
        )

        if training_error is None and optimizer_update_count == TOTAL_UPDATES and len(records) == TOTAL_UPDATES:
            save_started = time.perf_counter()
            torch.cuda.synchronize()
            trainer.save_model()
            torch.cuda.synchronize()
            checkpoint_path = Path(config.output_path) / f"epoch={trainer.epoch}_sample_step={trainer.sample_step}.ckpt"
            report["calibration_checkpoint"] = {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else None,
                "save_seconds": time.perf_counter() - save_started,
                "formal_checkpoint_reuse": False,
                "purpose": "isolated calibration-only checkpoint for measuring the official Trainer.save_model() I/O cost",
            }
        else:
            report["calibration_checkpoint"] = {
                "saved": False,
                "reason": "only a complete 110-update calibration writes an isolated calibration checkpoint",
            }

        training_complete = (
            training_error is None
            and optimizer_update_count == TOTAL_UPDATES
            and len(records) == TOTAL_UPDATES
            and len(timed_times) == MEASURED_UPDATES
        )
        if training_complete:
            planner_cost_report = run_native_planner_cost_probe(
                trainer, config, bounded_loader, torch, process_started
            )
        else:
            planner_cost_report = {
                "status": "SKIPPED_TRAINING_INCOMPLETE",
                "purpose": "cost only; no native evaluation or quality metric",
                "calls": [],
            }
        report["planner_cost_probe"] = planner_cost_report
        report["timing"]["planner_cost_probe"] = planner_cost_report
        report["training_status"] = "PASS" if training_complete else "PARTIAL" if training_error is None else "FAIL"

        report["ended_utc"] = utc_now()
        planner_status = planner_cost_report.get("status")
        report["status"] = (
            "PASS"
            if training_complete
            and validation_calls == 0
            and report.get("calibration_checkpoint", {}).get("bytes")
            and planner_status == "PASS"
            else "PARTIAL"
            if training_error is None
            else "FAIL"
        )
        atomic_json(report_path, report, job_id)
        print(
            f"GPU_CALIBRATION_{report['status']} updates={optimizer_update_count}/{TOTAL_UPDATES} "
            f"timed={len(timed_times)}/{MEASURED_UPDATES} native_eval_calls={validation_calls} "
            f"planner_cost_probe={planner_status}",
            flush=True,
        )
        return 0 if report["status"] == "PASS" else 2 if report["status"] == "PARTIAL" else 1
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()[-7000:],
        }
        report["ended_utc"] = utc_now()
        atomic_json(report_path, report, job_id)
        print(f"GPU_CALIBRATION_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
