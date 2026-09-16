"""Four-arm analytic padded-coordinate intervention screen for SmolVLA.

The CPU-prepared eight-sample manifest is the only source of sample selection.
This runner reuses the reviewed Flow 64763 runtime, preprocessor and W4
quantizer, but implements its own four-arm sampler recorder so it does not
inherit the old twelve-episode runner or its summary writer.  Allocation guard
is the first workload action; no model/data work is safe outside a real CCDS
SLURM compute allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
SCHEMA = "analytic-padding-screen-v1"
RAW_SCHEMA = "analytic-padding-raw-v1"
MANIFEST_SCHEMA = "padded-coordinate-raw-input-manifest-v1"
IDENTITY_SCHEMA = "smolvla-cpu-preparation-identity-v1"
ARM_NAMES = ("FPnative", "Qnative", "FPanalytic", "Qanalytic")
NOISE_SEEDS = (1801, 1802)
EPISODE_COUNT = 8
VELOCITY_STEPS = 10
HORIZON = 50
MAX_ACTION_DIM = 32
PHYSICAL_ACTION_DIM = 7
PAD_DIM = MAX_ACTION_DIM - PHYSICAL_ACTION_DIM
DT = -0.1
BITS = 4
QMAX = 7
PAD_TOLERANCE = 1e-5
MAX_WORKLOAD_SECONDS = 720.0
FLOW_SCREEN_SHA256 = "ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9"
def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)
def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def _file_identity(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size if path.is_file() else None}
def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)
def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
def _write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB ({len(encoded)} bytes)")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
def _check_deadline(started: float, limit: float, label: str) -> None:
    if time.monotonic() - started >= limit:
        raise TimeoutError(f"max-seconds reached at {label}")
def _load_guard() -> Any:
    return importlib.import_module("allocation_guard").require_allocation
def _load_flow(helper_dir: Path) -> Any:
    helper_dir = helper_dir.resolve()
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    flow = importlib.import_module("flow_screen")
    actual = _sha256_file(Path(flow.__file__).resolve())
    if actual != FLOW_SCREEN_SHA256:
        raise RuntimeError(f"flow_screen.py identity mismatch: {actual!r} != {FLOW_SCREEN_SHA256!r}")
    return flow
def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value
def _resolve_under(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise RuntimeError(f"{label} escapes asset root: {path}")
    return path
def _expert_state_names(policy: Any, modules: Sequence[Tuple[str, Any]]) -> set[str]:
    by_parameter: Dict[int, List[str]] = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        by_parameter.setdefault(id(parameter), []).append(name)
    names: set[str] = set()
    for label, module in modules:
        matched = by_parameter.get(id(module.weight), [])
        if not matched:
            raise RuntimeError(f"expert module has no named state weight: {label}")
        names.update(matched)
    if not names or not names.issubset(policy.state_dict()):
        raise RuntimeError("expert exclusion does not bind actual state_dict weights")
    return names
def _load_input_manifest(path: Path, asset_root: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any], List[Dict[str, Any]]]:
    raw = _load_json(path.resolve(), "padded input manifest")
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported padded manifest schema: {raw.get('schema')!r}")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) != EPISODE_COUNT:
        raise RuntimeError("padded manifest must contain exactly eight samples")
    identity_value = raw.get("base_identity_path")
    if not isinstance(identity_value, str) or not identity_value:
        raise RuntimeError("padded manifest has no base_identity_path")
    identity_path = _resolve_under(asset_root, identity_value, "base_identity_path")
    identity_hash = raw.get("base_identity_sha256", raw.get("identity_sha256"))
    actual_identity_hash = _sha256_file(identity_path)
    if not isinstance(identity_hash, str) or actual_identity_hash != identity_hash.casefold():
        raise RuntimeError("base identity SHA-256 mismatch")
    identity = _load_json(identity_path, "base identity")
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise RuntimeError("base identity schema mismatch")
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise RuntimeError(f"sample {index} is not a mapping")
        try:
            task_index = int(item["task_index"])
            episode_index = int(item.get("episode_index", item.get("dataset_episode_index")))
            length = int(item["length"])
            frame_index = int(item["frame_index"])
            task_text = str(item["task_text"])
            sample_value = str(item["sample_path"])
            sample_hash = str(item.get("sha256", item.get("sample_sha256"))).casefold()
            camera_keys = [str(value) for value in item["camera_keys"]]
            state_key = str(item["state_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"sample {index} has incomplete metadata") from exc
        if task_index not in range(4) or episode_index < 0 or length <= 0 or frame_index != length // 4:
            raise RuntimeError(f"sample {index} violates frozen task/frame metadata")
        if not task_text.strip() or not camera_keys or len(set(camera_keys)) != len(camera_keys) or not state_key:
            raise RuntimeError(f"sample {index} has malformed feature metadata")
        sample_path = _resolve_under(path.parent, sample_value, f"sample {index}")
        if asset_root.resolve() not in sample_path.parents:
            raise RuntimeError(f"sample {index} is outside the asset root")
        if not sample_path.is_file() or _sha256_file(sample_path) != sample_hash:
            raise RuntimeError(f"sample {index} path/hash mismatch")
        normalized.append({
            "episode_index": index,
            "dataset_episode_index": episode_index,
            "task_index": task_index,
            "task_id": str(task_index),
            "task_text": task_text,
            "length": length,
            "frame_index": frame_index,
            "sample_path": str(sample_path),
            "sample_sha256": sample_hash,
            "camera_keys": camera_keys,
            "state_key": state_key,
        })
    pairs = [(item["task_index"], item["dataset_episode_index"]) for item in normalized]
    if pairs != sorted(pairs) or len(set(pairs)) != EPISODE_COUNT:
        raise RuntimeError("padded samples must be sorted and task/episode distinct")
    if any(sum(item["task_index"] == task for item in normalized) != 2 for task in range(4)):
        raise RuntimeError("padded manifest must contain two samples for each task 0..3")
    return raw, identity, normalized
def _noise_tensors(torch: Any, device: Any) -> Tuple[List[Any], np.ndarray]:
    tensors = []
    for seed in NOISE_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        tensors.append(torch.randn((1, HORIZON, MAX_ACTION_DIM), generator=generator, dtype=torch.float32).to(device))
    return tensors, np.stack([tensor.detach().cpu().numpy()[0] for tensor in tensors]).astype(np.float32)
def _finite_shape(value: Any, shape: Tuple[int, ...], torch: Any, label: str) -> None:
    if not torch.is_tensor(value) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{label} must be finite with shape {shape}, got {getattr(value, 'shape', None)}")
class _AnalyticRecorder:
    def __init__(self, policy: Any, torch: Any, analytic: bool):
        self.policy = policy
        self.model = policy.model
        self.torch = torch
        self.analytic = analytic
    def run(self, batch: Mapping[str, Any], noise: Any) -> Dict[str, Any]:
        original = self.model.denoise_step
        predicted: List[Any] = []
        used: List[Any] = []
        inputs: List[Any] = []
        times: List[float] = []
        pad = noise[..., PHYSICAL_ACTION_DIM:].detach().clone()
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            x_t = kwargs.get("x_t")
            timestep = kwargs.get("timestep")
            if x_t is None or timestep is None:
                raise RuntimeError("official denoise_step did not expose x_t/timestep keywords")
            value = original(*args, **kwargs)
            _finite_shape(value, (1, HORIZON, MAX_ACTION_DIM), self.torch, "predicted velocity")
            inputs.append(x_t.detach().clone())
            predicted.append(value.detach().clone())
            times.append(float(timestep.detach().reshape(-1)[0].cpu().item()))
            actual = value.clone()
            if self.analytic:
                actual[..., PHYSICAL_ACTION_DIM:] = pad
            used.append(actual.detach().clone())
            return actual
        self.model.denoise_step = wrapped
        try:
            with self.torch.no_grad():
                self.policy.reset()
                output = self.policy.predict_action_chunk(_clone_batch(batch, self.torch), noise=noise.clone())
        finally:
            self.model.denoise_step = original
        if getattr(self.model.denoise_step, "__func__", None) is not getattr(original, "__func__", None):
            raise RuntimeError("denoise_step wrapper was not restored")
        _finite_shape(output, (1, HORIZON, PHYSICAL_ACTION_DIM), self.torch, "physical action output")
        if len(predicted) != VELOCITY_STEPS:
            raise RuntimeError(f"official sampler made {len(predicted)} denoise calls, expected {VELOCITY_STEPS}")
        return {
            "action": output.detach().cpu().numpy()[0].astype(np.float32),
            "predicted": self.torch.stack(predicted, dim=0).detach().cpu().numpy()[:, 0].astype(np.float32),
            "used": self.torch.stack(used, dim=0).detach().cpu().numpy()[:, 0].astype(np.float32),
            "inputs": self.torch.stack(inputs, dim=0).detach().cpu().numpy()[:, 0].astype(np.float32),
            "times": np.asarray(times, dtype=np.float32),
        }
def _clone_batch(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _clone_batch(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch(item, torch) for item in value)
    if isinstance(value, str):
        return value
    raise RuntimeError(f"unsupported batch value: {type(value).__name__}")
def _write_raw(output: Path, arrays: Mapping[str, Any]) -> None:
    _atomic_npz(output / "raw_analytic_padding.npz", **arrays)
def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_raw, base_identity, entries = _load_input_manifest(args.manifest.resolve(), args.asset_root.resolve())
    _check_deadline(started, args.max_seconds, "input manifest")
    engineering: Dict[str, Any] = {
        "schema": "analytic-padding-engineering-v1", "status": "manifest_verified", "allocation": dict(allocation),
        "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256_file(args.manifest.resolve()), "schema": manifest_raw["schema"], "sample_count": len(entries)},
        "base_identity": {"path": str(args.asset_root.resolve() / "smolvla" / "identity.json"), "sha256": _sha256_file(_resolve_under(args.asset_root, str(manifest_raw["base_identity_path"]), "base_identity_path"))},
        "episode_ids": [f"task{e['task_index']}:episode{e['dataset_episode_index']}:frame{e['frame_index']}" for e in entries],
        "complete8": False, "raw_npz": str((output / "raw_analytic_padding.npz").resolve()), "metrics_deferred_to_root_cpu_verifier": True,
    }
    _atomic_json(output / "engineering.json", engineering)
    flow = _load_flow(args.helper_dir)
    import torch
    args.manifest_data = {"identity": base_identity}
    policy, torch, runtime_identity = flow._load_runtime(args, torch)
    gpu_identity = flow._gpu_evidence(torch)
    if runtime_identity.get("checkpoint", {}).get("sha256") is None:
        raise RuntimeError("checkpoint hash is unavailable")
    if not runtime_identity.get("source_identity"):
        raise RuntimeError("runtime source identity is missing")
    _check_deadline(started, args.max_seconds, "runtime load")
    engineering.update({"flow_screen_sha256": FLOW_SCREEN_SHA256, "helper": {"path": str(Path(flow.__file__).resolve()), "sha256": FLOW_SCREEN_SHA256}, "source_hashes": runtime_identity.get("source_identity", {}), "runtime_identity": runtime_identity, "checkpoint": runtime_identity["checkpoint"], "gpu": gpu_identity})
    _atomic_json(output / "engineering.json", engineering)
    preprocessor, preprocessor_identity = flow._load_preprocessor(args.model_path.resolve(), args.vlm_path.resolve())
    batches = []
    for entry in entries:
        raw_batch = flow._load_raw_sample(Path(entry["sample_path"]), entry, torch)
        prepared, _ = flow._prepare_batch(raw_batch, preprocessor, torch)
        batches.append(flow._move_to_device(prepared, torch.device("cuda:0"), torch))
    noises, noise_np = _noise_tensors(torch, torch.device("cuda:0"))
    _check_deadline(started, args.max_seconds, "input batches")
    modules = flow._eligible_modules(policy, torch)
    expert_modules = modules.get("expert_W4", [])
    if not expert_modules:
        raise RuntimeError("expert W4 allowlist is empty")
    expert_state_names = _expert_state_names(policy, expert_modules)
    snapshot = flow._snapshot_weights(modules, torch)
    bypass_digest = flow._state_subset_digest(policy, expert_state_names, torch)
    arrays: Dict[str, Any] = {
        "schema": np.asarray(RAW_SCHEMA), "arm_names": np.asarray(ARM_NAMES), "noise": noise_np,
        "times": np.full((VELOCITY_STEPS,), np.nan, dtype=np.float32),
        "predicted_velocity": np.full((EPISODE_COUNT, 2, 4, VELOCITY_STEPS, HORIZON, MAX_ACTION_DIM), np.nan, dtype=np.float32),
        "used_velocity": np.full((EPISODE_COUNT, 2, 4, VELOCITY_STEPS, HORIZON, MAX_ACTION_DIM), np.nan, dtype=np.float32),
        "x_inputs": np.full((EPISODE_COUNT, 2, 4, VELOCITY_STEPS, HORIZON, MAX_ACTION_DIM), np.nan, dtype=np.float32),
        "actions": np.full((EPISODE_COUNT, 2, 4, HORIZON, PHYSICAL_ACTION_DIM), np.nan, dtype=np.float32),
        "episode_ids": np.asarray([f"task{e['task_index']}:episode{e['dataset_episode_index']}:frame{e['frame_index']}" for e in entries]),
        "completed": np.zeros((EPISODE_COUNT,), dtype=np.bool_),
    }
    _write_raw(output, arrays)
    engineering.update({"preprocessor": preprocessor_identity, "architecture": {"arms": list(ARM_NAMES), "velocity_shape": list(arrays["predicted_velocity"].shape), "action_shape": list(arrays["actions"].shape), "noise_shape": list(noise_np.shape), "times_shape": [VELOCITY_STEPS], "num_steps": VELOCITY_STEPS, "dt": DT}, "quantizer": {"scope": "expert_W4 only", "bits": BITS, "qmax": QMAX, "module_count": len(expert_modules), "expert_state_names": sorted(expert_state_names), "bypassed_state_digest_before": bypass_digest}})
    _atomic_json(output / "engineering.json", engineering)
    noop_result = _AnalyticRecorder(policy, torch, analytic=False).run(batches[0], noises[0].clone())
    with torch.no_grad():
        policy.reset()
        noop_direct = policy.predict_action_chunk(_clone_batch(batches[0], torch), noise=noises[0].clone())
    _finite_shape(noop_direct, (1, HORIZON, PHYSICAL_ACTION_DIM), torch, "no-op direct action")
    noop_exact = bool(np.array_equal(noop_result["action"], noop_direct.detach().cpu().numpy()[0].astype(np.float32)))
    if not noop_exact:
        raise RuntimeError("FPnative no-op recorder changed sampler action")
    _check_deadline(started, args.max_seconds, "no-op gate")
    engineering["no_op"] = {"action_exact_equal": True, "wrapper_restored": True, "sample": 0, "noise_seed": NOISE_SEEDS[0]}
    _atomic_json(output / "engineering.json", engineering)
    branch_checks: List[Dict[str, Any]] = []
    for episode, batch in enumerate(batches):
        for noise_index, noise in enumerate(noises):
            for arm_index, arm in enumerate(ARM_NAMES):
                flow._restore_weights(snapshot, modules, torch)
                if arm in ("Qnative", "Qanalytic"):
                    quantizer = flow._quantize_locus(expert_modules, torch)
                else:
                    quantizer = None
                if flow._state_subset_digest(policy, expert_state_names, torch) != bypass_digest:
                    raise RuntimeError(f"bypassed state changed before {arm}")
                result = _AnalyticRecorder(policy, torch, arm in ("FPanalytic", "Qanalytic")).run(batch, noise.clone())
                arrays["predicted_velocity"][episode, noise_index, arm_index] = result["predicted"]
                arrays["used_velocity"][episode, noise_index, arm_index] = result["used"]
                arrays["x_inputs"][episode, noise_index, arm_index] = result["inputs"]
                arrays["actions"][episode, noise_index, arm_index] = result["action"]
                if not np.isfinite(result["predicted"]).all() or not np.isfinite(result["used"]).all() or not np.isfinite(result["inputs"]).all() or not np.isfinite(result["action"]).all():
                    raise FloatingPointError(f"non-finite output episode={episode}, noise={noise_index}, arm={arm}")
                _check_deadline(started, args.max_seconds, f"episode {episode} noise {noise_index} arm {arm}")
                if episode == 0 and noise_index == 0 and arm_index == 0:
                    arrays["times"] = result["times"]
                    expected_times = np.asarray([1.0 + step * DT for step in range(VELOCITY_STEPS)], dtype=np.float32)
                    if not np.allclose(arrays["times"], expected_times, atol=1e-6, rtol=0.0):
                        raise RuntimeError(f"official sampler time grid mismatch: {arrays['times']!r}")
                elif not np.array_equal(arrays["times"], result["times"]):
                    raise RuntimeError("official sampler time grid changed across arms")
                branch_checks.append({"episode": episode, "noise": NOISE_SEEDS[noise_index], "arm": arm, "quantized": quantizer is not None})
        arrays["completed"][episode] = True
        _write_raw(output, arrays)
        engineering.update({"status": "partial", "completed_episodes": int(arrays["completed"].sum()), "branch_count": len(branch_checks)})
        _atomic_json(output / "engineering.json", engineering)
    flow._restore_weights(snapshot, modules, torch)
    if flow._state_subset_digest(policy, expert_state_names, torch) != bypass_digest:
        raise RuntimeError("bypassed state changed after final restore")
    predicted = arrays["predicted_velocity"]
    used = arrays["used_velocity"]
    inputs = arrays["x_inputs"]
    analytic = np.array([2, 3])
    native = np.array([0, 1])
    max_pad_path = float(np.max(np.abs(inputs[:, :, analytic, :, :, PHYSICAL_ACTION_DIM:] - arrays["times"][None, None, None, :, None, None] * noise_np[None, :, None, None, :, PHYSICAL_ACTION_DIM:])))
    first_fp = float(np.max(np.abs(predicted[:, :, 0, 0, :, :PHYSICAL_ACTION_DIM] - predicted[:, :, 2, 0, :, :PHYSICAL_ACTION_DIM])))
    first_q = float(np.max(np.abs(predicted[:, :, 1, 0, :, :PHYSICAL_ACTION_DIM] - predicted[:, :, 3, 0, :, :PHYSICAL_ACTION_DIM])))
    first_input_error = float(np.max(np.abs(inputs[:, :, 1:, 0] - inputs[:, :, :1, 0])))
    native_exact = bool(np.array_equal(predicted[:, :, native], used[:, :, native]))
    analytic_physical_exact = bool(np.array_equal(predicted[:, :, analytic, :, :, :PHYSICAL_ACTION_DIM], used[:, :, analytic, :, :, :PHYSICAL_ACTION_DIM]))
    analytic_pad_error = float(np.max(np.abs(used[:, :, analytic, :, :, PHYSICAL_ACTION_DIM:] - noise_np[None, :, None, None, :, PHYSICAL_ACTION_DIM:])))
    expected_times = np.asarray([1.0 + step * DT for step in range(VELOCITY_STEPS)], dtype=np.float32)
    time_grid_error = float(np.max(np.abs(arrays["times"] - expected_times)))
    gates = {"complete8": bool(arrays["completed"].all()), "no_op_action_exact": noop_exact, "native_used_equals_predicted": native_exact, "analytic_physical_slice_exact": analytic_physical_exact, "first_step_input_exact": first_input_error == 0.0, "first_step_fp_physical_exact": first_fp == 0.0, "first_step_q_physical_exact": first_q == 0.0, "analytic_used_pad_noise_exact": analytic_pad_error == 0.0, "analytic_x_pad_time_noise": max_pad_path <= PAD_TOLERANCE, "time_grid": time_grid_error <= 1e-6, "weight_restore_exact": True}
    engineering.update({"status": "complete" if all(gates.values()) else "engineering_failed", "complete8": bool(arrays["completed"].all()), "completed_episodes": int(arrays["completed"].sum()), "engineering_gates": {**gates, "first_step_input_max_abs": first_input_error, "first_step_fp_physical_equality_max_abs": first_fp, "first_step_q_physical_equality_max_abs": first_q, "analytic_used_pad_noise_max_abs": analytic_pad_error, "analytic_x_pad_minus_t_noise_max_abs": max_pad_path, "time_grid_max_abs": time_grid_error}, "weight_restore": {"method": "flow._restore_weights", "exact_elementwise": True}, "metrics_deferred_to_root_cpu_verifier": True})
    _atomic_json(output / "engineering.json", engineering)
    _write_raw(output, arrays)
    if not all(gates.values()):
        raise RuntimeError(f"engineering gate failed: {[key for key, value in gates.items() if not value]}")
    _write_summary(output / "summary.json", {"schema": SCHEMA, "status": "complete", "allocation": dict(allocation), "completed_episodes": int(arrays["completed"].sum()), "raw_npz": str((output / "raw_analytic_padding.npz").resolve()), "engineering": str((output / "engineering.json").resolve()), "metrics_deferred_to_root_cpu_verifier": True, "unresolved": ["CPU verifier must compute E0/S/E1/E2 and binding/gain gates", "analytic screen is an offline intervention and does not claim success, deployment or native W4 performance"], "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "complete", "episodes": EPISODE_COUNT, "raw": str(output / "raw_analytic_padding.npz")}, indent=2), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, required=True)
    parser.add_argument("--lerobot-source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_WORKLOAD_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_WORKLOAD_SECONDS:g}]")
    return args
def main() -> None:
    require_allocation = _load_guard()
    allocation = require_allocation()
    args = _parse_args()
    try:
        _run(args, allocation)
    except Exception as exc:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        _write_summary(output / "summary.json", {"schema": SCHEMA, "status": "implementation_failure", "allocation": allocation, "error": f"{type(exc).__name__}: {exc}", "raw_npz": str((output / "raw_analytic_padding.npz").resolve()), "engineering": str((output / "engineering.json").resolve()), "metrics_deferred_to_root_cpu_verifier": True})
        raise
if __name__ == "__main__":
    main()
