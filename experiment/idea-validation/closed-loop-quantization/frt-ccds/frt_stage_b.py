"""Stage B common-bank runner for the FRT CCDS pilot.

This module is deliberately separate from the Stage A entry point.  It uses
the source adapter, quantizer and fit routine from ``frt_runner`` and emits
the complete evaluator-by-bank matrix required by ``verify_frt.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


METHODS = ("clean", "random_same_norm", "frt")
SEEDS = (1201, 1202, 1203)
BANK_IDS = (
    "q0", "random_same_norm",
    "fresh:clean:1201", "fresh:clean:1202", "fresh:clean:1203",
    "fresh:random_same_norm:1201", "fresh:random_same_norm:1202", "fresh:random_same_norm:1203",
    "fresh:frt:1201", "fresh:frt:1202", "fresh:frt:1203",
)
EVALUATOR_IDS = (
    "fp32", "q0_rtn",
    "clean:1201", "clean:1202", "clean:1203",
    "random_same_norm:1201", "random_same_norm:1202", "random_same_norm:1203",
    "frt:1201", "frt:1202", "frt:1203",
)


def _base() -> Any:
    import frt_runner

    return frt_runner


def _slot_numpy(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _mse(error: np.ndarray, wz: np.ndarray) -> np.ndarray:
    weighted = error * wz.reshape((1,) * (error.ndim - 2) + wz.shape)
    return np.mean(weighted * weighted, axis=(-2, -1), dtype=np.float64)


def _map_hash(modules: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for path, module in modules:
        digest.update(path.encode("utf-8"))
        digest.update(module.weight.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _record_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    keys = []
    episodes = []
    indices = []
    env_seeds = []
    cem_seeds = []
    for position, row in enumerate(rows):
        episode = str(row.get("episode_id", row.get("episode", "")))
        record_index = int(row.get("record_index", row.get("local_index", position)))
        keys.append(f"{episode}:record-{record_index:03d}")
        episodes.append(episode)
        indices.append(int(row["dataset_index"]))
        env_seeds.append(int(row.get("env_seed", -1)))
        cem_seeds.append(int(row.get("cem_seed", -1)))
    if len(keys) != len(set(keys)):
        raise ValueError("DEV record keys are duplicated")
    return {
        "record_keys": np.asarray(keys, dtype="U"),
        "episode_ids": np.asarray(episodes, dtype="U"),
        "dataset_indices": np.asarray(indices, dtype=np.int64),
        "env_seeds": np.asarray(env_seeds, dtype=np.int64),
        "cem_seeds": np.asarray(cem_seeds, dtype=np.int64),
    }


def _fresh_bank(
    runtime: Mapping[str, Any],
    modules: Sequence[tuple[str, Any]],
    snapshot: Mapping[str, Any],
    dev_bank: Mapping[str, Any],
    adapter: Any,
) -> dict[str, Any]:
    """Collect a fitted model's fresh Q-theta direction on the DEV rows."""
    base = _base()
    core = base._core()
    torch = runtime["torch"]
    model = runtime["model"]
    device = next(model.parameters()).device
    data = base._bank_device(dev_bank, device)
    hard_snapshot = {path: module.weight.detach().clone() for path, module in modules}
    with torch.no_grad():
        theta_current = adapter.one_step(data["history"], data["action"])
        theta_history = adapter.append(data["history"], theta_current)
        theta_next = adapter.one_step(theta_history, data["next_action"])
        core.restore_modules(modules, snapshot)
        fp_theta_next = adapter.one_step(theta_history, data["next_action"])
        theta_delta = core.mask_action_delta(theta_next - fp_theta_next, adapter.action_mask(theta_next))
        theta_base_history = adapter.append(theta_history, fp_theta_next)
    core.restore_modules(modules, hard_snapshot)
    return {
        "x_history": theta_base_history.detach().cpu(),
        "next_action": data["next_action"].detach().cpu(),
        "delta": theta_delta.detach().cpu(),
        "history": theta_history.detach().cpu(),
        "fp_next": fp_theta_next.detach().cpu(),
    }


def _fp_targets(
    runtime: Mapping[str, Any],
    modules: Sequence[tuple[str, Any]],
    snapshot: Mapping[str, Any],
    adapter: Any,
    contexts: Sequence[Any],
    deltas: Sequence[Any],
    actions: Sequence[Any],
) -> list[Any]:
    base = _base()
    core = base._core()
    torch = runtime["torch"]
    model = runtime["model"]
    previous = {path: module.weight.detach().clone() for path, module in modules}
    core.restore_modules(modules, snapshot)
    result = []
    with torch.no_grad():
        for context, delta, action in zip(contexts, deltas, actions):
            device = next(model.parameters()).device
            result.append(core.transport(adapter, context.to(device), delta.to(device), action.to(device)).detach().cpu())
    core.restore_modules(modules, previous)
    return result


def _fit_config(manifest: Mapping[str, Any], a_summary: Mapping[str, Any]) -> tuple[dict[str, Any], int, int, float, float]:
    fit = manifest.get("fit", {})
    fit = dict(fit) if isinstance(fit, Mapping) else {}
    freeze = a_summary.get("fit_freeze", {})
    freeze = freeze if isinstance(freeze, Mapping) else {}
    steps = fit.get("fit_updates", freeze.get("fit_updates"))
    if steps is None:
        raise ValueError("Stage B requires manifest.fit.fit_updates or an explicit A fit_freeze")
    batch = fit.get("batch_size", freeze.get("batch_size"))
    if batch is None:
        raise ValueError("Stage B requires manifest.fit.batch_size or an explicit A fit_freeze")
    optimizer = fit.get("optimizer", {})
    optimizer = dict(optimizer) if isinstance(optimizer, Mapping) else {}
    lr = optimizer.get("lr", fit.get("lr", freeze.get("lr")))
    if lr is None:
        raise ValueError("Stage B requires an explicit frozen optimizer learning rate")
    loss = fit.get("loss", {})
    loss = dict(loss) if isinstance(loss, Mapping) else {}
    declared_lambda = manifest.get("loss", {}).get("lambda_T") if isinstance(manifest.get("loss"), Mapping) else None
    a_lambda = a_summary.get("calibration_freeze", {}).get("lambda_T") if isinstance(a_summary.get("calibration_freeze"), Mapping) else None
    lambda_t = a_lambda if a_lambda is not None else declared_lambda
    if lambda_t is None:
        raise ValueError("Stage B requires frozen lambda_T from the manifest or Stage A")
    loss["lambda_T"] = float(lambda_t)
    fit["fit_updates"] = int(steps)
    fit["batch_size"] = int(batch)
    optimizer["lr"] = float(lr)
    fit["optimizer"] = optimizer
    fit["loss"] = loss
    return fit, int(steps), int(batch), float(lr), float(lambda_t)


def _fit_record(
    result: Mapping[str, Any],
    modules: Sequence[tuple[str, Any]],
    seed: int,
    steps: int,
    batch: int,
    lr: float,
    n_records: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    base = _base()
    core = base._core()
    schedule = [base._batch_indices(n_records, min(batch, n_records), seed, step).tolist() for step in range(steps)]
    fit = dict(result)
    checkpoint = Path(str(result["checkpoint"]))
    fit.update({
        "status": "complete",
        "fit_updates_completed": int(steps),
        "fit_updates_expected": int(steps),
        "exposure_count": int(sum(len(item) for item in schedule)),
        "batch_schedule_hash": core.stable_hash(schedule),
        "initialization_id": core.stable_hash({"spec": "W4Spec", "seed": int(seed), "source": "Q0"}),
        "soft_schedule_id": core.stable_hash({"rounding": fit.get("rounding"), "spec": "W4Spec"}),
        "scale_learning_rate": float(lr),
        "rounding_learning_rate": float(lr),
        "checkpoint_sha256": core.sha256_file(checkpoint),
        "hard_map_sha256": _map_hash(modules),
        "binary_final": all(not hasattr(module, "parametrizations") or "weight" not in module.parametrizations for _, module in modules),
        "alpha_discarded": True,
        "exact_w4": bool(fit.get("hard_ledger") and all(row.get("hard") and row.get("q_min") == -7 and row.get("q_max") == 7 for row in fit["hard_ledger"])),
        "reload_equal": False,
    })
    return fit


def _reload_checkpoint(
    runtime: Mapping[str, Any],
    modules: Sequence[tuple[str, Any]],
    snapshot: Mapping[str, Any],
    adapter: Any,
    bank: Mapping[str, Any],
    fit: MutableMapping[str, Any],
) -> None:
    base = _base()
    core = base._core()
    torch = runtime["torch"]
    model = runtime["model"]
    device = next(model.parameters()).device
    checkpoint = torch.load(fit["checkpoint"], map_location=device)
    with torch.no_grad():
        before = adapter.one_step(bank["history"].to(device), bank["action"].to(device)).detach().cpu()
    core.restore_modules(modules, snapshot)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        after = adapter.one_step(bank["history"].to(device), bank["action"].to(device)).detach().cpu()
    fit["reload_equal"] = bool(torch.equal(before.to(device), after.to(device)))


def _score(
    runtime: Mapping[str, Any],
    modules: Sequence[tuple[str, Any]],
    adapter: Any,
    dev_bank: Mapping[str, Any],
    contexts: Sequence[Any],
    deltas: Sequence[Any],
    actions: Sequence[Any],
    fp_targets: Sequence[Any],
    wz: np.ndarray,
    clean_reference: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = _base()
    core = base._core()
    torch = runtime["torch"]
    model = runtime["model"]
    device = next(model.parameters()).device
    data = base._bank_device(dev_bank, device)
    with torch.no_grad():
        # ``fp_current`` in the per-record bank was collected one record at a
        # time.  A CUDA matmul can round differently for that batch shape than
        # for the common DEV batch used by this matrix.  Use one FP snapshot
        # reference collected with the same full DEV batch as every evaluator.
        clean_output = adapter.one_step(data["history"], data["action"])
        clean_error = _slot_numpy(clean_output - clean_reference.to(device))
        transport_errors = []
        for context, delta, action, target in zip(contexts, deltas, actions, fp_targets):
            predicted = core.transport(adapter, context.to(device), delta.to(device), action.to(device))
            transport_errors.append(_slot_numpy(predicted - target.to(device)))
    clean_mse = _mse(clean_error, wz)
    transport_error = np.stack(transport_errors, axis=0)
    transport_mse = _mse(transport_error, wz)
    return clean_error, transport_error, clean_mse, transport_mse


def _fp_clean_reference(
    runtime: Mapping[str, Any],
    modules: Sequence[tuple[str, Any]],
    snapshot: Mapping[str, Any],
    adapter: Any,
    dev_bank: Mapping[str, Any],
) -> Any:
    """Compute the shared FP clean target in the evaluator batch shape."""
    base = _base()
    core = base._core()
    torch = runtime["torch"]
    model = runtime["model"]
    device = next(model.parameters()).device
    previous = {path: module.weight.detach().clone() for path, module in modules}
    core.restore_modules(modules, snapshot)
    data = base._bank_device(dev_bank, device)
    with torch.no_grad():
        reference = adapter.one_step(data["history"], data["action"]).detach().cpu()
    core.restore_modules(modules, previous)
    return reference


def _load_fit_checkpoint(runtime: Mapping[str, Any], fit: Mapping[str, Any]) -> None:
    torch = runtime["torch"]
    model = runtime["model"]
    model.load_state_dict(torch.load(fit["checkpoint"], map_location=next(model.parameters()).device)["state_dict"], strict=True)


def run_stage_b(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    args: argparse.Namespace,
    allocation: Mapping[str, Any],
    gpu: Mapping[str, Any],
    output: Path,
    progress_path: Path | None = None,
    progress_state: MutableMapping[str, Any] | None = None,
    loaded: tuple[Any, Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all nine fits and emit a square common DEV evaluator matrix."""
    base = _base()
    core = base._core()
    if progress_state is None:
        progress_state = {"completed": []}
    output.mkdir(parents=True, exist_ok=True)
    if getattr(args, "a_summary", None):
        a_summary_path = Path(args.a_summary).resolve()
    else:
        a_summary_path = output / "stage_a_summary.json"
        if not a_summary_path.is_file():
            a_summary_path = output.parent / "stage_a_summary.json"
    a_summary = base._json(a_summary_path)
    if a_summary.get("schema") != "frt-stage-a-v1" or not a_summary.get("engineering_pass"):
        raise RuntimeError("Stage A engineering_pass is false; Stage B is locked")
    root = base._resolve(args.root or manifest.get("root") or manifest.get("runtime_root"), manifest_path.parent)
    if loaded is None:
        smoke, runtime, runtime_identity = base._load_runtime(manifest, root, getattr(args, "device", None))
    else:
        smoke, runtime, runtime_identity = loaded
    runtime_identity = dict(runtime_identity)
    if "gpu" not in runtime_identity and gpu.get("gpus"):
        runtime_identity["gpu"] = gpu["gpus"][0].get("name")
    base._check_identity(manifest, runtime_identity, gpu)
    modules, groups = base._target_modules(smoke, runtime, manifest)
    torch = runtime["torch"]
    model = runtime["model"]
    model.eval()
    adapter = core.SourceHistoryAdapter(model)
    cal_raw, dev_raw = base._split_sources(manifest, manifest_path.parent)
    cal_raw = base._validate_split_rows(cal_raw, "cal", require_fresh=True)
    dev_raw = base._validate_split_rows(dev_raw, "dev", require_fresh=True)
    base._validate_split_pair(cal_raw, dev_raw)
    for split, rows in (("cal", cal_raw), ("dev", dev_raw)):
        expected = base._declared_indices(manifest, split)
        if expected is not None and {int(row["dataset_index"]) for row in rows} != expected:
            raise ValueError(f"{split} dataset indices disagree with manifest")
    device = next(model.parameters()).device
    cal_rows = base._expand_batched_rows(cal_raw, adapter, torch, device, "cal")
    dev_rows = base._expand_batched_rows(dev_raw, adapter, torch, device, "dev")
    if len({str(row["episode_id"]) for row in cal_rows}) != 6 or len({str(row["episode_id"]) for row in dev_rows}) != 6:
        raise ValueError("CAL/DEV must each contain exactly six episode identities")
    snapshot = core.snapshot_modules(modules, cpu=True)
    cal_bank = base._build_bank(runtime, modules, snapshot, cal_rows, adapter, "cal", int(manifest.get("random_seed", 940000)))
    dev_bank = base._build_bank(runtime, modules, snapshot, dev_rows, adapter, "dev", int(manifest.get("random_seed", 940000)) + 1, wz=cal_bank["wz"])
    base._save_bank(output / "bank_cal.npz", cal_bank, core)
    base._save_bank(output / "bank_dev.npz", dev_bank, core)
    fit_cfg, steps, batch, lr, lambda_t = _fit_config(manifest, a_summary)
    fit_manifest = dict(manifest)
    fit_manifest["fit"] = fit_cfg
    fit_runs: list[dict[str, Any]] = []
    fresh: dict[str, dict[str, Any]] = {}
    fit_paths: list[str] = []
    fit_by_id: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for seed in SEEDS:
            key = f"{method}:{seed}"
            method_path = output / "methods" / f"{method}_seed_{seed}.json"
            if key in progress_state.get("completed", []) and method_path.is_file():
                fit = base._json(method_path)
                _load_fit_checkpoint(runtime, fit)
            else:
                core.set_seed(seed)
                fit = base._fit_method(runtime, modules, snapshot, cal_bank, adapter, method, seed, fit_manifest, output, progress_path or output / "progress.json", progress_state)
                fit = _fit_record(fit, modules, seed, steps, batch, lr, len(cal_rows), manifest)
                _reload_checkpoint(runtime, modules, snapshot, adapter, dev_bank, fit)
                core.atomic_json(method_path, fit)
                completed = list(progress_state.get("completed", []))
                if key not in completed:
                    completed.append(key)
                progress_state["completed"] = completed
                if progress_path is not None:
                    base._progress(progress_path, progress_state, completed=completed, current_method=None, current_step=None)
            _load_fit_checkpoint(runtime, fit)
            fresh[key] = _fresh_bank(runtime, modules, snapshot, dev_bank, adapter)
            fit["fresh_bank_id"] = f"fresh:{key}"
            if method_path.is_file():
                core.atomic_json(method_path, fit)
            fit_by_id[key] = fit
            fit_runs.append({key_name: fit.get(key_name) for key_name in ("method", "seed", "status", "fit_updates_completed", "fit_updates_expected", "batch_size", "lr", "lambda_transport", "exposure_count", "batch_schedule_hash", "initialization_id", "soft_schedule_id", "scale_learning_rate", "rounding_learning_rate", "checkpoint_sha256", "hard_map_sha256", "binary_final", "alpha_discarded", "reload_equal", "exact_w4")})
            fit_paths.append(str(method_path.resolve()))
            core.restore_modules(modules, snapshot)
    record_arrays = _record_arrays(dev_rows)
    wz_vector = np.asarray(dev_bank["wz"].detach().cpu().numpy(), dtype=np.float32)
    wz = wz_vector
    action_mask = dev_bank["action_mask"].detach().cpu().numpy().astype(bool) if hasattr(dev_bank.get("action_mask"), "detach") else np.asarray(adapter.action_mask(dev_bank["delta"][:1])[0, 0].cpu().numpy(), dtype=bool)
    active_mask = ~action_mask
    if wz.ndim == 1:
        wz = np.broadcast_to(wz.reshape(1, -1), action_mask.shape).copy()
    bank_deltas_t = [dev_bank["delta"].detach().cpu(), dev_bank["random_delta"].detach().cpu()]
    contexts = [dev_bank["x_history"], dev_bank["x_history"]]
    actions = [dev_bank["next_action"], dev_bank["next_action"]]
    for key in (f"{method}:{seed}" for method in METHODS for seed in SEEDS):
        bank_deltas_t.append(fresh[key]["delta"])
        contexts.append(fresh[key]["x_history"])
        actions.append(fresh[key]["next_action"])
    bank_deltas = np.stack([_slot_numpy(value) for value in bank_deltas_t], axis=0).astype(np.float32)
    # Freeze one clean FP reference using the same full DEV batch consumed by
    # every evaluator.  The per-record ``fp_current`` values remain useful for
    # provenance, but are intentionally not used for the common matrix because
    # their one-record CUDA matmul can differ by a few ulps.
    fp_clean_reference_t = _fp_clean_reference(runtime, modules, snapshot, adapter, dev_bank)
    fp_clean_reference = _slot_numpy(fp_clean_reference_t).astype(np.float32)
    fp_targets = _fp_targets(runtime, modules, snapshot, adapter, contexts, bank_deltas_t, actions)
    clean_error_rows = []
    transport_error_rows = []
    clean_mse_rows = []
    transport_mse_rows = []
    evaluator_states: list[str] = ["fp32", "q0_rtn"] + [f"{method}:{seed}" for method in METHODS for seed in SEEDS]
    for evaluator in evaluator_states:
        if evaluator == "fp32":
            core.restore_modules(modules, snapshot)
        elif evaluator == "q0_rtn":
            core.restore_modules(modules, snapshot)
            core.materialize_rtn(modules, core.W4Spec())
        else:
            core.restore_modules(modules, snapshot)
            _load_fit_checkpoint(runtime, fit_by_id[evaluator])
        clean_error, transport_error, clean_mse, transport_mse = _score(runtime, modules, adapter, dev_bank, contexts, bank_deltas_t, actions, fp_targets, wz, fp_clean_reference_t)
        clean_error_rows.append(clean_error.astype(np.float32))
        transport_error_rows.append(transport_error.astype(np.float32))
        clean_mse_rows.append(clean_mse.astype(np.float64))
        transport_mse_rows.append(transport_mse.astype(np.float64))
    core.restore_modules(modules, snapshot)
    clean_error = np.stack(clean_error_rows, axis=0)
    transport_error = np.stack(transport_error_rows, axis=0)
    clean_mse = np.stack(clean_mse_rows, axis=0)
    transport_mse = np.stack(transport_mse_rows, axis=0)
    common = {
        **record_arrays,
        "active_mask": active_mask,
        "action_mask": action_mask,
        "wz": wz,
        "bank_deltas": bank_deltas,
        "fp_clean_reference": fp_clean_reference,
        "clean_error": clean_error,
        "transport_error": transport_error,
        "clean_mse": clean_mse,
        "transport_mse": transport_mse,
        "bank_ids": np.asarray(BANK_IDS, dtype="U"),
        "evaluator_ids": np.asarray(EVALUATOR_IDS, dtype="U"),
        "bank_x_history": np.stack([value.numpy() for value in contexts], axis=0).astype(np.float32),
        "bank_next_action": np.stack([value.numpy() for value in actions], axis=0).astype(np.float32),
        "fp_transport_target": np.stack([value.numpy() for value in fp_targets], axis=0).astype(np.float32),
    }
    stage_b_arrays = output / "stage_b.npz"
    core.atomic_npz(stage_b_arrays, common)
    common_arrays = output / "common_bank.npz"
    try:
        if common_arrays.exists():
            common_arrays.unlink()
        os.link(stage_b_arrays, common_arrays)
    except OSError:
        core.atomic_npz(common_arrays, common)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    stage_b_meta = {
        "schema": "frt-stage-b-raw-v1", "manifest_sha256": manifest_hash,
        "runtime_identity": runtime_identity, "target_groups": base._target_modules_identity(groups),
        "completion_status": "complete", "complete": True,
        "common_bank_all_evaluators": True, "bank_ids": list(BANK_IDS), "evaluator_ids": list(EVALUATOR_IDS),
        "record_count": len(dev_rows), "record_keys": record_arrays["record_keys"].tolist(),
        "episode_ids": record_arrays["episode_ids"].tolist(), "dataset_indices": record_arrays["dataset_indices"].tolist(),
        "stage_a_wz_sha256": hashlib.sha256(wz_vector.reshape(-1).tobytes()).hexdigest(),
        "fit_config": {"fit_updates": steps, "batch_size": batch, "scale_learning_rate": lr, "rounding_learning_rate": lr, "schedule_hash": core.stable_hash({"steps": steps, "batch": batch})},
        "fit_runs": fit_runs, "lambda_T": lambda_t, "method_files": fit_paths,
        "clean_reference": {
            "array": "fp_clean_reference",
            "source": "FP snapshot evaluated once with the full DEV evaluator batch",
            "shape": list(fp_clean_reference.shape),
            "dtype": str(fp_clean_reference.dtype),
        },
    }
    core.atomic_json(output / "stage_b.json", stage_b_meta)
    summary = {
        "schema": "frt-stage-b-v1", "status": "complete", "engineering_pass": True,
        "allocation": dict(allocation), "gpu": gpu, "runtime_identity": runtime_identity,
        "target_groups": base._target_modules_identity(groups), "cal_records": len(cal_rows), "dev_records": len(dev_rows),
        "cal_episodes": sorted({str(row["episode_id"]) for row in cal_rows}), "dev_episodes": sorted({str(row["episode_id"]) for row in dev_rows}),
        "fit_updates": steps, "batch_size": batch, "lr": lr, "lambda_T": lambda_t,
        "methods": list(METHODS), "seeds": list(SEEDS), "fit_runs": fit_paths,
        "common_bank": str((output / "common_bank.npz").resolve()), "common_bank_shape": {"bank_deltas": list(bank_deltas.shape), "clean_error": list(clean_error.shape), "transport_error": list(transport_error.shape), "clean_mse": list(clean_mse.shape), "transport_mse": list(transport_mse.shape)},
        "test_opened": False, "execution": "fake_quant_emulation_only",
    }
    core.atomic_json(output / "stage_b_summary.json", summary)
    core.atomic_json(output / "summary.json", summary)
    if progress_path is not None:
        base._progress(progress_path, progress_state, status="complete", result="stage_b_summary.json")
    return summary


if __name__ == "__main__":
    raise SystemExit("Use the Stage B entry point with an explicit verified allocation context")
