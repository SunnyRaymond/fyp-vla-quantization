"""Run Wall-only Exp1 paired feature-error measurements on one PBS A100.

No torch/model import happens before the allocation and visible-GPU guards.
The five arms use ordinary FP32 operators with activation fake quantization;
this runner makes no native low-bit, storage, speed, or STaMP claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from allocation_guard import require_compute_allocation, verify_visible_a100
    from core import (
        ARMS,
        DEFAULT_EPISODES,
        HORIZONS,
        KNOWN_USED_EPISODES,
        PATCHES,
        PRIOR_EPISODES,
        SCHEMA,
        TAIL_DIM,
        VISUAL_DIM,
        make_state_record,
        mse,
        prepare_initial_treatments,
        requested_horizon,
        summarize_records,
        validate_episode_indices,
    )
except ModuleNotFoundError:  # direct import from another working directory
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    from allocation_guard import require_compute_allocation, verify_visible_a100
    from core import (  # type: ignore[no-redef]
        ARMS,
        DEFAULT_EPISODES,
        HORIZONS,
        KNOWN_USED_EPISODES,
        PATCHES,
        PRIOR_EPISODES,
        SCHEMA,
        TAIL_DIM,
        VISUAL_DIM,
        make_state_record,
        mse,
        prepare_initial_treatments,
        requested_horizon,
        summarize_records,
        validate_episode_indices,
    )


SOURCE_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CHECKPOINT_SHA256 = "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b"
STAMP_STATUS = {
    "name": "STaMP",
    "status": "pending",
    "implemented": False,
    "comparator_hook": "pending_stamp_comparator",
    "reason": "No reliable STaMP implementation was found in the pinned DINO-WM Wall runtime; it is not fabricated or executed.",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_smoke(path: Path | None):
    candidates = []
    if path is not None:
        candidates.append(path.resolve())
    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "smoke_runner.py",
            here.parent.parent / "world-model-quantization" / "experiments" / "dino-wm-wall" / "smoke_runner.py",
        ]
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("exp1_smoke_runner", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["exp1_smoke_runner"] = module
        spec.loader.exec_module(module)
        return module, candidate
    raise FileNotFoundError("pinned DINO-WM smoke_runner.py was not found; pass --smoke-runner explicitly")


def _to_tensor(value: Any, torch: Any, device: Any) -> Any:
    return value.to(device=device, dtype=torch.float32) if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32, device=device)


def _episode_inputs(runtime: Mapping[str, Any], index: int, frameskip: int, max_horizon: int, torch: Any) -> dict[str, Any]:
    dset = runtime["dset"]
    if index < 0:
        raise ValueError("episode index must be non-negative")
    if index >= len(dset):
        raise ValueError(f"episode index {index} is outside valid dataset length {len(dset)}")
    obs, actions, states, _ = dset[index]
    visual = obs.get("visual")
    proprio = obs.get("proprio")
    if visual is None or proprio is None:
        raise ValueError("WallDataset sample must contain visual and proprio")
    if tuple(visual.shape[1:]) != (3, 224, 224):
        raise ValueError(f"expected transformed visual [T,3,224,224], got {tuple(visual.shape)}")
    raw_actions = _to_tensor(actions, torch, "cpu")
    raw_proprio = _to_tensor(proprio, torch, "cpu")
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 2:
        raise ValueError(f"expected normalized Wall primitive actions [T,2], got {tuple(raw_actions.shape)}")
    if raw_proprio.ndim != 2 or raw_proprio.shape[1] != 2:
        raise ValueError(f"expected normalized Wall proprio [T,2], got {tuple(raw_proprio.shape)}")
    target_frame = max_horizon * frameskip
    if visual.shape[0] <= target_frame or raw_actions.shape[0] < target_frame:
        raise ValueError(f"episode {index} is too short for H={max_horizon}, frameskip={frameskip}")
    action_blocks = raw_actions[:target_frame].reshape(max_horizon, frameskip, 2).reshape(max_horizon, frameskip * 2)
    if hasattr(dset, "indices") and hasattr(dset, "dataset"):
        underlying_index = int(dset.indices[index])
        base_dset = dset.dataset
    elif hasattr(dset, "data_path") and type(dset).__name__ == "WallDataset":
        underlying_index = int(index)
        base_dset = dset
    else:
        raise RuntimeError("cannot prove valid-local to WallDataset episode identity; refusing to guess")
    data_path = Path(getattr(base_dset, "data_path", ""))
    if not data_path:
        raise RuntimeError("WallDataset data_path is missing; refusing an unidentifiable episode")
    return {
        "visual": _to_tensor(visual, torch, "cpu"),
        "proprio": raw_proprio,
        "actions": action_blocks,
        "states": _to_tensor(states, torch, "cpu"),
        "underlying_index": underlying_index,
        "trajectory_id": f"WallDataset:{data_path.as_posix()}#episode_{underlying_index:03d}",
        "dataset_path": data_path.as_posix(),
    }


def _predict_once(model: Any, current_z: Any, quantized_tail: np.ndarray | None, label: str, torch: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    if tuple(current_z.shape) != (1, 1, PATCHES, VISUAL_DIM + TAIL_DIM):
        raise ValueError(f"unexpected predictor state shape {tuple(current_z.shape)}")
    quantized_z = current_z.clone()
    if quantized_tail is not None:
        values = np.asarray(quantized_tail, dtype=np.float32)
        if values.shape != (PATCHES, TAIL_DIM) or not np.isfinite(values).all():
            raise ValueError(f"invalid intervention tail for {label}")
        quantized_z[0, 0, :, VISUAL_DIM:] = torch.as_tensor(values, dtype=torch.float32, device=device)
    expected_predictor_input = quantized_z[:, -1].reshape(1, PATCHES, VISUAL_DIM + TAIL_DIM)
    seen = []

    def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs:
            seen.append(inputs[0].detach().cpu().clone())

    before = current_z.clone()
    handle = model.predictor.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            prediction = model.predict(quantized_z)
    finally:
        handle.remove()
    audit = {
        "label": label,
        "hook_readback": bool(len(seen) == 1 and torch.equal(seen[0], expected_predictor_input.cpu())),
        "input_unchanged": bool(torch.equal(current_z, before)),
        "intervention": quantized_tail is not None,
    }
    audit["output_shape"] = list(prediction.shape)
    if not audit["hook_readback"] or not audit["input_unchanged"]:
        raise RuntimeError(f"predictor input readback gate failed for {label}")
    if tuple(prediction.shape) != (1, 1, PATCHES, VISUAL_DIM + TAIL_DIM):
        raise RuntimeError(f"unexpected predictor output shape {tuple(prediction.shape)} for {label}")
    if not bool(torch.isfinite(prediction).all().item()):
        raise FloatingPointError(f"non-finite predictor output for {label}")
    return prediction, audit


def _rollout_arm(model: Any, z0: Any, action_blocks: Any, arm: str, treatment: Mapping[str, Any], max_horizon: int, torch: Any, device: Any) -> tuple[dict[int, list[np.ndarray]], dict[str, Any]]:
    values = np.asarray(treatment["values"], dtype=np.float32)
    audit_spec = dict(treatment["audit"])
    outputs: dict[int, list[np.ndarray]] = {horizon: [] for horizon in HORIZONS}
    audits = []
    for draw, initial_tail in enumerate(values):
        current = z0.clone()
        prediction, audit = _predict_once(model, current, initial_tail, f"{arm}:initial_draw_{draw}", torch, device)
        audits.append(audit)
        for step in range(max_horizon):
            horizon = step + 1
            if requested_horizon(horizon):
                outputs[horizon].append(
                    prediction[0, 0, :, :VISUAL_DIM].detach().cpu().numpy().astype(np.float32, copy=True)
                )
            if step + 1 >= max_horizon:
                break
            next_z = prediction[:, -1:].clone()
            next_action = action_blocks[step + 1:step + 2].to(device=device).unsqueeze(0)
            with torch.no_grad():
                next_z = model.replace_actions_from_z(next_z, next_action)
            # Causal propagation stays FP32.  In particular, predictor-produced
            # non-broadcast proprio is never permanently overwritten.
            prediction, propagation_audit = _predict_once(model, next_z, None, f"{arm}:propagation_step_{step + 1}_draw_{draw}", torch, device)
            audits.append(propagation_audit)
    expected_draws = len(values)
    if any(len(outputs[horizon]) != expected_draws for horizon in HORIZONS):
        raise RuntimeError(f"missing requested horizon output for {arm}: {[len(outputs[h]) for h in HORIZONS]}")
    return outputs, {
        **audit_spec,
        "calls": len(audits),
        "quantized_calls": len(values),
        "propagation_calls": len(audits) - len(values),
        "hook_readback_all": bool(audits) and all(item["hook_readback"] for item in audits),
        "input_unchanged_all": bool(audits) and all(item["input_unchanged"] for item in audits),
        "output_shapes": sorted({tuple(item["output_shape"]) for item in audits}),
    }


def _state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _runtime_identity(runtime: Mapping[str, Any], smoke: Any, smoke_path: Path, root: Path) -> dict[str, Any]:
    identity = dict(smoke._checkpoint_identity(runtime))
    checkpoint = Path(runtime["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    actual_sha = _sha256(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA256 does not match the pinned Wall checkpoint")
    identity.update({
        "checkpoint_sha256": actual_sha,
        "source_commit_expected": SOURCE_COMMIT,
        "dinov2_source_commit_expected": DINOV2_COMMIT,
        "smoke_runner": str(smoke_path),
        "source_root": str((root / "source").resolve()),
    })
    if identity.get("source_commit") != SOURCE_COMMIT or identity.get("dinov2_source_commit") != DINOV2_COMMIT:
        raise RuntimeError("runtime source identity does not match the pinned DINO-WM revision")
    return identity


def run(args: argparse.Namespace, allocation: Mapping[str, Any], gpu: Mapping[str, Any]) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = validate_episode_indices(args.episode_indices, KNOWN_USED_EPISODES)
    max_horizon = max(HORIZONS)
    started = time.monotonic()
    smoke, smoke_path = _load_smoke(args.smoke_runner)
    runtime = smoke._runtime(args.root.resolve(), device="cuda:0")
    torch = runtime["torch"]
    device = runtime["device"]
    model = runtime["model"]
    model.eval()
    model_state_before = _state_digest(model)
    model_cfg = runtime["model_cfg"]
    frameskip = int(model_cfg.frameskip)
    if frameskip != args.expected_frameskip:
        raise RuntimeError(f"frameskip mismatch: runtime={frameskip}, expected={args.expected_frameskip}")
    if int(model_cfg.concat_dim) != 1 or int(model_cfg.num_hist) != 1 or int(model_cfg.num_pred) != 1:
        raise RuntimeError("Exp1 requires concat_dim=1, num_hist=1, num_pred=1")
    identity = _runtime_identity(runtime, smoke, smoke_path, args.root)
    manifest = {
        "schema": SCHEMA,
        "status": "running",
        "experiment": "Wall-only broadcast-coupling Exp1",
        "allocation": dict(allocation),
        "gpu": dict(gpu),
        "runtime_identity": identity,
        "episode_indices": list(episodes),
        "excluded_prior_episode_indices": list(KNOWN_USED_EPISODES),
        "trajectory_id_rule": "WallDataset:<data_path>#episode_<underlying_index>; identity is mandatory",
        "current_frame": 0,
        "frameskip": frameskip,
        "horizons": list(HORIZONS),
        "arms": list(ARMS),
        "base_seed": int(args.base_seed),
        "seed_rule": "SHA256(base_seed|episode_index|arm|step), first 31 bits",
        "broadcast_draw_seeds": [2501, 2502, 2503],
        "broadcast_offset_seed": 2601,
        "intervention_scope": "initial_exact_broadcast_tail_only; subsequent autoregressive calls remain FP32",
        "matched_pair": ["shared_stochastic", "balanced_broadcast"],
        "iid_patch_stochastic_role": "non-matched negative control",
        "stamp": STAMP_STATUS,
        "execution": "fake_activation_quantization_fp32_operators",
        "native_low_bit_claim": False,
    }
    _atomic_json(output / "manifest.json", manifest)
    records: list[dict[str, Any]] = []
    records_path = output / "per_state_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    with records_path.open("a", encoding="utf-8") as stream:
        for episode_index in episodes:
            sample = _episode_inputs(runtime, episode_index, frameskip, max_horizon, torch)
            visual = sample["visual"].to(device=device)
            proprio = sample["proprio"].to(device=device)
            action_blocks = sample["actions"]
            obs0 = {"visual": visual[None, 0:1], "proprio": proprio[None, 0:1]}
            initial_action = action_blocks[0:1].to(device=device).unsqueeze(0)
            with torch.no_grad():
                z0 = model.encode(obs0, initial_action)
            if tuple(z0.shape) != (1, 1, PATCHES, VISUAL_DIM + TAIL_DIM):
                raise RuntimeError(f"episode {episode_index} produced unexpected z shape {tuple(z0.shape)}")
            if not bool(torch.isfinite(z0).all().item()):
                raise FloatingPointError(f"non-finite initial z for episode {episode_index}")
            target_features: dict[int, np.ndarray] = {}
            for horizon in HORIZONS:
                frame = horizon * frameskip
                target_obs = {"visual": visual[None, frame:frame + 1], "proprio": proprio[None, frame:frame + 1]}
                with torch.no_grad():
                    target = model.encode_obs(target_obs)["visual"][0, 0, :, :VISUAL_DIM]
                target_features[horizon] = target.detach().cpu().numpy().astype(np.float32, copy=True)
            initial_tail = z0[0, 0, :, VISUAL_DIM:].detach().cpu().numpy().astype(np.float32, copy=True)
            if not np.array_equal(initial_tail, np.broadcast_to(initial_tail[0:1], initial_tail.shape)):
                raise RuntimeError("initial exact-broadcast z tail is not identical across patches")
            treatments = prepare_initial_treatments(initial_tail, args.base_seed, episode_index)
            raw_path = output / f"raw_episode_{episode_index:03d}.npz"
            np.savez_compressed(
                raw_path,
                episode_index=np.asarray(episode_index, dtype=np.int64),
                underlying_index=np.asarray(sample["underlying_index"], dtype=np.int64),
                initial_tail=initial_tail,
                fp32=treatments["FP32"]["values"],
                rtn=treatments["RTN"]["values"],
                shared_stochastic=treatments["shared_stochastic"]["values"],
                iid_patch_stochastic=treatments["iid_patch_stochastic"]["values"],
                balanced_broadcast=treatments["balanced_broadcast"]["values"],
            )
            arm_outputs: dict[str, dict[int, list[np.ndarray]]] = {}
            arm_audits: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                arm_outputs[arm], arm_audits[arm] = _rollout_arm(model, z0, action_blocks, arm, treatments[arm], max_horizon, torch, device)
            fp_outputs = arm_outputs["FP32"]
            for horizon in HORIZONS:
                metrics = {}
                for arm in ARMS:
                    predictions = arm_outputs[arm][horizon]
                    fp_predictions = fp_outputs[horizon]
                    draw_metrics = []
                    for draw, prediction in enumerate(predictions):
                        fp_reference = fp_predictions[min(draw, len(fp_predictions) - 1)]
                        draw_metrics.append({
                            "draw": draw,
                            "real_future_feature_mse": mse(prediction, target_features[horizon]),
                            "fp_reference_mse": mse(prediction, fp_reference),
                        })
                    metrics[arm] = {
                        "real_future_feature_mse": float(np.mean([item["real_future_feature_mse"] for item in draw_metrics])),
                        "fp_reference_mse": float(np.mean([item["fp_reference_mse"] for item in draw_metrics])),
                        "draw_metrics": draw_metrics,
                        "n_draws": len(predictions),
                        "calls": arm_audits[arm]["calls"],
                        "quantized_calls": arm_audits[arm]["quantized_calls"],
                        "propagation_calls": arm_audits[arm]["propagation_calls"],
                        "hook_readback_all": arm_audits[arm]["hook_readback_all"],
                        "input_unchanged_all": arm_audits[arm]["input_unchanged_all"],
                        "quantizer": arm_audits[arm],
                    }
                record = make_state_record(
                    episode_index=episode_index,
                    trajectory_id=sample["trajectory_id"],
                    current_frame=0,
                    future_frame=horizon * frameskip,
                    horizon=horizon,
                    arm_outputs=metrics,
                )
                record["raw_treatment_path"] = str(raw_path.resolve())
                record["raw_treatment_sha256"] = _sha256(raw_path)
                records.append(record)
                stream.write(json.dumps(record, default=_json_default) + "\n")
                stream.flush()
    state_after = _state_digest(model)
    if state_after != model_state_before:
        raise RuntimeError("model state changed during inference")
    summary = summarize_records(records, episodes)
    summary.update({
        "status": "complete",
        "allocation": dict(allocation),
        "gpu": dict(gpu),
        "runtime_identity": identity,
        "episode_indices": list(episodes),
        "excluded_prior_episode_indices": list(KNOWN_USED_EPISODES),
        "records": len(records),
        "base_seed": int(args.base_seed),
        "frameskip": frameskip,
        "model_state_before": model_state_before,
        "model_state_after": state_after,
        "model_state_unchanged": True,
        "stamp": STAMP_STATUS,
        "elapsed_seconds": time.monotonic() - started,
        "records_path": str(records_path),
    })
    _atomic_json(output / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["records"] = len(records)
    manifest["summary"] = str((output / "summary.json").resolve())
    manifest["elapsed_seconds"] = summary["elapsed_seconds"]
    _atomic_json(output / "manifest.json", manifest)
    return summary


def _parse_indices(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episode indices must be comma-separated integers") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="prepared DINO-WM Wall runtime root")
    parser.add_argument("--output", type=Path, required=True, help="artifact directory")
    parser.add_argument("--smoke-runner", type=Path, default=None, help="pinned smoke_runner.py path")
    parser.add_argument("--episode-indices", type=_parse_indices, default=DEFAULT_EPISODES, help="fresh valid-local indices, comma separated")
    parser.add_argument("--base-seed", type=int, default=910001)
    parser.add_argument("--expected-frameskip", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        allocation = require_compute_allocation()
        # This is the second guard, after the stdlib-only PBS/account/resource
        # check.  It runs before loading the smoke runner, torch, or a model.
        gpu = verify_visible_a100()
        summary = run(args, allocation, gpu)
        print(json.dumps({key: summary[key] for key in ("schema", "status", "episode_indices", "horizons", "arms", "records", "elapsed_seconds") if key in summary}, indent=2, default=_json_default), flush=True)
        return 0
    except Exception as exc:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "summary.json", {
            "schema": "broadcast-coupling-wall-exp1-summary-v1",
            "status": "failed",
            "engineering_pass": False,
            "decision": "implementation_inconclusive",
            "error": f"{type(exc).__name__}: {exc}",
            "stamp": STAMP_STATUS,
        })
        print(f"Exp1 refused/failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
