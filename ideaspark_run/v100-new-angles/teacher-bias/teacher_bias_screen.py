"""Bounded teacher-bias screen for the Wall epoch65 predictor.

This runner intentionally stops at recorded-feature comparison.  It does not
roll an environment, use a goal, or make a deployment claim.  The CPU
verifier consumes target_visual and pred_visual from the raw NPZ.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


INPUT_SCHEMA = "teacher-bias-recorded-future-manifest-v1"
SAMPLE_SCHEMA = "dino-wm-wall-recorded-future-raw-v1"
RAW_SCHEMA = "teacher-bias-raw-v1"
VALID_INDICES = tuple(range(124, 130))
FRAME_INDICES = (0, 5, 25)
ARM_NAMES = ("FP32", "W4", "W8")
CHECKPOINT_SHA256 = "8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b"
SOURCE_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SMOKE_SHA256 = "de5c5eb19f4b26614e71f9e2af36db21e9fd5bcc65188f3191772552bc750af9"
SCREEN_SHA256 = "51c2463a92a3bf84eabae735eeaa9771a7202add0b2227fe06c7fe1f2bd69d19"
PREDICTOR_BITS = {"W4": 4, "W8": 8}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _as_scalar_text(value: Any, label: str) -> str:
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise RuntimeError(f"{label} must be a scalar string")
    text = str(arr.item())
    if not text:
        raise RuntimeError(f"{label} is empty")
    return text


def _load_helpers(helper_dir: Path) -> tuple[Any, Any, dict[str, str]]:
    helper_dir = helper_dir.resolve()
    smoke_path = helper_dir / "smoke_runner.py"
    screen_path = helper_dir / "screen_runner.py"
    if not smoke_path.is_file() or not screen_path.is_file():
        raise RuntimeError("immutable smoke_runner.py and screen_runner.py are required")
    hashes = {"smoke_runner.py": _sha256(smoke_path), "screen_runner.py": _sha256(screen_path)}
    if hashes["smoke_runner.py"] != SMOKE_SHA256 or hashes["screen_runner.py"] != SCREEN_SHA256:
        raise RuntimeError(f"helper hash mismatch: {hashes}")
    sys.path.insert(0, str(helper_dir))
    for name in ("smoke_runner", "screen_runner"):
        sys.modules.pop(name, None)
    smoke = importlib.import_module("smoke_runner")
    screen = importlib.import_module("screen_runner")
    return smoke, screen, hashes


def _manifest_samples(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != INPUT_SCHEMA or manifest.get("raw_schema") != SAMPLE_SCHEMA:
        raise RuntimeError(f"manifest schema mismatch: {manifest.get('schema')!r}")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, dict):
        raise RuntimeError("manifest must carry source_identity")
    checkpoint_identity = manifest.get("checkpoint")
    if (source_identity.get("commit") != SOURCE_COMMIT or not isinstance(checkpoint_identity, dict)
            or checkpoint_identity.get("dinov2_source_commit") != DINOV2_COMMIT
            or checkpoint_identity.get("expected_sha256") != CHECKPOINT_SHA256):
        raise RuntimeError("manifest source identity mismatch")
    rows = manifest.get("samples")
    if not isinstance(rows, list) or len(rows) != len(VALID_INDICES):
        raise RuntimeError("manifest must contain exactly six samples")
    samples: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("sample record is not an object")
        idx = int(row.get("valid_local_index", -1))
        if idx not in VALID_INDICES or idx in seen:
            raise RuntimeError(f"invalid or duplicate validation index: {idx}")
        seen.add(idx)
        if any(bad <= idx <= 95 for bad in range(84, 96)):
            raise RuntimeError("reserved Wall indices 84..95 are forbidden")
        if list(row.get("frame_indices", [])) != list(FRAME_INDICES):
            raise RuntimeError("frame contract must be [0, 5, 25]")
        rel = row.get("sample_path")
        expected_hash = row.get("sha256")
        source_id = row.get("underlying_trajectory_id")
        if not isinstance(rel, str) or not isinstance(expected_hash, str) or not isinstance(source_id, (int, float)):
            raise RuntimeError("sample_path, sha256, and underlying_trajectory_id are required")
        source_id = int(source_id)
        trajectory_id = f"WallDataset:{source_id}"
        sample_path = (manifest_path.parent / rel).resolve()
        if manifest_path.parent not in sample_path.parents or not sample_path.is_file():
            raise RuntimeError(f"sample path is missing or escapes manifest directory: {rel}")
        actual_hash = _sha256(sample_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"sample hash mismatch for {idx}")
        with np.load(sample_path, allow_pickle=False) as raw:
            if "metadata_json" not in raw:
                raise RuntimeError(f"metadata_json missing for {idx}")
            metadata = json.loads(_as_scalar_text(raw["metadata_json"], "metadata_json"))
            if not isinstance(metadata, dict) or metadata.get("schema") != SAMPLE_SCHEMA:
                raise RuntimeError(f"sample metadata schema mismatch for {idx}")
            if int(metadata.get("valid_local_index", -1)) != idx or int(metadata.get("underlying_trajectory_id", -1)) != source_id:
                raise RuntimeError(f"sample metadata identity mismatch for {idx}")
            if list(metadata.get("frame_indices", [])) != list(FRAME_INDICES):
                raise RuntimeError(f"sample frame identity mismatch for {idx}")
            if "visual_0_5_25" not in raw or "proprio_0_5_25" not in raw or "model_actions_h5" not in raw:
                raise RuntimeError(f"visual/proprio missing for {idx}")
            visual = np.array(raw["visual_0_5_25"], copy=True)
            proprio = np.array(raw["proprio_0_5_25"], copy=True)
            action_blocks = np.array(raw["model_actions_h5"], copy=True)
        if proprio.shape != (3, 2) or not np.isfinite(proprio).all():
            raise RuntimeError(f"proprio must be finite [3,2] for {idx}")
        if action_blocks.shape != (5, 10) or not np.isfinite(action_blocks).all():
            raise RuntimeError(f"action blocks must be finite [5,10] for {idx}")
        normalization = str(metadata.get("dataset_normalization", "")).lower()
        if "normalize_action=true" not in normalization or "normalized once" not in normalization:
            raise RuntimeError("CPU preparation must provide already dataset-normalized proprio/actions")
        visual_space = "dataset_float_chw"
        if visual.ndim != 4 or visual.shape[0] != 3 or visual.shape[1] != 3 or visual.dtype not in (np.float32, np.float64):
            raise RuntimeError(f"prepared visual must be float [3,3,H,W] for {idx}")
        samples.append({
            "valid_local_index": idx,
            "trajectory_id": trajectory_id,
            "underlying_trajectory_id": source_id,
            "sample_path": str(sample_path),
            "sample_sha256": actual_hash,
            "metadata": metadata,
            "visual": visual,
            "visual_space": visual_space,
            "proprio": proprio.astype(np.float32, copy=False),
            "action_blocks": action_blocks.astype(np.float32, copy=False),
        })
    samples.sort(key=lambda x: x["valid_local_index"])
    if tuple(x["valid_local_index"] for x in samples) != VALID_INDICES:
        raise RuntimeError("validation samples are not exactly 124..129")
    return manifest, samples


def _runtime_structure(runtime: dict[str, Any], smoke: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = runtime["model"]
    cfg = runtime["model_cfg"]
    dset = runtime["dset"]
    def attr(obj: Any, key: str) -> Any:
        if not hasattr(obj, key):
            raise RuntimeError(f"missing structure attribute {key}")
        return getattr(obj, key)
    checks = {
        "num_hist": int(attr(model, "num_hist")),
        "num_pred": int(attr(model, "num_pred")),
        "concat_dim": int(attr(model, "concat_dim")),
        "num_action_repeat": int(attr(model, "num_action_repeat")),
        "num_proprio_repeat": int(attr(model, "num_proprio_repeat")),
        "model_action_dim": int(attr(model, "action_dim")),
        "model_proprio_dim": int(attr(model, "proprio_dim")),
        "dataset_action_dim": int(dset.action_dim),
        "dataset_proprio_dim": int(dset.proprio_dim),
        "frameskip": int(getattr(cfg, "frameskip")),
        "encoder_emb_dim": int(getattr(model.encoder, "emb_dim")),
        "model_emb_dim": int(getattr(model, "emb_dim")),
    }
    expected = {
        "num_hist": 1, "num_pred": 1, "concat_dim": 1, "num_action_repeat": 1,
        "num_proprio_repeat": 1, "model_action_dim": 10, "model_proprio_dim": 10,
        "dataset_action_dim": 2, "dataset_proprio_dim": 2, "frameskip": 5,
        "encoder_emb_dim": 384, "model_emb_dim": 404,
    }
    if checks != expected:
        raise RuntimeError(f"Wall epoch65 structure mismatch: {checks}")
    groups = smoke._linear_groups(model)
    predictor_groups = [g for g in groups if str(g.get("family", "")).lower() == "predictor"]
    if len(predictor_groups) != 6:
        raise RuntimeError(f"expected six predictor groups, got {len(predictor_groups)}")
    predictor_linears = [item for group in predictor_groups for item in group.get("linear", [])]
    if len(predictor_linears) != 24:
        raise RuntimeError(f"expected 24 predictor Linear weights, got {len(predictor_linears)}")
    checks["predictor_linear_count"] = len(predictor_linears)
    checks["predictor_linear_paths"] = [str(item["path"]) for item in predictor_linears]
    return groups, checks


def _source_identity(root: Path, runtime: dict[str, Any], smoke: Any, screen_hashes: dict[str, str]) -> dict[str, Any]:
    checkpoint = Path(runtime["checkpoint"]).resolve()
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA256 mismatch")
    identity = dict(smoke._checkpoint_identity(runtime))
    identity["checkpoint_sha256"] = CHECKPOINT_SHA256
    if identity.get("source_commit") != SOURCE_COMMIT or identity.get("dinov2_source_commit") != DINOV2_COMMIT:
        raise RuntimeError(f"source commit mismatch: {identity}")
    source_root = root / "source"
    source_files = [
        source_root / "models" / "visual_world_model.py",
        source_root / "datasets" / "wall_dset.py",
        source_root / "datasets" / "traj_dset.py",
        source_root / "datasets" / "img_transforms.py",
        source_root / "preprocessor.py",
        source_root / "plan.py",
        source_root / "models" / "dino.py",
        source_root / "env" / "__init__.py",
    ]
    source_hashes: dict[str, str] = {}
    for path in source_files:
        if not path.is_file():
            raise RuntimeError(f"required source file missing: {path}")
        source_hashes[str(path.relative_to(source_root))] = _sha256(path)
    identity["source_file_sha256"] = source_hashes
    identity["helper_sha256"] = screen_hashes
    identity["root"] = str(root)
    return identity


def _bind_prepared_runtime(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify the CPU-frozen executable inputs before loading model or data."""
    source = root / "source"
    actual_commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    modified = subprocess.run(["git", "-C", str(source), "diff", "--name-only", "HEAD"],
        check=True, capture_output=True, text=True, timeout=20).stdout.splitlines()
    if actual_commit != SOURCE_COMMIT or set(modified) - {"models/dino.py", "env/__init__.py"}:
        raise RuntimeError("source checkout differs from approved CPU preparation")
    records = list(manifest["source_identity"]["files"].values())
    records += [manifest["checkpoint"]["config"], manifest["checkpoint"]["file"]]
    for record in records:
        path = Path(record["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise RuntimeError("prepared runtime path is missing or outside model root")
        if path.stat().st_size != int(record["size"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"prepared runtime identity mismatch: {path.name}")
    expected_config = root / "checkpoints/outputs/wall_single/hydra.yaml"
    expected_checkpoint = root / "checkpoints/outputs/wall_single/checkpoints/model_latest.pth"
    if (Path(manifest["checkpoint"]["config"]["path"]).resolve() != expected_config.resolve()
            or Path(manifest["checkpoint"]["file"]["path"]).resolve() != expected_checkpoint.resolve()):
        raise RuntimeError("manifest runtime paths differ from immutable helper paths")
    return {"actual_git_commit": actual_commit, "modified_tracked_paths": modified,
            "prepared_source_config_checkpoint_verified": True,
            "checkpoint_config_sha256": manifest["checkpoint"]["config"]["sha256"]}


def _visual_tensor(sample: dict[str, Any], preprocessor: Any, torch: Any, device: Any) -> Any:
    if sample["visual_space"] == "uint8_hwc":
        visual = preprocessor.transform_obs_visual(sample["visual"][None, ...])
    else:
        visual = torch.as_tensor(sample["visual"], dtype=torch.float32).unsqueeze(0)
    visual = visual.to(device=device, dtype=torch.float32)
    if tuple(visual.shape[:3]) != (1, 3, 3):
        raise RuntimeError(f"unexpected visual tensor shape {tuple(visual.shape)}")
    return visual


def _make_observations(sample: dict[str, Any], preprocessor: Any, torch: Any, device: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visual = _visual_tensor(sample, preprocessor, torch, device)
    proprio = torch.as_tensor(sample["proprio"], dtype=torch.float32, device=device).unsqueeze(0)
    if tuple(proprio.shape) != (1, 3, 2):
        raise RuntimeError(f"unexpected proprio shape {tuple(proprio.shape)}")
    obs0 = {"visual": visual[:, 0:1], "proprio": proprio[:, 0:1]}
    targets = [{"visual": visual[:, i:i + 1], "proprio": proprio[:, i:i + 1]} for i in (1, 2)]
    return obs0, targets


def _cached_rollout(model: Any, z0: Any, actions: Any, torch: Any) -> tuple[dict[str, Any], Any]:
    z = z0.clone()
    for step in range(1, 5):
        z_pred = model.predict(z[:, -1:])
        z_new = z_pred[:, -1:]
        z_new = model.replace_actions_from_z(z_new, actions[:, step:step + 1])
        z = torch.cat((z, z_new), dim=1)
    z_pred = model.predict(z[:, -1:])
    z = torch.cat((z, z_pred[:, -1:]), dim=1)
    z_obs = model.separate_emb(z)[0]
    return z_obs, z


def _snapshot_equal(model: Any, snapshot: dict[str, Any], smoke: Any) -> bool:
    for name, value in snapshot.items():
        family, index, relative = name.split(".", 2)
        current = smoke._module_for_path(model, family, int(index), relative).weight.detach()
        if not bool(__import__("torch").equal(current, value)):
            return False
    return True


def _quantize_predictor(model: Any, predictor_groups: list[dict[str, Any]], bits: int, smoke: Any, torch: Any) -> list[dict[str, Any]]:
    qmax = 2 ** (bits - 1) - 1
    records: list[dict[str, Any]] = []
    for group in predictor_groups:
        for item in group.get("linear", []):
            path = str(item["path"])
            module = smoke._module_for_path(model, group["family"], group["index"], item["relative"])
            weight = module.weight.detach()
            fp_weight = weight.clone()
            rows = weight.reshape(weight.shape[0], -1)
            max_abs = rows.abs().amax(dim=1, keepdim=True)
            scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / float(qmax))
            codes = torch.round(rows / scale).clamp(-qmax, qmax)
            dequant = codes * scale
            with torch.no_grad():
                module.weight.copy_(dequant.reshape_as(weight))
            if not torch.equal(module.weight.detach(), dequant.reshape_as(weight)):
                raise RuntimeError(f"quantized readback mismatch: {path}")
            records.append({
                "path": path,
                "bits": bits,
                "shape": list(weight.shape),
                "fp_weight": fp_weight.cpu(),
                "codes": codes.detach().cpu().reshape(weight.shape).clone(),
                "scale": scale.detach().cpu().reshape(-1).clone(),
                "dequant_weight": dequant.detach().cpu().reshape(weight.shape).clone(),
            })
    return records


def _write_raw(path: Path, arrays: dict[str, Any]) -> None:
    _atomic_npz(path, **arrays)


def _run(args: argparse.Namespace, allocation: dict[str, Any]) -> None:
    started = time.monotonic()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "engineering.json").exists() or (out / "teacher_bias_raw.npz").exists():
        raise RuntimeError(f"output already contains teacher-bias results: {out}")
    manifest_path = Path(args.input_manifest).resolve()
    if _sha256(manifest_path) != args.input_sha256:
        raise RuntimeError("input manifest does not match the externally frozen SHA256")
    manifest, samples = _manifest_samples(manifest_path)
    helper_dir = Path(args.helper_dir).resolve()
    smoke, _screen, helper_hashes = _load_helpers(helper_dir)
    root = Path(args.root).resolve()
    prepared_runtime_binding = _bind_prepared_runtime(root, manifest)
    runtime = smoke._runtime(root)
    torch = runtime["torch"]
    device = runtime["device"]
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("a CUDA device is required")
    gpu_name = torch.cuda.get_device_name(device)
    if "V100" not in gpu_name.upper():
        raise RuntimeError(f"single V100 is required, got {gpu_name}")
    model = runtime["model"]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("model parameters could not be frozen")
    groups, structure = _runtime_structure(runtime, smoke)
    predictor_groups = [g for g in groups if str(g.get("family", "")).lower() == "predictor"]
    identity = _source_identity(root, runtime, smoke, helper_hashes)
    identity["prepared_runtime_binding"] = prepared_runtime_binding
    engineering: dict[str, Any] = {
        "schema": "teacher-bias-engineering-v1",
        "runner_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(out / "teacher_bias_protocol.zh.md"),
        "status": "running",
        "allocation": allocation,
        "input_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "sample_identity": [{k: s[k] for k in ("valid_local_index", "trajectory_id", "sample_sha256")} for s in samples],
        "runtime_identity": identity,
        "model_structure": structure,
        "arm_names": list(ARM_NAMES),
        "target_frames": list(FRAME_INDICES[1:]),
        "weight_scope": "predictor-only 24 Linear weights; encoder/other weights FP32",
        "quantization": "per-output-channel signed RTN, scale=max_abs/(2^(bits-1)-1), q=round(weight/scale)",
        "initial_feature_gate": "current FP encode_obs feature equals H5 start feature in z0",
        "rollout_path": "official model.rollout and cached z0->predict->replace_actions path; first FP trajectory allclose gate",
        "completed": [False] * 6,
        "quantization_evidence": {},
        "restore_evidence": [],
        "initial_feature_evidence": [],
    }
    _atomic_json(out / "engineering.json", engineering)
    preprocessor = smoke._preprocessor(runtime)
    all_groups_snapshot = smoke._snapshot_weights(model, groups)
    raw: dict[str, Any] | None = None
    quant_records: dict[str, Any] = {}
    no_op = None
    for si, sample in enumerate(samples):
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("workload deadline exceeded")
        obs0, targets = _make_observations(sample, preprocessor, torch, device)
        actions = torch.as_tensor(sample["action_blocks"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            target_visual_list = []
            target_proprio_list = []
            for target in targets:
                encoded_target = model.encode_obs(target)
                target_visual_list.append(encoded_target["visual"][0, 0].detach().cpu().numpy())
                target_proprio_list.append(encoded_target["proprio"][0, 0].detach().cpu().numpy())
            z0 = model.encode(obs0, actions[:, :1])
            if z0.ndim != 4 or z0.shape[-1] != 404:
                raise RuntimeError(f"unexpected encoded initial shape: {tuple(z0.shape)}")
            current_visual = model.encode_obs(obs0)["visual"][0, 0]
            initial_max_abs = float((current_visual - z0[0, 0, :, :384]).abs().max().item())
            if not torch.allclose(current_visual, z0[0, 0, :, :384], atol=1e-6, rtol=0):
                raise RuntimeError("H5 start and current FP encoder feature differ")
            engineering["initial_feature_evidence"].append({"sample": si, "max_abs": initial_max_abs, "atol": 1e-6, "rtol": 0, "passed": True})
            if si == 0:
                official_obs, official_z = model.rollout(obs0, actions)
                cached_obs, cached_z = _cached_rollout(model, z0, actions, torch)
                no_op = bool(torch.allclose(official_z, cached_z, atol=1e-6, rtol=0))
                if not no_op:
                    raise RuntimeError("official rollout and cached-initial-embedding path differ")
            else:
                cached_obs, cached_z = _cached_rollout(model, z0, actions, torch)
        p = int(target_visual_list[0].shape[0])
        if any(tuple(x.shape) != (p, 384) or not np.isfinite(x).all() for x in target_visual_list):
            raise RuntimeError("target visual feature shape or finiteness failure")
        if raw is None:
            raw = {
                "target_visual": np.full((6, 2, p, 384), np.nan, dtype=np.float32),
                "pred_visual": np.full((6, 3, 2, p, 384), np.nan, dtype=np.float32),
                "target_proprio": np.full((6, 2, 10), np.nan, dtype=np.float32),
                "raw_target_proprio": np.full((6, 2, 2), np.nan, dtype=np.float32),
                "pred_proprio": np.full((6, 3, 2, 10), np.nan, dtype=np.float32),
                "initial_z": np.full((6, p, 404), np.nan, dtype=np.float32),
                "initial_visual": np.full((6, p, 384), np.nan, dtype=np.float32),
                "action_blocks": np.full((6, 5, 10), np.nan, dtype=np.float32),
                "valid_local_indices": np.asarray(VALID_INDICES, dtype=np.int64),
                "trajectory_ids": np.asarray([s["trajectory_id"] for s in samples], dtype="U256"),
                "completed": np.zeros(6, dtype=np.bool_),
                "arm_names": np.asarray(ARM_NAMES),
                "target_frame_indices": np.asarray((5, 25), dtype=np.int64),
                "schema": np.asarray(RAW_SCHEMA),
            }
        elif raw["target_visual"].shape[2] != p:
            raise RuntimeError("patch dimension changed across samples")
        raw["target_visual"][si] = np.asarray(target_visual_list, dtype=np.float32)
        raw["initial_z"][si] = z0[0, 0].detach().cpu().numpy()
        raw["initial_visual"][si] = current_visual.detach().cpu().numpy()
        raw["action_blocks"][si] = sample["action_blocks"]
        raw["target_proprio"][si] = np.asarray(target_proprio_list, dtype=np.float32)
        raw["raw_target_proprio"][si] = sample["proprio"][1:, :]
        for ai, arm in enumerate(ARM_NAMES):
            if not _snapshot_equal(model, all_groups_snapshot, smoke):
                raise RuntimeError(f"pristine restore failed before {arm}")
            if arm in PREDICTOR_BITS:
                records = _quantize_predictor(model, predictor_groups, PREDICTOR_BITS[arm], smoke, torch)
                if arm not in quant_records:
                    quant_records[arm] = records
                    torch.save({"schema": "teacher-bias-quant-params-v1", "arms": quant_records}, out / "quant_params.pt")
                engineering["quantization_evidence"][arm] = {
                    "bits": PREDICTOR_BITS[arm], "linear_count": len(records),
                    "params_path": str(out / "quant_params.pt"), "writeback_readback": True,
                }
            if not _snapshot_equal(model, {k: v for k, v in all_groups_snapshot.items() if "predictor" not in k}, smoke):
                raise RuntimeError(f"non-predictor weights changed in {arm}")
            with torch.no_grad():
                z_obs, _ = _cached_rollout(model, z0, actions, torch)
            visual = z_obs["visual"][0]
            if visual.ndim != 3 or visual.shape[0] < 6 or visual.shape[-1] != 384:
                raise RuntimeError(f"unexpected prediction shape for {arm}: {tuple(visual.shape)}")
            pred = torch.stack((visual[1], visual[5]), dim=0).detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(pred).all():
                raise RuntimeError(f"nonfinite prediction for {arm}")
            raw["pred_visual"][si, ai] = pred
            predicted_proprio = z_obs["proprio"][0]
            raw["pred_proprio"][si, ai] = torch.stack((predicted_proprio[1], predicted_proprio[5]), dim=0).detach().cpu().numpy()
            smoke._restore_weights(model, all_groups_snapshot)
            if not _snapshot_equal(model, all_groups_snapshot, smoke):
                raise RuntimeError(f"exact restore failed after {arm}")
            engineering["restore_evidence"].append({"sample": si, "arm": arm, "exact": True})
        raw["completed"][si] = True
        engineering["completed"][si] = True
        _atomic_npz(out / "teacher_bias_raw.npz", **raw)
        _atomic_json(out / "engineering.json", engineering)
    if raw is None or not bool(raw["completed"].all()):
        raise RuntimeError("screen did not complete all six samples")
    if not _snapshot_equal(model, all_groups_snapshot, smoke):
        raise RuntimeError("final pristine restore failed")
    raw["metadata_json"] = np.asarray(json.dumps({
        "schema": RAW_SCHEMA,
        "input_manifest_sha256": engineering["input_manifest"]["sha256"],
        "sample_sha256": [s["sample_sha256"] for s in samples],
        "trajectory_ids": [s["trajectory_id"] for s in samples],
        "runtime_identity": identity,
        "model_structure": structure,
        "no_op_official_vs_cached": no_op,
        "cached_initial_embedding": True,
        "target_visual_shape": list(raw["target_visual"].shape),
        "pred_visual_shape": list(raw["pred_visual"].shape),
    }, sort_keys=True))
    _atomic_npz(out / "teacher_bias_raw.npz", **raw)
    engineering["status"] = "complete"
    engineering["quant_params_sha256"] = _sha256(out / "quant_params.pt")
    engineering["no_op_official_vs_cached"] = no_op
    engineering["raw_npz"] = {"path": str(out / "teacher_bias_raw.npz"), "sha256": _sha256(out / "teacher_bias_raw.npz")}
    _atomic_json(out / "engineering.json", engineering)
    summary = {
        "schema": "teacher-bias-summary-v1",
        "status": engineering["status"],
        "completed": int(raw["completed"].sum()),
        "raw_npz": engineering["raw_npz"],
        "input_manifest_sha256": engineering["input_manifest"]["sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": SOURCE_COMMIT,
        "dinov2_source_commit": DINOV2_COMMIT,
        "model_structure": {k: structure[k] for k in ("num_hist", "concat_dim", "model_emb_dim", "predictor_linear_count")},
        "arms": list(ARM_NAMES),
        "no_op_official_vs_cached": no_op,
        "note": "Science metrics and gates are computed independently from raw features by the CPU verifier.",
    }
    _atomic_json(out / "summary.json", summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--helper-dir", required=True)
    parser.add_argument("--max-seconds", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    guard = importlib.import_module("allocation_guard")
    allocation = guard.require_allocation()
    args = _parse_args()
    try:
        _run(args, allocation)
    except Exception as exc:
        out = Path(args.output).resolve()
        out.mkdir(parents=True, exist_ok=True)
        path = out / "engineering.json"
        evidence = _load_json(path) if path.exists() else {"schema": "teacher-bias-engineering-v1", "allocation": allocation}
        evidence["status"] = "failed"
        evidence["error"] = f"{type(exc).__name__}: {exc}"[:1200]
        _atomic_json(path, evidence)
        raise


if __name__ == "__main__":
    main()
