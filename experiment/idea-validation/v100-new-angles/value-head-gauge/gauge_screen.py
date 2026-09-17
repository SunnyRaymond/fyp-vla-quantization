"""Bounded TD-MPC2 value-head gauge screen; real CCDS V100 allocation required."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


TOP_DEFAULT = Path("/tc1home/UG/yguo017/v100_newangles_ccds")
SOURCE_COMMIT = "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
HF_REVISION = "73a50e2719ed8258c72c7d1fefd23b781d66e35e"
CHECKPOINT_SHA256 = "0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2"
CHECKPOINT_SIZE = 31_344_610
PARENT_MANIFEST_SHA256 = "9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369"
TASK = "cartpole-balance"
RESET_SEEDS = tuple(range(5209, 5217))
ACTION_SEED = 6301
POLICY_SEED_BASE = 7301
STATE_COUNT = 8
CANDIDATE_COUNT = 64
HORIZON = 3
OBS_DIM = 5
ACTION_DIM = 1
NUM_Q = 5
LATENT_DIM = 512
NUM_BINS = 101
BITS = 4
QMAX = 7
ARMS = ("FP-original", "FP-centered", "W4-RTN-original", "W4-RTN-centered")
MANIFEST_SCHEMA = "value-head-gauge-preparation-v1"
RAW_SCHEMA = "value-head-gauge-raw-v1"
Q_PREFIX = "_Qs.params."
DETACH_PREFIX = "_detach_Qs_params."
TARGET_PREFIX = "_target_Qs_params."
FINAL_SUFFIX = "2"
MAX_WORKLOAD_SECONDS = 480.0


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_load(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _atomic_npz(path: Path, np: Any, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".writing.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_small_summary(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB ({len(encoded)} bytes)")
    temporary = path.with_name(path.name + ".writing")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _resolve_child(root: Path, value: str, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"{label} escapes allowed root: {resolved}")
    return resolved


def _load_helper() -> Any:
    job_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(job_dir))
    helper = importlib.import_module("tdq_screen")
    helper_path = Path(getattr(helper, "__file__", "")).resolve()
    if helper_path.parent != job_dir:
        raise RuntimeError(f"tdq_screen was imported outside this job output: {helper_path}")
    if getattr(helper, "SOURCE_COMMIT", None) != SOURCE_COMMIT:
        raise RuntimeError("copied TD-MPC2 helper source commit differs from frozen commit")
    if getattr(helper, "CHECKPOINT_SHA256", None) != CHECKPOINT_SHA256:
        raise RuntimeError("copied TD-MPC2 helper checkpoint pin differs from frozen checkpoint")
    return helper


def _validate_parent_manifest(child: Mapping[str, Any], helper: Any) -> Dict[str, Any]:
    parent_ref = child.get("parent_manifest")
    if not isinstance(parent_ref, dict):
        raise RuntimeError("gauge manifest has no parent_manifest reference")
    if str(parent_ref.get("sha256", "")).casefold() != PARENT_MANIFEST_SHA256:
        raise RuntimeError("gauge parent manifest SHA does not match the frozen compatible TDQ parent")
    parent_path = Path(str(parent_ref.get("path", ""))).expanduser().resolve()
    if not parent_path.is_file() or _sha_file(parent_path).casefold() != PARENT_MANIFEST_SHA256:
        raise RuntimeError("frozen compatible TDQ parent manifest is missing or changed")
    parent = _json_load(parent_path, "compatible TDQ parent manifest")
    if parent.get("task") != TASK:
        raise RuntimeError("compatible parent task differs from cartpole-balance")
    parent_source = parent.get("source")
    parent_checkpoint = parent.get("checkpoint")
    if not isinstance(parent_source, dict) or parent_source.get("commit") != SOURCE_COMMIT:
        raise RuntimeError("compatible parent source is not pinned to the frozen TD-MPC2 commit")
    if not isinstance(parent_checkpoint, dict):
        raise RuntimeError("compatible parent has no checkpoint identity")
    if str(parent_checkpoint.get("sha256", "")).casefold() != CHECKPOINT_SHA256:
        raise RuntimeError("compatible parent does not bind the selected seed-3 checkpoint")
    if int(parent_checkpoint.get("size", -1)) != CHECKPOINT_SIZE:
        raise RuntimeError("compatible parent checkpoint size differs from the selected checkpoint")
    return {
        "path": str(parent_path),
        "sha256": PARENT_MANIFEST_SHA256,
        "raw": parent,
        "schema": parent.get("schema"),
        "source_commit": parent_source.get("commit"),
        "checkpoint_sha256": parent_checkpoint.get("sha256"),
    }


def _load_input_manifest(path: Path, helper: Any, np: Any) -> Tuple[Dict[str, Any], Any, Dict[str, Any], Dict[str, Any]]:
    path = path.resolve()
    raw = _json_load(path, "value-head gauge manifest")
    if raw.get("schema") != MANIFEST_SCHEMA or raw.get("task") != TASK:
        raise RuntimeError("gauge manifest schema/task mismatch")
    if raw.get("seed_order") != list(RESET_SEEDS):
        raise RuntimeError("gauge manifest reset seed order is not frozen")
    if raw.get("inference_ran") is not False or raw.get("full_rollout") is not False:
        raise RuntimeError("gauge preparation manifest permits inference or full rollout")
    config = raw.get("config")
    if not isinstance(config, dict) or config.get("resolved_task") != TASK or config.get("obs") != "state":
        raise RuntimeError("gauge manifest resolved config is not state/cartpole-balance")
    if config.get("model_loaded") is not False or config.get("checkpoint_loaded") is not False:
        raise RuntimeError("gauge preparation manifest reports a loaded model/checkpoint")
    dmcontrol = raw.get("dmcontrol")
    if not isinstance(dmcontrol, dict) or dmcontrol.get("observation_shape") != [OBS_DIM]:
        raise RuntimeError("gauge DMControl observation shape contract is not exact")
    if dmcontrol.get("env_steps") != 0 or dmcontrol.get("render_calls") != 0 or dmcontrol.get("reset_calls") != STATE_COUNT:
        raise RuntimeError("gauge preparation manifest has non-reset-only DMControl evidence")
    if raw.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise RuntimeError("gauge manifest source commit differs from frozen source")
    parent_identity = _validate_parent_manifest(raw, helper)
    asset_root = raw.get("asset_root")
    if str(Path(str(asset_root)).expanduser().resolve()) != str(path.parent):
        raise RuntimeError("gauge manifest asset_root does not bind its manifest directory")
    observations = raw.get("observations")
    if not isinstance(observations, dict) or not isinstance(observations.get("path"), str):
        raise RuntimeError("gauge manifest has no observations identity")
    observation_path = _resolve_child(path.parent, observations["path"], "gauge observations")
    expected_hash = str(observations.get("sha256", "")).casefold()
    actual_hash = _sha_file(observation_path).casefold() if observation_path.is_file() else ""
    if len(expected_hash) != 64 or actual_hash != expected_hash:
        raise RuntimeError("gauge observations SHA-256 mismatch")
    if int(observations.get("size", -1)) != observation_path.stat().st_size:
        raise RuntimeError("gauge observations size does not match manifest")
    try:
        with np.load(observation_path, allow_pickle=False) as packed:
            values = np.asarray(packed["observations"])
            seeds = np.asarray(packed["seeds"])
            metadata_text = str(np.asarray(packed["metadata_json"]).item())
    except Exception as exc:
        raise RuntimeError("cannot read gauge observations without pickle") from exc
    if values.shape != (STATE_COUNT, OBS_DIM) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise RuntimeError("gauge observations must be finite float32[8,5]")
    if seeds.shape != (STATE_COUNT,) or seeds.dtype != np.int64 or seeds.tolist() != list(RESET_SEEDS):
        raise RuntimeError("gauge observation seeds do not match the frozen order")
    try:
        metadata = json.loads(metadata_text)
    except ValueError as exc:
        raise RuntimeError("gauge observation metadata_json is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("task") != TASK or metadata.get("seeds") != list(RESET_SEEDS):
        raise RuntimeError("gauge observation metadata task/seeds mismatch")
    if metadata.get("observation_shape") != [STATE_COUNT, OBS_DIM] or metadata.get("env_steps") != 0 or metadata.get("render_calls") != 0:
        raise RuntimeError("gauge observation metadata shape/reset contract mismatch")
    if metadata.get("model_loaded") is not False or metadata.get("checkpoint_loaded") is not False:
        raise RuntimeError("gauge observation metadata reports model/checkpoint loading")
    if metadata.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("gauge observation metadata source commit mismatch")
    return raw, np.ascontiguousarray(values), metadata, parent_identity


def _storage_token(value: Any) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:
        return int(value.storage().data_ptr())


def _td_param(params: Any, suffix: str, torch: Any) -> Any:
    value = params[FINAL_SUFFIX, suffix]
    if not torch.is_tensor(value):
        raise RuntimeError(f"TD-MPC2 Q final parameter is not a tensor: {suffix}")
    return value


def _check_final_head(model: Any, torch: Any) -> Dict[str, Any]:
    live = model._Qs.params
    detach = model._detach_Qs_params
    target = model._target_Qs_params
    rows = []
    for suffix, expected_shape in (("weight", (NUM_Q, NUM_BINS, LATENT_DIM)), ("bias", (NUM_Q, NUM_BINS))):
        live_value = _td_param(live, suffix, torch)
        detach_value = _td_param(detach, suffix, torch)
        target_value = _td_param(target, suffix, torch)
        if any(tuple(value.shape) != expected_shape or value.dtype != torch.float32
               for value in (live_value, detach_value, target_value)):
            raise RuntimeError(
                f"unexpected final Q {suffix} shape/dtype: "
                f"live={tuple(live_value.shape)}/{live_value.dtype}, "
                f"detach={tuple(detach_value.shape)}/{detach_value.dtype}, "
                f"target={tuple(target_value.shape)}/{target_value.dtype}"
            )
        if _storage_token(live_value) != _storage_token(detach_value):
            raise RuntimeError(f"live/detach final Q {suffix} storage is not aliased")
        if _storage_token(live_value) == _storage_token(target_value):
            raise RuntimeError(f"live/target final Q {suffix} storage unexpectedly aliases")
        rows.append({
            "suffix": suffix,
            "live_name": Q_PREFIX + FINAL_SUFFIX + "." + suffix,
            "detach_name": DETACH_PREFIX + FINAL_SUFFIX + "." + suffix,
            "target_name": TARGET_PREFIX + FINAL_SUFFIX + "." + suffix,
            "shape": list(live_value.shape),
            "dtype": str(live_value.dtype),
            "live_detach_alias": True,
            "target_distinct": True,
        })
    return {"rows": rows, "live_param_entries": 2}


def _parameter_readback_exact(model: Any, expected_weight: Any, expected_bias: Any, torch: Any) -> Dict[str, Any]:
    """Prove each live/detached final parameter write reached fresh state_dict clones."""
    state = model.state_dict()
    expected = {
        Q_PREFIX + FINAL_SUFFIX + ".weight": expected_weight,
        DETACH_PREFIX + FINAL_SUFFIX + ".weight": expected_weight,
        Q_PREFIX + FINAL_SUFFIX + ".bias": expected_bias,
        DETACH_PREFIX + FINAL_SUFFIX + ".bias": expected_bias,
    }
    per_key = {}
    for name, value in expected.items():
        per_key[name] = bool(name in state and torch.equal(state[name], value))
    if not all(per_key.values()):
        raise RuntimeError(f"fresh state_dict parameter readback mismatch: {per_key}")
    return {"actual_parameter_readback_exact": True, "keys": per_key}


def _digest_excluding(model: Any, helper: Any, snapshot: Mapping[str, Any], transaction_names: set[str], torch: Any) -> Tuple[set[str], set[str], str, str]:
    target_names = {name for name in snapshot if name.startswith(TARGET_PREFIX)}
    bypass_names = set(snapshot) - set(transaction_names)
    target_digest = helper._state_digest(snapshot, sorted(target_names), torch)
    bypass_digest = helper._state_digest(snapshot, sorted(bypass_names), torch)
    return target_names, bypass_names, target_digest, bypass_digest


def _check_bypassed(model: Any, helper: Any, target_names: set[str], bypass_names: set[str], target_digest: str, bypass_digest: str, torch: Any) -> None:
    state = model.state_dict()
    if helper._state_digest(state, sorted(target_names), torch) != target_digest:
        raise RuntimeError("target Q or target metadata changed during value-head transaction")
    if helper._state_digest(state, sorted(bypass_names), torch) != bypass_digest:
        raise RuntimeError("non-final-Q or bypassed state changed during value-head transaction")


def _restore(model: Any, helper: Any, snapshot: Mapping[str, Any], snapshot_digest: str, target_names: set[str], bypass_names: set[str], target_digest: str, bypass_digest: str, torch: Any) -> None:
    helper._restore_exact(model, snapshot, snapshot_digest, torch)
    _check_bypassed(model, helper, target_names, bypass_names, target_digest, bypass_digest, torch)
    _check_final_head(model, torch)


def _prepare_cache(model: Any, observations: Any, cfg: Any, torch: Any, np: Any, started: float) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    if observations.shape != (STATE_COUNT, OBS_DIM):
        raise RuntimeError(f"gauge observations must have shape [{STATE_COUNT},{OBS_DIM}]")
    generator = torch.Generator(device="cpu").manual_seed(ACTION_SEED)
    actions_cpu = torch.empty((STATE_COUNT, CANDIDATE_COUNT, HORIZON, ACTION_DIM), dtype=torch.float32, device="cpu")
    actions_cpu.uniform_(-1.0, 1.0, generator=generator)
    device = torch.device("cuda:0")
    observations_gpu = torch.from_numpy(np.ascontiguousarray(observations)).to(device=device)
    actions_gpu = actions_cpu.to(device=device)
    terminal_latents = torch.empty((STATE_COUNT, CANDIDATE_COUNT, LATENT_DIM), dtype=torch.float32, device=device)
    terminal_actions = torch.empty((STATE_COUNT, CANDIDATE_COUNT, ACTION_DIM), dtype=torch.float32, device=device)
    with torch.no_grad():
        for state_index in range(STATE_COUNT):
            z = model.encode(observations_gpu[state_index:state_index + 1], None).repeat(CANDIDATE_COUNT, 1)
            if tuple(z.shape) != (CANDIDATE_COUNT, LATENT_DIM):
                raise RuntimeError(f"encoded latent shape mismatch at state {state_index}: {tuple(z.shape)}")
            for step in range(HORIZON):
                z = model.next(z, actions_gpu[state_index, :, step, :], None)
            if tuple(z.shape) != (CANDIDATE_COUNT, LATENT_DIM):
                raise RuntimeError(f"terminal latent shape mismatch at state {state_index}: {tuple(z.shape)}")
            terminal_latents[state_index].copy_(z)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(POLICY_SEED_BASE + state_index)
                policy_action, _ = model.pi(z, None)
            if tuple(policy_action.shape) != (CANDIDATE_COUNT, ACTION_DIM) or not bool(torch.isfinite(policy_action).all().item()):
                raise RuntimeError(f"terminal policy action shape/non-finite at state {state_index}")
            terminal_actions[state_index].copy_(policy_action)
            if time.monotonic() - started >= MAX_WORKLOAD_SECONDS:
                raise TimeoutError("max-seconds reached during FP cache")
    if not bool(torch.isfinite(terminal_latents).all().item()) or float(terminal_actions.abs().max().item()) > 1.0:
        raise RuntimeError("gauge FP cache is non-finite or outside native action range")
    if float(actions_cpu.abs().max().item()) > 1.0:
        raise RuntimeError("gauge candidate action escaped native range")
    candidate_np = actions_cpu.numpy().astype(np.float32, copy=True)
    latent_np = terminal_latents.detach().cpu().numpy().astype(np.float32, copy=True)
    action_np = terminal_actions.detach().cpu().numpy().astype(np.float32, copy=True)
    return actions_cpu, terminal_latents, terminal_actions, {
        "action_seed": ACTION_SEED,
        "action_distribution": "one CPU torch.Generator uniform[-1,1]",
        "policy_seed_by_state": [POLICY_SEED_BASE + i for i in range(STATE_COUNT)],
        "candidate_actions_shape": list(candidate_np.shape),
        "terminal_latents_shape": list(latent_np.shape),
        "terminal_actions_shape": list(action_np.shape),
        "fp_dynamics_roll_steps": HORIZON,
        "environment_steps": 0,
        "policy_cache_calls": STATE_COUNT,
        "candidate_actions_sha256": _sha_bytes(candidate_np.tobytes()),
        "terminal_latents_sha256": _sha_bytes(latent_np.tobytes()),
        "terminal_actions_sha256": _sha_bytes(action_np.tobytes()),
    }


def _q_outputs(model: Any, math_module: Any, cfg: Any, z: Any, action: Any, torch: Any) -> Tuple[Any, Any, Any]:
    logits = model.Q(z, action, None, return_type="all")
    if not torch.is_tensor(logits) or tuple(logits.shape) != (NUM_Q, int(z.shape[0]), NUM_BINS):
        raise RuntimeError(f"official Q(all) logits must be [5,N,101], got {getattr(logits, 'shape', None)}")
    probabilities = torch.softmax(logits, dim=-1)
    decoded = math_module.two_hot_inv(logits, cfg).squeeze(-1)
    if tuple(decoded.shape) != (NUM_Q, int(z.shape[0])) or not bool(torch.isfinite(decoded).all().item()):
        raise RuntimeError(f"decoded Q values must be finite [5,N], got {tuple(decoded.shape)}")
    if not bool(torch.isfinite(probabilities).all().item()):
        raise RuntimeError("Q softmax probabilities are non-finite")
    return logits, probabilities, decoded


def _rtn(weight: Any, torch: Any) -> Tuple[Any, Any, Any]:
    maximum = weight.abs().amax(dim=-1, keepdim=True)
    zero_rows = maximum == 0
    scale = torch.where(zero_rows, torch.ones_like(maximum), maximum / float(QMAX))
    codes = torch.clamp(torch.round(weight / scale), -QMAX, QMAX)
    codes = torch.where(zero_rows, torch.zeros_like(codes), codes)
    dequantized = codes * scale
    return dequantized, scale, codes.to(torch.int8)


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    torch.set_grad_enabled(False)
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engineering: Dict[str, Any] = {
        "schema": "value-head-gauge-engineering-v1",
        "status": "running",
        "allocation": dict(allocation),
        "hostname": socket.gethostname(),
        "parameters": {
            "arms": list(ARMS),
            "reset_seeds": list(RESET_SEEDS),
            "action_seed": ACTION_SEED,
            "policy_seed_base": POLICY_SEED_BASE,
            "candidate_actions": [STATE_COUNT, CANDIDATE_COUNT, HORIZON, ACTION_DIM],
            "obs_shape": [STATE_COUNT, OBS_DIM],
            "latent_shape": [STATE_COUNT, CANDIDATE_COUNT, LATENT_DIM],
            "num_q": NUM_Q,
            "num_bins": NUM_BINS,
            "bits": BITS,
            "qmin": -QMAX,
            "qmax": QMAX,
            "dtype": "float32 model and dequantized FP32 fake quantization",
            "final_q_weight": Q_PREFIX + FINAL_SUFFIX + ".weight",
            "final_q_bias": Q_PREFIX + FINAL_SUFFIX + ".bias",
            "max_seconds": args.max_seconds,
        },
    }
    try:
        helper = _load_helper()
        helper_path = Path(helper.__file__).resolve()
        if not args.freeze.is_file() or not args.draft.is_file():
            raise RuntimeError("frozen amendment or protocol draft is missing from the job output")
        engineering["protocol_identity"] = {
            "freeze_path": str(args.freeze.resolve()),
            "freeze_sha256": _sha_file(args.freeze),
            "draft_path": str(args.draft.resolve()),
            "draft_sha256": _sha_file(args.draft),
        }
        engineering["helper_identity"] = {"path": str(helper_path), "sha256": _sha_file(helper_path)}
        raw_manifest, observations, observation_metadata, parent_identity = _load_input_manifest(args.manifest, helper, np)
        engineering["manifest_identity"] = {
            "path": str(args.manifest.resolve()),
            "sha256": _sha_file(args.manifest),
            "schema": raw_manifest["schema"],
            "observation_path": str(_resolve_child(args.manifest.resolve().parent, raw_manifest["observations"]["path"], "gauge observations")),
            "observation_sha256": raw_manifest["observations"]["sha256"],
            "reset_seeds": list(RESET_SEEDS),
            "parent": parent_identity,
            "metadata": observation_metadata,
        }
        source_root = args.source_root.resolve()
        config_path = args.config.resolve()
        checkpoint = args.checkpoint.resolve()
        if not config_path.is_file() or source_root not in config_path.parents:
            raise RuntimeError("config path is not inside the pinned source tree")
        parent_raw = parent_identity.get("raw")
        source_identity = helper._source_identity(source_root)
        if source_identity.get("commit") != SOURCE_COMMIT:
            raise RuntimeError("source identity is not pinned to the frozen commit")
        if not isinstance(parent_raw, dict):
            raise RuntimeError("compatible TDQ parent raw manifest is unavailable for source binding")
        helper._assert_manifest_source(parent_raw, source_identity)
        checkpoint_identity = helper._checkpoint_identity(
            checkpoint, {"checkpoint": {"sha256": CHECKPOINT_SHA256}}
        )
        checkpoint_identity["compatibility"] = "CPU strict-compatible seed-3 asset selected before gauge outputs"
        engineering["source_identity"] = source_identity
        engineering["source_identity_binding"] = {
            "verified_against_parent_manifest": True,
            "selected_files": sorted(source_identity.get("selected_files", {})),
        }
        engineering["checkpoint_identity"] = checkpoint_identity
        engineering["gpu_identity"] = helper._gpu_evidence(torch)
        model, math_module, runtime_identity, cfg = helper._load_model(source_root, checkpoint, config_path, output, torch)
        if model.training or runtime_identity.get("model_eval") is not True or runtime_identity.get("compile") is not False:
            raise RuntimeError("TD-MPC2 model is not in required eval/non-compiled state")
        engineering["runtime_identity"] = runtime_identity
        engineering["actual_imports"] = {
            "helper_module": helper.__name__,
            "helper_path": str(helper_path),
            "torch": torch.__name__,
            "numpy": np.__name__,
            "world_model": type(model).__name__,
            "math_module": getattr(math_module, "__name__", type(math_module).__name__),
        }
        head_identity = _check_final_head(model, torch)
        engineering["q_head_identity"] = head_identity
        snapshot, snapshot_digest, _, _, _snapshot_q_names = helper._snapshot_model(model, torch)
        final_live_weight = Q_PREFIX + FINAL_SUFFIX + ".weight"
        final_live_bias = Q_PREFIX + FINAL_SUFFIX + ".bias"
        final_detach_weight = DETACH_PREFIX + FINAL_SUFFIX + ".weight"
        final_detach_bias = DETACH_PREFIX + FINAL_SUFFIX + ".bias"
        transaction_names = {final_live_weight, final_detach_weight, final_live_bias, final_detach_bias}
        for name in (final_live_weight, final_live_bias, final_detach_weight, final_detach_bias):
            if name not in snapshot:
                raise RuntimeError(f"final Q transaction key missing from strict snapshot: {name}")
        target_names, bypass_names, target_digest, bypass_digest = _digest_excluding(model, helper, snapshot, transaction_names, torch)
        _restore(model, helper, snapshot, snapshot_digest, target_names, bypass_names, target_digest, bypass_digest, torch)
        cache_actions, terminal_latents, terminal_actions, cache_identity = _prepare_cache(model, observations, cfg, torch, np, started)
        engineering["fp_terminal_cache"] = cache_identity
        _restore(model, helper, snapshot, snapshot_digest, target_names, bypass_names, target_digest, bypass_digest, torch)
        live_weight = _td_param(model._Qs.params, "weight", torch)
        live_bias = _td_param(model._Qs.params, "bias", torch)
        original_weight = live_weight.detach().clone()
        original_bias = live_bias.detach().clone()
        centered_weight = original_weight - original_weight.mean(dim=1, keepdim=True)
        centered_bias = original_bias - original_bias.mean(dim=1, keepdim=True)
        if not bool(torch.isfinite(original_weight).all().item()) or not bool(torch.isfinite(original_bias).all().item()):
            raise RuntimeError("pristine final Q head contains non-finite values")
        weight_denominator = original_weight.to(torch.float64).square().sum()
        if float(weight_denominator.item()) == 0.0:
            global_ratio = None
        else:
            pooled_mean = original_weight.to(torch.float64).mean(dim=1, keepdim=True)
            global_ratio = float((NUM_BINS * pooled_mean.square().sum() / weight_denominator).item())
        original_quantized, scale_original, codes_original = _rtn(original_weight, torch)
        centered_quantized, scale_centered, codes_centered = _rtn(centered_weight, torch)
        relative_scale_change = (scale_centered.to(torch.float64) - scale_original.to(torch.float64)).abs() / scale_original.to(torch.float64).abs().clamp_min(1e-30)
        scale_change_any = bool((relative_scale_change > 1e-6).any().item())
        code_change_any = bool(torch.ne(codes_original, codes_centered).any().item())
        global_binding = {
            "pooled_weight_common_mode_ratio": global_ratio,
            "weight_denominator": float(weight_denominator.item()),
            "scale_relative_change_max": float(relative_scale_change.max().item()),
            "scale_change_any_gt_1e-6": scale_change_any,
            "integer_code_change_any": code_change_any,
            "zero_row_scale_convention": "scale=1, code=0, dequant=0",
            "bound": bool(global_ratio is not None and np.isfinite(global_ratio) and global_ratio > 1e-8 and (scale_change_any or code_change_any)),
        }
        engineering["global_binding"] = global_binding
        if not global_binding["bound"]:
            engineering["status"] = "inconclusive_no_gauge"
        # Arrays are kept in CPU memory until one atomic write at the end.
        arm_weight_pre = np.zeros((len(ARMS), NUM_Q, NUM_BINS, LATENT_DIM), dtype=np.float32)
        arm_weight_post = np.zeros_like(arm_weight_pre)
        arm_bias = np.zeros((len(ARMS), NUM_Q, NUM_BINS), dtype=np.float32)
        arm_scales = np.zeros((len(ARMS), NUM_Q, NUM_BINS, 1), dtype=np.float32)
        arm_codes = np.full((len(ARMS), NUM_Q, NUM_BINS, LATENT_DIM), -128, dtype=np.int8)
        arm_weight_error = np.zeros_like(arm_weight_pre)
        arm_weight_error_quotient = np.zeros_like(arm_weight_pre)
        member_q = np.full((STATE_COUNT, len(ARMS), CANDIDATE_COUNT, NUM_Q), np.nan, dtype=np.float32)
        q_logits = np.full((STATE_COUNT, len(ARMS), NUM_Q, CANDIDATE_COUNT, NUM_BINS), np.nan, dtype=np.float32)
        probabilities = np.full_like(q_logits, np.nan)
        completed = np.zeros((len(ARMS),), dtype=np.bool_)
        arm_engineering = []
        arm_specs = (
            (original_weight, original_bias, None, None, None, "FP-original"),
            (centered_weight, centered_bias, None, None, None, "FP-centered"),
            (original_weight, original_bias, original_quantized, scale_original, codes_original, "W4-RTN-original"),
            (centered_weight, centered_bias, centered_quantized, scale_centered, codes_centered, "W4-RTN-centered"),
        )
        with torch.no_grad():
            for arm_index, (pre_weight, bias, post_weight, scale, codes, arm_name) in enumerate(arm_specs):
                _restore(model, helper, snapshot, snapshot_digest, target_names, bypass_names, target_digest, bypass_digest, torch)
                live_weight = _td_param(model._Qs.params, "weight", torch)
                live_bias = _td_param(model._Qs.params, "bias", torch)
                if post_weight is None:
                    live_weight.copy_(pre_weight)
                    arm_weight_post[arm_index] = pre_weight.detach().cpu().numpy()
                    arm_scales[arm_index].fill(0.0)
                else:
                    live_weight.copy_(post_weight)
                    arm_weight_post[arm_index] = post_weight.detach().cpu().numpy()
                    arm_scales[arm_index] = scale.detach().cpu().numpy()
                    arm_codes[arm_index] = codes.detach().cpu().numpy()
                live_bias.copy_(bias)
                arm_weight_pre[arm_index] = pre_weight.detach().cpu().numpy()
                arm_bias[arm_index] = bias.detach().cpu().numpy()
                arm_weight_error[arm_index] = arm_weight_post[arm_index] - arm_weight_pre[arm_index]
                arm_weight_error_quotient[arm_index] = arm_weight_error[arm_index] - arm_weight_error[arm_index].mean(axis=1, keepdims=True)
                _check_bypassed(model, helper, target_names, bypass_names, target_digest, bypass_digest, torch)
                alias = _check_final_head(model, torch)
                expected_runtime_weight = pre_weight if post_weight is None else post_weight
                readback = _parameter_readback_exact(model, expected_runtime_weight, bias, torch)
                for state_index in range(STATE_COUNT):
                    logits, probs, decoded = _q_outputs(
                        model, math_module, cfg, terminal_latents[state_index], terminal_actions[state_index], torch
                    )
                    q_logits[state_index, arm_index] = logits.detach().cpu().numpy().astype(np.float32, copy=False)
                    probabilities[state_index, arm_index] = probs.detach().cpu().numpy().astype(np.float32, copy=False)
                    member_q[state_index, arm_index] = decoded.transpose(0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
                    if time.monotonic() - started >= args.max_seconds:
                        raise TimeoutError(f"max-seconds reached during {arm_name}")
                completed[arm_index] = True
                arm_engineering.append({
                    "name": arm_name,
                    "alias": alias,
                    "quantized": post_weight is not None,
                    "weight_pre_sha256": _sha_bytes(arm_weight_pre[arm_index].tobytes()),
                    "weight_post_sha256": _sha_bytes(arm_weight_post[arm_index].tobytes()),
                    "bias_sha256": _sha_bytes(arm_bias[arm_index].tobytes()),
                    **readback,
                })
                _restore(model, helper, snapshot, snapshot_digest, target_names, bypass_names, target_digest, bypass_digest, torch)
        if not bool(completed.all()) or not np.isfinite(member_q).all() or not np.isfinite(q_logits).all() or not np.isfinite(probabilities).all():
            raise RuntimeError("value-head raw arrays are incomplete or non-finite")
        no_op_prob_diff = np.abs(probabilities[:, 1] - probabilities[:, 0]).astype(np.float64)
        no_op_q_diff = np.abs(member_q[:, 1] - member_q[:, 0]).astype(np.float64)
        no_op = {
            "probabilities_atol": 1e-7,
            "probabilities_rtol": 1e-6,
            "decoded_q_atol": 1e-5,
            "decoded_q_rtol": 1e-6,
            "probabilities_max_abs": float(no_op_prob_diff.max()),
            "decoded_q_max_abs": float(no_op_q_diff.max()),
            "probabilities_pass": bool(np.allclose(probabilities[:, 1], probabilities[:, 0], atol=1e-7, rtol=1e-6)),
            "decoded_q_pass": bool(np.allclose(member_q[:, 1], member_q[:, 0], atol=1e-5, rtol=1e-6)),
        }
        engineering["no_op"] = no_op
        engineering["arm_transactions"] = arm_engineering
        engineering["snapshot_restore"] = {
            "snapshot_digest": snapshot_digest,
            "target_digest": target_digest,
            "bypass_digest": bypass_digest,
            "transaction_names": sorted(transaction_names),
            "final_restore_pass": True,
            "target_unchanged": True,
            "bypassed_unchanged": True,
        }
        raw_arrays = {
            "schema": np.asarray(RAW_SCHEMA),
            "arm_names": np.asarray(ARMS),
            "completed": completed,
            "observations": observations,
            "candidate_actions": cache_actions.numpy().astype(np.float32, copy=True),
            "terminal_latents": terminal_latents.detach().cpu().numpy().astype(np.float32, copy=True),
            "terminal_actions": terminal_actions.detach().cpu().numpy().astype(np.float32, copy=True),
            "member_q": member_q,
            "q_logits": q_logits,
            "probabilities": probabilities,
            "final_weight_original": original_weight.detach().cpu().numpy().astype(np.float32, copy=True),
            "final_bias_original": original_bias.detach().cpu().numpy().astype(np.float32, copy=True),
            "final_weight_centered": centered_weight.detach().cpu().numpy().astype(np.float32, copy=True),
            "final_bias_centered": centered_bias.detach().cpu().numpy().astype(np.float32, copy=True),
            "final_weight_arm_pre": arm_weight_pre,
            "final_weight_arm_post": arm_weight_post,
            "final_bias_arm": arm_bias,
            "quant_scales": arm_scales,
            "quant_codes": arm_codes,
            "weight_error": arm_weight_error,
            "weight_error_quotient": arm_weight_error_quotient,
        }
        _atomic_npz(output / "raw_gauge.npz", np, **raw_arrays)
        # ``complete`` means raw production completed.  The independent CPU
        # verifier owns the scientific/engineering decision so it can report
        # implementation_inconclusive or inconclusive_no_gauge while still
        # accepting a complete raw artifact for provenance checks.
        final_status = "complete"
        engineering.update({
            "status": final_status,
            "producer_gate_flags": {
                "no_op_pass": bool(no_op["probabilities_pass"] and no_op["decoded_q_pass"]),
                "global_binding": bool(global_binding["bound"]),
            },
            "raw_schema": RAW_SCHEMA,
            "raw_path": str((output / "raw_gauge.npz").resolve()),
            "raw_shapes": {key: list(value.shape) for key, value in raw_arrays.items() if hasattr(value, "shape")},
            "elapsed_seconds": time.monotonic() - started,
            "science_gate": "deferred_to_root_value_head_verifier",
            "scope": "8 fresh reset states, 64 fixed H3 FP imagined inputs, five decoded Q members, no env step/success/native low-bit claim",
        })
        _atomic_json(output / "engineering.json", engineering)
        _write_small_summary(output / "summary.json", {
            "schema": "value-head-gauge-screen-v1",
            "status": engineering["status"],
            "allocation": dict(allocation),
            "source_commit": SOURCE_COMMIT,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "manifest_sha256": engineering["manifest_identity"]["sha256"],
            "raw_gauge": str((output / "raw_gauge.npz").resolve()),
            "engineering": str((output / "engineering.json").resolve()),
            "arms": list(ARMS),
            "completed": completed.tolist(),
            "no_op": no_op,
            "global_binding": global_binding,
            "science_gate": "deferred_to_root_value_head_verifier",
        })
        print(json.dumps({"status": final_status, "raw": str(output / "raw_gauge.npz"), "completed": int(completed.sum())}), flush=True)
    except Exception as exc:
        engineering.update({"status": "implementation_failure", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started})
        try:
            _atomic_json(output / "engineering.json", engineering)
            _write_small_summary(output / "summary.json", {
                "schema": "value-head-gauge-screen-v1",
                "status": engineering["status"],
                "allocation": dict(allocation),
                "error": engineering["error"],
                "engineering": str((output / "engineering.json").resolve()),
                "science_gate": "deferred_to_root_value_head_verifier",
            })
        finally:
            raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_WORKLOAD_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_WORKLOAD_SECONDS:g}]" )
    return args


def main() -> None:
    # Keep the allocation guard before argument parsing, helper import, hashes,
    # NumPy/Torch imports, model construction, and all tensor work.
    from allocation_guard import require_allocation

    allocation = require_allocation()
    args = _parse_args()
    _run(args, allocation)


if __name__ == "__main__":
    main()
