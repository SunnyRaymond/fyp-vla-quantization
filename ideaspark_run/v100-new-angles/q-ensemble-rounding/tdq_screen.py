"""Bounded TD-MPC2 five-critic stochastic-rounding raw screen.

The allocation guard is deliberately the first workload action in ``main``.
After it succeeds, this runner hashes the pinned source/checkpoint, validates
the CPU-prepared reset manifest, loads the official TD-MPC2 architecture with
strict state-dict coverage, and records only fixed-input Q outputs.  It does
not call the planner, an environment, a policy rollout, or the science gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple


TOP_DEFAULT = Path("/tc1home/UG/yguo017/v100_newangles_ccds")
SOURCE_COMMIT = "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
HF_REVISION = "73a50e2719ed8258c72c7d1fefd23b781d66e35e"
CHECKPOINT_SHA256 = "0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2"
CHECKPOINT_SIZE = 31_344_610
TASK = "cartpole-balance"
RESET_SEEDS = (5201, 5202, 5203, 5204, 5205, 5206, 5207, 5208)
ROUNDING_SEEDS = (4101, 4102, 4103)
PAIR_SEED_BASE = 8201
ACTION_SEED = 6201
POLICY_SEED_BASE = 7201
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
MAX_WORKLOAD_SECONDS = 720.0
RAW_SCHEMA = "tdq-coupling-raw-v1"
MANIFEST_SCHEMA = "tdmpc2-q-coupling-preparation-v1"
ARMS = ("FP32", "W4_RTN", "W4_independent_SR", "W4_stratified_SR")
Q_PREFIX = "_Qs.params."
DETACH_PREFIX = "_detach_Qs_params."
TARGET_PREFIX = "_target_Qs_params."
SOURCE_FILES = (
    "tdmpc2/common/__init__.py",
    "tdmpc2/common/parser.py",
    "tdmpc2/common/world_model.py",
    "tdmpc2/common/layers.py",
    "tdmpc2/common/math.py",
    "tdmpc2/tdmpc2.py",
    "tdmpc2/config.yaml",
    "tdmpc2/envs/dmcontrol.py",
    "tdmpc2/envs/wrappers/timeout.py",
    "docker/environment.yaml",
)


class ImplementationInconclusive(RuntimeError):
    """An explicit engineering comparison could not be accepted."""


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, np: Any, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_small_summary(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB ({len(encoded)} bytes)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": _sha256_file(resolved),
    }


def _check_deadline(started: float, maximum: float, label: str) -> None:
    if time.monotonic() - started >= maximum:
        raise TimeoutError(f"max-seconds reached at {label}")


def _resolve_child(root: Path, value: str, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"{label} escapes its allowed root: {resolved}")
    return resolved


def _load_guard() -> Any:
    return importlib.import_module("allocation_guard").require_allocation


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _load_input_manifest(path: Path) -> Dict[str, Any]:
    """Load only the pinned CPU preparation output; never select samples here."""
    import numpy as np

    path = path.resolve()
    raw = _load_json(path, "TD-MPC2 Q-coupling manifest")
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported manifest schema: {raw.get('schema')!r}")
    if raw.get("task") != TASK or raw.get("seed_order") != list(RESET_SEEDS):
        raise RuntimeError("manifest task or reset seed order is not frozen")
    config = raw.get("config")
    if not isinstance(config, dict) or any(
        config.get(key) != expected
        for key, expected in (("resolved_task", TASK), ("obs", "state"), ("num_q", NUM_Q), ("model_size", 5), ("compile", False))
    ):
        raise RuntimeError("manifest resolved config does not match the TD-MPC2 screen")
    dmcontrol = raw.get("dmcontrol")
    if not isinstance(dmcontrol, dict) or dmcontrol.get("observation_shape") != [OBS_DIM] or dmcontrol.get("action_dim") != ACTION_DIM:
        raise RuntimeError("manifest DMControl shape contract is not exact")
    expected_checkpoint = raw.get("checkpoint")
    if not isinstance(expected_checkpoint, dict):
        raise RuntimeError("manifest has no checkpoint identity")
    if (
        expected_checkpoint.get("revision") != HF_REVISION
        or str(expected_checkpoint.get("sha256", "")).casefold() != CHECKPOINT_SHA256
        or str(expected_checkpoint.get("expected_lfs_oid", "")).casefold() != CHECKPOINT_SHA256
        or int(expected_checkpoint.get("size", -1)) != CHECKPOINT_SIZE
    ):
        raise RuntimeError("manifest checkpoint identity does not match the frozen pin")
    source = raw.get("source")
    if not isinstance(source, dict) or source.get("commit") != SOURCE_COMMIT:
        raise RuntimeError("manifest source identity does not match the frozen commit")
    observations = raw.get("observations")
    if not isinstance(observations, dict) or not isinstance(observations.get("path"), str):
        raise RuntimeError("manifest has no observations file identity")
    observation_path = _resolve_child(path.parent, observations["path"], "observations path")
    expected_obs_hash = observations.get("sha256")
    if not isinstance(expected_obs_hash, str) or len(expected_obs_hash) != 64:
        raise RuntimeError("manifest has no observation SHA-256")
    actual_obs_hash = _sha256_file(observation_path)
    if actual_obs_hash is None or actual_obs_hash.casefold() != expected_obs_hash.casefold():
        raise RuntimeError("observations.npz SHA-256 mismatch")
    try:
        with np.load(observation_path, allow_pickle=False) as packed:
            values = np.asarray(packed["observations"])
            seeds = np.asarray(packed["seeds"])
            metadata_text = str(np.asarray(packed["metadata_json"]).item())
    except Exception as exc:
        raise RuntimeError("cannot read CPU-prepared observations.npz without pickle") from exc
    if values.shape != (STATE_COUNT, OBS_DIM) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise RuntimeError(f"observations must be finite float32[{STATE_COUNT},{OBS_DIM}]")
    if seeds.shape != (STATE_COUNT,) or seeds.dtype != np.int64 or seeds.tolist() != list(RESET_SEEDS):
        raise RuntimeError("observation seed array does not match the frozen order")
    try:
        metadata = json.loads(metadata_text)
    except ValueError as exc:
        raise RuntimeError("observations metadata_json is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != "tdmpc2-cartpole-reset-input-v1":
        raise RuntimeError("observations metadata schema mismatch")
    if metadata.get("source_commit") != SOURCE_COMMIT or metadata.get("observation_shape") != [STATE_COUNT, OBS_DIM]:
        raise RuntimeError("observations metadata source/shape mismatch")
    return {
        "raw": raw,
        "manifest_path": str(path),
        "manifest_sha256": _sha256_file(path),
        "observations_path": str(observation_path),
        "observations_sha256": actual_obs_hash,
        "observations": np.ascontiguousarray(values),
        "reset_seeds": list(RESET_SEEDS),
        "episode_ids": [f"{TASK}:reset_seed:{seed}" for seed in RESET_SEEDS],
        "metadata": metadata,
    }


def _source_identity(source_root: Path) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = (source_root / Path(*relative.split("/"))).resolve()
        if source_root.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"pinned TD-MPC2 source file is missing: {path}")
        selected[relative] = _file_identity(path)
    return {
        "repository": "https://github.com/nicklashansen/tdmpc2",
        "commit": SOURCE_COMMIT,
        "root": str(source_root.resolve()),
        "selected_files": selected,
    }


def _assert_manifest_source(manifest: Mapping[str, Any], source_identity: Mapping[str, Any]) -> None:
    expected = manifest.get("source", {}).get("selected_files")
    actual = source_identity.get("selected_files")
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise RuntimeError("manifest/source selected-file identity is incomplete")
    for relative, record in expected.items():
        if relative not in actual or not isinstance(record, dict):
            raise RuntimeError(f"manifest does not bind source file: {relative}")
        if str(record.get("sha256", "")).casefold() != str(actual[relative].get("sha256", "")).casefold():
            raise RuntimeError(f"source file hash mismatch: {relative}")


def _checkpoint_identity(path: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    actual = _file_identity(path)
    if actual["size_bytes"] != CHECKPOINT_SIZE or str(actual["sha256"]).casefold() != CHECKPOINT_SHA256:
        raise RuntimeError("local checkpoint does not match the pinned size/SHA-256")
    expected = manifest.get("checkpoint", {})
    if str(expected.get("sha256", "")).casefold() != str(actual["sha256"]).casefold():
        raise RuntimeError("local checkpoint disagrees with manifest identity")
    return {
        "repo": "nicklashansen/tdmpc2",
        "revision": HF_REVISION,
        "path": str(path.resolve()),
        "size_bytes": actual["size_bytes"],
        "sha256": actual["sha256"],
        "expected_lfs_oid": CHECKPOINT_SHA256,
    }


def _gpu_evidence(torch: Any) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing accidental CPU execution")
    device_index = int(torch.cuda.current_device())
    if device_index != 0:
        raise RuntimeError(f"screen requires the single allocated CUDA device as cuda:0, got cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device_index)
    name = str(properties.name)
    capability = tuple(torch.cuda.get_device_capability(device_index))
    if "v100" not in name.casefold() or capability != (7, 0) or int(properties.total_memory) < 30_000_000_000:
        raise RuntimeError(f"this screen requires a 32GB V100, got {name!r}, capability={capability}")
    return {
        "device": f"cuda:{device_index}",
        "name": name,
        "compute_capability": list(capability),
        "total_memory_bytes": int(properties.total_memory),
        "torch": str(torch.__version__),
        "cuda_runtime": str(getattr(torch.version, "cuda", None)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "verified_v100": True,
    }


def _resolve_config(source_root: Path, config_path: Path, checkpoint: Path, output: Path) -> Tuple[Any, Dict[str, Any]]:
    from omegaconf import OmegaConf

    from common import MODEL_SIZE

    raw = OmegaConf.load(str(config_path.resolve()))
    for key, value in MODEL_SIZE[5].items():
        raw[key] = value
    overrides: Dict[str, Any] = {
        "task": TASK,
        "obs": "state",
        "episodic": False,
        "model_size": 5,
        "num_q": NUM_Q,
        "compile": False,
        "multitask": False,
        "task_dim": 0,
        "tasks": [TASK],
        "obs_shape": {"state": [OBS_DIM]},
        "obs_shapes": {"state": [OBS_DIM]},
        "action_dim": ACTION_DIM,
        "action_dims": [ACTION_DIM],
        "episode_length": 500,
        "episode_lengths": [500],
        "seed_steps": 0,
        "num_bins": NUM_BINS,
        "vmin": -10,
        "vmax": 10,
        "bin_size": 20 / (NUM_BINS - 1),
        "checkpoint": str(checkpoint.resolve()),
        "data_dir": str(output.resolve()),
        "work_dir": str(output.resolve()),
        "task_title": TASK.replace("-", " ").title(),
        "exp_name": "tdq_coupling_screen",
        "enable_wandb": False,
        "save_video": False,
        "save_agent": False,
        "wandb_project": "disabled",
        "wandb_entity": "disabled",
        "seed": 1,
    }
    for key, value in overrides.items():
        raw[key] = value
    resolved = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(resolved, dict):
        raise RuntimeError("resolved TD-MPC2 config is not a mapping")
    # WorldModel only needs attribute-style configuration.  Avoid the training
    # parser/Hydra stack, whose optimizer and runner dependencies are outside
    # this inference-only screen.
    return SimpleNamespace(**dict(resolved)), dict(resolved)


def _metadata_signature(name: str, value: Any) -> Tuple[Any, ...]:
    """Canonicalize the only non-tensor TensorDict state entries we accept."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "__batch_size":
        if type(value).__name__ == "Size":
            return (leaf, "torch.Size", tuple(int(item) for item in value))
        if isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
            return (leaf, type(value).__name__, tuple(int(item) for item in value))
    elif leaf == "__device":
        if value is None:
            return (leaf, "NoneType", None)
        if isinstance(value, str):
            return (leaf, "str", value)
        if type(value).__name__ == "device":
            return (leaf, "torch.device", str(value))
    raise RuntimeError(f"unsupported non-tensor state metadata: {name}={type(value).__name__}")


def _unwrap_checkpoint(value: Any, torch: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint is not a mapping")
    if "model" in value:
        value = value["model"]
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("checkpoint model payload is empty")
    result = dict(value)
    for key, tensor in result.items():
        if not isinstance(key, str):
            raise RuntimeError("checkpoint model payload contains a non-string state key")
        if not torch.is_tensor(tensor):
            _metadata_signature(key, tensor)
    return result


def _module_identity(module: Any) -> Dict[str, Any]:
    source = Path(inspect.getfile(module)).resolve()
    return {"module": getattr(module, "__name__", type(module).__name__), **_file_identity(source)}


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_model(source_root: Path, checkpoint: Path, config_path: Path, output: Path, torch: Any) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    # The pinned repository is archived as ``source/tdmpc2/...`` while its
    # official modules import the sibling ``common`` package as top-level.
    # Add both roots explicitly, with the sibling package taking precedence.
    source_paths = (source_root.resolve(), (source_root / "tdmpc2").resolve())
    for source_path in source_paths:
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    cfg, resolved_cfg = _resolve_config(source_root, config_path, checkpoint, output)
    from common import math as td_math
    from common.layers import api_model_conversion
    from common.world_model import WorldModel

    # Official WorldModel.to() initializes target/detached Q storage.  Move it
    # to CUDA before loading so that device initialization cannot overwrite a
    # loaded checkpoint.
    model = WorldModel(cfg).to(torch.device("cuda:0"))
    parameter_device = next(model.parameters()).device
    if parameter_device != torch.device("cuda:0"):
        raise RuntimeError(f"official TDMPC2 model was not moved to cuda:0 before loading: {parameter_device}")
    payload = torch.load(str(checkpoint.resolve()), map_location="cpu", weights_only=False)
    source_state = _unwrap_checkpoint(payload, torch)
    target_state = model.state_dict()
    conversion_needed = "_detach_Qs_params.0.weight" not in source_state
    converted = api_model_conversion(target_state, source_state)
    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load had missing/unexpected keys: {incompatible}")
    parameter_device_after = next(model.parameters()).device
    if parameter_device_after != torch.device("cuda:0"):
        raise RuntimeError(f"checkpoint load changed model device unexpectedly: {parameter_device_after}")
    model.eval()
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"TD-MPC2 runtime parameter is not float32: {name}={parameter.dtype}")
    runtime = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {
            "torch": str(torch.__version__),
            "tensordict": _package_version("tensordict"),
            "omegaconf": _package_version("omegaconf"),
        },
        "source_identity": {
            "common_init": _module_identity(importlib.import_module("common.init")),
            "common_math": _module_identity(td_math),
            "common_layers": _module_identity(importlib.import_module("common.layers")),
            "world_model_module": _module_identity(importlib.import_module("common.world_model")),
        },
        "source_files_hashed": list(SOURCE_FILES),
        "source_files_hashed_but_not_imported_by_runner": [
            "tdmpc2/common/parser.py",
            "tdmpc2/tdmpc2.py",
            "tdmpc2/envs/dmcontrol.py",
            "tdmpc2/envs/wrappers/timeout.py",
            "docker/environment.yaml",
        ],
        "config_path": str(config_path.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "model_device_before_and_after_load": [str(parameter_device), str(parameter_device_after)],
        "model_eval": True,
        "grad_enabled_for_screen": False,
        "compile": False,
        "load": {
            "strict": True,
            "payload_wrapper": "model_if_present_else_explicit_state_dict",
            "official_api_model_conversion": True,
            "conversion_mode": "old_api_converted" if conversion_needed else "new_api_passthrough",
            "source_tensor_key_count_before_conversion": len(source_state),
            "target_state_key_count": len(target_state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        },
        "resolved_config": resolved_cfg,
    }
    return model, td_math, runtime, cfg


def _storage_token(value: Any) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:
        return int(value.storage().data_ptr())


def _is_linear_q_weight(name: str) -> bool:
    if not name.startswith(Q_PREFIX):
        return False
    suffix = name[len(Q_PREFIX):].split(".")
    return len(suffix) == 2 and suffix[0].isdigit() and suffix[1] == "weight"


def _inspect_q_bindings(model: Any, torch: Any) -> Dict[str, Any]:
    state = model.state_dict()
    if not all(isinstance(name, str) for name in state):
        raise RuntimeError("model state_dict contains a non-string key")
    live = {name[len(Q_PREFIX):]: value for name, value in state.items() if name.startswith(Q_PREFIX)}
    detach = {name[len(DETACH_PREFIX):]: value for name, value in state.items() if name.startswith(DETACH_PREFIX)}
    target = {name[len(TARGET_PREFIX):]: value for name, value in state.items() if name.startswith(TARGET_PREFIX)}
    if not live or set(live) != set(detach) or set(live) != set(target):
        raise RuntimeError("live/detach/target Q TensorDict state entries are not one-to-one")
    alias_rows = []
    for suffix, value in sorted(live.items()):
        detach_value = detach[suffix]
        target_value = target[suffix]
        tensor_entries = (torch.is_tensor(value), torch.is_tensor(detach_value), torch.is_tensor(target_value))
        if any(tensor_entries) and not all(tensor_entries):
            raise RuntimeError(f"Q binding mixes tensor and metadata entries: {suffix}")
        if not any(tensor_entries):
            live_meta = _metadata_signature(suffix, value)
            if live_meta != _metadata_signature(suffix, detach_value) or live_meta != _metadata_signature(suffix, target_value):
                raise RuntimeError(f"live/detach/target Q metadata differs: {suffix}")
            alias_rows.append({
                "suffix": suffix,
                "live_name": Q_PREFIX + suffix,
                "detach_name": DETACH_PREFIX + suffix,
                "target_name": TARGET_PREFIX + suffix,
                "metadata": list(live_meta),
                "live_detach_alias": "metadata_not_applicable",
                "target_distinct": "metadata_not_applicable",
            })
        else:
            # TensorDictParams.state_dict serializes clones in 0.7.2. Bind the
            # actual module tensors; state-dict values are evidence, not handles.
            key = tuple(suffix.split('.'))
            value = model._Qs.params[key]
            detach_value = model._detach_Qs_params[key]
            target_value = model._target_Qs_params[key]
            if not all(torch.equal(state[prefix + suffix], actual) for prefix, actual in (
                    (Q_PREFIX, value), (DETACH_PREFIX, detach_value), (TARGET_PREFIX, target_value))):
                raise RuntimeError(f'serialized and actual Q values differ: {suffix}')
            if _storage_token(value) != _storage_token(detach_value):
                raise RuntimeError(f"live and detach Q storage are not aliases: {suffix}")
            if _storage_token(value) == _storage_token(target_value):
                raise RuntimeError(f"target Q unexpectedly aliases live Q storage: {suffix}")
            alias_rows.append({
                "suffix": suffix,
                "live_name": Q_PREFIX + suffix,
                "detach_name": DETACH_PREFIX + suffix,
                "target_name": TARGET_PREFIX + suffix,
                "live_detach_alias": True,
                "target_distinct": True,
                "binding_source": "actual_TensorDict_parameters",
            })
    weight_names = sorted(Q_PREFIX + suffix for suffix in live if _is_linear_q_weight(Q_PREFIX + suffix))
    if len(weight_names) != 3:
        raise RuntimeError(f"expected exactly three Q Linear weights, got {weight_names}")
    shapes = {}
    for name in weight_names:
        value = live[name[len(Q_PREFIX):]]
        if value.ndim != 3 or int(value.shape[0]) != NUM_Q or value.dtype != torch.float32:
            raise RuntimeError(f"Q Linear weight must be float32 [5,out,in]: {name} {tuple(value.shape)} {value.dtype}")
        shapes[name] = list(value.shape)
    return {
        "q_weight_names": weight_names,
        "q_weight_shapes": shapes,
        "live_state_count": len(live),
        "detach_state_count": len(detach),
        "target_state_count": len(target),
        "alias_rows": alias_rows,
    }


def _clone_state_value(name: str, value: Any, torch: Any) -> Any:
    """Clone tensors and retain only the two explicit TensorDict metadata forms."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if name.rsplit(".", 1)[-1] == "__batch_size":
        if type(value).__name__ == "Size":
            return value
        if isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
            return type(value)(int(item) for item in value)
        raise RuntimeError(f"unsupported __batch_size metadata type: {name}={type(value).__name__}")
    if name.rsplit(".", 1)[-1] == "__device":
        if value is None or isinstance(value, str) or type(value).__name__ == "device":
            return value
        raise RuntimeError(f"unsupported __device metadata type: {name}={type(value).__name__}")
    raise RuntimeError(f"unsupported non-tensor state entry: {name}={type(value).__name__}")


def _state_value_equal(actual: Any, expected: Any, torch: Any) -> bool:
    if torch.is_tensor(actual) and torch.is_tensor(expected):
        return bool(torch.equal(actual.detach().cpu(), expected))
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        return False
    try:
        result = actual == expected
        return bool(result)
    except Exception:
        return False


def _state_digest(state: Mapping[str, Any], names: Sequence[str], torch: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        value = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(b"tensor\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(b"metadata\0")
            digest.update(repr(_metadata_signature(name, value)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _snapshot_model(model: Any, torch: Any) -> Tuple[Dict[str, Any], str, str, str, set[str]]:
    state = model.state_dict()
    if not all(isinstance(name, str) for name in state):
        raise RuntimeError("model state_dict contains a non-string key")
    snapshot = {name: _clone_state_value(name, value, torch) for name, value in state.items()}
    names = set(state)
    target_names = {name for name in names if name.startswith(TARGET_PREFIX)}
    live_weight_names = {name for name in names if _is_linear_q_weight(name)}
    detach_weight_names = {DETACH_PREFIX + name[len(Q_PREFIX):] for name in live_weight_names}
    if not detach_weight_names.issubset(names):
        raise RuntimeError("snapshot cannot bind every live Q weight to detached Q storage")
    transaction_names = live_weight_names | detach_weight_names
    bypass_names = names - transaction_names
    return (
        snapshot,
        _state_digest(snapshot, sorted(names), torch),
        _state_digest(snapshot, sorted(target_names), torch),
        _state_digest(snapshot, sorted(bypass_names), torch),
        transaction_names,
    )


def _restore_exact(model: Any, snapshot: Mapping[str, Any], expected_digest: str, torch: Any) -> None:
    incompatible = model.load_state_dict(snapshot, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"exact snapshot restore had missing/unexpected keys: {incompatible}")
    state = model.state_dict()
    for name, expected in snapshot.items():
        if name not in state or not _state_value_equal(state[name], expected, torch):
            raise RuntimeError(f"exact snapshot restore mismatch: {name}")
    actual_digest = _state_digest(state, sorted(snapshot), torch)
    if actual_digest != expected_digest:
        raise RuntimeError("full model digest differs after exact restore")


def _check_bypassed(model: Any, target_digest: str, bypass_digest: str, q_names: set[str], torch: Any) -> None:
    state = model.state_dict()
    target_names = [name for name in state if name.startswith(TARGET_PREFIX)]
    bypass_names = [name for name in state if name not in q_names]
    if _state_digest(state, target_names, torch) != target_digest:
        raise RuntimeError("target Q digest changed during live-Q transaction")
    if _state_digest(state, bypass_names, torch) != bypass_digest:
        raise RuntimeError("non-Q or bypassed parameter digest changed during live-Q transaction")


def _array_hash(value: Any, torch: Any) -> str:
    return _sha256_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def _quantize_q_weights(model: Any, weight_names: Sequence[str], torch: Any, recipe: str, seed: int | None) -> Dict[str, Any]:
    if recipe not in {"RTN", "independent_SR", "stratified_SR"}:
        raise ValueError(f"unknown Q quantizer recipe: {recipe}")
    generator = None
    if recipe != "RTN":
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(int(seed))
    records = []
    state = model.state_dict()
    with torch.no_grad():
        for name in weight_names:
            key = tuple(name[len(Q_PREFIX):].split('.'))
            weight = model._Qs.params[key]
            original = weight.detach().clone()
            maximum = original.abs().amax(dim=-1, keepdim=True)
            zero_rows = maximum == 0
            scale = torch.where(zero_rows, torch.ones_like(maximum), maximum / float(QMAX))
            normalized = original / scale
            uniform_hash = None
            permutation_hash = None
            if recipe == "RTN":
                integer = torch.round(normalized)
            elif recipe == "independent_SR":
                uniform = torch.rand(tuple(weight.shape), device=weight.device, dtype=weight.dtype, generator=generator)
                uniform_hash = _array_hash(uniform, torch)
                integer = torch.floor(normalized) + (uniform < (normalized - torch.floor(normalized))).to(weight.dtype)
            else:
                base_uniform = torch.rand((1, int(weight.shape[1]), int(weight.shape[2])), device=weight.device, dtype=weight.dtype, generator=generator)
                permutation_keys = torch.rand(tuple(weight.shape), device=weight.device, dtype=weight.dtype, generator=generator)
                permutation = torch.argsort(torch.argsort(permutation_keys, dim=0), dim=0)
                uniform = torch.remainder(base_uniform + permutation.to(weight.dtype) / 5.0, 1.0)
                uniform_hash = _array_hash(uniform, torch)
                permutation_hash = _array_hash(permutation, torch)
                fractional = normalized - torch.floor(normalized)
                integer = torch.floor(normalized) + (uniform < fractional).to(weight.dtype)
            integer = torch.clamp(integer, -QMAX, QMAX)
            integer = torch.where(zero_rows, torch.zeros_like(integer), integer)
            integer_hash = _array_hash(integer, torch)
            dequantized = integer * scale
            weight.copy_(dequantized)
            serialized_after = model.state_dict()[name]
            if not torch.equal(serialized_after, dequantized):
                raise RuntimeError(f'Quantized values did not reach actual model: {name}')
            records.append({
                "name": name,
                "shape": list(weight.shape),
                "numel": int(weight.numel()),
                "scale_shape": list(scale.shape),
                "scale_sha256": _array_hash(scale, torch),
                "scale_min": float(scale.min().item()),
                "scale_max": float(scale.max().item()),
                "zero_output_rows": int(zero_rows.sum().item()),
                "pre_weight_sha256": _array_hash(original, torch),
                "post_weight_sha256": _array_hash(weight, torch),
                "actual_parameter_readback_exact": True,
                "integer_dtype": str(integer.dtype),
                "integer_sha256": integer_hash,
                "weight_mse": float((dequantized.to(torch.float64) - original.to(torch.float64)).square().mean().item()),
                "uniform_sha256": uniform_hash,
                "permutation_sha256": permutation_hash,
            })
    return {
        "recipe": recipe,
        "seed": seed,
        "generator": "one explicit CUDA Generator per rounding seed, continuous across sorted three Linear weights" if recipe != "RTN" else None,
        "rounding": "floor(normalized)+Bernoulli(fraction), clamp[-7,7]" if recipe != "RTN" else "torch.round(normalized), clamp[-7,7]",
        "scale": "per-member per-output-row absmax/7; zero rows map to zero",
        "bits": BITS,
        "qmin": -QMAX,
        "qmax": QMAX,
        "grid_values": list(range(-QMAX, QMAX + 1)),
        "records": records,
    }


def _decode_q_values(model: Any, math_module: Any, cfg: Any, z: Any, action: Any, torch: Any) -> Any:
    logits = model.Q(z, action, None, return_type="all")
    if not torch.is_tensor(logits) or logits.ndim != 3 or tuple(logits.shape[:1]) != (NUM_Q,) or int(logits.shape[-1]) != NUM_BINS:
        raise RuntimeError(f"official Q(all) logits must be [5,N,101], got {getattr(logits, 'shape', None)}")
    values = math_module.two_hot_inv(logits, cfg).squeeze(-1)
    if tuple(values.shape) != (NUM_Q, int(z.shape[0])) or not bool(torch.isfinite(values).all().item()):
        raise RuntimeError(f"decoded scalar Q values must be finite [5,N], got {tuple(values.shape)}")
    return values


def _official_avg_check(model: Any, math_module: Any, cfg: Any, z: Any, action: Any, state_index: int, torch: Any, decoded: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    device = z.device
    if device.type != "cuda":
        raise RuntimeError("official Q(avg) check unexpectedly left CUDA")
    pair_seed = PAIR_SEED_BASE + int(state_index)
    with torch.random.fork_rng(devices=[int(device.index or 0)]):
        torch.manual_seed(pair_seed)
        rng_before = torch.cuda.get_rng_state(device=device)
        pair = torch.randperm(NUM_Q, device=device)[:2]
        torch.cuda.set_rng_state(rng_before, device=device)
        official_raw = model.Q(z, action, None, return_type="avg")
        if not torch.is_tensor(official_raw):
            raise RuntimeError(f"official Q(avg) returned a non-tensor: {type(official_raw).__name__}")
        official = official_raw.squeeze(-1) if official_raw.ndim == 2 and int(official_raw.shape[-1]) == 1 else official_raw
        expected = decoded[pair].mean(dim=0)
        if tuple(official.shape) != tuple(expected.shape):
            raise RuntimeError(f"official Q(avg) shape mismatch: {tuple(official.shape)} vs {tuple(expected.shape)}")
        difference = (official - expected).detach().to(torch.float64).abs()
        denominator = expected.detach().to(torch.float64).abs().clamp_min(1e-12)
        max_abs = float(difference.max().item())
        max_relative = float((difference / denominator).max().item())
        passed = bool(torch.allclose(official, expected, atol=1e-5, rtol=1e-6))
        if not passed:
            raise ImplementationInconclusive(
                f"official Q(avg) decode mismatch at state {state_index}: max_abs={max_abs:.9g}, max_relative={max_relative:.9g}"
            )
        return (
            official.detach().reshape(-1),
            pair.detach().cpu(),
            rng_before.detach().cpu(),
            {
                "state_index": state_index,
                "pair_seed": pair_seed,
                "official_raw_shape": list(official_raw.shape),
                "max_abs": max_abs,
                "max_relative": max_relative,
                "pass": passed,
            },
        )


def _prepare_terminal_cache(model: Any, observations: Any, cfg: Any, torch: Any, np: Any) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    if observations.shape != (STATE_COUNT, OBS_DIM):
        raise RuntimeError(f"observations must be [{STATE_COUNT},{OBS_DIM}]")
    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(ACTION_SEED)
    actions_cpu = torch.empty((STATE_COUNT, CANDIDATE_COUNT, HORIZON, ACTION_DIM), dtype=torch.float32, device="cpu")
    actions_cpu.uniform_(-1.0, 1.0, generator=action_generator)
    device = torch.device("cuda:0")
    observations_gpu = torch.from_numpy(np.ascontiguousarray(observations)).to(device=device)
    actions_gpu = actions_cpu.to(device=device)
    terminal_latents = torch.empty((STATE_COUNT, CANDIDATE_COUNT, LATENT_DIM), dtype=torch.float32, device=device)
    terminal_actions = torch.empty((STATE_COUNT, CANDIDATE_COUNT, ACTION_DIM), dtype=torch.float32, device=device)
    with torch.no_grad():
        for state_index in range(STATE_COUNT):
            z = model.encode(observations_gpu[state_index:state_index + 1], None).repeat(CANDIDATE_COUNT, 1)
            if tuple(z.shape) != (CANDIDATE_COUNT, LATENT_DIM):
                raise RuntimeError(f"encoded latent shape is not [{CANDIDATE_COUNT},{LATENT_DIM}]: {tuple(z.shape)}")
            for step in range(HORIZON):
                z = model.next(z, actions_gpu[state_index, :, step, :], None)
            if tuple(z.shape) != (CANDIDATE_COUNT, LATENT_DIM):
                raise RuntimeError("FP terminal latent shape changed during H3 roll")
            terminal_latents[state_index].copy_(z)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(POLICY_SEED_BASE + state_index)
                policy_action, _ = model.pi(z, None)
            if tuple(policy_action.shape) != (CANDIDATE_COUNT, ACTION_DIM) or not bool(torch.isfinite(policy_action).all().item()):
                raise RuntimeError(f"FP terminal policy action shape/non-finite at state {state_index}")
            terminal_actions[state_index].copy_(policy_action)
    if not bool(torch.isfinite(terminal_latents).all().item()):
        raise RuntimeError("FP terminal latent cache contains non-finite values")
    if float(terminal_actions.abs().max().item()) > 1.0:
        raise RuntimeError("FP terminal policy action escaped the native [-1,1] range")
    if float(actions_cpu.abs().max().item()) > 1.0:
        raise RuntimeError("candidate action escaped the native [-1,1] range")
    return (
        actions_cpu,
        terminal_latents,
        terminal_actions,
        {
            "action_seed": ACTION_SEED,
            "action_distribution": "one CPU torch.Generator uniform[-1,1]",
            "policy_seed_by_state": [POLICY_SEED_BASE + i for i in range(STATE_COUNT)],
            "candidate_actions_shape": list(actions_cpu.shape),
            "terminal_latents_shape": list(terminal_latents.shape),
            "terminal_actions_shape": list(terminal_actions.shape),
            "fp_dynamics_roll_steps": HORIZON,
            "environment_steps": 0,
            "policy_cache_calls": STATE_COUNT,
            "candidate_actions_sha256": _sha256_bytes(actions_cpu.numpy().tobytes()),
            "terminal_latents_sha256": _sha256_bytes(terminal_latents.detach().cpu().numpy().tobytes()),
            "terminal_actions_sha256": _sha256_bytes(terminal_actions.detach().cpu().numpy().tobytes()),
        },
    )


def _evaluate_transaction(model: Any, math_module: Any, cfg: Any, terminal_latents: Any, terminal_actions: Any, torch: Any, np: Any) -> Tuple[Any, Any, Any, Any, List[Dict[str, Any]]]:
    member_values = np.full((STATE_COUNT, CANDIDATE_COUNT, NUM_Q), np.nan, dtype=np.float32)
    official_values = np.full((STATE_COUNT, CANDIDATE_COUNT), np.nan, dtype=np.float32)
    pairs = np.full((STATE_COUNT, 2), -1, dtype=np.int64)
    rng_states: List[Any] = []
    checks: List[Dict[str, Any]] = []
    with torch.no_grad():
        for state_index in range(STATE_COUNT):
            decoded = _decode_q_values(model, math_module, cfg, terminal_latents[state_index], terminal_actions[state_index], torch)
            member_values[state_index] = decoded.transpose(0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
            official, pair, rng_before, check = _official_avg_check(
                model, math_module, cfg, terminal_latents[state_index], terminal_actions[state_index], state_index, torch, decoded
            )
            official_values[state_index] = official.detach().cpu().numpy().astype(np.float32, copy=False)
            pairs[state_index] = pair.numpy().astype(np.int64, copy=False)
            rng_states.append(rng_before.numpy().astype(np.uint8, copy=False))
            checks.append(check)
    if not np.isfinite(member_values).all() or not np.isfinite(official_values).all():
        raise RuntimeError("Q transaction produced non-finite raw values")
    return member_values, official_values, pairs, rng_states, checks


def _write_raw(output: Path, np: Any, member_q: Any, observations: Any, candidate_actions: Any, terminal_latents: Any, terminal_actions: Any, completed: Any, seeds: Any, episode_ids: Sequence[str], official_avg: Any, official_pair: Any, official_rng_state: Any | None) -> None:
    arrays = {
        "schema": np.asarray(RAW_SCHEMA),
        "arm_names": np.asarray(ARMS),
        "member_q": member_q,
        "observations": observations,
        "candidate_actions": candidate_actions,
        "terminal_latents": terminal_latents,
        "terminal_actions": terminal_actions,
        "completed": completed,
        "seeds": seeds,
        "episode_ids": np.asarray(episode_ids),
        "official_avg": official_avg,
        "official_pair": official_pair,
    }
    if official_rng_state is not None:
        arrays["official_rng_state"] = official_rng_state
    _atomic_npz(output / "raw_tdq.npz", np, **arrays)


def _copy_transaction(member_q: Any, official_avg: Any, official_pair: Any, completed: Any, rng_buffer: Any, seed_index: int, arm_index: int, result: Tuple[Any, Any, Any, Any, List[Dict[str, Any]]], np: Any) -> Dict[str, Any]:
    members, official, pairs, rng_rows, checks = result
    member_q[:, seed_index, arm_index] = members
    official_avg[:, seed_index, arm_index] = official
    official_pair[:, seed_index, arm_index] = pairs
    completed[seed_index, arm_index] = True
    if rng_buffer is None:
        rng_length = int(rng_rows[0].shape[0])
        rng_buffer = np.full((STATE_COUNT, len(ROUNDING_SEEDS), len(ARMS), rng_length), 0, dtype=np.uint8)
    if len(rng_rows) != STATE_COUNT or any(int(row.shape[0]) != int(rng_buffer.shape[-1]) for row in rng_rows):
        raise RuntimeError("CUDA RNG state length changed within official Q checks")
    for state_index, row in enumerate(rng_rows):
        rng_buffer[state_index, seed_index, arm_index] = row
    return {"rng_buffer": rng_buffer, "checks": checks}


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engineering: Dict[str, Any] = {
        "schema": "tdq-coupling-engineering-v1",
        "status": "running",
        "allocation": dict(allocation),
        "hostname": socket.gethostname(),
        "parameters": {
            "arms": list(ARMS),
            "reset_seeds": list(RESET_SEEDS),
            "rounding_seeds": list(ROUNDING_SEEDS),
            "candidate_actions": [STATE_COUNT, CANDIDATE_COUNT, HORIZON, ACTION_DIM],
            "obs_shape": [STATE_COUNT, OBS_DIM],
            "latent_shape": [STATE_COUNT, CANDIDATE_COUNT, LATENT_DIM],
            "num_q": NUM_Q,
            "num_bins": NUM_BINS,
            "bits": BITS,
            "qmin": -QMAX,
            "qmax": QMAX,
            "dtype": "float32 model and dequantized FP32 fake quantization",
            "max_seconds": args.max_seconds,
            "science_gate": "deferred_to_root_verify_tdq.py",
        },
        "raw_schema": RAW_SCHEMA,
        "raw_path": str((output / "raw_tdq.npz").resolve()),
    }
    _atomic_json(output / "engineering.json", engineering)
    try:
        source_root = args.source_root.resolve()
        checkpoint = args.checkpoint.resolve()
        config_path = args.config.resolve()
        source_identity = _source_identity(source_root)
        checkpoint_identity = _checkpoint_identity(checkpoint, _load_json(args.manifest.resolve(), "manifest checkpoint identity"))
        _atomic_json(output / "source_identity.json", source_identity)
        _atomic_json(output / "checkpoint_identity.json", checkpoint_identity)
        engineering.update({"source_identity": source_identity, "checkpoint_identity": checkpoint_identity})
        _atomic_json(output / "engineering.json", engineering)
        _check_deadline(started, args.max_seconds, "source/checkpoint identity")

        manifest = _load_input_manifest(args.manifest)
        _assert_manifest_source(manifest["raw"], source_identity)
        input_identity = {
            "manifest_path": manifest["manifest_path"],
            "manifest_sha256": manifest["manifest_sha256"],
            "schema": manifest["raw"].get("schema"),
            "observations_path": manifest["observations_path"],
            "observations_sha256": manifest["observations_sha256"],
            "reset_seeds": manifest["reset_seeds"],
            "episode_ids": manifest["episode_ids"],
            "observation_shape": list(manifest["observations"].shape),
            "flatten_order": manifest["raw"].get("dmcontrol", {}).get("flatten_order"),
            "environment_steps": manifest["metadata"].get("env_steps"),
            "render_calls": manifest["metadata"].get("render_calls"),
            "model_loaded_in_preparation": manifest["metadata"].get("model_loaded"),
        }
        _atomic_json(output / "input_identity.json", input_identity)
        engineering["input_identity"] = input_identity
        _atomic_json(output / "engineering.json", engineering)
        _check_deadline(started, args.max_seconds, "input identity")

        gpu_identity = _gpu_evidence(torch)
        _atomic_json(output / "gpu_identity.json", gpu_identity)
        engineering["gpu_identity"] = gpu_identity
        _atomic_json(output / "engineering.json", engineering)

        model, math_module, runtime_identity, cfg = _load_model(source_root, checkpoint, config_path, output, torch)
        bindings = _inspect_q_bindings(model, torch)
        runtime_identity["q_bindings"] = {
            "q_weight_names": bindings["q_weight_names"],
            "q_weight_shapes": bindings["q_weight_shapes"],
            "live_detach_alias_verified": True,
            "target_distinct_verified": True,
        }
        _atomic_json(output / "runtime_identity.json", runtime_identity)
        engineering["runtime_identity_path"] = str((output / "runtime_identity.json").resolve())
        engineering["runtime_identity"] = runtime_identity
        engineering["q_bindings"] = bindings
        _atomic_json(output / "engineering.json", engineering)
        _check_deadline(started, args.max_seconds, "strict runtime load")

        snapshot, snapshot_digest, target_digest, bypass_digest, q_names = _snapshot_model(model, torch)
        weight_names = bindings["q_weight_names"]
        engineering["transaction"] = {
            "snapshot_digest": snapshot_digest,
            "target_digest_before": target_digest,
            "bypassed_digest_before": bypass_digest,
            "live_q_state_prefix": Q_PREFIX,
            "detach_q_state_prefix": DETACH_PREFIX,
            "target_q_state_prefix": TARGET_PREFIX,
            "quantized_weight_names": weight_names,
        }
        _atomic_json(output / "engineering.json", engineering)

        candidate_actions, terminal_latents, terminal_actions, cache_identity = _prepare_terminal_cache(
            model, manifest["observations"], cfg, torch, np
        )
        engineering["fp_terminal_cache"] = cache_identity
        _atomic_json(output / "engineering.json", engineering)
        member_q = np.full((STATE_COUNT, len(ROUNDING_SEEDS), len(ARMS), CANDIDATE_COUNT, NUM_Q), np.nan, dtype=np.float32)
        observations = manifest["observations"].copy()
        candidate_actions_np = candidate_actions.numpy().astype(np.float32, copy=True)
        terminal_latents_np = terminal_latents.detach().cpu().numpy().astype(np.float32, copy=True)
        terminal_actions_np = terminal_actions.detach().cpu().numpy().astype(np.float32, copy=True)
        official_avg = np.full((STATE_COUNT, len(ROUNDING_SEEDS), len(ARMS), CANDIDATE_COUNT), np.nan, dtype=np.float32)
        official_pair = np.full((STATE_COUNT, len(ROUNDING_SEEDS), len(ARMS), 2), -1, dtype=np.int64)
        completed = np.zeros((len(ROUNDING_SEEDS), len(ARMS)), dtype=np.bool_)
        seeds_np = np.asarray(ROUNDING_SEEDS, dtype=np.int64)
        _write_raw(
            output, np, member_q, observations, candidate_actions_np, terminal_latents_np, terminal_actions_np,
            completed, seeds_np, manifest["episode_ids"], official_avg, official_pair, None,
        )
        _check_deadline(started, args.max_seconds, "FP cache")

        rng_buffer = None
        quantizer_records: Dict[str, Any] = {}
        official_checks: Dict[str, Any] = {}

        def run_one(recipe: str | None, seed: int | None, arm_index: int, seed_index: int, transaction_label: str) -> None:
            nonlocal rng_buffer
            _restore_exact(model, snapshot, snapshot_digest, torch)
            _check_bypassed(model, target_digest, bypass_digest, q_names, torch)
            quantizer = (
                {"recipe": "FP32_no_quantization", "seed": None, "records": []}
                if recipe is None
                else _quantize_q_weights(model, weight_names, torch, recipe, seed)
            )
            _check_bypassed(model, target_digest, bypass_digest, q_names, torch)
            result = _evaluate_transaction(model, math_module, cfg, terminal_latents, terminal_actions, torch, np)
            receipt = _copy_transaction(member_q, official_avg, official_pair, completed, rng_buffer, seed_index, arm_index, result, np)
            rng_buffer = receipt["rng_buffer"]
            checks = receipt["checks"]
            if not all(bool(row["pass"]) for row in checks):
                raise ImplementationInconclusive(f"official avg check failed in {transaction_label}")
            quantizer_records[transaction_label] = quantizer
            official_checks[transaction_label] = {
                "all_pass": True,
                "max_abs": max(float(row["max_abs"]) for row in checks),
                "max_relative": max(float(row["max_relative"]) for row in checks),
                "pair_seed_by_state": [int(row["pair_seed"]) for row in checks],
            }
            _atomic_json(output / "quantizer_transactions.json", quantizer_records)
            _check_bypassed(model, target_digest, bypass_digest, q_names, torch)
            engineering["completed"] = completed.tolist()
            engineering["last_transaction"] = transaction_label
            engineering["quantizer_transactions"] = {
                key: {"recipe": value["recipe"], "seed": value["seed"], "weight_mse": [row["weight_mse"] for row in value["records"]]}
                for key, value in quantizer_records.items()
            }
            engineering["official_avg_checks"] = official_checks
            _write_raw(
                output, np, member_q, observations, candidate_actions_np, terminal_latents_np, terminal_actions_np,
                completed, seeds_np, manifest["episode_ids"], official_avg, official_pair, rng_buffer,
            )
            _atomic_json(output / "engineering.json", engineering)
            _check_deadline(started, args.max_seconds, transaction_label)

        def copy_reference_arm(arm_index: int, label: str) -> None:
            if not bool(completed[0, arm_index]):
                raise RuntimeError(f"cannot replicate incomplete reference arm: {label}")
            member_q[:, 1:, arm_index] = member_q[:, :1, arm_index]
            official_avg[:, 1:, arm_index] = official_avg[:, :1, arm_index]
            official_pair[:, 1:, arm_index] = official_pair[:, :1, arm_index]
            completed[:, arm_index] = True
            if rng_buffer is None:
                raise RuntimeError(f"cannot replicate missing CUDA RNG evidence: {label}")
            rng_buffer[:, 1:, arm_index] = rng_buffer[:, :1, arm_index]
            engineering["completed"] = completed.tolist()
            engineering["last_transaction"] = f"{label} copied_to_all_rounding_seeds"
            _write_raw(
                output, np, member_q, observations, candidate_actions_np, terminal_latents_np, terminal_actions_np,
                completed, seeds_np, manifest["episode_ids"], official_avg, official_pair, rng_buffer,
            )
            _atomic_json(output / "engineering.json", engineering)
            _check_deadline(started, args.max_seconds, f"{label} replication")

        # FP32 and RTN are each evaluated once, then copied to all three raw seed slots.
        run_one(None, None, 0, 0, "FP32")
        copy_reference_arm(0, "FP32")
        run_one("RTN", None, 1, 0, "W4_RTN")
        copy_reference_arm(1, "W4_RTN")

        for arm_index, recipe in ((2, "independent_SR"), (3, "stratified_SR")):
            for seed_index, seed in enumerate(ROUNDING_SEEDS):
                run_one(recipe, int(seed), arm_index, seed_index, f"{ARMS[arm_index]} seed={seed}")
            _restore_exact(model, snapshot, snapshot_digest, torch)
            _check_bypassed(model, target_digest, bypass_digest, q_names, torch)

        _restore_exact(model, snapshot, snapshot_digest, torch)
        if not bool(completed.all()) or not np.isfinite(member_q).all() or not np.isfinite(official_avg).all():
            raise RuntimeError("raw TD-Q arrays are incomplete or non-finite")
        if not np.all((official_pair >= 0) & (official_pair < NUM_Q)):
            raise RuntimeError("official pair array is incomplete")
        if rng_buffer is None:
            raise RuntimeError("official CUDA RNG states were not recorded")
        _write_raw(output, np, member_q, observations, candidate_actions_np, terminal_latents_np, terminal_actions_np, completed, seeds_np, manifest["episode_ids"], official_avg, official_pair, rng_buffer)
        engineering.update({
            "status": "complete",
            "completed": completed.tolist(),
            "official_avg_checks": official_checks,
            "weight_restore": {"exact_elementwise": True, "final_full_digest": snapshot_digest, "target_unchanged": True, "bypassed_unchanged": True},
            "raw_shapes": {
                "member_q": list(member_q.shape),
                "observations": list(observations.shape),
                "candidate_actions": list(candidate_actions_np.shape),
                "terminal_latents": list(terminal_latents_np.shape),
                "terminal_actions": list(terminal_actions_np.shape),
                "completed": list(completed.shape),
                "official_avg": list(official_avg.shape),
                "official_pair": list(official_pair.shape),
                "official_rng_state": list(rng_buffer.shape),
            },
            "elapsed_seconds": time.monotonic() - started,
            "science_gate_deferred_to_root_cpu_verifier": True,
        })
        _atomic_json(output / "engineering.json", engineering)
        summary = {
            "schema": "tdq-coupling-screen-v1",
            "status": "complete",
            "allocation": dict(allocation),
            "source_commit": SOURCE_COMMIT,
            "source_identity_path": str((output / "source_identity.json").resolve()),
            "checkpoint": checkpoint_identity,
            "manifest": input_identity,
            "gpu": gpu_identity,
            "runtime_identity_path": str((output / "runtime_identity.json").resolve()),
            "arms": list(ARMS),
            "rounding_seeds": list(ROUNDING_SEEDS),
            "completed": completed.tolist(),
            "raw_tdq": str((output / "raw_tdq.npz").resolve()),
            "engineering": str((output / "engineering.json").resolve()),
            "quantizer_transactions": str((output / "quantizer_transactions.json").resolve()),
            "official_avg_allclose": {
                "atol": 1e-5,
                "rtol": 1e-6,
                "max_abs": max(float(row["max_abs"]) for row in official_checks.values()),
                "max_relative": max(float(row["max_relative"]) for row in official_checks.values()),
                "all_pass": all(bool(row["all_pass"]) for row in official_checks.values()),
            },
            "science_gate": "deferred_to_root_verify_tdq.py",
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_small_summary(output / "summary.json", summary)
        print(json.dumps({"status": "complete", "raw": str(output / "raw_tdq.npz"), "completed": int(completed.sum())}), flush=True)
    except Exception as exc:
        engineering.update({"status": "implementation_inconclusive" if isinstance(exc, ImplementationInconclusive) else "implementation_failure", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started})
        try:
            _atomic_json(output / "engineering.json", engineering)
            _write_small_summary(output / "summary.json", {
                "schema": "tdq-coupling-screen-v1",
                "status": engineering["status"],
                "allocation": dict(allocation),
                "error": engineering["error"],
                "raw_tdq": str((output / "raw_tdq.npz").resolve()),
                "engineering": str((output / "engineering.json").resolve()),
                "science_gate": "deferred_to_root_verify_tdq.py",
            })
        finally:
            raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_WORKLOAD_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_WORKLOAD_SECONDS:g}]")
    return args


def main() -> None:
    # The guard must precede argument parsing, torch/numpy imports, hashes,
    # manifest reads, checkpoint loading, and all tensor work.
    require_allocation = _load_guard()
    allocation = require_allocation()
    args = _parse_args()
    _run(args, allocation)


if __name__ == "__main__":
    main()
