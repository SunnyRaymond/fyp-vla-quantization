"""Bounded offline flow-geometry drift-ranking screen for SmolVLA.

The runner consumes a CPU-prepared, hashed manifest of twelve single-frame
SmolVLA raw samples.  It applies the checkpoint's saved preprocessor once per
sample, then executes the same processed batch and two explicit noise
tensors through FP32, text-backbone-only W4, and action-expert-only W4 arms.
The denoising solver is the official ten-step Euler solver; a temporary wrapper
records its raw ``v_t`` values and returns them unchanged.  The wrapper is
restored after every call, so this file does not introduce a new solver or
detector and makes no closed-loop or success claim.

All model loading, checkpoint/source hashing, input-file hashing, and tensor
work require a verified CCDS SLURM compute allocation.  The allocation guard
is the first workload action.  A missing or mismatched runtime/manifest fails
closed rather than selecting a guessed LeRobot API or input layout.

Input-manifest contract (``smolvla-raw-input-manifest-v1``)::

    {
      "schema": "smolvla-raw-input-manifest-v1",
      "samples": [{"sample_path": "samples/sample_01.npz",
                    "sha256": "...", "task_index": 0,
                    "task_text": "...", "episode_index": 123,
                    "length": 100, "frame_index": 25,
                    "camera_keys": ["observation.images.camera1"],
                    "state_key": "observation.state"}, ...],
      "identity_path": "identity.json"
    }

Each sample is a CPU-prepared raw NPZ (uint8 CHW images and finite float32
state).  The runner applies the checkpoint's saved LeRobot preprocessor locally
inside the verified allocation, including its tokenizer and normalization
state.  The manifest is the only accepted source of episode/frame selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "flow-geometry-drift-screen-v1"
INPUT_SCHEMA = "flow-geometry-drift-input-v1"
RAW_SCHEMA = "flow-geometry-drift-raw-v1"
RAW_INPUT_SCHEMA = "smolvla-raw-input-manifest-v1"
MODEL_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
BASE_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HORIZON = 50
PADDED_ACTION_DIM = 32
PHYSICAL_ACTION_DIM = 7
VELOCITY_STEPS = 10
PREFIX_STEPS = 8
PREFIX_CHUNK = 8
PREFIX_ACTION_DIM = 7
EPISODE_COUNT = 12
TASK_COUNT = 4
EPISODES_PER_TASK = 3
NOISE_SEEDS = (1701, 1702)
NOISE_SHAPE = (1, HORIZON, PADDED_ACTION_DIM)
DT = -0.1
BITS = 4
QMAX = 7
RHO_THRESHOLD = 0.5
RHO_MARGIN = 0.1
TOLERANCE = 1e-12
MAX_WORKLOAD_SECONDS = 2400.0
ARMS = ("FP32", "backbone_W4", "expert_W4")
LOCUS_NAMES = ("backbone_W4", "expert_W4")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def _check_deadline(started: float, max_seconds: float, label: str) -> None:
    if time.monotonic() - started >= max_seconds:
        raise TimeoutError(f"max-seconds reached at {label}")


def _load_allocation_guard() -> Any:
    # The job script snapshots the guard beside this runner. No path fallback.
    return importlib.import_module("allocation_guard").require_allocation


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.casefold()
    )


def _load_json_file(path: Path, label: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def _load_manifest(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"input manifest is missing: {path}")
    raw_manifest = _load_json_file(path, "raw input manifest")
    if raw_manifest.get("schema") != RAW_INPUT_SCHEMA:
        raise RuntimeError(f"unsupported input manifest schema: {raw_manifest.get('schema')!r}")
    samples = raw_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != EPISODE_COUNT:
        raise RuntimeError("raw input manifest must contain exactly twelve samples")
    identity_rel = raw_manifest.get("identity_path")
    if not isinstance(identity_rel, str) or not identity_rel:
        raise RuntimeError("raw input manifest has no identity_path")
    identity_path = (path.parent / identity_rel).resolve()
    if path.parent.resolve() not in identity_path.parents:
        raise RuntimeError("identity_path escapes the manifest directory")
    identity = _load_json_file(identity_path, "CPU preparation identity")
    if identity.get("schema") != "smolvla-cpu-preparation-identity-v1":
        raise RuntimeError("CPU preparation identity schema mismatch")
    checkpoint = identity.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("repo") != "lerobot/smolvla_libero" or checkpoint.get("revision") != MODEL_REVISION:
        raise RuntimeError("CPU preparation checkpoint identity mismatch")
    base_vlm = identity.get("base_vlm")
    if not isinstance(base_vlm, dict) or base_vlm.get("repo") != BASE_VLM_REPO or base_vlm.get("weights_downloaded") is not False:
        raise RuntimeError("CPU preparation base VLM identity mismatch")
    dataset = identity.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("repo") != DATASET_REPO or dataset.get("revision") != DATASET_REVISION:
        raise RuntimeError("CPU preparation dataset identity mismatch")
    selection_identity = identity.get("selection")
    if (
        not isinstance(selection_identity, dict)
        or int(selection_identity.get("count", -1)) != EPISODE_COUNT
        or not isinstance(selection_identity.get("task_ids"), list)
        or len(selection_identity["task_ids"]) != TASK_COUNT
        or not str(selection_identity.get("rule", "")).startswith("task_index ascending first 4; episode_index ascending first 3 distinct")
    ):
        raise RuntimeError("CPU preparation selection identity is not the frozen twelve-sample rule")

    normalized: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[int, int]] = set()
    for index, entry in enumerate(samples):
        if not isinstance(entry, dict):
            raise RuntimeError(f"raw input sample {index} is not a mapping")
        try:
            task_index = int(entry["task_index"])
            episode_index = int(entry["episode_index"])
            length = int(entry["length"])
            frame_index = int(entry["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"raw input sample {index} has incomplete selection metadata") from exc
        task_text = entry.get("task_text")
        sample_rel = entry.get("sample_path")
        camera_keys = entry.get("camera_keys")
        state_key = entry.get("state_key")
        recorded_hash = entry.get("sha256")
        if task_index < 0 or episode_index < 0 or length <= 0 or frame_index != length // 4:
            raise RuntimeError(f"raw input sample {index} violates frozen frame rule")
        if not isinstance(task_text, str) or not task_text.strip():
            raise RuntimeError(f"raw input sample {index} has no task text")
        if not isinstance(sample_rel, str) or not sample_rel or not _is_sha256(recorded_hash):
            raise RuntimeError(f"raw input sample {index} has no valid sample path/hash")
        if not isinstance(camera_keys, list) or not camera_keys or any(not isinstance(key, str) or not key for key in camera_keys):
            raise RuntimeError(f"raw input sample {index} has malformed camera_keys")
        if len(set(camera_keys)) != len(camera_keys) or not isinstance(state_key, str) or not state_key:
            raise RuntimeError(f"raw input sample {index} has malformed feature mapping")
        sample_path = (path.parent / sample_rel).resolve()
        if path.parent.resolve() not in sample_path.parents or not sample_path.is_file():
            raise RuntimeError(f"raw input sample file is missing or escapes manifest directory: {sample_path}")
        actual_hash = _sha256_file(sample_path)
        if actual_hash != recorded_hash:
            raise RuntimeError(f"raw sample hash mismatch at index {index}")
        pair = (task_index, episode_index)
        if pair in seen_pairs:
            raise RuntimeError(f"duplicate task/episode pair at index {index}")
        seen_pairs.add(pair)
        normalized.append({
            "episode_index": index,
            "dataset_episode_index": episode_index,
            "task_index": task_index,
            "task_id": str(task_index),
            "task_text": task_text,
            "length": length,
            "frame_index": frame_index,
            "sample_path": str(sample_path),
            "sample_sha256": recorded_hash,
            "camera_keys": list(camera_keys),
            "state_key": state_key,
        })
    if [(item["task_index"], item["dataset_episode_index"]) for item in normalized] != sorted(
        (item["task_index"], item["dataset_episode_index"]) for item in normalized
    ):
        raise RuntimeError("raw input samples are not sorted by task_index then episode_index")
    task_ids = sorted({item["task_index"] for item in normalized})
    if len(task_ids) != TASK_COUNT or any(
        sum(item["task_index"] == task_id for item in normalized) != EPISODES_PER_TASK for task_id in task_ids
    ):
        raise RuntimeError("raw input manifest must contain four tasks with three episodes each")
    for task_id in task_ids:
        episode_ids = [item["dataset_episode_index"] for item in normalized if item["task_index"] == task_id]
        if len(set(episode_ids)) != EPISODES_PER_TASK:
            raise RuntimeError(f"task {task_id} does not contain three distinct dataset episodes")
    return {
        "schema": INPUT_SCHEMA,
        "raw_schema": RAW_INPUT_SCHEMA,
        "lerobot_version": "0.4.4",
        "episodes": normalized,
        "identity": identity,
        "identity_path": str(identity_path),
        "raw_manifest": raw_manifest,
    }


def _module_identity(module: Any) -> Dict[str, Any]:
    source = Path(inspect.getfile(module)).resolve()
    return {
        "module": getattr(module, "__name__", type(module).__name__),
        **_file_identity(source),
    }


def _load_runtime(args: argparse.Namespace, torch: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    """Build the local architecture, strict-load one local checkpoint, then move it to V100."""
    model_path = args.model_path.resolve()
    checkpoint_path = args.checkpoint.resolve()
    vlm_path = args.vlm_path.resolve()
    for path, label in ((model_path, "SmolVLA model"), (checkpoint_path, "checkpoint"), (vlm_path, "local VLM")):
        if not path.exists():
            raise FileNotFoundError(f"{label} path is missing: {path}")
    if not checkpoint_path.is_file():
        raise RuntimeError("--checkpoint must name one local, unsharded checkpoint file")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.lerobot_source is not None:
        source_text = str(args.lerobot_source.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    try:
        from importlib.metadata import version

        lerobot_version = version("lerobot")
    except Exception as exc:
        raise RuntimeError("LeRobot package is unavailable in the approved runtime") from exc
    if lerobot_version != "0.4.4":
        raise RuntimeError(f"expected LeRobot 0.4.4, got {lerobot_version!r}")
    try:
        import lerobot
        from lerobot.policies.smolvla import configuration_smolvla, modeling_smolvla, smolvlm_with_expert
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.policies import PreTrainedConfig
    except Exception as exc:
        raise RuntimeError("official LeRobot v0.4.4 SmolVLA modules are unavailable") from exc

    config = PreTrainedConfig.from_pretrained(str(model_path), local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise RuntimeError("checkpoint config did not resolve to SmolVLAConfig")
    config.load_vlm_weights = False
    config.vlm_model_name = str(vlm_path)
    config.compile_model = False
    if not hasattr(config, "rtc_config"):
        raise RuntimeError("official SmolVLA config has no explicit rtc_config field")
    config.rtc_config = None
    # The architecture is created on CPU.  All parameters are converted to
    # float32 and moved to CUDA only after the complete local state dict loads.
    if hasattr(config, "device"):
        config.device = "cpu"

    policy = SmolVLAPolicy(config)
    # Official LeRobot checkpoints use safetensors.save_model, which omits
    # duplicate names for shared storage. The matching strict loader checks
    # complete model coverage while respecting those existing shared bindings.
    from safetensors.torch import load_model
    try:
        missing, unexpected = load_model(policy, str(checkpoint_path), strict=True, device="cpu")
        if missing or unexpected:
            raise RuntimeError(f"checkpoint coverage failed: {missing}, {unexpected}")
    except RuntimeError as exc:
        raise RuntimeError("strict complete checkpoint load failed") from exc
    policy.float().to(torch.device("cuda:0"))
    policy.eval()

    if bool(getattr(policy.config, "load_vlm_weights", True)):
        raise RuntimeError("load_vlm_weights must remain False for local complete-checkpoint loading")
    if bool(getattr(policy.config, "compile_model", True)):
        raise RuntimeError("compile_model must remain False")
    if getattr(policy.config, "rtc_config", None) is not None:
        raise RuntimeError("rtc_config must remain None")
    for owner, label in ((policy, "policy"), (getattr(policy, "model", None), "policy.model")):
        if owner is not None and getattr(owner, "rtc_processor", None) is not None:
            raise RuntimeError(f"{label}.rtc_processor must remain None")
    _assert_architecture(policy, torch)
    source_identity = {
        "lerobot_version": lerobot_version,
        "lerobot_package": _module_identity(lerobot),
        "configuration_smolvla": _module_identity(configuration_smolvla),
        "modeling_smolvla": _module_identity(modeling_smolvla),
        "smolvlm_with_expert": _module_identity(smolvlm_with_expert),
    }
    runtime_identity = {
        "source_identity": source_identity,
        "model_path": str(model_path),
        "model_config": _file_identity(model_path / "config.json"),
        "checkpoint": _file_identity(checkpoint_path),
        "vlm_path": str(vlm_path),
        "vlm_config": _file_identity(vlm_path / "config.json"),
        "processor_files": _processor_identity(model_path, vlm_path, args.manifest_data.get("identity", {})),
        "dtype": "float32",
        "device": str(next(policy.parameters()).device),
        "load_vlm_weights": False,
        "compile_model": False,
        "rtc_config": None,
        "rtc_processor": None,
        "model_rtc_processor": None,
        "attention_mode": str(getattr(policy.config, "attention_mode", None)),
        "attention_implementation": _attention_implementations(policy),
    }
    identity = args.manifest_data.get("identity", {})
    expected_checkpoint = _expected_download_hash(identity.get("checkpoint"), checkpoint_path.name)
    expected_config = _expected_download_hash(identity.get("checkpoint"), "config.json")
    if expected_checkpoint != runtime_identity["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint hash disagrees with CPU preparation identity")
    if expected_config != runtime_identity["model_config"]["sha256"]:
        raise RuntimeError("model config hash disagrees with CPU preparation identity")
    expected_vlm_config = _expected_download_hash(identity.get("base_vlm"), "config.json")
    if expected_vlm_config != runtime_identity["vlm_config"]["sha256"]:
        raise RuntimeError("base VLM config hash disagrees with CPU preparation identity")
    return policy, torch, runtime_identity


def _load_checkpoint_state(torch: Any, path: Path) -> Mapping[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for this checkpoint") from exc
        state = load_file(str(path), device="cpu")
    elif path.suffix in {".bin", ".pth", ".pt"}:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    else:
        raise RuntimeError("checkpoint suffix must be .safetensors, .bin, .pth, or .pt")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("checkpoint did not contain a non-empty state dict")
    if all(isinstance(value, torch.Tensor) for value in state.values()):
        return state
    # Only explicit common wrappers are accepted; arbitrary key stripping would
    # hide an incomplete checkpoint and violate the strict-loading gate.
    for wrapper in ("state_dict", "model"):
        candidate = state.get(wrapper)
        if isinstance(candidate, Mapping) and candidate and all(isinstance(value, torch.Tensor) for value in candidate.values()):
            return candidate
    raise RuntimeError("checkpoint wrapper is not an explicit tensor state_dict")


def _expected_download_hash(section: Any, filename: str) -> str:
    records = section.get("files") if isinstance(section, dict) else None
    if records is None and isinstance(section, dict):
        records = section.get("metadata_files")
    if not isinstance(records, list):
        raise RuntimeError(f"CPU preparation identity has no file records for {filename}")
    matches = [
        item for item in records
        if isinstance(item, dict) and Path(str(item.get("rfilename", ""))).name == filename
    ]
    if len(matches) != 1 or not _is_sha256(matches[0].get("downloaded_sha256")):
        raise RuntimeError(f"CPU preparation identity does not bind exactly one downloaded {filename}")
    return str(matches[0]["downloaded_sha256"]).casefold()


def _identity_records(section: Any, label: str) -> List[Mapping[str, Any]]:
    records = section.get("files") if isinstance(section, dict) else None
    if records is None and isinstance(section, dict):
        records = section.get("metadata_files")
    if not isinstance(records, list):
        raise RuntimeError(f"CPU preparation identity has no {label} file records")
    result = []
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("rfilename"), str):
            raise RuntimeError(f"CPU preparation identity has malformed {label} file records")
        result.append(item)
    return result


def _normalise_repo_filename(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _verify_identity_files(
    root: Path,
    section: Any,
    relative_names: Sequence[str],
    label: str,
) -> List[Dict[str, Any]]:
    """Bind every processor/tokenizer/config file used by runtime to CPU SHA evidence."""
    records = _identity_records(section, label)
    root = root.resolve()
    verified: List[Dict[str, Any]] = []
    for relative_name in relative_names:
        relative = _normalise_repo_filename(relative_name)
        matches = [
            item for item in records
            if _normalise_repo_filename(str(item.get("rfilename", ""))) == relative
        ]
        if len(matches) != 1 or not _is_sha256(matches[0].get("downloaded_sha256")):
            raise RuntimeError(f"CPU preparation identity does not bind exactly one downloaded {label} file: {relative}")
        local = (root / Path(*relative.split("/"))).resolve()
        if root not in local.parents or not local.is_file():
            raise RuntimeError(f"runtime {label} file is missing or escapes its local root: {local}")
        actual = _file_identity(local)
        expected = str(matches[0]["downloaded_sha256"]).casefold()
        if actual["sha256"] != expected:
            raise RuntimeError(f"runtime {label} SHA disagrees with CPU preparation identity: {relative}")
        verified.append({
            "rfilename": relative,
            "cpu_downloaded_sha256": expected,
            "actual": actual,
        })
    return verified


def _processor_identity(model_path: Path, vlm_path: Path, identity: Any) -> Dict[str, Any]:
    # These are the files consumed by SmolVLAConfig/DataProcessorPipeline and
    # TokenizerProcessorStep in the official v0.4.4 path.  Optional tokenizer
    # sidecars are included only when present and are still hash-bound.
    checkpoint = identity.get("checkpoint") if isinstance(identity, dict) else None
    base_vlm = identity.get("base_vlm") if isinstance(identity, dict) else None
    policy_names = (
        "config.json",
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    policy_files = _verify_identity_files(model_path, checkpoint, policy_names, "policy processor/config")
    tokenizer_required = (
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    )
    optional_sidecars = (
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
        "merges.txt",
    )
    for name in optional_sidecars:
        if (vlm_path / name).is_file():
            tokenizer_required += (name,)
    vlm_files = _verify_identity_files(vlm_path, base_vlm, tokenizer_required, "VLM processor/tokenizer/config")
    return {"policy_files": policy_files, "vlm_files": vlm_files}


def _load_preprocessor(model_path: Path, vlm_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load the checkpoint's saved processor with a local tokenizer override."""
    try:
        from lerobot.processor import DataProcessorPipeline

        preprocessor = DataProcessorPipeline.from_pretrained(
            str(model_path),
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            overrides={
                "tokenizer_processor": {"tokenizer_name": str(vlm_path)},
                "device_processor": {"device": "cuda:0"},
            },
        )
    except Exception as exc:
        raise RuntimeError("saved local SmolVLA policy preprocessor could not be loaded offline") from exc
    step_names = [type(step).__name__ for step in getattr(preprocessor, "steps", ())]
    if not step_names or "TokenizerProcessorStep" not in step_names:
        raise RuntimeError(f"saved policy preprocessor has no tokenizer step: {step_names}")
    return preprocessor, {
        "config_filename": "policy_preprocessor.json",
        "tokenizer_path": str(vlm_path),
        "device_override": "cuda:0",
        "steps": step_names,
        "local_files_only": True,
    }


def _attention_implementations(policy: Any) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    vlm = policy.model.vlm_with_expert
    objects = {
        "wrapper_config": getattr(vlm, "config", None),
        "vlm_config": getattr(vlm.vlm, "config", None),
        "vlm_text_config": getattr(vlm.get_vlm_model(), "config", None),
        "expert_config": getattr(vlm.lm_expert, "config", None),
    }
    for name, obj in objects.items():
        values[name] = getattr(obj, "_attn_implementation", None) if obj is not None else None
    # SmolVLA's expert/text path uses its own explicit eager interface.
    # Nested HF config labels do not select that interface.  In particular,
    # SDPA is not intrinsically unsupported on V100; its FP32 math path works.
    interface = vlm.get_attention_interface()
    if getattr(interface, "__func__", None) is not getattr(vlm.eager_attention_forward, "__func__", None):
        raise RuntimeError("SmolVLA text/expert attention is not the official eager interface")
    forbidden = {"flash_attention_2", "flash2"}
    unsupported = {
        name: value
        for name, value in values.items()
        if value is not None and str(value).casefold() in forbidden
    }
    if unsupported:
        raise RuntimeError(f"FlashAttention2 is not allowed on this V100 screen: {unsupported}")
    values["text_expert_interface"] = "official_eager_attention_forward"
    vision = vlm.get_vlm_model().vision_model
    values["vision_config"] = getattr(getattr(vision, "config", None), "_attn_implementation", None)
    values["effective_policy"] = "official_eager_text_expert; recorded_HF_vision_backend"
    return values


def _assert_architecture(policy: Any, torch: Any) -> None:
    config = policy.config
    if int(getattr(config, "chunk_size", -1)) != HORIZON:
        raise RuntimeError("SmolVLA chunk_size must be 50")
    if int(getattr(config, "max_action_dim", -1)) != PADDED_ACTION_DIM:
        raise RuntimeError("SmolVLA max_action_dim must be 32")
    if int(getattr(config, "num_steps", -1)) != VELOCITY_STEPS:
        raise RuntimeError("SmolVLA num_steps must be 10")
    action_feature = getattr(config, "action_feature", None)
    action_shape = getattr(action_feature, "shape", None)
    if action_shape is None or tuple(action_shape) != (PHYSICAL_ACTION_DIM,):
        raise RuntimeError(f"SmolVLA physical action feature must have shape (7,), got {action_shape!r}")
    expert = policy.model.vlm_with_expert
    text_layers = getattr(expert.get_vlm_model(), "text_model", None)
    text_layers = getattr(text_layers, "layers", None)
    expert_layers = getattr(getattr(expert, "lm_expert", None), "layers", None)
    if text_layers is None or expert_layers is None or len(text_layers) == 0 or len(expert_layers) == 0:
        raise RuntimeError("SmolVLM text_model.layers or lm_expert.layers is missing")
    for name, parameter in policy.named_parameters():
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"all runtime parameters must be float32, got {name}={parameter.dtype}")


def _move_to_device(value: Any, device: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, torch) for item in value)
    raise RuntimeError(f"input batch contains a non-tensor value of type {type(value)!r}")


def _clone_batch(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {key: _clone_batch(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch(item, torch) for item in value)
    raise RuntimeError(f"input batch contains unsupported value type {type(value)!r}")


def _load_raw_sample(path: Path, entry: Mapping[str, Any], torch: Any) -> Dict[str, Any]:
    """Turn one hashed CPU sample into the raw mapping consumed by LeRobot's preprocessor."""
    try:
        raw = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"cannot open raw sample without pickle: {path}") from exc
    try:
        try:
            metadata_array = np.asarray(raw["metadata_json"])
        except KeyError as exc:
            raise RuntimeError(f"raw sample has no metadata_json: {path}") from exc
        if metadata_array.ndim != 0 or metadata_array.dtype.kind not in {"U", "S"}:
            raise RuntimeError(f"raw sample metadata_json is not a scalar UTF-8-compatible string: {path}")
        try:
            metadata = json.loads(str(metadata_array.item()))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"raw sample metadata_json is not valid JSON: {path}") from exc
        if not isinstance(metadata, dict) or metadata.get("schema") != "smolvla-raw-input-sample-v1":
            raise RuntimeError(f"raw sample metadata schema mismatch: {path}")
        expected_metadata = {
            "task_index": int(entry["task_index"]),
            "task_text": str(entry["task_text"]),
            "episode_index": int(entry["dataset_episode_index"]),
            "frame_index": int(entry["frame_index"]),
            "state_key": str(entry["state_key"]),
            "camera_keys": [str(key) for key in entry["camera_keys"]],
        }
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if key == "camera_keys":
                actual = list(actual) if isinstance(actual, list) else actual
            if actual != expected:
                raise RuntimeError(f"raw sample metadata does not match manifest for {key}: {path}")
        state_key = str(entry["state_key"])
        state = np.asarray(raw[state_key])
        if state.dtype != np.float32 or state.ndim != 1 or not np.isfinite(state).all():
            raise RuntimeError(f"raw state must be finite float32 vector: {path}:{state_key}")
        batch: Dict[str, Any] = {
            state_key: torch.from_numpy(np.ascontiguousarray(state)),
            "task": [str(entry["task_text"])],
        }
        for key in entry["camera_keys"]:
            image = np.asarray(raw[str(key)])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
                raise RuntimeError(f"raw camera must be uint8 CHW with 3 channels: {path}:{key}")
            tensor = torch.from_numpy(np.ascontiguousarray(image)).to(dtype=torch.float32).div_(255.0)
            batch[str(key)] = tensor
        return batch
    except KeyError as exc:
        raise RuntimeError(f"raw sample is missing a manifest-declared key: {path}:{exc}") from exc
    finally:
        raw.close()


def _prepare_batch(
    raw_batch: Mapping[str, Any], preprocessor: Any, torch: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        batch = preprocessor(_clone_batch(raw_batch, torch))
    except Exception as exc:
        raise RuntimeError("checkpoint preprocessor failed on a CPU-prepared raw sample") from exc
    if not isinstance(batch, Mapping):
        raise RuntimeError("checkpoint preprocessor did not return a mapping")
    result = dict(batch)
    for target_key in ("action", "action_is_pad", "actions_id_pad"):
        if target_key in result and result[target_key] is not None:
            raise RuntimeError(f"preprocessed inference batch contains training target {target_key!r}")
    observation = {key: value for key, value in result.items() if str(key).startswith("observation.")}
    known_complementary = {
        "action", "action_is_pad", "actions_id_pad", "reward", "done", "truncated", "info",
        "task", "subtask", "index", "task_index", "episode_index",
        "next.reward", "next.done", "next.truncated",
    }
    unknown = set(result) - set(observation) - known_complementary
    if unknown:
        raise RuntimeError(f"preprocessed batch has unsupported non-observation keys: {sorted(map(str, unknown))}")
    if not observation:
        raise RuntimeError("preprocessed batch has no observation.* policy inputs")
    for key, value in observation.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or int(value.shape[0]) != 1:
            raise RuntimeError(f"preprocessed batch value {key!r} is not a batch-one tensor")
    required = {"observation.state", "observation.language.tokens", "observation.language.attention_mask"}
    if not required.issubset(observation):
        raise RuntimeError(f"preprocessed batch is missing required policy keys: {sorted(required - set(observation))}")
    return observation, {
        "keys": sorted(str(key) for key in result if key not in observation),
        "types": {str(key): type(value).__name__ for key, value in result.items() if key not in observation},
        "action_is_none": result.get("action", None) is None,
    }


def _noise_tensors(torch: Any, device: Any) -> Tuple[List[Any], np.ndarray]:
    values = []
    for seed in NOISE_SEEDS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(NOISE_SHAPE, dtype=torch.float32, generator=generator).to(device)
        values.append(noise)
    return values, np.stack([value.detach().cpu().numpy()[0] for value in values], axis=0).astype(np.float32)


def _eligible_modules(policy: Any, torch: Any) -> Dict[str, List[Tuple[str, Any]]]:
    wrapper = policy.model.vlm_with_expert
    roots = {
        "backbone_W4": ("model.vlm_with_expert.get_vlm_model().text_model.layers", wrapper.get_vlm_model().text_model.layers),
        "expert_W4": ("model.vlm_with_expert.lm_expert.layers", wrapper.lm_expert.layers),
    }
    result: Dict[str, List[Tuple[str, Any]]] = {}
    for locus, (prefix, layers) in roots.items():
        modules: List[Tuple[str, Any]] = []
        for layer_index, layer in enumerate(layers):
            for relative, module in layer.named_modules():
                if not relative:
                    continue
                if isinstance(module, torch.nn.Linear):
                    name = f"{prefix}.{layer_index}.{relative}"
                    if module.weight.ndim != 2 or module.weight.dtype != torch.float32:
                        raise RuntimeError(f"eligible module is not a float32 2-D Linear: {name}")
                    modules.append((name, module))
        if not modules:
            raise RuntimeError(f"no Linear modules found for {locus}")
        result[locus] = modules
    names = [name for values in result.values() for name, _ in values]
    if len(names) != len(set(names)):
        raise RuntimeError("backbone/expert W4 allowlists overlap")
    return result


def _snapshot_weights(modules: Mapping[str, Sequence[Tuple[str, Any]]], torch: Any) -> Dict[str, Any]:
    return {
        name: module.weight.detach().clone()
        for values in modules.values()
        for name, module in values
    }


def _restore_weights(snapshot: Mapping[str, Any], modules: Mapping[str, Sequence[Tuple[str, Any]]], torch: Any) -> None:
    by_name = {
        name: module
        for values in modules.values()
        for name, module in values
    }
    with torch.no_grad():
        for name, value in snapshot.items():
            by_name[name].weight.copy_(value)
            if not torch.equal(by_name[name].weight, value):
                raise RuntimeError(f"FP snapshot restoration mismatch at {name}")


def _quantize_locus(modules: Sequence[Tuple[str, Any]], torch: Any) -> Dict[str, Any]:
    details = []
    total = 0
    with torch.no_grad():
        for name, module in modules:
            weight = module.weight.detach()
            rows = weight.reshape(weight.shape[0], -1)
            maximum = rows.abs().amax(dim=1, keepdim=True)
            scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / float(QMAX))
            integer = torch.clamp(torch.round(rows / scale), -QMAX, QMAX)
            dequantized = torch.where(maximum == 0, torch.zeros_like(rows), integer * scale)
            module.weight.copy_(dequantized.reshape_as(weight))
            total += int(weight.numel())
            details.append({
                "name": name,
                "shape": list(weight.shape),
                "numel": int(weight.numel()),
                "out_channels": int(weight.shape[0]),
                "max_scale": float(scale.max().item()),
            })
    return {
        "bits": BITS,
        "qmin": -QMAX,
        "qmax": QMAX,
        "scheme": "symmetric_per_output_channel_RTN_dequantized_FP32",
        "linear_count": len(details),
        "numel": total,
        "modules": details,
    }


def _state_subset_digest(policy: Any, excluded: set[str], torch: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if name in excluded:
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class _DenoiseRecorder:
    def __init__(self, policy: Any, torch: Any):
        self.policy = policy
        self.model = policy.model
        self.torch = torch
        self.original = self.model.denoise_step
        self.values: List[Any] = []
        self.handle_active = False

    def run(self, batch: Mapping[str, Any], noise: Any, record: bool = True) -> Tuple[Any, np.ndarray | None]:
        self.values = []
        original = self.model.denoise_step

        if not record:
            # The no-op comparison must run with the method completely
            # unwrapped; otherwise two wrapped calls could agree while the
            # recording hook itself was still changing the execution.
            self.policy.reset()
            output = self.policy.predict_action_chunk(_clone_batch(batch, self.torch), noise=noise)
            if self.model.denoise_step.__func__ is not original.__func__:
                raise RuntimeError("denoise_step changed during the unwrapped call")
            if tuple(output.shape) != (1, HORIZON, PHYSICAL_ACTION_DIM):
                raise RuntimeError(f"physical action output shape must be [1,50,7], got {tuple(output.shape)}")
            return output.detach(), None

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            value = original(*args, **kwargs)
            if not self.torch.is_tensor(value) or tuple(value.shape) != (1, HORIZON, PADDED_ACTION_DIM):
                raise RuntimeError(f"denoise_step velocity shape must be [1,50,32], got {getattr(value, 'shape', None)}")
            if record:
                self.values.append(value.detach().clone())
            return value

        self.model.denoise_step = wrapped
        self.handle_active = True
        try:
            self.policy.reset()
            output = self.policy.predict_action_chunk(_clone_batch(batch, self.torch), noise=noise)
        finally:
            self.model.denoise_step = original
            self.handle_active = False
        if self.model.denoise_step.__func__ is not original.__func__:
            raise RuntimeError("denoise_step wrapper was not restored")
        if tuple(output.shape) != (1, HORIZON, PHYSICAL_ACTION_DIM):
            raise RuntimeError(f"physical action output shape must be [1,50,7], got {tuple(output.shape)}")
        if not record:
            return output.detach(), None
        if len(self.values) != VELOCITY_STEPS:
            raise RuntimeError(f"expected ten denoise calls, got {len(self.values)}")
        velocities = self.torch.stack(self.values, dim=0).detach().cpu().numpy()[:, 0]
        if velocities.shape != (VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM):
            raise RuntimeError(f"recorded velocity shape mismatch: {velocities.shape}")
        return output.detach(), velocities


def _record_noop_gate(policy: Any, batch: Mapping[str, Any], noise: Any, torch: Any) -> Dict[str, Any]:
    recorder = _DenoiseRecorder(policy, torch)
    recorded_actions, recorded_velocities = recorder.run(batch, noise, record=True)
    direct_actions, _ = recorder.run(batch, noise, record=False)
    if not torch.equal(recorded_actions, direct_actions):
        raise RuntimeError("no-op denoise recording changed the action output")
    if recorded_velocities is None or recorded_velocities.shape != (VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM):
        raise RuntimeError("no-op recording did not produce ten raw velocity tensors")
    if not np.isfinite(recorded_velocities).all():
        raise FloatingPointError("no-op recorded velocities are non-finite")
    return {
        "action_shape": list(recorded_actions.shape),
        "velocity_shape": list(recorded_velocities.shape),
        "action_exact_equal": True,
        "velocity_finite": True,
        "wrapper_restored": True,
        "steps": VELOCITY_STEPS,
        "dt": DT,
    }


def _write_raw(output: Path, velocities: np.ndarray, actions: np.ndarray, completed: np.ndarray, noise: np.ndarray, episode_ids: Sequence[str]) -> None:
    _atomic_npz(
        output / "raw_flow.npz",
        schema=np.asarray(RAW_SCHEMA),
        arm_names=np.asarray(ARMS),
        velocities=velocities,
        actions=actions,
        noise=noise,
        episode_ids=np.asarray(episode_ids),
        completed=completed,
    )


def _accel(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eq.11-style raw-velocity acceleration on the fixed 8x8x7 prefix."""
    prefix = values[..., :PREFIX_STEPS, :PREFIX_CHUNK, :PREFIX_ACTION_DIM]
    # Keep the denoising-step norm separate before summing it.  A direct
    # multi-axis norm would collapse the step axis too early and return one
    # value per arm batch instead of one accel value per episode/noise/arm.
    delta = np.diff(prefix, axis=-3)
    delta_flat = delta.reshape(*delta.shape[:-3], delta.shape[-3], -1)
    prefix_flat = prefix.reshape(*prefix.shape[:-3], prefix.shape[-3], -1)
    numerator = 8.0 * np.linalg.norm(delta_flat, axis=-1).sum(axis=-1)
    denominator = np.linalg.norm(prefix_flat, axis=-1).sum(axis=-1)
    degenerate = denominator <= TOLERANCE
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=~degenerate)
    return result, degenerate


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float | None, str | None]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != (EPISODE_COUNT,) or y.shape != (EPISODE_COUNT,):
        return None, "shape_mismatch"
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, "nonfinite"
    if float(np.max(x) - np.min(x)) <= TOLERANCE:
        return None, "x_near_constant"
    if float(np.max(y) - np.min(y)) <= TOLERANCE:
        return None, "y_near_constant"

    def average_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.shape[0], dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.shape[0]:
            stop = start + 1
            while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
            start = stop
        return ranks

    xr = average_rank(x)
    yr = average_rank(y)
    x_centered = xr - xr.mean()
    y_centered = yr - yr.mean()
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= TOLERANCE:
        return None, "rank_near_constant"
    return float(np.dot(x_centered, y_centered) / denom), None


def _metrics(velocities: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
    if velocities.shape != (EPISODE_COUNT, len(NOISE_SEEDS), len(ARMS), VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM):
        raise RuntimeError(f"unexpected velocity array shape: {velocities.shape}")
    if actions.shape != (EPISODE_COUNT, len(NOISE_SEEDS), len(ARMS), HORIZON, PHYSICAL_ACTION_DIM):
        raise RuntimeError(f"unexpected action array shape: {actions.shape}")
    velocity64 = velocities.astype(np.float64)
    action64 = actions.astype(np.float64)
    accel, degenerate = _accel(velocity64)
    action_prefix = action64[..., :PREFIX_CHUNK, :PREFIX_ACTION_DIM]
    fp_action = action_prefix[:, :, 0]
    drift = np.zeros((EPISODE_COUNT, len(NOISE_SEEDS), len(LOCUS_NAMES)), dtype=np.float64)
    qnorm = np.zeros_like(drift)
    for locus_index, arm in enumerate(LOCUS_NAMES):
        arm_index = ARMS.index(arm)
        drift[:, :, locus_index] = np.mean(np.square(action_prefix[:, :, arm_index] - fp_action), axis=(-2, -1))
        qnorm[:, :, locus_index] = np.linalg.norm(action_prefix[:, :, arm_index], axis=(-2, -1))
    episode_rows = []
    for episode in range(EPISODE_COUNT):
        episode_rows.append({
            "episode_index": episode,
            "accel_by_noise": accel[episode].tolist(),
            "drift_mse_by_noise": drift[episode].tolist(),
            "q_action_norm_by_noise": qnorm[episode].tolist(),
            "degenerate_accel_by_noise": degenerate[episode].tolist(),
        })

    locus_rows: Dict[str, Any] = {}
    decisions: List[str] = []
    for locus_index, locus in enumerate(LOCUS_NAMES):
        q_accel_episode = np.mean(accel[:, :, ARMS.index(locus)], axis=1)
        fp_accel_episode = np.mean(accel[:, :, ARMS.index("FP32")], axis=1)
        drift_episode = np.mean(drift[:, :, locus_index], axis=1)
        qnorm_episode = np.mean(qnorm[:, :, locus_index], axis=1)
        if bool(degenerate[:, :, ARMS.index(locus)].any()) or bool(degenerate[:, :, ARMS.index("FP32")].any()):
            locus_rows[locus] = {
                "status": "inconclusive",
                "reason": "degenerate_velocity_norm",
                "episode_accel": q_accel_episode.tolist(),
                "episode_drift_mse": drift_episode.tolist(),
                "episode_q_action_norm": qnorm_episode.tolist(),
            }
            decisions.append("inconclusive")
            continue
        rho_qd, reason_qd = _spearman(q_accel_episode, drift_episode)
        rho_fp, reason_fp = _spearman(fp_accel_episode, drift_episode)
        rho_qnorm, reason_qnorm = _spearman(qnorm_episode, drift_episode)
        reasons = [reason for reason in (reason_qd, reason_fp, reason_qnorm) if reason is not None]
        if reasons:
            locus_rows[locus] = {
                "status": "no_binding_locus",
                "reason": reasons[0],
                "episode_accel": q_accel_episode.tolist(),
                "episode_drift_mse": drift_episode.tolist(),
                "episode_q_action_norm": qnorm_episode.tolist(),
                "rho": {"q_accel_vs_drift": rho_qd, "fp_accel_vs_drift": rho_fp, "q_action_norm_vs_drift": rho_qnorm},
            }
            decisions.append("no_binding_locus")
            continue
        assert rho_qd is not None and rho_fp is not None and rho_qnorm is not None
        gates = {
            "q_accel_ge_0_5": rho_qd >= RHO_THRESHOLD - TOLERANCE,
            "q_accel_exceeds_fp_by_0_1": rho_qd >= rho_fp + RHO_MARGIN - TOLERANCE,
            "q_accel_exceeds_q_action_norm_by_0_1": rho_qd >= rho_qnorm + RHO_MARGIN - TOLERANCE,
        }
        status = "preliminary_go" if all(gates.values()) else "mechanism_no_go"
        locus_rows[locus] = {
            "status": status,
            "episode_accel": q_accel_episode.tolist(),
            "episode_drift_mse": drift_episode.tolist(),
            "episode_q_action_norm": qnorm_episode.tolist(),
            "rho": {"q_accel_vs_drift": rho_qd, "fp_accel_vs_drift": rho_fp, "q_action_norm_vs_drift": rho_qnorm},
            "rho_deltas": {"vs_fp": rho_qd - rho_fp, "vs_q_action_norm": rho_qd - rho_qnorm},
            "gates": gates,
            "spearman": {"tie_rule": "average rank for exact raw-value ties", "constant_range_tolerance": TOLERANCE},
        }
        decisions.append(status)
    if "inconclusive" in decisions:
        overall = "inconclusive"
    elif all(value == "preliminary_go" for value in decisions):
        overall = "preliminary_go"
    elif any(value == "preliminary_go" for value in decisions):
        overall = (
            "inconclusive_partial_binding"
            if any(value == "no_binding_locus" for value in decisions)
            else "scope_limited_preliminary_go"
        )
    elif all(value == "no_binding_locus" for value in decisions):
        overall = "no_binding_locus"
    elif any(value == "no_binding_locus" for value in decisions):
        overall = "inconclusive_partial_binding"
    else:
        overall = "mechanism_no_go"
    return {
        "primary": "offline_flow_accel_drift_ranking_diagnostic",
        "does_not_claim_online_detector": True,
        "does_not_claim_sequence_or_environment_success": True,
        "accel_formula": "8*sum_t=1..7(||v_t-v_(t-1)||_2)/sum_t=0..7(||v_t||_2), raw v, prefix steps/chunk/dim = 8/8/7",
        "drift_formula": "mean((action_Q[:8,:7]-action_FP32[:8,:7])^2) in normalized action space",
        "noise_aggregation": "mean over two fixed noise seeds within episode, then episode-level Spearman",
        "episode_rows": episode_rows,
        "loci": locus_rows,
        "overall_decision": overall,
    }


def _write_small_summary(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, default=_json_default).encode("utf-8")
    if len(encoded) >= 64 * 1024:
        raise RuntimeError(f"summary.json exceeds 64 KiB ({len(encoded)} bytes)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _gpu_evidence(torch: Any) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; refusing accidental CPU execution")
    index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(index)
    name = str(props.name)
    if "v100" not in name.casefold():
        raise RuntimeError(f"this screen requires a V100 allocation, got {name!r}")
    return {
        "device_index": index,
        "name": name,
        "total_memory_bytes": int(props.total_memory),
        "torch": str(torch.__version__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _run(args: argparse.Namespace, allocation: Mapping[str, Any]) -> None:
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "allocation": dict(allocation),
        "allocation_evidence": {
            "guard": "experiment/idea-validation/v100-new-angles/allocation_guard.py",
            "owner_partition_node_verified": True,
            "hostname_normalization": "campaign guard casefolds DNS short names",
        },
        "parameters": {
            "arms": list(ARMS),
            "loci": list(LOCUS_NAMES),
            "noise_seeds": list(NOISE_SEEDS),
            "noise_shape": list(NOISE_SHAPE),
            "velocity_shape": [EPISODE_COUNT, len(NOISE_SEEDS), len(ARMS), VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM],
            "action_shape": [EPISODE_COUNT, len(NOISE_SEEDS), len(ARMS), HORIZON, PHYSICAL_ACTION_DIM],
            "dt": DT,
            "bits": BITS,
            "qmax": QMAX,
            "execution": "FP32 fake-quantized dequantized weights; no native low-bit claim",
        },
    }
    try:
        manifest = _load_manifest(args.input_manifest.resolve())
        asset_identity = manifest["identity"]
        _write_small_summary(output / "asset_identity_brief.json", {
            "full_identity_sha256": _sha256_file(Path(manifest["identity_path"])),
            "full_identity_retained": manifest["identity_path"],
            "checkpoint": asset_identity["checkpoint"],
            "base_vlm": asset_identity["base_vlm"],
            "dataset": {key: asset_identity["dataset"][key] for key in ("repo", "revision")},
            "mapping": asset_identity.get("mapping"),
            "selection": asset_identity["selection"],
            "decode": asset_identity.get("decode"),
        })
        args.manifest_data = manifest
        summary["input_manifest"] = {
            "path": str(args.input_manifest.resolve()),
            "sha256": _sha256_file(args.input_manifest.resolve()),
            "schema": manifest["schema"],
            "raw_schema": manifest["raw_schema"],
            "identity_path": manifest["identity_path"],
            "identity_sha256": _sha256_file(Path(manifest["identity_path"])),
            "identity_selection": manifest["identity"].get("selection"),
            "episode_count": len(manifest["episodes"]),
            "episode_ids": [
                f"task{entry['task_index']}:episode{entry['dataset_episode_index']}:frame{entry['frame_index']}"
                for entry in manifest["episodes"]
            ],
            "raw_input_sha256": [entry["sample_sha256"] for entry in manifest["episodes"]],
        }
        _check_deadline(started, args.max_seconds, "input manifest")

        import torch

        policy, torch, runtime_identity = _load_runtime(args, torch)
        summary["runtime_identity"] = runtime_identity
        summary["source_identity"] = runtime_identity["source_identity"]
        summary["gpu"] = _gpu_evidence(torch)
        if runtime_identity["checkpoint"]["sha256"] is None:
            raise RuntimeError("checkpoint hash is unavailable")
        _check_deadline(started, args.max_seconds, "runtime load")

        device = torch.device("cuda:0")
        preprocessor, preprocessor_identity = _load_preprocessor(args.model_path.resolve(), args.vlm_path.resolve())
        summary["preprocessor"] = preprocessor_identity
        batches: List[Dict[str, Any]] = []
        complementary_evidence: List[Dict[str, Any]] = []
        for entry in manifest["episodes"]:
            raw_batch = _load_raw_sample(Path(entry["sample_path"]), entry, torch)
            prepared_batch, auxiliary = _prepare_batch(raw_batch, preprocessor, torch)
            batches.append(prepared_batch)
            complementary_evidence.append(auxiliary)
        summary["input_batches"] = {
            "count": len(batches),
            "keys": [list(batch.keys()) for batch in batches],
            "shapes": [{key: list(value.shape) for key, value in batch.items()} for batch in batches],
            "format": "official_saved_policy_preprocessor_v1",
            "known_complementary": complementary_evidence,
        }
        batches = [_move_to_device(batch, device, torch) for batch in batches]
        noises, noise_np = _noise_tensors(torch, device)
        _check_deadline(started, args.max_seconds, "input batches")

        modules = _eligible_modules(policy, torch)
        # Bind by the actual Parameter object rather than by a guessed textual
        # path.  ``get_vlm_model()`` is an accessor, so its conceptual name is
        # useful for the report but cannot safely identify state_dict entries.
        named_parameters = {id(parameter): name for name, parameter in policy.named_parameters()}
        state_names = set(policy.state_dict().keys())
        eligible_state_names: set[str] = set()
        for conceptual, module in ((name, module) for values in modules.values() for name, module in values):
            parameter_name = named_parameters.get(id(module.weight))
            if parameter_name is None or parameter_name not in state_names:
                raise RuntimeError(f"cannot bind W4 allowlist to an exact state_dict entry: {conceptual}")
            if parameter_name in eligible_state_names:
                raise RuntimeError(f"W4 allowlist state_dict binding is not one-to-one: {parameter_name}")
            eligible_state_names.add(parameter_name)
        if len(eligible_state_names) != sum(len(values) for values in modules.values()):
            raise RuntimeError("W4 allowlist state_dict binding is not one-to-one")
        snapshot = _snapshot_weights(modules, torch)
        bypass_digest = _state_subset_digest(policy, eligible_state_names, torch)
        summary["quantizer"] = {
            "scope": "SmolVLMWithExpertModel.get_vlm_model().text_model.layers vs lm_expert.layers",
            "allowlist_counts": {key: len(value) for key, value in modules.items()},
            "allowlist_state_entries": len(eligible_state_names),
            "bits": BITS,
            "qmax": QMAX,
            "scheme": "symmetric_per_output_channel_RTN_dequantized_FP32",
            "bypassed_modules_digest_before": bypass_digest,
        }

        first_batch = batches[0]
        noop_evidence = _record_noop_gate(policy, first_batch, noises[0], torch)
        summary["no_op"] = noop_evidence
        _check_deadline(started, args.max_seconds, "no-op gate")

        velocities = np.full((EPISODE_COUNT, len(NOISE_SEEDS), len(ARMS), VELOCITY_STEPS, HORIZON, PADDED_ACTION_DIM), np.nan, dtype=np.float32)
        actions = np.full((EPISODE_COUNT, len(NOISE_SEEDS), len(ARMS), HORIZON, PHYSICAL_ACTION_DIM), np.nan, dtype=np.float32)
        completed = np.zeros(EPISODE_COUNT, dtype=np.bool_)
        episode_ids = [
            f"task{entry['task_index']}:episode{entry['dataset_episode_index']}:frame{entry['frame_index']}"
            for entry in manifest["episodes"]
        ]
        quantizer_records: Dict[str, Any] = {}
        for arm_index, arm in enumerate(ARMS):
            _restore_weights(snapshot, modules, torch)
            if arm in LOCUS_NAMES:
                quantizer_records[arm] = _quantize_locus(modules[arm], torch)
            current_bypass = _state_subset_digest(policy, eligible_state_names, torch)
            if current_bypass != bypass_digest:
                raise RuntimeError(f"bypassed module state changed before arm {arm}")
            for episode, batch in enumerate(batches):
                for noise_index, noise in enumerate(noises):
                    recorder = _DenoiseRecorder(policy, torch)
                    action, velocity = recorder.run(batch, noise, record=True)
                    if velocity is None:
                        raise RuntimeError("denoise recorder returned no velocity array")
                    velocities[episode, noise_index, arm_index] = velocity
                    actions[episode, noise_index, arm_index] = action.cpu().numpy()[0]
                    if not np.isfinite(velocity).all() or not np.isfinite(actions[episode, noise_index, arm_index]).all():
                        raise FloatingPointError(f"non-finite output at episode={episode}, noise={noise_index}, arm={arm}")
                    _check_deadline(started, args.max_seconds, f"episode {episode} noise {noise_index} arm {arm}")
                # Progress is written only after all three arms for this
                # episode; partial raw arrays remain independently auditable.
                if arm == ARMS[-1]:
                    completed[episode] = True
                    _write_raw(output, velocities, actions, completed, noise_np, episode_ids)
            current_bypass = _state_subset_digest(policy, eligible_state_names, torch)
            if current_bypass != bypass_digest:
                raise RuntimeError(f"bypassed module state changed after arm {arm}")
        _restore_weights(snapshot, modules, torch)
        if _state_subset_digest(policy, eligible_state_names, torch) != bypass_digest:
            raise RuntimeError("bypassed module state changed after FP restoration")
        if not bool(completed.all()):
            raise RuntimeError("not all twelve episodes completed")
        if not np.isfinite(velocities).all() or not np.isfinite(actions).all():
            raise FloatingPointError("raw flow outputs contain non-finite values")
        metrics = _metrics(velocities, actions)
        summary.update({
            "status": "complete",
            "completed_episodes": int(completed.sum()),
            "quantizer": {**summary["quantizer"], "records": quantizer_records, "bypassed_modules_unchanged": True},
            "metrics": metrics,
            "raw_npz": str((output / "raw_flow.npz").resolve()),
            "elapsed_seconds": time.monotonic() - started,
        })
        _write_small_summary(output / "summary.json", summary)
        print(json.dumps({"status": "complete", "output": str(output), "decision": metrics["overall_decision"]}, indent=2), flush=True)
    except Exception as exc:
        summary.update({
            "status": "implementation_failure" if isinstance(exc, (RuntimeError, FileNotFoundError, ImportError)) else "inconclusive",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        })
        try:
            _write_small_summary(output / "summary.json", summary)
        except RuntimeError:
            _write_small_summary(output / "summary.json", {
                "schema": SCHEMA,
                "status": summary["status"],
                "allocation": summary.get("allocation"),
                "error": summary.get("error"),
                "raw_npz": str((output / "raw_flow.npz").resolve()),
            })
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--lerobot-source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_WORKLOAD_SECONDS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_WORKLOAD_SECONDS:
        parser.error(f"--max-seconds must be in (0, {MAX_WORKLOAD_SECONDS:g}]")
    return args


def main() -> None:
    args = _parse_args()
    # This is intentionally the first workload action.  It checks the real
    # running SLURM job, owner, normalized hostname, partition and nodelist.
    require_allocation = _load_allocation_guard()
    allocation = require_allocation()
    _run(args, allocation)


if __name__ == "__main__":
    main()
