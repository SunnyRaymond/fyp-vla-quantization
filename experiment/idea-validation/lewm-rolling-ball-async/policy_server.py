"""GPU-PBS-only LeWM+CEM server and offline planner smoke for Rolling Ball.

The predictor is the pinned vanilla LeWM checkpoint.  The task goal-bank cost
is an explicit adapter; CEM sampling and distribution updates come directly
from the pinned StableWorldModel solver.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LEWM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
STABLE_WM_COMMIT = "10c26dbd5677083fa31dba69eb738b973845e9a4"
REFLEXBENCH_COMMIT = "8bb931485093c6d98f8729774ad01bf824964e16"
DATASET_COMMIT = "9295b6e9878609a992047f0b8b65421a493299e7"
HORIZON = 5
HISTORY_SIZE = 3
ACTION_DIM = 8
ACTION_HORIZON = 1
NUM_SAMPLES = 300
TOPK = 30
CEM_STEPS = 30
SEED = 1234

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Task root containing data/ and pinned sources")
    parser.add_argument("--checkpoint", type=Path, help="Formal epoch-100 last.ckpt")
    parser.add_argument("--bundle", type=Path, help="Compact exported bundle root (replaces --root/--checkpoint)")
    parser.add_argument("--rented-host", help="Explicit non-PBS Linux host identity; only for a rented RTX 4090")
    parser.add_argument("--smoke", action="store_true", help="Run one validation-frame offline CEM probe")
    parser.add_argument("--output-dir", type=Path, help="PBS run directory for planner_smoke.json")
    parser.add_argument("--self-check", action="store_true", help="Check native protocol shapes using stdlib only")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def check_protocol_shape() -> dict[str, Any]:
    """Exercise only JSON/schema helpers; deliberately imports no ML packages."""
    info = {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "control_mode": "joint_pos",
        "model_name": "rolling-ball-vanilla-lewm-cem",
    }
    request = {
        "type": "vla",
        "num_envs": 1,
        "control_mode": "joint_pos",
        "action_format": "abs_joint",
        "state_format": "joint",
        "orientation_rep": "quat",
        "proprioception": {"joint_positions": [[0.0] * 7], "gripper_state": [[1.0]]},
        "images": {"fixed_cam": ["synthetic-base64-jpeg"]},
        "task_description": "catch the incoming ball",
        "step_ids": [0],
    }
    response = {
        "actions": [[[0.0] * ACTION_DIM]],
        "latency_s": 0.0,
        "cem_latency_s": 0.0,
        "request_id": "opaque",
        "server_received_unix_ns": 1,
        "server_completed_unix_ns": 2,
        "step_ids": [0],
    }
    reset_request = {"env_ids": [0]}
    _validate_vla_request_shape(request, require_image_payload=False)
    decoded = json.loads(json.dumps({"info": info, "request": request,
                                     "response": response, "reset": reset_request}))
    assert decoded["info"]["action_dim"] == ACTION_DIM
    assert decoded["info"]["action_horizon"] == ACTION_HORIZON
    assert len(decoded["response"]["actions"]) == 1
    assert len(decoded["response"]["actions"][0]) == ACTION_HORIZON
    assert len(decoded["response"]["actions"][0][0]) == ACTION_DIM
    assert decoded["reset"] == {"env_ids": [0]}
    return {"status": "PASS", "checks": ["native_vla_shape", "info_shape", "action_shape_1x1x8", "reset_shape"]}


def _validate_vla_request_shape(request: dict[str, Any], *, require_image_payload: bool = True) -> str:
    if request.get("type") != "vla":
        raise ValueError("Only ReflexBench VLA requests are supported")
    if int(request.get("num_envs", 0)) != 1:
        raise ValueError("This adapter supports num_envs=1 only")
    if request.get("control_mode") != "joint_pos":
        raise ValueError("control_mode must be joint_pos")
    if request.get("action_format") != "abs_joint":
        raise ValueError("action_format must be abs_joint")
    if request.get("state_format") not in (None, "joint"):
        raise ValueError("state_format must be joint when present")
    images = request.get("images")
    if not isinstance(images, dict) or "fixed_cam" not in images:
        raise ValueError("images.fixed_cam is required")
    fixed = images["fixed_cam"]
    if not isinstance(fixed, list) or len(fixed) != 1:
        raise ValueError("images.fixed_cam must have one environment entry")
    item = fixed[0]
    if isinstance(item, list):
        if not item:
            raise ValueError("images.fixed_cam history cannot be empty")
        item = item[-1]
    if not isinstance(item, str) or (require_image_payload and not item):
        raise ValueError("images.fixed_cam must contain a base64 JPEG string")
    return item


def require_gpu_pbs() -> tuple[str, str]:
    """Refuse all model/runtime imports unless running on an actual PBS GPU node."""
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile_name = os.environ.get("PBS_NODEFILE", "").strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not job_id or not nodefile_name:
        raise RuntimeError("PBS_JOBID and PBS_NODEFILE are required")
    if not visible or visible.lower() in ("-1", "none", "nodevfiles"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must identify an allocated GPU")
    nodefile = Path(nodefile_name)
    if not nodefile.is_file():
        raise RuntimeError("PBS_NODEFILE does not name a readable allocation file")
    host = socket.gethostname().split(".")[0].lower()
    if any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-compute host: {host}")
    allocated_hosts = {
        line.strip().split(".")[0].lower()
        for line in nodefile.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in allocated_hosts:
        raise RuntimeError(f"Current host {host} is absent from PBS_NODEFILE")
    return job_id, host


def require_rented_host(expected_host: str) -> tuple[str, str]:
    """Explicit non-PBS path for a user-identified rented Linux compute host."""
    if sys.platform != "linux":
        raise RuntimeError("--rented-host is supported only on Linux")
    if os.environ.get("PBS_JOBID") or os.environ.get("PBS_NODEFILE"):
        raise RuntimeError("--rented-host is a non-PBS path; use the default PBS guard inside PBS")
    if not expected_host or expected_host.strip() != expected_host:
        raise RuntimeError("--rented-host must be the exact expected hostname")
    host = socket.gethostname()
    if host != expected_host:
        raise RuntimeError(f"Rented-host identity mismatch: expected {expected_host!r}, got {host!r}")
    if any(tag in host.lower() for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-compute host: {host}")
    return f"rented:{host}", host


def require_rtx_4090(torch) -> str:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    if "4090" not in name.upper():
        raise RuntimeError(f"--rented-host requires RTX 4090; CUDA reports {name!r}")
    return name


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _verify_source(root: Path) -> tuple[Path, Path]:
    lewm = root / "upstream_lewm"
    stable = root / "upstream_stablewm"
    lewm_pin = _read_json(lewm / "PINNED.json")
    stable_pin = _read_json(stable / "PINNED.json")
    if lewm_pin.get("repository") != "lucas-maes/le-wm" or lewm_pin.get("commit") != LEWM_COMMIT:
        raise RuntimeError("upstream_lewm is not the frozen LeWM source")
    if stable_pin.get("repository") != "galilai-group/stable-worldmodel" or stable_pin.get("commit") != STABLE_WM_COMMIT:
        raise RuntimeError("upstream_stablewm is not the frozen StableWorldModel source")
    for rel in ("jepa.py", "utils.py", "config/train/model/lewm.yaml"):
        if not (lewm / rel).is_file():
            raise RuntimeError(f"Pinned LeWM source is missing {rel}")
    for rel in ("stable_worldmodel/solver/cem.py", "stable_worldmodel/policy.py"):
        if not (stable / rel).is_file():
            raise RuntimeError(f"Pinned StableWorldModel source is missing {rel}")
    return lewm, stable


class ProjectedGoalCost:
    """Task-level goal-bank cost around the raw pinned vanilla JEPA rollout."""

    def __new__(cls, *args: Any, **kwargs: Any):
        import torch.nn as nn

        class _CostModel(nn.Module):
            def __init__(self, world_model, mean, std, raw_low, raw_high, goal_bank, torch):
                super().__init__()
                self.world_model = world_model
                self.register_buffer("action_mean", torch.as_tensor(mean, dtype=torch.float32))
                self.register_buffer("action_std", torch.as_tensor(std, dtype=torch.float32))
                self.register_buffer("raw_low", torch.as_tensor(raw_low, dtype=torch.float32))
                self.register_buffer("raw_high", torch.as_tensor(raw_high, dtype=torch.float32))
                self.register_buffer("goal_bank", goal_bank.detach().float())
                self.torch = torch
                self.last_cost = None

            def project_normalized(self, normalized):
                raw = normalized.float() * self.action_std + self.action_mean
                projected_raw = self.torch.maximum(self.torch.minimum(raw, self.raw_high), self.raw_low)
                return (projected_raw - self.action_mean) / self.action_std

            def project_raw(self, normalized):
                raw = normalized.float() * self.action_std + self.action_mean
                return self.torch.maximum(self.torch.minimum(raw, self.raw_high), self.raw_low)

            def get_cost(self, info_dict, action_candidates):
                projected = self.project_normalized(action_candidates)
                predicted = self.world_model.rollout(
                    info_dict, projected, history_size=HISTORY_SIZE
                )["predicted_emb"]
                terminal = predicted[..., -1, :].float()
                # MSE over latent coordinates; minimize over the fixed 180 train goals.
                costs = []
                for bank_chunk in self.goal_bank.split(30, dim=0):
                    mse = (terminal[:, :, None, :] - bank_chunk[None, None, :, :]).square().mean(dim=-1)
                    costs.append(mse.amin(dim=-1))
                result = self.torch.stack(costs, dim=-1).amin(dim=-1)
                self.last_cost = result.detach()
                return result

        return _CostModel(*args, **kwargs)


class PlannerRuntime:
    def __init__(self, *, model, cost_model, solver, action_mean, action_std,
                 raw_low, raw_high, torch, np, image_transform, device):
        self.model = model
        self.cost_model = cost_model
        self.solver = solver
        self.action_mean = action_mean
        self.action_std = action_std
        self.raw_low = raw_low
        self.raw_high = raw_high
        self.torch = torch
        self.np = np
        self.image_transform = image_transform
        self.device = device
        self.lock = threading.Lock()
        self.previous_plan = None

    @staticmethod
    def info() -> dict[str, Any]:
        return {
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "control_mode": "joint_pos",
            "model_name": "rolling-ball-vanilla-lewm-cem",
        }

    def reset(self, env_ids: list[Any]) -> dict[str, str]:
        if not isinstance(env_ids, list) or any(int(x) != 0 for x in env_ids):
            raise ValueError("This N=1 server accepts reset env_ids [0] only")
        if not env_ids:
            return {"status": "ok"}
        with self.lock:
            self.previous_plan = None
            self.solver.torch_gen.manual_seed(SEED)
        return {"status": "ok"}

    def _decode_pixels(self, jpeg_b64: str):
        from PIL import Image

        raw = base64.b64decode(jpeg_b64, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            rgb = self.np.asarray(image.convert("RGB"), dtype=self.np.uint8)
        if rgb.shape != (224, 224, 3):
            raise ValueError(f"fixed_cam decoded to {rgb.shape}, expected 224x224 RGB")
        pixel = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        processed = self.image_transform({"pixels": pixel})["pixels"]
        if tuple(processed.shape) != (1, 3, 224, 224):
            raise RuntimeError(f"official image transform returned {tuple(processed.shape)}")
        return processed.unsqueeze(0).to(self.device, dtype=self.torch.float32)

    def predict(self, request: dict[str, Any], *, started: float | None = None,
                received_ns: int | None = None) -> dict[str, Any]:
        started = time.perf_counter() if started is None else started
        received_ns = time.time_ns() if received_ns is None else received_ns
        request_id = uuid.uuid4().hex
        image_b64 = _validate_vla_request_shape(request)
        pixels = self._decode_pixels(image_b64)
        info = {"pixels": pixels}

        with self.lock:
            if self.previous_plan is None:
                init_action = None
            else:
                # Native WorldModelPolicy warm-start: shift the prior full plan by one.
                init_action = self.previous_plan[:, 1:]
            self.torch.cuda.synchronize(self.device)
            cem_started = time.perf_counter()
            result = self.solver.solve(info, init_action=init_action)
            self.torch.cuda.synchronize(self.device)
            cem_latency_s = time.perf_counter() - cem_started
            plan = result["actions"].to(device=self.device, dtype=self.torch.float32)
            if tuple(plan.shape) != (1, HORIZON, ACTION_DIM):
                raise RuntimeError(f"CEM returned {tuple(plan.shape)}, expected [1,5,8]")
            first_raw = self.cost_model.project_raw(plan[:, :1])
            if not self.torch.isfinite(first_raw).all():
                raise FloatingPointError("Projected action contains NaN or infinity")
            self.previous_plan = result["actions"].detach().clone()

        actions = first_raw.detach().cpu().tolist()
        last_cost = self.cost_model.last_cost
        if last_cost is None or tuple(last_cost.shape) != (1, NUM_SAMPLES):
            raise RuntimeError("Final CEM cost tensor is missing or has an unexpected shape")
        if not self.torch.isfinite(last_cost).all().item():
            raise FloatingPointError("Final CEM candidate costs contain NaN or infinity")
        self.last_cost_summary = {
            "shape": list(last_cost.shape),
            "min": float(last_cost.min().item()),
            "max": float(last_cost.max().item()),
        }
        completed_ns = time.time_ns()
        response = {
            "actions": actions,
            "latency_s": time.perf_counter() - started,
            "cem_latency_s": cem_latency_s,
            "request_id": request_id,
            "server_received_unix_ns": received_ns,
            "server_completed_unix_ns": completed_ns,
        }
        if "step_ids" in request:
            response["step_ids"] = request["step_ids"]
        if len(actions) != 1 or len(actions[0]) != ACTION_HORIZON or len(actions[0][0]) != ACTION_DIM:
            raise RuntimeError("Response action shape is not [1,1,8]")
        return response


def build_runtime(root: Path, checkpoint_path: Path, torch, np, *, compact: bool = False) -> tuple[PlannerRuntime, dict[str, Any]]:
    lewm_dir, stable_dir = _verify_source(root)
    data_dir = root / "data"
    summary = _read_json(data_dir / "summary.json")
    if summary.get("status") != "PASS" or summary.get("task_index") != 3:
        raise RuntimeError("data/summary.json must be PASS for Rolling Ball task_index=3")
    split = _read_json(data_dir / "split.json")
    train_ids = [int(x) for x in split["train_episode_ids"]]
    val_ids = [int(x) for x in split["validation_episode_ids"]]
    if len(train_ids) != 180 or len(val_ids) != 20 or set(train_ids) & set(val_ids):
        raise RuntimeError("Dataset split must be the frozen disjoint 180/20 split")

    bundle_manifest = None
    if compact:
        bundle_manifest = _read_json(root / "BUNDLE.json")
        if bundle_manifest.get("status") != "PASS" or bundle_manifest.get("schema_version") != 1:
            raise RuntimeError("Compact bundle manifest is missing or not PASS")
        if bundle_manifest.get("source") != {
            "lewm": f"lucas-maes/le-wm@{LEWM_COMMIT}",
            "stable_worldmodel": f"galilai-group/stable-worldmodel@{STABLE_WM_COMMIT}",
            "reflexbench": f"LxRoboticsLab/ReflexBench@{REFLEXBENCH_COMMIT}",
            "dataset": f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}",
        }:
            raise RuntimeError("Compact bundle source pins differ from the frozen sources")
        if (bundle_manifest.get("dataset", {}).get("task_index") != 3
                or bundle_manifest.get("dataset", {}).get("train_episode_count") != 180
                or bundle_manifest.get("dataset", {}).get("validation_episode_count") != 20):
            raise RuntimeError("Compact bundle dataset metadata differs from the frozen split")

    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if compact and checkpoint_path != (root / "checkpoint" / "last.ckpt").resolve():
        raise RuntimeError("Compact mode must use its bundled checkpoint/last.ckpt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Formal checkpoint not found: {checkpoint_path}")
    training_summary = _read_json(checkpoint_path.parent / "summary.json")
    if not (
        training_summary.get("status") == "PASS_epoch100"
        and training_summary.get("completed_epochs") == 100
        and training_summary.get("smoke_only") is False
        and training_summary.get("baseline_checkpoint") == "last.ckpt (epoch 100)"
    ):
        raise RuntimeError("Sibling training summary does not verify the formal epoch-100 last.ckpt")
    if compact and bundle_manifest.get("checkpoint") != {
        "path": "checkpoint/last.ckpt",
        "training_job_id": training_summary.get("job_id"),
        "epoch": 100,
    }:
        raise RuntimeError("Compact bundle checkpoint provenance differs from its training summary")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != "vanilla_lewm_raw_model_state_v1" or int(checkpoint.get("epoch", -1)) != 100:
        raise RuntimeError("Planner requires the formal epoch-100 vanilla LeWM last.ckpt")
    ckpt_source = checkpoint.get("source", {})
    ckpt_split = checkpoint.get("split")
    if not isinstance(ckpt_split, dict):
        raise RuntimeError("Checkpoint is missing its training split provenance")
    if ([int(x) for x in ckpt_split.get("train_episode_ids", [])] != train_ids
            or [int(x) for x in ckpt_split.get("validation_episode_ids", [])] != val_ids):
        raise RuntimeError("Checkpoint split differs from data/split.json")
    if ckpt_source.get("job_id") != training_summary.get("job_id"):
        raise RuntimeError("Checkpoint training job ID differs from its sibling summary")
    if ckpt_source.get("lewm", {}).get("commit") != LEWM_COMMIT:
        raise RuntimeError("Checkpoint LeWM source commit differs from the frozen source")
    if ckpt_source.get("stable_worldmodel", {}).get("commit") != STABLE_WM_COMMIT:
        raise RuntimeError("Checkpoint StableWorldModel source commit differs from the frozen source")
    if ckpt_source.get("dataset") != f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}":
        raise RuntimeError("Checkpoint dataset identity differs from the frozen dataset")

    sys.path.insert(0, str(root / "deps"))
    sys.path.insert(0, str(stable_dir))
    sys.path.insert(0, str(lewm_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from gymnasium.spaces import Box
    from stable_worldmodel.policy import PlanConfig
    from stable_worldmodel.solver.cem import CEMSolver
    from utils import get_img_preprocessor

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("_target_") != "jepa.JEPA":
        raise RuntimeError("Checkpoint model_config must instantiate pinned jepa.JEPA")
    model = instantiate(OmegaConf.create(model_config))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device="cuda", dtype=torch.float32).eval()
    model.requires_grad_(False)
    if int(getattr(model.predictor, "num_frames", HISTORY_SIZE)) != HISTORY_SIZE:
        raise RuntimeError("Checkpoint predictor history does not match frozen history_size=3")

    stats = checkpoint.get("action_stats", {})
    bounds = checkpoint.get("action_bounds", {})
    mean = np.asarray(stats.get("mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(stats.get("std"), dtype=np.float32).reshape(-1)
    raw_low = np.asarray(bounds.get("min"), dtype=np.float32).reshape(-1)
    raw_high = np.asarray(bounds.get("max"), dtype=np.float32).reshape(-1)
    if any(arr.shape != (ACTION_DIM,) for arr in (mean, std, raw_low, raw_high)):
        raise RuntimeError("Checkpoint action stats/bounds must each have shape [8]")
    if not all(np.isfinite(arr).all() for arr in (mean, std, raw_low, raw_high)) or np.any(std <= 0):
        raise RuntimeError("Checkpoint action stats/bounds must be finite with positive std")
    if np.any(raw_low > raw_high):
        raise RuntimeError("Checkpoint action bounds are reversed")
    normalized_low = (raw_low - mean) / std
    normalized_high = (raw_high - mean) / std
    action_space = Box(
        low=np.broadcast_to(normalized_low, (1, ACTION_DIM)).copy(),
        high=np.broadcast_to(normalized_high, (1, ACTION_DIM)).copy(),
        dtype=np.float32,
    )

    goal_refs = checkpoint.get("train_goal_bank_terminal_frames")
    if not isinstance(goal_refs, list) or len(goal_refs) != len(train_ids):
        raise RuntimeError("Checkpoint must identify exactly one terminal training frame per train episode")
    goal_episode_ids = [int(item["episode_id"]) for item in goal_refs]
    if set(goal_episode_ids) != set(train_ids) or len(set(goal_episode_ids)) != len(goal_episode_ids):
        raise RuntimeError("Goal bank episode IDs do not exactly match the train split")

    state_available = False
    state = None
    if compact:
        goal_refs_path = data_dir / "goal_refs.json"
        if _read_json(goal_refs_path) != goal_refs:
            raise RuntimeError("Compact goal refs differ from the ordered checkpoint refs")
        goal_frames = np.load(data_dir / "goal_frames.npy", mmap_mode="r", allow_pickle=False)
        if goal_frames.dtype != np.uint8 or goal_frames.shape != (180, 224, 224, 3):
            raise RuntimeError(f"Unexpected compact goal frame array: {goal_frames.shape}/{goal_frames.dtype}")
        validation_meta = _read_json(data_dir / "validation_probe.json")
        if (int(validation_meta.get("episode_id", -1)) not in set(val_ids)
                or validation_meta.get("camera") != "observation.images.fixed_cam"
                or validation_meta.get("state_source") != "observation.state"):
            raise RuntimeError("Compact validation probe provenance is not from the fixed-camera validation split")
        with np.load(data_dir / "validation_probe.npz", allow_pickle=False) as probe:
            frame = np.asarray(probe["frame"])
            state = np.asarray(probe["state"])
        if frame.dtype != np.uint8 or frame.shape != (224, 224, 3):
            raise RuntimeError("Compact validation RGB probe must be uint8 [224,224,3]")
        if state.dtype != np.float32 or state.shape != (ACTION_DIM,):
            raise RuntimeError("Compact validation state probe must be float32 [8]")
        if (bundle_manifest.get("validation_probe", {}).get("episode_id") != int(validation_meta["episode_id"])
                or bundle_manifest.get("validation_probe", {}).get("dataset_row") != int(validation_meta["dataset_row"])
                or bundle_manifest.get("validation_probe", {}).get("frame_index") != int(validation_meta["frame_index"])):
            raise RuntimeError("Compact validation probe manifest and metadata disagree")
        validation_episode = int(validation_meta["episode_id"])
        validation_row = int(validation_meta["dataset_row"])
        validation_frame_index = int(validation_meta["frame_index"])
        state_available = True
        validation_frame = frame.copy()
        validation_state = state.copy()
        if bundle_manifest.get("goal_bank", {}).get("count") != 180:
            raise RuntimeError("Compact goal bank must contain exactly 180 frames")
    else:
        frames = np.load(data_dir / "frames.npy", mmap_mode="r", allow_pickle=False)
        episode_ids = np.load(data_dir / "episode_ids.npy", mmap_mode="r", allow_pickle=False)
        frame_indices = np.load(data_dir / "frame_indices.npy", mmap_mode="r", allow_pickle=False)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (224, 224, 3):
            raise RuntimeError(f"Unexpected frames.npy shape/dtype: {frames.shape}/{frames.dtype}")
        row_count = len(frames)
        if episode_ids.shape != (row_count,) or frame_indices.shape != (row_count,):
            raise RuntimeError("Episode/frame arrays do not align with frames.npy")
        summary_frames = summary.get("arrays", {}).get("frames", {})
        if summary_frames.get("shape") != list(frames.shape) or summary_frames.get("dtype") != "uint8":
            raise RuntimeError("summary.json frame array metadata differs from the prepared arrays")

        goal_rows = []
        for item in goal_refs:
            row = int(item["dataset_row"])
            eid = int(item["episode_id"])
            if not 0 <= row < row_count or int(episode_ids[row]) != eid:
                raise RuntimeError(f"Goal-bank row {row} does not belong to training episode {eid}")
            if int(frame_indices[row]) != 25:
                raise RuntimeError(f"Goal-bank row {row} is not the final available frame of episode {eid}")
            goal_rows.append(row)
        goal_frames = np.asarray(frames[goal_rows], dtype=np.uint8)

        state_path = data_dir / "state.npy"
        state_meta = summary.get("arrays", {}).get("state")
        if state_path.is_file() and isinstance(state_meta, dict):
            states = np.load(state_path, mmap_mode="r", allow_pickle=False)
            if (states.shape == (row_count, ACTION_DIM) and states.dtype == np.float32
                    and state_meta.get("shape") == [row_count, ACTION_DIM]
                    and state_meta.get("dtype") == "float32"):
                state = states
                state_available = True

        val_rows = np.flatnonzero(np.isin(episode_ids, np.asarray(val_ids, dtype=episode_ids.dtype)))
        if not len(val_rows):
            raise RuntimeError("No validation frames found in prepared arrays")
        val_order = np.lexsort((np.asarray(frame_indices[val_rows]), np.asarray(episode_ids[val_rows])))
        validation_row = int(val_rows[val_order[0]])
        validation_episode = int(episode_ids[validation_row])
        if validation_episode not in set(val_ids):
            raise RuntimeError("Selected smoke observation is not in validation split")
        validation_frame_index = int(frame_indices[validation_row])
        validation_frame = np.asarray(frames[validation_row], dtype=np.uint8).copy()
        validation_state = np.asarray(state[validation_row], dtype=np.float32).copy() if state_available else None

    image_transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    goal_embeddings = []
    with torch.inference_mode():
        for start in range(0, len(goal_frames), 30):
            raw = np.asarray(goal_frames[start:start + 30], dtype=np.uint8)
            pixels = torch.from_numpy(raw.copy()).permute(0, 3, 1, 2)
            transformed = image_transform({"pixels": pixels})["pixels"]
            transformed = transformed.unsqueeze(1).to("cuda", dtype=torch.float32)
            encoded = model.encode({"pixels": transformed})["emb"][:, 0]
            goal_embeddings.append(encoded.detach())
    goal_bank = torch.cat(goal_embeddings, dim=0)
    if tuple(goal_bank.shape) != (180, 192) or not torch.isfinite(goal_bank).all():
        raise RuntimeError(f"Unexpected train goal bank embedding shape/content: {tuple(goal_bank.shape)}")

    cost_model = ProjectedGoalCost(
        model, mean, std, raw_low, raw_high, goal_bank, torch
    ).to("cuda").eval()
    cost_model.requires_grad_(False)
    plan_config = PlanConfig(
        horizon=HORIZON,
        receding_horizon=1,
        history_len=HISTORY_SIZE,
        action_block=1,
        warm_start=True,
    )
    solver = CEMSolver(
        model=cost_model,
        batch_size=1,
        num_samples=NUM_SAMPLES,
        var_scale=1.0,
        n_steps=CEM_STEPS,
        topk=TOPK,
        device="cuda",
        seed=SEED,
    )
    solver.configure(action_space=action_space, n_envs=1, config=plan_config)

    runtime = PlannerRuntime(
        model=model,
        cost_model=cost_model,
        solver=solver,
        action_mean=mean,
        action_std=std,
        raw_low=raw_low,
        raw_high=raw_high,
        torch=torch,
        np=np,
        image_transform=image_transform,
        device="cuda",
    )
    context = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": 100,
        "training_job_id": training_summary.get("job_id"),
        "source": {
            "lewm": f"lucas-maes/le-wm@{LEWM_COMMIT}",
            "stable_worldmodel_solver": f"galilai-group/stable-worldmodel@{STABLE_WM_COMMIT}",
            "reflexbench_protocol": f"LxRoboticsLab/ReflexBench@{REFLEXBENCH_COMMIT}",
            "dataset": f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}",
        },
        "goal_bank_episode_count": 180,
        "goal_bank_frame_semantics": "last available training observation; per-episode success not verified",
        "validation_episode_id": validation_episode,
        "validation_dataset_row": validation_row,
        "validation_frame_index": validation_frame_index,
        "validation_proprioception_available": state_available,
        "validation_state_source": "observation.state" if state_available else None,
        "validation_provenance": {
            "episode_id": validation_episode,
            "dataset_row": validation_row,
            "frame_index": validation_frame_index,
            "camera": "observation.images.fixed_cam",
            "state_source": "observation.state" if state_available else None,
        },
        "split": {"train_episode_ids": train_ids, "validation_episode_ids": val_ids},
        "goal_refs": goal_refs,
        "checkpoint_source": ckpt_source,
        "frame": validation_frame,
        "state": validation_state,
    }
    return runtime, context


def _native_smoke_request(context: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(context["frame"], mode="RGB").save(buffer, format="JPEG", quality=90)
    payload: dict[str, Any] = {
        "type": "vla",
        "num_envs": 1,
        "control_mode": "joint_pos",
        "action_format": "abs_joint",
        "state_format": "joint",
        "orientation_rep": "quat",
        "images": {"fixed_cam": [base64.b64encode(buffer.getvalue()).decode("ascii")]},
        "task_description": "catch the incoming ball",
        "step_ids": [0],
    }
    state = context.get("state")
    if state is not None:
        if np_shape(state) != (ACTION_DIM,):
            raise RuntimeError("Smoke observation.state does not have 8 dimensions")
        payload["proprioception"] = {
            "joint_positions": [state[:7].astype(float).tolist()],
            # ReflexBench eval.py represents gripper state as 0/1 from finger width.
            "gripper_state": [[float(float(state[7]) > 0.5)]],
        }
    return payload


def np_shape(value) -> tuple[int, ...]:
    return tuple(value.shape)


def run_smoke(runtime: PlannerRuntime, context: dict[str, Any], *, job_id: str, host: str) -> dict[str, Any]:
    info = runtime.info()
    if info["action_dim"] != ACTION_DIM or info["action_horizon"] != ACTION_HORIZON:
        raise RuntimeError("GET /info schema/config mismatch")
    reset_result = runtime.reset({"env_ids": [0]}["env_ids"])
    if reset_result != {"status": "ok"}:
        raise RuntimeError("POST /reset schema check failed")
    request = _native_smoke_request(context)
    _validate_vla_request_shape(request)
    # Cross the same JSON boundary as the HTTP protocol before invoking planner.
    started = time.perf_counter()
    received_ns = time.time_ns()
    request = json.loads(json.dumps(request))
    response = runtime.predict(request, started=started, received_ns=received_ns)
    actions = runtime.np.asarray(response["actions"], dtype=runtime.np.float32)
    if actions.shape != (1, 1, ACTION_DIM) or not runtime.np.isfinite(actions).all():
        raise RuntimeError(f"Offline smoke output invalid: {actions.shape}")
    return {
        "status": "PASS_protocol_and_planner_shape",
        "job_id": job_id,
        "host": host,
        "checkpoint": context["checkpoint"],
        "checkpoint_epoch": context["checkpoint_epoch"],
        "source": context["source"],
        "cem": {"num_samples": NUM_SAMPLES, "topk": TOPK, "n_steps": CEM_STEPS,
                "horizon": HORIZON, "action_block": 1, "receding_horizon": 1,
                "seed": SEED, "dtype": "float32"},
        "action_shape": list(actions.shape),
        "latency_s": response["latency_s"],
        "cem_latency_s": response["cem_latency_s"],
        "final_cem_cost": runtime.last_cost_summary,
        "request_id": response["request_id"],
        "server_received_unix_ns": response["server_received_unix_ns"],
        "server_completed_unix_ns": response["server_completed_unix_ns"],
        "validation_episode_id": context["validation_episode_id"],
        "validation_dataset_row": context["validation_dataset_row"],
        "validation_frame_index": context["validation_frame_index"],
        "validation_proprioception_available": context["validation_proprioception_available"],
        "validation_state_source": context["validation_state_source"],
        "goal_bank_episode_count": context["goal_bank_episode_count"],
        "goal_bank_frame_semantics": context["goal_bank_frame_semantics"],
        "closed_loop_evaluated": False,
        "task_success": None,
        "observation_age_s": None,
        "rtf": None,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class PolicyHandler(BaseHTTPRequestHandler):
    runtime: PlannerRuntime

    def _write_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/info":
            self._write_response(404, {"error": "not found"})
            return
        self._write_response(200, self.runtime.info())

    def do_POST(self) -> None:
        if self.path not in ("/predict", "/reset"):
            self._write_response(404, {"error": "not found"})
            return
        started = time.perf_counter()
        received_ns = time.time_ns()
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 8 * 1024 * 1024:
                raise ValueError("request body must be between 1 byte and 8 MiB")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            if self.path == "/reset":
                result = self.runtime.reset(payload.get("env_ids"))
            else:
                result = self.runtime.predict(payload, started=started, received_ns=received_ns)
            self._write_response(200, result)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._write_response(400, {"error": str(exc)})
        except Exception as exc:
            self._write_response(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid logging request bodies or repetitive access details to the PBS log.
        return


def main() -> int:
    args = parse_args()
    if args.self_check:
        print(json.dumps(check_protocol_shape(), sort_keys=True))
        return 0
    if args.bundle is not None:
        if args.root is not None or args.checkpoint is not None:
            raise SystemExit("Use --bundle alone; do not combine it with --root/--checkpoint")
        root = args.bundle.resolve()
        checkpoint_path = root / "checkpoint" / "last.ckpt"
        compact = True
    else:
        if args.root is None or args.checkpoint is None:
            raise SystemExit("--root and --checkpoint are required unless --bundle is used")
        if args.rented_host:
            raise SystemExit("--rented-host is restricted to --bundle mode")
        root = args.root.resolve()
        checkpoint_path = args.checkpoint
        compact = False
    if args.smoke and args.output_dir is None:
        raise SystemExit("--output-dir is required with --smoke")
    if args.rented_host:
        job_id, host = require_rented_host(args.rented_host)
    else:
        job_id, host = require_gpu_pbs()

    # Numerical and model imports occur only after the PBS node/GPU guards pass.
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available inside the guarded compute environment")
    if args.rented_host:
        require_rtx_4090(torch)
    output_path = args.output_dir / "planner_smoke.json" if args.smoke else None
    failure_summary = {
        "status": "FAIL",
        "job_id": job_id,
        "host": host,
        "closed_loop_evaluated": False,
        "task_success": None,
        "observation_age_s": None,
        "rtf": None,
    }
    try:
        runtime, context = build_runtime(root, checkpoint_path, torch, np, compact=compact)
        if args.smoke:
            result = run_smoke(runtime, context, job_id=job_id, host=host)
            _write_json(output_path, result)
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        handler = type("BoundPolicyHandler", (PolicyHandler,), {"runtime": runtime})
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print(json.dumps({"status": "serving", "job_id": job_id, "host": host,
                          "bind": args.host, "port": args.port, "info": runtime.info(),
                          "checkpoint": context["checkpoint"],
                          "checkpoint_epoch": context["checkpoint_epoch"],
                          "source": context["source"]}), flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        finally:
            server.server_close()
    except Exception as exc:
        if output_path is not None:
            failure_summary["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(output_path, failure_summary)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
